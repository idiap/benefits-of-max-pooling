
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import time
import logging
import itertools
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
import numpy as np

import utils.lists

torch_inf = torch.tensor(float("Inf"))
torch_nan = torch.tensor(float("nan"))
# assert torch.__version__ >= (1, 10), "relies on torch.isin"

logging_fmt = '%(filename)s:%(lineno)d %(asctime)s %(message)s'
f_format = logging.Formatter(logging_fmt)

logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setFormatter(f_format)
logger_streamhandler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

for _ in logger.handlers:
    logger.removeHandler(_)  # make idempotent
logger.setLevel(logging.DEBUG)

logging.Formatter.converter = time.gmtime
logger.addHandler(logger_streamhandler)


def build_relu_layers(input_dim: int,
                      hidden_layer_widths: List[int],
                      output_dim: int,
                      include_bias: bool) -> List[torch.nn.Module]:
    num_layers = len(hidden_layer_widths)
    all_layer_widths = [input_dim] + hidden_layer_widths + [output_dim]
    assert hidden_layer_widths[-1] > 0
    final_linear_layer = torch.nn.Linear(hidden_layer_widths[-1],
                                         output_dim,
                                         bias=include_bias)
    layer_list = []
    for i in range(num_layers):
        w0 = all_layer_widths[i]
        w1 = all_layer_widths[i + 1]
        layer_list += [torch.nn.Linear(w0, w1, bias=include_bias),
                       torch.nn.ReLU()]
    layer_list += [final_linear_layer]
    return layer_list


def build_w1(size_out: int) -> torch.Tensor:
    nr = size_out * 4
    w1 = torch.zeros((size_out, nr))
    for idx in range(size_out):
        c0 = (idx + 0) * 4
        c1 = (idx + 1) * 4
        w1[idx, c0:c1] = torch.tensor([1, 1, 1, -1]) / 2
    return w1


def build_w0_general(dim: int,
                     mapping: List[tuple]) -> torch.Tensor:
    mapping_len = len(mapping)
    w0 = torch.zeros((4 * mapping_len, dim))
    for idx, mapping_elem in enumerate(mapping):
        w0[idx * 4 + 0, mapping_elem[0]] += +1
        w0[idx * 4 + 0, mapping_elem[1]] += -1
        w0[idx * 4 + 1, mapping_elem[0]] += -1
        w0[idx * 4 + 1, mapping_elem[1]] += +1
        w0[idx * 4 + 2, mapping_elem[0]] += +1
        w0[idx * 4 + 2, mapping_elem[1]] += +1
        w0[idx * 4 + 3, mapping_elem[0]] += -1
        w0[idx * 4 + 3, mapping_elem[1]] += -1
    return w0


def flatten_sequential_of_sequentials_to_layers(unflattened_model: Iterable[torch.nn.Module]) -> List[torch.nn.Module]:
    layers_out = []
    for layer in unflattened_model:
        if type(layer) == torch.nn.Sequential:
            to_add = flatten_sequential_of_sequentials_to_layers(layer)
        else:
            to_add = [layer]
        layers_out += to_add
    return layers_out


def _fuse_list_of_linears(lol: List[torch.nn.Linear]) -> torch.nn.Linear:
    fused_linear = lol[0]
    if len(lol) > 1:
        for lin in lol[1:]:
            fused_linear = fuse_linears(fused_linear, lin)
    return fused_linear


def interleave_lists(list1: list, list2: list) -> list:
    assert len(list1) == len(list2)
    return [*sum(zip(list2, list1), ())]


def canonicalize_network(network: torch.nn.Sequential) -> torch.nn.Sequential():
    is_linear = [type(_) == torch.nn.Linear for _ in network]

    assert is_linear[-1], "last layer must be linear by convention"
    assert is_linear[0], "first layer must be linear by convention"
    if all(is_linear):
        layers = [_fuse_list_of_linears(list(network))]
    else:
        assert type(network[1]) == torch.nn.ReLU, "second layer must be relu by convention"
        split_elem = None
        i = [idx if _ else split_elem for idx, _ in enumerate(is_linear)]
        spliti = utils.lists.splitter(i, split_elem)

        num_out = len(spliti)
        fused = [None] * num_out
        for idx in range(num_out):
            to_fuse = [network[_] for _ in spliti[idx]]
            fused[idx] = _fuse_list_of_linears(to_fuse)
        relu_layers = [torch.nn.ReLU()] * (num_out - 1)
        layers = interleave_lists(relu_layers, fused[:-1]) + [fused[-1]]
    canonicalized_network = torch.nn.Sequential(*layers)
    return canonicalized_network


def fuse_linears(lin1: torch.nn.Linear,
                 lin2: torch.nn.Linear) -> torch.nn.Linear:
    in_features = lin1.in_features
    out_features = lin2.out_features

    fused_lin = torch.nn.Linear(in_features, out_features)
    to_w = lin2.weight @ lin1.weight
    assert to_w.shape == fused_lin.weight.shape, "shape mismatch in weight assignment"
    fused_lin.weight = torch.nn.Parameter(to_w)

    b1 = lin1.bias if (lin1.bias is not None) else torch.zeros((lin1.out_features,),
                                                               dtype=lin1.weight.dtype,
                                                               device=lin1.weight.device)
    b2 = lin2.bias if (lin2.bias is not None) else torch.zeros((lin2.out_features,),
                                                               dtype=lin2.weight.dtype,
                                                               device=lin2.weight.device)

    to_b = (lin2.weight @ b1 + b2).flatten()
    assert to_b.shape == fused_lin.bias.shape, "shape mismatch in bias assignment"
    fused_lin.bias = torch.nn.Parameter(to_b)
    return fused_lin


def build_maxer_layer_general(input_dim: int,
                              mapping: List[tuple]) -> torch.nn.Sequential:
    size_out = len(mapping)

    lin0 = torch.nn.Linear(input_dim, 4 * size_out, bias=False)
    lin1 = torch.nn.Linear(4 * size_out, size_out, bias=False)

    w0 = build_w0_general(input_dim, mapping)
    w1 = build_w1(size_out)

    assert lin0.weight.shape == w0.shape
    assert lin1.weight.shape == w1.shape

    lin0.weight = torch.nn.Parameter(w0)
    lin1.weight = torch.nn.Parameter(w1)

    layers = [lin0, torch.nn.ReLU(), lin1]
    maxer_layer = torch.nn.Sequential(*layers)
    return maxer_layer


def _build_subpools(subsequent: List[tuple]) -> List[tuple]:
    dim = len(subsequent[0])
    assert all([dim == len(_) for _ in subsequent])
    cp0 = math.floor(dim / 2)

    is_odd_dim = dim % 2 == 1
    offset = 1 if is_odd_dim else 0
    subpools = [(_[:cp0+offset], _[cp0:]) for _ in subsequent]
    return subpools


def _minimal_decomp_necc_to_compute_subsequent(subsequent: List[tuple]):
    assert subsequent == sorted(subsequent)
    subpools = _build_subpools(subsequent)
    flattened = [x for _ in subpools for x in _]
    minimal_decomp = sorted(set(flattened))
    return minimal_decomp


def compute_normalized_form(dim: int, required_terms: List[int]) -> dict:
    xs = tuple([_ for _ in range(dim)])
    normalized_form = dict()

    if len(required_terms) > 0:
        done = False
        curr_rank = max(required_terms)
        curr_ranks = []

        last_rank = curr_rank
        iters = 0
        while not done:
            is_required = curr_rank in required_terms
            is_half = curr_rank == math.ceil(last_rank / 2)
            if is_required:
                decomp = sorted(list(itertools.combinations(xs, curr_rank)))
            elif is_half:
                subsequent = normalized_form[last_rank]
                decomp = _minimal_decomp_necc_to_compute_subsequent(subsequent)
            if is_required or is_half:
                normalized_form[curr_rank] = decomp
                last_rank = curr_rank
                curr_ranks.append(curr_rank)
            curr_rank = curr_rank - 1

            done = curr_rank <= 1
            iters += 1
        # print(required_terms, dim, curr_ranks, set(curr_ranks) - set(required_terms))
        # assert 2 in curr_ranks
    assert 0 not in normalized_form.keys()
    return normalized_form


def _gen_integer_subsets(dim: int, rank: int) -> List[tuple]:
    xs = tuple([_ for _ in range(dim)])
    integer_subsets = sorted(list(itertools.combinations(xs, rank)))
    return integer_subsets


def _preimager_general(preimage: List[tuple],
                       image: List[tuple]) -> List[tuple]:
    # Always works, but slow (quadratic in preimage size)
    assert preimage == sorted(preimage)
    assert image == sorted(image)
    preimage_size = len(preimage)
    pair_indices = _gen_integer_subsets(preimage_size, 2)
    melted = [tuple(sorted(set(preimage[_[0]] + preimage[_[1]]))) for _ in pair_indices]
    preimaged = [pair_indices[melted.index(_)] for _ in image]
    return preimaged


def preimager(dim: int,
              preimage: List[tuple],
              image: List[tuple]) -> List[tuple]:
    preimage_ord = len(preimage[0])
    image_ord = len(image[0])
    assert image_ord <= 2 * preimage_ord

    element_inverses = _preimager_general(preimage, image)
    return element_inverses


def compute_ords(dim: int) -> List[int]:
    ld = int(math.ceil(math.log2(dim - 1)))
    out = [math.ceil((dim - 1) / 2 ** _) for _ in range(ld)]
    return out


def compute_vals2(dim: int) -> List[int]:
    ld = int(math.ceil(math.log2(dim - 1)))
    vals = [None] * ld
    for j in range(1, ld):
        # j = 1
        roundby = 2 ** (ld - j - 1)
        start = 2 ** (ld - j) + 1
        first_value = 3 * 2 ** (ld - j - 1)
        vals[j] = first_value + math.ceil((dim - start) / roundby) * roundby
    vals[0] = 2 ** math.floor(math.log2(dim - 2)) - 1 + dim
    return vals


def compute_vals(dim: int) -> List[int]:
    ords = compute_ords(dim)
    ords1 = ords + [1]
    ld = int(math.ceil(math.log2(dim - 1)))
    assert len(ords) == ld
    vals = [None] * ld
    for j in range(1, ld):
        # j = 1
        vals[j] = (2 ** (ld - 1 - j)) * (1 + ords1[ld - 1 - j])
    vals[0] = 2 ** math.floor(math.log2(dim - 2)) - 1 + dim
    return vals


def _build_layer_mappings_old(required_terms: torch.Tensor,
                              dim: int) -> Dict[int, list]:
    nform = compute_normalized_form(dim, required_terms)
    ords = sorted(set(nform.keys()))
    subpool_orders = [_ for _ in ords if _ != 1]
    num_max_layers = len(subpool_orders)
    for v in nform.values():
        assert v == sorted(v)

    if 0 == num_max_layers:
        mappings = []
    elif 1 == num_max_layers:
        mappings = [nform[subpool_orders[0]]]
    else:
        mappings = [None] * num_max_layers
        for idx in range(num_max_layers - 1):
            # for idx in range(num_max_layers - 1):
            # idx = 0
            curr_layer_idx = num_max_layers - idx - 1
            curr_subpool_order = subpool_orders[curr_layer_idx]
            prev_subpool_order = subpool_orders[curr_layer_idx - 1]
            preimage = nform[prev_subpool_order]
            image = nform[curr_subpool_order]
            mappings[idx] = preimager(dim, preimage, image)
        mappings[-1] = nform[prev_subpool_order]
    layer_mappings = {k: v for k, v in zip(subpool_orders, mappings)}
    return layer_mappings


def _shim_up_layers(layers_in: List[torch.nn.Module]) -> List[torch.nn.Module]:
    is_relu = [type(_) == torch.nn.ReLU for _ in layers_in]
    if not any(is_relu):
        is_linear = [type(_) == torch.nn.Linear for _ in layers_in]
        assert all(is_linear)
        layers = layers_in
    else:
        layer_starts = torch.argwhere(torch.tensor(is_relu)).flatten() - 1
        num_relus = sum(is_relu)
        components = [None] * num_relus
        o = torch.ones(1, 1)

        for idx, layer_start in enumerate(layer_starts):
            # idx = 0; layer_start = layer_starts[idx]
            layer0_in = layers_in[layer_start]
            layer1_in = layers_in[layer_start + 2]

            layer0_out = torch.nn.Linear(layer0_in.in_features, layer0_in.out_features + 2, bias=False)
            layer1_out = torch.nn.Linear(layer0_in.out_features + 2, layer1_in.out_features + 1, bias=False)
            kill_layer = torch.nn.Linear(layer1_in.out_features + 1, layer1_in.out_features, bias=False)

            row0 = torch.ones((1, layer0_in.in_features)) / layer0_in.in_features
            weight0_out = torch.cat((+row0, -row0, layer0_in.weight), 0)

            top1 = torch.cat((+o, -o, torch.zeros((1, layer1_in.weight.shape[1]))), 1)
            left1 = torch.zeros((layer1_in.weight.shape[0], 2))

            weight1_out = torch.cat((top1, torch.cat((left1, layer1_in.weight), 1)), 0)
            kill_weight = torch.cat((torch.zeros((layer1_in.out_features, 1)),
                                     torch.eye(layer1_in.out_features)), 1)

            assert layer0_out.weight.shape == weight0_out.shape
            layer0_out.weight = torch.nn.Parameter(weight0_out)

            assert layer1_out.weight.shape == weight1_out.shape
            layer1_out.weight = torch.nn.Parameter(weight1_out)

            assert kill_layer.weight.shape == kill_weight.shape
            kill_layer.weight = torch.nn.Parameter(kill_weight)

            component = [layer0_out, torch.nn.ReLU(), layer1_out, kill_layer]
            components[idx] = component

        layers = utils.lists.flatten_list_of_lists(components)
    return layers


def cat_top_left(top: torch.Tensor,
                 left: torch.Tensor,
                 lower_right: torch.Tensor) -> torch.Tensor:
    return torch.cat((top, torch.cat((left, lower_right), 1)), 0)


def build_element_cardinalities(decomp0) -> Tuple[list, list]:
    subpools = build_subpools(decomp0)
    flattened = [x for _ in subpools for x in _]
    counts = [v for k, v in Counter(flattened).items()]
    count_counter = Counter(counts)
    sorted_keys = sorted(count_counter.keys(), reverse=True)
    sorted_values = [count_counter[k] for k in sorted_keys]
    # for k in sorted_keys:
    #     msg = f"{k}: {count_counter[k]}"
    return sorted_keys, sorted_values


def _maxpool_average_layers(layers_in: List[torch.nn.Module]) -> List[torch.nn.Module]:
    model_in = torch.nn.Sequential(*layers_in)
    is_relu = [type(_) == torch.nn.ReLU for _ in layers_in]
    if not any(is_relu):
        is_linear = [type(_) == torch.nn.Linear for _ in layers_in]
        assert all(is_linear)
        mpa_layers = layers_in
    else:
        dim = layers_in[0].in_features
        layer_starts = torch.argwhere(torch.tensor(is_relu)).flatten() - 1

        num_layer_starts = len(layer_starts)
        components = [None] * num_layer_starts
        c = torch.Tensor([[+1], [-1]])
        o = torch.ones(1, 1)
        o2 = torch.cat((+o, -o), 1)

        for idx, layer_start in enumerate(layer_starts):
            # idx = 0; layer_start = layer_starts[idx]
            # idx = 1; layer_start = layer_starts[idx]
            layer2_in = layers_in[layer_start + 2]
            layer2_out = torch.nn.Linear(layer2_in.in_features + 2 * idx,
                                         layer2_in.out_features + idx, bias=False)

            okron = torch.kron(torch.eye(idx), o2)
            top2 = torch.cat((okron, torch.zeros(idx, layer2_in.weight.shape[1])), 1)
            left2 = torch.zeros((layer2_in.weight.shape[0], 2 * idx))
            weight2_out = cat_top_left(top2, left2, layer2_in.weight)

            assert layer2_out.weight.shape == weight2_out.shape
            layer2_out.weight = torch.nn.Parameter(weight2_out)

            if idx < num_layer_starts - 1:
                layer4_in = layers_in[layer_start + 4]
                layer4_out = torch.nn.Linear(layer4_in.in_features + 1 + idx,
                                             layer4_in.out_features + 2 * (1 + idx),
                                             bias=False)

                kronc = torch.kron(torch.eye(1 + idx), c)
                top4 = torch.cat((kronc, torch.zeros(2 * (1 + idx), layer4_in.weight.shape[1])), 1)
                left4 = torch.zeros((layer4_in.weight.shape[0], 1 + idx))
                weight4_out = cat_top_left(top4, left4, layer4_in.weight)
                assert layer4_out.weight.shape == weight4_out.shape
                layer4_out.weight = torch.nn.Parameter(weight4_out)
                component = [layer2_out, layer4_out, torch.nn.ReLU()]
            else:
                component = [layer2_out]
            components[idx] = component

            do_check = True
            if do_check:
                running_layers = layers_in[:2] + utils.lists.flatten_list_of_lists(components[:idx + 1])
                running_model = torch.nn.Sequential(*running_layers)
                x = torch.rand((5, dim))
                y1 = running_model(x)
                y2 = model_in[:6 + 4 * idx](x)

                if idx + 1 == num_layer_starts:
                    torch.testing.assert_close(y1[:, idx + 1:], y2)
                    for i in range(idx):
                        torch.testing.assert_close(y1[:, i], model_in[:(4 * i)](x).mean(1))
                else:
                    torch.testing.assert_close(y1[:, (1 + idx) * 2:], y2)
                    for i in range(idx):
                        torch.testing.assert_close(y1[:, i * 2], model_in[:(4 * i)](x).mean(1))
        prefix_layers = layers_in[:2] + utils.lists.flatten_list_of_lists(components)

        do_debug = True
        if do_debug:
            mpa_model = torch.nn.Sequential(*prefix_layers)
            if any(is_relu):
                assert len(mpa_model) + num_layer_starts == len(model_in)
        suffix_layer = compute_suffix_layer(prefix_layers)
        mpa_layers = prefix_layers + [suffix_layer]
    return mpa_layers


def compute_suffix_layer(layers_in: List[torch.nn.Module]) -> torch.nn.Module:
    is_relu = [type(_) == torch.nn.ReLU for _ in layers_in]
    layer_starts = torch.argwhere(torch.tensor(is_relu)).flatten() - 1
    num_layer_starts = len(layer_starts)
    out_features = layers_in[-1].out_features
    lls = out_features - num_layer_starts

    suffix_layer = torch.nn.Linear(num_layer_starts + lls,
                                   num_layer_starts + 1,
                                   bias=False)
    lower_right = torch.zeros(num_layer_starts, lls)
    left = torch.eye(num_layer_starts)
    averager = torch.ones(1, lls) / lls
    bottom = torch.cat((torch.zeros(1, num_layer_starts), averager), 1)
    weight = torch.cat((torch.cat((left, lower_right), 1), bottom), 0)
    assert weight.shape == suffix_layer.weight.shape
    suffix_layer.weight = torch.nn.Parameter(weight)
    return suffix_layer


def _mapping_to_layers(layer_mappings: List[list]) -> List[torch.nn.Module]:
    subpool_orders = sorted(layer_mappings.keys())
    mappings = [layer_mappings[k] for k in subpool_orders]

    num_max_layers = len(layer_mappings)
    components = [None] * num_max_layers
    for idx in range(num_max_layers):
        # idx = 0
        curr_layer_idx = num_max_layers - idx - 1
        curr_layer_mapping = mappings[curr_layer_idx]
        if curr_layer_idx < num_max_layers - 1:
            next_layer_mapping = mappings[curr_layer_idx + 1]
            layer_dim = len(next_layer_mapping)
        else:
            layer_dim = 1 + max([x for _ in curr_layer_mapping for x in _])
        components[idx] = build_maxer_layer_general(layer_dim, curr_layer_mapping)
    layers = flatten_sequential_of_sequentials_to_layers(components)
    return layers


def _build_aggregator_layer(layer_mappings: Dict[int, list],
                            betas: torch.Tensor,
                            required_terms: List[int]) -> torch.nn.Linear:
    subpool_orders = sorted(layer_mappings.keys())
    subpool_orders1 = [1] + subpool_orders
    cols = [subpool_orders1.index(_) for _ in required_terms]

    # outputs each term in the penultimate layer, then weighted by betas
    # and summed further. add intercept as indep term.
    num_subpool_orders1 = len(layer_mappings) + 1
    aggregator_lin = torch.nn.Linear(num_subpool_orders1, 1, bias=True)

    beta0 = torch.zeros(aggregator_lin.weight.shape)
    beta0[0, cols] = betas[required_terms]
    aggregator_lin.weight = torch.nn.Parameter(beta0)
    aggregator_lin.bias = torch.nn.Parameter(betas[0])
    return aggregator_lin


def build_wholly_general_model(betas: torch.Tensor,
                               dim: int) -> torch.nn.Sequential:
    assert betas.numel() == dim + 1
    is_required_term = torch.logical_and(betas != 0, torch.arange(dim + 1) > 0)
    required_terms = torch.argwhere(is_required_term).flatten().tolist()
    layer_mappings = _build_layer_mappings_old(required_terms, dim)

    if 0 == len(layer_mappings):
        layer = torch.nn.Linear(in_features=dim, out_features=1, bias=False)
        layer.weight = torch.nn.Parameter(torch.ones(layer.weight.shape) / dim)
        mpaed_layers = [layer]
    else:
        initial_layers = _mapping_to_layers(layer_mappings)
        shimmed_layers = _shim_up_layers(initial_layers)
        mpaed_layers = _maxpool_average_layers(shimmed_layers)

    aggregator_layer = _build_aggregator_layer(layer_mappings,
                                               betas,
                                               required_terms)
    layers = mpaed_layers + [aggregator_layer]
    final_model = torch.nn.Sequential(*layers)
    return final_model


def _build_ords_for_layer_idx(layer_idx: int) -> List[int]:
    num_ords = 2 ** (layer_idx - 1)
    start = 2 ** (layer_idx - 1) + 1
    ords_for_layer_idx = list(range(start, start + num_ords))
    return ords_for_layer_idx


"""
A hierarchy of goodness: (1) iterative, (2) recursive, (3) while
"""


def magic(required_terms: List[int]) -> List[int]:
    assert required_terms == sorted(required_terms)
    assert required_terms[:2] == [0, 1]
    if [0, 1] == required_terms:  # base case
        out = required_terms
    else:
        cutpoint = 2 ** math.floor(math.log2(required_terms[-1] - 1))
        max_below_cutpoint = max(_ for _ in required_terms if _ <= cutpoint)
        healthy = (2 * max_below_cutpoint >= required_terms[-1])
        if healthy:
            out = magic(required_terms[:-1]) + [required_terms[-1]]
        else:
            to_pass = sorted(required_terms[:-1] + [math.ceil(required_terms[-1] / 2)])
            out = magic(to_pass) + [required_terms[-1]]
    return out


def _build_subpools_general(subsequent: List[tuple],
                            sz: int) -> List[tuple]:
    subpools = [(_[:sz], _[-sz:]) for _ in subsequent]
    return subpools


"""
Two possibilities: (1) It is required, and has all subsets guaranteed -> nothing really to do. 
                   (2) it was added, to be able to support a subsequent layer. 
                       Then it should be able to synthesize lower-order maxes of the same depth. 
"""


def get_subsets_of_sizes(superset: List[int],
                         sizes: List[int]) -> List[int]:
    dim = len(superset)
    subsets_of_size = []
    for size in sizes:
        subsets_of_size += list(itertools.combinations(superset, size))
    # subsets_of_size = sorted(subsets_of_size)
    expected_size = sum([math.comb(dim, _) for _ in sizes])
    assert expected_size == len(subsets_of_size)
    return subsets_of_size


def _build_curr_components(input_dim: int,
                           size_out: int,
                           layer_mapping: tuple) -> List[torch.nn.Module]:
    lin0 = torch.nn.Linear(input_dim, 4 * size_out, bias=False)
    lin1 = torch.nn.Linear(4 * size_out, size_out, bias=False)

    w0 = build_w0_general(input_dim, layer_mapping)
    w1 = build_w1(size_out)

    assert lin0.weight.shape == w0.shape
    assert lin1.weight.shape == w1.shape

    lin0.weight = torch.nn.Parameter(w0)
    lin1.weight = torch.nn.Parameter(w1)

    curr_components = [lin0, torch.nn.ReLU(), lin1]
    return curr_components


def build_layer_mappings_with_metadata(required_terms: List[int],
                                        dim: int) -> Tuple[list, dict]:
    xs = tuple([_ for _ in range(dim)])

    normalized_required_terms = magic(required_terms)
    nrt = torch.Tensor(normalized_required_terms).to(int)
    rrt = torch.Tensor(required_terms).to(int)  # raw required terms

    last_term = max(required_terms)
    depth = math.ceil(math.log2(last_term))
    n_term_layers = torch.ceil(torch.log2(nrt))
    r_term_layers = torch.ceil(torch.log2(rrt))
    torch.testing.assert_close(rrt[r_term_layers == depth],
                               nrt[n_term_layers == depth])

    layer_mappings = [None] * depth
    idents = [None] * depth
    orders = [None] * depth
    for idx in range(depth):
        # idx = 0
        # idx = 1
        idx_rev = depth - idx - 1
        n_prev_cols = (n_term_layers == idx_rev + 0)
        prev_ords = nrt[n_prev_cols].tolist()
        # terms that need to be computed at this layer
        max_prev_ord = max(prev_ords)

        if idx_rev == depth - 1:
            next_terms = []
        else:
            next_terms = idents[idx_rev + 1]
        r_curr_necc = rrt[r_term_layers == idx_rev + 1].tolist()
        curr_subsets = get_subsets_of_sizes(xs, r_curr_necc)
        to_append = sorted(set(curr_subsets) - set(next_terms))
        ordered_subsets = next_terms + to_append
        subpools = _build_subpools_general(ordered_subsets, max_prev_ord)
        curr_inputs = sorted(set(utils.lists.flatten_list_of_lists(subpools)))

        curr_mapping = [(curr_inputs.index(_[0]), curr_inputs.index(_[1]))
                        for _ in subpools]

        ords = [len(_) for _ in ordered_subsets]
        idents[idx_rev] = curr_inputs
        layer_mappings[idx_rev] = curr_mapping
        orders[idx_rev] = ords

    metadata = {
        "idents": idents,
        "orders": orders,
        "required_terms": required_terms
    }
    return layer_mappings, metadata


def _build_base_components(dim: int,
                           max_ord: int,
                           layer_mappings: List[tuple],) -> List[list]:
    depth = len(layer_mappings)
    mapping_lens = [len(_) for _ in layer_mappings]

    dim_out = math.comb(dim, max_ord)
    in_sizes = [dim] + mapping_lens
    out_sizes = mapping_lens + [dim_out]
    base_components = [None] * depth

    for idx, layer_mapping in enumerate(layer_mappings):
        input_dim = in_sizes[idx]
        size_out = out_sizes[idx]
        curr_components = _build_curr_components(input_dim, size_out, layer_mapping)
        base_components[idx] = curr_components
    return base_components


def torch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dtype_out = a.dtype
    assert b.dtype == dtype_out
    return torch.tensor(np.kron(a, b)).to(dtype_out)


def _shim_component(base_component: List[torch.nn.Module],
                    i: torch.Tensor) -> List[torch.nn.Module]:
    o = torch.ones(1, 1)
    o2 = torch.cat((+o, -o), 1)
    c = torch.tensor([[+1], [-1]]).to(torch.float32)

    layer0_in = base_component[0]
    assert type(base_component[1]) == torch.nn.ReLU
    layer1_in = base_component[2]
    num_avgs = i.shape[0]

    layer0_out = torch.nn.Linear(layer0_in.in_features,
                                 layer0_in.out_features + 2 * num_avgs,
                                 bias=False)
    layer1_out = torch.nn.Linear(layer0_in.out_features + 2 * num_avgs,
                                 layer1_in.out_features + 1 * num_avgs,
                                 bias=False)
    kill_layer = torch.nn.Linear(layer1_in.out_features + 1 * num_avgs,
                                 layer1_in.out_features,
                                 bias=False)
    row0 = i / i.sum(1, keepdims=True)
    if 0 == row0.numel():
        rkron = row0
    else:
        rkron = torch_kron(row0, c)

    # weight0_out = torch.cat((+row0, -row0, layer0_in.weight), 0)
    weight0_out = torch.cat((+rkron, layer0_in.weight), 0)
    okron = torch.kron(torch.eye(num_avgs), o2)

    top1 = torch.cat((okron, torch.zeros((num_avgs, layer1_in.weight.shape[1]))), 1)
    left1 = torch.zeros((layer1_in.weight.shape[0], 2 * num_avgs))

    weight1_out = torch.cat((top1, torch.cat((left1, layer1_in.weight), 1)), 0)
    kill_weight = torch.cat((torch.zeros((layer1_in.out_features, num_avgs)),
                             torch.eye(layer1_in.out_features)), 1)

    assert layer0_out.weight.shape == weight0_out.shape
    assert layer0_out.weight.dtype == weight0_out.dtype
    layer0_out.weight = torch.nn.Parameter(weight0_out)

    assert layer1_out.weight.shape == weight1_out.shape
    assert weight1_out.dtype == layer1_out.weight.dtype
    layer1_out.weight = torch.nn.Parameter(weight1_out)

    assert kill_layer.weight.shape == kill_weight.shape
    assert kill_layer.weight.dtype == kill_weight.dtype
    kill_layer.weight = torch.nn.Parameter(kill_weight)

    shimmed_component = [layer0_out, torch.nn.ReLU(), layer1_out, kill_layer]
    return shimmed_component


def _shim_final_component(last_layer: torch.nn.Linear,
                          i: torch.Tensor) -> List[torch.nn.Module]:
    num_avgs = i.shape[0]
    dim_out = last_layer.out_features
    expand_layer = torch.nn.Linear(in_features=dim_out,
                                   out_features=dim_out + num_avgs,
                                   bias=False)
    row = i / i.sum(1, keepdims=True)
    expand_weight = torch.cat((row, torch.eye(dim_out)), 0)
    assert expand_weight.shape == expand_layer.weight.shape
    assert expand_weight.dtype == expand_layer.weight.dtype
    expand_layer.weight = torch.nn.Parameter(expand_weight)

    kill_layer = torch.nn.Linear(in_features=dim_out + num_avgs,
                                 out_features=dim_out,
                                 bias=False)
    kill_weight = torch.cat((torch.zeros((dim_out, num_avgs)),
                             torch.eye(dim_out)), 1)
    assert kill_weight.shape == kill_layer.weight.shape
    assert kill_weight.dtype == kill_layer.weight.dtype
    kill_layer.weight = torch.nn.Parameter(kill_weight)

    shimmed_final_component = [expand_layer, kill_layer]
    return shimmed_final_component


def _add_shim_layers(dim: int,
                     base_components: List[list],
                     orders: List[list],
                     required_terms: List[int]) -> List[torch.nn.Module]:
    depth = len(orders)
    shimmed_components = [None] * (depth + 1)

    do_print = False
    # do_print = True
    # for idx, ords in enumerate(orders):
    for idx in range(depth):
        # idx = 0; ords = orders[idx]
        # idx = depth - 1; ords = orders[idx]
        base_component = base_components[idx]
        if 0 == idx:
            ords = [1] * dim
        else:
            ords = orders[idx - 1]
        this_layer_terms = sorted(set(ords).intersection(set(required_terms)))
        if do_print:
            print(idx, this_layer_terms)
        ord_counter = Counter(ords)
        for tlt in this_layer_terms:
            assert ord_counter[tlt] == math.comb(dim, tlt), "unexpected number of terms"
        # assert idx < depth - 1
        i = (torch.tensor(ords).reshape(-1, 1) == \
             torch.tensor(this_layer_terms).reshape(1, -1)).to(torch.float32).T
        shimmed_components[idx] = _shim_component(base_component, i)

    ords = orders[depth - 1]
    this_layer_terms = sorted(set(ords).intersection(set(required_terms)))
    i = (torch.tensor(ords).reshape(-1, 1) == \
         torch.tensor(this_layer_terms).reshape(1, -1)).to(torch.float32).T

    last_component = base_components[-1]
    last_layer = last_component[-1]
    shimmed_components[depth] = _shim_final_component(last_layer, i)
    return shimmed_components


def _finalize_layers(dim: int,
                     shimmed_components: List[list],
                     orders: List[list],
                     required_terms: List[int]) -> List[list]:
    c = torch.Tensor([[+1], [-1]])
    o = torch.ones(1, 1)
    o2 = torch.cat((+o, -o), 1)

    layers_in = utils.lists.flatten_list_of_lists(shimmed_components)
    model = torch.nn.Sequential(*layers_in)

    req_terms0 = sorted(set(required_terms) - {0})
    last_term = max(required_terms)
    depth = math.ceil(math.log2(last_term))

    components_out = [None] * (1 + depth)
    cum_term_count = 0
    for idx in range(depth):
        if 0 == idx:
            ords = [1] * dim
        else:
            ords = orders[idx - 1]
        this_layer_terms = sorted(set(ords).intersection(set(required_terms)))
        this_term_count = len(this_layer_terms)

        layer_start = idx * 4
        layer2_in = layers_in[layer_start + 2]
        layer4_in = layers_in[layer_start + 4]
        in_dim = layer2_in.in_features + 2 * cum_term_count
        middle_dim = layer2_in.out_features + cum_term_count
        out_dim = layer4_in.out_features + 2 * (cum_term_count + this_term_count)
        layer2_out = torch.nn.Linear(in_dim, middle_dim, bias=False)
        layer4_out = torch.nn.Linear(middle_dim, out_dim, bias=False)

        okron = torch.kron(torch.eye(cum_term_count), o2)
        top2 = torch.cat((okron, torch.zeros(cum_term_count, layer2_in.weight.shape[1])), 1)
        left2 = torch.zeros((layer2_in.weight.shape[0], 2 * cum_term_count))
        weight2_out = cat_top_left(top2, left2, layer2_in.weight)

        kronc = torch.kron(torch.eye(cum_term_count + this_term_count), c)
        zeros_right = torch.zeros(2 * (cum_term_count + this_term_count),
                                  layer4_in.weight.shape[1])
        top4 = torch.cat((kronc, zeros_right), 1)
        left4 = torch.zeros((layer4_in.weight.shape[0],
                             cum_term_count + this_term_count))
        weight4_out = cat_top_left(top4, left4, layer4_in.weight)

        assert weight2_out.shape == layer2_out.weight.shape
        assert weight2_out.dtype == layer2_out.weight.dtype
        layer2_out.weight = torch.nn.Parameter(weight2_out)

        assert weight4_out.shape == layer4_out.weight.shape
        assert weight4_out.dtype == layer4_out.weight.dtype
        layer4_out.weight = torch.nn.Parameter(weight4_out)
        if idx < depth - 1:
            component = [layer2_out, layer4_out, torch.nn.ReLU()]
        else:
            component = [layer2_out]
        cum_term_count += this_term_count
        components_out[idx] = component

    ords = orders[depth - 1]
    this_layer_terms = sorted(set(ords).intersection(set(required_terms)))
    this_term_count = len(this_layer_terms)
    total_terms = len(req_terms0)
    in_dim = components_out[idx][-1].out_features
    prev_terms = total_terms - this_term_count  # number of terms already computed
    assert in_dim == total_terms - this_term_count + len(ords)
    layer2_out = torch.nn.Linear(in_dim, total_terms, bias=False)
    i = (torch.tensor(ords).reshape(-1, 1) == \
         torch.tensor(this_layer_terms).reshape(1, -1)).to(torch.float32).T
    row = i / i.sum(1, keepdims=True)

    weight2_out = torch.cat((torch.cat((torch.eye(prev_terms), torch.zeros(prev_terms, len(ords))), 1),
                             torch.cat((torch.zeros(this_term_count, prev_terms), row), 1)), 0)
    assert layer2_out.weight.shape == weight2_out.shape
    assert layer2_out.weight.dtype == weight2_out.dtype
    layer2_out.weight = torch.nn.Parameter(weight2_out)

    components_out[-1] = [layer2_out]
    final_layers = list(model[:2]) + utils.lists.flatten_list_of_lists(components_out)
    return final_layers


def build_layers_from_mappings_and_metadata(dim: int,
                                             layer_mappings: List[tuple],
                                             metadata: Dict[str, Any]) -> List[torch.nn.Module]:
    required_terms = metadata["required_terms"]
    idents = metadata["idents"]
    orders = metadata["orders"]
    max_ord = max(required_terms)

    depth = len(layer_mappings)
    assert len(idents) == depth
    base_components = _build_base_components(dim, max_ord, layer_mappings)
    shimmed_components = _add_shim_layers(dim, base_components, orders, required_terms)
    all_layers_flat = _finalize_layers(dim, shimmed_components, orders, required_terms)
    return all_layers_flat


def get_aggregator_layer(betas: torch.Tensor) -> List[torch.nn.Module]:
    betas1 = betas[1:]
    cols = (betas1 != 0.0) | (0 == torch.arange(len(betas1)))
    num_nonzero = sum(cols)
    aggregator_lin = torch.nn.Linear(num_nonzero, 1, bias=True)

    beta_reduced = torch.zeros(aggregator_lin.weight.shape)
    beta_reduced[0, :] = betas1[cols]
    aggregator_lin.weight = torch.nn.Parameter(beta_reduced)
    aggregator_lin.bias = torch.nn.Parameter(betas[0])
    return aggregator_lin


def build_shallowest_network(betas: torch.Tensor) -> torch.nn.Sequential:
    dim = betas.numel() - 1
    cond = torch.logical_or(betas != 0, torch.arange(dim + 1) < 2)
    required_terms = torch.argwhere(cond).flatten().tolist()
    layer_mappings, metadata = build_layer_mappings_with_metadata(required_terms, dim)
    layers = build_layers_from_mappings_and_metadata(dim, layer_mappings, metadata)
    aggregator_layer = get_aggregator_layer(betas)
    all_layers = layers + [aggregator_layer]
    shallowest_network = torch.nn.Sequential(*all_layers)
    if False:
        csn = canonicalize_network(shallowest_network)
        mapping_lens = [len(_) for _ in layer_mappings]

    do_check = True
    if do_check:
        if (betas[:-1] == 0).all() and (betas[-1] == 1.0):
            x = torch.rand(3, dim)
            actual = shallowest_network(x).flatten()
            expected = x.max(1).values
            torch.testing.assert_close(actual, expected)
    return shallowest_network


if __name__ == "__main__":
    dim = 9
    # rset = [0, 2, 4, 8]
    rset = [0, 2, 3, 8]
    betas = torch.zeros(dim+1,)
    betas[rset] = torch.rand(len(rset),)
    model1 = build_wholly_general_model(betas, dim)
    model2 = build_shallowest_network(betas)

    model1f = canonicalize_network(model1)
    model2f = canonicalize_network(model2)

    x = torch.randn((11, dim))

    y1 = model1f(x)
    y2 = model2f(x)
