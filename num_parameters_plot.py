
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import functools
import math
import os
import pickle
import logging
import warnings
import sys
from typing import Callable, Iterable, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import seaborn
import torch
import numpy as np
import matplotlib.lines
import matplotlib.pyplot as plt

import tools
import networks
import depth_analysis

import utils.plotting
import utils.path_config
import utils.config
import utils.logging

# warnings.filterwarnings("error")

standard_streamhandler = utils.logging.get_standard_streamhandler()

logging_level = 15

logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
logger.addHandler(standard_streamhandler)

FigAx = Tuple[matplotlib.figure.Figure, np.array]

palette_name = "deep"
w_o_h = 1.5
# if False:
#     palette = seaborn.color_palette(palette_name, 8)
#     seaborn.palplot(palette)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def do_model_size_analysis(dim: int) -> dict:
    paths = utils.path_config.get_paths()
    model_filename = f"model_size_analysis{dim}.pkl"
    model_filedir = paths["results"]
    model_fullfilename = os.path.join(model_filedir, model_filename)

    if os.path.exists(model_fullfilename):
        with open(model_fullfilename, 'rb') as fh:
            model_size_analysis = pickle.load(fh)
        logger.info(f"Loading from '{model_fullfilename}'")
    else:
        z = torch.zeros(1)
        s = tuple(range(2, dim))
        rsets = tools.powerset(s)
        if () in rsets:
            rsets.remove(())
        num_sets = len(rsets)

        parameter_counts = [None] * num_sets
        criterions = [None] * num_sets
        incl_d1 = [None] * num_sets
        ks_lens = [None] * num_sets

        # ks_sets = [[0, 1]] + [[0, 1] + list(_) for _ in rsets]
        ks_sets = [[0, 1] + list(_) for _ in rsets]
        # for idx, ks0 in enumerate(rsets):
        for idx, ks in enumerate(ks_sets):
            # idx = 21; ks = ks_sets[idx]
            # idx = 29; ks = ks_sets[idx]
            # idx = 0; ks = rsets[idx]
            soln = depth_analysis.get_dk_situation(dim, ks)
            betas = torch.cat((soln["argmin"], z)).to(torch.float32)

            shallowest_network = networks.build_shallowest_network(betas, dim)
            shallowest_network_fused = networks.canonicalize_network(shallowest_network)
            parameter_counts[idx] = count_parameters(shallowest_network_fused)

            max_r = max(ks)
            num_relu = sum([type(_) == torch.nn.ReLU for _ in shallowest_network_fused])
            assert num_relu == int(math.log2(max_r - 1)) + 1
            print(idx, max_r, num_relu)

            criterions[idx] = soln["criterion"]
            incl_d1[idx] = dim - 1 in ks
            ks_lens[idx] = len(ks)

        assert 2 ** (dim - 2) - 1 == len(criterions)
        model_size_analysis = {
            "parameter_counts": parameter_counts,
            "criterions": criterions,
            "incl_d1": incl_d1,
            "ks_lens": ks_lens,
            "dim": dim
        }
        with open(model_fullfilename, 'wb') as fh:
            pickle.dump(model_size_analysis, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Dumping to '{model_fullfilename}'")
    return model_size_analysis


def model_size_error_scatterplot(model_size_analysis: dict,
                                 fig_scl: float) -> FigAx:
    parameter_counts = torch.tensor(model_size_analysis["parameter_counts"])
    criterions = torch.tensor(model_size_analysis["criterions"])

    ks_lens = torch.tensor(model_size_analysis["ks_lens"])
    rset_sizes = sorted(set(ks_lens.tolist()))
    markersize = 2
    figsize = (w_o_h * fig_scl, fig_scl)
    logger.info(f"figsize = {figsize}")
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax = axs[0, 0]
    palette = seaborn.color_palette(palette_name, len(rset_sizes))

    for idx, s in enumerate(rset_sizes):
        # d = min(ks_lens) + 1
        # idx = 0; s = rset_sizes[idx]
        # inds = [s == _ for _ in ks_lens]
        inds = s == ks_lens
        col = palette[idx]
        ax.semilogx(criterions[inds],
                    parameter_counts[inds], ".", color=col,
                    markersize=markersize)
    fig.tight_layout(pad=.2)
    return fig, axs


def model_index_size_plot(model_size_analysis: dict,
                          fig_scl: float) -> FigAx:
    dim = model_size_analysis["dim"]
    parameter_counts = model_size_analysis["parameter_counts"]
    ks_lens = model_size_analysis["ks_lens"]
    n = len(ks_lens)
    diff_inds = torch.tensor(ks_lens).diff()
    vs = (1 + torch.argwhere(diff_inds).flatten()).tolist()

    fig, axs = plt.subplots(1, 1,
                            figsize=(w_o_h * fig_scl, fig_scl),
                            squeeze=False)
    linewidth = .9
    ax = axs[0, 0]
    ax.plot(parameter_counts, linewidth=linewidth, color="k")

    ylim = ax.get_ylim()
    a = .1
    y_above = a * ylim[0] + (1 - a) * ylim[1]
    y_below = (1 - a) * ylim[0] + a * ylim[1]

    assert len(vs) == dim - 3
    palette = seaborn.color_palette(palette_name, dim - 3)

    for idx, v in enumerate(vs):
        # idx = 0; v = vs[idx]
        # idx = 1; v = vs[idx]
        # ax.axvline(v)
        if v <= 3 * n // 4:
            y = y_above
        else:
            y = y_below
        if 0 == idx:
            x0 = 0
        else:
            x0 = vs[idx - 1]
        x = (x0 + v) / 2
        col = palette[idx]
        # print(x0, col)
        xy = (x0, ylim[0])
        rect = matplotlib.patches.Rectangle(xy,
                                            v - x0,
                                            ylim[1] - ylim[0],
                                            linewidth=1,
                                            facecolor=col,
                                            alpha=.9)
        ax.add_patch(rect)
        if (idx > 0) and (idx < len(vs) - 1):
            ax.text(x, y, f"{ks_lens[vs[idx]]}",
                    color="k", ha='center')
    # ax.set_xlim(*xlim)
    ax.set_xlim(0, n - 1)
    # fig.tight_layout()
    fig.tight_layout(pad=.2)
    return fig, axs


def get_model_size_for_ks_builder(dim: int, ks_builder: Callable) -> List[int]:
    ks = ks_builder(dim)
    layer_mappings, metadata = networks.build_layer_mappings_with_metadata(ks, dim)

    mapping_lens = [len(_) for _ in layer_mappings]
    idealized = [4 * _ for _ in mapping_lens]
    model_size = idealized
    return model_size


def _widths_versus_dim_plot(xs: np.array,
                            all_widths_data_mat: np.array,
                            r_title: str,
                            do_legend: bool,
                            figsize: Tuple[float, float]) -> Tuple[FigAx, str]:
    linewidth = .9
    marker_size = 1.4
    max_depth = all_widths_data_mat.shape[1]
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    lines = axs[0, 0].semilogy(xs, all_widths_data_mat, "o-", linewidth=linewidth, ms=marker_size)

    axs[0, 0].set_xlabel(f"$d$")
    # axs[0, 0].set_title(f"$R = {r_title}$")
    axs[0, 0].grid(True)
    axs[0, 0].set_title(f"$R = {r_title}$")
    # axs[0, 0].set_ylabel("Layer width")
    if do_legend:
        # axs[0, 0].legend(lines, list(range(1, max_depth + 1)), title="layer")
        axs[0, 0].legend(lines, list(range(1, max_depth + 1)))
    fig.tight_layout()
    desc = f"""
    This plot shows the layer widths for all networks depths {xs.min()} -> {xs.max()} [x axis] 
    This is for the R = {r_title} features.
    """
    return fig, axs, desc


@functools.lru_cache
def widths_versus_dim_analysis(depth: int,
                               ks_builder: Callable) -> np.ndarray:
    dims = 2 ** (depth - 1) + 2 + np.arange(2 ** (depth - 1))
    sizes_actual = np.ndarray((len(dims), depth))
    for idx, dim in enumerate(dims):
        # idx = 1; dim = dims[idx]
        ms = get_model_size_for_ks_builder(dim, ks_builder)
        assert depth == int(math.ceil(math.log2(dim - 1)))
        sizes_actual[idx] = np.array(ms)
        logger.info(f"  dim = {dim} complete")
    return sizes_actual


def plot_model_size_analysis(model_size_analysis: dict,
                             fig_scl: float) -> Tuple[list, list]:
    dim = model_size_analysis["dim"]

    all_figs = []
    all_idents = []

    fig, axs = model_index_size_plot(model_size_analysis, fig_scl)
    all_figs += [fig]
    all_idents += [f"model_index_size_{dim}"]

    fig, axs = model_size_error_scatterplot(model_size_analysis, fig_scl)
    all_figs += [fig]
    all_idents += [f"model_size_error_{dim}"]

    return all_figs, all_idents


def build_num_parameters_plots(dim: int, cfg_plot: DictConfig):
    fig_scl = cfg_plot.fig_scl
    fig_format = cfg_plot.fig_format

    paths = utils.path_config.get_paths()
    filepath = paths["plots"]
    model_size_analysis = do_model_size_analysis(dim)
    # fig_format = "pdf"

    all_figs, all_idents = plot_model_size_analysis(model_size_analysis, fig_scl)

    # fig, axs = model_size_scatterplot(model_size_analysis, fig_scl)
    for fig, ident in zip(all_figs, all_idents):
        fig_path = utils.plotting.smart_save_fig(fig, ident, fig_format, filepath)
        logger.info(f"Saving {fig_path}")


def do_width_versus_dim_plot_for_rstr(r_str: str,
                                      depths: List[int],
                                      do_legend: bool,
                                      figsize: Tuple[float, float],
                                      fig_format: str):
    if r_str == "01d1":
        ks_builder = lambda d: sorted(set([0, 1, d - 1]))
        r_title = "\\{0, 1, d - 1\\}"
    elif r_str == "01d2d1":
        ks_builder = lambda d: sorted(set([0, 1, d - 2, d - 1]))
        r_title = "\\{0, 1, d - 2, d - 1\\}"
    elif r_str == "all_subsets":
        ks_builder = lambda d: sorted(set(list(range(d))))
        r_title = "\\{0, 1, 2, ..., d - 2, d - 1\\}"

    max_depth = max(depths)
    all_dimensions = [None] * len(depths)
    all_widths_data = [None] * len(depths)
    for idx, depth in enumerate(depths):
        # idx = 0; depth = depths[idx]
        widths_data = widths_versus_dim_analysis(depth, ks_builder)
        dimensions = 2 ** (depth - 1) + 2 + np.arange(2 ** (depth - 1))
        all_widths_data[idx] = widths_data
        all_dimensions[idx] = dimensions
        logger.info(f"depth = {depth} complete")

    _ = all_widths_data[0]
    to_cat = [np.hstack((_, np.full((_.shape[0], max_depth - depth), math.nan))) for
              _, depth in zip(all_widths_data, depths)]
    all_widths_data_mat = np.vstack(to_cat)
    xs = np.concatenate(all_dimensions)

    fig, axs, desc = _widths_versus_dim_plot(xs, all_widths_data_mat, r_title, do_legend, figsize)

    paths = utils.path_config.get_paths()
    filepath = paths["plots"]
    ident = f"width_versus_dim{r_str}"

    fig_path = utils.plotting.smart_save_fig(fig, ident, fig_format, filepath)
    logger.info(f"Saving '{fig_path}'")


def build_widths_versus_dim_plots(cfg_plot: DictConfig):
    fig_scl = cfg_plot.fig_scl
    fig_format = cfg_plot.fig_format

    # extra_scl = 2 / 3
    # extra_scl = 3 / 4
    # extra_scl = 4 / 5
    extra_scl = .9
    figsize = (1.25 * fig_scl * extra_scl, fig_scl * extra_scl)
    logger.info(f"figsize = {figsize}")

    depths1 = [2, 3, 4, 5]
    depths2 = [2, 3, 4]
    do_width_versus_dim_plot_for_rstr("01d1", depths1, True, figsize, fig_format)
    do_width_versus_dim_plot_for_rstr("01d2d1", depths1, False, figsize, fig_format)
    # do_width_versus_dim_plot_for_rstr("all_subsets", depths2, figsize, fig_format)


@hydra.main(version_base=None,
            config_path="config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device, dtype = utils.config.setup(cfg)

    cfg_plot = cfg.plots
    fig_format = cfg_plot.fig_format
    if "pgf" == fig_format:
        utils.plotting.initialise_pgf_plots(cfg_plot.texsystem,
                                            cfg_plot.font_family)
    build_widths_versus_dim_plots(cfg_plot)

    dim = 12
    build_num_parameters_plots(dim, cfg_plot)


if __name__ == "__main__":
    main()
