
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math
import argparse
import logging
import pprint
import os
import pickle
import warnings
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
import hydra
from omegaconf import DictConfig, OmegaConf

import utils.logging
import utils.config

import depth_analysis
import utils.plotting
import networks
import utils.argparsers
import utils.path_config
import ntk

FigAx = Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

paths = utils.path_config.get_paths()

# warnings.filterwarnings("error")

logging_level = 15

logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

log_dir_str = paths["logs"]
standard_streamhandler = utils.logging.get_standard_streamhandler()
standard_filehandler = utils.logging.get_standard_filehandler(log_dir_str)

logger.addHandler(standard_streamhandler)
logger.addHandler(standard_filehandler)

for handler in logger.handlers:
    logger.info(handler)


def _get_x(dataset_name: str,
           num_rows: int,
           dim: int,
           dtype: torch.dtype) -> torch.Tensor:
    if "unitcube_uniform" == dataset_name:
        x = torch.rand(num_rows, dim, dtype=dtype)
    elif "gaussian" == dataset_name:
        x = torch.randn(num_rows, dim, dtype=dtype)
    elif "unitcube_dirichlet" == dataset_name:
        alpha = np.ones(dim,) / dim
        dirichlet_sample = torch.tensor(np.random.dirichlet(alpha, size=num_rows), dtype=dtype)
        x = dirichlet_sample
    elif "unitcube_sobol" == dataset_name:
        # https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html
        se = torch.quasirandom.SobolEngine(dim, scramble=True)
        sobol_sample = se.draw(num_rows).to(dtype)
        x = sobol_sample
    else:
        raise ValueError(f"dataset_name {dataset_name} not configured")
    return x


def _init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def _init_weights_kaiming(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def generate_results(fitting_results_mult: list,
                     mults: List[float]) -> Dict[str, Any]:
    num_mults = len(mults)
    optimize_criterion_values = dict(train=torch.full((num_mults,), math.nan),
                                     test=torch.full((num_mults,), math.nan))
    evaluate_criterion_values = dict(train=torch.full((num_mults,), math.nan),
                                     test=torch.full((num_mults,), math.nan))
    for idx, mult in enumerate(mults):
        # idx = 0; mult = mults[idx]
        frm = fitting_results_mult[idx]
        terminal_epoch = frm["terminal_epoch"]
        ocv = frm['optimize_criterion_values']
        ecv = frm["evaluate_criterion_values"]

        optimize_criterion_values["train"][idx] = ocv["train"][terminal_epoch]
        optimize_criterion_values["test"][idx] = ocv["test"][terminal_epoch]

        evaluate_criterion_values["train"][idx] = ecv["train"][terminal_epoch]
        evaluate_criterion_values["test"][idx] = ecv["test"][terminal_epoch]

    results = {
        "evaluate_criterion_values": evaluate_criterion_values,
        "optimize_criterion_values": optimize_criterion_values,
        "mults": mults
    }
    return results


def _mutate_dataloader_by_model(dataloader: torch.utils.data.DataLoader,
                                model: torch.nn.Module,
                                device: torch.device) -> torch.utils.data.DataLoader:
    dataloader_kwargs = {
        "batch_size": dataloader.batch_size,
        "shuffle": True
    }
    new_y_batches = [None] * len(dataloader)
    for batch_idx, (x, y) in enumerate(dataloader):
        # (x, y) = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        new_y_batches[batch_idx] = (y - model(x).reshape(y.shape)).detach()
    new_y = torch.cat(new_y_batches)
    dataset = torch.utils.data.TensorDataset(dataloader.dataset.tensors[0], new_y)
    new_dataloader = torch.utils.data.DataLoader(dataset,
                                                 **dataloader_kwargs)
    return new_dataloader


def initialize_weights(model: torch.nn.Module,
                       initialization_name: str) -> torch.nn.Module:
    if initialization_name is None:
        pass
    elif "xavier" == initialization_name:
        model.apply(_init_weights_xavier)
    elif "kaiming" == initialization_name:
        model.apply(_init_weights_kaiming)
    else:
        raise ValueError(f"initialization_name = {initialization_name} not configured")
    return model


def test_loop(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              optimize_criterion: Callable,
              evaluate_criterion: Callable,
              device: torch.device) -> Tuple[float, float]:
    dataloader_len = len(dataloader)

    sizes = torch.full((dataloader_len,), math.nan)
    optimize_losses = torch.full((dataloader_len,), math.nan)
    evaluate_losses = torch.full((dataloader_len,), math.nan)

    for batch_idx, (x, y) in enumerate(dataloader):
        # (x, y) = next(iter(dataloader))
        x, y = x.to(device), y.to(device)

        y_pred = model(x).flatten()
        evaluate_loss = evaluate_criterion(y_pred, y)
        optimize_loss = optimize_criterion(y_pred, y)
        sizes[batch_idx] = y.shape[0]
        if False:
            print(y[:3])
            print(y_pred[:3])
            print(x[:3, :])
        evaluate_losses[batch_idx] = evaluate_loss.item()
        optimize_losses[batch_idx] = optimize_loss.item()

    optimize_loss = (sizes * optimize_losses).sum() / sizes.sum()
    evaluate_loss = (sizes * evaluate_losses).sum() / sizes.sum()
    return optimize_loss, evaluate_loss


def train_loop(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               optimize_criterion: Callable,
               evaluate_criterion: Callable,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    train_dataloader_len = len(train_dataloader)
    train_sizes = torch.full((train_dataloader_len,), torch.nan)
    optimize_losses = torch.full((train_dataloader_len,), torch.nan)
    evaluate_losses = torch.full((train_dataloader_len,), torch.nan)

    for batch_idx, (x, y) in enumerate(train_dataloader):
        # (x, y) = next(iter(dataloader))
        x, y = x.to(device), y.to(device)

        y_pred = model(x).flatten()
        optimize_loss_value = optimize_criterion(y_pred, y)

        optimizer.zero_grad()
        optimize_loss_value.backward()
        optimizer.step()

        evaluate_loss_value = evaluate_criterion(y_pred, y)

        train_sizes[batch_idx] = y.shape[0]
        optimize_losses[batch_idx] = optimize_loss_value.item()
        evaluate_losses[batch_idx] = evaluate_loss_value.item()

    optimize_loss_value = (train_sizes * optimize_losses).sum() / train_sizes.sum()
    evaluate_loss_value = (train_sizes * evaluate_losses).sum() / train_sizes.sum()
    return optimize_loss_value, evaluate_loss_value


def do_fitting(cfg_training: DictConfig,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               optimize_criterion: Callable,
               evaluate_criterion: Callable,
               optim_class: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, Any]:
    num_epochs = cfg_training.num_epochs
    patience_epochs = cfg_training.patience_epochs
    min_improvement = cfg_training.min_improvement

    optimizer = optim_class(model.parameters())
    optimize_criterion_values = dict(test=torch.full((num_epochs,), math.nan),
                                     train=torch.full((num_epochs,), math.nan))
    evaluate_criterion_values = dict(test=torch.full((num_epochs,), math.nan),
                                     train=torch.full((num_epochs,), math.nan))
    best_err = math.inf
    epoch_iterator = range(num_epochs)

    # nr = 3
    # dim = model[0].in_features
    # x = torch.rand(nr, dim, device=device)
    # test_x = x
    for epoch_idx in epoch_iterator:
        optimize_loss_value, evaluate_loss_value = train_loop(model,
                                                              train_dataloader,
                                                              optimize_criterion,
                                                              evaluate_criterion,
                                                              optimizer,
                                                              device)
        optimize_criterion_values["train"][epoch_idx] = optimize_loss_value
        evaluate_criterion_values["train"][epoch_idx] = evaluate_loss_value

        optimize_loss_value, evaluate_loss_value = test_loop(model,
                                                          test_dataloader,
                                                          optimize_criterion,
                                                          evaluate_criterion,
                                                          device)

        optimize_criterion_values["test"][epoch_idx] = optimize_loss_value
        evaluate_criterion_values["test"][epoch_idx] = evaluate_loss_value

        test_err = evaluate_loss_value
        best_err = min(best_err, test_err)

        e0 = max(0, epoch_idx - patience_epochs)

        improvements = evaluate_criterion_values["test"][e0:epoch_idx] - best_err
        stop_early_loss_improvement = epoch_idx >= patience_epochs and \
                                      (min_improvement is not None) and \
                                      max(improvements) < min_improvement
        if stop_early_loss_improvement:
            break
    if False:
        xs = np.arange(epoch_idx)
        ys = optimize_criterion_values["test"][:epoch_idx].numpy()
        plt.plot(xs, ys)

    fitting_results = {
        "optimize_criterion_values": optimize_criterion_values,
        "evaluate_criterion_values": evaluate_criterion_values,
        "model": model,
        "terminal_epoch": epoch_idx
    }
    return fitting_results


def get_criterion(criterion_name: str) -> Callable:
    if criterion_name == "linf":
        criterion = lambda y1, y2: torch.linalg.norm(y1.squeeze() - y2.squeeze(),
                                                     ord=math.inf)
    elif "l2" == criterion_name:
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"'{criterion_name}' not configured")
    return criterion


def model_name_to_betas(dim: int,
                        model_name: str) -> torch.Tensor:
    if "maxer" == model_name:
        betas = torch.zeros((dim + 1,))
        betas[-1] = 1.0
    elif "bigapprox" == model_name:
        ks = list(range(dim))
        soln = depth_analysis.get_dk_situation(dim, ks)
        betas = torch.zeros((dim + 1,))
        try:
            betas[:-1] = soln["argmin"].clone().detach()
        except:
            betas[:-1] = soln["argmin"]
    elif "mediumapprox" == model_name:
        ks = [0, 1, dim - 2, dim - 1]
        soln = depth_analysis.get_dk_situation(dim, ks)
        betas = torch.zeros((dim + 1,))
        betas[:-1] = soln["argmin"].clone().detach()
    elif "smallapprox" == model_name:
        ks = [0, 1, dim - 1]
        soln = depth_analysis.get_dk_situation(dim, ks)
        betas = torch.zeros((dim + 1,))
        betas[:-1] = soln["argmin"].clone().detach()
    else:
        raise ValueError(f"I do not know about model_name {model_name}")
    return betas


def build_response(x: torch.Tensor) -> torch.Tensor:

    x_max = x.max(1).values
    response = []
    return response


def _f(x: torch.Tensor,
       maxer_model: torch.nn.Module,
       device: torch.device) -> torch.Tensor:
    device_in = x.device
    split_size = 300
    num_components = int(math.ceil(x.shape[0] / split_size))
    components = [None] * num_components
    for idx, x_ in enumerate(x.split(split_size)):
        x_ = x_.to(device)
        with torch.no_grad():
            components[idx] = (x_.max(1).values - maxer_model(x_).flatten()).to(device_in)
    y = torch.cat(components)
    return y


def get_dataloaders(cfg_data: DictConfig,
                    cfg_dataloader: DictConfig,
                    dtype: torch.dtype,
                    device: torch.device) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dim = cfg_data.dim
    dataset_name = cfg_data.dataset_name
    num_rows_train = cfg_data.num_rows
    num_rows_test = cfg_data.num_rows

    x_train = _get_x(dataset_name, num_rows_train, dim, dtype)
    x_test = _get_x(dataset_name, num_rows_test, dim, dtype)

    if "max" == cfg_data.response_variable_name:
        y_train = x_train.max(1).values
        y_test = x_test.max(1).values
    elif "relative" == cfg_data.response_variable_name:
        x_train_top2 = torch.topk(x_train, 2, dim=1, sorted=True).values
        x_test_top2 = torch.topk(x_test, 2, dim=1, sorted=True).values

        y_train = (x_train_top2[:, 0] - x_train_top2[:, 1]) / dim
        y_test = (x_test_top2[:, 0] - x_test_top2[:, 1]) / dim
    elif "mean" == cfg_data.response_variable_name:
        y_train = x_train.mean(1)
        y_test = x_test.mean(1)
    else:
        raise ValueError(f"Do not know {cfg_data.response_variable_name}")

    if False:
        x_train_top2 = torch.topk(x_train, 2, dim=1, sorted=True).values
        x_test_top2 = torch.topk(x_test, 2, dim=1, sorted=True).values

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    dataloader_kwargs = {
        "batch_size": cfg_dataloader.batch_size,
        "shuffle": cfg_dataloader.shuffle
    }
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    return train_dataloader, test_dataloader


def get_model(dim: int, cfg_model: DictConfig) -> torch.nn.Sequential:
    if "single_layer" == cfg_model.model_name:
        hidden_width = cfg_model.single_layer_width
        model = torch.nn.Sequential(torch.nn.Linear(dim, hidden_width),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_width, 1))
    else:
        model_name = cfg_model.model_name
        betas = model_name_to_betas(dim, model_name)
        model = networks.canonicalize_network(networks.build_shallowest_network(betas))
    return model


def get_optim_class(cfg_optimizer: DictConfig) -> torch.optim.Optimizer:
    if "adam" == cfg_optimizer.optimizer_name:
        optim_kwargs = {"lr": 0.005,
                        "betas": (0.9, 0.999)}
        optim_class = functools.partial(torch.optim.Adam, **optim_kwargs)
    elif "adamw" == cfg_optimizer.optimizer_name:
        # optim_kwargs = {}
        # optim_class = functools.partial(torch.optim.AdamW, **optim_kwargs)
        optim_class = torch.optim.AdamW
    elif "sgd" == cfg_optimizer.optimizer_name:
        optim_class = torch.optim.SGD
    else:
        raise ValueError(f"{cfg_optimizer.optimizer_name} not configured")

    return optim_class


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None,
            config_path="config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device, dtype = utils.config.setup(cfg)
    experiment_ident = cfg.experiment.ident
    logger.info(f"running experiment {experiment_ident}")
    train_dataloader, test_dataloader = get_dataloaders(cfg.data,
                                                        cfg.dataloader,
                                                        dtype,
                                                        device)
    optim_class = get_optim_class(cfg.optimizer)
    optimize_criterion = get_criterion(cfg.training.optimize_criterion_name)
    evaluate_criterion = lambda y1, y2: torch.linalg.norm(y1.squeeze() - y2.squeeze(),
                                                          ord=math.inf)
    model = get_model(cfg.data.dim, cfg.model)
    model_hidden_layer_widths = [_.out_features for _ in model if type(_) == torch.nn.Linear][:-1]
    mults = cfg.model.mults

    num_mults = len(mults)
    # fitting_results_mult = [None] * num_mults
    fitting_results_mult = dict()
    for idx, mult in enumerate(mults):
        # idx = 3; mult = mults[idx]
        # idx = 4; mult = mults[idx]
        # idx = 5; mult = mults[idx]
        # idx = len(mults) - 6; mult = mults[idx]
        hidden_layer_widths = [int(math.ceil(mult * _)) for _ in
                               model_hidden_layer_widths]
        logger.info(f"hidden_layer_widths = {hidden_layer_widths}")
        layers = networks.build_relu_layers(cfg.data.dim,
                                            hidden_layer_widths,
                                            output_dim=1,
                                            include_bias=cfg.model.include_bias)
        model = torch.nn.Sequential(*layers).to(dtype).to(device)
        model = initialize_weights(model, cfg.training.initialization_name)
        num_params = count_parameters(model)

        if False:
            done = False
            iter_num = 0
            # max_iter = 50000
            max_iter = 5000
            print_every = 100
            # x, y = train_dataloader.dataset.tensors
            # x, y = x.to(device), y.to(device)
            optimizer = torch.optim.Adam(model.parameters())
            # optimizer = torch.optim.AdamW(model.parameters())
            while not done:
                for (x, y) in train_dataloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x).flatten()
                    optimize_loss_value = optimize_criterion(y_pred, y)

                    optimizer.zero_grad()
                    optimize_loss_value.backward()
                    optimizer.step()

                iter_num += 1
                if 0 == iter_num % print_every:
                    print(iter_num, optimize_loss_value)
                done = iter_num > max_iter

                if False:
                    sample_ntk = ntk.empirical_ntk_ntk_vps(model, x, x).detach()
                    sample_ntk_2d = sample_ntk.squeeze()
                    assert (sample_ntk_2d > 0).all()
                    print(sample_ntk_2d.min())

        fitting_results_mult[mult] = do_fitting(cfg.training,
                                                train_dataloader,
                                                test_dataloader,
                                                model,
                                                optimize_criterion,
                                                evaluate_criterion,
                                                optim_class,
                                                device)
        logger.info(f"[{idx} / {num_mults}] num_params = {num_params} complete")
        if False:
            fitting_results_mult[mult]["optimize_criterion_values"]
            model = fitting_results_mult[mult]["model"]
            nr = 33
            x = torch.rand(nr, cfg.data.dim, device=device)
            y_pred = model(x)
            y = x.max(1).values
            err = optimize_criterion(y, y_pred)

    pickle_filename = f"{cfg.prng.seed}.pkl"
    pickle_filedir = os.path.join(paths["results"], experiment_ident, cfg.model.model_name)
    os.makedirs(pickle_filedir, exist_ok=True)
    pickle_fullfilename = os.path.join(pickle_filedir, pickle_filename)
    logger.info(f"Saving results to {pickle_fullfilename}")
    with open(pickle_fullfilename, 'wb') as handle:
        pickle.dump(fitting_results_mult, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Done")


@hydra.main(config_name="config",
            config_path=".",
            version_base=None)
def caller_default(cfg: DictConfig) -> None:
    return_hydra_config = True
    overrides = []
    overrides += [f"prng.seed=5"]
    cfg = hydra.compose("config", overrides, return_hydra_config)
    main(cfg)


def caller(base_dir: str,
           experiment_dir: str) -> None:
    overrides_fullfilename = os.path.join(base_dir, experiment_dir, ".hydra", "overrides.yaml")
    out_path = os.path.join(base_dir, experiment_dir)
    return_hydra_config = True

    f = os.path.join(base_dir, experiment_dir, ".hydra")
    with hydra.initialize_config_dir(f, "app", None):
        overrides = list(OmegaConf.load(overrides_fullfilename))
        overrides += [f"++paths.out={out_path}"]
        cfg = hydra.compose("config", overrides, return_hydra_config)
        main(cfg)


if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--base_dir", type=str)
    # parser.add_argument("--experiment_dir", type=str)
    #
    # args = parser.parse_args()
    # base_dir = args.base_dir
    # experiment_dir = args.experiment_dir
    #
    # if base_dir is None and experiment_dir is None:
    #     caller_default()
    # else:
    #     caller(base_dir, experiment_dir)
