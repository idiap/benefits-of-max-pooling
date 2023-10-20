
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import warnings
import datetime as dt
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import torch
import matplotlib
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import utils.logging
import utils.config
import utils.plotting

import empirical_error
import order_statistics
import utils.plotting
import utils.argparsers
import utils.path_config

FigAx = Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
warnings.filterwarnings("error")
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


def get_r_str(model_name: str) -> str:
    if "smallapprox" == model_name:
        r_str = "$R = \\{0, 1, d - 1\\}$"
    elif "mediumapprox" == model_name:
        r_str = "$R = \\{0, 1, d - 2, d - 1\\}$"
    elif "bigapprox" == model_name:
        r_str = "$R = \\{0, 1, ..., d - 2, d - 1\\}$"
    else:
        raise ValueError(f"do not know {model_name}")
    return r_str


def get_theoretical_err(model_name: str, dim: int) -> float:
    if "bigapprox" == model_name:
        theoretical_err = 1 / 2 ** dim
    elif "mediumapprox" == model_name:
        theoretical_err = 1 / dim ** 2
    elif "smallapprox" == model_name:
        theoretical_err = 1 / (2 * (dim + 1))
    else:
        raise ValueError(f"do not know {model_name}")
    return theoretical_err


@hydra.main(version_base=None,
            config_path="config",
            config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device, dtype = utils.config.setup(cfg)
    dataloader, _ = empirical_error.get_dataloaders(cfg.data,
                                                    cfg.dataloader,
                                                    dtype,
                                                    device)
    # evaluate_criterion = lambda y1, y2: (y1.squeeze() - y2.squeeze()).abs().mean()
    model = empirical_error.get_model(cfg.data.dim, cfg.model)
    r_str = get_r_str(cfg.model.model_name)
    betas = empirical_error.model_name_to_betas(cfg.data.dim, cfg.model.model_name)

    shapes = torch.empty((len(dataloader),))
    losses = torch.empty((len(dataloader),))

    diffs = [None] * len(dataloader)
    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x).flatten()
        # diffs[idx] = (y_pred - y).abs()
        diffs[idx] = (y_pred - y)
        # loss_value = evaluate_criterion(y_pred, y)
        # shapes[idx] = y.shape[0]
        # losses[idx] = loss_value.detach()

    all_diffs = torch.cat(diffs)
    all_diffs_sorted = all_diffs.sort().values.detach()
    plot_x = np.arange(all_diffs_sorted.numel()) / all_diffs_sorted.numel()
    plot_y = all_diffs_sorted

    max_err = get_theoretical_err(cfg.model.model_name, cfg.data.dim)
    # ts = np.linspace(-max_err, +max_err, 100)
    ts = np.linspace(0, +max_err, 100)
    order_statistic_weights = order_statistics.build_order_statistic_weights(cfg.data.dim)
    # a = order_statistic_weights @ betas[1:]
    a = torch.eye(cfg.data.dim)[-1, :] - (betas[1:-1] @ order_statistic_weights).flip(0)
    # probs = np.array([analytical_cdf(a, _ - betas[0].item()) for _ in ts])
    # analytical_cdf(a, .003)

    if "pgf" == cfg.plots.fig_format:
        font_family = "serif"
        utils.plotting.initialise_pgf_plots("pdflatex",
                                            font_family)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(2.15, 2.15))
    axs[0, 0].plot(plot_x, plot_y)
    print(f"median = {plot_y.median()}")

    # axs[0, 0].plot(plot_x, all_diffs_sorted)
    axs[0, 0].grid()
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].set_ylim(-max_err, max_err)
    # axs[0, 0].set_title(r_str)
    axs[0, 0].axhline(max_err)

    # fmtr = matplotlib.ticker.StrMethodFormatter('{x:+.3f}')
    fmtr = matplotlib.ticker.FuncFormatter(lambda _, __: f'{_:.0f}' if _ == 0 else f'{_:+.3f}')
    axs[0, 0].yaxis.set_major_formatter(fmtr)

    fig.tight_layout()
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    full_filepath = paths["plots"]
    ident = f"empirical_error_{cfg.model.model_name}_{cfg.data.dataset_name}"
    fullfilename = utils.plotting.smart_save_fig(fig, ident, cfg.plots.fig_format, full_filepath)

    logger.info(f"saved {fullfilename}")

    desc = f"""
    I generate {cfg.data.num_rows} {cfg.data.dataset_name} variates, and
    plot the CDF of the empirical error. 
    The axis labels are scaled to between 0 and the L_inf error.
    """
    logger.info(desc)
    if False:
        d = cfg.data.dim
        np.array([.5, -d / (d - 2), d * (d - 1) / (d - 2)]) / (d + 1)


def analytical_cdf(a: np.ndarray, t: float):
    # notation from https://pubsonline.informs.org/doi/epdf/10.1287/ijoc.14.2.124.121
    a = np.atleast_2d(a)
    d = np.cumsum(a[::-1])[::-1]
    d0 = np.concatenate((d, np.zeros((1,))))
    d0_sorted = np.sort(d0)
    aa = (d0_sorted[1:] - d0_sorted[:-1])[::-1]
    aa[aa < 1e-7] = 0

    tt = t - d0_sorted[0]
    prob = analytical_cdf_positive_coefs(aa, tt)
    return prob


def analytical_cdf_nonzero_coefs(d: np.ndarray, x: float, kset: List[int]) -> float:
    assert (d >= 0).all()

    c = np.empty((d.size,))
    c = np.cumsum(d[::-1])[::-1]
    n = d.size
    aw = np.argwhere(x <= c)
    r = aw.max()
    prod_terms = np.full((n,), np.nan)

    for j in range(n):
        # j = 0
        inds = np.arange(n) != j
        to_prod = (c[j] - c)[inds]
        prod_terms[j] = np.prod(to_prod)

    sum_terms = (c - x) ** n / c / prod_terms
    prob = 1 - sum_terms[:r + 1].sum()
    # if False:
    #     v = 1 / c / prod_terms
    #     # sum_terms = (c - x) ** n / c / prod_terms
    #     prob2 = 1 - ((c[:r + 1] - x) ** n * v[:r + 1]).sum()

    prob = 0.0
    return prob


def analytical_cdf_positive_coefs(d: np.ndarray, x: float) -> float:
    # Weisberg (https://www.jstor.org/stable/pdf/2239815.pdf) notation for S = n:
    # Pr[sum_{s=1}^S d_s * U_{(s)} <= x] = 1 - sum_{j=1}^r ()
    # where r = largest integer such that c_r >= x
    kset = np.argwhere(d != 0).flatten().tolist()
    prob = analytical_cdf_nonzero_coefs(d, x, kset)
    return prob


def analytical_cdf_computation1():
    d = 4
    n_samples = 20000
    # n_samples = 200
    a = np.random.rand(d, )
    u = np.sort(np.random.rand(n_samples, d), 1)

    sample = (u @ a).flatten()
    do_check = True
    if do_check:
        t = +.51
        empirical_prob = (sample <= t).mean()
        analytical_prob = analytical_cdf_positive_coefs(a, t)
        assert math.fabs(empirical_prob - analytical_prob) < 1e-2

    max_val = a.sum()
    ts = np.linspace(0, max_val, 100)

    probs = np.array([analytical_cdf_positive_coefs(a, _) for _ in ts])

    fig, axs = plt.subplots(1, 1, squeeze=False)
    plt_x = np.arange(n_samples) / n_samples
    plt_y = np.sort(sample)
    axs[0, 0].plot(plt_x, plt_y)
    axs[0, 0].set_ylim(0, max_val)
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].grid()

    axs[0, 0].plot(probs, ts)


def analytical_cdf_computation2():
    # https://pubsonline.informs.org/doi/epdf/10.1287/ijoc.14.2.124.121
    d = 4
    n_samples = 20000
    # a = np.random.rand(d, ) - .5
    a = np.array([-0.044, -0.489,  0.45, -0.109])
    u = np.sort(np.random.rand(n_samples, d), 1)
    sample = (u @ a).flatten()
    fig, axs = plt.subplots(1, 1, squeeze=False)
    plt_x = np.arange(n_samples) / n_samples
    plt_y = np.sort(sample)
    axs[0, 0].plot(plt_x, plt_y)

    do_check = True
    if do_check:
        t = +.11
        empirical_prob = (sample <= t).mean()
        analytical_prob = analytical_cdf(a, t)
        # print(empirical_prob, analytical_prob)
        assert math.fabs(empirical_prob - analytical_prob) < 1e-2


def analytical_cdf_computation3():
    # https://pubsonline.informs.org/doi/epdf/10.1287/ijoc.14.2.124.121
    d = 4
    n_samples = 20000
    a = np.random.rand(d,)
    u = np.sort(np.random.rand(n_samples, d), 1)

    sample = (u @ a).flatten()

    max_val = a.sum()
    ts = np.linspace(0, max_val, 100)

    probs = np.array([analytical_cdf_positive_coefs(a, _) for _ in ts])

    fig, axs = plt.subplots(1, 1, squeeze=False)

    empirical_avg = sample.mean()

    plt_x = np.arange(n_samples) / n_samples
    plt_y = np.sort(sample)
    axs[0, 0].plot(plt_x, plt_y)
    axs[0, 0].set_ylim(0, max_val)
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].grid()

    axs[0, 0].plot(probs, ts)


def analytical_cdf_computation4():
    # https://pubsonline.informs.org/doi/epdf/10.1287/ijoc.14.2.124.121
    d = 4
    n_samples = 20000
    a = np.random.rand(d,)
    a[0] = 0

    u = np.sort(np.random.rand(n_samples, d), 1)

    sample = (u @ a).flatten()

    max_val = a.sum()
    ts = np.linspace(0, max_val, 100)

    probs = np.array([analytical_cdf_positive_coefs(a, _) for _ in ts])

    fig, axs = plt.subplots(1, 1, squeeze=False)

    empirical_avg = sample.mean()

    plt_x = np.arange(n_samples) / n_samples
    plt_y = np.sort(sample)
    axs[0, 0].plot(plt_x, plt_y)
    axs[0, 0].set_ylim(0, max_val)
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].grid()

    axs[0, 0].plot(probs, ts)


if __name__ == '__main__':
    # first_date = dt.datetime.utcnow().date() + dt.timedelta(weeks=8)
    main()
