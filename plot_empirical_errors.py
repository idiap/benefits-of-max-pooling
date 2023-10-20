
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import sys
import shutil
import math
import logging
import pickle
import os
import warnings
from typing import Any, Dict, Iterable, List, Tuple

import torch
from cycler import cycler

import matplotlib.figure
import matplotlib
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import seaborn
import numpy as np

import utils.logging
import utils.path_config
import utils.plotting
import utils.config

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

subfigure_sizes = {
    "large": 5,
    "medium": 1.9,
    "small": 1.5
}


def _plot_kernel(mults: List[float],
                 means1: torch.Tensor,
                 means2: torch.Tensor,
                 stds1: torch.Tensor) -> FigAx:
    num_mults = len(mults)
    kwargs = {
        "color": "green",
        "alpha": .25
    }
    sz = subfigure_sizes["medium"]

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)

    axs[0, 0].plot(mults, means1, linestyle='--', marker='o')
    axs[0, 0].plot(mults, means2, linestyle='-', color="k")
    axs[0, 0].fill_between(mults,
                           means1 - 1.96 * stds1,
                           means1 + 1.96 * stds1,
                           **kwargs)
    axs[0, 0].grid(axis="y")
    axs[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    axs[0, 0].ticklabel_format(useMathText=True, axis='x')
    axs[0, 0].xaxis.set_ticks(torch.arange(1, num_mults + 1, 2))

    # https://stackoverflow.com/questions/11577665/change-x-axes-scale-in-matplotlib
    axs[0, 0].yaxis.major.formatter._useMathText = True
    axs[0, 0].xaxis.major.formatter._useMathText = True
    fig.tight_layout()
    return fig, axs


def load_all_seeds(full_filedir: str) -> list:
    filenames = sorted(os.listdir(full_filedir))
    all_seeds = [None] * len(filenames)
    for idx, filename in enumerate(filenames):
        assert f"{idx}.pkl" == filename
        pickle_fullfilename = os.path.join(full_filedir, filename)
        with open(pickle_fullfilename, 'rb') as handle:
            all_seeds[idx] = pickle.load(handle)
    return all_seeds


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_stats(model_seed_result, test_or_train: str):
    assert test_or_train in ["test", "train"]
    mults = sorted(model_seed_result[0].keys())
    num_mults = len(mults)
    num_seed = len(model_seed_result)
    accs = np.zeros((num_seed, num_mults))
    param_counts = np.zeros((num_seed, num_mults))

    for idx, res in enumerate(model_seed_result):
        assert len(res) == num_mults
        for seed_idx, seed in enumerate(res):
            assert sorted(res.keys()) == mults
            rs = res[seed]
            val = rs["optimize_criterion_values"]["test"][rs["terminal_epoch"]]
            accs[idx, seed_idx] = val
            try:
                param_counts[idx, seed_idx] = count_parameters(rs["model"])
            except Exception as e:
                param_counts[idx, seed_idx] = rs["num_params"]
    assert (param_counts.std(0) == 0).all()

    stats = {
        "test": accs,
        "params": param_counts.mean(0),
        "mults": mults
    }
    return stats


def do_earlystopping_learning_curves_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    # model_name = "smallapprox"
    # model_name = "mediumapprox"
    model_name = "bigapprox"
    # full_filedir = os.path.join(paths["results"], f"stopping-always", model_name)
    full_filedir = os.path.join(paths["results"],
                                f"dataset_size20_000",
                                model_name)
    test_or_train = "train"

    filenames = sorted(os.listdir(full_filedir))
    model_seed_result = [None] * len(filenames)
    for idx, filename in enumerate(filenames):
        # idx = 0; filename = filenames[idx]
        assert f"{idx}.pkl" == filename
        pickle_fullfilename = os.path.join(full_filedir, filename)
        with open(pickle_fullfilename, 'rb') as handle:
            loaded = pickle.load(handle)
        for k, v in loaded.items():
            v["num_params"] = count_parameters(v["model"])
            del v["model"]
        model_seed_result[idx] = loaded

    mults = sorted(model_seed_result[0].keys())
    num_mults = len(mults)
    num_seed = len(model_seed_result)
    seed_res_list = [None] * num_seed
    for idx, res in enumerate(model_seed_result):
        assert len(res) == num_mults
        res_list = [None] * len(res)
        for seed_idx, seed in enumerate(res):
            assert sorted(res.keys()) == mults
            rs = res[seed]
            res_list[seed_idx] = rs["optimize_criterion_values"][test_or_train]
        seed_res_list[idx] = torch.stack(res_list, 1)
    xx = torch.stack(seed_res_list, 1)
    xxx = xx.mean(1)

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    lines = axs[0, 0].semilogy(xxx)

    legend_labels = [str(_) for _ in mults]
    axs[0, 0].legend(lines, legend_labels)
    return fig, axs


def do_dataset_size_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    # extra_scl = .8
    extra_scl = .775
    rotation = 45
    fontsize = 7
    markersize = 1.75

    # sizes = ["100", "500", "1_000", "5_000", "10_000"]
    sizes = ["100", "500", "1_000", "5_000", "10_000", "20_000", "50_000", "100_000"]

    test_or_train = "train"
    # test_or_train = "test"
    # model_name = "single_layer"
    model_name = "bigapprox"
    f = lambda _: os.path.join(paths["results"], f"dataset_size{_}", model_name)

    stats = [build_stats(load_all_seeds(f(_)), test_or_train) for _ in sizes]
    param_counts = stats[0]["params"].astype(int).tolist()

    avgs = [_["test"].mean(0) for _ in stats]
    mults = stats[0]["mults"]

    y = np.stack(avgs, 1)
    legend_labels = [_.replace("_", ",") for _ in sizes]

    # https://stackoverflow.com/a/24544116
    figsize = (w_o_h * fig_scl * extra_scl, fig_scl * extra_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    lines = axs[0, 0].semilogx(mults, y, "o-", markersize=markersize)

    axs[0, 0].legend(lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.0)
    axs[0, 0].set_xticks(mults)
    axs[0, 0].set_xticklabels(param_counts, rotation=rotation, fontsize=fontsize)
    axs[0, 0].grid()
    fig.tight_layout()
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    full_filepath = paths["plots"]
    ident = f"dataset_size_{model_name}_{test_or_train}"
    fullfilename = utils.plotting.smart_save_fig(fig, ident, cfg_plot.fig_format, full_filepath)
    logger.info(f"saved {fullfilename}")
    return fig, axs


def load_data_for_train_test_plot(full_filedir: str):
    filenames = sorted(os.listdir(full_filedir))
    model_seed_result = [None] * len(filenames)
    for idx, filename in enumerate(filenames):
        # idx = 0; filename = filenames[idx]
        assert f"{idx}.pkl" == filename
        pickle_fullfilename = os.path.join(full_filedir, filename)
        with open(pickle_fullfilename, 'rb') as handle:
            loaded = pickle.load(handle)
        model_seed_result[idx] = loaded

    mults = sorted(model_seed_result[0].keys())
    num_mults = len(mults)
    num_seed = len(model_seed_result)
    seed_res_test_list = [None] * num_seed
    seed_res_train_list = [None] * num_seed
    for idx, res in enumerate(model_seed_result):
        assert len(res) == num_mults
        res_test_list = [None] * len(res)
        res_train_list = [None] * len(res)
        for seed_idx, seed in enumerate(res):
            assert sorted(res.keys()) == mults
            rs = res[seed]
            res_test_list[seed_idx] = rs["optimize_criterion_values"]["test"]
            res_train_list[seed_idx] = rs["optimize_criterion_values"]["train"]
        seed_res_test_list[idx] = torch.stack(res_test_list, 1)
        seed_res_train_list[idx] = torch.stack(res_train_list, 1)
    test_x = torch.stack(seed_res_test_list, 1)
    train_x = torch.stack(seed_res_train_list, 1)
    return test_x, train_x, mults


def do_train_test_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    model_name = "bigapprox"
    f = lambda _: load_data_for_train_test_plot(os.path.join(paths["results"],
                                                             f"dataset_size{_}",
                                                             model_name))

    test_x, train_x, mults = f("10_000")
    train_curve = train_x.mean(1)
    test_curve = test_x.mean(1)

    if False:
        num_rows = 3
        num_plots = len(mults)
        num_cols = int(math.ceil(num_plots / num_rows))
        fig, axs = plt.subplots(num_rows, num_cols)

        for idx in range(num_plots):
            row = idx // num_rows
            col = idx % num_rows
            # print(row, col)
            axs[row, col].semilogy(train_curve[:, idx], "k")
            axs[row, col].semilogy(test_curve[:, idx], "b")
            axs[row, col].set_title(mults[idx])

        utils.plotting.smart_save_fig(fig,
                                      "traintest_learning_curve",
                                      cfg_plot.fig_format,
                                      full_filepath)

    tt = torch.stack((train_curve[-1, :], test_curve[-1, :]), 1)

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    lines = axs[0, 0].semilogy(mults, tt, "o-")
    axs[0, 0].grid()
    axs[0, 0].legend(lines, ["train", "test"])

    fig_full_path = utils.plotting.smart_save_fig(fig,
                                                  "traintest_terminal",
                                                  cfg_plot.fig_format,
                                                  paths["plots"])
    logger.info(f"saved {fig_full_path}")
    return fig, axs


def deepdive_plot_on_ax(ax,
                        approx_name: str,
                        color):
    f = lambda _, __: load_data_for_approx_size(
        os.path.join(paths["results"], f"baseline", _), __)

    markersize = 1.75
    rotation = 45
    fontsize = 7

    approx, mults, param_counts = f(approx_name, "test")
    mean = approx.mean(0)
    std = approx.std(0)
    mn = approx.min(0).values
    mx = approx.max(0).values

    train_approx, _, __ = f(approx_name, "train")
    train_mean = train_approx.mean(0)

    # https://stackoverflow.com/questions/27878217/how-do-i-extend-the-margin-at-the-bottom-of-a-figure-in-matplotlib

    fmt = lambda _: str(int(_)) if int(_) == _ else f"{_:.2f}".replace("0.", ".")
    xticklabels = [" " if idx < 3 else fmt(_) for idx, _ in enumerate(param_counts)]
    do_semilogy = False
    if do_semilogy:
        ax.semilogy(mults, mean, c=color, linestyle="-", marker="o", markersize=markersize)
        ax.semilogy(mults, train_mean, c=color, linestyle="dashed")
        ax.semilogy(mults, mn, c=color, linestyle=":")
        ax.semilogy(mults, mx, c=color, linestyle=":")
    else:
        ax.plot(mults, mean, c=color, linestyle="-", marker="o", markersize=markersize)
        ax.plot(mults, train_mean, c=color, linestyle="dashed")
        ax.plot(mults, mn, c=color, linestyle=":")
        ax.plot(mults, mx, c=color, linestyle=":")

    ax.set_xticks(mults)
    xl = ax.get_xlim()
    ax.set_xlim(0, xl[1])
    ax.set_xticklabels(xticklabels, rotation=rotation, fontsize=fontsize)
    # if not do_yticks:
    #     ax.set_yticks([])

    kwargs = {
        "color": color,
        "alpha": .25
    }
    ax.fill_between(mults,
                    mean - 1.96 * std,
                    mean + 1.96 * std,
                    **kwargs)


def do_deepdive_plot(cfg_plot: DictConfig):
    fig_scl = cfg_plot.fig_scl

    # extra_scl = .75
    extra_scl = .7
    this_fig_scl = extra_scl * fig_scl
    figsize = (this_fig_scl * 3, this_fig_scl)
    fig, axs = plt.subplots(1, 3, figsize=figsize,
                            squeeze=False, sharey=True, constrained_layout=True)

    palette = seaborn.color_palette(cfg_plot.palette_name, 3)

    deepdive_plot_on_ax(axs[0, 0], "smallapprox", palette[0])
    deepdive_plot_on_ax(axs[0, 1], "mediumapprox", palette[1])
    deepdive_plot_on_ax(axs[0, 2], "bigapprox", palette[2])

    # axs[0, 0].set_ylabel(("err$(R, \\mu)$"))
    axs[0, 0].set_title("$R = \\{0, 1, d - 1\\}$", fontsize=10)
    axs[0, 1].set_title("$R = \\{0, 1, d - 2, d - 1\\}$", fontsize=10)
    axs[0, 2].set_title("$R = \\{0, 1, ..., d - 2, d - 1\\}$", fontsize=10)
    # if not do_yticks:
    #     ax.set_yticks([])

    st = fig.suptitle('$x$-axis: \# of parameters, (shared) $y$-axis: \\textsc{err}$(R, \\mu)$', fontsize=10)
    # fig.subplots_adjust(wspace=0.05)
    # fig.subplots_adjust(bottom=0.2, left=.15)
    # fig.subplots_adjust(left=.15, top=.15)
    ident = "deepdive"
    fig_format = cfg_plot.fig_format

    filepath = paths["plots"]
    filename = "{}.{}".format(ident, fig_format)
    os.makedirs(filepath, exist_ok=True)
    fullfile_path = os.path.join(filepath, filename)

    plt.savefig(fullfile_path, bbox_extra_artists=[st], bbox_inches='tight')
    logger.info(f"saved {fullfile_path}")
    return fig, axs


def do_relerr_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    fig_full_path = utils.plotting.smart_save_fig(fig,
                                                  "relerr",
                                                  cfg_plot.fig_format,
                                                  paths["plots"])
    logger.info(f"saved {fig_full_path}")
    return fig, axs


def load_data_for_approx_size(full_filedir: str, train_or_test) -> torch.tensor:
    # train_or_test = "test"
    filenames = sorted(os.listdir(full_filedir))
    num_seeds = len(filenames)
    # field_name = "optimize_criterion_values"
    field_name = "evaluate_criterion_values"
    model_seed_result = [None] * num_seeds
    for idx, filename in enumerate(filenames):
        # idx = 0; filename = filenames[idx]
        assert f"{idx}.pkl" == filename
        pickle_fullfilename = os.path.join(full_filedir, filename)
        with open(pickle_fullfilename, 'rb') as handle:
            loaded = pickle.load(handle)
            g = {k: v[field_name][train_or_test][v["terminal_epoch"]]
                 for k, v in loaded.items()}
        model_seed_result[idx] = g

    mults = sorted(model_seed_result[0].keys())
    num_mults = len(mults)
    data_for_approx_size = torch.full((num_seeds, num_mults), math.nan)
    for idx in range(num_seeds):
        data_for_approx_size[idx, :] = torch.tensor([model_seed_result[idx][m] for m in mults])
    # plt.plot(mults, data_for_approx_size.T)
    # plt.plot(mults, data_for_approx_size.T)
    if False:
        gg = loaded[mults[0]]["model"]
    param_counts = [count_parameters(loaded[_]["model"]) for _ in mults]
    return data_for_approx_size, mults, param_counts


def do_approx_size_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl
    extra_scl = .75
    fontsize = 7
    markersize = 1.75

    f = lambda _, __: load_data_for_approx_size(os.path.join(paths["results"],
                                                             f"baseline",
                                                             _), __)
    bigapprox, mults, _ = f("bigapprox", "test")
    mediumapprox, m1, _ = f("mediumapprox", "test")
    smallapprox, m2, _ = f("smallapprox", "test")

    assert mults == m1 and mults == m2
    y = torch.stack((smallapprox, mediumapprox, bigapprox)).mean(1).T

    figsize = (w_o_h * fig_scl * extra_scl, fig_scl * extra_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False, constrained_layout=True)

    lines = axs[0, 0].plot(mults, y, "o-", markersize=markersize)
    legend_entries = ["$R = \\{0, 1, d - 1\\}$",
                      "$R =  \\{0, 1, d-2, d - 1\\}$",
                      "$R =  \\{0, 1, ..., d-2, d - 1\\}$"]
    axs[0, 0].legend(lines, legend_entries)
    desc = """
    Test error averaged over 10 random seeds for small ($R = \\{0, 1, d - 1\\}$), 
    medium ($R =  \\{0, 1, d-2, d - 1\\}$), and large ($R =  \\{0, 1, ..., d-2, d - 1\\}$), 
    models. $ d = 8$.
    """

    fmt = lambda _: str(int(_)) if int(_) == _ else f"{_:.2f}".replace("0.", ".")
    xticklabels = [" " if idx < 3 else fmt(_) for idx, _ in enumerate(mults)]

    axs[0, 0].set_xticks(mults)
    axs[0, 0].set_xticklabels(xticklabels, fontsize=fontsize)
    axs[0, 0].grid()
    axs[0, 0].set_xlabel("$\\mu$")
    axs[0, 0].set_ylabel("\\textsc{err}$(R, \\mu)$")

    # plt.tight_layout()
    plots_filepath = paths["plots"]
    fullfile_path = utils.plotting.smart_save_fig(fig,
                                                  "approxsize",
                                                  cfg_plot.fig_format,
                                                  plots_filepath)
    logger.info(f"saved {fullfile_path}")
    return fig, axs


def do_single_layer_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)

    plots_filepath = paths["plots"]
    fullfile_path = utils.plotting.smart_save_fig(fig,
                                                  "singlelayer",
                                                  cfg_plot.fig_format,
                                                  plots_filepath)
    logger.info(f"saving {fullfile_path}")
    return fig, axs


def do_average_plot(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl

    result = "relative"
    # result = "mean"
    fullfilename = os.path.join(paths["results"], result, "bigapprox")
    data = load_data_for_approx_size(fullfilename, "train")

    figsize = (w_o_h * fig_scl, fig_scl)
    fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)

    mults = data[1]
    errs = data[0]
    axs[0, 0].plot(mults, errs.T)

    plots_filepath = paths["plots"]
    fullfile_path = utils.plotting.smart_save_fig(fig,
                                                  "singlelayer",
                                                  cfg_plot.fig_format,
                                                  plots_filepath)
    logger.info(f"saving {fullfile_path}")
    return fig, axs


def do_experiment_plots(cfg_plot: DictConfig):
    w_o_h = cfg_plot.w_o_h
    fig_scl = cfg_plot.fig_scl
    extra_scl = .75
    fontsize = 7
    markersize = 1.75
    #
    # experiments = [
    #     "initialization-xavier",
    #     "initialization-kaiming",
    #     # "criterion-l2",
    #     "optimizer-adamw",
    #     # "optimizer-sgd",
    #     "data-sobol",
    #     "data-dirichlet"
    #    ]

    experiments = [
        "criterion-l2",
       ]

    for experiment in experiments:
        f = lambda _: load_data_for_approx_size(os.path.join(paths["results"],
                                                                 experiment,
                                                                 _), "test")
        bigapprox, mults, _ = f("bigapprox")
        mediumapprox, m1, _ = f("mediumapprox")
        smallapprox, m2, _ = f("smallapprox")

        assert mults == m1 and mults == m2
        y = torch.stack((smallapprox, mediumapprox, bigapprox)).mean(1).T

        figsize = (w_o_h * fig_scl * extra_scl, fig_scl * extra_scl)
        fig, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False,
                                constrained_layout=True)

        lines = axs[0, 0].plot(mults, y, "o-", markersize=markersize)
        legend_entries = ["$R = \\{0, 1, d - 1\\}$",
                          "$R =  \\{0, 1, d-2, d - 1\\}$",
                          "$R =  \\{0, 1, ..., d-2, d - 1\\}$"]
        axs[0, 0].legend(lines, legend_entries)

        fmt = lambda _: str(int(_)) if int(_) == _ else f"{_:.2f}".replace("0.",
                                                                           ".")
        xticklabels = [" " if idx < 3 else fmt(_) for idx, _ in
                       enumerate(mults)]

        axs[0, 0].set_xticks(mults)
        axs[0, 0].set_xticklabels(xticklabels, fontsize=fontsize)
        axs[0, 0].grid()
        axs[0, 0].set_xlabel("$\\mu$")
        axs[0, 0].set_ylabel("\\textsc{err}$(R, \\mu)$")

        plots_filepath = paths["plots"]
        fullfile_path = utils.plotting.smart_save_fig(fig,
                                                      f"experiments_{experiment}",
                                                      cfg_plot.fig_format,
                                                      plots_filepath)
        logger.info(f"saved {fullfile_path}")


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
    do_experiment_plots(cfg_plot)
    fig, axs = do_average_plot(cfg_plot)
    fig, axs = do_deepdive_plot(cfg_plot)
    fig, axs = do_approx_size_plot(cfg_plot)
    fig, axs = do_earlystopping_learning_curves_plot(cfg_plot)
    fig, axs = do_dataset_size_plot(cfg_plot)
    fig, axs = do_train_test_plot(cfg_plot)
    fig, axs = do_relerr_plot(cfg_plot)


if __name__ == "__main__":
    main()
