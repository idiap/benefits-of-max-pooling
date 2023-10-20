
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import random
import math
import itertools
from typing import Iterable, List, Tuple

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt

import utils.path_config

FigAx = Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

seed = 1338
random.seed(seed)


def plot_colorlist_on_col(colorlist: List[float],
                          col: int,
                          row0: int,
                          axs):
    numcolors = len(colorlist)
    colors = [(1 - val,) * 3 for val in colorlist]
    for row in range(row0):
        axs[row, col].axis("off")
    for row in range(row0 + numcolors, axs.shape[0]):
        axs[row, col].axis("off")
    for idx, row in enumerate(range(row0, row0 + numcolors)):
        # axs[idx].axis('off')
        # axs[row, col].axes.get_xaxis().set_visible(False)
        # axs[row, col].get_yaxis().set_visible(False)
        color = colors[idx]
        axs[row, col].set_facecolor(color)
        # axs[idx].set_title("{:.4}".format(colorlist[idx]))


def _maxes_for_order(vals: List[float],
                     k: int) -> List[float]:
    dim = len(vals)
    xs = tuple([_ for _ in range(dim)])
    inds = sorted(list(itertools.combinations(xs, k)))
    indmaxes = [None] * len(inds)
    for idx in range(len(inds)):
        indmaxes[idx] = max([vals[_] for _ in inds[idx]])
    return indmaxes, inds


def do_calc(vals: List[float]) -> List[list]:
    dim = len(vals)
    calc = [None] * dim
    for idx in range(1, dim):
        calc[idx] = _maxes_for_order(vals, idx)
    return calc


def plot_calc(calc: List[list],
              plot_height: float,
              plot_width: float) -> FigAx:
    figsize = (plot_width, plot_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dim = len(calc[1][0])
    ncol = dim - 1
    nrow = max([math.comb(dim, _) for _ in range(dim)])

    frac_for_bottom = .2
    buffer_rel_sq_size = .1
    horz_offset = frac_for_bottom / 2

    # need to have nrow boxes of size sq_size plus margins in plottable area
    # (sq_size * (1 + 2 * buffer_rel_sq_size)) * nrow <= 1 - frac_for_bottom
    # Same condition for columns:
    # (sq_size * (1 + 2 * buffer_rel_sq_size)) * ncol <= 1 - frac_for_bottom
    # iff sq_size <= ( ) / (1 + 2 * buffer_rel_sq_size)
    sq_size = (1 - frac_for_bottom - buffer_rel_sq_size) / ((1 + 2 * buffer_rel_sq_size) * max(nrow, ncol))
    buffer = buffer_rel_sq_size * sq_size

    x_intercept = (horz_offset + buffer)
    x_slope = (2 * buffer + sq_size)
    y_slope = (1 - frac_for_bottom) / nrow

    for c in range(ncol):
        # c = 0
        # c = 1
        c1_calc = calc[c + 1][0]
        num_terms = len(c1_calc)

        x = x_intercept + x_slope * c
        val = sum(c1_calc) / num_terms
        y = frac_for_bottom / 2 - sq_size / 2
        color = (1 - val,) * 3
        rect = matplotlib.patches.Rectangle((x, y),
                                            sq_size,
                                            sq_size,
                                            edgecolor="black",
                                            facecolor=color)
        ax.add_patch(rect)
        blank_rows_above = (nrow - num_terms) / 2
        y_intercept = frac_for_bottom + y_slope * blank_rows_above

        for r in range(num_terms):
            # for r in [1]:
            y = y_intercept + r * y_slope
            val = c1_calc[r]
            color = (1 - val,) * 3
            # print(r, c, x, y)
            assert x + sq_size <= 1
            assert y + sq_size <= 1
            rect = matplotlib.patches.Rectangle((x, y),
                                                sq_size,
                                                sq_size,
                                                edgecolor="black",
                                                facecolor=color)
            ax.add_patch(rect)
    ax.axis("off")
    return fig, ax


def _add_centered_box_row(yval: float,
                          xintercept: float,
                          xslope: float,
                          colors_grayscale: Iterable[float],
                          width, height,
                          ax: matplotlib.axes.Axes):
    for idx, color_grayscale in enumerate(colors_grayscale):
        color = (1 - color_grayscale,) * 3
        tl = (xintercept + idx * xslope, yval)

        rect = matplotlib.patches.Rectangle(tl,
                                            width,
                                            height,
                                            edgecolor="black",
                                            facecolor=color)
        ax.add_patch(rect)


def plot_calc_horiz(calc: List[list],
                    plot_height: float,
                    plot_width: float) -> FigAx:
    figsize = (plot_width, plot_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dim = len(calc[1][0])
    nbox_width = max([math.comb(dim, _) for _ in range(dim)])

    n_subplot = dim - 1
    right_frac = .15

    width_buffer_rel = .1
    box_size = (1 - width_buffer_rel) / ((1 + 2 * width_buffer_rel) * nbox_width)
    width_buffer = width_buffer_rel * box_size

    subplot_height = 1 / n_subplot

    xslope = (2 * width_buffer + box_size)
    for subplot_idx in range(n_subplot):
        # subplot_idx = 0
        subplot_bot_y = 1 - (subplot_idx + 1) * subplot_height

        calcn = calc[subplot_idx + 1]
        colorsn = calcn[0]
        ncoln = len(colorsn)

        xintercept = (1 - right_frac) * ((nbox_width - ncoln) / nbox_width) / 2
        yval1 = subplot_bot_y + box_size / 2
        height = 2.25 * box_size
        width = box_size

        _add_centered_box_row(yval1, xintercept, xslope, colorsn, width, height, ax)
        avgcol = sum(colorsn) / ncoln

        tl = (1 - box_size, yval1)

        rect = matplotlib.patches.Rectangle(tl,
                                            width,
                                            height,
                                            edgecolor="black",
                                            facecolor=(1 - avgcol,) * 3)
        ax.add_patch(rect)
        # ax.hlines(y=yval1, xmin=0, xmax=1, color="r")
        # ax.hlines(y=yval1 + height, xmin=0, xmax=1, color="b")
        ax.text(1 + box_size / 4,
                yval1 + height / 2,
                f"$r = {subplot_idx + 1}$")
    ax.axis("off")
    return fig, ax


if __name__ == "__main__":
    plot_width = 6.0
    plot_height = plot_width / 2.61

    # dim = 6
    # dim = 4
    dim = 5

    if 4 == dim:
        orig_order = [1, 3, 0, 2]
    elif 5 == dim:
        orig_order = [1, 3, 0, 4, 2]
    elif 6 == dim:
        orig_order = [2, 1, 5, 0, 4, 3]
    else:
        orig_order = list(range(dim))
    grayscales = [1 / dim + (_ / dim + random.random() / (2 * dim)) * (1 - 1 / dim) for _ in orig_order]
    calc = do_calc(grayscales)
    # fig, ax = plot_calc(calc, plot_height, plot_width)

    paths = utils.path_config.get_paths()
    # filepath = os.path.join(paths["plots"], "statistical_sorting.png")
    filepath = paths["plots"]
    ident = "statistical_sorting"

    fig_format = "pgf"
    # fig_format = "pdf"
    fig, ax = plot_calc_horiz(calc, plot_height, plot_width)
    fig.tight_layout()
    fig_path = utils.plotting.smart_save_fig(fig, ident, fig_format, filepath)
    print(f"Check '{fig_path}'")
