
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Any, Dict, List

import torch

import tools

import networks
import depth_analysis


def _row_vals(d: int, k: int) -> torch.tensor:
    row = torch.tensor([math.comb(d - 1 - _, k - 1) for _ in range(d)])
    assert math.fabs(row.sum() - math.comb(d, k)) < 1e-14  # Pascal's formula
    return row


def build_order_statistic_weights(d: int) -> torch.tensor:
    # mapping from d order statistics to d - 1 subpool max averages
    # rows are the subpool max averages
    # columns are the order statistics (largest on the right)
    # Called B(d) in the paper
    loadings_matrix = torch.zeros((d - 1, d))
    for k in range(1, d):
        # k = d - 2
        loadings_matrix[k - 1, :] = _row_vals(d, k) / math.comb(d, k)
    return loadings_matrix


if __name__ == "__main__":
    d = 5
    k = d - 2
    row = torch.tensor([math.comb(d - 1 - _, k - 1) for _ in range(d)]) / math.comb(d, k)
    print(row)

    d = 8
    print(f"dim = {d}")
    index_set = list(range(0, d - 1))
    index_power_set = tools.powerset(tuple(index_set))

    for idx, ks in enumerate(index_power_set):
        # idx = len(index_power_set) * 3 // 4; rows = index_power_set[idx]
        if () == ks:
            continue

        soln = depth_analysis.get_dk_situation(d, ks)
        criterion = soln['criterion']
        argmin = soln['argmin']

        print(idx, ks, f"{criterion:.3f}", argmin)

        betas = argmin
        z = torch.zeros(1)
        soln = depth_analysis.get_dk_situation(d, ks)
        argmin = soln["argmin"]
        betas = torch.cat((torch.tensor(argmin), z)).to(torch.float32)

        model = networks.build_wholly_general_model(betas, d)
