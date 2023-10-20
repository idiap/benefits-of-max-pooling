
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
import itertools
import fractions
import functools
from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np

import cvxpy
import gurobipy

import networks
import tools
import order_statistics


np.random.seed(seed=112)
torch.manual_seed(112)
random.seed(112)

ArrayLike = Union[torch.Tensor, np.ndarray]

np.set_printoptions(linewidth=1000, precision=3)
torch.set_printoptions(linewidth=1000)

# https://www.gurobi.com/downloads/end-user-license-agreement-academic/

torch_inf = torch.tensor(float("Inf"))
torch_nan = torch.tensor(float("nan"))
torch_zero = torch.tensor(float(0.0))
torch_one = torch.tensor(float(1.0))


@functools.lru_cache(16)
def _build_full_system(d: int) -> np.ndarray:
    # this is called B(d) in the paper
    order_statistic_weights = order_statistics.build_order_statistic_weights(d)
    assert order_statistic_weights.shape == (d - 1, d)
    vd = np.tril(np.ones((d + 1, d)), -1).T

    assert (vd.T == tools.generate_increasing_unit_cube_v(d)[:, 1:]).all()
    # this is called K(d) in the paper
    coef1 = order_statistic_weights @ vd

    # how much of each subpool weight term you will pick up at each vertex
    # add a column of ones for the intercept
    coef_matrix = np.concatenate((np.ones((1, d + 1)), coef1), 0)
    return coef_matrix


def _get_dk_situation_gurobi(d: int, ks: List[int]) -> dict:
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    omitted_betas = ~np.in1d(cols, ks)

    # vertices in lambda space are canonical bases
    check_points = np.concatenate((np.zeros((1, d)),
                                   np.eye(d)), 0)  # in lambda space
    target_function_response = check_points.max(1)
    num_check_points = check_points.shape[0]

    m = gurobipy.Model("coefs")
    g = m.addMVar(shape=1)
    # d betas: 0 (constant) through d - 1
    betas = m.addMVar(shape=d, lb=-float('inf'), ub=float('inf'))

    for idx in range(num_check_points):
        resp = target_function_response[idx]
        fitted = betas @ full_coef_matrix[:, idx]
        # is K lambda[idx]

        m.addConstr(g >= +1 * (resp - fitted))
        m.addConstr(g >= -1 * (resp - fitted))

    for idx in range(d):
        if omitted_betas[idx]:
            m.addConstr(0 == betas[idx])

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = betas.x

    if 0 in ks:
        assert math.fabs(argmin[0] - criterion) < 1e-8
    dk_situation = {
        "criterion": criterion,
        "argmin": torch.tensor(argmin, dtype=torch.float32)
    }
    return dk_situation


def _get_dk_situation_hardcoded(d: int, ks: List[int]) -> dict:
    # used for machines where gurobi might not be possible
    if 8 == d and [0, 1, 2, 3, 4, 5, 6, 7] == ks:
        criterion = 0.0039062479786478865
        argmin = np.array([+3.906e-03, 6.250e-02, -4.375e-01, 1.750e+00,
                           -4.375e+00, 7.000e+00, -7.000e+00, 4.000e+00])
    elif 8 == d and [0, 1, 6, 7] == ks:
        criterion = 0.01562500030559015
        argmin = np.array([0.016, 0.05, 0.0, 0.0, 0.0, 0.0, -1.05, 2.0])
    elif 8 == d and [0, 1, 7] == ks:
        criterion = 0.05555555555555558
        argmin = np.array([0.056, -0.148, 0., 0., 0., 0., 0., 1.037])
    elif 9 == d and [0, 1, 2, 3, 4, 5, 6, 7, 8] == ks:
        criterion = 0.001953121955016228
        argmin = np.array([1.953e-03, -3.516e-02, 2.812e-01, -1.312e+00,
                           3.937e+00, -7.875e+00, 1.050e+01, -9.000e+00,
                           4.500e+00])
    elif 9 == d and [0, 1, 8] == ks:
        criterion = 0.04999999999999995
        argmin = np.array([+0.05, -0.129, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.029])
    elif 9 == d and [0, 1, 7, 8] == ks:
        criterion = 0.012345678916956409
        argmin = np.array([0.012, 0.037, 0., 0., 0., 0., 0., -1.037, 2.])
        """
        8 [0, 1, 6, 7] {'criterion':, 'argmin': }
        9 [0, 1, 7, 8] {'criterion': 0.012345678916956409, 'argmin': array([ 0.012,  0.037,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.037,  2.   ])}
        """
    elif 11 == d and [0, 1, 9, 10] == ks:
        criterion = 0.008264463140533368
        argmin = np.array([0.0083, 0.0227, 0.0000, 0.0000, 0.0000,
                           0.0000, 0.0000, 0.0000, 0.0000, -1.0227, 2.0000])
    elif 11 == d and ks == [0, 1, 10]:
        criterion = 0.041666666666666734
        argmin = np.array([0.0417, -0.1019, 0.0000, 0.0000, 0.0000, 0.0000,
                           0.0000, 0.0000, 0.0000, 0.0000, 1.0185])
    elif 11 == d and [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] == ks:
        criterion = 0.0004882823386501006
        argmin = np.array([4.8828e-04, -1.0742e-02, 1.0742e-01, -6.4453e-01,
                           2.5781e+00, -7.2188e+00, 1.4438e+01, -2.0625e+01, 2.0625e+01,
                          -1.3750e+01, 5.5000e+00])
    elif 10 == d and [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] == ks:
        criterion = 0.0009765612325424011
        argmin = np.array([9.7656e-04, 1.9531e-02, -1.7578e-01, 9.3750e-01, -3.2812e+00,
                           7.8750e+00, -1.3125e+01, 1.5000e+01, -1.1250e+01, 5.0000e+00])
    elif 10 == d and [0, 1, 8, 9] == ks:
        criterion = 0.010000000668423471
        argmin = np.array([0.0100, 0.0286, 0.0000, 0.0000, 0.0000,
                           0.0000, 0.0000, 0.0000, -1.0286, 2.0000])
    elif 10 == d and [0, 1, 9] == ks:
        criterion = 0.045454547186343675
        argmin = np.array([0.0455, -0.1136, 0.0000, 0.0000, 0.0000,
                           0.0000, 0.0000, 0.0000, 0.0000, 1.0227])
    else:
        raise ValueError(f"d, ks = {d}, {ks} not hardcoded yet")
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    try:
        dk_situation = _get_dk_situation_gurobi(d, ks)
    except:
        dk_situation = _get_dk_situation_hardcoded(d, ks)
    return dk_situation


def get_dk_situation2(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    omitted_betas = ~included_betas

    num_omitted_betas = omitted_betas.sum()
    num_included_betas = included_betas.sum()
    num_betas = int(num_included_betas)

    # vertices in lambda space are canonical bases
    check_points = np.concatenate((np.zeros((1, d)),
                                   np.eye(d)), 0)  # in lambda space
    target_function_response = check_points.max(1)
    num_check_points = check_points.shape[0]

    m = gurobipy.Model("coefs")
    g = m.addMVar(shape=1)
    # d betas: 0 (constant) through d - 1
    betas = m.addMVar(shape=num_betas, lb=-float('inf'), ub=float('inf'))

    for idx in range(num_check_points):
        # idx = 0
        resp = target_function_response[idx]
        fitted = betas @ full_coef_matrix[included_betas, :][:, idx]

        m.addConstr(g >= +1 * (resp - fitted))
        m.addConstr(g >= -1 * (resp - fitted))

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = np.zeros((d,))
    argmin[included_betas] = betas.x

    if 0 in ks:
        assert math.fabs(argmin[0] - criterion) < 1e-8
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation3(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]

    num_included_betas = included_betas.sum()
    num_betas = int(num_included_betas)

    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))
    num_points = target_function_response.shape[0]

    m = gurobipy.Model("coefs")
    g = m.addMVar(shape=1)
    betas = m.addMVar(shape=num_betas, lb=-float('inf'), ub=float('inf'))

    for idx in range(num_points):
        # idx = 0
        resp = target_function_response[idx]
        fitted = betas @ coef_matrix[:, idx]

        # val =
        m.addConstr(g >= +1 * (resp - fitted))
        m.addConstr(g >= -1 * (resp - fitted))

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = np.zeros((d,))
    argmin[included_betas] = betas.x

    if 0 in ks:
        assert math.fabs(argmin[0] - criterion) < 1e-8
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation4(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]

    num_included_betas = included_betas.sum()
    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    b = -1 * target_function_response
    a = coef_matrix

    nr = a.shape[0]
    nc = a.shape[1]
    m = gurobipy.Model("coefs")
    g = m.addMVar(shape=1)
    x = m.addMVar(shape=nr, lb=-float('inf'), ub=float('inf'))

    for idx in range(nc):
        # idx = 0
        val = x @ a[:, idx] + b[idx]
        m.addConstr(g >= +1 * val)
        m.addConstr(g >= -1 * val)

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    criterion = tools.smart_optimize(m, is_verbose = False)
    argmin = np.zeros((d,))
    argmin[included_betas] = x.x

    if 0 in ks:
        assert math.fabs(argmin[0] - criterion) < 1e-8
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation5(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]
    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    a = coef_matrix
    b = -1 * target_function_response.reshape(-1, 1)

    argmin0, criterion = l_inf_optimization(a, b)

    argmin = np.zeros((d,))
    argmin[included_betas] = argmin0
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation6(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]

    num_included_betas = included_betas.sum()
    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    a = coef_matrix.T
    b = -1 * target_function_response.reshape(-1, 1)

    x = cvxpy.Variable((num_included_betas, 1))
    objective = cvxpy.Minimize(cvxpy.norm(a @ x + b, math.inf))
    constraints = []
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(verbose=False)

    criterion = prob.value
    argmin = np.zeros((d,))
    argmin[included_betas] = x.value.flatten()
    dk_situation = {
        "criterion": criterion,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation7(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]

    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    a = coef_matrix.T
    b = -1 * target_function_response.reshape(-1, 1)
    nc, nr = a.shape

    nu = cvxpy.Variable((nc, 1))
    objective = cvxpy.Maximize(nu.T @ b - cvxpy.norm(nu, 1) ** 2 / 2)
    constraints = [a.T @ nu == 0]
    prob_dual = cvxpy.Problem(objective, constraints)

    prob_dual.solve(verbose=False)
    value = (prob_dual.value * 2) ** .5

    rows = (np.abs(nu.value) > 1e-15).flatten()
    row_signs = np.sign(nu.value)[rows]
    rhs = value * row_signs - b[rows]
    xvalue = np.linalg.pinv(a[rows, :]) @ rhs

    argmin = np.zeros((d,))
    argmin[included_betas] = xvalue.flatten()
    dk_situation = {
        "criterion": value,
        "argmin": argmin
    }
    return dk_situation


def get_dk_situation8(d: int, ks: List[int]) -> dict:
    assert all(d >= _ for _ in ks)
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)
    coef_matrix = full_coef_matrix[included_betas, :]

    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    a = coef_matrix.T
    b = -1 * target_function_response.reshape(-1, 1)
    nc, nr = a.shape

    nu = cvxpy.Variable((nc, 1))
    objective = cvxpy.Maximize(nu.T @ b - cvxpy.norm(nu, 1) ** 2 / 2)
    constraints = [a.T @ nu == 0]
    prob_dual = cvxpy.Problem(objective, constraints)
    prob_dual.solve(verbose=False)
    value = (prob_dual.value * 2) ** .5

    rows = (np.abs(nu.value) > 1e-15).flatten()
    row_signs = np.sign(nu.value)[rows]
    rhs = value * row_signs - b[rows]
    xvalue = np.linalg.pinv(a[rows, :]) @ rhs

    argmin = np.zeros((d,))
    argmin[included_betas] = xvalue.flatten()
    dk_situation = {
        "criterion": value,
        "argmin": argmin
    }
    return dk_situation


def l_inf_optimization(a: np.ndarray,
                       b: np.ndarray) -> Tuple[np.array, float]:
    nr = a.shape[0]
    nc = a.shape[1]
    m = gurobipy.Model("coefs")
    g = m.addMVar(shape=1)
    x = m.addMVar(shape=nr, lb=-float('inf'), ub=float('inf'))

    for idx in range(nc):
        # idx = 0
        val = x @ a[:, idx] + b[idx]
        m.addConstr(g >= +1 * val)
        m.addConstr(g >= -1 * val)

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = x.x
    value = criterion
    return argmin, value


def _max_absval(signed_crit: gurobipy.MLinExpr,
                optim_val: gurobipy.MVar,
                m: gurobipy.Model) -> Tuple[float, np.ndarray]:
    is_verbose = False
    m.setObjective(-1 * signed_crit, gurobipy.GRB.MINIMIZE)
    criterion1 = -1 * tools.smart_optimize(m, is_verbose)
    argopt1 = optim_val.x

    m.setObjective(+1 * signed_crit, gurobipy.GRB.MAXIMIZE)
    criterion2 = +1 * tools.smart_optimize(m, is_verbose)
    argopt2 = optim_val.x

    criterion = max(criterion1, criterion2)
    argopt = argopt1 if criterion == criterion1 else argopt2
    return criterion, argopt


def verify_weight_optimality_v_form1(w: np.ndarray) -> Dict[str, Any]:
    # Old form, without lambda1 substituted out
    d = w.shape[0]
    order_statistic_weights = order_statistics.build_order_statistic_weights(d)
    xweights = w[1:].reshape(1, -1) @ order_statistic_weights
    # do_vrepr_calc = False
    m = gurobipy.Model("coefs")

    vertices = tools.generate_increasing_unit_cube_v(d)[:, 1:]
    v_repr_dim = vertices.shape[0]
    multiplier_weights = xweights @ vertices.T
    lams = m.addMVar(shape=v_repr_dim, lb=0, ub=1)

    m.addConstr(1 == lams.sum())
    signed_crit = (lams[1:].sum() - w[0] - multiplier_weights @ lams)
    criterion, argopt = _max_absval(signed_crit, lams, m)
    argmax = vertices.T @ argopt
    inner_situation = {
        "criterion": criterion,
        "argmax": argmax
    }
    return inner_situation


def build_kd(d: int) -> torch.Tensor:
    order_statistic_weights = order_statistics.build_order_statistic_weights(d)
    v = torch.triu(torch.ones((d, d + 1)), 1)
    b = order_statistic_weights
    bv = b @ v
    assert bv.shape == (d - 1, d + 1)
    kd = bv[:, 1:]
    return kd


def extremely_simple_optimizer2(kdrows: np.ndarray) -> dict:
    """
           -g <= kdrows^T @ beta[1:] - 1 + beta[0] <= +g
    iff
        1 - g <= kdrows^T @ beta                   <= 1 + g.
    Use a closed-form of Farkas' lemma to prove if empty.
    """
    nr, d = kdrows.shape
    m = gurobipy.Model("coefs")

    coefmat = np.concatenate((np.concatenate((np.ones((1, 1)), np.zeros((1, nr))), 1),
                              np.concatenate((np.ones((d, 1)), kdrows.T), 1)), 0)

    responses = np.concatenate((np.zeros((1, 1)), np.ones((d, 1)))).flatten()
    betas = m.addMVar(shape=nr + 1, lb=-float('inf'), ub=float('inf'))
    g = m.addMVar(shape=1)

    for idx in range(d + 1):
        # m.addConstr(+1 * (kdrows.T[idx, :] @ betas) <= +1 + g)
        # m.addConstr(-1 * (kdrows.T[idx, :] @ betas) <= -1 * (+1 - g))
        # m.addConstr(coefmat[idx, :] @ betas - responses[idx] <= +g)
        # m.addConstr(coefmat[idx, :] @ betas - responses[idx] >= -g)
        m.addConstr(coefmat[idx, :] @ betas <= responses[idx] + g)
        m.addConstr(coefmat[idx, :] @ betas >= responses[idx] - g)

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = betas.x

    solution = {
        "criterion": criterion,
        "argmin": argmin
    }
    return solution


def extremely_simple_optimizer(kdrows: np.ndarray) -> np.ndarray:
    """
    -g <= kdrows^T @ beta[1:] - 1 + beta[0] <= +g
    iff
    1 - g <= kdrows^T @ beta <= 1 + g
    Use a closed-form of Farkas' lemma to prove if empty.
    """

    nr, d = kdrows.shape
    m = gurobipy.Model("coefs")

    betas = m.addMVar(shape=nr + 1, lb=-float('inf'), ub=float('inf'))
    g = m.addMVar(shape=1)
    for idx in range(d):
        # m.addConstr(+1 * (kdrows.T[idx, :] @ betas) <= +1 + g)
        # m.addConstr(-1 * (kdrows.T[idx, :] @ betas) <= -1 * (+1 - g))
        m.addConstr(kdrows.T[idx, :] @ betas[1:] - 1 + betas[0] <= +g)
        m.addConstr(kdrows.T[idx, :] @ betas[1:] - 1 + betas[0] >= -g)
    m.addConstr(betas[0] <= +g)
    m.addConstr(betas[0] >= -g)

    m.setObjective(g, gurobipy.GRB.MINIMIZE)
    is_verbose = False
    criterion = tools.smart_optimize(m, is_verbose)
    argmin = betas.x

    solution = {
        "criterion": criterion,
        "argmin": argmin
    }
    return solution


def verify_weight_optimality_v_form3(betas: np.ndarray) -> Dict[str, Any]:
    # New form, without lambda1 substituted out
    d = betas.shape[0]
    kd = build_kd(d)

    betakd = betas[1:].T @ kd.numpy()

    m = gurobipy.Model("coefs")
    lambdas = m.addMVar(shape=d, lb=0.0, ub=1.0)
    # ones = torch.ones((d, 1))
    m.addConstr(lambdas.sum() <= +1)
    signed_crit = (1 - betakd).numpy() @ lambdas - betas[0]
    criterion, argopt = _max_absval(signed_crit, lambdas, m)
    inner_situation = {
        "criterion": criterion,
        "argmax": argopt
    }
    return inner_situation


def verify_weight_optimality_h_form(w: np.ndarray) -> Dict[str, Any]:
    d = w.shape[0]
    order_statistic_weights = order_statistics.build_order_statistic_weights(d)
    xweights = w[1:].reshape(1, -1) @ order_statistic_weights

    m = gurobipy.Model("coefs")
    x = m.addMVar(shape=d, lb=0, ub=1)
    for idx in range(d - 1):
        m.addConstr(x[idx] >= x[idx + 1])
    signed_crit = (x[0] - w[0] - xweights @ x)
    criterion, argmax = _max_absval(signed_crit, x, m)
    inner_situation = {
        "criterion": criterion,
        "argmax": argmax
    }
    return inner_situation


def _print_dk_table_for_d(d: int):
    frac_limit = 2 ** d
    s = tuple(range(d))
    rsets = tools.powerset(s)
    if () in rsets:
        rsets.remove(())

    for rset in rsets:
        dk_situation = get_dk_situation(d, rset)
        criterion = dk_situation["criterion"]
        argmin = dk_situation["argmin"]
        criterion_fraction = fractions.Fraction(criterion).limit_denominator(frac_limit)
        print(d, rset, argmin, criterion_fraction)


def print_dk_table():
    dmax = 6
    for d in range(2, dmax):
        # d = dmax - 3
        _print_dk_table_for_d(d)


def test_tabular_computation1():
    # d = 6
    d = 8
    # d = 9
    # d = 10

    # d = 16
    # rset = (0, 3, 4, 6)
    # rset = (0, 3, 5, 6)
    rset = (0, 2, 5, 6)
    # rset = tuple(range(d))

    assert 0 in rset
    assert 0 == rset[0]

    # dk_situation1 = get_dk_situation(d, rset)
    # dk_situation2 = get_dk_situation2(d, rset)
    # dk_situation3 = get_dk_situation3(d, rset)
    # dk_situation4 = get_dk_situation4(d, rset)
    # dk_situation5 = get_dk_situation5(d, rset)
    dk_situation6 = get_dk_situation6(d, rset)
    dk_situation7 = get_dk_situation7(d, rset)
    dk_situation8 = get_dk_situation8(d, rset)

    dk_situation = dk_situation8
    # print(dk_situation)
    # print(dk_situation2)
    # print(dk_situation3)
    # print(dk_situation4)
    # print(dk_situation5)
    print(dk_situation6)
    print(dk_situation7)
    print(dk_situation8)

    criterion = dk_situation["criterion"]
    argmin = dk_situation["argmin"]
    if tuple(range(d)) == rset:
        assert math.fabs(1 / 2 ** d - criterion) < 1e-14
        assert math.fabs(argmin[0] - criterion) < 1e-14

    # opt_betas = argmin[list(rset[1:])]
    # print(opt_betas)
    # print(reduced_coef_matrix)
    """
    High level argument:
      lower bound on criterion is given by the value at vertices. 
      Lower bound on the optimization over beta is given by duality -- we can demonstrate 
      that the minimum is at least > 0.
      
    """
    # cols = list(range(1, d))
    # omitted_betas = ~np.in1d(cols, rset)


def test_max_fourier_expansion():
    b = np.array([[1], [1], [1], [-1]])
    a = np.array([[3, 3, 1, 1],
                  [1, -1, -1, 1],
                  [-1, -1, 1, 1],
                  [-3, 3, -1, 1]])
    abcd = np.linalg.solve(a, b)

    max3 = lambda _1, _2, _3: abcd[0] * (_1 + _2 + _3) + abcd[1] * (_1 * _2 + _1 * _3 + _2 * _3) + abcd[2] * _1 * _2 * _3 + abcd[3]

    for x1 in [-1, +1]:
        for x2 in [-1, +1]:
            for x3 in [-1, +1]:
                tmp = 1 / 4 * (x1 + x2 + x3) - 1 / 4 * (x1 * x2 + x1 * x3 + x2 * x3) + 3 / 4 * x1 * x2 * x3 + 1 / 4
                assert max3(x1, x2, x3) == max(x1, x2, x3)


def _comparison_term(d: int, ks: List[int]) -> Dict[str, Any]:
    assert all(d >= _ for _ in ks)
    assert 0 in ks
    full_coef_matrix = _build_full_system(d)

    cols = list(range(d))
    included_betas = np.in1d(cols, ks)

    selector_matrix = np.eye(d)[included_betas, :]
    coef_matrix = selector_matrix @ full_coef_matrix
    np.testing.assert_close(coef_matrix, full_coef_matrix[included_betas, :])
    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    a = coef_matrix.T
    b = -1 * target_function_response.reshape(-1, 1)
    nc, nr = a.shape
    num_included_betas = included_betas.sum()

    beta = cvxpy.Variable((num_included_betas, 1))
    nu = cvxpy.Variable((nc, 1))
    objective_dual = cvxpy.Maximize(nu.T @ b - cvxpy.norm(nu, 1) ** 2 / 2)
    objective_prim = cvxpy.Minimize(cvxpy.norm(a @ beta + b, math.inf))

    # at_mat = a.T
    at_mat = selector_matrix @ full_coef_matrix
    constraints_dual = [at_mat @ nu == 0]
    constraints_prim = []
    prob_dual = cvxpy.Problem(objective_dual, constraints_dual)
    prob_prim = cvxpy.Problem(objective_prim, constraints_prim)

    prob_prim.solve(verbose=False)
    prob_dual.solve(verbose=False)
    value = (prob_dual.value * 2) ** .5

    rows = (np.abs(nu.value) > 1e-15).flatten()
    row_signs = np.sign(nu.value)[rows]
    rhs = value * row_signs - b[rows]
    nonzero_beta_value = np.linalg.pinv(a[rows, :]) @ rhs

    atol = 5e-6
    np.testing.assert_close(beta.value.flatten(),
                            nonzero_beta_value.flatten(),
                               atol=atol, rtol=0.0)
    argmin = np.zeros((d,))
    argmin[included_betas] = nonzero_beta_value.flatten()

    ct = {
        "criterion": value,
        "argmin": argmin,
        "nu": nu.value,
        "at_term": a.T
    }
    return ct


def test_betastar_nustar_relationship() -> None:
    d = 7
    # ks = [0, 3, 4, 6]
    # ks = [0, 3, 4, 5]
    # ks = [0, 1, 2, 3, 4, 5, 6]
    index_set1 = set(range(1, d))
    c = 0

    full_coef_matrix = _build_full_system(d)
    order_statistic_weights = order_statistics.build_order_statistic_weights(d)
    cols = list(range(d))

    target_function_response = np.concatenate((np.zeros((1,)), np.ones((d,))))

    # consider adding a term to ks
    # max_subset_size = d - 4
    max_subset_size = d - 2
    for base_subset_size in range(max_subset_size):
        subsets = itertools.combinations(index_set1, base_subset_size)
        for subset in subsets:
            if subset == ():
                continue
            ks0 = [0] + list(subset)
            ct0 = _comparison_term(d, ks0)
            assert 1 + base_subset_size == len(ks0)  # 1 for intercept
            for addtl_row in list(index_set1 - set(ks0)):
                # print(f"c = {c}")
                c += 1
                ks1 = sorted(ks0 + [addtl_row])
                assert 1 + len(ks0) == len(ks1)  # 1 for intercept, 1 for

                ct1 = _comparison_term(d, ks1)

                beta0 = ct0["argmin"]
                beta1 = ct1["argmin"]
                print(beta0, beta1)


def analysis_01d1():
    z = torch.zeros(1)
    # test_tabular_computation1()
    for d in range(4, 19):
        # d = 6
        ks = [0, 1, d - 1]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]

        expected_d1 = d * (d - 1) / ((d - 2) * (d + 1))
        assert math.fabs(argmin[-1] - 1 / (1 - 2 / (d * (d - 1)))) < 1e-6
        assert math.fabs(argmin[-1] - expected_d1) < 1e-6
        assert math.fabs(argmin[1] - -d / (d - 2) / (d + 1)) < 1e-5
        assert math.fabs(argmin[0] - 1 / (2 * (d + 1))) < 1e-5
        assert math.fabs(criterion - argmin[0]) < 1e-5

        betas = torch.cat((torch.tensor(argmin), z)).to(torch.float32)
        model = networks.build_shallowest_network(betas)

        xcrit = torch.triu(torch.ones((d + 1, d)), 0)
        ycrit = model(xcrit).flatten()
        y_act = xcrit.max(1).values
        err_crit = (y_act - ycrit)[[0, d - 2, d - 1, d]]

        print(ycrit[[0, d - 2, d - 1, d]])
        print(xcrit[[0, d - 2, d - 1, d]])

        assert ((err_crit.abs() - criterion).abs() < 1e-6).all()
        assert (torch.sign(err_crit) == torch.tensor([+1, -1] * 2)).all()


def analysis_01d2d1():
    for d in range(4, 19):
        # d = 6
        print((d + 1) * (d - 2) / d / (d - 1))
        ks = [0, 1, d - 2, d - 1]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        # criterion = dk_situation["criterion"]
        expected_d1 = 2.0
        # expected_1 = 1 / ((d - 2) * (d - 1) / 2 - 1)
        expected_1 = 2 / (d * (d - 3))
        expected_2 = -(1 + expected_1)

        expected_11 = 2 / ((d - 2) * (d - 1) - 2)
        expected_111 = 2 / (d * (d - 3))
        assert math.fabs(expected_111 - expected_1) < 1e-6
        assert math.fabs(expected_11 - expected_1) < 1e-6

        assert math.fabs(argmin[1] - expected_1) < 1e-6
        assert math.fabs(argmin[-2] - expected_2) < 1e-6
        assert math.fabs(argmin[-1] - expected_d1) < 1e-6
        assert math.fabs(argmin[0] - 1 / d ** 2) < 1e-8


def analysis_01():
    for d in range(4, 19):
        # d = 6
        ks = [0, 1]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        print(argmin)
        criterion = dk_situation["criterion"]
        # expected = (1 / 2) * 1 / (1 / (d - 1) + 1)
        expected = (1 / 2) * (d - 1) / d
        actual = criterion
        assert math.fabs(actual - expected) < 1e-5
        assert math.fabs(1 - argmin[1]) < 1e-5


def analysis_0():
    for d in range(4, 19):
        # d = 6
        ks = [0]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]
        expected = 1 / 2
        actual = criterion
        assert math.fabs(actual - expected) < 1e-5


def analysis_012():
    z = torch.zeros(1)
    for d in range(3, 25):
        # d = 6
        ks = [0, 1, 2]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]
        # print(d, 1 / (1 / criterion - 2))
        t1 = (d - 1) // 2
        t2 = d // 2

        expected0 = 1 / 2 * (t1 * t2) / ((t1 * t2) + d)
        # print(expected0)
        assert math.fabs(argmin[0] - expected0) < 1e-4

        # expected1 = (1 / (d - 1) - 1) * (4 / t1 + 8) / (2 + (2 * d) / (t1 * t2))
        expected1 = (1 / (d - 1) - 1) * (2 / t1 + 4) / (1 + d / (t1 * t2))
        # expected1 = (-d / (d - 1)) * (4 / t1 + 8) / (2 + (2 * d) / (t1 * t2))
        assert math.fabs(argmin[1] - expected1) < 1e-4

        expected2 = 2 * (1 / t1 + 2) / (1 + d / (t1 * t2))
        assert math.fabs(argmin[2] - expected2) < 1e-4

        # print(d, t1, t2,
        #       1 / (argmin[2] / argmin[0] / 4 - 2) / t1,
        #       argmin[1] / argmin[0] / 4 + 2)
        # print((d - 1) / 2, (criterion - 2))
        # print(2 / (criterion - 1), 1 / d - 2)


def analysis_0123():
    z = torch.zeros(1)
    # test_tabular_computation1()
    for d in range(4, 40):
        # d = 6
        ks = [0, 1, 2, 3]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]
        print(d, d / criterion - 2)


def analysis_012d1():
    dvals = list(range(4, 19))
    vals = np.empty((len(dvals),))
    crits = np.empty((len(dvals),))
    for idx, d in enumerate(dvals):
        # d = 6
        ks = [0, 1, 2, d - 1]
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]
        print(d, ks, 1 / criterion)
        vals[idx] = 1 / criterion
        crits[idx] = criterion

    import matplotlib.pyplot as plt
    plt.plot(vals)
    # probably also some rounding-related weirdness going on


def analysis_s2():
    dvals = list(range(4, 19))

    for idx, d in enumerate(dvals):
        # d = 6
        betas = torch.zeros(d + 1,)
        betas[-3] = 1.0
        network = networks.build_shallowest_network(betas)

        w1 = math.comb(d - 1, d - 3) / math.comb(d, d - 2)
        w2 = math.comb(d - 2, d - 3) / math.comb(d, d - 2)
        w3 = math.comb(d - 3, d - 3) / math.comb(d, d - 2)

        torch.testing.assert_close(w1, (d - 2) / d)
        torch.testing.assert_close(w2, 2 * (d - 2) / (d * (d - 1)))
        torch.testing.assert_close(w3, 2 * 1 / (d * (d - 1)))
        # print(w1, w2, w3)
        assert math.fabs(1 - sum([w1, w2, w3])) < 1e-8

        x = torch.randn(11, d)
        q = torch.tensor([(d - 3), (d - 2), (d - 1)]) / (d - 1)
        qs = torch.quantile(x, q=q, dim=1, interpolation="nearest")
        actual = torch.tensor([[w3], [w2], [w1]]).T @ qs
        y = network(x)
        torch.testing.assert_close(y.flatten(), actual.flatten())


def kd_byhand(d: int) -> torch.tensor:
    kd = torch.empty((d - 1, d))
    for r in range(1, 1 + d - 1):
        for c in range(1, 1 + d):
            kd[r - 1, c - 1] = sum([math.comb(d - i, r - 1) for i in range(1, c + 1)])
        kd[r - 1, :] = kd[r - 1, :] / math.comb(d, r)
    return kd


def analysis_alld1():
    z = torch.zeros(1)
    dvals = list(range(5, 11))
    for idx, d in enumerate(dvals):
        # idx = 2; d = dvals[idx]
        ks = list(range(d))

        # kd = build_kd(d)
        dk_situation = get_dk_situation(d=d, ks=ks)
        argmin = dk_situation["argmin"]
        criterion = dk_situation["criterion"]

        betas = torch.cat((argmin, z)).to(torch.float32)

        x_crit = torch.triu(torch.ones((d + 1, d)), 0)
        model = networks.build_shallowest_network(betas)
        y_crit = model(x_crit).flatten()
        y_act = x_crit.max(1).values
        # print(d, y_crit)
        err_crit = (y_act - y_crit)
        assert torch.abs(err_crit).std() < 1e-5, "for all subsets, err at all vertices equalized"

        order_statistic_weights = order_statistics.build_order_statistic_weights(d)
        bd = order_statistic_weights

        top = torch.cat((torch.eye(1), torch.zeros(1, d - 1)), 1)
        bot = torch.cat((torch.zeros((d, 1)), bd.T), 1)

        gg = torch.cat((top, bot), 0)
        betas_nonzero = betas[:-1]
        resp = gg @ betas_nonzero

        for row in range(1, d + 1):
            terms1 = [-(-1 / 2) ** (d - k) for k in range(1, d)]
            terms2 = [math.comb(d - row, k - 1) for k in range(1, d)]
            terms = [t1 * t2 for t1, t2, in zip(terms1, terms2)]
            # terms = [-(-1 / 2) ** (d - k) * math.comb(d - row, k - 1) for k in range(1, d)]
            print(sum(terms) - resp[row])


if __name__ == "__main__":
    d = 10
    ks = [0, 1, d - 2, d - 1]
    # ks = [0, 1, d - 1]
    # ks = list(range(d))
    get_dk_situation(d=10, ks=[0, 1, 8, 9])

    dk_situation = get_dk_situation(d, ks)
    print(d)
    print(ks)
    print(dk_situation)

    order_statistic_weights = np.array(order_statistics.build_order_statistic_weights(d))
    am = dk_situation["argmin"]
    print(am[1:].T @ order_statistic_weights)
    # test_dual_system()
    analysis_alld1()

    analysis_s2()

    analysis_01d2d1()

    analysis_01d1()

    analysis_01()
    analysis_012d1()

    analysis_012()
    # analysis_0123()

    # analysis_01d3d2d1()
    analysis_01d2d1()

    test_tabular_computation1()
    test_betastar_nustar_relationship()

    # test_tabular_computation1()
    get_dk_situation8(d=8, ks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
