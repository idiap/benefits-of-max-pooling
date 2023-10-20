
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Kyle Matoba <kmatoba@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import logging
import functools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
import scipy.spatial
import scipy.linalg
import scipy.sparse
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1337)))

import cvxpy
import gurobipy

logger = logging.getLogger(__name__)

# https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html#sec:StatusCodes

success_statuses = [gurobipy.GRB.OPTIMAL]
infeasible_statuses = [gurobipy.GRB.INFEASIBLE]
unbounded_statuses = [gurobipy.GRB.UNBOUNDED]
failure_statuses = infeasible_statuses + unbounded_statuses
whitelisted_statuses = success_statuses + failure_statuses


@functools.lru_cache(maxsize=16)
def powerset(i: tuple) -> List[tuple]:
    if len(i) >= 22:
        warnings.warn("Computing the power set of collections this large can take significant amounts of time")
    s = list(i)
    chained = itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )
    return list(chained)


def onehot_encode(q: np.ndarray) -> np.ndarray:
    dim = q.shape[1]
    q_argmaxes = np.argmax(q, axis=1)
    nr = len(q_argmaxes)
    ohed = np.zeros((nr, dim))
    ohed[np.arange(nr), q_argmaxes] = 1
    return ohed


def vec(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (-1, 1))


@functools.lru_cache(maxsize=16)
def gen_all_01_rows(n: int) -> np.ndarray:
    index = tuple(range(n))
    index_powerset = powerset(index)
    r, c = zip(*[([i] * len(x), x) for i, x in enumerate(index_powerset)])

    i = list(itertools.chain.from_iterable(r))
    j = list(itertools.chain.from_iterable(c))
    ones = [1] * len(i)
    uc = scipy.sparse.coo_matrix((ones, (i, j)), dtype=float)
    all_01_rows = uc.toarray()
    return all_01_rows


@functools.lru_cache(maxsize=16)
def positive_orthant_v_repr(n: int) -> np.ndarray:
    v_repr = np.hstack([np.zeros((n, 1)), np.eye(n)])
    return v_repr


@functools.lru_cache(maxsize=16)
def positive_orthant_h_repr(n: int) -> np.ndarray:
    h_repr = np.hstack([np.zeros((n, 1)), np.eye(n)])
    return h_repr


@functools.lru_cache(maxsize=16)
def unit_cube_v_repr(n: int) -> np.ndarray:
    all_01_rows = gen_all_01_rows(n)
    v_repr = np.hstack([np.ones((all_01_rows.shape[0], 1)), all_01_rows])
    return v_repr


@functools.lru_cache(maxsize=16)
def unit_cube_h_repr(n: int) -> np.ndarray:
    # remember: b - Ax >= 0
    # 1 - x >= 0 iff 1 >= x
    is_le1 = np.hstack([np.ones((n, 1)), -1 * np.eye(n)])
    # 0 + x >= 0 iff x >= 0
    is_ge0 = np.hstack(([np.zeros((n, 1)), np.eye(n)]))
    h_repr = np.vstack([is_le1,
                        is_ge0])
    return h_repr


@functools.lru_cache(maxsize=16)
def cross_polytope_v_repr(n: int) -> np.ndarray:
    eye_n = np.eye(n)
    eye_pm = np.vstack((+1 * eye_n, -1 * eye_n))
    v_repr = np.concatenate((np.ones((2 * n, 1)), eye_pm), axis=1)
    return v_repr


@functools.lru_cache(maxsize=16)
def simplex_h_repr(n: int) -> np.ndarray:
    sums_to_less_than1 = np.hstack([np.ones((1, 1)), -1 * np.ones((1, n))])
    is_greater_than0 = np.hstack(([np.zeros((n, 1)), np.eye(n)]))
    h_repr = np.vstack([sums_to_less_than1, is_greater_than0])
    return h_repr


@functools.lru_cache(maxsize=16)
def nth_canonical_basis(n: int, dim: int) -> np.ndarray:
    return vec((n == np.arange(dim)).astype(float))


def find_point_multipliers(point: np.ndarray,
                           basis_v_repr: np.ndarray) -> np.ndarray:
    """
    this function finds the minimum norm Lagrange
    multipliers that deliver a point.
    """
    n = point.shape[0]
    vertices = basis_v_repr[:, 1:]
    v, _ = vertices.shape
    assert _ == n

    is_ray = 0 == basis_v_repr[:, 0]
    is_pol = ~is_ray

    weights = cvxpy.Variable(v)
    eye_v = np.eye(v)
    constraints = [0 <= weights, vertices.T @ weights == point.flatten()]
    if np.any(is_pol):
        ipf = vec(is_pol).T
        constraints = constraints + [ipf @ weights == 1]
    # objective = cvxpy.Minimize((1 / 2) * cvxpy.quad_form(weights, eye_v))
    objective = cvxpy.Minimize(cvxpy.norm(weights, 1))
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()

    point_multipliers = np.vstack(weights.value)
    disc = vec(point) - vertices.T @ point_multipliers
    assert np.linalg.norm(disc, np.inf) < 1e-13
    return point_multipliers


def build_bounding_box_h_form(x_lower: np.ndarray,
                              x_upper: np.ndarray) -> np.ndarray:
    m, _ = x_lower.shape
    assert 1 == _
    assert m, 1 == x_upper.shape
    # x >= x_lower iff
    # -x_lower + I x >= 0
    # and
    # x <= x_upper iff
    # +x_upper - I x >= 0
    eyem = np.eye(m)
    h_lower = np.hstack([-1 * x_lower, +1 * eyem])
    h_upper = np.hstack([+1 * x_upper, -1 * eyem])

    h_lower = h_lower[np.isfinite(x_lower).flatten(), :]
    h_upper = h_upper[np.isfinite(x_upper).flatten(), :]

    h_bounds = np.vstack([h_lower, h_upper])
    return h_bounds


def generate_increasing_unit_cube_v(dim: int) -> np.ndarray:
    increasing_unit_cube_v = np.tril(np.ones(dim + 1))
    return increasing_unit_cube_v


def apply_linear_transformation_to_v_repr(aa: np.ndarray,
                                          w: np.ndarray,
                                          b: np.ndarray) -> np.ndarray:
    # aa = v_repr of polytope
    """
    Given a set a = {Rl, l >= 0, 1'l = 1} compute {wx + b : x in a}
    fix an x, since x in a, there exists an l such that

    x = sum_j r_j l_j

    thus,
    wx + b

    w (sum_j r_j l_j) + b =
    sum_j wr_j l_j + b =
    sum_j (wr_j + b)l_j
    """
    dim_in = aa.shape[1] - 1
    dim_out = len(b)
    assert dim_out, dim_in == w.shape

    if np.all(0 == aa[:, 0]):
        # no vertices, just rays
        # (harmlessly) append a vertex of zero, so that everything else goes through below
        origin_vertex = np.hstack((np.eye(1), np.zeros((1, dim_in))))
        aa = np.vstack([origin_vertex, aa])
    is_ray = (0 == aa[:, 0])
    is_pol = ~is_ray

    vertices = aa[:, 1:]
    to_add = vec(is_pol) @ b.T
    transformed_vertices = (w @ vertices.T).T + to_add
    transformed_a = np.hstack((vec(aa[:, 0]), transformed_vertices))
    return transformed_a


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def build_polytope_where_nth_coordinate_is_greatest(n: int,
                                                    dim: int,
                                                    margin: float) -> np.ndarray:
    iota = nth_canonical_basis(n, dim)
    one = np.ones((dim, 1))

    c_full = one @ iota.T - np.eye(dim)
    class_c = c_full[1 != iota.flatten(), :]
    nr = class_c.shape[0]
    thresholds = np.ones((nr, 1)) * margin
    polytope_where_nth_coordinate_is_greatest = np.hstack((thresholds, class_c))
    return polytope_where_nth_coordinate_is_greatest


def build_prototype_from_v_form(v_gen: np.ndarray,
                                v_lin: np.ndarray) -> np.ndarray:
    assert 0 == v_lin.shape[0], "Not supporting nonempty v_lin yet"
    # uniform over vertices, zero rays:
    is_vertex = (0 != v_gen[:, 0])
    to_average = v_gen[is_vertex, :]
    prototype = np.mean(to_average, axis=0)
    return prototype


def build_prototype_from_h_form(h_ineq: np.ndarray,
                                h_lin: np.ndarray,
                                objective_sense: cvxpy.problems.objective,
                                p: float,
                                fall_back_to_vacuous_criterion: bool) -> np.ndarray:
    assert objective_sense == cvxpy.Minimize, "only minimize supported for now"
    kwargs = {
        "solver": cvxpy.OSQP,
        'max_iter': 15000,
        'eps_abs': 1e-5,
        'eps_rel': 1e-5
    }

    verbose = False
    dim = h_ineq.shape[1]
    x = cvxpy.Variable(dim)

    iota0 = vec(0 == np.arange(dim)).astype(float)

    constraints = [iota0.T @ x == 1]
    if 0 < h_lin.shape[0]:
        constraints += [h_lin @ x == 0]

    if 0 < h_ineq.shape[0]:
        constraints += [h_ineq @ x >= 0]

    objective = objective_sense(cvxpy.norm(x, p))  # try to get non-zero-ish answers
    prob = cvxpy.Problem(objective, constraints)

    try:
        value = prob.solve(verbose=verbose, **kwargs)
    except Exception as e:
        warnings.warn("Caught {} but continuing".format(e))
        if fall_back_to_vacuous_criterion:
            objective = cvxpy.Minimize(0)
            prob = cvxpy.Problem(objective, constraints)
            try:
                value = prob.solve(verbose=verbose)
            except Exception as e:
                warnings.warn("Caught {} but continuing".format(e))
                value = np.inf
        else:
            value = np.inf
    # prob.solve(verbose=True)
    ie = np.isinf(value)
    if ie:
        prototype = None
    else:
        prototype = vec(x.value)
    return prototype


def _minkowski_sum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert 2 == len(x.shape)
    assert 2 == len(y.shape)

    rx, cx = x.shape
    ry, cy = y.shape

    assert cx == cy

    xlist = x.tolist()
    ylist = y.tolist()

    xyprod = itertools.product(xlist, ylist)
    xy = np.array([np.array(ix) + np.array(iy) for ix, iy in xyprod])
    total_sum = xy
    return total_sum


def convex_union_v_reprs(v1: np.ndarray,
                         v2: np.ndarray) -> np.ndarray:
    v1_row_set = _matrix_to_row_set(v1)
    v2_row_set = _matrix_to_row_set(v2)
    v12_union = set.union(v1_row_set, v2_row_set)
    return _row_set_to_matrix(v12_union)


def _matrix_to_row_set(m: np.ndarray) -> set:
    return set(map(tuple, m))


def _row_set_to_matrix(rs: set) -> np.ndarray:
    return np.array(list(rs))


def same_unique_rows(m1: np.ndarray, m2: np.ndarray) -> bool:
    """ Check whether matrices have the same unique rows """
    assert 2 == m1.ndim and 2 == m2.ndim, "Only 2d arrays supported"
    return _matrix_to_row_set(m1) == _matrix_to_row_set(m2)


def is_inequality_valid(a: np.ndarray,
                        b: float,
                        poly_a: np.ndarray,
                        poly_b: np.ndarray,
                        tol: float = 0) -> bool:
    dim = len(a)
    x = cvxpy.Variable(dim)

    constraints = [poly_a @ x <= poly_b]
    objective = cvxpy.Maximize(a @ x)
    prob = cvxpy.Problem(objective, constraints)
    solver = cvxpy.GUROBI
    # solver = cvxpy.MOSEK
    prob.solve(solver=solver)

    value = prob.value
    tf = value <= b + tol
    return tf


def compute_valid_inequalities_for_poly(ineqs: np.ndarray,
                                        poly: np.ndarray) -> np.ndarray:
    tol = 1e-8
    num_ineqs = ineqs.shape[0]
    are_inequalities_valid_for_poly = np.full((num_ineqs,), False)

    poly_a = -1 * poly[:, 1:]
    poly_b = poly[:, 0]

    for idx in range(num_ineqs):
        # idx = 0
        a = -1 * ineqs[idx, 1:]
        b = ineqs[idx, 0]
        iv = is_inequality_valid(a, b, poly_a, poly_b, tol)
        are_inequalities_valid_for_poly[idx] = iv
    return are_inequalities_valid_for_poly


def _inner_optimisation(h1: np.ndarray,
                        h2: np.ndarray,
                        hbar: np.ndarray) -> float:
    dim = hbar.shape[1]

    assert 1, dim == h1.shape
    assert 1, dim == h2.shape
    xe = cvxpy.Variable(dim + 1)
    iota_0 = vec(0 == np.arange(dim + 1)).astype(float)
    iota_e = vec(dim == np.arange(dim + 1)).astype(float)

    c1 = np.hstack([-1 * h1, -1])
    c2 = np.hstack([-1 * h2, -1])
    cenv = np.hstack([hbar, np.zeros((hbar.shape[0], 1))])

    constraints = [c1 @ xe >= 0,
                   c2 @ xe >= 0,
                   cenv @ xe >= 0,
                   iota_0.T @ xe == 1]

    objective = cvxpy.Maximize(iota_e.T @ xe)
    prob = cvxpy.Problem(objective, constraints)

    # solver = cvxpy.MOSEK
    solver = cvxpy.GUROBI
    prob.solve(solver=solver)
    # prob.solve()
    eps_star = prob.value
    return eps_star


def drop_rows_positive_proportional_to_another(r: np.ndarray,
                                               tol: float) -> bool:
    """
    r = np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]])
    """
    # np.set_printoptions(linewidth=1000)
    dotprods = r @ r.T
    norms = vec(np.diag(dotprods) ** .5)
    denominators = norms @ norms.T
    quotients = dotprods / denominators

    # all_zero = np.all(np.abs(r) < tol, axis=1)

    triu_quotients = np.triu(quotients, k=+1)
    is_positively_proportional_to_another_row = np.any(triu_quotients >= 1 - tol, axis=1)
    dropped_r = r[~is_positively_proportional_to_another_row, :]
    return dropped_r


def h_tuple_to_matrix(h_tuple: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    h = h_tuple[0]
    h_lin = h_tuple[1]

    # ax == b iff
    # ax <= b
    # ax >= b iff -ax <= -b
    # h_lin_ineq =
    h_matrix = np.vstack([h, h_lin, -1 * h_lin])
    return h_matrix


def is_h_form_empty(h_ineq: np.ndarray,
                    h_lin: np.ndarray) -> bool:
    dim = h_ineq.shape[1]
    p1 = cvxpy.Variable(dim)

    iota0 = vec(0 == np.arange(dim)).astype(float)

    constraints = [iota0.T @ p1 == 1]  # first column is of ones
    if 0 < h_lin.shape[0]:
        constraints += [h_lin @ p1 == 0]

    if 0 < h_ineq.shape[0]:
        constraints += [h_ineq @ p1 >= 0]
    objective = cvxpy.Minimize(0)

    prob = cvxpy.Problem(objective, constraints)

    # solver = cvxpy.MOSEK
    # solver = cvxpy.GUROBI
    # value = prob.solve(solver=solver)
    value = prob.solve()
    ie = np.isinf(value)
    return ie


def compute_hull_volume(vertices: np.ndarray) -> float:
    dim = vertices.shape[1]
    zero_rows = (0 == vertices.shape[0])
    if zero_rows or (np.linalg.matrix_rank(vertices) < dim):
        hull_volume = 0
    else:
        hull = scipy.spatial.ConvexHull(vertices)
        hull_volume = hull.volume
    return hull_volume


def compute_minimum_volume_outer_box(h_ineq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Algorithm from 2.4 of "Inner and outer approximations of polytopes using boxes"
    # http://cse.lab.imtlucca.it/~bemporad/publications/papers/compgeom-boxes.pdf
    a = -1 * h_ineq[:, 1:]
    b = np.vstack(h_ineq[:, 0])
    dim = a.shape[1]

    lower = np.empty((dim,))
    upper = np.empty((dim,))

    for idx in range(dim):
        # idx = 0
        c = nth_canonical_basis(idx, dim)

        m_l = gurobipy.Model("lower")
        m_u = gurobipy.Model("upper")

        lb = np.array([-1 * gurobipy.GRB.INFINITY] * dim)
        x_l = m_l.addMVar(shape=dim, lb=lb)
        x_u = m_u.addMVar(shape=dim, lb=lb)

        m_l.addConstr(a @ x_l <= b.flatten())
        m_u.addConstr(a @ x_u <= b.flatten())

        m_l.setObjective(c.T @ x_l, gurobipy.GRB.MINIMIZE)
        m_u.setObjective(c.T @ x_u, gurobipy.GRB.MAXIMIZE)

        criterion_l = smart_optimize(m_l)
        criterion_u = smart_optimize(m_u)

        # assert np.all(lower <= upper),
        lower[idx] = criterion_l
        upper[idx] = criterion_u
    return lower, upper


def compute_maximum_volume_inner_box(h_ineq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Algorithm 1 from https://doi.org/10.1016/S0925-7721(03)00048-8
    a = -1 * h_ineq[:, 1:]
    b = np.vstack(h_ineq[:, 0])
    a_plus = np.clip(a, a_min=0, a_max=None)

    dim = a.shape[1]
    x = cvxpy.Variable(dim)
    y = cvxpy.Variable(dim)

    constraints = [a @ x + a_plus @ y <= b.flatten()]
    objective = cvxpy.Maximize(cvxpy.sum(cvxpy.log(y)))
    prob = cvxpy.Problem(objective, constraints)

    try:
        verbose = False
        # solver = cvxpy.MOSEK
        # solver = cvxpy.GUROBI
        # prob.solve(solver=solver, verbose=verbose)
        prob.solve(verbose=verbose)
    except Exception as err:
        print(err)

    prob_value = prob.value
    assert prob_value is not None
    if np.isinf(prob_value):
        lower = None
        upper = None
    else:
        x_value = x.value
        y_value = y.value

        lower = np.vstack(x_value)
        upper = np.vstack(x_value + y_value)
    return lower, upper


def refine_status(m: gurobipy.Model) -> gurobipy.GRB.Status:
    if m.Status == gurobipy.GRB.INF_OR_UNBD:
        # https://www.gurobi.com/documentation/9.0/refman/py_model_copy.html
        m_copy = m.copy()
        m_copy.Params.DualReductions = 0
        m_copy.optimize()
        refined_status = m_copy.Status
    else:
        refined_status = m.Status
    assert refined_status in whitelisted_statuses, f"Unknown status code {refined_status}"
    return refined_status


def smart_optimize(m: gurobipy.Model,
                   is_verbose: bool = False) -> float:
    m.setParam('OutputFlag', is_verbose)
    m.optimize()
    refined_status = refine_status(m)
    if refined_status in success_statuses:
        criterion = m.objVal
    elif refined_status in infeasible_statuses:
        # The default 1 value indicates that the objective is to minimize the objective.
        # Setting this attribute to -1 changes the sense to maximization.
        # https://www.gurobi.com/documentation/9.0/refman/modelsense.html
        model_sense = m.getAttr(gurobipy.GRB.Attr.ModelSense)
        criterion = model_sense * float("inf")
    elif refined_status == gurobipy.GRB.UNBOUNDED:
        m_copy = m.copy()
        m_copy.Params.DualReductions = 0
        m_copy.setObjective(0.0)
        m_copy.optimize()
        refined_status = m_copy.Status

        raise ValueError(f"Why did you give me an unbounded problem?")
    else:
        raise ValueError(f"Not configured to handle status {refined_status}")
    return criterion


def _solve_dual_problem(a: np.ndarray,
                        b: np.ndarray,
                        a_eq: Optional[np.ndarray],
                        b_eq: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    has_eq_constrs = a_eq is not None
    if has_eq_constrs:
        assert b_eq is not None
    if not has_eq_constrs:
        a_eq = np.empty((0, a.shape[1]))
        b_eq = np.empty((0, 1))
    assert a_eq.shape[0] == b_eq.shape[0]

    n_ineq_cons, dim = a.shape
    n_eq_cons = a_eq.shape[0]
    assert a_eq.shape[1] == dim
    assert a_eq.shape[0] == b_eq.shape[0]

    one = np.ones((n_ineq_cons, 1))

    # Equation 8.5 in Fukuda 2015
    m = gurobipy.Model("matrix1")
    s = m.addMVar(shape=n_ineq_cons)
    t = m.addMVar(shape=1)

    m.setObjective(b.T @ s + t, gurobipy.GRB.MINIMIZE)
    m.addConstr(a.T @ s == 0)
    m.addConstr(one.T @ s + t == 1)
    criterion = smart_optimize(m)
    argmin = s.x
    return argmin, criterion


def _solve_primal_problem(a: np.ndarray,
                          b: np.ndarray,
                          a_eq: np.ndarray,
                          b_eq: np.ndarray) -> Tuple[np.ndarray, float]:
    n_ineq_cons, dim = a.shape
    n_eq_cons, _ = a_eq.shape
    assert _ == dim
    assert n_eq_cons == b_eq.shape[0]
    assert n_ineq_cons == b.shape[0]

    a_sp = scipy.sparse.csr_matrix(a)
    a_sp_eq = scipy.sparse.csr_matrix(a_eq)

    c_sp = scipy.sparse.csr_matrix(([1.0], ([0], [0])),
                                   shape=(1, 1 + dim))
    c = c_sp.toarray().flatten()
    a_aug_sp = scipy.sparse.hstack((np.ones((n_ineq_cons, 1)), a_sp))
    a_aug_eq_sp = scipy.sparse.hstack((np.zeros((n_eq_cons, 1)), a_sp_eq))

    m = gurobipy.Model("matrix1")
    lb = np.array([-1 * gurobipy.GRB.INFINITY] * (dim + 1))
    ub = np.array([1.0] + [+1 * gurobipy.GRB.INFINITY] * dim)

    x = m.addMVar(shape=dim + 1, lb=lb, ub=ub, name="x")
    m.setObjective(c.T @ x, gurobipy.GRB.MAXIMIZE)
    #  m.setParam('OutputFlag', is_verbose)
    m.addConstr(a_aug_sp @ x <= b.flatten(), name="ineq")
    m.addConstr(a_aug_eq_sp @ x == b_eq.flatten(), name="eq")

    criterion = smart_optimize(m)

    if np.isfinite(criterion):
        assert criterion == x.x[0]
        argmax = x.x
        if False:
            test_val = argmax * .9
            b.flatten() - a @ argmax
            b.flatten() - a @ test_val
    else:
        argmax = None
    if criterion < -1e-14:
        pass
        # warnings.warn("This polytope seems to be empty")
        # raise ValueError("This polytope seems to be empty")
    return argmax, criterion


def h_repr_dim(h_ineq: np.ndarray, h_lin: np.ndarray) -> int:
    """
    A new try, attempting to code up in more idiomatic gurobipy
    """
    a = -1 * h_ineq[:, 1:]
    b = np.reshape(h_ineq[:, 0], (-1, 1))
    n_ineq_cons, dim = a.shape

    aeq = -1 * h_lin[:, 1:]
    beq = np.reshape(h_lin[:, 0], (-1, 1))

    curr_a = a
    curr_b = b
    curr_aeq = aeq
    curr_beq = beq
    curr_dim = dim

    done = False
    while not done:
        pp_argmax, pp_criterion = _solve_primal_problem(curr_a, curr_b, curr_aeq, curr_beq)
        done = pp_criterion > 1e-14
        # print(curr_a.shape, pp_criterion, curr_dim)
        if not done:
            assert pp_criterion > -1e-14
            curr_dim = curr_dim - 1
            assert curr_dim >= 0, "dim needs to be >= 0"
            dp_argmax, dp_criterion = _solve_dual_problem(curr_a, curr_b, curr_aeq, curr_beq)
            assert np.any(dp_argmax > 0), "This should not happen"
            idx = _find_indices_to_fix_at_equality(dp_argmax)

            curr_aeq = np.vstack((curr_aeq, curr_a[idx, :]))
            curr_beq = np.vstack((curr_beq, curr_b[idx, :]))
            curr_a = np.delete(curr_a, idx, axis=0)
            curr_b = np.delete(curr_b, idx, axis=0)

    polytope_dim = curr_dim
    assert dim == polytope_dim + curr_aeq.shape[0]
    # polytope_dim = curr_a.shape[0]
    return polytope_dim


def _find_indices_to_fix_at_equality(dp_argmax: np.ndarray) -> Iterable[int]:
    # NB: this can be done much better as follows:
    """
    By Gaussian elimination, we can recognize all other inequalities
    in Ax ≤ b that are forced to be equalities provided AIx = bI. Let us merge I with these 
    dependent equality indices,
    """
    indices = np.argmax(dp_argmax > 0)
    return indices


def v_repr_dim(v_repr: np.ndarray) -> int:
    # Section 8.3 of https://people.inf.ethz.ch/fukudak/lect/pclect/notes2015/PolyComp2015.pdf
    # For P = {x : x = Vl + Rm, 1'l = 1, l >= 0, m >= 0}
    # dim P = rank [[V, R]; [1', 0']] - 1
    assert np.all(np.in1d(v_repr[:, 0], [0, 1]))
    if 0 == v_repr.shape[0]:
        polytope_dim = -1
    else:
        polytope_dim = np.linalg.matrix_rank(v_repr.T) - 1
    return polytope_dim


def is_full_dim_h_repr(h_repr: np.ndarray,
                       h_lin: np.ndarray) -> bool:
    a = -1 * h_repr[:, 1:]
    b = np.vstack(h_repr[:, 0])

    a_eq = -1 * h_lin[:, 1:]
    b_eq = np.reshape(h_lin[:, 0], (-1, 1))

    argmax, criterion = _solve_primal_problem(a, b, a_eq, b_eq)
    is_full_dim = criterion > 0
    return is_full_dim


def outer_n(dims: int, grid: np.ndarray) -> np.ndarray:
    args = (grid for _ in range(dims))
    out = np.meshgrid(*args)
    to_stack = tuple(_.flatten() for _ in out)
    stacked = np.column_stack(to_stack)
    return stacked


if __name__ == "__main__":
    n = 6
    # point = rs.uniform(size=(n, 1))
    # basis_v_repr = unit_cube_v_repr(n)
    # # basis_v_repr = np.concatenate((np.ones((n, 1)), basis), axis=1)
    # point_multipliers = find_point_multipliers(point, basis_v_repr)
    # print(np.round(point_multipliers, 3))
