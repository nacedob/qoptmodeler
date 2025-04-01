from types import NoneType
from typing import Tuple
import numpy as np


def optimization_problem_to_qubo(Q: np.ndarray,
                                 l: np.ndarray,
                                 A: np.ndarray = None,
                                 b: np.ndarray = None,
                                 penalty_factor: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a general optimization problem represented by a quadratic objective function Q,
    linear constraints l, and optionally a system of linear equalities A and b, into a QUBO format.
    argmin  x.T @ Q @ x + l.T @ x
    subject to: A @ x == b
               x_i \in \{0,1\}

    :param Q: np.ndarray, the quadratic objective function contribution.
    :param l: np.ndarray, the linear objective function contribution.
    :param A: np.ndarray, the equality constraints matrix.
    :param b: np.ndarray, the right-hand side of the equality constraints.
    :param penalty_factor: float, the penalty factor for constraints violations.
    :return: Tuple
    """
    # Data consistency
    assert isinstance(Q, np.ndarray), 'Q must be a numpy array'
    assert isinstance(l, np.ndarray), 'l must be a numpy array'
    assert isinstance(A, (np.ndarray, NoneType)), 'If provided, A must be a numpy array'
    assert isinstance(b, (np.ndarray, NoneType)), 'If provided, b must be a numpy array'
    assert (A is None and b is None) or (A is not None and b is not None), \
        "Both A and b must either be None or not None"
    if A is not None and b is not None:
        assert A.shape[0] == b.shape[0], 'The number of rows in A must match the number of rows in b'
        assert Q.shape[0] == A.shape[1], 'The number of columns in Q must match the number of rows in A'

    # Convert to QUBO format
    Q_tilde = Q
    l_tilde = l
    if A is not None:
        penalty_factor = penalty_factor or _compute_default_penalty_factor(Q)
        Q_tilde += penalty_factor * A.T @ A
        l_tilde -= 2 * penalty_factor * b.T @ A
    return Q_tilde, l_tilde


def _compute_default_penalty_factor(Q: np.ndarray) -> float:
    """
    Computes a default penalty factor based on the size of the quadratic objective function
    """
    return Q.max().max() * np.prod(Q.shape)
