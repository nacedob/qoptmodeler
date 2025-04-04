from types import NoneType
from typing import Tuple
import numpy as np


class QuantumTranslator:

    def __init__(self,
                 quad_cost_matrix: np.ndarray,
                 lin_cost_matrix: np.ndarray,
                 lhs_eq_matrix: np.ndarray = None,
                 rhs_eq_vector: np.ndarray = None,
                 lhs_ineq_matrix: np.ndarray = None,
                 rhs_ineq_vector: np.ndarray = None,
                 penalty_factor: float = None) -> None:
        self.Q = np.asarray(quad_cost_matrix)
        self.l = np.asarray(lin_cost_matrix)
        self.G = np.asarray(lhs_ineq_matrix)
        self.h = np.asarray(rhs_ineq_vector)
        self.A = np.asarray(lhs_eq_matrix)
        self.b = np.asarray(rhs_eq_vector)
        self.penalty_factor = penalty_factor or self._compute_default_penalty_factor()

        self._assert_input_data()

    def _assert_input_data(self):
        # Assert data types --------------------
        assert isinstance(self.Q, np.ndarray), f'quad_cost_matrix must be of type np.nadarray. Got {type(self.Q)}'
        assert isinstance(self.l, np.ndarray), f'lin_cost_matrix must be of type np.nadarray. Got {type(self.l)}'
        assert isinstance(self.G,
                          (np.ndarray, NoneType)), f'lhs_ineq_matrix must be of type np.nadarray. Got {type(self.G)}'
        assert isinstance(self.h,
                          (np.ndarray, NoneType)), f'rhs_ineq_vector must be of type np.nadarray. Got {type(self.h)}'
        assert isinstance(self.A,
                          (np.ndarray, NoneType)), f'lhs_eq_matrix must be of type np.nadarray. Got {type(self.A)}'
        assert isinstance(self.b,
                          (np.ndarray, NoneType)), f'rhs_eq_vector must be of type np.nadarray. Got {type(self.b)}'

        # Assert dimensions --------------------
        # 1. Cost function
        assert self.Q.shape[0] == self.Q.shape[1], \
            'Quadratic cost matrix must be square'
        assert self.l.shape[0] == self.Q.shape[0], \
            'Linear cost matrix must have the same number of rows as the quadratic cost matrix'

        # 2. Equality constraints
        if self.A.shape != () or self.b.shape != ():
            assert self.A.shape != () and self.b.shape != (), (
                "Both lhs_ineq_constraint and rhs_ineq_constraint must be provided together."
            )
            assert self.A.shape[1] == self.Q.shape[1], \
                'Equality constraint matrix must have the same number of columns as the linear cost matrix'
            assert self.b.shape[0] == self.A.shape[0], \
                'Equality constraint vector must have the same number of rows as the equality constraint matrix'

        # 3. Inequality constraints
        if self.G.shape != () or self.h.shape != ():
            assert self.G.shape != () and self.h.shape != (), (
                "Both lhs_eq_constraint and rhs_eq_constraint must be provided together."
            )
            assert self.G.shape[1] == self.Q.shape[1], \
                'Inequality constraint matrix must have the same number of columns as the quadratic cost matrix'
            assert self.h.shape[0] == self.G.shape[0], \
                'Inequality constraint vector must have the same number of rows as the inequality constraint matrix'

        # For now
        if self.G.shape != () or self.h.shape != ():
            raise NotImplementedError("Conversion to QUBO format with inequality constraints is not yet implemented.")

    def _compute_default_penalty_factor(self) -> float:
        """
        Computes a default penalty factor based on the size of the quadratic objective function
        """
        return self.Q.max().max() * np.prod(self.Q.shape)

    def to_qubo(self) -> np.ndarray:
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
        :return: np.ndarray, the QUBO representation as a ndarray
        """
        # Data consistency

        # Convert to QUBO format
        qubo = self.Q.copy()
        l_tilde = self.l.copy()
        if self.A.shape != ():
            qubo += self.penalty_factor * self.A.T @ self.A
            l_tilde -= 2 * (self.penalty_factor * self.b.reshape(1, -1) @ self.A).reshape(l_tilde.shape)
        qubo += np.diag(l_tilde)
        return qubo

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a Quadratic Unconstrained Binary Optimization (QUBO) problem into an Ising model.
        Reference: https://learning.quantum.ibm.com/tutorial/quantum-approximate-optimization-algorithm

        Args:
        - Q (np.ndarray): Square matrix representing the QUBO coefficients.
        - l (np.ndarray, optional): Vector representing the linear terms in the QUBO problem. Default is None.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - J (np.ndarray): The Ising model matrix.
            - h (np.ndarray): The Ising model vector.

        Raises:
        - AssertionError: If Q is not a square matrix or if the shape of l does not match Q when provided.
        """
        # Ising transformation
        J = self.to_qubo()
        h = -(J + J.T) @ np.ones(J.shape[0])
        return J, h
