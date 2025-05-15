from types import NoneType
from typing import Tuple, Union, Sequence, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp


class QuantumTranslator:

    def __init__(self,
                 quad_cost_matrix: np.ndarray,
                 lin_cost_matrix: np.ndarray,
                 lhs_eq_matrix: np.ndarray = None,
                 rhs_eq_vector: np.ndarray = None,
                 lhs_ineq_matrix: np.ndarray = None,
                 rhs_ineq_vector: np.ndarray = None,
                 number_slacks: Union[int, Sequence] = None,
                 penalty_factor: float = None) -> None:
        """
        Initialize the QuantumTranslator with optimization problem parameters.

        - The objective function is of the form x^T Q x + l^T x.
        - The equality constraints are of the form Ax = b.
        - The inequality constraints are of the form Gx <= h.

        Parameters
        ----------
        quad_cost_matrix : np.ndarray
            The quadratic cost matrix Q in the objective function x^T Q x.
        lin_cost_matrix : np.ndarray
            The linear cost vector l in the objective function l^T x.
        lhs_eq_matrix : np.ndarray, optional
            The left-hand side matrix A of equality constraints Ax = b.
        rhs_eq_vector : np.ndarray, optional
            The right-hand side vector b of equality constraints Ax = b.
        lhs_ineq_matrix : np.ndarray, optional
            The left-hand side matrix G of inequality constraints Gx <= h.
        rhs_ineq_vector : np.ndarray, optional
            The right-hand side vector h of inequality constraints Gx <= h.
        penalty_factor : float, optional
            The penalty factor for constraint violations. If None, a default value is computed.
        """
        self.Q = np.asarray(quad_cost_matrix)
        self.l = np.asarray(lin_cost_matrix)
        self.G = np.asarray(lhs_ineq_matrix)
        self.h = np.asarray(rhs_ineq_vector)
        self.A = np.asarray(lhs_eq_matrix)
        self.b = np.asarray(rhs_eq_vector)
        self.slack_matrix, self.n_total_slacks = self._get_slack_matrix(number_slacks)
        self.penalty_factor = penalty_factor or self._compute_default_penalty_factor()

        self._assert_input_data()

    def _assert_input_data(self):
        # Assert data types --------------------
        assert isinstance(self.Q, np.ndarray), f'quad_cost_matrix must be of type np.ndarray. Got {type(self.Q)}'
        assert isinstance(self.l, np.ndarray), f'lin_cost_matrix must be of type np.ndarray. Got {type(self.l)}'
        assert isinstance(self.G,
                          (np.ndarray, NoneType)), f'lhs_ineq_matrix must be of type np.ndarray. Got {type(self.G)}'
        assert isinstance(self.h,
                          (np.ndarray, NoneType)), f'rhs_ineq_vector must be of type np.ndarray. Got {type(self.h)}'
        assert isinstance(self.A,
                          (np.ndarray, NoneType)), f'lhs_eq_matrix must be of type np.ndarray. Got {type(self.A)}'
        assert isinstance(self.b,
                          (np.ndarray, NoneType)), f'rhs_eq_vector must be of type np.ndarray. Got {type(self.b)}'

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

    def _get_slack_matrix(self, number_slacks: Optional[int, Sequence[int]]) -> Tuple[Optional[np.ndarray], Optional[int]]:

        if number_slacks is None:
            return None, None

        n_ineq = self.G.shape[0]
        if isinstance(number_slacks, int):
            assert number_slacks >= 1, "number_slacks must be a positive integer or a sequence of positive integers."
            n_slacks_total = n_ineq * number_slacks  # total number of slack variables
            coeff_slack = 2 ** np.arange(number_slacks)
            slack_block = np.zeros((n_ineq, n_slacks_total))
            for i in range(n_ineq):
                slack_block[i, i * number_slacks:(i + 1) * number_slacks] = coeff_slack
        elif isinstance(number_slacks, Sequence) or isinstance(number_slacks, np.ndarray):
            assert len(number_slacks) == n_ineq
            n_slacks_total = sum(number_slacks)  # total number of slack variables
            slack_block = np.zeros((n_ineq, n_slacks_total))
            col_offset = 0
            for i, k in enumerate(number_slacks):
                assert isinstance(k,
                                  int), "number_slacks must be a positive integer or a sequence of positive integers."
                assert k >= 1, "number_slacks must be a positive integer or a sequence of positive integers."
                slack_block[i, col_offset: col_offset + k] = [2 ** j for j in range(k)]
                col_offset += k
        else:
            raise TypeError("number_slacks must be a positive integer or a sequence of positive integers.")

        return slack_block, n_slacks_total

    def _compute_default_penalty_factor(self) -> float:
        """
        Computes a default penalty factor based on the size of the quadratic objective function
        """
        return self.Q.max().max() * np.prod(self.Q.shape)

    def to_qubo(self, slack_bits: Union[int, Sequence[int], np.ndarray] = 1) -> np.ndarray:
        """
        Converts a general optimization problem represented by a quadratic objective function Q,
        linear constraints l, and optionally a system of linear equalities A and b, into a QUBO format.

        :param Q: np.ndarray, the quadratic objective function contribution.
        :param l: np.ndarray, the linear objective function contribution.
        :param A: np.ndarray, the equality constraints matrix.
        :param b: np.ndarray, the right-hand side of the equality constraints.
        :param penalty_factor: float, the penalty factor for constraints violations.
        :param slack_bits: int, number of bits per slack variable (default 1).
        :return: np.ndarray, the QUBO representation as a ndarray
        """
        # Data consistency
        qubo = self.Q.copy()
        l_tilde = self.l.copy()

        # Handling equality constraints if any
        if self.A.shape != ():
            qubo += self.penalty_factor * (self.A.T @ self.A)
            l_tilde -= 2 * (self.penalty_factor * self.b.reshape(1, -1) @ self.A).reshape(l_tilde.shape)

        # Handling inequality constraints if any (including slack variables)
        if self.G.shape != ():
            n = self.Q.shape[0]  # number of original variables

            # Extend QUBO matrix to include slack variables
            qubo_extended = np.zeros((n + self.n_total_slacks, n + self.n_total_slacks))
            qubo_extended[:n, :n] = qubo

            G_modified = np.hstack([self.G, self.slack_matrix])
            qubo_extended += self.penalty_factor * (G_modified.T @ G_modified - 2 * np.diag(self.h.T @ G_modified))

            return qubo_extended

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
        qubo = self.to_qubo()
        J, h = self.qubo_to_ising(qubo)
        return J, h

    def qubo_to_ising(self, qubo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        J = qubo
        h = -(J + J.T) @ np.ones(J.shape[0])
        return J, h

    def to_hamiltonian(self) -> SparsePauliOp:
        """
        Converts the optimization problem to a Sparse Pauli Hamiltonian.

        Uses the binary optimization to QUBO conversion, then converts the QUBO
        to a SparsePauliOp Hamiltonian for use in quantum algorithms.

        Returns:
        - SparsePauliOp: The Hamiltonian in the SparsePauliOp format.
        """
        # Convert to QUBO matrix
        qubo = self.to_qubo()

        # Convert QUBO to Sparse Pauli Hamiltonian
        sparse_hamiltonian = self.binary_optimization_to_hamiltonian(qubo)

        return sparse_hamiltonian

    def binary_optimization_to_hamiltonian(self, qubo: np.ndarray) -> SparsePauliOp:
        """
        Converts a QUBO matrix to a Sparse Pauli Hamiltonian.

        Args:
        - qubo: np.ndarray of shape (n, n), the QUBO matrix.

        Returns:
        - SparsePauliOp: The Hamiltonian in SparsePauliOp format.
        """
        n = qubo.shape[0]
        pauli_terms = []

        # For each qubit, generate the corresponding Pauli terms
        for i in range(n):
            for j in range(i, n):
                coeff = qubo[i, j]
                if coeff != 0:  # Skip zero coefficients
                    if i == j:
                        # Diagonal terms are just Z^i
                        pauli_terms.append((coeff, 'I' * i + 'Z' + 'I' * (n - i - 1)))
                    else:
                        # Off-diagonal terms are just Z^i Z^j
                        pauli_terms.append((coeff, 'I' * i + 'Z' + 'I' * (j - i - 1) + 'Z' + 'I' * (n - j - 1)))

        # Create SparsePauliOp from the Pauli terms
        sparse_hamiltonian = SparsePauliOp.from_list(pauli_terms)

        return sparse_hamiltonian
