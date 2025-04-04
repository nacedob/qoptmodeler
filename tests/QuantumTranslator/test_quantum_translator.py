import pytest
from src.QuantumTranslator import QuantumTranslator
import numpy as np


# --------------------------------
# _compute_default_penalty_factor tests
# --------------------------------
def test_compute_default_penalty_factor():
    size = np.random.randint(1, 100)  # Random size between 1 and 100
    random_constant = np.random.rand() * 100  # Random constant between 0 and 100
    uniform_q = random_constant * np.ones((size, size))
    translator = QuantumTranslator(uniform_q, np.zeros(size))
    penalty_factor = translator._compute_default_penalty_factor()
    assert isinstance(penalty_factor, float)
    assert np.isclose(penalty_factor, random_constant * size ** 2)


# --------------------------------
# to_qubo tests
# --------------------------------

def generate_random_problem_data(seed: int = None) -> dict:
    if seed is None:
        np.random.seed(seed)

    # Generate random problem parameters
    size = 4  # np.random.randint(1, 100)  # Random size between 1 and 100
    n_constraints = np.random.randint(1, 100)  # Random number of constraints between 1 and 100

    # Generate random problem data
    random_q = np.random.rand(size, size)  # Random quadratic term
    random_l = np.random.rand(size, 1)  # Random linear term
    random_a = np.random.rand(n_constraints, size)  # Random equality constraints matrix
    random_b = np.random.rand(n_constraints, 1)  # Random equality constraints vector
    random_penalty_factor = np.random.rand() * 100  # Random penalty factor between 0 and 100

    return {
        'Q': random_q,
        'l': random_l,
        'A': random_a,
        'b': random_b,
        'penalty_factor': random_penalty_factor,
        'size': size,
        'n_constraints': n_constraints
    }


def test_full_optimization_problem_to_qubo():
    """
    Test to_qubo for a full defined problem
    """
    problem_data = generate_random_problem_data()
    qtranslator = QuantumTranslator(quad_cost_matrix=problem_data['Q'],
                                    lin_cost_matrix=problem_data['l'],
                                    lhs_eq_matrix=problem_data['A'],
                                    rhs_eq_vector=problem_data['b'],
                                    penalty_factor=problem_data['penalty_factor'])

    # Convert optimization problem to QUBO format and check results
    qubo = qtranslator.to_qubo()

    assert isinstance(qubo, np.ndarray), "Qubo is not np.ndarray"
    assert qubo.shape == (problem_data['size'], problem_data['size']), "Quadratic term shape is incorrect"


def test_no_penalty_factor_optimization_problem_to_qubo():
    """
    Test to_qubo for a problem where the penalty factor is not provided
    """
    problem_data = generate_random_problem_data()
    qtranslator = QuantumTranslator(quad_cost_matrix=problem_data['Q'],
                                    lin_cost_matrix=problem_data['l'],
                                    lhs_eq_matrix=problem_data['A'],
                                    rhs_eq_vector=problem_data['b'],
                                    penalty_factor=None)
    actual_qubo = qtranslator.to_qubo()
    default_penalty_factor = qtranslator._compute_default_penalty_factor()
    qtranslator_def_penalty = QuantumTranslator(quad_cost_matrix=problem_data['Q'],
                                                lin_cost_matrix=problem_data['l'],
                                                lhs_eq_matrix=problem_data['A'],
                                                rhs_eq_vector=problem_data['b'],
                                                penalty_factor=default_penalty_factor)
    expected_qubo = qtranslator_def_penalty.to_qubo()
    assert np.allclose(actual_qubo, expected_qubo), "Qubo is not as expected"


def test_no_constraints_optimization_problem_to_qubo():
    """
    Test to_qubo for a problem without constraints
    """
    problem_data = generate_random_problem_data()
    # Convert optimization problem to QUBO format and check results
    qtranslator = QuantumTranslator(quad_cost_matrix=problem_data['Q'],
                                    lin_cost_matrix=problem_data['l'],
                                    lhs_eq_matrix=None,
                                    rhs_eq_vector=None,
                                    penalty_factor=problem_data['penalty_factor'])
    qubo = qtranslator.to_qubo()

    expected_qubo = problem_data['Q'] + np.diag(problem_data['l'])
    assert np.allclose(qubo, expected_qubo), "Qubo is not as expected"


# --------------------------------
# to_ising tests
# --------------------------------
def test_full_to_ising():
    """
    Test to_ising for a full defined problem
    """
    problem_data = generate_random_problem_data()
    # Convert optimization problem to QUBO format and check results
    qutranslator = QuantumTranslator(quad_cost_matrix=problem_data['Q'],
                                     lin_cost_matrix=problem_data['l'],
                                     lhs_eq_matrix=problem_data['A'],
                                     rhs_eq_vector=problem_data['b'],
                                     penalty_factor=problem_data['penalty_factor'])
    ising_q, ising_l = qutranslator.to_ising()
    qubo = qutranslator.to_qubo()
    expected_ising_l = - (qubo.sum(axis=0).flatten() + qubo.sum(axis=1).flatten())

    # Quadratic term
    assert isinstance(ising_q, np.ndarray), "Quadratic term is not np.ndarray"
    assert ising_q.shape == (problem_data['size'], problem_data['size']), "Quadratic term shape is incorrect"
    assert np.allclose(qubo, ising_q), "Quadratic term shape is incorrect"

    # Linear term
    assert isinstance(ising_l, np.ndarray), "Linear term is not np.ndarray"
    assert ising_l.shape == (problem_data['size'],), "Linear term shape is incorrect"
    assert np.allclose(ising_l, expected_ising_l), "Linear term shape is incorrect"


if __name__ == "__main__":
    pytest.main(['-V', __file__])
