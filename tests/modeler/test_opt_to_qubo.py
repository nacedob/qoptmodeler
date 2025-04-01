import pytest
from src.modeler.genopt_to_qubo import optimization_problem_to_qubo, _compute_default_penalty_factor
import numpy as np


# --------------------------------
# _compute_default_penalty_factor tests
# --------------------------------
def test_compute_default_penalty_factor():
    size = np.random.randint(1, 100)  # Random size between 1 and 100
    random_constant = np.random.rand() * 100  # Random constant between 0 and 100
    uniform_q = random_constant * np.ones((size, size))
    penalty_factor = _compute_default_penalty_factor(uniform_q)
    assert isinstance(penalty_factor, float)
    assert np.isclose(penalty_factor, random_constant * size ** 2)


# --------------------------------
# optimization_problem_to_qubo tests
# --------------------------------

def generate_random_problem_data(seed: int = None) -> dict:
    if seed is None:
        np.random.seed(seed)

    # Generate random problem parameters
    size = np.random.randint(1, 100)  # Random size between 1 and 100
    n_constraints = np.random.randint(1, 100)  # Random number of constraints between 1 and 100

    # Generate random problem data
    random_q = np.random.rand(size, size)  # Random quadractic term
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
    Test optimization_problem_to_qubo for a full defined problem
    """
    problem_data = generate_random_problem_data()
    # Convert optimization problem to QUBO format and check results
    qubo_q, qubo_l = optimization_problem_to_qubo(Q=problem_data['Q'],
                                                  l=problem_data['l'],
                                                  A=problem_data['A'],
                                                  b=problem_data['b'],
                                                  penalty_factor=problem_data['penalty_factor'])
    # Quadratic term
    assert isinstance(qubo_q, np.ndarray), "Quadratic term is not np.ndarray"
    assert qubo_q.shape == (problem_data['size'], problem_data['size']), "Quadratic term shape is incorrect"

    # Linear term
    assert isinstance(qubo_l, np.ndarray), "Linear term is not np.ndarray"
    assert qubo_l.shape == (problem_data['size'], 1), "Linear term shape is incorrect"


def test_no_penalty_factor_optimization_problem_to_qubo():
    """
    Test optimization_problem_to_qubo for a full defined problem
    """
    problem_data = generate_random_problem_data()
    # Convert optimization problem to QUBO format and check results
    qubo_q, qubo_l = optimization_problem_to_qubo(Q=problem_data['Q'],
                                                  l=problem_data['l'],
                                                  A=problem_data['A'],
                                                  b=problem_data['b'],
                                                  penalty_factor=None)
    default_penalty_factor = _compute_default_penalty_factor(problem_data['Q'])
    expected_qubo_q, expected_qubo_l = optimization_problem_to_qubo(Q=problem_data['Q'],
                                                                    l=problem_data['l'],
                                                                    A=problem_data['A'],
                                                                    b=problem_data['b'],
                                                                    penalty_factor=default_penalty_factor)
    # Quadratic term
    assert np.allclose(qubo_q, expected_qubo_q), "Quadratic term is not as expected"

    # Linear term
    assert np.allclose(qubo_l, expected_qubo_l), "Linear term is not as expected"


def test_no_constraints_optimization_problem_to_qubo():
    """
    Test optimization_problem_to_qubo for a full defined problem
    """
    problem_data = generate_random_problem_data()
    # Convert optimization problem to QUBO format and check results
    qubo_q, qubo_l = optimization_problem_to_qubo(Q=problem_data['Q'],
                                                  l=problem_data['l'],
                                                  A=None,
                                                  b=None,
                                                  penalty_factor=problem_data['penalty_factor'])
    # Quadratic term
    assert np.allclose(qubo_q, problem_data['Q']), "Quadratic term is not as expected"

    # Linear term
    assert np.allclose(qubo_l, problem_data['l']), "Linear term is not as expected"


if __name__ == "__main__":
    pytest.main(['-V', __file__])
