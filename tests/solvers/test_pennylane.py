import numpy as np
import pytest
from src.qoptmodeler import QAOASolver
from src.qoptmodeler.solvers.utils import int_to_bits
from time import perf_counter
from icecream import ic

np.random.seed(42)


# Define the test data
@pytest.fixture(scope="module")
def easy_problem():
    size = 3
    J = np.random.rand(size, size)
    J = (J + J.T) / 2
    h = np.random.rand(size)
    return (size, J, h)


@pytest.fixture(scope="module")
def hard_problem():
    size = 6
    J = np.random.rand(size, size)
    J = (J + J.T) / 2
    h = np.random.rand(size)
    return (size, J, h)


@pytest.fixture(scope="module")
def solver():
    return QAOASolver(solver='pennylane', n_layers=2)


# Test Pennylane QAOA function
def test_pennylane(easy_problem, solver):
    size, J, h = easy_problem
    solution = solver.solve(J=J, h=h, silent=False, epochs=10)
    ic(solution)

    # Returned solution must be a numpy array with n_qubits elements, each being 0 or 1
    assert isinstance(solution, np.ndarray)
    assert len(solution) == size
    assert all(bit in [0, 1] for bit in solution)


# Test max time constraint
def test_max_time(hard_problem, solver):
    size, J, h = hard_problem
    max_time = 10
    start = perf_counter()
    solution = solver.solve(J=J, h=h, silent=False, epochs=100,
                            stopping_conditions={'max_time': max_time})
    measured_time = perf_counter() - start

    assert abs(measured_time - max_time) < 0.5


# Skip test for visualization circuit
@pytest.mark.skip(reason="For some reason it does not run. You can run it in a terminal and it will run")
def test_visualization_circuit(easy_problem, solver):
    size, J, h = easy_problem
    solver.visualize_circuit(J, h)


# Test limit output
def test_limit_output(hard_problem, solver):
    size, J, h = hard_problem
    possible_ints = [2]
    possible_bits = [int_to_bits(intt, length=J.shape[0]) for intt in possible_ints]
    solution = solver.solve(J=J, h=h, silent=False, epochs=2,
                            possible_result_ints=possible_ints)
    ic(possible_bits, solution)
    assert np.all(solution == possible_bits[0])


if __name__ == '__main__':
    pytest.main(['-V', __file__])
