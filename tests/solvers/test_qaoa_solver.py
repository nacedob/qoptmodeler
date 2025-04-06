import pytest
from src.qoptmodeler.solvers.QAOASolver import QAOASolver
import numpy as np


@pytest.fixture
def solver():
    return QAOASolver(solver='pennylane', n_layers=4)


def test_init(solver):
    assert solver.solver == 'pennylane'
    assert solver.n_layers == 4


def test_init_with_options():
    options = {
        'init_state': np.array([0, 1]),
        'epochs': 10,
        'silent': True
    }
    solver = QAOASolver(solver='pennylane', n_layers=4, options=options)
    assert solver.solver == 'pennylane'
    assert solver.n_layers == 4
    assert solver.options == options


def test_update_options(solver):
    new_options = {
        'epochs': 20,
        'silent': False
    }
    updated_options = solver._update_options(new_options)
    assert updated_options['epochs'] == 20
    assert updated_options['silent'] == False


def test_check_options(solver):
    options = {
        'init_state': np.array([0, 1]),
        'epochs': 10,
        'silent': True
    }
    solver._check_options(options)


def test_check_options_with_invalid_type(solver):
    options = {
        'init_state': 'invalid',
        'epochs': 10,
        'silent': True
    }
    with pytest.raises(AssertionError):
        solver._check_options(options)


def test_check_options_with_unknown_key(solver):
    options = {
        'unknown_key': 'value',
        'epochs': 10,
        'silent': True
    }
    with pytest.raises(AssertionError):
        solver._check_options(options)


if __name__ == '__main__':
    pytest.main(['-V', __file__])
