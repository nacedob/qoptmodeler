from typing import Sequence
from .BaseSolver import BaseSolver
from math import isclose
from warnings import warn
import pennylane as qml
from time import perf_counter
from pennylane import numpy as qnp
import numpy as np
from scipy.sparse import spmatrix
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp


class QAOASolver(BaseSolver):
    solver_list = ['pennylane', 'qiskit']

    def __init__(self,
                 solver: str = 'pennylane',
                 n_layers: int = 4,
                 options: dict = None):
        """
        Quantum Approximate Optimization Algorithm.

        The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm
        that uses a classical optimizer to optimize the parameters of a quantum circuit. The circuit is
        iteratively run with different parameters until the optimal solution is found.

        Parameters
        ----------
        solver : str, default='pennylane'
            The solvers to use. Can be 'pennylane' or 'qiskit'.

        n_layers : int, default=4
            The number of layers in the QAOA ansatz.

        options : dict, optional
            Additional solvers-specific options. List of options:

            - init_state : np.ndarray - Initial state of the system.
            - transition_matrix_mixer : np.ndarray - Transition matrix mixer.
            - epochs : int - Number of epochs.
            - silent : bool - Whether to suppress output.
            - sample : bool - Whether to sample the output.
            - probs : bool - Whether to compute probabilities.
            - shots : int - Number of shots.
            - optimizer : str - Optimizer to use.
            - lr : float - Learning rate.
            - beta1 : float - Beta 1 value.
            - beta2 : float - Beta 2 value.
            - decay : float - Decay rate.
            - momentum : float - Momentum value.
            - stopping_conditions : dict - Stopping conditions.
            - possible_result_ints : list[int] - Possible result integers.
        """
        # Assertions
        assert solver in self.solver_list, f'Invalid solvers: {solver}. Available solvers: {self.solver_list}'
        assert isinstance(n_layers, int), f'n_layers should be integer. Got {n_layers}.'
        assert n_layers >= 1, f'n_layers should be >= 1. Got {n_layers}.'

        # Initialize base solvers
        super(QAOASolver, self).__init__(solver, options)

        # Set solvers-specific attributes
        self.n_layers = n_layers

    def _check_options(self, options: dict = None) -> None:
        expected_options_dict = {
            'init_state': np.ndarray,
            'transition_matrix_mixer': np.ndarray,
            'epochs': int,
            'silent': bool,
            'sample': bool,
            'shots': int,
            'optimizer': str,
            'optimizer_options': dict,
            'stopping_conditions': dict,
            'possible_result_ints': Sequence,
        }

        options_ = options or self.options

        # Check if there are unexpected keys in options
        unexpected_keys = set(options_.keys()) - set(expected_options_dict.keys())
        assert not unexpected_keys, (
            f'Unexpected options: {unexpected_keys}. Expected options: {expected_options_dict.keys()}'
        )

        # Check the types of the expected options
        for key, value in options_.items():
            assert isinstance(value, expected_options_dict[key]), (
                f'Invalid value for option {key}. Expected type: {expected_options_dict[key]}, got {type(value)}.'
            )

    def solve(self, J: np.ndarray, h: np.ndarray, **kwargs):
        if self.solver == 'pennylane':
            return self._solve_pennylane(J, h, **kwargs)
        else:
            raise ValueError(f"Invalid solvers: {self.solver}")

    def _solve_pennylane(self, J: np.ndarray, h: np.ndarray, **kwargs) -> np.ndarray:

        options = self._update_options(kwargs)
        solution = self.qaoa_pennylane(J=J, h=h, solver='cpu', n_layers=self.n_layers, **options)
        return solution

    def _create_cost_operator_pennylane(self,
                                        J: np.ndarray, h: np.ndarray, params: qml.typing.TensorLike, n_qubits: int):
        # single-qubit terms
        for i, hi in enumerate(h):
            if hi != 0:
                qml.RZ(2 * params * hi, wires=i)
        # two-qubit terms
        for i in range(n_qubits):
            for j in range(i):
                qml.MultiRZ(2 * params * J[i, j], wires=[i, j])

    def _create_mixer_operator_pennylane(self,
                                         params, n_qubits, transition_matrix_mixer=None):
        if transition_matrix_mixer in [None, 'x']:
            for i in range(n_qubits):
                qml.RX(2 * params, wires=i)
        else:
            assert isinstance(transition_matrix_mixer,
                              (np.ndarray, spmatrix)), f'Invalid mixer: {transition_matrix_mixer}'
            transition_matrix_mixer_ = transition_matrix_mixer.toarray() if isinstance(transition_matrix_mixer,
                                                                                       spmatrix) \
                else transition_matrix_mixer
            transition_matrix_mixer_ = qnp.array(transition_matrix_mixer_, requires_grad=False)
            mixer_matrix = qnp.exp(-1j * params * transition_matrix_mixer_)
            qml.QubitUnitary(mixer_matrix, wires=range(n_qubits))

    def _qaoa_circuit_pennylane(self,
                                params_cost: qnp.ndarray,
                                params_mixer: qnp.ndarray,
                                J: np.ndarray,
                                h: np.ndarray,
                                n_qubits: int,
                                init_state: qml.typing.TensorLike = None,
                                transition_matrix_mixer: np.ndarray = None,
                                draw=False):
        # Apply the initial layer of Hadamard gates to all qubits
        if init_state is not None:
            qml.StatePrep(init_state, range(n_qubits))
        else:
            for q in range(n_qubits):
                qml.Hadamard(wires=q)

        # Repeat p layers of the cost and mixing operators
        for layer in range(len(params_cost)):
            self._create_cost_operator_pennylane(J, h, params_cost[layer], n_qubits)
            self._create_mixer_operator_pennylane(params_mixer[layer], n_qubits, transition_matrix_mixer)

        if draw:
            qml.drawer.use_style("pennylane")
            fig, ax = qml.draw_mpl(self._qaoa_circuit_pennylane)(params_cost, params_mixer, J, h, n_qubits)
            plt.show()

    def _cost_function_qaoa_pennylane(self,
                                      params: qnp.ndarray,
                                      n_qubits: int,
                                      J: np.ndarray,
                                      h: np.ndarray,
                                      init_state: qml.typing.TensorLike = None,
                                      transition_matrix_mixer: np.ndarray = None,
                                      sample: bool = False,
                                      probs: bool = False,
                                      draw=False):
        cost_params = params[0]
        mixer_params = params[1]
        self._qaoa_circuit_pennylane(cost_params, mixer_params, J, h, n_qubits,
                                     draw=draw, init_state=init_state, transition_matrix_mixer=transition_matrix_mixer)

        if sample:
            return qml.sample()
        elif probs:
            return qml.probs(wires=range(n_qubits))
        else:
            # We define the Ising Hamiltonian (that defines our optimization problem)
            EIz = qml.Hamiltonian(h, [qml.PauliZ(i) for i in range(n_qubits)])
            EIzz = qml.Hamiltonian([J[i, j] for i in range(n_qubits) for j in range(i)],
                                   [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(n_qubits) for j in range(i)])
            # And measure its expectation value
            return qml.expval(EIzz + EIz)

    def _get_optimizer(self, optimizer: str, optimizer_options: dict):

        if optimizer == 'adam':
            opt = qml.AdamOptimizer(stepsize=optimizer_options.get('lr') or 0.01,
                                    beta1=optimizer_options.get('beta1') or 0.9,
                                    beta2=optimizer_options.get('beta2') or 0.99)
        elif optimizer == 'gd':
            opt = qml.GradientDescentOptimizer(stepsize=optimizer_options.get('lr') or 0.01)
        elif optimizer == 'rms':
            opt = qml.RMSPropOptimizer(stepsize=optimizer_options.get('lr') or 0.01,
                                       decay=optimizer_options.get('decay') or 0.9)
        elif optimizer == 'momentum':
            opt = qml.MomentumOptimizer(stepsize=optimizer_options.get('lr') or 0.01,
                                        momentum=optimizer_options.get('momentum') or 0.9)
        elif optimizer == 'nesterov':
            opt = qml.NesterovMomentumOptimizer(stepsize=optimizer_options.get('lr') or 0.01,
                                                momentum=optimizer_options.get('momentum') or 0.9)
        else:
            raise ValueError(f"Optimizer {optimizer} not recognized. Choose between 'adam', 'gd', 'rms' or 'momentum'")
        return opt

    def qaoa_pennylane(self,
                       J: np.ndarray,
                       h: np.ndarray,
                       n_layers: int,
                       solver: str = 'cpu',
                       init_state: np.ndarray = None,
                       transition_matrix_mixer: np.ndarray = None,
                       epochs: int = 10,
                       silent: bool = True,
                       sample: bool = False,
                       probs: bool = False,
                       shots: int = 1000,
                       optimizer: str = 'adam',
                       lr: float = None,
                       beta1: float = None,
                       beta2: float = None,
                       decay: float = None,
                       momentum: float = None,
                       stopping_conditions: dict = None,
                       possible_result_ints: list[int] = None
                       ) -> np.ndarray:
        """
        Quantum Approximate Optimization Algorithm (QAOA) implementation using PennyLane.

        :param J: (np.ndarray) Quadratic coefficients of the QUBO problem.
        :param h: (np.ndarray) Linear coefficients of the QUBO problem.
        :param n_layers: (int) Number of QAOA layers.
        :param solver: (str) Solver type, only 'cpu' is supported for now.
        :param init_state: (np.ndarray, optional) Initial state for the quantum circuit.
        :param transition_matrix_mixer: (np.ndarray, optional) Transition matrix for the mixer Hamiltonian.
        :param epochs: (int) Number of optimization epochs.
        :param silent: (bool) If True, suppresses output during optimization.
        :param sample: (bool) If True, returns samples instead of expectation values.
        :param probs: (bool) If True, returns probabilities instead of expectation values.
        :param shots: (int) Number of shots for sampling.
        :param optimizer: (str) Optimizer type, e.g., 'adam', 'gd', 'rms', 'momentum'.
        :param lr: (float, optional) Learning rate for the optimizer.
        :param beta1: (float, optional) Beta1 parameter for the Adam optimizer.
        :param beta2: (float, optional) Beta2 parameter for the Adam optimizer.
        :param decay: (float, optional) Decay parameter for the RMSprop optimizer.
        :param momentum: (float, optional) Momentum parameter for the Momentum optimizer.
        :param stopping_conditions: (dict, optional) Stopping conditions for the optimization. Supported keys are:
                                                     'max_time', 'patience', 'patience_tolerance'.
        :param possible_result_ints: (list[int], optional) Limits the output to a given list of results, in integer format.
        :return: Solution bits (np.ndarray).
        """
        if solver != 'cpu':
            raise NotImplementedError("Only CPU solvers is implemented for PennyLane")

        # Process stopping conditions ------------------------
        max_time = patience = patience_tolerance = None  # init all to None
        if stopping_conditions is not None:
            if 'max_time' in stopping_conditions:
                max_time = stopping_conditions['max_time']
            if 'patience' in stopping_conditions:
                patience = int(stopping_conditions['patience'])
                if 'patience_tolerance' in stopping_conditions:
                    patience_tolerance = stopping_conditions['patience_tolerance']
                else:
                    patience_tolerance = 1e-6

        patience_counter = 0
        best_cost = float('inf')

        # Init optimization ------------------------
        n_qubits = J.shape[0]
        qnode = qml.QNode(self._cost_function_qaoa_pennylane,
                          qml.device("default.qubit", shots=shots))
        qnode_fixed_args = {'n_qubits': n_qubits, 'J': J, 'h': h, 'init_state': init_state,
                            'transition_matrix_mixer': transition_matrix_mixer, 'sample': sample, 'probs': probs,
                            'draw': False}

        opt = self._get_optimizer(optimizer,
                                  optimizer_options={'lr': lr, 'beta1': beta1, 'beta2': beta2, 'decay': decay,
                                                     'momentum': momentum})

        # Initialize parameters ------------------------
        params = qnp.random.rand(2, n_layers, requires_grad=True)

        # Start training ------------------------
        start_time = perf_counter()
        for it in range(epochs):
            from icecream import ic
            x = opt.step(qnode, params, **qnode_fixed_args)
            if patience is not None or (not silent and it % max(1, epochs // 5) == 0):
                current_cost = qnode(x, **qnode_fixed_args)

            # Stopping conditions
            if max_time is not None:
                if perf_counter() - start_time > max_time:
                    warn('Max time reached. Stopping optimization.', stacklevel=2)
                    break
            if patience is not None:
                if isclose(current_cost, best_cost, abs_tol=patience_tolerance):
                    best_cost = current_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter == patience:
                        warn('Patience reached. Stopping optimization.', stacklevel=2)
                        break
            if not silent and it % (epochs // 5 or 1) == 0:
                print(f"[INFO] Iter: {it + 1:3d} | Energy: {current_cost:0.6f}")
        else:
            # This is exectuced just in case the loop ends without breaking
            if not silent and epochs % max(1, epochs // 5) != 0:  # then current cost is not up-to-date
                current_cost = qnode(x, **qnode_fixed_args)
                print(f"[INFO] Optimization finished with value {current_cost:0.6f}.")

        # Get most probable (with no shots, so prob is exact) among possible list of ints ------------------------
        qnode_fixed_args_ = qnode_fixed_args.copy()
        del qnode_fixed_args_['probs']
        qnode = qml.QNode(self._cost_function_qaoa_pennylane, qml.device("default.qubit"))
        probs = qnode(params, probs=True, **qnode_fixed_args_)

        if possible_result_ints:
            # Limit output to possible results
            probs = probs[possible_result_ints]
            max_index = np.argmax(probs)
            correct_int = possible_result_ints[max_index]
        else:
            correct_int = np.argmax(probs)

        # Convert to binary format ------------------------
        binary_rep = f"{correct_int:0{n_qubits}b}"
        solution = np.array([int(bit) for bit in binary_rep])

        return solution

    def visualize_circuit(self, J: np.ndarray, h: np.ndarray,
                          params_cost: Sequence = None, params_mixer: Sequence = None, init_state: np.ndarray = None,
                          transition_matrix_mixer: np.ndarray = None) -> None:
        if params_cost is None:
            params_cost = qnp.random.rand(self.n_layers)
        if params_mixer is None:
            params_mixer = qnp.random.rand(self.n_layers)

        self._qaoa_circuit_pennylane(params_cost=params_cost, params_mixer=params_mixer, J=J, h=h, n_qubits=J.shape[0],
                                     init_state=init_state, transition_matrix_mixer=transition_matrix_mixer, draw=True)
