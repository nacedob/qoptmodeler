import os
from math import isclose
from time import perf_counter
from typing import Sequence, Optional, Union
from warnings import warn, filterwarnings, resetwarnings

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from pennylane.typing import TensorLike
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, EstimatorOptions, SamplerOptions
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeFez
from scipy.optimize import minimize
from .BaseSolver import BaseSolver
from .utils import MaxTimeWarning, to_bitstring, invert_integer_binary_representation


class MinimizeStopper(object):
    """
    A class to stop the optimization when the maximum time is reached.
    Used for qiskit optimization with scipy interface
    """

    def __init__(self, max_sec: float):
        self.max_sec = max_sec
        self.start = perf_counter()

    def __call__(self, xk=None):
        if perf_counter() - self.start > self.max_sec:
            warn('max_time is attained', category=MaxTimeWarning, stacklevel=2)
            global optimal_params_global
            optimal_params_global = xk
            raise TimeoutError


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
        if self.solver == 'pennylane':
            expected_options_dict = {'init_state': np.ndarray, 'epochs': int, 'silent': bool, 'sample': bool,
                                     'shots': int, 'optimizer': str, 'optimizer_options': dict,
                                     'stopping_conditions': dict, 'possible_result_ints': (list, tuple, set)}
        elif self.solver == 'qiskit':
            expected_options_dict = {'backend': str, 'backend_training': str, 'noise': bool, 'init_state': np.ndarray,
                                     'epochs': (np.integer, int), 'shots': int,
                                     'tolerance': float, 'max_time': (int, np.integer, np.floating, float),
                                     'max_time_on_qpu': (int, np.integer, np.floating, float), 'silent': bool,
                                     'save_intermediate_cost_function': bool, 'possible_result_ints': list,
                                     'error_correction_options': dict, 'save_job_id_path': str,
                                     'save_job_id_prefix': str, 'wait': bool, 'init_params': list, 'token': str,
                                     'optimization_level': int}
        else:
            raise ValueError(f"Invalid solvers: {self.solver}")

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
        options = self._update_options(kwargs)
        if self.solver == 'pennylane':
            return self.qaoa_pennylane(J=J, h=h, solver='cpu', **options)
        elif self.solver == 'qiskit':
            return self.qaoa_qiskit(J=J, h=h, **options)
        else:
            raise ValueError(f"Invalid solvers: {self.solver}")

    # ----------------------------------------------------
    # QAOA with pennylane
    # ----------------------------------------------------
    def qaoa_pennylane(self,
                       J: np.ndarray,
                       h: np.ndarray,
                       solver: str = 'cpu',
                       init_state: np.ndarray = None,
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
        :param solver: (str) Solver type, only 'cpu' is supported for now.
        :param init_state: (np.ndarray, optional) Initial state for the quantum circuit.
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
        qnode_fixed_args = {'n_qubits': n_qubits, 'J': J, 'h': h, 'init_state': init_state, 'sample': sample,
                            'probs': probs,
                            'draw': False}

        opt = self._get_optimizer_pennylane(optimizer,
                                            optimizer_options={'lr': lr, 'beta1': beta1, 'beta2': beta2, 'decay': decay,
                                                               'momentum': momentum})

        # Initialize parameters ------------------------
        params = qnp.random.rand(2, self.n_layers, requires_grad=True)

        # Start training ------------------------
        start_time = perf_counter()
        for it in range(epochs):
            x = opt.step(qnode, params, **qnode_fixed_args)
            if patience is not None or (not silent and it % max(1, epochs // 5) == 0):
                current_cost = qnode(x, **qnode_fixed_args)

            # Stopping conditions
            if max_time is not None:
                if perf_counter() - start_time > max_time:
                    warn('Max time reached. Stopping optimization.', category=MaxTimeWarning, stacklevel=2)
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

    def _create_cost_operator_pennylane(self,
                                        J: np.ndarray,
                                        h: np.ndarray,
                                        params: TensorLike,
                                        n_qubits: int):
        # single-qubit terms
        for i, hi in enumerate(h):
            if hi != 0:
                qml.RZ(2 * params * hi, wires=i)
        # two-qubit terms
        for i in range(n_qubits):
            for j in range(i):
                qml.MultiRZ(2 * params * J[i, j], wires=[i, j])

    def _create_mixer_operator_pennylane(self, params: TensorLike, n_qubits: int) -> None:
        for i in range(n_qubits):
            qml.RX(2 * params, wires=i)

    def _qaoa_circuit_pennylane(self,
                                params_cost: qnp.ndarray,
                                params_mixer: qnp.ndarray,
                                J: np.ndarray,
                                h: np.ndarray,
                                n_qubits: int,
                                init_state: qml.typing.TensorLike = None,
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
            self._create_mixer_operator_pennylane(params_mixer[layer], n_qubits)

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
                                      sample: bool = False,
                                      probs: bool = False,
                                      draw=False):
        cost_params = params[0]
        mixer_params = params[1]
        self._qaoa_circuit_pennylane(cost_params, mixer_params, J, h, n_qubits,
                                     draw=draw, init_state=init_state)

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

    def _get_optimizer_pennylane(self, optimizer: str, optimizer_options: dict):

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

    # ----------------------------------------------------
    # QAOA with Qiskit
    # ----------------------------------------------------

    def qaoa_qiskit(self,
                    J: np.ndarray,
                    h: np.ndarray,
                    backend: str = 'cpu',
                    backend_training: str = None,
                    noise: bool = False,
                    init_state: [np.ndarray, QuantumCircuit] = None,
                    epochs: Optional[int] = None,
                    shots: int = 1000,
                    tolerance=None,
                    max_time: float = None,
                    max_time_on_qpu: float = None,
                    silent: bool = True,
                    save_intermediate_cost_function: bool = False,
                    possible_result_ints: list[int] = None,
                    error_correction_options: dict = None,
                    save_job_id_path: str = None,
                    save_job_id_prefix: str = None,
                    wait: bool = True,
                    init_params: list = None,
                    token: str = None,
                    optimization_level: int = 3,
                    ) -> Union[np.ndarray, str]:
        """
        Perform the Quantum Approximate Optimization Algorithm (QAOA) using IBM's Qiskit.

        Parameters:
            J (np.ndarray): Quadratic coefficients of the cost function.
            h (np.ndarray): Linear coefficients of the cost function.
            n_layers (int): Number of QAOA layers (p-parameter).
            backend (str, optional): Optimization backend type, e.g., 'cpu', 'gpu', or 'ibmq'. Defaults to 'cpu'.
            noise (bool, optional): Whether to use a noisy quantum simulator. Defaults to False.
            backend_training (str, optional): Backend or training backend for optimization, if different from `backend`.
                                             Defaults to None.
            init_state (Union[np.ndarray, QuantumCircuit], optional): Initial quantum state, either as a state vector
                                                            (np.ndarray) or as a Qiskit QuantumCircuit. Defaults to None.
            epochs (int, optional): Number of optimization epochs. Defaults to 10.
            shots (int, optional): Number of shots for sampling. Defaults to 1000.
            tolerance (float, optional): Convergence tolerance for the optimizer. Defaults to None.
            max_time (float, optional): Maximum allowed time for the training process. Defaults to None.
            max_time_on_qpu (float, optional): Maximum allowed time that a process can be running on a real qpu, either for
                                                evaluation or training. Defaults to None.
            silent (bool, optional): If True, suppress optimizer progress output. Defaults to True.
            save_intermediate_cost_function (bool, optional): Whether to save intermediate cost function values during
                                                              optimization. Defaults to False.
            possible_result_ints (list[int], optional): Limits the output to a specific set of valid results,
                                                        given as integers.
                Note: these integers correspond to bitstrings where the bit order matches the correct physical qubit order
                      (little-endian).
            error_correction_options (dict, optional): Options for applying post-processing error correction.
                                                       Defaults to None.
            save_job_id_path (str, optional): Path where the job ID (if applicable, e.g., for IBMQ runs) will be saved.
                                              Defaults to None.
            save_job_id_prefix (str, optional): Prefix to prepend to the saved job ID filename. Defaults to None.
            wait (bool, optional): If True, wait for quantum jobs to complete (only relevant for cloud execution).
                                    Defaults to True.
            init_params (list, optional): Initial guess for QAOA parameters (angles). Defaults to None.
            token (str, optional): Token to load the IBM account. If None, it tries to load a presaved environment variable
                                   `IBM_TOKEN` or a presaved QiskitRuntimeService. Defaults to None.
            optimization_level (int, optional): Optimization level for the optimizer [0, 1, 2, 3]. Defaults to 3.

        Returns:
            Union[Tuple[np.ndarray, float], str]:
            If wait = True
                - np.ndarray: The optimized solution state (bitstring probabilities).
                - float: The final cost function value.
            If using a remote backend and wait = True, it returns job ID (str) instead.

        """

        if backend_training is not None and 'fake' in backend.lower() and 'fake' not in backend_training.lower():
            raise ValueError("Cannot train on real backend and execute on simulator.")

        if backend_training == 'fake':
            backend_training = backend.replace('ibm_', 'fake_')

        filterwarnings("ignore", category=DeprecationWarning)
        filterwarnings("ignore", category=UserWarning)
        filterwarnings("ignore", category=PendingDeprecationWarning)

        n_qubits = h.shape[0]
        service = self._get_ibmruntimeservice(token)

        # error correction errors
        default_error_correction = {
            'zne': True,
            'twirling': True,
            'dynamical_decoupling': True,
            'trex': True,
            'shots': shots,
            'max_time_on_qpu': max_time_on_qpu,
        }
        if error_correction_options is None:
            error_correction_options = {}
        for k, v in default_error_correction.items():
            if k not in error_correction_options:
                error_correction_options[k] = v

        cost_hamiltonian = self._create_cost_operator_qiskit(n_qubits, J, h)

        qc = self._create_qaoa_circuit_qiskit(cost_hamiltonian=cost_hamiltonian,
                                              layers=self.n_layers,
                                              measure=False,  # Otherwise transpiler fails
                                              init_state=init_state)
        # Get backend
        if backend_training is None:
            backend_training = backend
        backend_training_ = self._get_backend_qiskit(backend=backend_training, noise=noise, service=service,
                                                     qubits=J.shape[0])
        backend_execution = self._get_backend_qiskit(backend=backend, noise=noise, service=service, qubits=J.shape[0])
        qc = self._transpile_circuit_qiskit(qc,
                                            backend_execution,
                                            optimization_level)  # transpile in the final circuit to avoid future compabilities problems
        qc.measure_all()

        # Initialize parameters
        if init_params is None:
            initial_gamma = np.pi
            initial_beta = np.pi / 2
            init_params = np.tile([initial_gamma, initial_beta], self.n_layers)
        assert len(init_params) == 2 * self.n_layers, 'Init_params length must be 2 * n_layers'
        if save_intermediate_cost_function:
            objective_func_vals = []  # global variable

        # Get optimal paramters
        session = Session(backend=backend_training_)
        estimator = self._get_estimator_qiskit(session=session, backend=backend_training, **error_correction_options)

        start_time = perf_counter()
        if max_time is not None:
            def wrapped_cost_func(*args):
                elapsed_time = perf_counter() - start_time
                if elapsed_time > max_time:
                    raise TimeoutError(f"Optimization exceeded max_time of {max_time} seconds.")
                return self.cost_func_estimator_qiskit(*args)
        else:
            wrapped_cost_func = self.cost_func_estimator_qiskit
        global optimal_params_global
        global counter
        optimal_params_global = None
        counter = 0
        try:
            result = minimize(
                wrapped_cost_func,
                init_params,
                args=(qc, cost_hamiltonian, estimator, save_intermediate_cost_function,
                      save_job_id_path, save_job_id_prefix),
                method="COBYLA",
                tol=tolerance,
                callback=MinimizeStopper(max_time) if max_time is not None else None,
                options={'maxiter': epochs} if epochs is not None else None,
                bounds=((0, 2 * np.pi),) * 2 * self.n_layers,
            )
            optimal_params_global = result.x

        except TimeoutError:
            assert optimal_params_global is not None, \
                'Some problem happened during optimization and result was not stored before ' \
                'max_time was reached'
        finally:
            session.close()

        if not silent:
            print('Optimization completed. Results:')
            print(optimal_params_global)
            print(f'Optimization time: {round(perf_counter() - start_time, 3)} seconds.')

        # Execute the optimized circuit
        qc = self._transpile_circuit_qiskit(qc, backend_execution, optimization_level)
        optimized_circuit = qc.assign_parameters(optimal_params_global)

        sampler = self._get_sampler_qiskit(backend=backend_execution, **error_correction_options)
        pub = (optimized_circuit,)
        job = sampler.run([pub], shots=int(shots))

        if not wait:
            return job.job_id()

        solution = self._solution_from_job_qiskit(job=job, n_qubits=n_qubits, possible_result_ints=possible_result_ints)
        resetwarnings()
        return solution

    def _solution_from_job_qiskit(self,
                                  job,
                                  n_qubits: int,
                                  possible_result_ints: list[int] = None) -> np.ndarray:
        int_counts = job.result()[0].data.meas.get_int_counts()
        shots = len(int_counts)
        final_distribution_int = {key: val / shots for key, val in int_counts.items()}
        if possible_result_ints is not None:
            final_distribution_int = {k: v
                                      for k, v in final_distribution_int.items()
                                      if invert_integer_binary_representation(k, num_bits=n_qubits)
                                      in possible_result_ints}
        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())
        most_likely = keys[np.argmax(np.abs(values))]
        solution = to_bitstring(most_likely, n_qubits)
        solution.reverse()
        return np.array(solution)

    def get_solution_from_job_id_qiskit(self,
                                        job_id: str,
                                        n_qubits: int,
                                        service: Optional[QiskitRuntimeService] = None,
                                        possible_result_ints: list[int] = None,
                                        token: str = None) -> np.ndarray:
        if service is None:
            service = self._get_ibmruntimeservice(token)
        job = service.job(job_id)
        solution = self._solution_from_job_qiskit(job=job, n_qubits=n_qubits, possible_result_ints=possible_result_ints)
        return solution

    def _get_ibmruntimeservice(self, token: str = None) -> QiskitRuntimeService:
        IBM_TOKEN = token or os.environ.get('IBM_TOKEN')

        if IBM_TOKEN is None:
            warn('No IBM_TOKEN environment variable found. Trying to load a presaved QuantumRuntimeService.',
                 stacklevel=2)
            return QiskitRuntimeService()

        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='ibm-q-lantik/quantum-mads/benchmarking',
            token=IBM_TOKEN
        )
        return service

    def _create_cost_operator_qiskit(self, n_qubits: int, J: np.ndarray, h: np.ndarray) -> SparsePauliOp:
        pauli_list = []
        for i in range(n_qubits):
            if h[i] != 0:
                pauli_h = ["I"] * n_qubits
                pauli_h[i] = "Z"
                pauli_list.append(("".join(pauli_h)[::-1], h[i]))
            for j in range(i):
                if J[i, j] != 0:
                    pauli_J = ["I"] * n_qubits
                    pauli_J[i], pauli_J[j] = 'Z', 'Z'
                    pauli_list.append(("".join(pauli_J)[::-1], J[i, j]))
        return SparsePauliOp.from_list(pauli_list)


    def _create_qaoa_circuit_qiskit(self,
                                    cost_hamiltonian: SparsePauliOp,
                                    layers: int = 1,
                                    measure: bool = True,
                                    init_state: np.ndarray = None) -> QuantumCircuit:
        if init_state is not None:
            if isinstance(init_state, np.ndarray):
                n_qubits = int(np.log2(init_state.shape[0]))
                init_state_qc = QuantumCircuit(n_qubits)
                init_state_qc.initialize(init_state, qubits=range(n_qubits))
            elif isinstance(init_state, QuantumCircuit):
                init_state_qc = init_state
            else:
                raise ValueError("Invalid init_state. It should be either a numpy array or a QuantumCircuit."
                                 " Got %s" % type(init_state))
        else:
            init_state_qc = None
        qc = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=layers, initial_state=init_state_qc)
        if measure:
            qc.measure_all()
        return qc

    def _get_backend_qiskit(self,
                            backend: str,
                            noise: bool,
                            service: Optional[QiskitRuntimeService] = None,
                            qubits: int = None,
                            token: str = None) -> BackendV2:
        if 'fake' not in backend and backend != 'cpu':
            assert backend in ('ibm_marrakesh', 'ibm_fez', 'ibm_aachen', 'ibm_kingston'), \
                (f'Invalid backend {backend}. Available: "ibm_marrakesh", "ibm_fez", "ibm_aachen", "ibm_kingston", '
                 f'"fake_marrakesh", "fake_fez", "fake_aachen", "fake_kingston"')
            if service is None:
                service = self._get_ibmruntimeservice(token)
            return service.backend(backend)

        elif 'fez' in backend:  # and 'fake' in backend:
            if noise:
                return FakeFez()
            else:
                return AerSimulator(configuration=FakeFez().configuration())
        elif 'marrakesh' in backend:  # and 'fake' in backend:
            if noise:
                return FakeMarrakesh()
            else:
                return AerSimulator(configuration=FakeMarrakesh().configuration())
        elif 'kingston' in backend or 'aachen' in backend:  # and 'fake' in backend:
            warn('IBM has not yet implemented "fake_kingston" and "fake_aachen". Using "fake_marrakesh" instead',
                 stacklevel=2)
            if noise:
                return FakeMarrakesh()
            else:
                return AerSimulator(configuration=FakeMarrakesh().configuration())
        assert backend == 'cpu', (
            'Invalid backend. Available: "ibm_marrakesh", "ibm_fez", "ibm_aachen", "ibm_kingston", '
            '"fake_marrakesh", "fake_fez", "fake_aachen", "fake_kingston", "cpu"'
        )
        if qubits < 30:
            return AerSimulator()
        return AerSimulator(configuration=FakeMarrakesh().configuration())

    def _get_estimator_qiskit(self,
                              session, backend: str,
                              shots: int, max_time_on_qpu: float = None,
                              zne: bool = True,
                              dynamical_decoupling: bool = True,
                              twirling: bool = True,
                              trex: bool = True) -> Estimator:
        options_estimator = self._get_error_options_qiskit(estimator=True,
                                                           backend=backend,
                                                           shots=shots,
                                                           max_time_on_qpu=max_time_on_qpu,
                                                           zne=zne,
                                                           dynamical_decoupling=dynamical_decoupling,
                                                           twirling=twirling,
                                                           trex=trex)
        estimator = Estimator(mode=session, options=options_estimator)
        return estimator

    def _get_sampler_qiskit(self,
                            backend,
                            shots: int, max_time_on_qpu: float = None,
                            zne: bool = True,
                            dynamical_decoupling: bool = True,
                            twirling: bool = True,
                            trex: bool = True) -> Sampler:
        solver = 'cpu' if backend.configuration().simulator else 'qpu'
        options_sampler = self._get_error_options_qiskit(sampler=True,
                                                         backend=solver,
                                                         shots=shots,
                                                         max_time_on_qpu=max_time_on_qpu,
                                                         zne=zne,
                                                         dynamical_decoupling=dynamical_decoupling,
                                                         twirling=twirling,
                                                         trex=trex)
        sampler = Sampler(mode=backend, options=options_sampler)
        return sampler

    def _transpile_circuit_qiskit(self, circuit: QuantumCircuit, back: BackendV2,
                                  optimization_level: int) -> QuantumCircuit:
        # Transpile circuit to backend
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=back)
        circuit = pm.run(circuit)
        return circuit

    def cost_func_estimator_qiskit(self,
                                   params: np.ndarray,
                                   circuit: QuantumCircuit,
                                   hamiltonian: SparsePauliOp,
                                   estimator: Estimator,
                                   save_intermediate_cost_function: bool,
                                   save_job_id_path: str = None,
                                   save_job_id_prefix: str = None):
        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
        circuit_ = circuit.assign_parameters(params, inplace=False)
        pub = (circuit_, isa_hamiltonian)
        job = estimator.run([pub])
        global counter
        self._save_job_id_qiskit(job.job_id(), save_job_id_path, save_job_id_prefix, counter)
        counter += 1

        results = job.result()[0]
        cost = results.data.evs

        if save_intermediate_cost_function:
            global objective_func_vals
            objective_func_vals.append(cost)

        return cost

    def _save_job_id_qiskit(self, job_id: str, save_job_id_path: str, save_job_id_prefix: str,
                            counter_: [str, int]) -> None:
        if save_job_id_path:
            # Check path existance
            if not os.path.exists(save_job_id_path):
                os.makedirs(os.path.dirname(save_job_id_path), exist_ok=True)
                # If it does not exist, create it
                with open(save_job_id_path, "w") as f:
                    pass
            # Create text to be appended to the txt
            text = save_job_id_prefix if save_job_id_prefix else ""
            text += f' Counter: {counter_} // Job ID: {job_id}'
            # Append the text to the file
            with open(save_job_id_path, 'a') as f:
                f.write(text + '\n')

    def _get_error_options_qiskit(self,
                                  estimator: bool = False,
                                  backend: str = 'cpu',
                                  sampler: bool = False,
                                  shots: int = None,
                                  max_time_on_qpu: float = None,
                                  zne: bool = True,
                                  dynamical_decoupling: bool = True,
                                  twirling: bool = True,
                                  trex: bool = True):
        if not estimator and not sampler:
            raise ValueError("At least one error correction method (estimator or sampler) must be enabled.")
        elif estimator:
            options = EstimatorOptions()
            # ZNE - Suppress gate noise
            if zne:
                options.resilience.zne_mitigation = True
                options.resilience.zne.noise_factors = (1, 3, 5)  # optional as default is (1, 3, 5)
                options.resilience.zne.extrapolator = "linear"
            # TREX - error for measurement
            if trex:
                options.resilience_level = 0
                options.resilience.measure_mitigation = True  # optional as default is `True`
                options.resilience.measure_noise_learning.num_randomizations = 32  # optional
        else:
            options = SamplerOptions()
        # COMMON OPTIONS FOR SAMPLER AND ESTIMATOR
        # Shots
        if shots:
            options.default_shots = shots
        else:
            raise ValueError('No shots specified')
        # Max execution time
        if backend != 'cpu' and max_time_on_qpu is not None:
            options.max_execution_time = int(max_time_on_qpu)
        # Dynamical Decoupling - suppress decoherence errors during idle times
        if dynamical_decoupling:
            options.dynamical_decoupling.enable = True
            options.dynamical_decoupling.sequence_type = "XY4"
        # Twirling - error for coherent errors
        if twirling:
            options.twirling.enable_gates = True
            options.twirling.num_randomizations = "auto"

        # estimator.options.dynamical_decoupling.enable = True
        # estimator.options.dynamical_decoupling.sequence_type = "XY4"
        # estimator.options.twirling.enable_gates = True
        # estimator.options.twirling.num_randomizations = "auto"
        return options

    def visualize_circuit(self, J: np.ndarray, h: np.ndarray,
                          params_cost: Sequence = None, params_mixer: Sequence = None, init_state: np.ndarray = None) -> None:
        if self.solver == 'pennylane':
            if params_cost is None:
                params_cost = qnp.random.rand(self.n_layers)
            if params_mixer is None:
                params_mixer = qnp.random.rand(self.n_layers)

            self._qaoa_circuit_pennylane(params_cost=params_cost, params_mixer=params_mixer, J=J, h=h,
                                         n_qubits=J.shape[0], init_state=init_state, draw=True)
        elif self.solver == 'qiskit':
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
