# qoptmodeler 🚀

**qoptmodeler** is a Python package designed to adapt general optimization problems to quantum computing. It converts quadratic cost functions with linear inequality/equality constraints into suitable formats such as **QUBO (Quadratic Unconstrained Binary Optimization)** and **Ising models**, making them ready for quantum solvers ⚛️.

## ✨ Features

- 🧮 **Problem Transformation**: Convert quadratic cost functions and constraints into QUBO or Ising models.
- 💡 **Quantum Compatibility**: Prepare models for execution on quantum hardware or simulators.
- 🔧 **Flexible Input Formats**: Accepts cost functions and constraints as matrices.
- 🛠 **Integration**: Works with quantum computing frameworks for execution.
- 🧑‍💻 **Optimization with Quantum Libraries**: Includes Python code for solving problems using **PennyLane** and **Qiskit**.

## 📦 Installation

You can install qoptmodeler via pip:

```bash
pip install git+https://github.com/nacedob/qoptmodeler.git
```

## 🚀 Usage

### Example

```python
from qoptmodeler import QuantumTranslator
import numpy as np

# Define cost function and constraints as matrices
cost_matrix_quad = np.array([[1, 0.5], [0.5, 2]])
cost_matrix_lin = np.array([[1, 2]])
inequality_constraints = np.array([[-1, 0], [0, -1]])
equality_constraints = np.array([[1, 1]])

# Initialize the converter
converter = QuantumTranslator(cost_matrix_quad, cost_matrix_lin, inequality_constraints, equality_constraints)

# Convert to QUBO format
Q = converter.to_qubo()

# Convert to Ising model
J, h = converter.to_ising()

# Ready for quantum execution ⚡
```

## 🧠 Quantum Optimization with PennyLane & Qiskit

qoptmodeler provides implementations to optimize problems using PennyLane and Qiskit. Examples include:

- **PennyLane-based quantum optimization** ✨
- **Qiskit variational solvers** 🔬

More details and code examples can be found in the `examples/` directory.

## 📂 Code Structure

The repository is organized as follows:
```
qoptmodeler/
│── src/
│   │── QuantumTranslator/   # Code for translating problems to QUBO and ising model
│   │── optimizers/          # Quantum and classical optimization methods
│── tests/                   # Unit tests for different components
```

## ✅ Testing

To ensure correctness, run the test suite using:
```bash
pytest test/
```

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## 🤝 Contributing

Contributions are welcome! 🎉 If you have ideas, improvements, or bug fixes, feel free to fork the repository and submit a pull request.

## 📜 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

---

🚀 Happy Quantum Computing!