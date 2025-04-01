from icecream import ic
import numpy as np
from typing import Tuple
from types import NoneType

def qubo_to_ising(Q: np.ndarray, l: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:

    # Data consistency
    assert Q.shape[0] == Q.shape[1], "Input matrix Q must be square."
    if l is not None and l.shape[0] != Q.shape.shape[0]:
        raise AssertionError("Input vector l must have the same shape as Q.")

    # Ising transformationç
    raise NotImplementedError("Implement the Ising transformation")