from warnings import warn
from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class BaseSolver(ABC):

    def __init__(self, solver: str, options: dict = None):
        self.solver = solver
        self.options = options if options is not None else {}

    @abstractmethod
    def solve(self, J: np.ndarray, h: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def _check_options(self, options: dict = None) -> None:
        pass

    def _update_options(self, kwargs: Union[dict, None]) -> dict:
        # Create a copy of self.options
        new_options = self.options.copy()

        # Check for duplicate keys
        duplicate_keys = set(new_options) & set(kwargs)

        # Raise a warning if there are any duplicate keys
        if duplicate_keys:
            warn(message=f"Overriding options for the following keys: {', '.join(duplicate_keys)}",
                 category=UserWarning,
                 stacklevel=2)

        # Update new_options with kwargs, where kwargs takes precedence
        new_options.update(kwargs)

        # Check options validity
        self._check_options(new_options)

        # Return the updated dictionary with updated options
        return new_options

    def __str__(self):
        return f'{self.__class__.__name__} with solver: {self.solver}'
