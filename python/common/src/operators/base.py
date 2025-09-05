from abc import ABC, abstractmethod
from typing import Any

Array = Any


class PDEOperator(ABC):
    """Abstract spatial PDE operator (e.g. FD stencil, FEM/DG assembly)."""

    @property
    @abstractmethod
    def mesh(self) -> Any:
        """Mesh associated with this operator."""

    @property
    @abstractmethod
    def backend(self) -> Any:
        """Array backend (NumPy, Torch, etc.) used by this operator."""

    @abstractmethod
    def assemble(self) -> None:
        """Perform any one-time assembly (build matrices or operators) prior to time-stepping."""

    @abstractmethod
    def evaluate(self, u: Array, t: float) -> Array:
        """Compute the RHS (time-derivative) for state `u` at time `t`."""
