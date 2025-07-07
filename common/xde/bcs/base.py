from abc import ABC, abstractmethod
from typing import Any

Array = Any


class BoundaryConditions(ABC):
    """Interface for applying boundary conditions to solution vectors."""

    @abstractmethod
    def add_dirichlet(self, indices: Any, values: Any) -> None:
        """Register Dirichlet conditions (u[i] = value) at given node indices."""

    @abstractmethod
    def add_neumann(self, edges: Any, values: Any) -> None:
        """Register Neumann conditions (flux) on given boundary edges/faces."""

    @abstractmethod
    def apply(self, u: Array) -> Array:
        """Apply all registered BCs to vector u and return a new array."""
