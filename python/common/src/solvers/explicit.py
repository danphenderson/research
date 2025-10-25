from abc import ABC, abstractmethod
from typing import Any


class Solver(ABC):
    """Facade that runs the PDE solve (assemble, stepping, BC)."""

    @abstractmethod
    def solve(self, initial_state: Any, t0: float, t_final: float, dt: float) -> Any:
        """Run time integration from t0 to t_final and return final solution."""

    @abstractmethod
    def get_solution(self) -> Any:
        """Return the current solution vector."""
