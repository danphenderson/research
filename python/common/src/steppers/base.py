from abc import ABC, abstractmethod


class TimeStepper(ABC):
    """Abstract time integration method (explicit)."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Return the order of accuracy of the method."""

    @abstractmethod
    def step(self, operator: Any, u: Any, t: float, dt: float) -> Any:
        """Advance solution `u` at time `t` by one step of size `dt`."""
