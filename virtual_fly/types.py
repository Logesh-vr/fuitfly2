from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass
class Observation:
    time_s: float
    position_mm: np.ndarray
    odor: np.ndarray  # left, right; normalized concentration
    joints_rad: np.ndarray
    contact: np.ndarray
    vision: np.ndarray | None = None


@dataclass
class Activity:
    rates_hz: dict[str, float]
    spike_times_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    spike_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


@dataclass(frozen=True)
class MotorCommand:
    left: float
    right: float


class Brain(Protocol):
    def step(self, sensory_input: dict[str, float], dt: float) -> Activity: ...


class Body(Protocol):
    def observe(self) -> Observation: ...
    def act(self, motor_command: MotorCommand) -> None: ...
