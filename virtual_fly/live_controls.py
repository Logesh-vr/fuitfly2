"""Validated commands shared by the local UI and simulation worker."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Controls:
    paused: bool = True
    mode: str = "brain"
    odor_on: bool = True
    odor_strength: float = 1.
    source_x: float = 16.
    source_y: float = 5.
    boost_left: float = 0.
    boost_right: float = 0.
    manual_speed: float = .8
    manual_turn: float = 0.
    speed_cap: float = 0.
    azimuth: float = 135.
    elevation: float = -25.
    distance: float = 10.
    inner_connected: bool = True
    inner_paused: bool = False
    inner_odor_on: bool = True
    inner_odor_strength: float = 1.
    inner_boost_left: float = 0.
    inner_boost_right: float = 0.
    inner_source_x: float = 16.
    inner_source_y: float = 5.

    def update(self, patch):
        validated = validate_patch(patch)
        for key, value in validated.items():
            setattr(self, key, value)
        return asdict(self)


RANGES = {
    "inner_odor_strength": (0., 3.), "inner_boost_left": (0., 300.), "inner_boost_right": (0., 300.),
    "inner_source_x": (-10000., 10000.), "inner_source_y": (-10000., 10000.),
    "odor_strength": (0., 3.), "source_x": (-10000., 10000.), "source_y": (-10000., 10000.),
    "boost_left": (0., 300.), "boost_right": (0., 300.),
    "manual_speed": (0., 1.2), "manual_turn": (-1., 1.), "speed_cap": (0., 1.),
    "azimuth": (-3600., 3600.), "elevation": (-89., -5.), "distance": (4., 50.),
}


def validate_patch(patch):
    if not isinstance(patch, dict) or not patch:
        raise ValueError("Expected a nonempty control object")
    result = {}
    for key, value in patch.items():
        if key in ("paused", "odor_on", "inner_connected", "inner_paused", "inner_odor_on"):
            if type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
        elif key == "mode":
            if value not in ("brain", "manual"):
                raise ValueError("Mode must be brain or manual")
        elif key in RANGES:
            lo, hi = RANGES[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} must be a finite number between {lo} and {hi}")
        else:
            raise ValueError(f"Unknown control {key}")
        result[key] = value
    return result
