"""Transparent, configurable sensory stimulation and motor decoding."""
import math
import numpy as np
from .types import MotorCommand


class Encoder:
    def __init__(self, config):
        self.config = config

    def encode(self, observation):
        result = {}
        for group, mapping in self.config.items():
            channel = mapping["channel"]
            if channel in ("odor_left", "odor_right"):
                value = observation.odor[0 if channel.endswith("left") else 1]
            elif channel in ("vision_left", "vision_right"):
                if observation.vision is None:
                    raise ValueError("Visual encoder requires body.vision: true")
                value = observation.vision[0 if channel.endswith("left") else 1].mean() / 255.
            elif channel == "contact":
                value = np.linalg.norm(observation.contact, axis=-1).mean()
            elif channel == "proprioception":
                value = np.abs(observation.joints_rad).mean()
            else:
                raise ValueError(f"Unknown sensory channel: {channel}")
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite sensory input: {channel}")
            result[group] = float(np.clip(mapping.get("baseline_hz", 0.) + mapping["gain_hz"] * value,
                                           0, mapping["max_hz"]))
        return result


class Decoder:
    def __init__(self, config):
        self.config = config
        self.filtered = np.zeros(2)
        if config["smoothing_tau_s"] < 0 or not 0 < config["max_drive"] <= 1.5:
            raise ValueError("Invalid decoder smoothing or drive limit")

    def decode(self, activity, dt):
        c = self.config
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        rates = np.array([activity.rates_hz[c["left_group"]], activity.rates_hz[c["right_group"]]])
        if not np.isfinite(rates).all() or np.any(rates < 0):
            raise ValueError("Population rates must be finite and nonnegative")
        alpha = 1. if c["smoothing_tau_s"] == 0 else -math.expm1(-dt / c["smoothing_tau_s"])
        self.filtered += alpha * (rates - self.filtered)
        mean_rate = self.filtered.mean()
        if mean_rate <= c["stand_threshold_hz"]:
            return MotorCommand(0., 0.)
        speed = np.clip(c["speed_gain_per_hz"] * mean_rate, 0, c["max_drive"])
        turn = c["turn_gain_per_hz"] * (self.filtered[0] - self.filtered[1])
        drive = np.clip([speed - turn, speed + turn], 0, c["max_drive"])
        return MotorCommand(float(drive[0]), float(drive[1]))
