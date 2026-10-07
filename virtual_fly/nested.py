"""Independent inner physics and a deliberately engineered virtual keyboard."""
from copy import deepcopy
from io import BytesIO
import numpy as np
import mujoco
from PIL import Image
from .body import FlyBody
from .types import MotorCommand, Activity
from .interface import Encoder, Decoder
from .timing import step_count


def keyboard_command(outer, config, connected=True):
    if not connected:
        return "DISCONNECTED", MotorCommand(0., 0.)
    speed = (outer.left + outer.right) / 2
    turn = outer.right - outer.left
    if speed < config["forward_threshold"]:
        return "IDLE", MotorCommand(0., 0.)
    base = config["walking_drive"]
    if abs(turn) < config["turn_threshold"]:
        return "FORWARD", MotorCommand(base, base)
    amount = config["turn_drive"] * np.sign(turn)
    return ("LEFT" if turn > 0 else "RIGHT"), MotorCommand(
        float(np.clip(base - amount, 0, 1.2)), float(np.clip(base + amount, 0, 1.2)))


class InnerSimulation:
    def __init__(self, body_config, config, seed):
        self.config = config
        for key in ("forward_threshold", "turn_threshold", "walking_drive", "turn_drive"):
            value = config[key]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1.2:
                raise ValueError(f"Invalid nested configuration: {key}")
        boost = config["bridge_stimulation_hz"]
        if not isinstance(boost, (int, float)) or not np.isfinite(boost) or not 0 <= boost <= 300:
            raise ValueError("bridge_stimulation_hz must be between 0 and 300")
        settings = deepcopy(body_config)
        settings.update(render=False, vision=False, workstation=False)
        self.body = FlyBody(settings, seed)
        self.renderer = None
        try:
            self.renderer = mujoco.Renderer(self.body.sim.mj_model, height=300, width=500)
        except BaseException:
            self.body.close()
            raise
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 135, -30, 10
        self.key = "IDLE"
        self.drive = MotorCommand(0., 0.)
        self.brain = None
        self.brain_time = 0.
        self.brain_tick = 0
        self.rates, self.stimulation, self.counts = {}, {}, []

    def load_brain(self, config, decoder_config, seed, indices, stop_event):
        from .inner_brain import InnerBrain
        self.encoder = Encoder(deepcopy(config["encoder"]))
        self.decoder = Decoder(deepcopy(decoder_config))
        self.brain = InnerBrain(config["brain"], seed, indices, stop_event)
        self.counts = [0] * len(indices)

    def step(self, outer, dt, connected, paused, controls):
        self.key, _ = keyboard_command(outer, self.config, connected)
        if paused:
            self.key, self.drive = "PAUSED", MotorCommand(0., 0.)
            return
        observation = self.body.observe()
        self.stimulation = self.encoder.encode(observation)
        # The computer now changes sensory stimulation; it never writes leg drives.
        boost = self.config["bridge_stimulation_hz"]
        for name in self.stimulation:
            left = name.endswith("left")
            direct = controls.inner_boost_left if left else controls.inner_boost_right
            bridge = boost if self.key == "FORWARD" or self.key == ("LEFT" if left else "RIGHT") else 0.
            odor = self.stimulation[name] * controls.inner_odor_strength if controls.inner_odor_on else 0.
            self.stimulation[name] = min(odor + direct + bridge, 1000.)
        result = self.brain.step(self.stimulation, dt)
        self.rates, self.counts = result["rates"], result["counts"]
        self.brain_time += dt
        self.brain_tick += 1
        self.drive = self.decoder.decode(Activity(self.rates), dt)
        for _ in range(step_count(dt, self.body.dt)):
            self.body.act(self.drive)

    def snapshot(self):
        observation = self.body.observe()
        return {"time_s": self.body.steps * self.body.dt,
                "position": observation.position_mm.tolist(), "key": self.key,
                "drive": [self.drive.left, self.drive.right],
                "odor": observation.odor.tolist(), "source": self.body.odor_source.tolist(),
                "rates": self.rates, "stimulation": self.stimulation,
                "brain_time_s": self.brain_time, "brain_tick": self.brain_tick,
                "brain_view_counts": self.counts,
                "brain_ready": self.brain is not None}

    def frame(self):
        self.camera.lookat[:] = self.body.observe().position_mm
        self.renderer.update_scene(self.body.sim.mj_data, camera=self.camera)
        buffer = BytesIO()
        Image.fromarray(self.renderer.render()).save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    def close(self):
        if self.brain is not None:
            self.brain.close()
        if self.renderer is not None:
            self.renderer.close()
        self.body.close()
