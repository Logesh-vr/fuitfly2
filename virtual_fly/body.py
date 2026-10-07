"""Native Windows MuJoCo/FlyGym body with an existing hybrid walking controller."""
from pathlib import Path
import numpy as np
import mujoco
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridTurningController, HybridControllerObservation, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)
from .types import Observation, MotorCommand


class FlyBody:
    def __init__(self, config, seed=7):
        self.config = config
        self.fly = make_locomotion_fly(name="virtual_fly", add_adhesion=True, colorize=True)
        self.vision_enabled = config.get("vision", False)
        if self.vision_enabled:
            self.fly.add_vision()
        camera = self.fly.add_tracking_camera(
            name="side", pos_offset=(-0.5, -7.5, 0),
            rotation=Rotation3D("euler", (1.57, 0, 0)), fovy=35,
        )
        world = FlatGroundWorld()
        if config.get("workstation", False):
            # Visual workstation only: tactile keyboard interaction is a later milestone.
            parts = [
                ("monitor", [6, 0, 2.3], [.2, 2., 1.3], [.09, .13, .18, 1]),
                ("screen", [5.78, 0, 2.3], [.025, 1.8, 1.1], [.15, .48, .4, 1]),
                ("stand", [6, 0, .6], [.3, .35, .6], [.18, .22, .28, 1]),
                ("keyboard", [3.5, 0, .15], [.9, 1.8, .15], [.16, .2, .26, 1]),
                ("tower", [6, 3, 1], [.7, .6, 1], [.12, .16, .22, 1]),
            ]
            for name, position, size, color in parts:
                world.mjcf_root.worldbody.add_geom(
                    name=f"workstation_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=position, size=size, rgba=color, contype=0, conaffinity=0)
        self.odor_source = np.asarray(config["odor_source_mm"], dtype=float)
        self.odor_sigma = float(config["odor_sigma_mm"])
        if self.odor_source.shape != (3,) or not np.isfinite(self.odor_source).all() or self.odor_sigma <= 0:
            raise ValueError("Invalid odor source or spatial spread")
        world.mjcf_root.worldbody.add_geom(
            name="odor_source_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=self.odor_source, size=[.4, 0, 0], rgba=[.2, .85, .25, .8],
            contype=0, conaffinity=0,
        )
        world.add_fly(self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                      bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                      add_ground_contact_sensors=False)
        self.sim = Simulation(world)
        self.dt = float(self.sim.timestep)
        if not np.isclose(self.dt, config["physics_dt_s"], rtol=0, atol=1e-12):
            raise ValueError(f"Body timestep is {self.dt}, requested {config['physics_dt_s']}")
        self.renderer = None
        if config.get("render", True):
            self.renderer = self.sim.set_renderer(
                [camera], camera_res=tuple(config["camera_resolution"]),
                playback_speed=config["playback_speed"], output_fps=config["fps"],
            )
        steps = PreprogrammedSteps()
        dofs = self.fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(
            timestep=self.dt, preprogrammed_steps=steps, output_dof_order=dofs,
        )
        self.sim.reset()
        self.controller.reset(seed=seed)
        apply_locomotion_action(self.sim, self.fly.name, LocomotionAction(
            joint_angles=steps.default_pose_by_dof_order(dofs),
            adhesion_onoff=np.ones(6, dtype=bool),
        ))
        self.sim.warmup()
        self.thorax_idx = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self.antenna_indices = [self.fly.get_bodysegs_order().index(BodySegment(f"{side}_funiculus"))
                                for side in ("l", "r")]
        self.feet = [BodySegment(f"{leg}_tarsus5") for leg in ("lf", "lm", "lh", "rf", "rm", "rh")]
        self.steps = 0

    def observe(self):
        pos = np.array(self.sim.get_body_positions(self.fly.name)[self.thorax_idx], copy=True)
        sensors = self.sim.get_body_positions(self.fly.name)[self.antenna_indices]
        # Analytic static Gaussian field sampled at actual left/right antenna positions.
        odor = np.exp(-np.sum((sensors - self.odor_source) ** 2, axis=1) / (2 * self.odor_sigma ** 2))
        joints = self.sim.get_joint_angles(self.fly.name).copy()
        contacts = self.sim.get_bodysegment_contact_forces(self.fly.name, self.feet).copy()
        vision = self.sim.get_raw_vision(self.fly.name) if self.vision_enabled else None
        return Observation(self.steps * self.dt, pos, odor, joints, contacts, vision)

    def act(self, motor_command: MotorCommand):
        drive = np.array([motor_command.left, motor_command.right], dtype=float)
        if not np.isfinite(drive).all() or np.any(np.abs(drive) > 1.5):
            raise ValueError("Motor drives must be finite and within [-1.5, 1.5]")
        obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
        action = self.controller.step(drive, obs)
        apply_locomotion_action(self.sim, self.fly.name, action)
        self.sim.step_with_profile()
        self.steps += 1

    def move_odor_source(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Odor source must have three finite coordinates")
        self.odor_source = position.copy()
        marker = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "odor_source_marker")
        if marker < 0:
            raise RuntimeError("Odor marker missing from body model")
        self.sim.mj_model.geom_pos[marker] = position
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

    def render(self):
        if self.renderer is not None:
            self.sim.render_as_needed_with_profile()

    def save_video(self, path):
        if self.renderer is not None:
            self.renderer.save_video(Path(path))

    def close(self):
        if self.sim.eye_renderer is not None:
            self.sim.eye_renderer.close()
        if self.renderer is not None:
            close = getattr(self.renderer, "close", None)
            if close:
                close()
