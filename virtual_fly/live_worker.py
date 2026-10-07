"""Simulation in a separate process; browser controls never wait for Brian2."""
import csv
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import queue
import time
import traceback
import numpy as np
import yaml
from .live_controls import Controls
from .types import MotorCommand
from .timing import step_count


def simulation_worker(config, command_queue, result_queue, stop_event, output):
    body = renderer = brain = inner = None
    controls = Controls(source_x=config["body"]["odor_source_mm"][0],
                        source_y=config["body"]["odor_source_mm"][1])
    seq, tick, brain_time, factor, brain_tick = 0, 0, 0., 0., 0
    brain_activity = []
    brain_view_indices = np.empty(0, dtype=np.int32)
    rates, stimulation = {}, {}
    command = MotorCommand(0., 0.)
    state = {"status": "loading", "message": "Building the fly body...", "controls": asdict(controls)}

    def publish(frame=None):
        # Bounded queue drops stale frames; it never accumulates video in memory.
        packet = {"state": dict(state), "frame": frame,
                  "inner_frame": inner.frame() if inner is not None and frame is not None else None}
        try:
            result_queue.put_nowait(packet)
        except queue.Full:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                result_queue.put_nowait(packet)
            except queue.Full:
                pass

    try:
        publish()
        from .body import FlyBody
        from .interface import Encoder, Decoder
        import mujoco
        from PIL import Image
        config["body"]["render"] = False
        config["body"]["vision"] = False
        nested_config = config.get("nested", {})
        config["body"]["workstation"] = nested_config.get("enabled", False)
        body = FlyBody(config["body"], config["seed"])
        if nested_config.get("enabled", False):
            from .nested import InnerSimulation
            inner = InnerSimulation(config["body"], nested_config, config["seed"] + 1)
        renderer = mujoco.Renderer(body.sim.mj_model, height=540, width=900)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        decoder_config = yaml.safe_load(Path(config["decoder_file"]).read_text())
        (output / "config.yaml").write_text(yaml.safe_dump(config))
        (output / "decoder.yaml").write_text(yaml.safe_dump(decoder_config))

        def refresh(status, message):
            observation = body.observe()
            camera.lookat[:] = observation.position_mm
            camera.azimuth, camera.elevation, camera.distance = controls.azimuth, controls.elevation, controls.distance
            renderer.update_scene(body.sim.mj_data, camera=camera)
            frame = renderer.render()
            buffer = BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
            quat = body.sim.get_body_rotations(body.fly.name)[body.thorax_idx]
            w, x, y, z = quat
            heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z)))
            state.update(status=status, message=message, controls=asdict(controls), applied_seq=seq,
                         tick=tick, sim_time_s=body.steps * body.dt, brain_time_s=brain_time,
                         position=observation.position_mm.tolist(), heading=heading,
                         source=body.odor_source.tolist(), odor=observation.odor.tolist(),
                         rates=rates, stimulation=stimulation, drive=[command.left, command.right],
                         brain_view_counts=brain_activity, brain_tick=brain_tick,
                         brain_window_s=config["control_dt_s"],
                         realtime_factor=factor, log_directory=str(output.resolve()))
            state["inner"] = inner.snapshot() if inner is not None else None
            publish(buffer.getvalue())
            return observation

        refresh("loading", "Loading the full connectome. Controls will apply when ready...")
        from .brain import ConnectomeBrain
        config["brain"]["record_spikes"] = False
        brain = ConnectomeBrain(config["brain"], config["seed"])
        refresh("loading", "Preparing the 3D brain activity view...")
        from .brain_view import build_brain_geometry
        brain_view_indices, geometry = build_brain_geometry(brain)
        geometry_tmp = output / "brain_geometry.tmp"
        geometry_tmp.write_text(json.dumps(geometry, separators=(",", ":")), encoding="utf-8")
        geometry_tmp.replace(output / "brain_geometry.json")
        brain_activity = [0] * len(brain_view_indices)
        if inner is not None:
            refresh("loading", "Loading the inner fly's independent full connectome brain...")
            inner.load_brain(config, decoder_config, config["seed"] + 1, brain_view_indices, stop_event)
        (output / "config.yaml").write_text(yaml.safe_dump(config))
        (output / "selected_groups.json").write_text(json.dumps(brain.selected_groups()))
        encoder, decoder = Encoder(config["encoder"]), Decoder(decoder_config)
        dt = config["control_dt_s"]
        n_substeps = step_count(dt, body.dt)
        refresh("paused", "Ready. Press Run, or Step to advance 10 ms.")
        with (output / "live.csv").open("w", newline="", encoding="utf-8") as log, \
             (output / "nested.jsonl").open("w", encoding="utf-8") as nested_log, \
             (output / "events.jsonl").open("w", encoding="utf-8") as events:
            writer = csv.writer(log)
            writer.writerow(["tick", "body_time_s", "brain_time_s", "mode", "x_mm", "y_mm", "z_mm",
                             "source_x_mm", "source_y_mm", "odor_left", "odor_right", "stim_left_hz", "stim_right_hz",
                             "descending_left_hz", "descending_right_hz", "left_drive", "right_drive", "step_wall_s"])
            one_step = False
            next_due = 0.
            last_refresh = 0.
            while not stop_event.is_set():
                changed = False
                while True:
                    try:
                        request = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    seq = request["seq"]
                    if request.get("action") == "step":
                        controls.paused, one_step = True, True
                    else:
                        controls.update(request["patch"])
                    events.write(json.dumps({"wall_time": time.time(), "body_time_s": body.steps * body.dt,
                                             "request": request}) + "\n")
                    events.flush()
                    changed = True
                if changed:
                    body.move_odor_source([controls.source_x, controls.source_y, config["body"]["odor_source_mm"][2]])
                    if inner is not None:
                        inner.body.move_odor_source([controls.inner_source_x, controls.inner_source_y,
                                                    config["body"]["odor_source_mm"][2]])
                if controls.paused and not one_step:
                    if changed or time.perf_counter() - last_refresh > 1:
                        refresh("paused", "Paused. Move the source, orbit the camera, or press Run.")
                        last_refresh = time.perf_counter()
                    stop_event.wait(.025)
                    continue
                if time.perf_counter() < next_due and not one_step:
                    if changed:
                        refresh("running", "Pacing to the selected speed cap.")
                    stop_event.wait(.025)
                    continue
                started = time.perf_counter()
                observation = body.observe()
                if controls.mode == "brain":
                    state.update(status="computing", message="Computing the next brain step; changes apply at its boundary.",
                                 controls=asdict(controls), applied_seq=seq)
                    publish()
                    stimulation = encoder.encode(observation)
                    for name in stimulation:
                        value = stimulation[name] * controls.odor_strength if controls.odor_on else 0.
                        boost = controls.boost_left if name.endswith("left") else controls.boost_right
                        stimulation[name] = min(value + boost, 1000.)
                    activity = brain.step(stimulation, dt)
                    brain_activity = brain.last_spike_counts[brain_view_indices].tolist()
                    brain_tick += 1
                    rates = activity.rates_hz
                    brain_time += dt
                    command = decoder.decode(activity, dt)
                else:
                    # Explicit alternative control mode: brain is frozen and NOT driving the body.
                    stimulation, rates = {}, {}
                    drive = np.clip([controls.manual_speed - controls.manual_turn,
                                     controls.manual_speed + controls.manual_turn], 0., 1.2)
                    command = MotorCommand(float(drive[0]), float(drive[1]))
                for _ in range(n_substeps):
                    body.act(command)
                if inner is not None:
                    state.update(status="computing", message="Computing the inner fly's independent brain step.")
                    publish()
                    inner.step(command, dt, controls.inner_connected, controls.inner_paused, controls)
                    nested_log.write(json.dumps({"outer_tick": tick + 1,
                        "outer_time_s": body.steps * body.dt, "outer_mode": controls.mode,
                        "outer_drive": [command.left, command.right],
                        "connected": controls.inner_connected, "paused": controls.inner_paused,
                        **inner.snapshot()}) + "\n")
                    nested_log.flush()
                tick += 1
                elapsed = time.perf_counter() - started
                factor = dt / max(elapsed, 1e-9)
                after = refresh("paused" if controls.paused else "running",
                                "Brain controls walking." if controls.mode == "brain" else "Manual walking: brain is frozen and bypassed.")
                writer.writerow([tick, body.steps * body.dt, brain_time, controls.mode, *after.position_mm,
                                 controls.source_x, controls.source_y, *observation.odor,
                                 stimulation.get("olfactory_left", 0.), stimulation.get("olfactory_right", 0.),
                                 rates.get("descending_left", 0.), rates.get("descending_right", 0.),
                                 command.left, command.right, elapsed])
                log.flush()
                one_step = False
                next_due = started + dt / controls.speed_cap if controls.speed_cap else 0.
        state.update(status="stopped", message="Session ended. Logs have been saved.")
        publish()
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        state.update(status="error", message=error)
        publish()
    finally:
        if inner is not None:
            inner.close()
        if renderer is not None:
            renderer.close()
        if body is not None:
            body.close()
