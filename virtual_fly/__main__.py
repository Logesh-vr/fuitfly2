import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml
from .timing import Pacer, step_count
from .types import MotorCommand


def body_run(args, config):
    from .body import FlyBody
    from .reporting import save_motor_report
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    body = FlyBody(config["body"], config["seed"])
    command = MotorCommand(**config["scripted"])
    duration = args.duration if args.duration is not None else config["duration_s"]
    n = step_count(duration, body.dt)
    pacer = Pacer(body.dt, args.realtime)
    initial = body.observe().position_mm
    rows = [[0., *initial, command.left, command.right]]
    try:
        for i in range(n):
            body.act(command)
            body.render()
            if (i + 1) % 100 == 0 or i + 1 == n:
                obs = body.observe()
                rows.append([obs.time_s, *obs.position_mm, command.left, command.right])
            pacer.tick()
        elapsed = time.perf_counter() - pacer.start
        displacement = float(np.linalg.norm(body.observe().position_mm[:2] - initial[:2]))
        summary = dict(milestone="body", duration_s=duration, wall_s=elapsed,
                       realtime_factor=duration / elapsed, displacement_mm=displacement,
                       odor_source_mm=config["body"]["odor_source_mm"],
                       missed_deadlines=pacer.missed_deadlines, max_lag_s=pacer.max_lag_s)
        body.save_video(output / "walking.mp4")
        save_motor_report(output, rows, summary)
        (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if not np.isfinite(np.asarray(rows)).all() or displacement < 0.1:
            raise RuntimeError("Body verification failed: invalid values or no walking displacement")
    finally:
        body.close()


def brain_run(args, config):
    from .brain import ConnectomeBrain
    from .reporting import save_brain_report
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    brain = ConnectomeBrain(config["brain"], config["seed"])
    dt = config["control_dt_s"]
    n = step_count(args.duration if args.duration is not None else config["duration_s"], dt)
    activities, times = [], []
    pacer = Pacer(dt, args.realtime)
    for i in range(n):
        t = i * dt
        stim = config["stimulation"]
        rates = stim["rates_hz"] if stim["onset_s"] <= t < stim["offset_s"] else {}
        activity = brain.step(rates, dt)
        activities.append(activity)
        times.append((i + 1) * dt)
        pacer.tick()
        if (i + 1) % 5 == 0:
            print(f"Brain {i + 1}/{n}: {activity.rates_hz}", flush=True)
    elapsed = time.perf_counter() - pacer.start
    save_brain_report(output, times, activities, brain.selected_groups())
    summary = dict(milestone="brain", neurons=len(brain.ids), connections=brain.synapse_count,
                   group_sizes=brain.group_sizes, duration_s=n * dt, wall_s=elapsed,
                   realtime_factor=n * dt / elapsed,
                   missed_deadlines=pacer.missed_deadlines, max_lag_s=pacer.max_lag_s,
                   total_network_spikes=int(brain.monitor.num_spikes))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    driven_groups = [name for name, rate in config["stimulation"]["rates_hz"].items() if rate > 0]
    if driven_groups and n * dt > config["stimulation"]["onset_s"]:
        if not any(a.rates_hz[name] > 0 for a in activities for name in driven_groups):
            raise RuntimeError("Stimulation did not activate the selected sensory groups")


def closed_loop_run(args, config):
    from .brain import ConnectomeBrain
    from .body import FlyBody
    from .interface import Encoder, Decoder
    from .loop import run_loop
    from .reporting import save_motor_report, save_brain_report
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.no_stimulation:
        for mapping in config["encoder"].values():
            mapping["gain_hz"] = 0.
            mapping["baseline_hz"] = 0.
    brain = ConnectomeBrain(config["brain"], config["seed"])
    decoder_config = yaml.safe_load(Path(config["decoder_file"]).read_text(encoding="utf-8"))
    encoder, decoder = Encoder(config["encoder"]), Decoder(decoder_config)
    body = FlyBody(config["body"], config["seed"])
    initial = body.observe().position_mm
    rows = [[0., *initial, 0., 0.]]
    times, activities, sensory_rows = [], [], []
    source = np.asarray(config["body"]["odor_source_mm"])
    duration = args.duration if args.duration is not None else config["duration_s"]
    stimulus_names = list(config["encoder"])
    def record(t, before, stim, activity, command, after):
        rows.append([t, *after.position_mm, command.left, command.right])
        times.append(t)
        activities.append(activity)
        sensory_rows.append([before.time_s, *before.odor, *[stim[name] for name in stimulus_names]])
        if len(times) % 10 == 0:
            print(f"Loop {t:.2f}/{duration:.2f}s; odor={before.odor.round(3)}; "
                  f"drive=({command.left:.2f}, {command.right:.2f})", flush=True)
    try:
        pacer = run_loop(body, brain, encoder, decoder, duration, config["control_dt_s"],
                         args.realtime, record)
        elapsed = time.perf_counter() - pacer.start
        positions = np.asarray(rows)[:, 1:4]
        distances = np.linalg.norm(positions[:, :2] - source[:2], axis=1)
        summary = dict(milestone="closed_loop", no_stimulation=args.no_stimulation,
                       odor_source_mm=config["body"]["odor_source_mm"],
                       duration_s=duration, wall_s=elapsed, realtime_factor=duration / elapsed,
                       neurons=len(brain.ids), connections=brain.synapse_count,
                       initial_distance_mm=float(distances[0]), final_distance_mm=float(distances[-1]),
                       minimum_distance_mm=float(distances.min()),
                       approach_success=bool(distances[0] - distances[-1] >= config["task"]["approach_improvement_mm"]),
                       arrival_success=bool(distances.min() <= config["task"]["arrival_radius_mm"]),
                       missed_deadlines=pacer.missed_deadlines, max_lag_s=pacer.max_lag_s)
        body.save_video(output / "closed_loop.mp4")
        save_motor_report(output, rows, summary)
        save_brain_report(output, times, activities, brain.selected_groups())
        np.savetxt(output / "sensory_stimulation.csv", sensory_rows, delimiter=",",
                   header="sense_time_s,odor_left,odor_right," + ",".join(stimulus_names), comments="")
        (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (output / "decoder.yaml").write_text(yaml.safe_dump(decoder_config), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if not np.isfinite(np.asarray(rows)).all():
            raise RuntimeError("Nonfinite body state in closed loop")
    finally:
        body.close()


def main():
    parser = argparse.ArgumentParser(description="Virtual fruit fly milestone runner")
    parser.add_argument("mode", choices=["body", "brain", "closed_loop", "live"])
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no-stimulation", action="store_true", help="Closed-loop ablation with zero sensory drive")
    parser.add_argument("--port", type=int, default=8765, help="Local interactive control panel port")
    parser.add_argument("--no-browser", action="store_true", help="Start live server without opening a browser")
    args = parser.parse_args()
    args.config = args.config or f"configs/{'closed_loop' if args.mode == 'live' else args.mode}.yaml"
    args.output = args.output or f"outputs/{args.mode}"
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "live":
        from .live import live_run
        live_run(args, config)
        return
    {"body": body_run, "brain": brain_run, "closed_loop": closed_loop_run}[args.mode](args, config)


if __name__ == "__main__":
    main()
