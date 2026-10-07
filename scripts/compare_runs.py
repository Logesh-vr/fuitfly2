"""Compare recorded experiments without rerunning or altering them."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import virtual_fly  # configure project-local caches
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def main():
    root = Path("outputs")
    config = yaml.safe_load((root / "closed_loop" / "config.yaml").read_text())
    source = np.array(config["body"]["odor_source_mm"][:2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    results = {}
    for name in ("closed_loop", "no_stimulation", "scripted_baseline"):
        run_config = yaml.safe_load((root / name / "config.yaml").read_text())
        if not np.array_equal(source, run_config["body"]["odor_source_mm"][:2]):
            raise ValueError("Comparison runs must use the same odor source")
        frame = pd.read_csv(root / name / "motor_trajectory.csv")
        distance = np.linalg.norm(frame[["x_mm", "y_mm"]].to_numpy() - source, axis=1)
        axes[0].plot(frame.x_mm, frame.y_mm, label=name)
        axes[1].plot(frame.time_s, distance, label=name)
        results[name] = dict(initial_distance_mm=float(distance[0]), final_distance_mm=float(distance[-1]),
                             approach_mm=float(distance[0] - distance[-1]),
                             minimum_distance_mm=float(distance.min()))
    axes[0].scatter(*source, marker="*", s=150, label="Odor source")
    axes[0].set(xlabel="x (mm)", ylabel="y (mm)", title="Trajectories", aspect="equal")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Distance to odor source (mm)", title="Approach comparison")
    rates = pd.read_csv(root / "closed_loop" / "population_rates.csv")
    for name in ("descending_left", "descending_right"):
        axes[2].plot(rates.time_s, rates[name], label=name)
    axes[2].set(xlabel="Simulation time (s)", ylabel="Mean firing rate (Hz)", title="Descending output activity")
    axes[2].legend(fontsize=7)
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    (root / "comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
