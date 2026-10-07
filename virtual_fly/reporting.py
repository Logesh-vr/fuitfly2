from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_motor_report(out, rows, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    values = np.asarray(rows)
    np.savetxt(out / "motor_trajectory.csv", values, delimiter=",",
               header="time_s,x_mm,y_mm,z_mm,left_drive,right_drive", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(values[:, 1], values[:, 2])
    axes[0].scatter(values[[0, -1], 1], values[[0, -1], 2], c=["green", "red"])
    if "odor_source_mm" in metadata:
        source = metadata["odor_source_mm"]
        axes[0].scatter([source[0]], [source[1]], marker="*", s=100, label="Odor source")
        axes[0].legend()
    axes[0].set(xlabel="x (mm)", ylabel="y (mm)", title="Fly trajectory", aspect="equal")
    axes[1].plot(values[:, 0], values[:, 4], label="Left")
    axes[1].plot(values[:, 0], values[:, 5], label="Right")
    axes[1].set(xlabel="Time (s)", ylabel="Dimensionless motor drive", title="Motor commands")
    axes[1].legend()
    fig.savefig(out / "trajectory_motor.png", dpi=160)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_brain_report(out, times, activities, group_ids):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(activities[0].rates_hz)
    rates = np.array([[a.rates_hz[name] for name in names] for a in activities])
    np.savetxt(out / "population_rates.csv", np.column_stack([times, rates]), delimiter=",",
               header="time_s," + ",".join(names), comments="")
    spike_times = np.concatenate([a.spike_times_s for a in activities])
    spike_ids = np.concatenate([a.spike_ids for a in activities])
    # Preserve 64-bit root IDs as decimal strings; never convert through float.
    with (out / "selected_spikes.csv").open("w", encoding="utf-8") as handle:
        handle.write("time_s,root_id\n")
        for t, root in zip(spike_times, spike_ids):
            handle.write(f"{t:.8f},{int(root)}\n")
    (out / "selected_groups.json").write_text(json.dumps(group_ids, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for i, name in enumerate(names):
        axes[0].plot(times, rates[:, i], label=name)
    axes[0].set(ylabel="Mean rate per neuron (Hz)", title="Population activity")
    axes[0].legend(fontsize=8)
    unique, indices = np.unique(spike_ids, return_inverse=True)
    axes[1].scatter(spike_times, indices, s=3)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Selected neuron index",
                title=f"Selected-neuron raster ({len(unique)} active neurons)")
    fig.savefig(out / "brain_activity.png", dpi=160)
    plt.close(fig)
