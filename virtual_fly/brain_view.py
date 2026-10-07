"""A bounded display sample with real activity/edges and explicitly schematic positions."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def build_brain_geometry(brain, max_neurons=6000, max_edges=10000):
    rng = np.random.default_rng(723)
    selected_groups = {**brain.inputs, **brain.outputs}
    important = np.unique(np.concatenate(list(selected_groups.values())))
    # Include every selected sensory/descending neuron before sampling other neurons.
    if len(important) > max_neurons:
        important = np.sort(rng.choice(important, max_neurons, replace=False))
    other = np.setdiff1d(np.arange(len(brain.ids)), important)
    extra = rng.choice(other, min(max_neurons - len(important), len(other)), replace=False)
    indices = np.sort(np.concatenate([important, extra])).astype(np.int32)
    lookup = np.full(len(brain.ids), -1, dtype=np.int32)
    lookup[indices] = np.arange(len(indices))
    groups = yaml.safe_load(Path(brain.config["groups_file"]).read_text())
    annotations = pd.read_csv(groups["annotations"], dtype={"root_id": str}).set_index("root_id")
    labels = {}
    kinds = {}
    for name, members in selected_groups.items():
        for member in members:
            labels[int(member)] = name
            kinds[int(member)] = "sensory" if name in brain.inputs else "output"
    positions, neurons = [], []
    for index in indices:
        root = str(brain.ids[index])
        row = annotations.loc[root] if root in annotations.index else {}
        side = row.get("side", "unknown")
        cell_class = row.get("class", "unknown")
        super_class = row.get("super_class", "unknown")
        side = str(side) if pd.notna(side) else "unknown"
        cell_class = str(cell_class) if pd.notna(cell_class) else "unknown"
        kind = kinds.get(int(index), "internal")
        sign = -1 if side == "left" else 1 if side == "right" else 0
        if kind == "sensory":
            center, scale = [sign * .75, 1.1, .0], [.55, .38, .5]
        elif kind == "output":
            center, scale = [sign * .6, -.45, -.8], [.42, .5, .3]
        elif super_class == "optic":
            center, scale = [sign * 1.9, .15, .12], [.65, .8, .7]
        else:
            center, scale = [sign * .7, .05, .2], [.85, .9, .75]
        vector = rng.normal(size=3)
        vector /= max(np.linalg.norm(vector), 1e-12)
        point = np.array(center) + vector * rng.random() ** (1 / 3) * np.array(scale)
        positions.extend(np.round(point, 4).tolist())
        neurons.append({"id": root, "group": labels.get(int(index), cell_class),
                        "kind": kind, "class": cell_class, "side": side})
    # Reservoir by random priority across all connections between displayed neurons.
    edge_i = edge_j = np.empty(0, dtype=np.int32)
    weights = priorities = np.empty(0)
    for start in range(0, brain.synapse_count, 500000):
        end = min(start + 500000, brain.synapse_count)
        pre = lookup[np.asarray(brain.synapses.i[start:end], dtype=np.int32)]
        post = lookup[np.asarray(brain.synapses.j[start:end], dtype=np.int32)]
        keep = (pre >= 0) & (post >= 0)
        if not keep.any():
            continue
        edge_i = np.concatenate([edge_i, pre[keep]])
        edge_j = np.concatenate([edge_j, post[keep]])
        weights = np.concatenate([weights, np.asarray(brain.synapses.w_[start:end])[keep] * 1000])
        priorities = np.concatenate([priorities, rng.random(int(keep.sum()))])
        if len(priorities) > max_edges:
            retain = np.argpartition(priorities, max_edges - 1)[:max_edges]
            edge_i, edge_j, weights, priorities = [array[retain] for array in (edge_i, edge_j, weights, priorities)]
    geometry = {
        "layout": "Schematic grouping by annotation and side; NOT anatomical coordinates or neuron morphology.",
        "total_neurons": len(brain.ids), "total_connections": brain.synapse_count,
        "displayed_neurons": len(indices), "displayed_connections": len(edge_i),
        "positions": positions, "neurons": neurons,
        "edges": [[int(i), int(j), round(float(w), 4)] for i, j, w in zip(edge_i, edge_j, weights)],
        "edge_note": "Sampled actual directed model connections; weights in mV. Bright edges mark a spiking source, not measured signal propagation.",
    }
    return indices, geometry
