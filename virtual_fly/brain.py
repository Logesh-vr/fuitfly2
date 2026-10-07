"""Persistent Brian2 adapter around the published Shiu network constructor."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import brian2 as b2
from .types import Activity
from .timing import step_count


def select_groups(specs, annotations, ids):
    lookup = {str(value): i for i, value in enumerate(ids)}
    result = {}
    for name, selector in specs.items():
        if "ids" in selector:
            chosen = [str(v) for v in selector["ids"]]
            missing = set(chosen) - lookup.keys()
            if missing:
                raise ValueError(f"{name}: IDs absent from model: {sorted(missing)[:5]}")
        else:
            mask = np.ones(len(annotations), dtype=bool)
            for key, value in selector.items():
                if key not in annotations:
                    raise ValueError(f"Unknown annotation column {key}")
                mask &= annotations[key].eq(value).fillna(False).to_numpy(dtype=bool)
            chosen = annotations.loc[mask, "root_id"].astype(str).tolist()
            chosen = [value for value in chosen if value in lookup]
        if not chosen or len(set(chosen)) != len(chosen):
            raise ValueError(f"{name}: selection is empty or contains duplicate IDs")
        result[name] = np.array([lookup[value] for value in chosen], dtype=np.int32)
    return result


class ConnectomeBrain:
    def __init__(self, config, seed=7):
        self.config = config
        groups = yaml.safe_load(Path(config["groups_file"]).read_text(encoding="utf-8"))
        if groups["release"] != 783:
            raise ValueError("This dataset adapter is verified for materialization 783 only")
        for key in ("completeness", "connectivity", "annotations", "model_source"):
            if not Path(groups[key]).is_file():
                raise FileNotFoundError(f"Missing {groups[key]}; run scripts/download_data.py")
        manifest_path = Path("data_manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("completeness", "connectivity", "annotations", "model_source"):
                path = groups[key]
                if path in manifest and hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest[path]["sha256"]:
                    raise ValueError(f"Dataset/source checksum mismatch: {path}")
        b2.prefs.codegen.target = config.get("codegen", "numpy")
        b2.defaultclock.dt = config["neural_dt_s"] * b2.second
        b2.seed(seed)
        self.ids = pd.read_csv(groups["completeness"], index_col=0).index.to_numpy(dtype=np.int64)
        annotations = pd.read_csv(groups["annotations"], dtype={"root_id": str})
        self.inputs = select_groups(groups["inputs"], annotations, self.ids)
        self.outputs = select_groups(groups["outputs"], annotations, self.ids)
        if set(self.inputs) & set(self.outputs):
            raise ValueError("Input and output group names must be distinct")
        all_inputs = np.concatenate(list(self.inputs.values()))
        if len(np.unique(all_inputs)) != len(all_inputs):
            raise ValueError("Input groups must not overlap")
        spec = importlib.util.spec_from_file_location("shiu_published_model", groups["model_source"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        params = dict(module.default_params)
        # Upstream reset includes `w = 0`, but w belongs to Synapses, not NeuronGroup.
        # Remove this undefined assignment; retain membrane and synaptic-state reset.
        params["eq_rst"] = "v = v_rst; g = 0 * mV"
        print(f"Loading {len(self.ids):,} neurons and published connectivity...", flush=True)
        self.neurons, self.synapses, self.monitor = module.create_model(
            groups["completeness"], groups["connectivity"], params)
        self.record_spikes = config.get("record_spikes", True)
        if not self.record_spikes:
            # Continuous live sessions retain counts, not an ever-growing event history.
            self.monitor = b2.SpikeMonitor(self.neurons, record=False)
        self.poisson = b2.PoissonGroup(len(all_inputs), rates=0 * b2.Hz)
        self.stimulation = b2.Synapses(self.poisson, self.neurons,
                                      on_pre="v_post += stimulus_weight",
                                      namespace={"stimulus_weight": params["w_syn"] * params["f_poi"]})
        self.stimulation.connect(i=np.arange(len(all_inputs)), j=all_inputs)
        self.neurons.rfc[all_inputs] = 0 * b2.ms
        offset = 0
        self.input_slices = {}
        for name, indices in self.inputs.items():
            self.input_slices[name] = slice(offset, offset + len(indices))
            offset += len(indices)
        self.network = b2.Network(self.neurons, self.synapses, self.monitor,
                                  self.poisson, self.stimulation)
        self.previous_counts = np.zeros(len(self.ids), dtype=np.int64)
        self.last_spike_counts = np.zeros(len(self.ids), dtype=np.int64)
        self.spike_offset = 0
        self.raster_indices = np.unique(np.concatenate([
            indices[:config.get("raster_neurons_per_group", 24)]
            for indices in {**self.inputs, **self.outputs}.values()
        ]))
        self.group_sizes = {k: len(v) for k, v in {**self.inputs, **self.outputs}.items()}
        self.synapse_count = len(self.synapses)
        print(f"Loaded {self.synapse_count:,} directed connections; groups: {self.group_sizes}", flush=True)

    def step(self, sensory_input, dt):
        step_count(dt, self.config["neural_dt_s"])
        unknown = sensory_input.keys() - self.inputs.keys()
        if unknown:
            raise ValueError(f"Unknown input groups: {unknown}")
        rates = np.zeros(len(self.poisson))
        for name, rate in sensory_input.items():
            if not np.isfinite(rate) or rate < 0 or rate * self.config["neural_dt_s"] > 1:
                raise ValueError(f"Invalid stimulation rate for {name}: {rate}")
            rates[self.input_slices[name]] = rate
        self.poisson.rates = rates * b2.Hz
        self.network.run(dt * b2.second, namespace={})
        counts = np.asarray(self.monitor.count[:], dtype=np.int64)
        delta = counts - self.previous_counts
        self.last_spike_counts = delta
        self.previous_counts = counts.copy()
        population_rates = {name: float(delta[idx].sum() / (len(idx) * dt))
                            for name, idx in {**self.inputs, **self.outputs}.items()}
        if not self.record_spikes:
            return Activity(population_rates)
        new_i = np.asarray(self.monitor.i[self.spike_offset:], dtype=np.int32)
        new_t = np.asarray(self.monitor.t[self.spike_offset:] / b2.second)
        self.spike_offset = len(self.monitor.i)
        keep = np.isin(new_i, self.raster_indices)
        return Activity(population_rates, new_t[keep], self.ids[new_i[keep]])

    def selected_groups(self):
        return {name: [str(v) for v in self.ids[idx]]
                for name, idx in {**self.inputs, **self.outputs}.items()}
