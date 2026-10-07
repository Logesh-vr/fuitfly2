from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from virtual_fly.brain import ConnectomeBrain, select_groups


def test_selection_preserves_large_ids_and_model_order():
    ids = np.array([720575940596125868, 720575940597856265], dtype=np.int64)
    annotations = pd.DataFrame({"root_id": ids[::-1].astype(str), "side": ["right", "left"]})
    result = select_groups({"left": {"side": "left"}}, annotations, ids)
    assert result["left"].tolist() == [0]
    with pytest.raises(ValueError, match="absent"):
        select_groups({"missing": {"ids": ["123"]}}, annotations, ids)
    with pytest.raises(ValueError, match="empty"):
        select_groups({"missing": {"side": "center"}}, annotations, ids)


@pytest.fixture
def small_model(tmp_path):
    # Deliberately synthetic fixture, never used by any milestone/demo.
    pd.DataFrame({"Completed": [True, True]}, index=[101, 102]).to_csv(tmp_path / "neurons.csv")
    pd.DataFrame({"Presynaptic_Index": [0], "Postsynaptic_Index": [1],
                  "Excitatory x Connectivity": [100]}).to_parquet(tmp_path / "edges.parquet")
    pd.DataFrame({"root_id": [101, 102]}).to_csv(tmp_path / "annotations.csv", index=False)
    groups = dict(release=783, completeness=str(tmp_path / "neurons.csv"),
                  connectivity=str(tmp_path / "edges.parquet"),
                  annotations=str(tmp_path / "annotations.csv"),
                  model_source="vendor/shiu/model.py",
                  inputs={"input": {"ids": ["101"]}}, outputs={"output": {"ids": ["102"]}})
    (tmp_path / "groups.yaml").write_text(yaml.safe_dump(groups))
    return dict(groups_file=str(tmp_path / "groups.yaml"), neural_dt_s=.0001,
                codegen="numpy", raster_neurons_per_group=2)


@pytest.mark.skipif(not Path("vendor/shiu/model.py").exists(), reason="Run scripts/download_data.py first")
def test_persistent_brain_stimulation_and_rates(small_model):
    brain = ConnectomeBrain(small_model, seed=1)
    quiet = brain.step({}, .01)
    assert quiet.rates_hz == {"input": 0., "output": 0.}
    driven = brain.step({"input": 1000.}, .03)
    assert driven.rates_hz["input"] > 0
    assert driven.rates_hz["output"] > 0
    assert driven.spike_times_s.min() >= .01
    assert driven.spike_times_s.max() < .04
    for name, root in [("input", 101), ("output", 102)]:
        assert driven.rates_hz[name] == pytest.approx(np.sum(driven.spike_ids == root) / .03)
    brain.step({}, .01)
    assert float(brain.network.t) == pytest.approx(.05)
    with pytest.raises(ValueError, match="Unknown"):
        brain.step({"typo": 1}, .01)


def test_inner_brain_has_independent_clock_and_activity(small_model):
    import multiprocessing as mp
    from virtual_fly.inner_brain import InnerBrain
    outer = ConnectomeBrain(small_model, seed=1)
    outer.step({}, .01)
    inner = InnerBrain(small_model, 2, np.array([0, 1]), mp.get_context("spawn").Event())
    try:
        driven = inner.step({"input": 1000.}, .02)
        assert driven["rates"]["input"] > 0
        assert driven["counts"][0] / .02 == pytest.approx(driven["rates"]["input"])
        assert float(outer.network.t) == pytest.approx(.01)
        assert outer.last_spike_counts.tolist() == [0, 0]
        outer.step({}, .01)
        assert float(outer.network.t) == pytest.approx(.02)
    finally:
        inner.close()
    assert not inner.process.is_alive()
