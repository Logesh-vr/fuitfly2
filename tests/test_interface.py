from pathlib import Path
import numpy as np
import pytest
import yaml
from virtual_fly.interface import Encoder, Decoder
from virtual_fly.types import Observation, Activity


def observation(odor):
    return Observation(0, np.zeros(3), np.array(odor), np.zeros(42), np.zeros((6, 3)))


def decoder():
    config = yaml.safe_load(Path("configs/decoder.yaml").read_text())
    config["smoothing_tau_s"] = 0
    return Decoder(config)


def test_encoder_preserves_laterality_and_clamps():
    config = yaml.safe_load(Path("configs/closed_loop.yaml").read_text())["encoder"]
    result = Encoder(config).encode(observation([.2, 2.]))
    assert result == {"olfactory_left": 60., "olfactory_right": 300.}


def test_encoder_rejects_nonfinite_sensory_data():
    config = yaml.safe_load(Path("configs/closed_loop.yaml").read_text())["encoder"]
    with pytest.raises(ValueError):
        Encoder(config).encode(observation([float("nan"), .1]))


def test_equal_population_rates_produce_equal_drives():
    command = decoder().decode(Activity({"descending_left": 3., "descending_right": 3.}), .01)
    assert command.left == pytest.approx(.6)
    assert command.right == pytest.approx(.6)


def test_opposite_rate_asymmetries_reverse_turn_command():
    left = decoder().decode(Activity({"descending_left": 4., "descending_right": 2.}), .01)
    right = decoder().decode(Activity({"descending_left": 2., "descending_right": 4.}), .01)
    assert left.left == right.right
    assert left.right == right.left
    assert left.left < left.right


def test_zero_activity_stands_and_extreme_activity_saturates():
    d = decoder()
    zero = d.decode(Activity({"descending_left": 0., "descending_right": 0.}), .01)
    assert zero.left == zero.right == 0.
    maximum = d.decode(Activity({"descending_left": 1e6, "descending_right": 1e6}), .01)
    assert maximum.left == maximum.right == 1.2


def test_missing_population_is_not_silently_treated_as_zero():
    with pytest.raises(KeyError):
        decoder().decode(Activity({"descending_left": 2}), .01)


def test_smoothing_depends_on_elapsed_time_not_number_of_calls():
    config = yaml.safe_load(Path("configs/decoder.yaml").read_text())
    a, b = Decoder(config), Decoder(config)
    activity = Activity({"descending_left": 2., "descending_right": 2.})
    for _ in range(10):
        x = a.decode(activity, .01)
    y = b.decode(activity, .1)
    assert x.left == pytest.approx(y.left)
