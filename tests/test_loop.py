from virtual_fly.loop import run_loop
from virtual_fly.types import MotorCommand
import pytest


def test_loop_order_substeps_and_neural_time():
    events = []
    class Body:
        dt = .001
        def observe(self):
            events.append("sense")
            return "observation"
        def act(self, command):
            assert command == MotorCommand(.4, .6)
            events.append("act")
        def render(self):
            events.append("render")
    class Encoder:
        def encode(self, obs):
            assert obs == "observation"
            events.append("encode")
            return {"input": 100}
    class Brain:
        def step(self, stim, dt):
            assert stim == {"input": 100}
            assert dt == .01
            events.append("brain")
            return "activity"
    class Decoder:
        def decode(self, activity, dt):
            assert activity == "activity"
            events.append("decode")
            return MotorCommand(.4, .6)
    pacer = run_loop(Body(), Brain(), Encoder(), Decoder(), .02, .01)
    assert events == (["sense", "encode", "brain", "decode"] + ["act", "render"] * 10) * 2
    assert pacer.steps == 2


def test_noninteger_physics_ratio_fails_before_running():
    class Body:
        dt = .003
    with pytest.raises(ValueError):
        run_loop(Body(), None, None, None, .02, .01)
