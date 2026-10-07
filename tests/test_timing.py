import pytest
from virtual_fly.timing import Pacer, step_count


class Clock:
    def __init__(self):
        self.now = 10.
    def clock(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


def test_absolute_deadlines_do_not_accumulate_compute_time():
    clock = Clock()
    pacer = Pacer(.01, True, clock.clock, clock.sleep)
    for _ in range(100):
        clock.now += .003
        pacer.tick()
    assert clock.now == pytest.approx(11.)
    assert pacer.missed_deadlines == 0


def test_overrun_is_reported_without_dropping_steps():
    clock = Clock()
    pacer = Pacer(.01, True, clock.clock, clock.sleep)
    clock.now += .035
    pacer.tick()
    assert pacer.steps == 1
    assert pacer.missed_deadlines == 1
    assert pacer.max_lag_s == pytest.approx(.025)


def test_fast_mode_never_sleeps():
    def no_sleep(_):
        pytest.fail("Fast mode must not sleep")
    pacer = Pacer(.01, False, lambda: 0, no_sleep)
    pacer.tick()
    assert pacer.steps == 1


@pytest.mark.parametrize("duration,dt", [(1, .3), (0, .01), (1, 0), (float('nan'), .1)])
def test_invalid_timing_rejected(duration, dt):
    with pytest.raises(ValueError):
        step_count(duration, dt)


def test_integer_multirate_clock():
    assert step_count(.01, .0001) == 100
    assert step_count(1., .01) == 100
