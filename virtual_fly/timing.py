import math
import time


def step_count(duration: float, dt: float) -> int:
    if not math.isfinite(duration) or not math.isfinite(dt) or duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    count = round(duration / dt)
    if count < 1 or not math.isclose(count * dt, duration, rel_tol=0, abs_tol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    return count


class Pacer:
    """Absolute wall-clock deadlines; never skip simulated steps to catch up."""
    def __init__(self, dt, realtime=False, clock=time.perf_counter, sleep=time.sleep):
        step_count(dt, dt)
        self.dt, self.realtime, self.clock, self.sleep = dt, realtime, clock, sleep
        self.start = clock()
        self.steps = 0
        self.missed_deadlines = 0
        self.max_lag_s = 0.0

    def tick(self):
        self.steps += 1
        if not self.realtime:
            return
        remaining = self.start + self.steps * self.dt - self.clock()
        if remaining > 0:
            self.sleep(remaining)
        else:
            self.missed_deadlines += 1
            self.max_lag_s = max(self.max_lag_s, -remaining)
