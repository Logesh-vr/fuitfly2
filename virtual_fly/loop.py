import time
from .timing import Pacer, step_count


def run_loop(body, brain, encoder, decoder, duration, dt, realtime=False, on_step=None,
             clock=time.perf_counter, sleep=time.sleep):
    """Sense at t; integrate brain to t+dt; hold decoded drive across body's dt.

    This is a discrete sample-and-hold coupling, not continuous simultaneous integration.
    Brain and body both advance exactly dt on every iteration.
    """
    n = step_count(duration, dt)
    physics_steps = step_count(dt, body.dt)
    pacer = Pacer(dt, realtime, clock, sleep)
    for i in range(n):
        observation = body.observe()
        stimulation = encoder.encode(observation)
        activity = brain.step(stimulation, dt)
        command = decoder.decode(activity, dt)
        for _ in range(physics_steps):
            body.act(command)
            body.render()
        if on_step:
            on_step((i + 1) * dt, observation, stimulation, activity, command, body.observe())
        pacer.tick()
    return pacer
