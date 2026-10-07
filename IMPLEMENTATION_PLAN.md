# Fly operating a virtual fly simulation

## Intended outcome
An outer connectome-controlled fly operates a virtual workstation whose display
shows an independently simulated inner fly. Both worlds remain interactive for
the duration of the local session. Native Windows; CPU physics and neural solver.

This is an engineered interface experiment, not evidence of computer literacy,
intent, consciousness, or recursive cognition in the connectome model.

## Implemented extension: independent inner brain
The user requested a full brain for the inner fly after milestone 1. The inner
world now runs its own full connectome in an isolated Brian2 process, with its
own sensory encoder, motor decoder, clock, seed and live neural activity view.
This supersedes milestone 1's original no-second-brain budget and direct motor
bridge: virtual keys now add sensory stimulation, never directly drive legs.
Disconnecting removes those additions while inner odor sensing continues.
Independent-clock/activity/cleanup backend regression test passes. Manual visual
and full-scale interaction verification remains with the user. Physical typing
and screen feedback below are still outstanding.

## Milestone 1 — independent inner world and explicit command bridge
- Add an optional workstation prop to the outer arena.
- Run a second FlyGym/MuJoCo body with its own state and walking controller.
- Show the inner world in a dedicated computer-screen panel alongside the outer
  body and brain activity. The screen panel is the workstation's display; the
  initial 3D monitor is a decorative prop, not a textured live display.
- Translate outer motor drives into discrete forward/left/right/idle commands,
  with configurable thresholds. Show exactly which command was issued.
- Allow the user to disconnect the bridge and pause the inner simulation.
- Advance each world by the same fixed interval per completed outer step. Global
  pause/step affects both. Disconnected inner physics continues with zero drive;
  inner pause freezes its clock. No second whole-brain model is allocated.
- Log outer command, virtual keys, inner command, inner position and inner time.

Acceptance: user can run/pause/step, see two independent positions and clocks,
change stimulation/manual drive, and inspect the resulting virtual command.
Browser testing is performed manually by the user, as requested.

## Milestone 2 — embodied keyboard interaction
Add physical keyboard targets and foot-contact sensing with debouncing. Replace
the direct motor bridge with key events from measured contacts. Add an explicit
approach/operate task controller; label any scripted assistance. Render the live
inner framebuffer on the 3D monitor. Success means measured contacts produce
logged key events, and moving away prevents typing.

## Milestone 3 — screen feedback and a measurable task
Encode inner target error or screen observations into selected outer sensory
groups. Start with a documented engineered error encoding. Measure inner target
distance, completion time and key efficiency against disconnected/scripted
baselines. The initial command bridge is feed-forward; this milestone closes
the inner-world feedback path. Learning requires a separate explicit training
objective and mechanism; the existing connectome alone does not supply it.

## Milestone 4 — optional recursion and performance
Only after the two-world task works, allow another bounded nesting level with
per-world time budgets. Benchmark CPU time and memory before adding neural
models. Never imply real-time performance or create unbounded recursive worlds.

## Delivery and verification policy
Implement milestone 1 first, preserving existing batch modes. Use pinned existing
libraries, bounded frame queues and streamed logs. Keep the bridge independent
of rendering. Document completed work and outstanding milestones in README.
No automated browser/site testing; user performs visual and interaction checks.
