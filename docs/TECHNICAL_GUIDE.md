# Virtual fruit fly — native Windows

A working three-module experiment connecting the published Shiu/FlyWire spiking
network to a MuJoCo/NeuroMechFly walking body. All three milestones have been run
locally. **The full loop currently runs about 92 times slower than real time.**
Real-time pacing is implemented, but this reference CPU backend does not meet the
real-time/near-real-time performance objective.

## Your machine

- 24 GiB installed RAM: two 12 GiB modules; approximately 23.7 GiB OS-visible.
- NVIDIA GeForce RTX 3050 A Laptop GPU, 4094 MiB VRAM, driver 591.66.
- Windows reports **Windows 11 Home Single Language, build 26200**, rather than
  Windows 10. Everything was executed directly on Windows; no WSL was used.
- Python 3.12.12; FlyGym 2.1.0; MuJoCo 3.9.0; Brian2 2.10.1.
- Neural simulation and physics use CPU. CUDA acceleration is not implemented.
  Rendering uses MuJoCo's local OpenGL renderer. GPU availability alone does not
  accelerate the chosen Brian2 backend.

## Setup

The environment and data are already installed in this folder. To recreate them,
install [uv](https://docs.astral.sh/uv/getting-started/installation/), open PowerShell,
change into the project directory, then run:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements.txt` pins every installed dependency, and setup downloads Python
3.12.12 into `.python`. It does not change system Python or PATH. The project-local
`.venv` is used explicitly, so shell activation is unnecessary. Brian2 creates its
standard `.brian` cache in your user directory. Plot, mesh, and Numba caches use
`.cache` within the project. Internet is needed for initial installation/downloads.
The full connectivity file is approximately 101 MB; environment installation takes
additional disk space. These exact dependencies were verified on the Windows 11
installation above; Windows 10 has not been separately tested.

## Run each milestone

### Nested fly with its own brain

The inner fly now runs an independent instance of the same full Shiu/FlyWire
connectome model as the outer fly (138,639 neurons and 15,091,983 connections per
brain). It has a separate Brian2 process, clock, random seed, encoder, decoder,
and MuJoCo body. Its own antenna odor readings stimulate its brain; its own
measured descending activity drives its legs. No outer motor commands are copied
into the inner legs. The two neural processes roughly double neural-state memory
requirements and compute sequentially; the original single-brain speed benchmark
below does not describe this configuration.

Press **Run simulation** and use **Fly's computer** for inner odor strength,
left/right stimulation, source placement, bridge disconnection, and inner pause.
The inner brain has its own interactive 3D activity view and firing-rate readout.
Both views share the schematic geometry because they use the same connectome and
sample indices; their measured spike counts and neural times are independent.
Global pause/step controls both loops. Inner pause freezes its brain and body.
Outer manual mode freezes only the outer brain; the inner brain still runs.

The outer virtual keyboard adds sensory stimulation to the inner brain: FORWARD
adds to both olfactory groups; LEFT or RIGHT adds to the corresponding group;
IDLE and DISCONNECTED add nothing. The mapping is an experimental design choice,
not a biologically established steering mechanism. Disconnect leaves the inner
fly's own odor loop active. Thresholds and `bridge_stimulation_hz` are in
`configs/nested.yaml`; both brains use `configs/neurons.yaml`, the configured
encoder and decoder. Restart after changing files. Set `enabled: false` to omit
the inner world entirely. `nested.jsonl` records inner neural time, measured
rates, displayed spike counts, stimulation, motor drives and trajectory each step.

The 3D PC remains a decorative prop; its live display is the separate UI panel.
Physical typing and inner-screen feedback into the outer brain remain future
milestones in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md). This does not
establish learned computer use. A small synthetic-model backend test verifies
independent clocks, activity and process cleanup; browser verification is manual.

### Interactive live lab

Double-click **Start Interactive Fly.cmd**, or run:

```powershell
.\.venv\Scripts\python.exe -m virtual_fly live
```

Open **http://127.0.0.1:8765/**. The session starts paused and has no fixed duration.
Press **Run simulation** or **Step 10 ms**. Click the arena map to move the odor
source; switch odor on/off, adjust its stimulation strength, or add independent
left/right sensory stimulation. Drag the body view to orbit and scroll to zoom.
**Manual walking** explicitly bypasses and freezes the neural model, exposing walking
and turning drives. Switch back to **Whole-brain closed loop** to resume neural
control. Controls apply at completed step boundaries, so neural computation still
limits the response latency. A pacing cap only slows execution; it does not make
the solver faster.

The **3D brain activity** panel sits alongside the fly simulation, so both views
remain visible while you adjust stimulation. On narrow phones the panels stack;
neither view is hidden behind a tab. Stimulation controls sit to the right on wide
screens and below the two views on smaller screens.
The brain panel displays a rotatable, zoomable sample of up to 6,000
neurons, prioritizing the configured sensory and descending populations. Other neurons
are sampled deterministically. Every shown node corresponds to a real neuron in the
loaded model. Node brightness reflects its actual spike count in the latest completed
neural interval, with the chosen glow scale controlling visual intensity only. Hover
a neuron to see its FlyWire root ID, annotation, side, spike count, and interval rate.
Filters show active, sensory, descending, or other neurons.

Up to 10,000 displayed edges are sampled from actual model connections between those
neurons. Pink edges have negative weights; blue edges have positive weights. A bright
edge indicates a spiking presynaptic neuron, not a directly measured transmission or
causal pathway. **Positions are schematic groups arranged by annotation and side,
not anatomical coordinates, reconstructed neuron shapes, or measured axon routes.**
Paused/manual modes keep the last neural snapshot visible and label it accordingly.

The 3D renderer is a locally bundled, pinned **Three.js 0.180.0**, with its MIT license
in `virtual_fly/static/three/LICENSE`; it uses browser WebGL and loads no CDN at runtime.
The live server binds only to loopback. Neural simulation remains on CPU. The browser
and solver run separately. Live mode keeps spike counts, bounded display buffers,
and a streamed CSV instead of accumulating every spike and video frame in RAM.
Each launch writes to a new timestamped directory under `outputs/live/`, including
the effective configs, `live.csv`, control events, and `brain_geometry.json`.
Use **End session** or Ctrl+C to stop the server. Closing only the browser tab does
not stop the simulation. Restart the launcher after Python backend changes, then
refresh the browser. The interactive/3D changes are left for your manual UI testing.

### Batch experiments

```powershell
# 1. Scripted walking body; 1 simulated second, 5-second slow-motion video
.\.venv\Scripts\python.exe -m virtual_fly body

# 2. Whole brain; baseline, left olfactory stimulation, recovery
.\.venv\Scripts\python.exe -m virtual_fly brain

# 3. Sense -> encode -> brain -> decode -> physics -> render
.\.venv\Scripts\python.exe -m virtual_fly closed_loop

# Controls, in the same odor arena
.\.venv\Scripts\python.exe -m virtual_fly closed_loop --no-stimulation --output outputs/no_stimulation
.\.venv\Scripts\python.exe -m virtual_fly body --config configs/closed_loop.yaml --output outputs/scripted_baseline
.\.venv\Scripts\python.exe scripts/compare_runs.py
```

Use `--output outputs/another_run` to retain previous experiments; the default paths
overwrite that mode's previous outputs. Use `--duration 2.0` to change simulated run
time. Use `--realtime` to pace against absolute wall-clock deadlines. The default
runs as fast as possible. Late steps are recorded, never skipped or shortened.
Video playback speed is independent of simulation speed and wall-clock pacing.
No window is required. Videos are ordinary MP4 files.

## Measured verification

These are single-seed engineering checks, not biological validation. Times exclude
initial network/model construction and final file encoding; the CPU was not isolated
from other processes.

| Milestone | Simulated time | Wall time | Verified result |
| --- | ---: | ---: | --- |
| Body | 1.00 s | 10.26 s | Walked 13.46 mm; video visually inspected |
| Brain | 0.30 s | 22.53 s | Sensory stimulation activated descending populations |
| Closed loop | 1.00 s | 91.83 s | Source distance decreased from 16.24 to 6.68 mm |
| Zero-stimulation control | 1.00 s | 83.76 s | Zero motor commands; only about 0.11 mm distance reduction |
| Scripted baseline | 1.00 s | 9.23 s | Source distance decreased to 4.42 mm, closer than the neural trial |

The approach criterion is a final distance reduction of at least 1 mm. Arrival is
separately defined as coming within 2 mm: **the first neural trial did not arrive**.
The source is ahead and slightly left of the initial fly. Straight scripted walking
also approaches it; this favorable arena does not establish odor-gradient steering.
Proving chemotaxis requires side/rear targets, odor-free and shuffled/symmetric-input
controls, multiple seeds, and decoder calibration. No direct odor-to-motor fallback
is hidden in the neural loop.

The no-stimulation body's small motion comes from settling and the locomotion
controller's initial state. Zero drive means the controller relaxes its CPG amplitude;
it is not a rigid-body freeze.

## Files and interfaces

```text
virtual_fly/
  brain.py        Persistent published Brian2 network adapter and group selection
  body.py         MuJoCo/FlyGym body and hybrid walking controller
  interface.py    Encoder and decoder
  loop.py         Fixed-interval coupling and physics substeps
  timing.py       Deadline pacing and integer-step validation
  types.py        Observation, Activity, MotorCommand, module protocols
  reporting.py    CSV/JSON/PNG output
  __main__.py     Independent milestone commands
configs/          Body, brain, closed loop, neuron groups, decoder YAML
scripts/          Windows setup, data download, experiment comparison
tests/            Encoder/decoder, timing, loop ordering, neural adapter tests
data_manifest.json  Source URLs and SHA-256 hashes
requirements.txt  Exact dependency versions
data/             Downloaded v783 connectivity and annotations (not committed)
vendor/shiu/      Pinned upstream source and MIT license (not committed)
outputs/          Videos, logs, plots, run configurations (not committed)
```

- `brain.step({group_name: stimulation_hz}, dt_seconds) -> Activity`: input Poisson
  event rates, output mean spikes/second/neuron and selected spike events. State
  persists across calls. Outputs also include input-population rates for diagnostics.
- `body.observe() -> Observation`: time in seconds, thorax position in mm, two
  antenna odor concentrations, joint angles in radians, contact force vectors, and
  optional left/right eye RGB images. Contact forces retain MuJoCo model units.
- `body.act(MotorCommand(left, right))`: advance one physics step (0.1 ms).
  Commands are dimensionless CPG amplitude drives, not mm/s or radians/s.
- Every 10 ms control interval samples the body, integrates the neural model by
  10 ms, decodes its interval firing rates, then holds the resulting drives across
  100 physics steps. Reflexes update at each physics step. This sequential coupling
  introduces a discretization delay; it is not continuous simultaneous integration.
- Physics, neural, and control timestep ratios must be integral. Tests verify exact
  ordering, substep counts, no skipped steps, deadline overrun reporting, and pacing
  without accumulated computation-time drift.

## Configuration and biological assumptions

The supplied [murthylab/codex](https://github.com/murthylab/codex) repository is the
FlyWire connectome explorer, not a spiking simulator. This project uses the published
[Shiu model](https://github.com/philshiu/Drosophila_brain_model), pinned to commit
`91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`, with that repository's v783 dataset:
**138,639 neurons and 15,091,983 directed connection rows**. These connection rows
carry synapse-count weights; they are not individual anatomical synapses. No
connectome subset or toy network is substituted in the milestone runs.

`configs/neurons.yaml` selects neurons from matching v783 Codex annotations:

- Left/right olfactory input populations: 1,116 / 1,133 neurons in this model.
- Left/right descending output populations: 646 / 647 neurons in this model.
- An annotation selection is intersected with model IDs, because the published model
  does not contain every annotated neuron. Actual resolved IDs are saved per run.
- Select other modalities using `class: visual`, `class: mechanosensory`, or
  `class: gustatory`; alternatively supply `ids: ["exact root id", ...]`. Explicit
  IDs must exist, selections must be nonempty, and input groups must not overlap.
- Preserve large root IDs as strings in YAML/JSON. The loader never converts them
  through floating point. Connectivity's pre/post index-to-root mappings were checked
  against the completeness table on the downloaded dataset.

The adapter calls the upstream `create_model` constructor and uses its LIF parameters,
signed connectivity weights, synaptic delays, refractory behavior, and linear state
integration. Two explicit adaptations are made:

1. The upstream reset expression contains `w = 0`, although `w` is a synapse variable,
   not a neuron variable. The adapter removes that undefined assignment and retains
   the voltage and synaptic-state reset. Upstream source is preserved unchanged.
2. To allow time-varying sensory rates, a vectorized Brian2 PoissonGroup with
   one-to-one voltage-injection synapses replaces fixed-rate per-neuron PoissonInput
   objects. The voltage kick and zero sensory refractory period follow upstream
   parameters. Random draws/scheduling are not bit-for-bit reproduction of the
   paper's trials. This project also uses a newer Brian2 than the publication.

`configs/closed_loop.yaml` specifies a static Gaussian odor field, sampled at actual
left/right antennal funiculus positions. This simple analytic field is not a turbulent
odor plume and has no receptor-specific odor chemistry or adaptation. Normalized
concentration maps linearly to a capped Poisson stimulation rate. The green sphere
is a non-colliding source marker. Set `body.vision: true` to expose FlyGym's raw eye
images; visual control has not been validated. Available encoder channels are
`odor_left`, `odor_right`, `vision_left`, `vision_right`, `contact`, `proprioception`.
The vision encoder uses mean brightness; contact/proprioception use mean magnitude.
Gustatory neuron selection is supported, but a biological taste sensor is not modeled.
The optional sensory readout was smoke-tested: two 512-by-450 RGB eye images,
66 joint angles, six 3D foot-contact force vectors, and two odor values.

`configs/decoder.yaml` defines:

```text
filtered_rates = exponential smoothing of left/right mean descending rates
speed_drive = clip(speed_gain * mean(filtered_rates), 0, max_drive)
turn_drive = turn_gain * (filtered_left - filtered_right)
left_drive  = clip(speed_drive - turn_drive, 0, max_drive)
right_drive = clip(speed_drive + turn_drive, 0, max_drive)
```

Below the mean-rate threshold both commands are zero. Side labels are anatomical
annotations, not proof of motor function. Both sensory and motor mappings are
engineering choices, **not learned from the connectome**. Broad descending-population
averages may discard relevant activity or combine antagonistic functions.

The brain connectome omits the full ventral nerve cord. FlyGym's existing hybrid
controller supplies the missing leg control layer using preprogrammed steps and
reflex rules. This initial implementation covers walking and standing. Wings are
visible, but flight aerodynamics, wing control, and grooming are not implemented.
This is not a biologically complete fly or validated biological behavior.

## Outputs and limits

Each run stores its resolved configuration and `summary.json`. Body runs save
`walking.mp4`, `motor_trajectory.csv`, and `trajectory_motor.png`. Brain runs save
`population_rates.csv`, `selected_spikes.csv`, `selected_groups.json`, and
`brain_activity.png`. Closed-loop runs combine those logs with `closed_loop.mp4`,
`sensory_stimulation.csv`, and a saved decoder configuration. `outputs/comparison.png`
compares the neural trial with both controls.

Firing rates use the latest 10 ms interval; decoder smoothing reduces sampling noise.
Rasters record a deterministic selection of at most 24 neurons per configured group.
Internally, the upstream spike monitor records all spikes, and video frames are held
in memory until saving. Long runs can consume substantial RAM; current defaults are
short bounded experiments. Crashes may leave an incomplete output directory.

The reference implementation prioritizes a verifiable full network and native
Windows compatibility. Faster execution will require profiling and potentially
compiled Brian2 code generation, a separately validated GPU neural backend, and
body-controller optimizations. The 4 GB GPU memory is a constraint to measure, not
a guarantee that this entire network can run on it. No CUDA toolkit or compiler was
installed, and no claim of near-real-time performance is made.

## Tests and attribution

The original 19-test backend suite passed during initial development; the added inner-brain isolation test passed separately. Neural unit tests use a clearly labeled two-neuron synthetic
fixture to verify the adapter's persistence and rate accounting; actual milestone
verification uses the complete downloaded network. Brian2 emits upstream Pyparsing
deprecation warnings; tests still pass.

Body: [FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym), Apache-2.0, and
MuJoCo. The adapter uses FlyGym's provided hybrid turning controller. Brain: Shiu
et al.'s [published source](https://github.com/philshiu/Drosophila_brain_model), MIT;
its license is retained in `vendor/shiu/LICENSE`. Annotations come from the
[Codex data loader's published v783 endpoint](https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py).
`data_manifest.json` records original URLs and file hashes. The downloader refuses
checksum changes. Consult the source projects/publications for data citation and
redistribution terms; this project does not grant new rights to upstream datasets.
