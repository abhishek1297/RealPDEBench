# Synthetic Solver Workflow

This guide describes the current Processing-based controlled-cylinder and FSI
solvers, the trajectory runners, and the plotting workflow.

## 1. Build the Docker image

From the repository root:

```bash
docker build --network host -t realpdebench .
```

The image contains:

- Processing 3.5.4 and its bundled Java runtime
- Xvfb and xauth for headless Processing execution
- The Conda environment from `environment.yml`
- Gym 0.26.2 and the RealPDEBench Python package

The Processing download can be overridden for a cluster mirror:

```bash
docker build --network host \
  --build-arg PROCESSING_URL=https://example.org/processing-3.5.4-linux64.tgz \
  -t realpdebench .
```

## 2. Controlled-cylinder solver

The controlled-cylinder parameter row has two values:

```text
[Re, dia_0]
```

- `Re`: Reynolds number. Larger values produce a more strongly unsteady wake.
- `dia_0`: cylinder diameter in the solver's nondimensional parameterization.

The runner uses a zero control action (`v1=0`) and appends the parameter row to
form the solver action:

```text
[v1, Re, dia_0]
```

### Inline parameters

```bash
docker run --rm \
  -v "$PWD/trajectories:/workspace/RealPDEBench/trajectories" \
  realpdebench \
  python -u realpdebench/data/sim_generation/controlled_cylinder/run_trajectories.py \
  --parameters '[[100, 0.1], [1000, 0.1], [2000, 0.1]]' \
  --steps 100 \
  --output-dir trajectories/ccylinder/
```

Each row creates one independent solver trajectory:

```text
trajectories/ccylinder/trajectory_000.npy
trajectories/ccylinder/trajectory_001.npy
trajectories/ccylinder/trajectory_002.npy
```

### Parameters from a NumPy file

The file must contain an array with shape `(N, 2)`:

```python
import numpy as np

parameters = np.array([
    [100.0, 0.1],
    [1000.0, 0.1],
    [2000.0, 0.1],
])
np.save("parameters.npy", parameters)
```

Run it with:

```bash
docker run --rm \
  -v "$PWD:/workspace/RealPDEBench" \
  realpdebench \
  python -u realpdebench/data/sim_generation/controlled_cylinder/run_trajectories.py \
  --parameters-file parameters.npy \
  --steps 100 \
  --output-dir trajectories/ccylinder/
```

## 3. FSI solver

The FSI parameter row has five values:

```text
[Re, epi_C, massR_C, dia_0, dia_1]
```

- `Re`: Reynolds number.
- `epi_C`: structural damping parameter (`cr/c` in the Processing code).
- `massR_C`: cylinder mass ratio.
- `dia_0`: diameter of the first cylinder.
- `dia_1`: diameter of the second cylinder.

The runner uses two zero control actions (`v1=0`, `v2=0`) and forms:

```text
[v1, v2, Re, epi_C, massR_C, dia_0, dia_1]
```

Example:

```bash
docker run --rm \
  -v "$PWD/trajectories:/workspace/RealPDEBench/trajectories" \
  realpdebench \
  python -u realpdebench/data/sim_generation/fsi/run_trajectories.py \
  --parameters '[[1000, 0.8, 15.0, 0.1, 0.1]]' \
  --steps 100 \
  --output-dir trajectories/fsi/
```

### Current FSI caveat

The FSI Processing sketch and Python decoder now both use a 384x384x3 field,
and a one-frame FSI trajectory has been smoke-tested end to end. Longer runs
should still be checked for numerical stability and physical validity.

Also, the Processing FSI setup currently consumes `Re`, `epi_C`, and `massR_C`
for the structural model. The `dia_0` and `dia_1` values are accepted by the
Python action interface but are currently not used by the Processing setup
path.

## 4. Loading a trajectory

Each `.npy` file contains a pickled dictionary. Load it with:

```python
import numpy as np

trajectory = np.load(
    "trajectories/trajectory_000.npy",
    allow_pickle=True,
).item()

fields = trajectory["fields"]
observations = trajectory["observations"]
boundaries = trajectory["boundaries"]
actions = trajectory["actions"]
parameters = trajectory["parameters"]
rewards = trajectory["rewards"]
done = trajectory["done"]
```

For controlled cylinder, `fields` has shape:

```text
(time, 128, 128, 3)
```

The final axis contains `u`, `v`, and `p`. `boundaries` has shape:

```text
(time, 1, 40, 2)
```

For FSI, the intended field shape is:

```text
(time, 384, 384, 3)
```

with two body boundaries.

The runner performs three initialization calls before collecting frames because
the wrapper's first responses are warm-up placeholders while Processing aligns
its internal stepping cadence. Therefore, `--steps 100` means 100 saved frames,
plus the initialization calls.

## 5. Plotting

The plotting utility is [plot_trajectory.py](plot_trajectory.py). It supports
`u`, `v`, `p`, and `speed`.

Interactive animation:

```bash
python3 plot_trajectory.py \
  trajectories/trajectory_000.npy \
  --component speed
```

Single frame:

```bash
python3 plot_trajectory.py \
  trajectories/trajectory_000.npy \
  --component p \
  --frame 0
```

Save a GIF:

```bash
python3 plot_trajectory.py \
  trajectories/trajectory_000.npy \
  --component speed \
  --save trajectories/trajectory_000.gif
```

If the host does not have Matplotlib, run the plotter in the image:

```bash
docker run --rm \
  -e MPLBACKEND=Agg \
  -v "$PWD:/workspace/RealPDEBench" \
  realpdebench \
  python plot_trajectory.py \
  trajectories/trajectory_000.npy \
  --component speed \
  --save trajectories/trajectory_000.gif
```

## 6. Runtime caveats

- Processing, Java, Xvfb, and X11 libraries are required by the current
  Processing implementation, even though the goal is numerical output only.
- The `Gym has been unmaintained` message is a warning and does not cause the
  solver failure.
- `xauth` must be installed because `xvfb-run` uses it. It is included in the
  Docker image.
- Build again after changing Python or Processing source files so the changes
  are copied into the image.
- The runner uses one Processing solver per parameter row. Do not run multiple
  rows concurrently with the same X display number unless the display ranges
  are separated.
- Generated trajectories should be written to a mounted host directory so
  they survive container removal.
- For cluster jobs, distribute parameter rows across scheduler jobs and give
  each job a separate output directory. Avoid global `pkill` commands because
  they can terminate other jobs on a shared node.
