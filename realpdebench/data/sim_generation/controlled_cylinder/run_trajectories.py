"""Run controlled-cylinder trajectories for one or more IC parameter rows.

Examples (run from this directory)::

    python run_trajectories.py --parameters '[[100, 0.1], [200, 0.12]]'
    python run_trajectories.py --parameters-file parameters.npy --steps 200

Each parameter row is ``[Re, dia_0]``. The initial state is created by
``reset()`` followed by the first ``step()`` with that row appended to the
control action. Results are written to ``trajectories/trajectory_XXX.npy``.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from env.flow_field_env import env


FIELD_SHAPE = (128, 128, 3)
BOUNDARY_SHAPE = (1, 40, 2)
FIELD_SIZE = int(np.prod(FIELD_SHAPE))
BOUNDARY_SIZE = int(np.prod(BOUNDARY_SHAPE))
SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_INTERVAL = 10
TIME_PER_SAVED_STEP = 0.1
DISPLAY_START = 20
OUTPUT_DIR = Path("trajectories")

CONFIG = SimpleNamespace(
    observation_dim=49235,
    action_dim=3,
    action_interval=ACTION_INTERVAL,
    num_structure=1,
    path_env=str(SCRIPT_DIR) + "/",
    name_env="controlled_cylinder",
    all_features=["angle", "CD", "CL"],
    dict_action={"v1": 0, "Re": 1, "dia_0": 2},
    dict_observe={"angle": 3, "CD": 4, "CL": 5},
    dict_state={"u": 49152, "bd": 49232, "angle": 49234,
                "CD": 49232, "CL": 49233},
    store_step=0,
)


def load_parameters(parameters, parameters_file):
    if parameters_file:
        rows = np.load(parameters_file)
    elif parameters:
        rows = np.asarray(json.loads(parameters), dtype=float)
    else:
        raise ValueError("provide --parameters or --parameters-file")

    rows = np.atleast_2d(rows).astype(float)
    if rows.shape[1] != 2:
        raise ValueError("controlled-cylinder parameters must have shape (N, 2): [Re, dia_0]")
    return rows


def run_one(parameters, steps, network_port):
    solver = env(config=CONFIG, network_port=network_port)
    observations = []
    fields = []
    boundaries = []
    actions = []
    rewards = []
    done = []

    try:
        solver.reset()
        control = np.zeros(1, dtype=float)
        action = np.concatenate((control, parameters))

        # The wrapper's first three calls initialize Processing and align its
        # internal 10-frame cadence; only the third response contains fields.
        for _ in range(3):
            observation, reward, is_done, _ = solver.step(action)
        observations_to_collect = [observation] + [None] * (steps - 1)

        for step_index, initial_observation in enumerate(observations_to_collect):
            if initial_observation is None:
                observation, reward, is_done, _ = solver.step(action)
            else:
                observation = initial_observation
            observation = np.asarray(observation, dtype=float).reshape(-1)
            if observation.size != CONFIG.observation_dim:
                raise RuntimeError(
                    f"unexpected observation size {observation.size}; "
                    f"expected {CONFIG.observation_dim}"
                )

            observations.append(observation)
            fields.append(observation[:FIELD_SIZE].reshape(FIELD_SHAPE))
            boundaries.append(
                observation[FIELD_SIZE:FIELD_SIZE + BOUNDARY_SIZE].reshape(BOUNDARY_SHAPE)
            )
            actions.append(action.copy())
            rewards.append(float(np.asarray(reward).mean()))
            finished = bool(np.asarray(is_done).any())
            done.append(finished)
            print(
                f"trajectory step {step_index + 1}/{steps} "
                f"(t={(step_index + 1) * TIME_PER_SAVED_STEP:.3f})",
                flush=True,
            )
            if finished:
                break
            action = np.concatenate((control, parameters))
    finally:
        solver.close()

    return {
        "observations": np.asarray(observations),
        "fields": np.asarray(fields),
        "boundaries": np.asarray(boundaries),
        "actions": np.asarray(actions),
        "parameters": parameters,
        "rewards": np.asarray(rewards),
        "done": np.asarray(done),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", help="JSON list of [Re, dia_0] rows")
    parser.add_argument("--parameters-file", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    rows = load_parameters(args.parameters, args.parameters_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, parameters in enumerate(rows):
        print(f"running controlled-cylinder trajectory {index}: {parameters.tolist()}")
        result = run_one(parameters, args.steps, DISPLAY_START + index)
        output = args.output_dir / f"trajectory_{index:03d}.npy"
        np.save(output, result, allow_pickle=True)
        print(f"saved {output}")


if __name__ == "__main__":
    main()