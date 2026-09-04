"""Run fluid-structure-interaction trajectories for IC parameter rows.

Each row is ``[Re, epi_C, massR_C, dia_0, dia_1]``. For example::

    python run_trajectories.py --parameters '[[100, 0.1, 2.0, 0.1, 0.1]]'

The script resets the Processing solver, initializes it with the first step,
then saves a trajectory to ``trajectories/trajectory_XXX.npy``. The saved
``observations`` array is the decoder output and ``fields`` is shaped as
``(time, 384, 384, 3)``.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from env.flow_field_env import env


FIELD_SHAPE = (384, 384, 3)
BOUNDARY_SHAPE = (2, 40, 2)
FIELD_SIZE = int(np.prod(FIELD_SHAPE))
BOUNDARY_SIZE = int(np.prod(BOUNDARY_SHAPE))
SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_INTERVAL = 10
TIME_PER_SAVED_STEP = 1.0
DISPLAY_START = 40
OUTPUT_DIR = Path("trajectories")

CONFIG = SimpleNamespace(
    observation_dim=442534,
    action_dim=7,
    action_interval=ACTION_INTERVAL,
    num_structure=2,
    path_env=str(SCRIPT_DIR) + "/",
    name_env="fluid_structure_interaction",
    all_features=["angle", "CD", "CL"],
    dict_action={"v1": 0, "v2": 1, "Re": 2, "epi_C": 3,
                 "massR_C": 4, "dia_0": 5, "dia_1": 6},
    dict_observe={"angle": 3, "CD": 4, "CL": 5},
    dict_state={"bd": 442528, "angle": 442532, "CD": 442528,
                "CL": 442530},
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
    if rows.shape[1] != 5:
        raise ValueError(
            "FSI parameters must have shape (N, 5): "
            "[Re, epi_C, massR_C, dia_0, dia_1]"
        )
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
        control = np.zeros(2, dtype=float)
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
    parser.add_argument("--parameters", help="JSON list of five-value parameter rows")
    parser.add_argument("--parameters-file", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    rows = load_parameters(args.parameters, args.parameters_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, parameters in enumerate(rows):
        print(f"running FSI trajectory {index}: {parameters.tolist()}")
        result = run_one(parameters, args.steps, DISPLAY_START + index)
        output = args.output_dir / f"trajectory_{index:03d}.npy"
        np.save(output, result, allow_pickle=True)
        print(f"saved {output}")


if __name__ == "__main__":
    main()