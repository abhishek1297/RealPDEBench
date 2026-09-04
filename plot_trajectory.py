"""Plot a trajectory saved by the solver runner.

Examples:
    python plot_trajectory.py trajectories/trajectory_000.npy
    python plot_trajectory.py trajectories/trajectory_000.npy --component speed
    python plot_trajectory.py trajectories/trajectory_000.npy --save trajectory.gif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


COMPONENTS = {"u": 0, "v": 1, "p": 2}


def load_trajectory(path):
    trajectory = np.load(path, allow_pickle=True).item()
    fields = np.asarray(trajectory["fields"])
    if fields.ndim != 4 or fields.shape[-1] != 3:
        raise ValueError(
            f"expected fields with shape (time, height, width, 3), got {fields.shape}"
        )
    return trajectory, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--component", choices=["u", "v", "p", "speed"],
                        help="plot one field; default: plot all four fields")
    parser.add_argument("--interval", type=int, default=100, help="milliseconds between frames")
    parser.add_argument("--save", type=Path, help="save animation as .gif or .mp4")
    parser.add_argument("--frame", type=int, help="show one frame instead of animating")
    args = parser.parse_args()

    trajectory, fields = load_trajectory(args.trajectory)
    components = [args.component] if args.component else ["u", "v", "p", "speed"]
    values = {}
    for component in components:
        if component == "speed":
            component_values = np.hypot(fields[..., 0], fields[..., 1])
        else:
            component_values = fields[..., COMPONENTS[component]]
        # Processing stores values as [x, y]; imshow expects [y, x].
        values[component] = component_values.transpose(0, 2, 1)

    frame_count = fields.shape[0]
    if frame_count == 0:
        raise ValueError("trajectory contains no frames")

    frame_index = 0 if args.frame is None else args.frame
    if not 0 <= frame_index < frame_count:
        raise ValueError(f"--frame must be between 0 and {frame_count - 1}")

    if len(components) == 1:
        figure, axes = plt.subplots(figsize=(7, 6))
        axes = [axes]
    else:
        figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        axes = axes.ravel()

    images = []
    titles = []
    for axis, component in zip(axes, components):
        image = axis.imshow(values[component][frame_index], origin="lower", cmap="viridis")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04).set_label(component)
        axis.set_xlabel("x grid index")
        axis.set_ylabel("y grid index")
        images.append(image)
        titles.append(axis.set_title(f"{component}, frame {frame_index}"))

    boundaries = np.asarray(trajectory.get("boundaries", []))
    boundary_lines = []
    if boundaries.ndim == 4 and boundaries.shape[0] == frame_count:
        for axis in axes:
            for body in range(boundaries.shape[1]):
                line, = axis.plot([], [], "w-", linewidth=1.5)
                boundary_lines.append((line, body))

    def draw(frame):
        for image, title, component in zip(images, titles, components):
            image.set_data(values[component][frame])
            title.set_text(f"{component}, frame {frame}")
        boundary_index = 0
        for axis in axes:
            for body in range(boundaries.shape[1] if boundaries.ndim == 4 else 0):
                line, _ = boundary_lines[boundary_index]
                boundary_index += 1
                boundary = boundaries[frame, body]
                line.set_data(boundary[:, 0], boundary[:, 1])
        return (*images, *titles, *(line for line, _ in boundary_lines))

    if args.frame is not None:
        draw(frame_index)
        plt.tight_layout()
        plt.show()
        return

    animation = FuncAnimation(
        figure,
        draw,
        frames=frame_count,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    if args.save:
        suffix = args.save.suffix.lower()
        if suffix == ".gif":
            animation.save(args.save, writer=PillowWriter(fps=max(1, 1000 // args.interval)))
        elif suffix == ".mp4":
            animation.save(args.save, writer=FFMpegWriter(fps=max(1, 1000 // args.interval)))
        else:
            raise ValueError("--save must end in .gif or .mp4")
        print(f"saved {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
