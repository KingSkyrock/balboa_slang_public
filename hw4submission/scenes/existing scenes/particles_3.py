# For 4.1
# uv run python main.py 4_1 scenes/hw4/particles_3.py

import numpy as np

def make_scene():
    rng = np.random.default_rng(1)
    count = 500

    # Turns out if you normalize a bunch of 3D
    # normal distribution points, they uniformly
    # distribute on a sphere!
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = rng.uniform(0.5, 0.9, count)

    positions = directions * radii[:, None]
    positions[:, 1] += 0.4

    # Make every particle meet at the same point after 1 second
    dt = 1.0 / 60.0
    steps = 60
    T = steps * dt

    target = np.array([0.0, 0.5, 0.0])
    gravity = np.array([0.0, -9.8, 0.0])

    velocities = (
        (target - positions) / T
        - 0.5 * gravity * dt * (steps + 1)
    )

    return {
        'positions': positions.astype(np.float32),
        'velocities': velocities.astype(np.float32),
        'show_particles': True,
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        'cam_pos': (3.0, 0.6, -3.0),
        'cam_target': (0.0, 0.5, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
