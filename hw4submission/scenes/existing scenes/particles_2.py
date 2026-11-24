# For 4.1
# uv run python main.py 4_1 scenes/hw4/particles_2.py

import numpy as np

def make_scene():
    rng = np.random.default_rng(1)
    count = 300

    theta = rng.uniform(0, 2 * np.pi, count)
    speed = rng.uniform(0.5, 1.5, count)

    positions = rng.uniform(-0.05, 0.05, (count, 3)).astype(np.float32)
    positions[:, 1] -= 0.8

    velocities = np.zeros_like(positions)
    velocities[:, 0] = speed * np.cos(theta)
    velocities[:, 1] = rng.uniform(3.0, 4.0, count)
    velocities[:, 2] = speed * np.sin(theta)

    return {
        'positions': positions,
        'velocities': velocities,
        'show_particles': True,
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        # Camera configuration
        'cam_pos': (1.5, 0.1, -1.0),
        'cam_target': (0.0, -0.5, 0.0),
        'cam_up': (0.0, 1.0, 0.0)
    }
