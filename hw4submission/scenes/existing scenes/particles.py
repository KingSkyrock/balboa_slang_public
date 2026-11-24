# For 4.1
# uv run python main.py 4_1 scenes/hw4/particles.py

import numpy as np

def make_scene():
    rng = np.random.default_rng(1)
    positions = (2.0 * rng.random((50, 3), dtype=np.float32) - 1.0) * 0.8
    positions[:, 1] += 0.2
    velocities = (2.0 * rng.random((50, 3), dtype=np.float32) - 1.0) * 0.05

    return {
        'positions': positions,
        'velocities': velocities,
        'show_particles': True,
        # The planes are stored as (a, b, c, d)
        # where ax+by+cz=d
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        # Camera configuration
        'cam_pos': (1.5, 0.1, -1.0),
        'cam_target': (0.0, -0.5, 0.0),
        'cam_up': (0.0, 1.0, 0.0)
    }
