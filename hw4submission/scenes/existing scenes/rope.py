# For 4.2.
# uv run python main.py 4_2 scenes/hw4/rope.py

import numpy as np

def make_scene():
    count = 24
    # A horizontal line at y=0.6
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1.8, count)
    positions[:, 1] = 0.6
    # For visualization purpose. Not used in simulation
    edges = np.stack([np.arange(count - 1),
                      np.arange(1, count)], axis=1).astype(np.uint32)

    inv_masses = np.ones(count, dtype=np.float32)
    inv_masses[0] = 0.0
    rest_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)
    return {
        "positions": positions,
        "edges": edges,
        "inv_masses": inv_masses,
        "rest_lengths": rest_lengths,
        "substeps": 4, # number of timesteps we take per frame
        "iterations": 10, # number of Jacobi iterations we run per timestep
        # The planes are stored as (a, b, c, d)
        # where ax+by+cz=d
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        # Camera configuration
        'cam_pos': (1.5, -0.8, -1.5),
        'cam_target': (0.0, -0.4, 0.0),
        'cam_up': (0.0, 1.0, 0.0)
    }
