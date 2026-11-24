# For 4.2
# uv run python main.py 4_2 scenes/hw4/crazy_rope.py

import numpy as np

def make_scene():
    count = 24
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1.8, count)
    positions[:, 1] = 0.6

    edges = np.stack([np.arange(count - 1),
                      np.arange(1, count)], axis=1).astype(np.uint32)

    inv_masses = np.ones(count, dtype=np.float32)
    inv_masses[0] = 0.0

    # Three increasingly heavy beads.
    inv_masses[8] = 0.25
    inv_masses[16] = 0.1
    inv_masses[-1] = 0.05

    # Strong rotational kick around the pinned endpoint.
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    velocities = np.zeros_like(positions)
    velocities[:, 1] = 14.0 * t
    velocities[:, 2] = 3.0 * t

    rest_lengths = np.linalg.norm(
        np.diff(positions, axis=0), axis=1).astype(np.float32)

    return {
        'positions': positions,
        'velocities': velocities,
        'edges': edges,
        'inv_masses': inv_masses,
        'rest_lengths': rest_lengths,
        'substeps': 8,
        'iterations': 20,
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        'cam_pos': (2.0, 1.5, -3.0),
        'cam_target': (0.2, 0.5, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
