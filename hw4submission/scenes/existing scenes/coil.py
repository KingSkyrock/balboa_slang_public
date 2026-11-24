# For 4.2
# uv run python main.py 4_2 scenes/hw4/coil.py

import numpy as np

def make_scene():
    count = 4096
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)

    theta = 2.0 * np.pi * 35.0 * t
    radius = 0.2 + 5.0 * np.sin(0.5 * np.pi * t) ** 1.5

    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = radius * np.cos(theta)
    positions[:, 1] = 20.0 - 15.0 * t
    positions[:, 2] = radius * np.sin(theta)

    omega = 35.0
    velocities = np.zeros_like(positions)
    velocities[:, 0] = -omega * positions[:, 2]
    velocities[:, 2] = omega * positions[:, 0]

    velocities[:, 1] = 50.0 * t
    velocities[0] = 0.0

    inv_masses = np.ones(count, dtype=np.float32)

    edges = np.stack([np.arange(count - 1),
                      np.arange(1, count)], axis=1).astype(np.uint32)

    rest_lengths = np.linalg.norm(
        np.diff(positions, axis=0), axis=1).astype(np.float32)

    return {
        'positions': positions,
        'velocities': velocities,
        'edges': edges,
        'inv_masses': inv_masses,
        'rest_lengths': rest_lengths,
        'substeps': 4,
        'iterations': 10,
        'planes': np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32),
        'cam_pos': (16.0, 25.0, -21.0),
        'cam_target': (0.0, 8.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
