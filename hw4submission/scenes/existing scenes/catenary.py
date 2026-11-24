# For 4.2.
# uv run python main.py 4_2 scenes/hw4/catenary.py
# Btw in case you are interested in the name
# https://en.wikipedia.org/wiki/Catenary
import numpy as np

def make_scene():
    count = 64

    t = np.linspace(0.0, 1.0, count)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = -0.9 + 1.8 * t
    # Make it arch up at the beginning
    positions[:, 1] = 0.6 + 0.5 * np.sin(np.pi * t)

    # For visualization only
    edges = np.stack([np.arange(count - 1),
                      np.arange(1, count)], axis=1).astype(np.uint32)

    # Pin both endpoints
    inv_masses = np.ones(count, dtype=np.float32)
    inv_masses[0] = 0.0
    inv_masses[-1] = 0.0

    rest_lengths = np.linalg.norm(
        np.diff(positions, axis=0), axis=1).astype(np.float32)

    return {
        'positions': positions,
        'edges': edges,
        'inv_masses': inv_masses,
        'rest_lengths': rest_lengths,
        'substeps': 8,
        'iterations': 20,
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        'cam_pos': (1.2, 1.2, -1.2),
        'cam_target': (0.0, 0.4, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
