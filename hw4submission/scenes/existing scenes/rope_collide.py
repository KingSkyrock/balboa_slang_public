# For 4.5. 
# uv run python main.py 4_5 scenes/hw4/rope_collide.py

import numpy as np

def make_scene():
    count = 72
    t = np.linspace(0, 1, count, dtype=np.float32)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = -1.6 + 3.2 * t
    positions[:, 1] = 1.1

    edges = np.stack((np.arange(count - 1),
                      np.arange(1, count)), axis=1).astype(np.uint32)
    rest_lengths = np.linalg.norm(
        np.diff(positions, axis=0), axis=1).astype(np.float32)
    inv_masses = np.ones(count, dtype=np.float32)
    inv_masses[0] = 0.0
    inv_masses[-1] = 0.2

    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'edges': edges,  # Visualization only
        'planes': np.array([[0, 1, 0, -1]], dtype=np.float32),
        'spheres': np.array([
            [ 0.6,  0.0,  0.0, 0.50],
            [-0.7, -0.3,  0.0, 0.25],
        ], dtype=np.float32),
        'collision_distance': 0.025,
        'contact_margin': 0.06,
        'static_friction': 0.45,
        'kinetic_friction': 0.30,
        'substeps': 8,
        'iterations': 20,
        'cam_pos': (2.0, 2.0, -2.5),
        'cam_target': (0.0, 0.15, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
