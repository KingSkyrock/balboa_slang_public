# For 4.3. 
# uv run python main.py 4_3 scenes/hw4/spin_cloth.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid

def make_scene():
    nx, ny = 23, 27
    sx, sy = 1.2, 1.7
    positions, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    positions[:, 1] = 0.75

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[0] = 0.0

    # Some rotating velocities
    anchor = positions[0].copy()
    omega = np.array([3.5, -1.0, 3.0], dtype=np.float32)
    velocities = np.cross(omega, positions - anchor).astype(np.float32)
    velocities[0] = 0.0

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),
        'substeps': 6,
        'iterations': 10,
        'planes': np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32),
        'cam_pos': (2.0, 1.5, -2.0),
        'cam_target': (0.0, 0.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
