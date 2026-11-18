# For 4.3.
# uv run python main.py 4_3 scenes/hw4/cloth.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid

def make_scene():
    nx, ny = 25, 20
    sx, sy = 1.5, 1.2
    positions, triangles, dx, dy = \
        make_cloth_grid(nx, ny, sx, sy)
    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[0] = 0.0
    inv_masses[nx - 1] = 0.0
    return {
        'positions': positions,
        'inv_masses': inv_masses,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),
        'substeps': 4,
        'iterations': 10,
        # The planes are stored as (a, b, c, d)
        # where ax+by+cz=d
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        # Camera configuration
        'cam_pos': (1.6, 0.5, -1.6),
        'cam_target': (0.0, 0.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0)
    }
