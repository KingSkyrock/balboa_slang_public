# For 4.3
# uv run python main.py 4_3 scenes/hw4/big_cloth.py
import numpy as np
from scenes.hw4.geometry import make_cloth_grid

def make_scene():
    nx, ny = 256, 192
    sx, sy = 2.4, 1.8

    positions, triangles, dx, dy = \
        make_cloth_grid(nx, ny, sx, sy)

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[:nx] = 0.0  # Pin the entire top edge

    # Some crazy initial velocities
    u = np.tile(np.linspace(0, 1, nx), ny)
    v = np.repeat(np.linspace(0, 1, ny), nx)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 2.0 * v * np.sin(4 * np.pi * u)
    velocities[:, 1] = 4.0 * v * np.sin(2 * np.pi * u)
    velocities[:, 2] = 5.0 * v

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),
        'substeps': 6,
        'iterations': 10,
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),
        'cam_pos': (2.5, 0.5, -2.5),
        'cam_target': (0.0, 0.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
