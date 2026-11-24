# For 4.3
# uv run python main.py 4_3 scenes/hw4/hammock.py
import numpy as np
from scenes.hw4.geometry import make_cloth_grid

def make_scene():
    nx, ny = 30, 25
    sx, sy = 1.8, 1.5
    positions, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)

    grid_y = np.repeat(np.arange(ny), nx)
    grid_x = np.tile(np.arange(nx), ny)
    u = grid_x / (nx - 1)
    v = grid_y / (ny - 1)

    # The attachment corners are closer together than the cloth's rest size.
    # Keep the original dx/dy rest lengths: this gives the hammock slack.
    positions[:, 0] *= 0.8
    positions[:, 2] *= 0.8
    positions[:, 1] = 0.6

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[[0, nx - 1, (ny - 1) * nx, nx * ny - 1]] = 0.0

    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),
        'substeps': 4,
        'iterations': 10,
        'planes': np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32),
        'cam_pos': (1.5, 1.5, -1.6),
        'cam_target': (0.0, 0.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
