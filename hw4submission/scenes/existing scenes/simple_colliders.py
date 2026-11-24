# For 4.5
# uv run python main.py 4_5 scenes/hw4/simple_colliders.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

def make_scene():
    nx, ny = 25, 20
    sx, sy = 1.5, 1.2
    dx, dy = sx / (nx - 1), sy / (ny - 1)

    cloth, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    cloth_edges, cloth_lengths = make_cloth_constraints(nx, ny, dx, dy)

    return {
        'positions': cloth,
        'inv_masses': np.ones(len(cloth), dtype=np.float32),
        'constraint_edges': np.asarray(cloth_edges, dtype=np.uint32),
        'rest_lengths': np.asarray(cloth_lengths, dtype=np.float32),

        'triangles': np.asarray(triangles, dtype=np.uint32),
        'cloth_grid_size': (nx, ny),

        # Planes: dot(n, x) = d, allowed region dot(n, x) >= d.
        'planes': np.array([
            [0.0, 1.0, 0.0, -1.0],
        ], dtype=np.float32),

        # Spheres: center (xyz), radius (w).
        'spheres': np.array([
            [0.0, -0.3, 0.5, 0.6],
        ], dtype=np.float32),

        'collision_distance': 0.015,
        'contact_margin': 0.03,
        'static_friction': 0.6,
        'kinetic_friction': 0.4,

        'substeps': 4,
        'iterations': 10,

        # Camera configuration
        'cam_pos': (2.0, 0.1, -2.0),
        'cam_target': (0.0, -0.5, 0.0),
        'cam_up': (0.0, 1.0, 0.0)
    }
