# For 4.5
# uv run python main.py 4_5 scenes/hw4/cloth_tunnel.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

def make_scene():
    # Five alternating spheres
    radius = 0.95
    spheres = np.array([
        [-1.00, 8.00, 0.0, radius],
        [ 1.00, 5.90, 0.0, radius],
        [-1.00, 3.80, 0.0, radius],
        [ 1.00, 1.70, 0.0, radius],
        [-1.00, -0.40, 0.0, radius],
    ], dtype=np.float32)

    # Cloth above the first sphere
    nx, ny = 30, 25
    sx, sy = 1.0, 0.8
    positions, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    # Center cloth around z=0
    positions[:, 0] -= 0.20
    positions[:, 1] = 9.20
    edges, rest_lengths = make_cloth_constraints(nx, ny, dx, dy)
    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': np.ones(
            len(positions), dtype=np.float32),
        'constraint_edges': np.array(
            edges, dtype=np.uint32),
        'rest_lengths': np.array(
            rest_lengths, dtype=np.float32),
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'planes': np.array([
            [0, 1, 0, -1.65],
        ], dtype=np.float32),
        'spheres': spheres,

        'collision_distance': 0.02,
        'contact_margin': 0.05,
        'static_friction': 0.15,
        'kinetic_friction': 0.10,

        'substeps': 8,
        'iterations': 16,

        'cam_pos': (8.0, 5.0, -8.0),
        'cam_target': (0.0, 3.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
