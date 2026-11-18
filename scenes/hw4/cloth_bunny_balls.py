# For 4.6
# uv run python main.py 4_6 scenes/hw4/cloth_bunny_balls.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints
from scenes.hw4.cloth_bunny import make_scene as bunny_scene

def make_scene():
    scene = bunny_scene()

    bunny = scene['collision_mesh']['vertices']
    bmin = bunny.min(axis=0)
    bmax = bunny.max(axis=0)

    r = 0.6
    spheres = np.array([
        [bmin[0] - r - 0.1, r, 0.0, r],
        [bmax[0] + r + 0.1, r, 0.0, r],
        [0.0, r, bmin[2] - r - 0.1, r],
        [0.0, r, bmax[2] + r + 0.1, r],
    ], dtype=np.float32)

    nx, ny = 80, 60
    sx = bmax[0] - bmin[0] + 4 * r + 0.7
    sy = bmax[2] - bmin[2] + 4 * r + 0.7

    positions, triangles, dx, dy = make_cloth_grid(
        nx, ny, sx, sy)

    x = positions[:, 0].copy()
    positions[:, 0] = np.cos(0.2) * x
    positions[:, 1] = bmax[1] + 0.8 + np.sin(0.2) * x

    edges, rest_lengths = make_cloth_constraints(nx, ny, dx, dy)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = 0.4

    scene.update({
        'positions': positions.astype(np.float32),
        'velocities': velocities,
        'inv_masses': np.ones(
            len(positions), dtype=np.float32),

        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'spheres': spheres,

        'collision_distance': 0.03,
        'contact_margin': 0.04,
        'static_friction': 0.6,
        'kinetic_friction': 0.4,

        'substeps': 6,
        'iterations': 20,

        'cam_pos': (3.0, 3.0, -3.0),
        'cam_target': (0.0, 1.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    })

    return scene
