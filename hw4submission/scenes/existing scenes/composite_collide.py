# For 4.5
# uv run python main.py 4_5 scenes/hw4/composite_collide.py

import numpy as np
from scenes.hw4.geometry import \
    make_cloth_grid, make_cloth_constraints, make_rope

def make_scene():
    nx, ny = 80, 60
    sx, sy = 4.0, 3.0
    rope_segments = 12

    cloth, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)

    # Move the cloth closer to the center
    cloth[:, 0] *= 0.9
    cloth[:, 2] *= 0.9
    # Put the cloth above the spheres
    cloth[:, 1] = 2.0

    cloth_edges, cloth_lengths = make_cloth_constraints(nx, ny, dx, dy)

    # Four ropes attached to the corners
    corners = [0, nx - 1, (ny - 1) * nx, nx * ny - 1]

    rope_parts = []
    rope_edges = []
    rope_lengths = []
    anchor_ids = []

    for attach_id in corners:
        attach = cloth[attach_id]
        anchor = attach + np.array([0.0, 1.5, 0.0])

        start = len(cloth) + len(rope_parts) * rope_segments
        anchor_ids.append(start)

        parts, edges, lengths = make_rope(
            anchor, attach, attach_id, start,
            rope_segments, rest_scale=0.9)

        rope_parts.append(parts)
        rope_edges.append(edges)
        rope_lengths.append(lengths)

    rope_edges = np.concatenate(rope_edges)
    rope_lengths = np.concatenate(rope_lengths)

    positions = np.concatenate([cloth, *rope_parts]).astype(np.float32)
    velocities = np.zeros_like(positions)

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[anchor_ids] = 0.0

    spheres = np.array([
        [ 0.0, 0.65,  0.00, 0.8],
        [-1.0, 0.55, -0.50, 0.6],
        [ 1.0, 0.55,  0.50, 0.6],
        [ 0.0, 0.45,  0.90, 0.4],
    ], dtype=np.float32)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': np.concatenate([cloth_edges, rope_edges]),
        'rest_lengths': np.concatenate([cloth_lengths, rope_lengths]),
        'edges': rope_edges,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'planes': np.array([
            [0, 1, 0, 0.0],
        ], dtype=np.float32),
        'spheres': spheres,

        'collision_distance': 0.02,
        'contact_margin': 0.05,
        'static_friction': 0.45,
        'kinetic_friction': 0.30,

        'substeps': 6,
        'iterations': 20,

        'cam_pos': (3.5, 4.0, -4.0),
        'cam_target': (0.0, 1.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
