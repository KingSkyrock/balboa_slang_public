# For HW 4.5
# uv run python main.py 4_5 scenes/hw4/suspended_cloth_collide.py

import numpy as np
from scenes.hw4.geometry import (
    make_cloth_grid,
    make_cloth_constraints,
)

def make_scene():
    # Large horizontal cloth
    nx, ny = 80, 60
    sx, sy = 4.0, 3.0
    cloth, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    cloth[:, 1] = 2.6
    cloth_count = len(cloth)
    cloth_edges, cloth_rest = make_cloth_constraints(
        nx, ny, dx, dy)

    # Two fixed anchors above the cloth
    anchors = np.array([
        [-1.0, 4.4, 0.0],
        [ 1.0, 4.4, 0.0],
    ], dtype=np.float32)

    # Four attachment vertices near the cloth's center
    cx, cy = nx // 2, ny // 2
    attachments = [
        (cx - 7, cy - 5, 0),
        (cx - 7, cy + 5, 0),
        (cx + 7, cy - 5, 1),
        (cx + 7, cy + 5, 1),
    ]

    # Attach ropes to the cloth
    rope_segments = 16
    rope_parts = []
    rope_edges = []
    rope_rest = []
    anchor_ids = [cloth_count, cloth_count + 1]
    for x, y, anchor_index in attachments:
        attach_id = y * nx + x
        anchor_id = anchor_ids[anchor_index]

        a = anchors[anchor_index]
        b = cloth[attach_id]

        t = (np.arange(1, rope_segments, dtype=np.float32)
             / rope_segments)
        rope = ((1 - t[:, None]) * a
                + t[:, None] * b).astype(np.float32)

        start = cloth_count + 2 + len(rope_parts) * (rope_segments - 1)
        rope_parts.append(rope)

        ids = ([anchor_id]
            + list(range(start, start + rope_segments - 1))
            + [attach_id])

        segment_length = np.linalg.norm(b - a) / rope_segments

        for i, j in zip(ids[:-1], ids[1:]):
            rope_edges.append((i, j))
            rope_rest.append(segment_length)

    positions = np.ascontiguousarray(
        np.concatenate([cloth, anchors, *rope_parts]),
        dtype=np.float32)

    rope_edges = np.array(rope_edges, dtype=np.uint32)
    constraints = np.concatenate(
        [cloth_edges, rope_edges], axis=0)
    rest_lengths = np.concatenate([
        cloth_rest,
        np.array(rope_rest, dtype=np.float32),
    ])

    # Pinn the anchors
    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[anchor_ids] = 0.0

    # Downward initial velocity
    velocities = np.zeros_like(positions)
    velocities[:cloth_count, 1] = -1.0

    # Colliders
    spheres = np.array([
        [-1.2, 1.0, -0.5, 0.8],
        [ 1.2, 1.0,  0.5, 0.8],
        [ 0.0, 0.8,  1.0, 0.6],
        [ 0.0, 0.8, -1.0, 0.6],
    ], dtype=np.float32)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,

        'constraint_edges': constraints,
        'rest_lengths': rest_lengths,

        'triangles': triangles,
        'edges': rope_edges,
        'show_edges': True,
        'cloth_grid_width': nx,
        'cloth_grid_height': ny,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'planes': np.array([
            [0, 1, 0, -0.15],
        ], dtype=np.float32),
        'spheres': spheres,

        'collision_distance': 0.02,
        'contact_margin': 0.05,
        'static_friction': 0.35,
        'kinetic_friction': 0.20,

        'substeps': 8,
        'iterations': 20,

        'cam_pos': (2.5, 4.0, -3.5),
        'cam_target': (0.0, 2.0, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
