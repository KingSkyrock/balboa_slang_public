# For 4.4
# uv run python main.py 4_4 scenes/hw4/big_hammock.py
import numpy as np
from scenes.hw4.geometry import \
    make_cloth_grid, make_cloth_constraints, make_rope

def make_scene():
    nx, ny = 256, 192
    sx, sy = 5.0, 3.8
    rope_segments = 12

    # Cloth
    cloth, triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    cloth_edges, cloth_lengths = make_cloth_constraints(nx, ny, dx, dy)

    # 10 suspension ropes
    attachments = [
        (0, 0), (nx // 3, 0), (2 * nx // 3, 0), (nx - 1, 0),
        (nx - 1, ny // 2), (nx - 1, ny - 1),
        (2 * nx // 3, ny - 1), (nx // 3, ny - 1),
        (0, ny - 1), (0, ny // 2),
    ]

    rope_parts = []
    rope_edges = []
    rope_lengths = []
    anchor_ids = []
    for u, v in attachments:
        attach_id = v * nx + u
        attach = cloth[attach_id]
        radial = attach[[0, 2]] / np.linalg.norm(attach[[0, 2]])
        anchor = attach + [0.3 * radial[0], 2.0, 0.3 * radial[1]]

        start = len(cloth) + len(rope_parts) * rope_segments
        anchor_ids.append(start)

        parts, edges, lengths = make_rope(
            anchor, attach, attach_id, start,
            rope_segments, rest_scale=0.8)

        rope_parts.append(parts)
        rope_edges.append(edges)
        rope_lengths.append(lengths)

    rope_edges = np.concatenate(rope_edges)
    rope_lengths = np.concatenate(rope_lengths)

    # Distort the cloth to create interesting motion.
    x = cloth[:, 0]
    z = cloth[:, 2]
    distort = np.cos(np.pi * x / sx) * np.cos(np.pi * z / sy)
    cloth[:, 1] += 0.25 * distort
    cloth[:, 0] += 0.25 * distort * np.sin(2 * np.pi * z / sy)

    positions = np.concatenate([cloth, *rope_parts]).astype(np.float32)
    velocities = np.zeros_like(positions)
    velocities[:len(cloth), 1] = -0.25 * distort
    velocities[:len(cloth), 2] = (
        0.25 * distort * np.sin(2 * np.pi * x / sx))

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[anchor_ids] = 0.0

    rope_edges = np.array(rope_edges, dtype=np.uint32)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': np.concatenate([cloth_edges, rope_edges]),
        'rest_lengths': np.concatenate([
            cloth_lengths, np.array(rope_lengths, dtype=np.float32)]),
        'edges': rope_edges,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),
        'substeps': 12,
        'iterations': 20,
        'planes': np.array([[0, 1, 0, -1]], dtype=np.float32),
        'cam_pos': (4.0, 3.5, -2.0),
        'cam_target': (1.0, 0.5, 1.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
