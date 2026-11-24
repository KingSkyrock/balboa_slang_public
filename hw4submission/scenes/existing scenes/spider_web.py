# For 4.4
# uv run python main.py 4_4 scenes/hw4/spider_web.py
# inspired by Shadertoy user middle: https://www.shadertoy.com/view/4fc3zf
import numpy as np

def make_scene():
    spokes = 24
    rings = 9
    radius = 1.35

    # Construct a concentric ring
    positions = [[0.0, 0.65, 0.0]]
    for r in range(1, rings + 1):
        rho = radius * r / rings
        for s in range(spokes):
            a = 2.0 * np.pi * s / spokes
            positions.append([
                rho * np.cos(a),
                0.65 + rho * np.sin(a),
                0.0
            ])
    positions = np.array(positions, dtype=np.float32)

    def index(r, s):
        return 0 if r == 0 else 1 + (r - 1) * spokes + s % spokes

    # Constraints
    edges = []
    for r in range(1, rings + 1):
        for s in range(spokes):
            edges.append((index(r - 1, s), index(r, s)))
            edges.append((index(r, s), index(r, s + 1)))
    edges = np.array(edges, dtype=np.uint32)

    # Compute desired rest lengths from the ring
    rest_lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]],
        axis=1).astype(np.float32)

    # Deform the positions to move away from the rest position
    # Do a little bit of twisting
    positions[0] += [0.25, 0.15, -1.5]
    for r in range(1, rings):
        taper = (1.0 - r / rings) ** 2.0
        for s in range(spokes):
            a = 2.0 * np.pi * s / spokes
            i = index(r, s)
            positions[i, 0] += 0.2 * taper
            positions[i, 1] += 0.2 * taper
            positions[i, 2] -= 1.5 * taper * (
                0.8 + 0.2 * np.cos(3.0 * a))

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[index(rings, 0):] = 0.0

    return {
        'positions': positions,
        'velocities': np.zeros_like(positions),
        'inv_masses': inv_masses,
        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'edges': edges,
        'substeps': 8,
        'iterations': 24,
        'planes': np.array([[0, 1, 0, -1]], dtype=np.float32),
        'cam_pos': (1.2, 1.5, -3.0),
        'cam_target': (0.0, 0.65, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
