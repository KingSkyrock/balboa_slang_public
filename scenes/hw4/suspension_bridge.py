# For HW 4.4
# uv run python main.py 4_4 scenes/hw4/suspension_bridge.py

import numpy as np
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

def make_scene():
    nx, ny = 30, 10
    sx, sz = 3.0, 1.0

    # Cloth "deck" of the bridge
    cloth, triangles, dx, dz = make_cloth_grid(nx, ny, sx, sz)
    cloth[:, 1] = 0.35
    cloth_edges, _ = make_cloth_constraints(nx, ny, dx, dz)

    # Two curved suspension cables, with vertical hangers
    pieces = [cloth]
    cable_edges = []
    anchor_ids = []

    for side in (0, ny - 1):
        start = len(cloth) + (len(pieces) - 1) * nx
        ids = np.arange(start, start + nx, dtype=np.uint32)

        t = np.linspace(-1, 1, nx, dtype=np.float32)
        cable = np.zeros((nx, 3), dtype=np.float32)
        cable[:, 0] = cloth[:nx, 0]
        cable[:, 1] = 1.20 + 0.65 * t * t
        cable[:, 2] = cloth[side * nx, 2]
        pieces.append(cable)

        anchor_ids.extend([ids[0], ids[-1]])

        # Adjacent cable particles
        cable_edges.extend(zip(ids[:-1], ids[1:]))

        # Hangers connecting the cable to the cloth
        for x in range(0, nx, 2):
            cable_edges.append((ids[x], side * nx + x))

    positions = np.concatenate(pieces)
    cable_edges = np.array(cable_edges, dtype=np.uint32)
    edges = np.concatenate([cloth_edges, cable_edges])

    # Rest lengths from initial positions
    rest_lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]],
        axis=1).astype(np.float32)

    # Pin the cable endpoints
    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[anchor_ids] = 0.0

    # Make the middle of the deck heavier
    t = np.arange(nx) - (nx - 1) / 2
    load = np.exp(-(t / (nx / 5)) ** 2)
    inv_masses[:len(cloth)] = np.tile(
        1.0 / (1.0 + 3.0 * load), ny)

    # Some velocities to disturb the rest state
    velocities = np.zeros_like(positions)
    u = np.tile(np.linspace(0, 1, nx), ny)
    v = np.repeat(np.linspace(-1, 1, ny), nx)
    velocities[:len(cloth), 1] = 2.2 * np.sin(np.pi * u) * v

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'edges': cable_edges,
        'triangles': triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dz),
        'substeps': 8,
        'iterations': 20,
        'planes': np.array([[0, 1, 0, -1]], dtype=np.float32),
        'cam_pos': (2.5, 2.5, -2.5),
        'cam_target': (0.0, 0.6, 0.0),
        'cam_up': (0, 1, 0),
    }
