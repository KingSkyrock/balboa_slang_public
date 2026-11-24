# For HW 4.4
# uv run python main.py 4_4 scenes/hw4/branching_pendulum.py
import numpy as np

def make_scene():
    positions = [[0.0, 2.0, 0.0]]
    edges = []
    depths = [0]
    # pointer to the parent + offset
    frontier = [(0, 0.0)]
    lengths = [0.7, 0.5, 0.4, 0.3, 0.2]

    # create a tree of mass spring
    for level, length in enumerate(lengths):
        next_frontier = []
        for parent, heading in frontier:
            branch_count = 3 if level == 0 else 2
            for child in range(branch_count):
                if level == 0:
                    a = 2.0 * np.pi * child / 3.0
                else:
                    a = heading + (-0.6 if child == 0 else 0.6)
                v = np.array([0.7 * np.cos(a), -0.8,
                              0.7 * np.sin(a)], dtype=np.float32)
                v /= np.linalg.norm(v)
                p = np.array(positions[parent]) + length * v
                idx = len(positions)
                positions.append(p.tolist())
                edges.append((parent, idx))
                depths.append(level + 1)
                next_frontier.append((idx, a))
        frontier = next_frontier

    positions = np.array(positions, dtype=np.float32)
    edges = np.array(edges, dtype=np.uint32)
    rest_lengths = np.linalg.norm(
        positions[edges[:, 0]] - positions[edges[:, 1]], axis=1).astype(np.float32)

    inv_masses = np.ones(len(positions), dtype=np.float32)
    inv_masses[0] = 0.0
    # make things heavier at the end
    inv_masses[np.array(depths) == len(lengths)] = 0.2

    velocities = np.zeros_like(positions)
    # apply some rotation
    velocities[:, 0] = -2.5 * positions[:, 2]
    velocities[:, 2] = 2.5 * positions[:, 0]
    velocities[:, 1] = 0.7 * np.sin(4.0 * positions[:, 0])
    velocities[0] = 0.0

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,
        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'edges': edges,
        'substeps': 6,
        'iterations': 20,
        'planes': np.array([[0.0, 1.0, 0.0, -1.0]], dtype=np.float32),
        'cam_pos': (1.5, 2.5, -2.5),
        'cam_target': (0.0, 0.85, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
