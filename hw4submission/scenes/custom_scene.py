# For 4.6

import numpy as np
import trimesh
from pathlib import Path
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

def make_scene():

    # its saying "HELLO 167 :)"
    pixel_grid = [["X"," ","X"," ","X","X","X"," ","X"," "," "," ","X"," "," "," ","X","X","X",],
                  ["X"," ","X"," ","X"," "," "," ","X"," "," "," ","X"," "," "," ","X"," ","X",],
                  ["X","X","X"," ","X","X","X"," ","X"," "," "," ","X"," "," "," ","X"," ","X",],
                  ["X"," ","X"," ","X"," "," "," ","X"," "," "," ","X"," "," "," ","X"," ","X",],
                  ["X"," ","X"," ","X","X","X"," ","X","X","X"," ","X","X","X"," ","X","X","X",],
                  [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," ",],
                  ["X"," ","X","X"," "," ","X","X","X"," "," "," ","X"," ","X"," "," "," "," ",],
                  ["X"," ","X"," "," "," "," "," ","X"," "," "," ","X"," ","X"," "," "," "," ",],
                  ["X"," ","X","X","X"," "," ","X"," "," "," "," "," "," "," "," "," "," "," ",],
                  ["X"," ","X"," ","X"," "," ","X"," "," "," ","X"," "," "," ","X"," "," "," ",],
                  ["X"," ","X","X","X"," "," ","X"," "," "," "," ","X","X","X"," "," "," "," ",],
                  [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," ",]]

    bear = trimesh.load_mesh(
        Path(__file__).with_name('custom_assets') / 'bear-sp22.ply',
        process=False)
    bear_vertices = np.asarray(
        bear.vertices, dtype=np.float32).copy()
    bear_triangles = np.asarray(
        bear.faces, dtype=np.uint32)
    bear_normals = np.asarray(
        bear.vertex_normals, dtype=np.float32).copy()

    bmin = bear_vertices.min(axis=0)
    bmax = bear_vertices.max(axis=0)
    bear_vertices -= (bmin + bmax) / 2
    bear_vertices *= 1.6 / (bmax[1] - bmin[1])
    bear_vertices[:, 1] -= bear_vertices[:, 1].min()
    bear_vertices[:, [0, 2]] *= -1
    bear_normals[:, [0, 2]] *= -1
    bear_top = bear_vertices[:, 1].max()

    offsets = np.array([
        [-2, 0, -2],
        [ 2, 0, -2],
        [-2, 0,  2],
        [ 2, 0,  2],
    ], dtype=np.float32)
    small_scale = 0.5
    num_vertices = len(bear_vertices)
    vertices = np.concatenate(
        [bear_vertices] +
        [bear_vertices * small_scale + o for o in offsets])
    triangles = np.concatenate([
        bear_triangles + i * num_vertices
        for i in range(len(offsets) + 1)
    ]).astype(np.uint32)
    normals = np.tile(bear_normals, (len(offsets) + 1, 1))

    nx, ny = 120, 120
    sx, sy = 4.6, 4.6
    positions, cloth_triangles, dx, dy = make_cloth_grid(
        nx, ny, sx, sy)
    positions[:, 1] = bear_top + 15
    edges, rest_lengths = make_cloth_constraints(
        nx, ny, dx, dy)

    cam_pos = (3.6, 4.2, -3.6)
    cam_target = (0.0, 0.0, 0.0)
    cam_up = (0.0, 1.0, 0.0)

    forward = np.subtract(cam_target, cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    cell = 0.17
    segments = 4
    text_center = np.array([0.0, 2.5, 0.0])

    # row and column coordinates
    pixels = [(r, c) for r, row in enumerate(pixel_grid)
              for c, p in enumerate(row) if p == "X"]
    #print(pixels)
    mid_r = (min(r for r, c in pixels) + max(r for r, c in pixels)) / 2
    mid_c = (min(c for r, c in pixels) + max(c for r, c in pixels)) / 2

    ids = {}
    string_positions = []
    for r, c in pixels:
        ids[(r, c)] = len(string_positions)
        string_positions.append(
            text_center + cell * ((c - mid_c) * right - (r - mid_r) * up))

    string_edges = []
    for r, c in pixels:
        # only add diagonal if no other neighbor
        neighbors = [(r, c + 1), (r + 1, c)]
        for dc in [-1, 1]:
            if (r, c + dc) not in ids and (r + 1, c) not in ids:
                neighbors.append((r + 1, c + dc))
        print(neighbors)
        for n in neighbors:
            if n not in ids:
                continue
            prev = ids[(r, c)]
            for k in range(1, segments):
                t = k / segments
                string_positions.append(
                    (1 - t) * string_positions[ids[(r, c)]] +
                    t * string_positions[ids[n]]
                )
                string_edges.append((prev, len(string_positions) - 1))
                prev = len(string_positions) - 1
            string_edges.append((prev, ids[n]))

    string_positions = np.array(string_positions, dtype=np.float32)
    string_edges = np.array(string_edges, dtype=np.uint32)
    string_rest = np.linalg.norm(
        string_positions[string_edges[:, 0]] -
        string_positions[string_edges[:, 1]], axis=1
    ).astype(np.float32)

    string_edges = string_edges + np.uint32(len(positions))
    positions = np.concatenate([positions, string_positions])
    edges = np.concatenate([edges, string_edges])
    rest_lengths = np.concatenate([rest_lengths, string_rest])

    inv_masses = np.ones(
        len(positions), dtype=np.float32
    )

    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,

        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'triangles': cloth_triangles,
        'edges': string_edges,
        'cloth_grid_width': nx,
        'cloth_grid_height': ny,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'collision_mesh': {
            'vertices': np.ascontiguousarray(vertices),
            'triangles': np.ascontiguousarray(triangles),
            'vertex_normals': np.ascontiguousarray(normals),
        },

        'planes': np.array([
            [0, 1, 0, 0],
        ], dtype=np.float32),
        'spheres': np.zeros((0, 4), dtype=np.float32),

        'collision_distance': 0.025,
        'contact_margin': 0.05,
        'static_friction': 0.55,
        'kinetic_friction': 0.35,

        'substeps': 12,
        'iterations': 40,

        'cam_pos': cam_pos,
        'cam_target': cam_target,
        'cam_up': cam_up,
    }
