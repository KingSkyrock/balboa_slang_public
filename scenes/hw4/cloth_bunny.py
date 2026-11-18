# For 4.6
# uv run python main.py 4_6 scenes/hw4/cloth_bunny.py

import numpy as np
import trimesh
from pathlib import Path
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

def make_scene():
    mesh = trimesh.load_mesh(
        Path(__file__).with_name('bunny.ply'),
        process=False)

    vertices = np.asarray(
        mesh.vertices, dtype=np.float32).copy()
    triangles = np.asarray(
        mesh.faces, dtype=np.uint32)
    normals = np.asarray(
        mesh.vertex_normals, dtype=np.float32).copy()

    # Normalize bunny height to size ~= 1.6
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    scale = 1.6 / (bmax[1] - bmin[1])
    vertices -= (bmin + bmax) / 2
    vertices *= scale
    vertices[:, 1] -= vertices[:, 1].min()
    bunny_top = vertices[:, 1].max()
    # Make the bunny face us
    vertices[:, 0] *= -1
    vertices[:, 2] *= -1
    normals[:, 0] *= -1
    normals[:, 2] *= -1

    nx, ny = 80, 80
    sx, sy = 3.0, 3.0
    positions, cloth_triangles, dx, dy = make_cloth_grid(
        nx, ny, sx, sy)
    positions[:, 1] = bunny_top + 0.45
    edges, rest_lengths = make_cloth_constraints(
        nx, ny, dx, dy)
    inv_masses = np.ones(
        len(positions), dtype=np.float32)

    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,

        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'triangles': cloth_triangles,
        'cloth_grid_width': nx,
        'cloth_grid_height': ny,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'collision_mesh': {
            'vertices': np.ascontiguousarray(vertices),
            'triangles': np.ascontiguousarray(triangles),
            'vertex_normals': np.ascontiguousarray(normals),
        },

        # Ground at bunny's feet.
        'planes': np.array([
            [0, 1, 0, 0],
        ], dtype=np.float32),
        'spheres': np.zeros((0, 4), dtype=np.float32),

        'collision_distance': 0.025,
        'contact_margin': 0.05,
        'static_friction': 0.55,
        'kinetic_friction': 0.35,

        'substeps': 6,
        'iterations': 20,

        'cam_pos': (2.5, 2.5, -2.5),
        'cam_target': (0.0, 0.85, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
