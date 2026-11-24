# For 4.6
# uv run python main.py 4_6 scenes/hw4/rope_bunny.py

import numpy as np
import trimesh
from pathlib import Path

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

    # Coil rope
    count = 512
    t = np.linspace(0.0, 2.0, count, dtype=np.float32)
    theta = 6.0 * np.pi * t
    radius = 0.1 + 0.6 * t
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = radius * np.cos(theta)
    positions[:, 2] = radius * np.sin(theta)
    positions[:, 1] = bunny_top + 0.3 + 1.0 * t
    positions[:, 2] += 0.1

    edges = np.stack(
        [np.arange(count - 1), np.arange(1, count)],
        axis=1).astype(np.uint32)
    rest_lengths = np.linalg.norm(
        positions[1:] - positions[:-1], axis=1).astype(np.float32)

    inv_masses = np.ones(count, dtype=np.float32)
    velocities = np.zeros_like(positions)

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': inv_masses,

        'constraint_edges': edges,
        'rest_lengths': rest_lengths,

        # For visualization
        'edges': edges,
        'show_edges': True,

        'collision_mesh': {
            'vertices': np.ascontiguousarray(vertices),
            'triangles': np.ascontiguousarray(triangles),
            'vertex_normals': np.ascontiguousarray(normals),
        },

        # Ground at bunny's feet
        'planes': np.array([
            [0, 1, 0, 0],
        ], dtype=np.float32),
        'spheres': np.zeros((0, 4), dtype=np.float32),

        'collision_distance': 0.025,
        'contact_margin': 0.05,
        'static_friction': 0.55,
        'kinetic_friction': 0.35,

        'substeps': 4,
        'iterations': 10,

        'cam_pos': (2.7, 2.4, -3.0),
        'cam_target': (0.0, 0.95, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
