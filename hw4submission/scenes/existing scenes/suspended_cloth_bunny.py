# For 4.6
# uv run python main.py 4_6 scenes/hw4/suspended_cloth_bunny.py

import numpy as np
from scenes.hw4.suspended_cloth_collide import make_scene as suspended_scene
from scenes.hw4.cloth_bunny import make_scene as bunny_scene

def make_scene():
    scene = suspended_scene()
    bunny = bunny_scene()

    scene['collision_mesh'] = bunny['collision_mesh']

    # Replace the spheres with the bunny.
    scene['spheres'] = np.zeros((0, 4), dtype=np.float32)
    scene['planes'] = np.array([
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Give the suspended cloth a larger initial impulse.
    cloth_count = scene['cloth_grid_size'][0] * scene['cloth_grid_size'][1]
    scene['velocities'][:cloth_count, 1] = -2.5

    scene['substeps'] = 6
    scene['iterations'] = 24
    scene['static_friction'] = 0.5
    scene['kinetic_friction'] = 0.3

    scene['cam_pos'] = (4.0, 3.0, -4.0)
    scene['cam_target'] = (0, 1.5, 0)

    return scene
