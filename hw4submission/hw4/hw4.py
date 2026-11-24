import numpy as np
import slangpy as spy
import buffer
import window
import device
import runpy
from . import hw4_app
from . import bvh
from . import record
import math

FRAME_DT = 1.0 / 60.0

def run_hw4(app, step, args):
    output = getattr(args, 'output', None)
    if output is None:
        app.run(step)
    else:
        record.record_hw4(
            app, step, output,
            duration=getattr(args, 'duration', 5.0),
            fps=getattr(args, 'fps', 30),
            dt=FRAME_DT)

def load_scene(args):
    module = runpy.run_path(args.scene)
    scene = module['make_scene']()
    return scene

def hw4_1(args):
    # 4.1: independent Newtonian particles

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)
    shader = device.load_compute_shader(app.device, 'hw4/hw4_1.slang')

    def step(command_encoder):
        shader.kernel.dispatch(
            thread_count=[app.num_particles, 1, 1],
            command_encoder=command_encoder,
            positions=app.positions,
            velocities=app.velocities,
            num_particles=app.num_particles,
            dt=FRAME_DT)

    run_hw4(app, step, args)

def hw4_2(args):
    # 4.2: position-based dynamics rope

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)
    num_particles = app.num_particles
    substeps = scene.get('substeps', 4)
    iterations = scene.get('iterations', 10)
    dt = FRAME_DT / substeps

    # Some temporary buffers we need for the simulation
    old_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    scratch_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    rest_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['rest_lengths'], 'float')

    # The compute shaders we will use. 
    predict = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'predict')
    project = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'project_constraints')
    update = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'update_velocity')

    def step(command_encoder):
        # This function is executed every frame before the rendering.
        nonlocal scratch_positions
        for _ in range(substeps):
            # Predict position
            predict.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

            # Project constraints using Jacobi iterations
            src, dst = app.positions, scratch_positions
            for _ in range(iterations):
                project.kernel.dispatch(
                    thread_count=[num_particles, 1, 1],
                    command_encoder=command_encoder,
                    positions_in=src, positions_out=dst,
                    inv_masses=app.inv_masses, rest_lengths=rest_buffer,
                    num_particles=num_particles)
                command_encoder.global_barrier()
                src, dst = dst, src
            app.positions, scratch_positions = src, dst

            # Update velocity
            update.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

    run_hw4(app, step, args)

def hw4_3(args):
    # 4.3: position-based dynamics cloth

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)
    num_particles = app.num_particles
    substeps = scene.get('substeps', 4)
    iterations = scene.get('iterations', 10)
    dt = FRAME_DT / substeps

    old_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    scratch_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')

    predict = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'predict')
    project = device.load_compute_shader(
        app.device, 'hw4/hw4_3.slang', 'project_constraints')
    update = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'update_velocity')

    def step(command_encoder):
        nonlocal scratch_positions
        for _ in range(substeps):
            # Predict position
            predict.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

            # Project constraints using Jacobi iterations
            src, dst = app.positions, scratch_positions
            for _ in range(iterations):
                project.kernel.dispatch(
                    thread_count=[num_particles, 1, 1],
                    command_encoder=command_encoder,
                    positions_in=src, positions_out=dst,
                    inv_masses=app.inv_masses,
                    grid_size=spy.int2(scene['cloth_grid_size']),
                    spacing=scene['spacing'])
                command_encoder.global_barrier()
                src, dst = dst, src
            app.positions, scratch_positions = src, dst

            # Update velocity
            update.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

    run_hw4(app, step, args)

def build_csr(num_particles, edges, rest_lengths):
    # Build a Compressed Sparse Row (CSR) matrix from the given edges. 
    adjacency = [[] for _ in range(num_particles)]
    for (i, j), length in zip(edges, rest_lengths):
        adjacency[i].append((j, length))
        adjacency[j].append((i, length))
    row_ptr = np.zeros(num_particles + 1, dtype=np.uint32)
    for i in range(num_particles):
        row_ptr[i + 1] = row_ptr[i] + len(adjacency[i])
    size = int(row_ptr[-1])
    neighbors = np.empty(size, dtype=np.uint32)
    neighbor_rest = np.empty(size, dtype=np.float32)
    for i, incident in enumerate(adjacency):
        for k, (j, length) in enumerate(incident, start=int(row_ptr[i])):
            neighbors[k] = j
            neighbor_rest[k] = length
    return row_ptr, neighbors, neighbor_rest

def hw4_4(args):
    # 4.4: general mass-spring systems using a CSR matrix

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)
    num_particles = app.num_particles
    substeps = scene.get('substeps', 4)
    iterations = scene.get('iterations', 10)
    dt = FRAME_DT / substeps
    row_ptr, neighbors, neighbor_rest = build_csr(
        num_particles, scene['constraint_edges'], scene['rest_lengths'])

    old_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    scratch_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    row_ptr_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, row_ptr, 'uint')
    neighbor_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbors, 'uint')
    neighbor_rest_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbor_rest, 'float')

    predict = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'predict')
    solve = device.load_compute_shader(
        app.device, 'hw4/hw4_4.slang', 'project_constraints')
    update = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'update_velocity')

    def step(command_encoder):
        nonlocal scratch_positions
        for _ in range(substeps):
            # Predict position
            predict.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

            # Project constraints using Jacobi iterations
            src, dst = app.positions, scratch_positions
            for _ in range(iterations):
                solve.kernel.dispatch(
                    thread_count=[num_particles, 1, 1],
                    command_encoder=command_encoder,
                    positions_in=src, positions_out=dst,
                    inv_masses=app.inv_masses,
                    row_ptr=row_ptr_buffer, neighbors=neighbor_buffer,
                    neighbor_rest=neighbor_rest_buffer,
                    num_particles=num_particles)
                command_encoder.global_barrier()
                src, dst = dst, src
            app.positions, scratch_positions = src, dst

            # Update velocity
            update.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

    run_hw4(app, step, args)

def hw4_5(args):
    # 4.5: collision with static geometry

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)
    num_particles = app.num_particles
    substeps = scene.get('substeps', 4)
    iterations = scene.get('iterations', 10)
    dt = FRAME_DT / substeps

    row_ptr, neighbors, neighbor_rest = build_csr(
        num_particles, scene['constraint_edges'], scene['rest_lengths'])

    old_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    scratch_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    row_ptr_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, row_ptr, 'uint')
    neighbor_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbors, 'uint')
    neighbor_rest_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbor_rest, 'float')
    # One contact per particle
    contact_planes = buffer.structured_typed_buffer_from_numpy(
        app.device, np.zeros((num_particles, 4), dtype=np.float32), 'float4')

    collision_distance = float(scene['collision_distance'])
    contact_margin = float(scene['contact_margin'])

    predict = device.load_compute_shader(
        app.device, 'hw4/hw4_2.slang', 'predict')
    detect = device.load_compute_shader(
        app.device, 'hw4/hw4_5.slang', 'detect_collision')
    project = device.load_compute_shader(
        app.device, 'hw4/hw4_5.slang', 'project_constraints')
    update = device.load_compute_shader(
        app.device, 'hw4/hw4_5.slang', 'update_velocity')

    def step(command_encoder):
        nonlocal scratch_positions
        for _ in range(substeps):
            # Predict position
            predict.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

            # Generate contact
            detect.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, inv_masses=app.inv_masses,
                planes=app.planes, spheres=app.spheres,
                contact_planes=contact_planes,
                num_particles=num_particles,
                num_planes=app.num_planes,
                num_spheres=app.num_spheres,
                collision_distance=collision_distance,
                contact_margin=contact_margin)
            command_encoder.global_barrier()

            # Project constraints using Jacobi iteration
            src, dst = app.positions, scratch_positions
            for _ in range(iterations):
                project.kernel.dispatch(
                    thread_count=[num_particles, 1, 1],
                    command_encoder=command_encoder,
                    positions_in=src, positions_out=dst,
                    old_positions=old_positions,
                    inv_masses=app.inv_masses,
                    row_ptr=row_ptr_buffer, neighbors=neighbor_buffer,
                    neighbor_rest=neighbor_rest_buffer,
                    contact_planes=contact_planes,
                    num_particles=num_particles,
                    static_friction=scene['static_friction'],
                    kinetic_friction=scene['kinetic_friction'])
                command_encoder.global_barrier()
                src, dst = dst, src
            app.positions, scratch_positions = src, dst

            # Update velocity
            update.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

    run_hw4(app, step, args)

def hw4_6(args):
    # 4.6: static triangle mesh collision with stackless BVH

    scene = load_scene(args)
    app = hw4_app.HW4App(scene)

    predict = device.load_compute_shader(app.device, 'hw4/hw4_2.slang', 'predict')
    detect = device.load_compute_shader(
        app.device, 'hw4/hw4_6.slang', 'detect_collision')
    project = device.load_compute_shader(
        app.device, 'hw4/hw4_5.slang', 'project_constraints')
    update = device.load_compute_shader(
        app.device, 'hw4/hw4_5.slang', 'update_velocity')

    num_particles = app.num_particles
    substeps = scene.get('substeps', 4)
    iterations = scene.get('iterations', 10)
    dt = FRAME_DT / substeps

    row_ptr, neighbors, neighbor_rest = build_csr(
        num_particles, scene['constraint_edges'], scene['rest_lengths'])

    old_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    scratch_positions = buffer.structured_typed_buffer_from_numpy(
        app.device, scene['positions'].copy(), 'float3')
    row_ptr_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, row_ptr, 'uint')
    neighbor_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbors, 'uint')
    neighbor_rest_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, neighbor_rest, 'float')
    # One contact per particle
    contact_planes = buffer.structured_typed_buffer_from_numpy(
        app.device, np.zeros((num_particles, 4), dtype=np.float32), 'float4')

    collision_distance = float(scene['collision_distance'])
    contact_margin = float(scene['contact_margin'])

    # Collision mesh buffers & BVH building
    mesh = scene['collision_mesh']
    mesh_vertices = np.ascontiguousarray(mesh['vertices'], dtype=np.float32)
    mesh_vertex_normals = np.ascontiguousarray(mesh['vertex_normals'], dtype=np.float32)
    mesh_triangles = np.ascontiguousarray(mesh['triangles'], dtype=np.uint32)
    bvh_nodes = []
    mesh_triangles = bvh.build_bvh(mesh_vertices, mesh_triangles, bvh_nodes)
    mesh_vertices_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, mesh_vertices, 'float3')
    mesh_vertex_normals_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, mesh_vertex_normals, 'float3')
    mesh_triangles_buffer = buffer.structured_typed_buffer_from_numpy(
        app.device, mesh_triangles, 'uint3')
    bvh_nodes_buffer = buffer.create_structured_buffer(
        app.device, detect.module.BVHNode, bvh_nodes)

    # One contact per particle
    contact_planes = buffer.structured_typed_buffer_from_numpy(
        app.device, np.zeros((num_particles, 4), dtype=np.float32), 'float4')

    collision_distance = float(scene['collision_distance'])
    contact_margin = float(scene['contact_margin'])

    def step(command_encoder):
        nonlocal scratch_positions
        for _ in range(substeps):
            # Predict position
            predict.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

            # Generate contact
            detect.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, inv_masses=app.inv_masses,
                planes=app.planes, spheres=app.spheres,
                mesh_vertices=mesh_vertices_buffer,
                mesh_triangles=mesh_triangles_buffer,
                mesh_vertex_normals=mesh_vertex_normals_buffer,
                bvh_nodes=bvh_nodes_buffer,
                contact_planes=contact_planes,
                num_particles=num_particles,
                num_planes=app.num_planes,
                num_spheres=app.num_spheres,
                num_bvh_nodes=len(bvh_nodes),
                collision_distance=collision_distance,
                contact_margin=contact_margin)
            command_encoder.global_barrier()

            # Project the constraints
            src, dst = app.positions, scratch_positions
            for _ in range(iterations):
                project.kernel.dispatch(
                    thread_count=[num_particles, 1, 1],
                    command_encoder=command_encoder,
                    positions_in=src, positions_out=dst,
                    old_positions=old_positions,
                    inv_masses=app.inv_masses,
                    row_ptr=row_ptr_buffer, neighbors=neighbor_buffer,
                    neighbor_rest=neighbor_rest_buffer,
                    contact_planes=contact_planes,
                    num_particles=num_particles,
                    static_friction=scene['static_friction'],
                    kinetic_friction=scene['kinetic_friction'])
                command_encoder.global_barrier()
                src, dst = dst, src
            app.positions, scratch_positions = src, dst

            # Reconstruct velocities
            update.kernel.dispatch(
                thread_count=[num_particles, 1, 1],
                command_encoder=command_encoder,
                positions=app.positions, old_positions=old_positions,
                velocities=app.velocities, inv_masses=app.inv_masses,
                num_particles=num_particles, dt=dt)
            command_encoder.global_barrier()

    run_hw4(app, step, args)
