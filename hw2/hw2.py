import slangpy as spy
import device
import buffer
import window
import math
from . import hw2_scene
import time
import statistics
import numpy as np
import image
from PIL import Image
import os

def hw2_1(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_1.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    a = img.width / img.height
    def to_screen(p):
        # HW2.1: implement this
        # access "s" using args.s
        return (0.0, 0.0)

    sp0 = to_screen(args.p0)
    sp1 = to_screen(args.p1)
    sp2 = to_screen(args.p2)

    # HW2.1: implement this
    aabb_min = (0, 0)
    aabb_max = (10, 10)

    vars = {'sp0': sp0,
            'sp1': sp1,
            'sp2': sp2,
            'aabb_min': aabb_min,
            'color': args.color}

    def render(command_encoder, vars, img):
        # Clear the image to background color
        command_encoder.clear_texture_float(img, clear_value=spy.float4(0.5, 0.5, 0.5, 1.0))

        if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
            # the entire triangle is outside the screen
            return

        # Launch the kernel 
        compute_shader.kernel.dispatch(
            thread_count=[aabb_max[0] - aabb_min[0] + 1,
                          aabb_max[1] - aabb_min[1] + 1,
                          1],
            command_encoder=command_encoder,
            image=img,
            **vars,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(vars, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, vars, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw2_2(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    color_image = buffer.create_texture_2d(slang_device, args.width, args.height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, args.width, args.height, num_channels=1)

    a = args.width / args.height

    mesh = hw2_scene.parse_triangle_mesh(args.mesh)

    def to_screen(p):
        # HW2.2: implement this (basically the same as HW2.1)
        return (0.0, 0.0)

    # splat the triangles onto the screen
    screen_vertices = [to_screen(mesh.vertices[vid, :]) for vid in range(mesh.vertices.shape[0])]

    def render(command_encoder, vars, img):
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(0.5, 0.5, 0.5, 1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for tri_id in range(mesh.faces.shape[0]):
            # HW2.2: implement this
            # you need to retrieve/compute:
            # sp0, sp1, sp2 from screen_vertices
            # inv_depth0, inv_depth1, inv_depth2 from the Z coordinates
            # (recall that camera-space depth is -z, which we assume is positive)
            # aabb_min & aabb_max like in 2.1
            # c0, c1, c2 from mesh.vertex_colors
            # use mesh.faces[tri_id] to obtain the vertex indices

            if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                continue

            shade_vars = {'sp0': sp0,
                          'sp1': sp1,
                          'sp2': sp2,
                          'inv_depth0': inv_depth0,
                          'inv_depth1': inv_depth1,
                          'inv_depth2': inv_depth2,
                          'aabb_min': aabb_min,
                          'c0': c0,
                          'c1': c1,
                          'c2': c2,
                          's': args.s,
                          'a': a}
            compute_shader.kernel.dispatch(
                thread_count=[aabb_max[0] - aabb_min[0] + 1,
                              aabb_max[1] - aabb_min[1] + 1,
                              1],
                command_encoder=command_encoder,
                color_image=color_image,
                depth_image=depth_image,
                **shade_vars
            )
            # Synchronize to avoid race conditions on the depth buffer
            command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        app.run({}, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, {}, color_image)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

def hw2_3(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    color_image = buffer.create_texture_2d(slang_device, width, height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, width, height, num_channels=1)

    a = width / height

    # HW 2.3: implement this
    # change np.zeros into the right matrices
    projection = np.zeros((4, 4), dtype=np.float32)
    camera_to_screen = np.zeros((4, 4), dtype=np.float32)
    # use camera.cam_to_world
    view_matrix = np.zeros((4, 4), dtype=np.float32)
    world_to_screen = np.zeros((4, 4), dtype=np.float32)
    
    screen_vertices_list = []
    inv_depth_list = []
    for mesh in scene.meshes:
        # HW 2.3: implement this
        # change np.zeros into the right matrix
        # use mesh.model_matrix
        MVP = np.zeros((4, 4), dtype=np.float32)
        ones = np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)
        vertices = np.concatenate(
            [mesh.vertices, ones],
            axis=1
        )
        # transform vertices using MVP & homogeneous coordinates division
        # The resulting vertices should contain
        # (screen_x, screen_y, inverse_depth).
        # inverse_depth should be positive
        # ---
        # HW 2.3: your code here
        # ---
        inv_depth_list.append(np.ascontiguousarray(vertices[:, 2]))
        screen_vertices_list.append(np.ascontiguousarray(vertices[:, :2]))

    def render(command_encoder, vars, img):
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for mesh_id in range(len(scene.meshes)):
            mesh = scene.meshes[mesh_id]
            screen_vertices = screen_vertices_list[mesh_id]
            inv_depth = inv_depth_list[mesh_id]
            for tri_id in range(mesh.faces.shape[0]):
                # HW 2.3: implement this
                # (basically the same as HW 2.2)

                if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                    continue

                shade_vars = {'sp0': sp0,
                              'sp1': sp1,
                              'sp2': sp2,
                              'inv_depth0': inv_depth0,
                              'inv_depth1': inv_depth1,
                              'inv_depth2': inv_depth2,
                              'aabb_min': aabb_min,
                              'c0': c0,
                              'c1': c1,
                              'c2': c2,
                              's': camera.s,
                              'a': a}
                compute_shader.kernel.dispatch(
                    thread_count=[aabb_max[0] - aabb_min[0] + 1,
                                  aabb_max[1] - aabb_min[1] + 1,
                                  1],
                    command_encoder=command_encoder,
                    color_image=color_image,
                    depth_image=depth_image,
                    **shade_vars
                )
                # Synchronize to avoid race conditions on the depth buffer
                command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        app.run({}, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, {}, color_image)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

def interpolate_transformation(transform_keyframes, t):
    # HW 2.4: implement this
    # Find the two keyframes surrounding t and linearly
    # interpolate the corresponding transformation parameters.
    #
    # Return a list of transformations in the same format
    # as the "transform" field in the scene JSON.
    interp_transforms = []

    return interp_transforms
    
def hw2_4(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw2/hw2_2.slang')
    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    color_image = buffer.create_texture_2d(slang_device, width, height, num_channels=4)
    depth_image = buffer.create_texture_2d(slang_device, width, height, num_channels=1)

    a = width / height

    # HW 2.4 TODO: implement this (basically the same as 2.3)
    # change np.zeros into the right matrices
    projection = np.zeros((4, 4), dtype=np.float32)
    camera_to_screen = np.zeros((4, 4), dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.4 TODO: insert your code here and fill in world_to_screen
        # by using interpolate_transformation to compute the view matrix at time t
        world_to_screen = np.zeros((4, 4), dtype=np.float32)
        
        screen_vertices_list = []
        inv_depth_list = []
        for mesh in scene.meshes:
            # HW 2.4 TODO: insert your code here and fill in the MVP matrix
            # by using interpolate_transformation to compute the model matrix at time t
            MVP = np.zeros((4, 4), dtype=np.float32)
            ones = np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)
            vertices = np.concatenate(
                [mesh.vertices, ones],
                axis=1
            )
            # HW 2.4 TODO: insert your code here (basically the same as 2.3)
            inv_depth_list.append(np.ascontiguousarray(vertices[:, 2]))
            screen_vertices_list.append(np.ascontiguousarray(vertices[:, :2]))

        
        # Clear color buffer
        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )
        # Clear depth buffer
        command_encoder.clear_texture_float(
            depth_image,
            clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
        )
        for mesh_id in range(len(scene.meshes)):
            mesh = scene.meshes[mesh_id]
            screen_vertices = screen_vertices_list[mesh_id]
            inv_depth = inv_depth_list[mesh_id]
            for tri_id in range(mesh.faces.shape[0]):
                # HW 2.4 TODO: implement this
                # (basically the same as HW 2.2 & 2.3)

                if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                    continue

                shade_vars = {'sp0': sp0,
                        'sp1': sp1,
                        'sp2': sp2,
                        'inv_depth0': inv_depth0,
                        'inv_depth1': inv_depth1,
                        'inv_depth2': inv_depth2,
                        'aabb_min': aabb_min,
                        'c0': c0,
                        'c1': c1,
                        'c2': c2,
                        's': camera.s,
                        'a': a}
                compute_shader.kernel.dispatch(
                    thread_count=[aabb_max[0] - aabb_min[0] + 1,
                                  aabb_max[1] - aabb_min[1] + 1,
                                  1],
                    command_encoder=command_encoder,
                    color_image=color_image,
                    depth_image=depth_image,
                    **shade_vars
                )
                # Synchronize to avoid race conditions on the depth buffer
                command_encoder.global_barrier()

    if args.output == None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())

def hw2_5(args):
    slang_device = spy.create_device()
    transform_vertices_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'transform_vertices')
    clear_buffers_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'clear_buffers')
    rasterize_depth_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'rasterize_depth')
    rasterize_id_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'rasterize_id')
    shade_shader = \
        device.load_compute_shader(slang_device,
            'hw2/hw2_5.slang', 'shade')

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Preprocessing: merge all meshes into one list
    # so that we can launch them on GPU
    vertices_list = []
    faces_list = []
    colors_list = []
    # need to retain the offset list for animation
    vertex_offsets = []
    vertex_offset = 0
    for mesh in scene.meshes:
        vertices_list.append(mesh.vertices)
        # the indexing needs to be updated to the new array
        faces_list.append(mesh.faces + vertex_offset)
        colors_list.append(mesh.vertex_colors)
        vertex_offsets.append(vertex_offset)
        vertex_offset += len(mesh.vertices)
    vertices_host = np.concatenate(vertices_list)
    faces_host = np.concatenate(faces_list).astype(np.uint32)
    vertex_colors_host = np.concatenate(colors_list)
    num_vertices = vertices_host.shape[0]
    num_faces = faces_host.shape[0]

    # Allocate/upload buffer
    vertices = buffer.structured_typed_buffer_from_numpy(\
        slang_device, vertices_host, 'float3')
    projected_vertices = spy.Tensor.empty(
        slang_device, shape=(num_vertices,), dtype='float3')
    faces = buffer.structured_typed_buffer_from_numpy(\
        slang_device, faces_host, 'uint3')
    vertex_colors = buffer.structured_typed_buffer_from_numpy(\
        slang_device, vertex_colors_host, 'float3')
    depth_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    triangle_id_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    color_image = buffer.create_texture_2d(\
        slang_device, width, height, num_channels=4)

    # HW 2.5 TODO: implement this (basically the same as 2.3 and 2.4)
    # change np.zeros into the right matrices
    projection = np.zeros((4, 4), dtype=np.float32)
    camera_to_screen = np.zeros((4, 4), dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.5 TODO: implement this (basically the same as 2.4)
        world_to_screen = np.zeros((4, 4), dtype=np.float32)

        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1], 
                                   scene.background[2],
                                   1.0)
        )

        # Transform the vertices per mesh because each mesh
        # has a different model matrix
        for mesh, offset in zip(scene.meshes, vertex_offsets):
            # HW 2.5 TODO: implement this (basically the same as 2.4)
            MVP = np.zeros((4, 4), dtype=np.float32)
            transform_vertices_shader.kernel.dispatch(\
                thread_count=[mesh.vertices.shape[0], 1, 1],
                command_encoder=command_encoder,
                num_vertices=mesh.vertices.shape[0],
                vertex_offset=offset,
                MVP=spy.float4x4(MVP),
                vertices=vertices,
                projected_vertices=projected_vertices.storage
            )

        clear_buffers_shader.kernel.dispatch(
            thread_count=[width * height, 1, 1],
            command_encoder=command_encoder,
            num_pixels=width * height,
            depth_buffer=depth_buffer.storage,
            triangle_id_buffer=triangle_id_buffer.storage)

        command_encoder.global_barrier()

        rasterize_depth_shader.kernel.dispatch(\
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            num_triangles=num_faces,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            projected_vertices=projected_vertices.storage,
            depth_buffer=depth_buffer.storage
        )

        command_encoder.global_barrier()

        rasterize_id_shader.kernel.dispatch(\
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            num_triangles=num_faces,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            projected_vertices=projected_vertices.storage,
            depth_buffer=depth_buffer.storage,
            triangle_id_buffer=triangle_id_buffer.storage
        )

        command_encoder.global_barrier()

        shade_shader.kernel.dispatch(\
            thread_count=[width, height, 1],
            command_encoder=command_encoder,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            vertex_colors=vertex_colors,
            projected_vertices=projected_vertices.storage,
            triangle_id_buffer=triangle_id_buffer.storage,
            image=color_image)

        command_encoder.global_barrier()

    if args.output is None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())

def hw2_6(args):
    slang_device = spy.create_device()
    program = slang_device.load_program(
        "hw2/hw2_6.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Upload meshes to GPU
    gpu_meshes = []
    for mesh in scene.meshes:
        vertices = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertices, 'float3')
        vertex_colors = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_colors, 'float3')
        indices_host = np.ascontiguousarray(
            mesh.faces.reshape(-1), dtype=np.uint32)
        indices = slang_device.create_buffer(
            size=indices_host.nbytes,
            usage=spy.BufferUsage.index_buffer,
            data=indices_host
        )
        render_state = spy.RenderState({
            'viewports': [spy.Viewport.from_size(width, height)],
            'scissor_rects': [spy.ScissorRect.from_size(width, height)],
            'index_buffer': spy.BufferOffsetPair(indices),
            'index_format': spy.IndexFormat.uint32,
        })
        gpu_meshes.append({
            'vertices': vertices,
            'indices': indices,
            'vertex_colors': vertex_colors,
            'render_state': render_state
        })

    color_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=(spy.TextureUsage.render_target |
               spy.TextureUsage.shader_resource |
               spy.TextureUsage.copy_source)
    )
    depth_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.d32_float,
        usage=spy.TextureUsage.depth_stencil
    )
    color_view = color_image.create_view()
    depth_view = depth_image.create_view()

    pipeline = slang_device.create_render_pipeline(
        program=program,
        input_layout=None,
        primitive_topology=spy.PrimitiveTopology.triangle_list,
        targets=[
            spy.ColorTargetDesc({"format": color_image.format})
        ],
        depth_stencil=spy.DepthStencilDesc({
            "format": spy.Format.d32_float,
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
        }),
        rasterizer=spy.RasterizerDesc({
            "cull_mode": spy.CullMode.none
        }),
    )

    render_pass_desc = spy.RenderPassDesc({
        'color_attachments': [
            spy.RenderPassColorAttachment({
                'view': color_view,
                'load_op': spy.LoadOp.clear,
                'store_op': spy.StoreOp.store,
                'clear_value': spy.float4(\
                    scene.background[0],
                    scene.background[1], 
                    scene.background[2], 1.0),
            })
        ],
        'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
            'view': depth_view,
            'depth_load_op': spy.LoadOp.clear,
            'depth_store_op': spy.StoreOp.dont_care,
            'depth_clear_value': 1.0,
        }),
    })

    clip_projection = np.array([
        [1 / (camera.s * a), 0, 0, 0],
        [0, 1 / camera.s, 0, 0],
        [0, 0, -1, -camera.z_near],
        [0, 0, -1, 0],
    ], dtype=np.float32)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.6 TODO: compute the view matrix
        # remember to handle animation if keyframes exist
        view_matrix = np.zeros((4, 4), dtype=np.float32)

        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh in zip(scene.meshes, gpu_meshes):
            # HW 2.6 TODO: compute the model matrix
            # remember to handle animation if keyframes exist
            model_matrix = np.zeros((4, 4), dtype=np.float32)

            MVP = clip_projection @ view_matrix @ model_matrix

            root = render_pass.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root)

            cursor.find_field("MVP").write(spy.float4x4(MVP))
            cursor.find_field("vertices").write(gpu_mesh['vertices'])
            cursor.find_field("vertex_colors").write(gpu_mesh['vertex_colors'])

            render_pass.set_render_state(gpu_mesh['render_state'])
            render_pass.draw_indexed(spy.DrawArguments({
                "vertex_count": 3 * len(mesh.faces),
                "instance_count": 1,
                "start_index_location": 0,
            }))

        render_pass.end()

    if args.output is None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())

def create_Kd_texture(slang_device, material, scene_dir):
    if 'texture' in material:
        filename = os.path.join(scene_dir, material['texture'])
        rgba = np.asarray(
            Image.open(filename).convert('RGBA'),
            dtype=np.uint8)
    else:
        color = material['Kd']
        rgba = np.array(
            [[[*color, 1.0]]], dtype=np.float32)
        rgba = np.uint8(np.round(np.clip(rgba, 0, 1) * 255))

    height, width = rgba.shape[:2]

    return slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba8_unorm_srgb,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(rgba, dtype=np.uint8, copy=True, order='C').ravel()
    )

def hw2_7(args):
    slang_device = spy.create_device()
    program = slang_device.load_program(
        "hw2/hw2_7.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Upload meshes to GPU
    gpu_meshes = []
    for mesh in scene.meshes:
        vertices = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertices, 'float3')
        vertex_normals = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_normals, 'float3')
        vertex_uvs = buffer.structured_typed_buffer_from_numpy(
            slang_device, mesh.vertex_uvs, 'float2')
        indices_host = np.ascontiguousarray(
            mesh.faces.reshape(-1), dtype=np.uint32)
        indices = slang_device.create_buffer(
            size=indices_host.nbytes,
            usage=spy.BufferUsage.index_buffer,
            data=indices_host
        )

        render_state = spy.RenderState({
            'viewports': [spy.Viewport.from_size(width, height)],
            'scissor_rects': [spy.ScissorRect.from_size(width, height)],
            'index_buffer': spy.BufferOffsetPair(indices),
            'index_format': spy.IndexFormat.uint32,
        })

        gpu_meshes.append({
            'vertices': vertices,
            'indices': indices,
            'vertex_normals': vertex_normals,
            'vertex_uvs': vertex_uvs,
            'render_state': render_state
        })

    color_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=(spy.TextureUsage.render_target |
               spy.TextureUsage.shader_resource |
               spy.TextureUsage.copy_source)
    )

    depth_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.d32_float,
        usage=spy.TextureUsage.depth_stencil
    )
    color_view = color_image.create_view()
    depth_view = depth_image.create_view()

    pipeline = slang_device.create_render_pipeline(
        program=program,
        input_layout=None,
        primitive_topology=spy.PrimitiveTopology.triangle_list,
        targets=[
            spy.ColorTargetDesc({"format": color_image.format})
        ],
        depth_stencil=spy.DepthStencilDesc({
            "format": spy.Format.d32_float,
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
        }),
        rasterizer=spy.RasterizerDesc({
            "cull_mode": spy.CullMode.none
        }),
    )

    render_pass_desc = spy.RenderPassDesc({
        'color_attachments': [
            spy.RenderPassColorAttachment({
                'view': color_view,
                'load_op': spy.LoadOp.clear,
                'store_op': spy.StoreOp.store,
                'clear_value': spy.float4(\
                    scene.background[0],
                    scene.background[1], 
                    scene.background[2], 1.0),
            })
        ],
        'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
            'view': depth_view,
            'depth_load_op': spy.LoadOp.clear,
            'depth_store_op': spy.StoreOp.dont_care,
            'depth_clear_value': 1.0,
        }),
    })

    clip_projection = np.array([
        [1 / (camera.s * a), 0, 0, 0],
        [0, 1 / camera.s, 0, 0],
        [0, 0, -1, -camera.z_near],
        [0, 0, -1, 0],
    ], dtype=np.float32)

    Kd_textures = [
        create_Kd_texture(slang_device,
                          mesh.material,
                          os.path.dirname(args.scene))
        for mesh in scene.meshes
    ]

    Kd_sampler = slang_device.create_sampler()

    def render_frame(command_encoder, t):
        t = t % scene.duration
        # HW 2.7 TODO: compute cam_to_world and view matrix
        # remember to handle animation if keyframes exist
        cam_to_world = np.zeros((4, 4), dtype=np.float32)
        view_matrix = np.zeros((4, 4), dtype=np.float32)

        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh, Kd_texture in zip(scene.meshes, gpu_meshes, Kd_textures):
            # HW 2.7 TODO: compute the model matrix
            # remember to handle animation if keyframes exist
            model_matrix = np.zeros((4, 4), dtype=np.float32)

            normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
            normal_matrix = np.asarray(normal_matrix, dtype=np.float32)

            MVP = clip_projection @ view_matrix @ model_matrix

            root = render_pass.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root)

            cursor.find_field("MVP").write(spy.float4x4(MVP))
            cursor.find_field("model_matrix").write(spy.float4x4(model_matrix))
            cursor.find_field("normal_matrix").write(spy.float3x3(normal_matrix))
            cursor.find_field("vertices").write(gpu_mesh['vertices'])
            cursor.find_field("vertex_normals").write(gpu_mesh['vertex_normals'])
            cursor.find_field("vertex_uvs").write(gpu_mesh['vertex_uvs'])

            cursor.find_field("light_dir").write(spy.float3(scene.light_dir))
            cursor.find_field("light_color").write(spy.float3(scene.light_color))
            cursor.find_field("camera_position").write(
                spy.float3(cam_to_world[:3, 3]))

            cursor.find_field("Kd_texture").write(Kd_texture)
            cursor.find_field("Kd_sampler").write(Kd_sampler)
            cursor.find_field("Ks").write(spy.float3(mesh.material['Ks']))
            cursor.find_field("exponent").write(mesh.material['exponent'])

            render_pass.set_render_state(gpu_mesh['render_state'])
            render_pass.draw_indexed(spy.DrawArguments({
                "vertex_count": 3 * len(mesh.faces),
                "instance_count": 1,
                "start_index_location": 0,
            }))

        render_pass.end()

    if args.output == None:
        app = window.App(slang_device, color_image)
        vars = {'time': 0.0}
        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run(vars, render, update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        image.save_png(args.output, color_image.to_numpy())
