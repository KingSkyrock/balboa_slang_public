import numpy as np
import slangpy as spy
import buffer
import window
import device

particle_radius = 3.0
width = 800
height = 600

def make_collider_mesh(planes, spheres,
                       collision_mesh=None,
                       plane_size=100.0,
                       sphere_stacks=12,
                       sphere_slices=24):
    """
    Given the plane & sphere & mesh colliders,
    make a mesh out of all of them. Returns a list of
    vertices, normals, and albedos (Kd).
    """

    vertices = []
    normals = []
    albedos = []

    def triangle(a, b, c, na, nb, nc, color):
        vertices.extend([a, b, c])
        normals.extend([na, nb, nc])
        albedos.extend([color, color, color])

    # Plane convention: dot(n, x) = d, with unit n
    # Render infinite collision planes as large finite quads
    if planes is not None:
        for plane in planes:
            n = np.asarray(plane[:3], dtype=np.float32)
            d = float(plane[3])
            length = np.linalg.norm(n)
            center = n * (d / (length * length))
            n = n / length

            # Build an orthogonal frame
            tmp = (np.array([1.0, 0.0, 0.0], dtype=np.float32)
                   if abs(n[1]) > 0.9 else
                   np.array([0.0, 1.0, 0.0], dtype=np.float32))
            u = np.cross(tmp, n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)

            a = center - plane_size * u - plane_size * v
            b = center + plane_size * u - plane_size * v
            c = center + plane_size * u + plane_size * v
            d = center - plane_size * u + plane_size * v
            gray = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            triangle(a, b, c, n, n, n, gray)
            triangle(a, c, d, n, n, n, gray)

    # Smooth-shaded UV sphere meshes
    def sphere_normal(theta, phi):
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.cos(theta),
            np.sin(theta) * np.sin(phi),
        ], dtype=np.float32)

    if spheres is not None:
        for sphere in spheres:
            center = np.asarray(sphere[:3], dtype=np.float32)
            radius = float(sphere[3])
            gray = np.array([0.7, 0.5, 0.5], dtype=np.float32)
            for y in range(sphere_stacks):
                t0 = np.pi * y / sphere_stacks
                t1 = np.pi * (y + 1) / sphere_stacks
                for x in range(sphere_slices):
                    p0 = 2.0 * np.pi * x / sphere_slices
                    p1 = 2.0 * np.pi * (x + 1) / sphere_slices
                    n00 = sphere_normal(t0, p0)
                    n01 = sphere_normal(t0, p1)
                    n10 = sphere_normal(t1, p0)
                    n11 = sphere_normal(t1, p1)
                    triangle(center + radius * n00,
                             center + radius * n01,
                             center + radius * n11,
                             n00, n01, n11, gray)
                    triangle(center + radius * n00,
                             center + radius * n11,
                             center + radius * n10,
                             n00, n11, n10, gray)

    # Static triangle meshes
    if collision_mesh is not None:
        mesh_vertices = collision_mesh['vertices']
        mesh_triangles = collision_mesh['triangles']
        red = np.array([0.7, 0.5, 0.5], dtype=np.float32)
        vns = collision_mesh['vertex_normals']
        for i, j, k in mesh_triangles:
            a, b, c = mesh_vertices[i], mesh_vertices[j], mesh_vertices[k]
            triangle(a, b, c, vns[i], vns[j], vns[k], red)

    return np.array(vertices, dtype=np.float32), \
           np.array(normals, dtype=np.float32), \
           np.array(albedos, dtype=np.float32)

class HW4App:
    def __init__(self, scene):
        """
        The scene is a dictionary that contains:
        positions -- a Nx3 float numpy array containing all initial positions of particles
        velocities -- a Nx3 float numpy array containing all initial velocities of particles
        inv_masses -- a Nx1 float numpy array that contains 1/mass for each particle.
        edges -- a Mx2 int numpy array containing which two particles have an edge
                 purely for visualization. This is mostly for the springs/ropes.
        triangles -- a Tx3 int numpy array containing which 3 particles form a triangle
                     purely for visualization. This is mostly for the clothes.
        cloth_grid_size -- a 2-item tuple (w, h) that specifies the size of the cloth.
                           we only allow 1 cloth in the scene. 
                           Must be specified if you are simulating/rendering a cloth.
        spacing -- a 2-item tuple (sx, sy) that specifies how far away each node
                   is from each other in a cloth. 
                   Must be specified if you are simulating/rendering a cloth.
        substeps -- number of timesteps taken per frame
        iterations -- number of iterations of the Jacobi iteration
        show_particles -- a boolean flag for visualizing the particles.
                          mostly use for 4.1 and debugging.
        planes -- static plane colliders. They are stored in a Px4 numpy arrays
                  where P is the number of planes. Each plane is stored in 4 numbers
                  (a, b, c, d) that represents the plane ax+by+cx=d.
                  During simulation, these planes are infinite planes.
                  But during visualization we render them as large finite quads.
                  We only use it for visualization here,
                  but your simulation code should account for them (4.5 & 4.6).
        spheres -- static sphere colliders. They are stored in a Sx4 numpy arrays
                   where S is the number of spheres. Each sphere is stored in 4 numbers
                   (x, y, z, r) where (x, y, z) is the center and r is the radius.
                   We only use it for visualization here,
                   but your simulation code should account for them (4.5 & 4.6).
        collision_mesh -- a dictionary with entires 'vertices' and 'triangles'
                          that represents a mesh
        cam_pos -- a 3-item tuple of the position of the camera (default 0, 0, 0)
        cam_target -- a 3-item tuple of the tatget of the camera (default 0, 0, -1)
        cam_up -- a 3-item tuple of the up vector of the camera (default 0, 1, 0)
        """
        self.device = spy.create_device()

        # Upload things to GPU
        host_positions = np.ascontiguousarray(
            scene['positions'], dtype=np.float32)
        self.num_particles = len(host_positions)
        self.positions = buffer.structured_typed_buffer_from_numpy(
            self.device, host_positions, 'float3'
        )
        if 'velocities' in scene:
            host_velocities = np.ascontiguousarray(
                scene['velocities'], dtype=np.float32)
            self.velocities = buffer.structured_typed_buffer_from_numpy(
                self.device, host_velocities, 'float3'
            )
        else:
            # Initialize the velocities to zero
            self.velocities = buffer.structured_typed_buffer_from_numpy(
                self.device, np.zeros_like(host_positions), 'float3'
            )
        if 'inv_masses' in scene:
            host_inv_masses = np.ascontiguousarray(
                scene['inv_masses'], dtype=np.float32)
            self.inv_masses = buffer.structured_typed_buffer_from_numpy(
                self.device, host_inv_masses, 'float3'
            )
        else:
            # Initialize the masses to 1
            self.inv_masses = buffer.structured_typed_buffer_from_numpy(
                self.device, np.ones_like(host_positions), 'float3'
            )

        self.num_edges = 0
        self.edges = None
        if 'edges' in scene:
            host_edges = np.ascontiguousarray(scene['edges'], dtype=np.uint32)
            self.num_edges = len(host_edges)
            self.edges = buffer.structured_typed_buffer_from_numpy(
                self.device, host_edges, 'uint2'
            )
        self.num_triangles = 0
        self.triangles = None
        if 'triangles' in scene:
            host_triangles = np.ascontiguousarray(scene['triangles'], dtype=np.uint32)
            self.num_triangles = len(host_triangles)
            self.triangles = buffer.structured_typed_buffer_from_numpy(
                self.device, host_triangles, 'uint3')
        self.cloth_grid_size = scene.get('cloth_grid_size', None)

        # The static collider geometry
        host_planes = None
        self.num_planes = 0
        if 'planes' in scene:
            host_planes = np.ascontiguousarray(scene['planes'], dtype=np.float32)
            self.num_planes = len(host_planes)
            self.planes = buffer.structured_typed_buffer_from_numpy(
                self.device, host_planes, 'float4')
        host_spheres = None
        self.num_spheres = 0
        if 'spheres' in scene:
            host_spheres = np.ascontiguousarray(scene['spheres'], dtype=np.float32)
            self.num_spheres = len(host_spheres)
            self.spheres = buffer.structured_typed_buffer_from_numpy(
                self.device, host_spheres, 'float4')
        static_positions, static_normals, static_albedos = \
            make_collider_mesh(host_planes, host_spheres,
                scene.get('collision_mesh', None))
        self.num_static_vertices = len(static_positions)
        self.static_positions = None
        self.static_normals = None
        self.static_albedos = None
        if self.num_static_vertices > 0:
            self.static_positions = buffer.structured_typed_buffer_from_numpy(
                self.device, static_positions, 'float3')
            self.static_normals = buffer.structured_typed_buffer_from_numpy(
                self.device, static_normals, 'float3')
            self.static_albedos = buffer.structured_typed_buffer_from_numpy(
                self.device, static_albedos, 'float3')

        self.color_image = self.device.create_texture(
            width=width,
            height=height,
            format=spy.Format.rgba32_float,
            usage=(spy.TextureUsage.render_target |
                   spy.TextureUsage.shader_resource |
                   spy.TextureUsage.copy_source),
        )
        self.depth_image = self.device.create_texture(
            width=width,
            height=height,
            format=spy.Format.d32_float,
            usage=spy.TextureUsage.depth_stencil,
        )
        self.color_view = self.color_image.create_view()
        self.depth_view = self.depth_image.create_view()

        targets = [spy.ColorTargetDesc({'format': self.color_image.format})]
        depth_stencil = spy.DepthStencilDesc({
            'format': spy.Format.d32_float,
            'depth_test_enable': True,
            'depth_write_enable': True,
            'depth_func': spy.ComparisonFunc.less,
        })
        rasterizer = spy.RasterizerDesc({'cull_mode': spy.CullMode.none})

        self.show_particles = scene.get('show_particles', False)

        def pipeline(vertex_name, fragment_name, topology):
            program = self.device.load_program(
                'hw4/hw4_viewer.slang', [vertex_name, fragment_name])
            return self.device.create_render_pipeline(
                program=program, input_layout=None,
                primitive_topology=topology, targets=targets,
                depth_stencil=depth_stencil, rasterizer=rasterizer)

        self.static_pipeline = (
            pipeline('static_vertex_main', 'surface_fragment_main',
                     spy.PrimitiveTopology.triangle_list)
            if self.num_static_vertices > 0 else None)
        self.particle_pipeline = (
            pipeline('particle_vertex_main', 'particle_fragment_main',
                     spy.PrimitiveTopology.triangle_list)
            if self.show_particles else None)
        self.line_pipeline = (
            pipeline('line_vertex_main', 'line_fragment_main',
                     spy.PrimitiveTopology.line_list)
            if self.num_edges > 0 else None)
        self.cloth_pipeline = (
            pipeline('cloth_vertex_main', 'surface_fragment_main',
                     spy.PrimitiveTopology.triangle_list)
            if self.num_triangles > 0 else None)

        self.render_pass_desc = spy.RenderPassDesc({
            'color_attachments': [spy.RenderPassColorAttachment({
                'view': self.color_view,
                'load_op': spy.LoadOp.clear,
                'store_op': spy.StoreOp.store,
                'clear_value': spy.float4(0.2, 0.2, 0.5, 1.0),
            })],
            'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
                'view': self.depth_view,
                'depth_load_op': spy.LoadOp.clear,
                'depth_store_op': spy.StoreOp.dont_care,
                'depth_clear_value': 1.0,
            }),
        })

        s = 0.6
        aspect = width / height
        z_near = 0.1
        projection = np.array([
            [1.0 / (s * aspect), 0.0, 0.0, 0.0],
            [0.0, 1.0 / s, 0.0, 0.0],
            [0.0, 0.0, -1.0, -z_near],
            [0.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)

        cam_pos = scene.get('cam_pos', (0, 0, 0))
        cam_pos = np.array(cam_pos, dtype=np.float32)
        cam_target = scene.get('cam_target', (0, 0, -1))
        cam_target = np.array(cam_target, dtype=np.float32)
        cam_up = scene.get('cam_up', (0, 1, 0))
        cam_up = np.array(cam_up, dtype=np.float32)

        z = cam_pos - cam_target
        z /= np.linalg.norm(z)
        x = np.cross(cam_up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        # Direct inverse of the camera to world matrix
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = z
        view[:3, 3] = -view[:3, :3] @ cam_pos

        self.MVP = spy.float4x4(projection @ view)

        self.render_state = spy.RenderState({
            'viewports': [spy.Viewport.from_size(width, height)],
            'scissor_rects': [spy.ScissorRect.from_size(width, height)],
        })

    def draw(self, command_encoder):
        render_pass = command_encoder.begin_render_pass(self.render_pass_desc)

        def bind(pipeline):
            root = render_pass.bind_pipeline(pipeline)
            cursor = spy.ShaderCursor(root)
            cursor.find_field('MVP').write(self.MVP)
            return cursor

        render_pass.set_render_state(self.render_state)

        # Draw the static geometry
        if self.static_pipeline is not None:
            cursor = bind(self.static_pipeline)
            cursor.find_field('static_positions').write(self.static_positions)
            cursor.find_field('static_normals').write(self.static_normals)
            cursor.find_field('static_albedos').write(self.static_albedos)
            render_pass.draw(spy.DrawArguments({
                'vertex_count': self.num_static_vertices,
                'instance_count': 1,
            }))

        # Draw the cloth
        if self.cloth_pipeline is not None:
            cursor = bind(self.cloth_pipeline)
            cursor.find_field('positions').write(self.positions)
            cursor.find_field('triangles').write(self.triangles)
            cursor.find_field('cloth_grid_width').write(self.cloth_grid_size[0])
            cursor.find_field('cloth_grid_height').write(self.cloth_grid_size[1])
            render_pass.draw(spy.DrawArguments({
                'vertex_count': 3 * self.num_triangles,
                'instance_count': 1,
            }))

        # Draw the lines
        if self.line_pipeline is not None:
            cursor = bind(self.line_pipeline)
            cursor.find_field('positions').write(self.positions)
            cursor.find_field('edges').write(self.edges)
            render_pass.draw(spy.DrawArguments({
                'vertex_count': 2, 'instance_count': self.num_edges,
            }))

        # Draw the particles
        if self.particle_pipeline is not None:
            cursor = bind(self.particle_pipeline)
            cursor.find_field('positions').write(self.positions)
            cursor.find_field('viewport_size').write(spy.float2(width, height))
            cursor.find_field('particle_radius').write(particle_radius)
            render_pass.draw(spy.DrawArguments({
                'vertex_count': 6, 'instance_count': self.num_particles,
            }))

        render_pass.end()

    def run(self, step=None):
        running = False
        viewer = window.App(self.device, self.color_image)

        def on_keyboard_event(event):
            nonlocal running
            if event.key == spy.KeyCode.space and event.is_key_press():
                running = not running

        viewer.window.on_keyboard_event = on_keyboard_event

        def render(command_encoder, vars, img):
            nonlocal running
            if running and step is not None:
                step(command_encoder)
            self.draw(command_encoder)

        viewer.run({}, render)
