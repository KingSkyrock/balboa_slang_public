import numpy as np
import slangpy as spy
import buffer
import device
import window
import image
import functools
from . import hw1_scene

def make_render_callback(kernel):
    def render(command_encoder, vars, image):
        kernel.dispatch(
            thread_count=[image.width, image.height, 1],
            command_encoder=command_encoder,
            image=image,
            **vars,
        )
    return render

def hw1_1(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_1.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    vars = {'center': args.center,
            'radius': args.radius,
            'color': args.color}

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(vars, make_render_callback(compute_shader.kernel))
    else:
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **vars)
        image.save_png(args.output, img.to_numpy())

def hw1_2(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_2.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    points_np = np.asarray(args.points, dtype=np.float32).reshape(-1, 2)
    polyline = buffer.structured_buffer_from_numpy(slang_device, points_np)

    vars = {'polyline': polyline,
            'num_polyline': points_np.shape[0],
            'is_closed': args.closed,
            'use_fill_color': args.fill_color is not None,
            'fill_color': args.fill_color if args.fill_color is not None else [0.0, 0.0, 0.0],
            'use_stroke_color': args.stroke_color is not None,
            'stroke_color': args.stroke_color if args.stroke_color is not None else [0.0, 0.0, 0.0],
            'stroke_width': args.stroke_width}

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(vars, make_render_callback(compute_shader.kernel))
    else:
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **vars)
        image.save_png(args.output, img.to_numpy())    

def hw1_3(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_3.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device)
    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, make_render_callback(compute_shader.kernel))
    else:
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())

def hw1_4(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_4.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device)
    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, make_render_callback(compute_shader.kernel))
    else:
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())

def hw1_5(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_4.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    if args.output == None:
        app = window.App(slang_device, img)
        def update(t, bindings):
            # The following is slightly inefficient: 
            # we reupload the scene regardless of whether things have changed or not.
            # Can you think of a better way to optimize this?
            bindings = hw1_scene.upload_scene(scene,
                                              compute_shader.module,
                                              slang_device,
                                              t % scene.duration)
            return bindings
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=0.0)
        app.run(bindings, make_render_callback(compute_shader.kernel), update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=t)
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())

def hw1_6(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_6.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    if args.output == None:
        app = window.App(slang_device, img)
        def update(t, bindings):
            # The following is slightly inefficient: 
            # we reupload the scene regardless of whether things have changed or not.
            # Can you think of a better way to optimize this?
            bindings = hw1_scene.upload_scene(scene,
                                              compute_shader.module,
                                              slang_device,
                                              t % scene.duration)
            return bindings
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=0.0)
        app.run(bindings, make_render_callback(compute_shader.kernel), update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=t)
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())

def hw1_7(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_7.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    if args.output == None:
        app = window.App(slang_device, img)
        def update(t, bindings):
            # The following is slightly inefficient: 
            # we reupload the scene regardless of whether things have changed or not.
            # Can you think of a better way to optimize this?
            bindings = hw1_scene.upload_scene(scene,
                                              compute_shader.module,
                                              slang_device,
                                              t % scene.duration)
            return bindings
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=0.0)
        app.run(bindings, make_render_callback(compute_shader.kernel), update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=t)
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())

def hw1_8(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw1/hw1_8.slang')
    scene = hw1_scene.parse_scene(args.scene)
    img = buffer.create_texture_2d(slang_device, scene.resolution[0], scene.resolution[1])
    if args.output == None:
        app = window.App(slang_device, img)
        def update(t, bindings):
            # The following is slightly inefficient: 
            # we reupload the scene regardless of whether things have changed or not.
            # Can you think of a better way to optimize this?
            bindings = hw1_scene.upload_scene(scene,
                                              compute_shader.module,
                                              slang_device,
                                              t % scene.duration)
            return bindings
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=0.0)
        app.run(bindings, make_render_callback(compute_shader.kernel), update)
    else:
        t = args.time % scene.duration if hasattr(args, 'time') else 0.0
        bindings = hw1_scene.upload_scene(scene, compute_shader.module, slang_device, t=t)
        compute_shader.kernel.dispatch(
            thread_count=[img.width, img.height, 1],
            image=img,
            **bindings)
        image.save_png(args.output, img.to_numpy())
