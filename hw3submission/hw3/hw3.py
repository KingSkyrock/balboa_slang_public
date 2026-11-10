import slangpy as spy
import device
import buffer
import window
import math
import time
import statistics
import numpy as np
import image
from PIL import Image
import os
from . import hw3_scene
import json

def hw3_1(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_1.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    with open(args.scene) as f:
        scene = json.load(f)

    background = scene.get('background', (0.5, 0.5, 0.5))

    spheres = [
        {'center': obj['center'], 'radius': obj['radius']}
        for obj in scene['objects']
        if obj['type'] == 'sphere'
    ]

    bindings = {
        'background': spy.float3(background),
        'spheres': buffer.create_structured_buffer(
            slang_device, compute_shader.module.Sphere, spheres),
        'num_spheres': len(spheres)
    }

    def render(command_encoder, bindings, img):
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw3_2(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_2.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    bindings, _, _ = hw3_scene.upload_scene(
        args.scene, compute_shader.module, slang_device)

    def render(command_encoder, bindings, img):
        # Launch the kernel 
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw3_3(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_3.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    bindings, _, _ = hw3_scene.upload_scene(
        args.scene, compute_shader.module, slang_device)

    def render(command_encoder, bindings, img):
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw3_4(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_4.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    bindings, scene, shapes = hw3_scene.upload_scene(
        args.scene, compute_shader.module, slang_device)

    def update_scene(t,
                     bindings,
                     scene,
                     shapes,
                     module,
                     slang_device):
        # Update the bindings & shapes in-place
        # the caller is responsible for passing the arrays/dictionaries

        # Interpolate the animation
        for i, obj in enumerate(scene['objects']):
            M = hw3_scene.get_transformation(obj, t)
            shapes[i]['obj_to_world'] = M
            shapes[i]['world_to_obj'] = np.linalg.inv(M)
        # Update the GPU buffer for shapes
        bindings['shapes'] = buffer.create_structured_buffer(
            slang_device, module.Shape, shapes)
        # Interpolate the animation for camera
        camera = scene.get('camera', {})
        bindings['camera']['cam_to_world'] = \
            hw3_scene.get_transformation(camera, t)

    duration = scene.get('duration', 0.0)

    def update(t, bindings):
        if duration > 0:
            t = t % duration
            update_scene(t, bindings, scene, shapes,
                         compute_shader.module, slang_device)
        return bindings

    def render(command_encoder, bindings, img):
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render, update)
    else:
        t = args.time % duration if \
            hasattr(args, 'time') and duration > 0 else 0.0
        update(t, bindings)
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw3_5(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_5.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    bindings, scene, shapes = hw3_scene.upload_scene(
        args.scene, compute_shader.module, slang_device)

    def update_scene(t,
                     bindings,
                     scene,
                     shapes,
                     module,
                     slang_device):
        # Update the bindings & shapes in-place
        # the caller is responsible for passing the arrays/dictionaries

        # Interpolate the animation
        for i, obj in enumerate(scene['objects']):
            M = hw3_scene.get_transformation(obj, t)
            shapes[i]['obj_to_world'] = M
            shapes[i]['world_to_obj'] = np.linalg.inv(M)
        # Update the GPU buffer for shapes
        bindings['shapes'] = buffer.create_structured_buffer(
            slang_device, module.Shape, shapes)
        # Interpolate the animation for camera
        camera = scene.get('camera', {})
        bindings['camera']['cam_to_world'] = \
            hw3_scene.get_transformation(camera, t)

    duration = scene.get('duration', 0.0)

    def update(t, bindings):
        if duration > 0:
            t = t % duration
            update_scene(t, bindings, scene, shapes,
                         compute_shader.module, slang_device)
        return bindings

    def render(command_encoder, bindings, img):
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render, update)
    else:
        t = args.time % duration if \
            hasattr(args, 'time') and duration > 0 else 0.0
        update(t, bindings)
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())

def hw3_6(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(slang_device, 'hw3/hw3_6.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    bindings, scene, shapes = hw3_scene.upload_scene(
        args.scene, compute_shader.module, slang_device, use_bvh=True)

    def update_scene(t,
                     bindings,
                     scene,
                     shapes,
                     module,
                     slang_device):
        # Update the bindings & shapes in-place
        # the caller is responsible for passing the arrays/dictionaries

        # Interpolate the animation
        for i, obj in enumerate(scene['objects']):
            M = hw3_scene.get_transformation(obj, t)
            shapes[i]['obj_to_world'] = M
            shapes[i]['world_to_obj'] = np.linalg.inv(M)
        # Update the GPU buffer for shapes
        bindings['shapes'] = buffer.create_structured_buffer(
            slang_device, module.Shape, shapes)
        # Interpolate the animation for camera
        camera = scene.get('camera', {})
        bindings['camera']['cam_to_world'] = \
            hw3_scene.get_transformation(camera, t)

    duration = scene.get('duration', 0.0)

    def update(t, bindings):
        if duration > 0:
            t = t % duration
            update_scene(t, bindings, scene, shapes,
                         compute_shader.module, slang_device)
        return bindings

    def render(command_encoder, bindings, img):
        compute_shader.kernel.dispatch(
            thread_count=[args.width, args.height, 1],
            command_encoder=command_encoder,
            image=img,
            **bindings,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(bindings, render, update)
    else:
        t = args.time % duration if \
            hasattr(args, 'time') and duration > 0 else 0.0
        update(t, bindings)
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, bindings, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())
