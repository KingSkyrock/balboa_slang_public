import json
import slangpy as spy
import numpy as np
import buffer
import math

# Must match Slang's struct definition
CIRCLE = 0
POLYLINE = 1
QUADRATIC_BEZIER = 2

class Scene:
    """
        A Scene contains the following:
        resolution -> (int, int)
        background -> (float, float, float, float) (RGBA)
        duration -> float
        shapes -> [Shape]
        circles -> [Circle]
        polylines -> [Polyline]
        quadratic_beziers -> [QuadraticBezier]
        points -> [(float, float)]

        where
        Shape: {'type': CIRCLE | POLYLINE | QUADRATIC_BEZIER,
                'index': int,
                'use_fill': bool,
                'fill_rgba': (float, float, float, float),
                'use_stroke': bool,
                'stroke_rgba': (float, float, float, float),
                'stroke_width': float,
                'keyframes': Transformation}
        (what type of the shape is, and the index in the corresponding array,
         and stroke/fill rgba)

        Circle: {'center': (x, y),
                 'radius': float}

        Polyline: {'pts_begin': int,
                   'pts_end': int,
                   'closed': bool}
        (pts_begin and pts_end point to corresponding locations in the points array)

        QuadraticBezier: {'pts_begin': int,
                          'pts_end': int,
                          'closed': bool}
        (pts_begin and pts_end point to corresponding locations in the points array)

        Transformation: {'time': float,
                         'transform': [{'scale'|'translate'|'rotate'|'shear_x'|'shear_y': transformation values}]}
    """
    def __init__(self,
                 resolution,
                 background,
                 duration,
                 samples_per_axis,
                 shapes,
                 circles,
                 polylines,
                 quadratic_beziers,
                 points):
        self.resolution = resolution
        self.background = background
        self.duration = duration
        self.samples_per_axis = samples_per_axis
        self.shapes = shapes
        self.circles = circles
        self.polylines = polylines
        self.quadratic_beziers = quadratic_beziers
        self.points = points

    def __repr__(self):
        return (
            f"Scene(\n"
            f"  resolution={self.resolution},\n"
            f"  background={self.background},\n"
            f"  shapes={self.shapes},\n"
            f"  circles={self.circles},\n"
            f"  polylines={self.polylines},\n"
            f"  quadratic_beziers={self.quadratic_beziers},\n"
            f"  points={self.points}\n"
            f")"
        )

# helper function
def add_curves(obj, new_points, closed,
               points, shape_type, curves, shapes):
    # sometimes new_points might be [[1.0, 2.0], [3.0, 4.0], ...]
    # below we normalize the new_points into a flat array
    new_points = np.asarray(new_points, dtype=np.float32).reshape(-1).tolist()

    pts_begin = len(points) // 2
    points += new_points
    pts_end = len(points) // 2

    curves.append({
        'pts_begin': pts_begin,
        'pts_end': pts_end,
        'closed': closed,
    })

    shapes.append({
        'type': shape_type,
        'index': len(curves) - 1,
        'use_fill': 'fill_rgba' in obj,
        'fill_rgba': obj.get('fill_rgba', (0.0, 0.0, 0.0, 0.0)),
        'use_stroke': 'stroke_rgba' in obj,
        'stroke_rgba': obj.get('stroke_rgba', (0.0, 0.0, 0.0, 0.0)),
        'stroke_width': obj.get('stroke_width', 1.0),
    })

def parse_scene(filename : str) -> Scene:
    with open(filename, 'r') as f:
        scene = json.load(f)
        
        resolution = scene['resolution']
        background = scene['background']
        duration = scene['duration'] if 'duration' in scene else 1.0
        samples_per_axis = scene['samples_per_axis'] if 'samples_per_axis' in scene else 4
        shapes = []
        circles = []
        polylines = []
        quadratic_beziers = []
        points = []

        for obj in scene['objects']:
            if obj['type'] == 'circle':
                center = obj['center'] if 'center' in obj else (0.0, 0.0)
                radius = obj['radius'] if 'radius' in obj else 1.0
                circles.append({
                    'center': center,
                    'radius': radius
                })

                shapes.append({
                    'type': CIRCLE,
                    'index': len(circles) - 1,
                    'use_fill': 'fill_rgba' in obj,
                    'fill_rgba': obj['fill_rgba'] if 'fill_rgba' in obj else (0.0, 0.0, 0.0, 0.0),
                    'use_stroke': 'stroke_rgba' in obj,
                    'stroke_rgba': obj['stroke_rgba'] if 'stroke_rgba' in obj else (0.0, 0.0, 0.0, 0.0),
                    'stroke_width': obj['stroke_width'] if 'stroke_width' in obj else 1.0
                })
            elif obj['type'] == 'polyline':
                add_curves(obj, obj['points'], obj['closed'] if 'closed' in obj else True,
                           points, POLYLINE, polylines, shapes)
            elif obj['type'] == 'triangle':
                p0 = obj['p0'] if 'p0' in obj else [0.0, 0.0]
                p1 = obj['p1'] if 'p1' in obj else [1.0, 0.0]
                p2 = obj['p2'] if 'p2' in obj else [0.0, 1.0]
                tri_pts = p0 + p1 + p2
                add_curves(obj, tri_pts, True,
                           points, POLYLINE, polylines, shapes)
            elif obj['type'] == 'rectangle':
                x0, y0 = obj['p_min'] if 'p_min' in obj else [0.0, 0.0]
                x1, y1 = obj['p_max'] if 'p_max' in obj else [1.0, 1.0]

                rect_points = [
                    x0, y0, x1, y0, x1, y1, x0, y1,
                ]

                add_curves(obj, rect_points, True,
                           points, POLYLINE, polylines, shapes)
            elif obj['type'] == 'quadratic_bezier':
                add_curves(obj, obj['points'], obj['closed'] if 'closed' in obj else True,
                           points, QUADRATIC_BEZIER, quadratic_beziers, shapes)

            if 'transform' in obj:
                shapes[-1]['transform'] = obj['transform']
            if 'transform_keyframes' in obj:
                shapes[-1]['transform_keyframes'] = obj['transform_keyframes']

    points = np.array(points, dtype=np.float32).reshape(-1, 2)

    return Scene(resolution,
                 background,
                 duration,
                 samples_per_axis,
                 shapes,
                 circles,
                 polylines,
                 quadratic_beziers,
                 points)

def compose_transformation(transforms):
    """
        Given a list of transformations, compose them into a matrix.
    """

    F = np.eye(3, dtype=np.float32)

    for transform in transforms:
        if "shear_y" in transform:
            F = F @ np.matrix([[1, 0, 0], [transform["shear_y"][0], 1, 0], [0, 0, 1]])
        elif "shear_x" in transform:
            F = F @ np.matrix([[1, transform["shear_x"][0], 0], [0, 1, 0], [0, 0, 1]])
        elif "scale" in transform:
            F = F @ np.matrix([[transform["scale"][0], 0, 0], [0, transform["scale"][1], 0], [0, 0, 1]])
        elif "rotate" in transform:
            angle = transform["rotate"][0] * np.pi / 180
            cosAngle = np.cos(angle)
            sinAngle = np.sin(angle)
            F = F @ np.matrix([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])
        elif "translate" in transform:
            F = F @ np.matrix([[1, 0, transform["translate"][0]], [0, 1, transform["translate"][1]], [0, 0, 1]])
    F = np.asarray(F)
    return F

def interpolate_transformation(transform_keyframes, t):
    # TODO: your code here

    # Should never happen?
    assert False

def upload_scene(scene, module, slang_device, t=0.0):
    # Keyframe parameters are interpolated on the CPU;
    # For simplicity, we re-upload the scene buffers every frame,
    # even though most scene data does not change.
    # This can be slow in practice.

    shapes = []
    for shape in scene.shapes:
        new_shape = shape.copy()
        new_shape.pop("transform", None)
        new_shape.pop("transform_keyframes", None)
        new_shape['world_to_obj'] = np.eye(3, dtype=np.float32)

        # compose_transformation returns object -> world.
        # the shader receives the inverse because hit tests are done in object space
        if 'transform' in shape:
            obj_to_world = compose_transformation(shape['transform'])
            new_shape['world_to_obj'] = np.linalg.inv(obj_to_world)
        if 'transform_keyframes' in shape:
            transforms = interpolate_transformation(shape['transform_keyframes'], t)
            obj_to_world = compose_transformation(transforms)
            new_shape['world_to_obj'] = np.linalg.inv(obj_to_world)
        shapes.append(new_shape)

    return {
        'background': scene.background,
        'samples_per_axis': scene.samples_per_axis,
        'shapes': buffer.create_structured_buffer(slang_device, module.Shape, shapes),
        'num_shapes': len(shapes),
        'circles': buffer.create_structured_buffer(slang_device, module.Circle, scene.circles),
        'polylines': buffer.create_structured_buffer(slang_device, module.Polyline, scene.polylines),
        'quadratic_beziers': buffer.create_structured_buffer(slang_device, module.QuadraticBezier, scene.quadratic_beziers),
        'points': buffer.structured_buffer_from_numpy(slang_device, scene.points),
    }
