import numpy as np

def make_cloth_grid(nx, ny, sx, sy):
    dx, dy = sx / (nx - 1), sy / (ny - 1)
    positions = np.empty((ny, nx, 3), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            positions[y, x] = (-sx / 2 + x * dx, 0.8, sy / 2 - y * dy)

    triangles = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            i = y * nx + x
            triangles.append((i, i + 1, i + nx))
            triangles.append((i + 1, i + nx + 1, i + nx))
    return (np.ascontiguousarray(positions.reshape(-1, 3)),
            np.asarray(triangles, dtype=np.uint32), dx, dy)

def make_cloth_constraints(nx, ny, dx, dy):
    offsets = [
        (1, 0), (0, 1),   # structural
        (1, 1), (-1, 1),  # shear
        (2, 0), (0, 2),   # bending
    ]
    edges, rest_lengths = [], []
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            for ox, oy in offsets:
                xx, yy = x + ox, y + oy
                if 0 <= xx < nx and 0 <= yy < ny:
                    edges.append((i, yy * nx + xx))
                    rest_lengths.append(np.hypot(ox * dx, oy * dy))
    return (np.asarray(edges, dtype=np.uint32).reshape(-1, 2),
            np.asarray(rest_lengths, dtype=np.float32))

def make_rope(anchor, attach, attach_id, start_id,
              segments, rest_scale):
    # Make a rope that attaches to a vertex on a cloth
    # anchor: 3D position of the end point of the rope
    # attach: 3D position of the attached point on the cloth
    #         (not included in the generated rope)
    # attach_id: the ID of the vertex on the cloth
    # start_id: the ID of the beginning vertex of the rope
    # segments: how many segments for the rope
    # rest_scale: a scalar to determine the rest length
    t = np.arange(segments, dtype=np.float32) / segments
    positions = (1 - t[:, None]) * anchor + t[:, None] * attach

    ids = list(range(start_id, start_id + segments)) + [attach_id]
    edges = np.array(list(zip(ids[:-1], ids[1:])), dtype=np.uint32)

    segment_length = rest_scale * np.linalg.norm(anchor - attach) / segments
    rest_lengths = np.full(segments, segment_length, dtype=np.float32)

    return positions.astype(np.float32), edges, rest_lengths