// Common scene definitions
enum ShapeType {
    Sphere,
    TriangleMesh
};

struct Camera {
    float4x4 cam_to_world;
    float fov_x;
};

struct Sphere {
    float3 center;
    float radius;
};

struct TriangleMesh {
    uint vertex_offset;
    uint triangle_begin;
    uint triangle_end;
    uint bvh_root_index; // Used in 3.6
};

struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

struct Shape {
    ShapeType type;
    // Index the shape into the type-specific arrays
    // (spheres and triangle_mesh)
    int index;

    float3 Kd;
    float3 Kr;
    
    // The following are always affine transforms.
    // Maps world-space positions to the shape's object space.
    float4x4 world_to_obj;
    // Inverse of above.
    float4x4 obj_to_world;
};

struct PointLight {
    float3 position;
    float3 intensity;
};

static const float EPSILON = 1e-3;

// Polynomial solvers
// Return true if a root is found.
// The two real roots are stored in t
bool solve_quadratic(float a, float b, float c,
                     out float2 t) {
    // From https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h#L613
    if (a == 0) {
        if (b == 0) {
            t = float2(0.0, 0.0);
            return false;
        }
        t = float2(-c / b, -c / b);
        return true;
    }

    float discrim = b * b - 4 * a * c;
    if (discrim < 0) {
        t = float2(0, 0);
        return false;
    }
    float root_discrim = sqrt(discrim);

    float q;
    if (b < 0) {
        q = -0.5f * (b - root_discrim);
    } else {
        q = -0.5f * (b + root_discrim);
    }
    t.x = q / a;
    t.y = c / q;
    return true;
}

float3 xform_pt(float4x4 xform, float3 pt) {
    return mul(xform, float4(pt, 1.0)).xyz;
}

float3 xform_vec(float4x4 xform, float3 vec) {
    return mul(xform, float4(vec, 0.0)).xyz;
}

bool intersect_tri(float3 p0, float3 p1, float3 p2,
                   float3 ray_org,
                   float3 ray_dir,
                   float t_min,
                   float t_max,
                   out float3 b,
                   out float t) {
    // return the barycentric coordinates b
    // and intersection time t
    b = float3(0.0, 0.0, 0.0);
    t = t_max;

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 s1 = cross(ray_dir, e2);
    float divisor = dot(s1, e1);
    if (divisor == 0) {
        return false;
    }

    float inv_divisor = 1 / divisor;
    float3 s = ray_org - p0;
    float u = dot(s, s1) * inv_divisor;
    float3 s2 = cross(s, e1);
    float v = dot(ray_dir, s2) * inv_divisor;
    t = dot(e2, s2) * inv_divisor;

    if (t > t_min && t < t_max && u >= 0 && v >= 0 && u + v <= 1) {
        b.x = 1 - u - v;
        b.y = u;
        b.z = v;
        return true;
    }

    t = t_max;
    return false;
}

bool intersect_aabb(float3 p_min,
                    float3 p_max,
                    float3 ray_org,
                    float3 ray_dir,
                    float t_min,
                    float t_max) {
    // Return true if intersect
    // https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/rayintersectionwithanaabb
    for (int i = 0; i < 3; i++) {
        // This code was designed to work even when ray_dir[i] == 0
        // See https://pbr-book.org/4ed/Shapes/Basic_Shape_Interface
        float inv_dir = float(1) / ray_dir[i];
        float t0 = (p_min[i] - ray_org[i]) * inv_dir;
        float t1 = (p_max[i] - ray_org[i]) * inv_dir;
        if (inv_dir < 0) {
            // swap
            float _tmp = t1;
            t1 = t0;
            t0 = _tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max < t_min) {
            return false;
        }
    }
    return true;
}
