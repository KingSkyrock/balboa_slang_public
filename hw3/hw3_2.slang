import hw3_common;

// arbitrary large number for t_max when not doing shadow
static const float really_big_number = 0x5F3759DFp67;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    float a = float(width) / float(height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float2 p = to_camera(float2(tid.x + 0.5, tid.y + 0.5), 1, a, width, height);
    float3 o = float3(0, 0, 0);
    float3 d = float3(p, -1);

    float closest_t;
    float3 closest_normal;
    float3 closest_Kd;
    if (is_ray_intersect(o, d, EPSILON, really_big_number, shapes, num_shapes, spheres,
                         closest_t, closest_normal, closest_Kd)) {
        float3 hit_point = o + closest_t * d;
        float3 color = float3(0, 0, 0);
        for (int i = 0; i < num_lights; i++) {
            PointLight light = lights[i];
            float3 dist_to_light = light.position - hit_point;
            float dist_squared = dot(dist_to_light, dist_to_light);
            float3 l = dist_to_light / sqrt(dist_squared);

            float shadow_t;
            float3 shadow_normal;
            float3 shadow_Kd;
            if (!is_ray_intersect(hit_point, dist_to_light, EPSILON, 1.0,
                                  shapes, num_shapes, spheres, shadow_t,
                                  shadow_normal, shadow_Kd)) {
                color += closest_Kd * light.intensity
                       * max(dot(closest_normal, l), 0.0) / dist_squared;
            }
        }
        image[tid] = float4(color, 1.0);
    } else {
        image[tid] = float4(background, 1.0);
    }
}

bool is_ray_intersect(float3 o, float3 d, float t_min, float t_max,
                       StructuredBuffer<Shape> shapes, int num_shapes,
                       StructuredBuffer<Sphere> spheres,
                       out float closest_t, out float3 closest_normal, out float3 closest_Kd) {
    closest_t = -1; // -1 means not hit
    closest_normal = float3(0, 0, 0);
    closest_Kd = float3(0, 0, 0);
    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];
        Sphere sphere = spheres[shape.index];
        float3 xc = sphere.center;
        float r = sphere.radius;

        float2 t;
        float3 dist = o - xc;
        float A = dot(d, d);
        float B = 2 * dot(d, dist);
        float C = dot(dist, dist) - r * r;
        bool hit = solve_quadratic(A, B, C, t);
        float t_hit = 0;
        if (hit) {
            float t1 = t.x;
            float t2 = t.y;
            bool t1_valid = t1 > t_min && t1 < t_max;
            bool t2_valid = t2 > t_min && t2 < t_max;
            if (t1_valid && t2_valid) {
                t_hit = min(t1, t2);
            } else if (t1_valid) {
                t_hit = t1;
            } else if (t2_valid) {
                t_hit = t2;
            } else {
                hit = false;
            }
        }
        if (hit) {
            if (closest_t < 0 || t_hit < closest_t) {
                float3 hit_point = o + t_hit * d;
                closest_t = t_hit;
                closest_normal = (hit_point - xc) / r;
                closest_Kd = shape.Kd;
            }
        }
    }
    return closest_t > 0;
}

float2 to_camera(float2 point, float s, float a, float width, float height) {
    float new_x = point.x * 2 * s * a / width - s * a;
    float new_y = point.y * 2 * -s / height + s;
    return float2(new_x, new_y);
}
