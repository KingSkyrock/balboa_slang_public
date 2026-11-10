import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
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

    float closest_t = -1; // -1 means not hit
    float3 closest_normal = float3(0, 0, 0);

    for (int i = 0; i < num_spheres; i++) {
        Sphere s = spheres[i];
        float3 xc = s.center;
        float r = s.radius;

        float2 t;
        float3 dist = o - xc;
        float A = dot(d, d);
        float B = 2 * dot(d, dist);
        float C = dot(dist, dist) - r * r;
        bool hit = solve_quadratic(A, B, C, t);
        float t_hit = 0;
        float3 hit_point;
        if (hit) {
            float t1 = t.x;
            float t2 = t.y;
            if (t1 > 0 && t2 > 0) {
                t_hit = min(t1, t2);
            } else if (t1 > 0 || t2 > 0) {
                t_hit = max(t1, t2);
            } else {
                hit = false;
            }
        }
        if (hit) {
            if (closest_t < 0 || t_hit < closest_t) {
                hit_point = o + t_hit * d;
                closest_t = t_hit;
                closest_normal = (hit_point - xc) / r;
            }
        }
    }

    if (closest_t > 0) {
        image[tid] = float4((closest_normal + 1) / 2, 1.0);
    } else {
        image[tid] = float4(background, 1.0);
    }
}

float2 to_camera(float2 point, float s, float a, float width, float height) {
    float new_x = point.x * 2 * s * a / width - s * a;
    float new_y = point.y * 2 * -s / height + s;
    return float2(new_x, new_y);
}