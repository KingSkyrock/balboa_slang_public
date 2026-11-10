import hw3_common;

// arbitrary large number for t_max when not doing shadow
static const float really_big_number = 0x5F3759DFp67;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float4x4 c2w = camera.cam_to_world;

    float a = float(width) / float(height);
    float s = tan(radians(camera.fov_x) / 2) / a;
    float2 p = to_camera(float2(tid.x + 0.5, tid.y + 0.5), s, a, width, height);
    float3 o = xform_pt(c2w, float3(0, 0, 0));
    float3 d = xform_vec(c2w, float3(p, -1));

    float closest_t;
    float3 closest_normal;
    float3 closest_Kd;
    float3 closest_Kr;
    float3 color = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {
        if (!is_ray_intersect(o, d, EPSILON, really_big_number, shapes, num_shapes, spheres, 
                            meshes, vertices, vertex_normals, triangles,
                            closest_t, closest_normal, closest_Kd, closest_Kr)) {
            color += throughput * background;
            break; // background so the loop can end now
        }
        float3 hit_point = o + closest_t * d;

        // compute contribution from point lights
        float3 contribution = 0.0;
        for (int i = 0; i < num_lights; i++) {
            PointLight light = lights[i];
            float3 dist_to_light = light.position - hit_point;
            float dist_squared = dot(dist_to_light, dist_to_light);
            float3 l = dist_to_light / sqrt(dist_squared);

            float shadow_t;
            float3 shadow_normal;
            float3 shadow_Kd;
            float3 shadow_Kr;
            if (!is_ray_intersect(hit_point, dist_to_light, EPSILON, 1.0,
                                  shapes, num_shapes, spheres, 
                                  meshes, vertices, vertex_normals, triangles, 
                                  shadow_t, shadow_normal, shadow_Kd, shadow_Kr)) {
                contribution += closest_Kd * light.intensity
                              * max(dot(closest_normal, l), 0.0) / dist_squared;
            }
        }
        color += throughput * contribution;
        throughput *= closest_Kr;
        o = hit_point;
        d = reflect(d, closest_normal);
    }

    image[tid] = float4(color, 1.0);
}

bool is_ray_intersect(float3 o, float3 d, float t_min, float t_max,
                       StructuredBuffer<Shape> shapes, int num_shapes,
                       StructuredBuffer<Sphere> spheres,
                       StructuredBuffer<TriangleMesh> meshes,
                       StructuredBuffer<float3> vertices,
                       StructuredBuffer<float3> vertex_normals,
                       StructuredBuffer<uint3> triangles,
                       out float closest_t, out float3 closest_normal, out float3 closest_Kd,
                       out float3 closest_Kr) {
    closest_t = -1; // -1 means not hit
    closest_normal = float3(0, 0, 0);
    closest_Kd = float3(0, 0, 0);
    closest_Kr = float3(0, 0, 0);
    for (int i = 0; i < num_shapes; i++) {
        Shape shape = shapes[i];
        if (shape.type == ShapeType.Sphere) {
            Sphere sphere = spheres[shape.index];
            float3 xc = sphere.center;
            float r = sphere.radius;

            float3 o_obj = xform_pt(shape.world_to_obj, o);
            float3 d_obj = xform_vec(shape.world_to_obj, d);

            float2 t;
            float3 dist = o_obj - xc;
            float A = dot(d_obj, d_obj);
            float B = 2 * dot(d_obj, dist);
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
                    float3 hit_point = o_obj + t_hit * d_obj;
                    closest_t = t_hit;
                    closest_Kd = shape.Kd;
                    closest_Kr = shape.Kr;

                    float3 n_obj = (hit_point - xc) / r;
                    closest_normal = normalize(
                        xform_vec(
                            transpose(shape.world_to_obj), n_obj
                        )
                    );
                }
            }
        } else if (shape.type == ShapeType.TriangleMesh) {
            TriangleMesh mesh = meshes[shape.index];

            float3 o_obj = xform_pt(shape.world_to_obj, o);
            float3 d_obj = xform_vec(shape.world_to_obj, d);

            float t_hit = t_max;
            float3 n_obj = float3(0, 0, 0);
            bool hit = false;

            // only using uint because triangle begin and end are in uint
            for (uint j = mesh.triangle_begin; j < mesh.triangle_end; j++) {
                uint3 triangle = triangles[j] + mesh.vertex_offset;
                float3 p0 = vertices[triangle.x];
                float3 p1 = vertices[triangle.y];
                float3 p2 = vertices[triangle.z];
                float3 bary;
                float t;

                if (intersect_tri(p0, p1, p2, o_obj, d_obj, t_min, t_hit, bary, t)) {
                    //if this goes through it means we found a closer triangle
                    float3 n0 = vertex_normals[triangle.x];
                    float3 n1 = vertex_normals[triangle.y];
                    float3 n2 = vertex_normals[triangle.z];

                    n_obj = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);

                    t_hit = t;
                    hit = true;
                }
            }

            if (hit) {
                if (closest_t < 0 || t_hit < closest_t) {
                    closest_t = t_hit;
                    closest_Kd = shape.Kd;
                    closest_Kr = shape.Kr;
                    closest_normal = normalize(
                        xform_vec(
                            transpose(shape.world_to_obj), n_obj
                        )
                    );
                }
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
