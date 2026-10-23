[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

bool pixelHitsFace(float2 p, float2 sp0, float2 sp1, float2 sp2) {
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float sign01 = sign(dot((p - sp0), n01));
    float sign12 = sign(dot((p - sp1), n12));
    float sign20 = sign(dot((p - sp2), n20));

    return (sign01 >= 0 && sign12 >= 0 && sign20 >= 0) || (sign01 <= 0 && sign12 <= 0 && sign20 <= 0);
}

float3 interpolate_barycentric(float2 p, float3 sp0, float3 sp1, float3 sp2, float s, float a, uint width, uint height) {
    float3 p_prime = float3(to_camera(p, s, a, width, height), 0);

    float3 p0_prime = float3(to_camera(sp0.xy, s, a, width, height), 0);
    float3 p1_prime = float3(to_camera(sp1.xy, s, a, width, height), 0);
    float3 p2_prime = float3(to_camera(sp2.xy, s, a, width, height), 0);

    float full_area = length(cross(p0_prime - p1_prime, p0_prime - p2_prime));
    float b0_area = length(cross(p_prime - p1_prime, p_prime - p2_prime));
    float b1_area = length(cross(p0_prime - p_prime, p0_prime - p2_prime));
    float b2_area = length(cross(p0_prime - p1_prime, p0_prime - p_prime));

    float b0_prime = b0_area / full_area;
    float b1_prime = b1_area / full_area;
    float b2_prime = b2_area / full_area;

    float denom = (b0_prime * sp0.z + b1_prime * sp1.z + b2_prime * sp2.z);
    float b0 = b0_prime * sp0.z / denom;
    float b1 = b1_prime * sp1.z / denom;
    float b2 = b2_prime * sp2.z / denom;

    return float3(b0, b1, b2);
}

float interpolate_barycentric_depth(float2 p, float3 sp0, float3 sp1, float3 sp2, float s, float a, uint width, uint height) {
    float3 b = interpolate_barycentric(p, sp0, sp1, sp2, s, a, width, height);
    float b0 = b.x;
    float b1 = b.y;
    float b2 = b.z;
    float depth = b0 / sp0.z + b1 / sp1.z + b2 / sp2.z;
    return depth;
}

float3 interpolate_barycentric_color(float2 p, float3 sp0, float3 sp1, float3 sp2, float s, float a, uint width, uint height, float3 c0, float3 c1, float3 c2) {
    float3 b = interpolate_barycentric(p, sp0, sp1, sp2, s, a, width, height);
    float b0 = b.x;
    float b1 = b.y;
    float b2 = b.z;

    float3 c = b0 * c0 + b1 * c1 + b2 * c2;
    return c;
}


float2 to_camera(float2 point, float s, float a, float width, float height) {
    float new_x = point.x * 2 * s * a / width - s * a;
    float new_y = point.y * 2 * -s / height + s;
    return float2(new_x, new_y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {

    if (tid >= num_triangles)
        return;
    uint3 face = faces[tid];
    float3 sp0 = projected_vertices[face.x];
    float3 sp1 = projected_vertices[face.y];
    float3 sp2 = projected_vertices[face.z];

    float min_x = fmin3(sp0.x, sp1.x, sp2.x);
    float min_y = fmin3(sp0.y, sp1.y, sp2.y);
    float max_x = fmax3(sp0.x, sp1.x, sp2.x);
    float max_y = fmax3(sp0.y, sp1.y, sp2.y);

    // guard against negative with max. unecessary tho. might remove later
    uint2 aabb_min = uint2(max(uint(min_x), 0), max(uint(min_y), 0));
    uint2 aabb_max = uint2(min(uint(max_x), width - 1), min(uint(max_y), height - 1));

    for (uint x = aabb_min.x; x <= aabb_max.x; x++) {
        for (uint y = aabb_min.y; y <= aabb_max.y; y++) {
            float2 p_screen = float2(x + 0.5, y + 0.5);
            if (pixelHitsFace(p_screen, sp0.xy, sp1.xy, sp2.xy)) {
                float depth = interpolate_barycentric_depth(p_screen, sp0, sp1, sp2, s, a, width, height);
                uint depth_bits = asuint(depth);
                InterlockedMin(depth_buffer[x + y * width], depth_bits);
            }
        }
    }
// depth_buffer[x, y] = 0xffffffff for all x, y
// parallel for each face:
// for x, y in screen AABB:
// if pixel_center hits face:
// d = interpolate_barycentric
// uint depth_bits = asuint(d);
// InterlockedMin(depth_buffer[x, y], depth_bits);

}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO
    
    if (tid >= num_triangles)
        return;

    uint3 face = faces[tid];

    float3 sp0 = projected_vertices[face.x];
    float3 sp1 = projected_vertices[face.y];
    float3 sp2 = projected_vertices[face.z];
    
    float min_x = fmin3(sp0.x, sp1.x, sp2.x);
    float min_y = fmin3(sp0.y, sp1.y, sp2.y);
    float max_x = fmax3(sp0.x, sp1.x, sp2.x);
    float max_y = fmax3(sp0.y, sp1.y, sp2.y);

    uint2 aabb_min = uint2(max(uint(min_x), 0), max(uint(min_y), 0));
    uint2 aabb_max = uint2(min(uint(max_x), width - 1), min(uint(max_y), height - 1));

    for (uint x = aabb_min.x; x <= aabb_max.x; x++) {
        for (uint y = aabb_min.y; y <= aabb_max.y; y++) {
            float2 p_screen = float2(x + 0.5, y + 0.5);
            if (pixelHitsFace(p_screen, sp0.xy, sp1.xy, sp2.xy)) {
                float depth = interpolate_barycentric_depth(p_screen, sp0, sp1, sp2, s, a, width, height);
                if (asuint(depth) == depth_buffer[x + y * width]) {
                    InterlockedMin(triangle_id_buffer[x + y * width], tid);
                }
            }
        }
    }

    // id[x, y] = 0xffffffff for all x, y
    // parallel for each face:
    //     for x, y in screen AABB:
    //         if pixel_center hits face:
    //             d = interpolate_barycentric
    //         if asuint(d) == depth_buffer[x, y]:
    //             InterlockedMin(id[x, y], face_id)
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint3 face = faces[tid.x + tid.y * width];

    float3 sp0 = projected_vertices[face.x];
    float3 sp1 = projected_vertices[face.y];
    float3 sp2 = projected_vertices[face.z];

    float3 c0 = vertex_colors[face.x];
    float3 c1 = vertex_colors[face.y];
    float3 c2 = vertex_colors[face.z];

    float x = float(tid.x) + 0.5;
    float y = float(tid.y) + 0.5;
    float2 p_screen = float2(x, y);

    uint face_id = triangle_id_buffer[tid.x + tid.y * width];
    if (face_id != 0xffffffff) {
        image[tid] = float4(interpolate_barycentric_color(p_screen, sp0, sp1, sp2, s, a, width, height, c0, c1, c2), 1);
    }

    // HW 2.5 TODO
}
