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

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}

float SDF_plane(float3 p, float4 plane) {
    float3 n = plane.xyz;
    float d = plane.w;
    float len_n = length(n);
    if (len_n == 0) return 0; // arbitrary
    return (dot(n, p) - d) / len_n;
}

float SDF_sphere(float3 p, float4 sphere) {
    float3 c = sphere.xyz;
    float r = sphere.w;
    return length(p - c) - r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    float3 p = positions[tid];      
    contact_planes[tid] = float4(0, 0, 0, 0);  

    float min_dist = contact_margin + collision_distance;

    for (int i = 0; i < num_planes; i++) {
        float4 plane = planes[i];
        float sdf = SDF_plane(p, plane);
        if (sdf < min_dist) {
            min_dist = sdf;
            float4 contact_plane = plane;
            contact_plane.w += collision_distance;
            contact_planes[tid] = contact_plane;
        }
    }
    for (int i = 0; i < num_spheres; i++) {
        float4 sphere = spheres[i];
        float3 center = sphere.xyz;
        float sdf = SDF_sphere(p, sphere);
        if (sdf < min_dist) {
            min_dist = sdf;
            float3 n;
            if (length(p - center) == 0) {
                n = float3(1, 0, 0); // arbitrary
            } else {
                n = normalize(p - center);
            }
            float d = dot(n, center + sphere.w * n);
            contact_planes[tid] = float4(n, d + collision_distance);
        }
    }

    float tri_min_dist = contact_margin + collision_distance;
    bool tri_found = false;
    float4 tri_plane = float4(0, 0, 0, 0);

    uint node_index = 0;
    uint end_index = uint(bvh_nodes[node_index].next_index);
    while (node_index < end_index) {
        BVHNode node = bvh_nodes[node_index];

        if (point_aabb_distance(p, node.bbox_min, node.bbox_max) > tri_min_dist) {
            node_index = uint(node.next_index);
            continue;
        }
        
        if (node.is_leaf) {
            uint tri_begin = node.triangle_begin;
            uint tri_end = node.triangle_end;

            float closest_tri_dist = -1;
            uint3 closest_tri;
            float3 closest_tri_bary;
            float3 closest_tri_point;

            for (int i = tri_begin; i < tri_end; i++) {
                uint3 tri = mesh_triangles[i];
                
                float3 v0 = mesh_vertices[tri.x];
                float3 v1 = mesh_vertices[tri.y];
                float3 v2 = mesh_vertices[tri.z];

                ClosestPoint closest_point = closest_point_triangle(p, v0, v1, v2);
                float dist_to_tri = length(closest_point.position - p);
                if (closest_tri_dist == -1 || dist_to_tri < closest_tri_dist) {
                    closest_tri_dist = dist_to_tri;
                    closest_tri = tri;
                    closest_tri_bary = closest_point.bary;
                    closest_tri_point = closest_point.position;
                }
            }

            if (closest_tri_dist < tri_min_dist && closest_tri_dist != -1) {
                tri_min_dist = closest_tri_dist;
                tri_found = true;
                float3 v0 = mesh_vertices[closest_tri.x];
                float3 v1 = mesh_vertices[closest_tri.y];
                float3 v2 = mesh_vertices[closest_tri.z];

                float3 n0 = mesh_vertex_normals[closest_tri.x];
                float3 n1 = mesh_vertex_normals[closest_tri.y];
                float3 n2 = mesh_vertex_normals[closest_tri.z];

                float b0 = closest_tri_bary.x;
                float b1 = closest_tri_bary.y;
                float b2 = closest_tri_bary.z;
                
                // Face normal
                float3 ng = normalize(cross(v1 - v0, v2 - v0));
                // Phong normal
                float3 ns = normalize(b0 * n0 + b1 * n1 + b2 * n2);
                if (dot(ng, ns) < 0.0) {
                    ng = -ng;
                }
                // Use ng as the normal of our contact plane.
                float d = dot(ng, closest_tri_point);
                tri_plane = float4(ng, d);
            }

        }
        node_index++;
    }

    if (tri_found) {
        float sdf = SDF_plane(p, tri_plane);
        if (sdf < min_dist) {
            min_dist = sdf;
            contact_planes[tid] = float4(tri_plane.xyz, tri_plane.w + collision_distance);
        }
    }
}
