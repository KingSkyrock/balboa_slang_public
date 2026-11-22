#define GRAVITY float3(0, -9.8, 0)

float3 compute_correction(float3 p1, float3 p2, float w1, float w2, float l) {
    float3 d = p2 - p1;
    float length_d = length(d);
    float1 w_sum = w1 + w2;
    if (length_d == 0 || w_sum == 0) return float3(0);
    return w1 / (w_sum) * (length_d - l) * d / length_d;
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
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    float3 p = positions[tid];      
    contact_planes[tid] = float4(0, 0, 0, 0);  

    float min_sdf = contact_margin + collision_distance;

    for (int i = 0; i < num_planes; i++) {
        float4 plane = planes[i];
        float sdf = SDF_plane(p, plane);
        if (sdf < min_sdf) {
            min_sdf = sdf;
            float4 contact_plane = plane;
            contact_plane.w += collision_distance;
            contact_planes[tid] = contact_plane;
        }
    }
    min_sdf = contact_margin + collision_distance;
    for (int i = 0; i < num_spheres; i++) {
        float4 sphere = spheres[i];
        float3 center = sphere.xyz;
        float sdf = SDF_sphere(p, sphere);
        if (sdf < min_sdf) {
            min_sdf = sdf;
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
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
    if (tid >= num_particles) {
        return;
    }
    float3 p = positions_in[tid];
    float3 old_p = old_positions[tid];
    float w = inv_masses[tid];

    float count = 0;
    float3 other_correction = float3(0, 0, 0);

    for (uint i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        uint n = neighbors[i];
        float rest_length = neighbor_rest[i];
        other_correction += compute_correction(p, positions_in[n], w, inv_masses[n], rest_length);
        count++;
    }

    float4 plane = contact_planes[tid];
    float3 n = plane.xyz;

    float penetration = -min(0, SDF_plane(p, plane));
    float3 tangent = p - old_p - dot(p - old_p, n) * n;

    float3 contact_correction = float3(0, 0, 0);

    contact_correction += n * penetration;
    
    if (length(tangent) <= static_friction * penetration) {
        contact_correction -= tangent;
    } else {
        contact_correction -= normalize(tangent) * kinetic_friction * penetration; 
    }

    positions_out[tid] = p + contact_correction + (count == 0 ? 0 : other_correction / count);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    
    if (inv_masses[tid] == 0) return;

    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}
