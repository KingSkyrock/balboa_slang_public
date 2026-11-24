#define GRAVITY float3(0, -9.8, 0)
#define FLOOR_Y -0.97

float3 compute_correction(float3 p1, float3 p2, float w1, float w2, float l) {
    float3 d = p2 - p1;
    float length_d = length(d);
    float1 w_sum = w1 + w2;
    if (length_d == 0 || w_sum == 0) return float3(0);
    return w1 / (w_sum) * (length_d - l) * d / length_d;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    old_positions[tid] = positions[tid];
    if (inv_masses[tid] == 0) return;
    float3 v = velocities[tid] + GRAVITY * dt;
    float3 p = positions[tid] + v * dt;
    positions[tid] = p;
    velocities[tid] = v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    float3 p = positions_in[tid];
    float w = inv_masses[tid];
    int n1 = int(tid) - 1;
    int n2 = tid + 1;
    float num_neighbors = 0;
    float3 total_correction = float3(0, 0, 0);
    if (n1 >= 0) {
        float3 p2 = positions_in[n1];
        float w2 = inv_masses[n1];
        float l = rest_lengths[n1];
        total_correction += compute_correction(p, p2, w, w2, l);
        num_neighbors++;
    }
    if (n2 < num_particles) {
        float3 p2 = positions_in[n2];
        float w2 = inv_masses[n2];
        float l = rest_lengths[tid];
        total_correction += compute_correction(p, p2, w, w2, l);
        num_neighbors++;
    }

    positions_out[tid] = p + total_correction / num_neighbors;
    // TODO: HW 4.2. Your code below.
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    if (inv_masses[tid] == 0) return;

    if (positions[tid].y < FLOOR_Y) {
        positions[tid].y = FLOOR_Y;
    }

    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
    // TODO: HW 4.2. Your code below.
}
