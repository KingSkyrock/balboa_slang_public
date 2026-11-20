[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 gravity = float3(0, -9.8, 0);
    float floor_y = -0.97;
    float3 v = velocities[tid] + gravity * dt;
    float3 p = positions[tid] + v * dt;
    if (p.y < floor_y) {
        p.y = floor_y;
        v.y = -v.y;
    }
    positions[tid] = p;
    velocities[tid] = v;
}
