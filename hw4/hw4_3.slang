#define GRAVITY float3(0, -9.8, 0)
#define FLOOR_Y -0.97

static const int2 grid_offsets[12] = {
    int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), 
    int2(1, 1), int2(1, -1), int2(-1, 1), int2(-1, -1), 
    int2(2, 0), int2(-2, 0), int2(0, 2), int2(0, -2), 
};

float3 compute_correction(float3 p1, float3 p2, float w1, float w2, float l) {
    float3 d = p2 - p1;
    float length_d = length(d);
    float1 w_sum = w1 + w2;
    if (length_d == 0 || w_sum == 0) return float3(0);
    return w1 / (w_sum) * (length_d - l) * d / length_d;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }

    float3 p = positions_in[tid];
    float w = inv_masses[tid];
    int2 grid_pos = int2(tid % grid_size.x, tid / grid_size.x);

    float count = 0;
    float3 total_correction = float3(0, 0, 0);
    
    for (int go = 0; go < 12; go++) {
        int2 n = grid_pos + grid_offsets[go];
        if (n.x >= 0 && n.y >= 0 && n.x < grid_size.x && n.y < grid_size.y) {
            int i = n.y * grid_size.x + n.x;
            float rest_length = length(grid_offsets[go] * spacing);
            total_correction += compute_correction(p, positions_in[i], w, inv_masses[i], rest_length);
            count++;
        }
    }

    positions_out[tid] = p + total_correction / count;
}
