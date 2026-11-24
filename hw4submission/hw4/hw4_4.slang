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
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    float3 p = positions_in[tid];
    float w = inv_masses[tid];

    float count = 0;
    float3 total_correction = float3(0, 0, 0);

    for (uint i = row_ptr[tid]; i < row_ptr[tid + 1]; i++) {
        uint n = neighbors[i];
        float rest_length = neighbor_rest[i];
        total_correction += compute_correction(p, positions_in[n], w, inv_masses[n], rest_length);
        count++;
    }

    if (count == 0) {
        positions_out[tid] = p;
        return;
    }

    positions_out[tid] = p + total_correction / count;
}
