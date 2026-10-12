[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    float x = float(tid.x) + 0.5;
    float y = float(height) - float(tid.y) - 0.5;
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    float distance = length(float2(x, y) - center);
    if (distance <= radius) {
        image[tid] = float4(color, 1.0);
    } else {
        image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    }
}
