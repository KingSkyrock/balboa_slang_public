import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    uint x = tid.x;
    uint y = height - tid.y - 1;
    float2 q = float2(x, y);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    float radius = stroke_width / 2;
    for (int i = 0; i < num_polyline - 1; i++) {
        float2 p0 = polyline[i];
        float2 p1 = polyline[i + 1];
        float d = length(p1-p0);
        float l = dot(p1-p0, q-p0) / d;
        float2 q_prime = p0 + l * (p1-p0)/ d;
        if (l > 0 && l < d && length(q_prime - q) < radius) {
            image[tid] = float4(stroke_color, 1.0);
            return;
        }
        // draw caps
        if (i != 0 && i != num_polyline - 1 || is_closed) {
            float2 center = float2(p0.x, p0.y);
            if (length(center - q) < radius) {
                image[tid] = float4(stroke_color, 1.0);
                return;
            }
        }
    }
    if (is_closed) {
        // Draw stroke to close the polyline
        float2 p0 = polyline[num_polyline - 1];
        float2 p1 = polyline[0];
        float d = length(p1-p0);
        float l = dot(p1-p0, q-p0) / d;
        float2 q_prime = p0 + l * (p1-p0)/ d;
        if (l > 0 && l < d && length(q_prime - q) < radius) {
            image[tid] = float4(stroke_color, 1.0);
            return;
        }
        // draw cap for p0 point
        float2 center = float2(p0.x, p0.y);
        if (length(center - q) < radius) {
            image[tid] = float4(stroke_color, 1.0);
            return;
        }
        // Now fill the color
        
    } 

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    return;
}
