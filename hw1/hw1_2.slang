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
    int intersections = 0; // Use non-zero value to indicate that the point is inside the polygon
    for (int i = 0; i < num_polyline - 1 + int(is_closed); i++) {
        float2 p0, p1;
        if (is_closed && i == num_polyline - 1) {
            p0 = polyline[i];
            p1 = polyline[0];
        } else {
            p0 = polyline[i];
            p1 = polyline[i + 1];
        }
        // draw strokes
        if (use_stroke_color) {
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
        // fill color
        if (is_closed && use_fill_color) {
            float y_diff = p1.y - p0.y;
            float x_diff = p1.x - p0.x;
            float t, s;
            if (y_diff != 0) {
                t = (q.y - p0.y) / y_diff;
                s = p0.x + t * x_diff - q.x;
                if (s >= 0 && ((p0.y >= q.y) != (p1.y >= q.y))) {
                    intersections += p0.y < p1.y ? 1 : -1;
                }
            }
        }
    }
    if (intersections != 0) {
        image[tid] = float4(fill_color, 1.0);
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    return;
}
