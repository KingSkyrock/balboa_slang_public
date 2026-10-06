import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    float2 q = float2(float(tid.x) + 0.5, float(height) - float(tid.y) - 0.5);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    for (int i = 0; i < num_shapes; i++) {
        
        // Transform using world_to_obj
        float3 q_transform_ = mul(shapes[i].world_to_obj, float3(q, 1));
        q = float2(q_transform_.x, q_transform_.y);

        // Shape properties
        ShapeType type = shapes[i].type;
        int index = shapes[i].index;
        float stroke_radius = shapes[i].stroke_width / 2;
        float4 fill_rgba = shapes[i].fill_rgba;
        float4 stroke_rgba = shapes[i].stroke_rgba;
        bool use_stroke = shapes[i].use_stroke;
        bool use_fill = shapes[i].use_fill;

        if (type == ShapeType.Polyline) {
            Polyline pl = polylines[index];
            int lineStart = pl.pts_begin;
            int lineEnd = pl.pts_end - 1;
            int intersections = 0;
            for (int j = lineStart; j < lineEnd + int(pl.closed); j++) {
                float2 p0, p1;
                if (pl.closed && j == lineEnd) {
                    p0 = points[j];
                    p1 = points[lineStart];
                } else {
                    p0 = points[j];
                    p1 = points[j + 1];
                }
                 // draw strokes
                if (use_stroke) {
                    float d = length(p1-p0);
                    float l = dot(p1-p0, q-p0) / d;
                    float2 q_prime = p0 + l * (p1-p0)/ d;
                    if (l > 0 && l < d && length(q_prime - q) < stroke_radius) {
                        image[tid] = float4(stroke_rgba);
                        return;
                    }
                    // draw caps
                    if (j != 0 && j != lineEnd || pl.closed) {
                        float2 center = float2(p0.x, p0.y);
                        if (length(center - q) < stroke_radius) {
                            image[tid] = float4(stroke_rgba);
                            return;
                        }
                    }
                }
                // fill color
                if (pl.closed && use_fill) {
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
                image[tid] = float4(fill_rgba);
                return;
            }
        } else if (type == ShapeType.Circle) {
            float2 center = circles[index].center;
            float radius = circles[index].radius - stroke_radius;
            float radius_including_stroke = circles[index].radius + stroke_radius;
            float1 distance = sqrt(pow(center.x - q.x, 2) + pow(center.y - q.y, 2));
            if (use_fill && distance <= radius) {
                image[tid] = fill_rgba;
                return;
            } else if (use_stroke && distance > radius && distance <= radius_including_stroke) {
                image[tid] = stroke_rgba;
                return;
            }
        } else {
            // ignore quadratic beziers
        }
    }
    image[tid] = float4(background, 1.0);
    return;
}
