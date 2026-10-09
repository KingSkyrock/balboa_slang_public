import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    float subpixel_division = 1.0 / samples_per_axis;
    float2 q = float2(float(tid.x) + subpixel_division / 2, 
                      float(height) - float(tid.y) - 1 + subpixel_division / 2);
    float3 final_color = float3(0);
    for (int subpixel_x = 0; subpixel_x < samples_per_axis; subpixel_x++) {
        for (int subpixel_y = 0; subpixel_y < samples_per_axis; subpixel_y++) {
            float3 subpixel_color = float3(background); // default is bg color
            float2 q_sub = float2(q.x + subpixel_division * subpixel_x,
                           q.y + subpixel_division * subpixel_y);
            for (int i = num_shapes - 1; i >= 0; i--) {
                // Transform using world_to_obj
                float3 q_transform_ = mul(shapes[i].world_to_obj, float3(q_sub, 1));
                float2 q_obj = float2(q_transform_.x, q_transform_.y);

                // Shape properties
                ShapeType type = shapes[i].type;
                int index = shapes[i].index;
                float stroke_radius = shapes[i].stroke_width / 2;
                float3 fill_rgb = shapes[i].fill_rgba.rgb;
                float3 stroke_rgb = shapes[i].stroke_rgba.rgb;
                float fill_alpha = shapes[i].fill_rgba.a;
                float stroke_alpha = shapes[i].stroke_rgba.a;
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
                            float l = dot(p1-p0, q_obj-p0) / d;
                            float2 q_prime = p0 + l * (p1-p0)/ d;
                            if (l > 0 && l < d && length(q_prime - q_obj) < stroke_radius) {
                                subpixel_color = stroke_rgb * stroke_alpha + subpixel_color * (1 - stroke_alpha);
                            }
                            // draw caps
                            if (j != lineStart && j != lineEnd || pl.closed) {
                                float2 center = float2(p0.x, p0.y);
                                if (length(center - q_obj) < stroke_radius) {
                                    subpixel_color = stroke_rgb * stroke_alpha + subpixel_color * (1 - stroke_alpha);
                                }
                            }
                        }
                        // fill color
                        if (pl.closed && use_fill) {
                            float y_diff = p1.y - p0.y;
                            float x_diff = p1.x - p0.x;
                            float t, s;
                            if (y_diff != 0) {
                                t = (q_obj.y - p0.y) / y_diff;
                                s = p0.x + t * x_diff - q_obj.x;
                                if (s >= 0 && ((p0.y >= q_obj.y) != (p1.y >= q_obj.y))) {
                                    intersections += p0.y < p1.y ? 1 : -1;
                                }
                            }
                        }
                    }
                    if (intersections != 0) {
                        subpixel_color = fill_rgb * fill_alpha + subpixel_color * (1 - fill_alpha);
                    }
                } else if (type == ShapeType.Circle) {
                    float2 center = circles[index].center;
                    float radius = circles[index].radius;
                    float distance = length(q_obj - center);
                    if (use_stroke && abs(distance - radius) <= stroke_radius) {
                        subpixel_color = stroke_rgb * stroke_alpha + subpixel_color * (1 - stroke_alpha);
                    } else if (use_fill && distance <= radius) {
                        subpixel_color = fill_rgb * fill_alpha + subpixel_color * (1 - fill_alpha);
                    }
                } else {
                    // ignore quadratic beziers
                }
            }
            final_color += subpixel_color;
        }
    }
    image[tid] = float4(final_color * subpixel_division * subpixel_division, 1.0);
}