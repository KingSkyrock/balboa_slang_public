[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    float x = float(ipixel.x) + 0.5;
    float y = float(ipixel.y) + 0.5;
    float2 p_screen = float2(x, y);

    float3 p_prime = float3(to_camera(float2(x, y), s, a, width, height), 0);

    float3 p0_prime = float3(to_camera(sp0, s, a, width, height), 0);
    float3 p1_prime = float3(to_camera(sp1, s, a, width, height), 0);
    float3 p2_prime = float3(to_camera(sp2, s, a, width, height), 0);

    float full_area = length(cross(p0_prime - p1_prime, p0_prime - p2_prime));
    float b0_area = length(cross(p_prime - p1_prime, p_prime - p2_prime));
    float b1_area = length(cross(p0_prime - p_prime, p0_prime - p2_prime));
    float b2_area = length(cross(p0_prime - p1_prime, p0_prime - p_prime));

    float b0_prime = b0_area / full_area;
    float b1_prime = b1_area / full_area;
    float b2_prime = b2_area / full_area;

    float denom = (b0_prime * inv_depth0 + b1_prime * inv_depth1 + b2_prime * inv_depth2);
    float b0 = b0_prime * inv_depth0 / denom;
    float b1 = b1_prime * inv_depth1 / denom;
    float b2 = b2_prime * inv_depth2 / denom;

    float depth = b0 / inv_depth0 + b1 / inv_depth1 + b2 / inv_depth2;

    float3 c = b0 * c0 + b1 * c1 + b2 * c2;

    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float sign01 = sign(dot((p_screen - sp0), n01));
    float sign12 = sign(dot((p_screen - sp1), n12));
    float sign20 = sign(dot((p_screen - sp2), n20));

    if ((sign01 >= 0 && sign12 >= 0 && sign20 >= 0) || (sign01 <= 0 && sign12 <= 0 && sign20 <= 0)) {
        if (depth < depth_image[ipixel]) {
            depth_image[ipixel] = depth;
            color_image[ipixel] = float4(c, 1.0);
        }
    }
}

float2 to_camera(float2 point, float s, float a, float width, float height) {
    /*
    Inverse of 
        x_screen = args.width  * (x_camera + s * a) / (2 * s * a)
        y_screen = args.height * (y_camera - s) / (2 * -s)
    */
    float new_x = point.x * 2 * s * a / width - s * a;
    float new_y = point.y * 2 * -s / height + s;
    return float2(new_x, new_y);
}