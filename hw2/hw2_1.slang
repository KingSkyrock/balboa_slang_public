[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    float x = float(ipixel.x) + 0.5;
    float y = float(ipixel.y) + 0.5;
    float2 p = float2(x, y);

    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float sign01 = sign(dot((p - sp0), n01));
    float sign12 = sign(dot((p - sp1), n12));
    float sign20 = sign(dot((p - sp2), n20));

    if ((sign01 >= 0 && sign12 >= 0 && sign20 >= 0) || (sign01 <= 0 && sign12 <= 0 && sign20 <= 0)) {
        image[ipixel] = float4(color, 1.0);
        return;
    }

    // TODO: HW 2.1
}