StructuredBuffer<float3> positions;
StructuredBuffer<uint2> edges;
StructuredBuffer<uint3> triangles;
StructuredBuffer<float3> static_positions;
StructuredBuffer<float3> static_normals;
StructuredBuffer<float3> static_albedos;
uniform float4x4 MVP;
uniform float2 viewport_size;
uniform float particle_radius;
uniform int cloth_grid_width;
uniform int cloth_grid_height;

struct ParticleOutput {
    float4 position : SV_Position;
    float2 disk_xy : TEXCOORD0;
};

[shader("vertex")]
ParticleOutput particle_vertex_main(uint vid : SV_VertexID,
                                  uint iid : SV_InstanceID) {
    // Similar to https://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/
    float2 offsets[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2( 1.0,  1.0),
        float2(-1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0)
    };
    float2 offset = offsets[vid];
    float4 clip = mul(MVP, float4(positions[iid], 1.0));
    // Shift the vertex by the offset, adjusted by
    // perspective projection through multiplication of clip.w
    // and the viewport size, so that the size of the particle
    // on screen is constant.
    clip.xy += clip.w * offset * (2.0 * particle_radius / viewport_size);
    ParticleOutput output;
    output.position = clip;
    output.disk_xy = offset;
    return output;
}

[shader("fragment")]
float4 particle_fragment_main(ParticleOutput input) : SV_Target {
    if (dot(input.disk_xy, input.disk_xy) > 1.0) {
        // Hit the quad but didn't hit the disk
        discard;
    }
    return float4(0.99, 0.15, 0.15, 1.0);
}

struct LineOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
LineOutput line_vertex_main(uint vid : SV_VertexID,
                            uint iid : SV_InstanceID) {
    uint2 edge = edges[iid];
    uint particle_index = (vid == 0) ? edge.x : edge.y;
    LineOutput output;
    output.position = mul(MVP, float4(positions[particle_index], 1));
    return output;
}

[shader("fragment")]
float4 line_fragment_main(LineOutput input) : SV_Target {
    return float4(0.15, 0.99, 0.15, 1.0);
}

struct SurfaceOutput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float3 albedo : TEXCOORD1;
};

[shader("vertex")]
SurfaceOutput cloth_vertex_main(uint vid : SV_VertexID) {
    uint3 tri = triangles[vid / 3];
    uint pid = tri[vid % 3];

    float3 p = positions[pid];

    uint x = pid % cloth_grid_width;
    uint y = pid / cloth_grid_width;

    uint x0 = max(int(x) - 1, 0);
    uint x1 = min(x + 1, cloth_grid_width - 1);
    uint y0 = max(int(y) - 1, 0);
    uint y1 = min(y + 1, cloth_grid_height - 1);

    float3 dx = positions[y * cloth_grid_width + x1]
              - positions[y * cloth_grid_width + x0];
    float3 dy = positions[y1 * cloth_grid_width + x]
              - positions[y0 * cloth_grid_width + x];
    float3 n = cross(dx, dy);
    if (length(n) > 1e-12) {
        n = normalize(n);
    } else{
        n = float3(0, 0, -1);
    }

    SurfaceOutput output;
    output.position = mul(MVP, float4(p, 1.0));
    output.normal = n;
    output.albedo = float3(0.25, 0.8, 0.8);
    return output;
}

[shader("vertex")]
SurfaceOutput static_vertex_main(uint vid : SV_VertexID) {
    SurfaceOutput output;
    output.position = mul(MVP, float4(static_positions[vid], 1.0));
    output.normal = static_normals[vid];
    output.albedo = static_albedos[vid];
    return output;
}

[shader("fragment")]
float4 surface_fragment_main(SurfaceOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.4, 0.8, -0.6));
    // Two-sided lighting
    float cosine = abs(dot(normalize(input.normal), light_dir));
    float intensity = 0.30 + 0.70 * cosine;
    return float4(input.albedo * intensity, 1.0);
}

