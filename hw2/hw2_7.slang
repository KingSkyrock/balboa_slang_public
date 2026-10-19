StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;
uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;
Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.7 TODO: implement
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.7 TODO: implement
    return float4(1.0);
}
