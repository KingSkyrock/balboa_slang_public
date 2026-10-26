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
    
    //get vertex position + normal + uv
    float3 position = vertices[vid];
    float3 normal = vertex_normals[vid];
    float2 uv = vertex_uvs[vid];

    //calculate positions after transform
    output.position = mul(MVP, float4(position, 1.0)); //use 1.0 at the end so we can do transform
    output.world_position = mul(model_matrix, float4(position, 1.0)).xyz;
    output.world_normal = mul(normal_matrix, normal);
    output.uv = uv;

    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.7 TODO: implement
    // access texture
    float3 Kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;

    float3 surface_normal = normalize(input.world_normal);
    float3 light_direction = normalize(light_dir);
    float3 view_direction = normalize(camera_position - input.world_position); //direction from surface to camera

    float3 diffuse = Kd * light_color * max(dot(surface_normal, light_direction), 0.0);
    
    //get specular
    float3 r = reflect(-light_direction, surface_normal);
    float3 specular = float3(0.0, 0.0, 0.0);
    if (dot(surface_normal, light_direction) > 0.0) {
        specular = Ks * light_color * pow(max(0.0, dot(r, view_direction)), exponent);
    }
    
    float3 ambient = 0.2 * Kd;

    float3 color = ambient + diffuse + specular;
    return float4(color, 1.0);
}
