StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_colors;
uniform float4x4 MVP;

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid : SV_VertexID) {
    VertexOutput output;
    // HW 2.6: your code here

    //get vertex position
    float3 position = vertices[vid];
    
    output.color = vertex_colors[vid];

    //calculate position after transform
    output.position = mul(MVP, float4(position, 1.0)); //use 1.0 at the end so we can do transform
    
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.6: your code here
    return float4(input.color, 1.0);
}
