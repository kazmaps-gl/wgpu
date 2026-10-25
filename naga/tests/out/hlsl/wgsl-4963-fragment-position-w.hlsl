struct VertexOutput {
    float4 position : SV_Position;
    float clip_w : LOC0;
};

struct VertexOutput_vs_main {
    float clip_w_1 : LOC0;
    float4 position_1 : SV_Position;
};

struct FragmentInput_fs_struct {
    float clip_w_2 : LOC0;
    float4 position_2 : SV_Position;
};

struct FragmentInput_fs_argument {
    float clip_w_3 : LOC0;
    float4 position_3 : SV_Position;
};

VertexOutput ConstructVertexOutput(float4 arg0, float arg1) {
    VertexOutput ret = (VertexOutput)0;
    ret.position = arg0;
    ret.clip_w = arg1;
    return ret;
}

VertexOutput_vs_main vs_main(uint index : SV_VertexID)
{
    float4 position_4 = float4(float(index), 0.0, 0.5, 2.0);
    const VertexOutput vertexoutput = ConstructVertexOutput(position_4, position_4.w);
    const VertexOutput_vs_main vertexoutput_1 = { vertexoutput.clip_w, vertexoutput.position };
    return vertexoutput_1;
}

float4 fs_struct(FragmentInput_fs_struct fragmentinput_fs_struct) : SV_Target0
{
    VertexOutput in_ = { float4(fragmentinput_fs_struct.position_2.xyz, 1.0 / fragmentinput_fs_struct.position_2.w), fragmentinput_fs_struct.clip_w_2 };
    return float4(in_.position.xyz, (in_.clip_w * in_.position.w));
}

float4 fs_argument(FragmentInput_fs_argument fragmentinput_fs_argument) : SV_Target0
{
    float clip_w = fragmentinput_fs_argument.clip_w_3;
    float4 position = float4(fragmentinput_fs_argument.position_3.xyz, 1.0 / fragmentinput_fs_argument.position_3.w);
    return float4(position.xyz, (clip_w * position.w));
}
