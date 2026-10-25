// In a fragment shader WGSL's `position.w` is 1/w of the clip position, while D3D's
// `SV_Position.w` is w itself: the HLSL backend must invert w of a fragment input, whether
// the position is a member of a struct argument or an argument of its own. The vertex
// output keeps the position as it is.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) clip_w: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    let position = vec4(f32(index), 0.0, 0.5, 2.0);
    return VertexOutput(position, position.w);
}

@fragment
fn fs_struct(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.position.xyz, in.clip_w * in.position.w);
}

@fragment
fn fs_argument(@location(0) clip_w: f32, @builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(position.xyz, clip_w * position.w);
}
