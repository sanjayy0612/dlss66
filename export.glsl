#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D depth_tex;
layout(set = 0, binding = 1) uniform sampler2D normal_tex;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D output_tex;
void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_tex);
    if (xy.x >= size.x || xy.y >= size.y) return;
    vec2 uv = (vec2(xy) + vec2(0.5)) / vec2(size);
    float depth = texture(depth_tex, uv).r;
    vec4 packed_normal = texture(normal_tex, uv);
    vec3 normal = depth <= 0.0 ? vec3(0.0) : normalize(packed_normal.xyz * 2.0 - 1.0);
    imageStore(output_tex, xy, vec4(depth, normal));
}
