#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int WIDTH = 256;
const int HEIGHT = 144;
const int PIXELS = WIDTH * HEIGHT;
const int MAX_INPUT_FLOATS = 442368;

layout(set = 0, binding = 0) uniform sampler2D color_tex;
layout(set = 0, binding = 1) uniform sampler2D depth_tex;
layout(set = 0, binding = 2) uniform sampler2D normal_tex;
layout(set = 0, binding = 3, std430) writeonly buffer InputBuffer {
    float input_values[MAX_INPUT_FLOATS];
};

layout(push_constant, std430) uniform Params {
    float near_plane;
    float far_plane;
    vec2 light_view_xy;
    float light_view_z;
    float padding0;
    vec2 render_size;
} params;

float linear_to_srgb(float value) {
    value = max(value, 0.0);
    return value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    if (xy.x >= WIDTH || xy.y >= HEIGHT) return;

    vec2 uv = (vec2(xy) + vec2(0.5)) / vec2(WIDTH, HEIGHT);
    vec3 rgb = texture(color_tex, uv).rgb;
    float device_depth = texture(depth_tex, uv).r;
    vec3 packed_normal = texture(normal_tex, uv).xyz;
    int pixel = xy.y * WIDTH + xy.x;

    for (int channel = 0; channel < 3; channel++) {
        float component = channel == 0 ? rgb.r : (channel == 1 ? rgb.g : rgb.b);
        input_values[channel * PIXELS + pixel] = linear_to_srgb(component);
    }

    bool valid = device_depth > 0.0;
    float meters = valid
        ? params.near_plane * params.far_plane /
          (params.near_plane + device_depth * (params.far_plane - params.near_plane))
        : 0.0;
    input_values[3 * PIXELS + pixel] = clamp(meters / 10.0, 0.0, 1.0);

    vec3 normal = valid ? normalize(packed_normal * 2.0 - 1.0) : vec3(0.0);
    input_values[4 * PIXELS + pixel] = normal.x;
    input_values[5 * PIXELS + pixel] = normal.y;
    input_values[6 * PIXELS + pixel] = normal.z;
    input_values[7 * PIXELS + pixel] = valid ? 1.0 : 0.0;
    input_values[8 * PIXELS + pixel] = params.light_view_xy.x;
    input_values[9 * PIXELS + pixel] = params.light_view_xy.y;
    input_values[10 * PIXELS + pixel] = params.light_view_z;
}
