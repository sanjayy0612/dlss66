#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int WIDTH = 256;
const int HEIGHT = 144;
const int PIXELS = WIDTH * HEIGHT;
const int MAX_ACTIVATION_FLOATS = 442368;

layout(set = 0, binding = 0, std430) readonly buffer OutputBuffer {
    float output_values[MAX_ACTIVATION_FLOATS];
};
layout(rgba16f, set = 0, binding = 1) uniform image2D color_image;

float srgb_to_linear(float value) {
    value = max(value, 0.0);
    return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}

void main() {
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    ivec2 output_size = imageSize(color_image);
    if (xy.x >= output_size.x || xy.y >= output_size.y) return;
    vec2 source = (vec2(xy) + vec2(0.5)) * vec2(WIDTH, HEIGHT) / vec2(output_size) - vec2(0.5);
    ivec2 low = ivec2(floor(source));
    ivec2 high = low + ivec2(1);
    vec2 blend = fract(source);
    low = clamp(low, ivec2(0), ivec2(WIDTH - 1, HEIGHT - 1));
    high = clamp(high, ivec2(0), ivec2(WIDTH - 1, HEIGHT - 1));
    int p00 = low.y * WIDTH + low.x;
    int p10 = low.y * WIDTH + high.x;
    int p01 = high.y * WIDTH + low.x;
    int p11 = high.y * WIDTH + high.x;
    vec3 c00 = vec3(output_values[p00], output_values[PIXELS + p00], output_values[2 * PIXELS + p00]);
    vec3 c10 = vec3(output_values[p10], output_values[PIXELS + p10], output_values[2 * PIXELS + p10]);
    vec3 c01 = vec3(output_values[p01], output_values[PIXELS + p01], output_values[2 * PIXELS + p01]);
    vec3 c11 = vec3(output_values[p11], output_values[PIXELS + p11], output_values[2 * PIXELS + p11]);
    vec3 neural_rgb = mix(mix(c00, c10, blend.x), mix(c01, c11, blend.x), blend.y);
    vec4 color = imageLoad(color_image, xy);
    color.rgb = vec3(srgb_to_linear(neural_rgb.r),
        srgb_to_linear(neural_rgb.g), srgb_to_linear(neural_rgb.b));
    imageStore(color_image, xy, color);
}
