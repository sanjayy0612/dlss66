#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int WIDTH = 256;
const int HEIGHT = 144;
const int MAX_ACTIVATION_FLOATS = 442368;

layout(set = 0, binding = 0, std430) readonly buffer InputBuffer {
    float input_values[MAX_ACTIVATION_FLOATS];
};
layout(set = 0, binding = 1, std430) readonly buffer MaskBuffer {
    float mask_logits[MAX_ACTIVATION_FLOATS];
};
layout(set = 0, binding = 2, std430) readonly buffer StrengthBuffer {
    float strengths[MAX_ACTIVATION_FLOATS];
};
layout(set = 0, binding = 3, std430) writeonly buffer OutputBuffer {
    float output_values[MAX_ACTIVATION_FLOATS];
};

layout(push_constant, std430) uniform Params {
    float mask_bias;
    float max_delta;
} params;

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (id.x >= WIDTH || id.y >= HEIGHT || id.z >= 3) {
        return;
    }
    int pixel = id.y * WIDTH + id.x;
    float logit = mask_logits[pixel];
    float mask = logit >= params.mask_bias ? 1.0 / (1.0 + exp(-(logit - params.mask_bias))) : 0.0;
    float strength = tanh(strengths[id.z * WIDTH * HEIGHT + pixel]);
    float rgb = input_values[id.z * WIDTH * HEIGHT + pixel];
    output_values[id.z * WIDTH * HEIGHT + pixel] = clamp(rgb + params.max_delta * mask * strength, 0.0, 1.0);
}
