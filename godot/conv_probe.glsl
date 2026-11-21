#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int WIDTH = 256;
const int HEIGHT = 144;
const int INPUT_CHANNELS = 11;
const int OUTPUT_CHANNELS = 12;

layout(set = 0, binding = 0, std430) readonly buffer WeightBuffer {
    float weights[80152];
};

layout(set = 0, binding = 1, std430) readonly buffer InputBuffer {
    float input_values[405504];
};

layout(set = 0, binding = 2, std430) writeonly buffer OutputBuffer {
    float output_values[442368];
};

layout(push_constant, std430) uniform Params {
    int weight_offset;
    int bias_offset;
} params;

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (id.x >= WIDTH || id.y >= HEIGHT || id.z >= OUTPUT_CHANNELS) {
        return;
    }

    float value = weights[params.bias_offset + id.z];
    for (int input_channel = 0; input_channel < INPUT_CHANNELS; input_channel++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                ivec2 sample_position = id.xy + ivec2(kx, ky);
                if (sample_position.x < 0 || sample_position.x >= WIDTH ||
                    sample_position.y < 0 || sample_position.y >= HEIGHT) {
                    continue;
                }
                int input_index = input_channel * WIDTH * HEIGHT +
                                  sample_position.y * WIDTH + sample_position.x;
                int kernel_index = (id.z * INPUT_CHANNELS + input_channel) * 9 +
                                   (ky + 1) * 3 + (kx + 1);
                value += input_values[input_index] * weights[params.weight_offset + kernel_index];
            }
        }
    }
    int output_index = id.z * WIDTH * HEIGHT + id.y * WIDTH + id.x;
    output_values[output_index] = value;
}
