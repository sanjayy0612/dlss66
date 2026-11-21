#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const int MAX_ACTIVATION_FLOATS = 442368;
const int MAX_WEIGHT_FLOATS = 80152;
const int MODE_DIRECT = 0;
const int MODE_AVERAGE_POOL_2 = 1;
const int MODE_BILINEAR_UPSAMPLE = 2;

layout(set = 0, binding = 0, std430) readonly buffer WeightBuffer {
    float weights[MAX_WEIGHT_FLOATS];
};
layout(set = 0, binding = 1, std430) readonly buffer InputABuffer {
    float input_a[MAX_ACTIVATION_FLOATS];
};
layout(set = 0, binding = 2, std430) readonly buffer InputBBuffer {
    float input_b[MAX_ACTIVATION_FLOATS];
};
layout(set = 0, binding = 3, std430) writeonly buffer OutputBuffer {
    float output_values[MAX_ACTIVATION_FLOATS];
};

layout(push_constant, std430) uniform Params {
    int width;
    int height;
    int a_width;
    int a_height;
    int a_channels;
    int b_channels;
    int output_channels;
    int mode;
    int apply_relu;
    int weight_offset;
    int bias_offset;
} params;

float load_a(int channel, int x, int y) {
    int index = channel * params.a_width * params.a_height + y * params.a_width + x;
    return input_a[index];
}

float sample_a(int channel, ivec2 position) {
    if (position.x < 0 || position.x >= params.width ||
        position.y < 0 || position.y >= params.height) {
        return 0.0;
    }

    if (params.mode == MODE_AVERAGE_POOL_2) {
        int base_x = position.x * 2;
        int base_y = position.y * 2;
        return 0.25 * (load_a(channel, base_x, base_y) +
                       load_a(channel, base_x + 1, base_y) +
                       load_a(channel, base_x, base_y + 1) +
                       load_a(channel, base_x + 1, base_y + 1));
    }

    if (params.mode == MODE_BILINEAR_UPSAMPLE) {
        vec2 source_position = (vec2(position) + vec2(0.5)) *
                               vec2(params.a_width, params.a_height) /
                               vec2(params.width, params.height) - vec2(0.5);
        ivec2 low = ivec2(floor(source_position));
        ivec2 high = low + ivec2(1);
        vec2 fraction = fract(source_position);
        low = clamp(low, ivec2(0), ivec2(params.a_width - 1, params.a_height - 1));
        high = clamp(high, ivec2(0), ivec2(params.a_width - 1, params.a_height - 1));
        float top = mix(load_a(channel, low.x, low.y), load_a(channel, high.x, low.y), fraction.x);
        float bottom = mix(load_a(channel, low.x, high.y), load_a(channel, high.x, high.y), fraction.x);
        return mix(top, bottom, fraction.y);
    }

    return load_a(channel, position.x, position.y);
}

float sample_b(int channel, ivec2 position) {
    int index = channel * params.width * params.height + position.y * params.width + position.x;
    return input_b[index];
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID);
    if (id.x >= params.width || id.y >= params.height || id.z >= params.output_channels) {
        return;
    }

    int input_channels = params.a_channels + params.b_channels;
    float value = weights[params.bias_offset + id.z];
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            ivec2 position = id.xy + ivec2(kx, ky);
            for (int channel = 0; channel < params.a_channels; channel++) {
                int kernel_index = ((id.z * input_channels + channel) * 3 + (ky + 1)) * 3 + (kx + 1);
                value += sample_a(channel, position) * weights[params.weight_offset + kernel_index];
            }
            for (int channel = 0; channel < params.b_channels; channel++) {
                int joined_channel = params.a_channels + channel;
                int kernel_index = ((id.z * input_channels + joined_channel) * 3 + (ky + 1)) * 3 + (kx + 1);
                float sample_value = 0.0;
                if (position.x >= 0 && position.x < params.width &&
                    position.y >= 0 && position.y < params.height) {
                    sample_value = sample_b(channel, position);
                }
                value += sample_value * weights[params.weight_offset + kernel_index];
            }
        }
    }
    if (params.apply_relu != 0) {
        value = max(value, 0.0);
    }
    int output_index = id.z * params.width * params.height + id.y * params.width + id.x;
    output_values[output_index] = value;
}
