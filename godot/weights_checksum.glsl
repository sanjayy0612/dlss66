#[compute]
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer WeightBuffer {
    float weights[80152];
};

layout(set = 0, binding = 1, std430) writeonly buffer SummaryBuffer {
    float summary[4];
};

void main() {
    float sum = 0.0;
    float absolute_sum = 0.0;
    float maximum_absolute = 0.0;
    for (int index = 0; index < 80152; index++) {
        float value = weights[index];
        sum += value;
        absolute_sum += abs(value);
        maximum_absolute = max(maximum_absolute, abs(value));
    }
    summary[0] = sum;
    summary[1] = absolute_sum;
    summary[2] = maximum_absolute;
    summary[3] = 80152.0;
}
