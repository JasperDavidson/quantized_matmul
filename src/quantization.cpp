#include "quantization.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>

// Range clipping might be useful to implement here in the future --> minimize the effect of outliers on the quantization
// Useful when there is a large dataset of (e.g.) tensors to check through and generate ranges from (a calibration process)
std::pair<std::vector<int8_t>, float> symmetric_quantize_int8(const std::vector<float>& data) {
    if (data.empty()) {
        return std::pair<std::vector<int8_t>, float>({}, 0);
    }

    float max_value = 0.0;

    for (float val : data) {
        max_value = std::max(max_value, std::fabs(val));
    }

    if (max_value == 0) {
        return std::pair<std::vector<int8_t>, float>(std::vector<int8_t>(data.size(), 0), 1);
    }


    float scale = (pow(2, 7) - 1) / max_value;
    std::vector<int8_t> quantized(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        quantized[i] = static_cast<int>(std::round(scale * data[i]));
    }

    return std::pair<std::vector<int8_t>, float>(quantized, scale);
}

std::vector<float> dequantize_symmetric_int8(const std::vector<int8_t>& quantized_data, float scale) {
    std::vector<float> dequantized_data(quantized_data.size());

    for (size_t i = 0; i < quantized_data.size(); ++i) {
        dequantized_data[i] = static_cast<float>(quantized_data[i]) / scale;
    }

    return dequantized_data;
}

float calculate_quant_error(const std::vector<float>& original_data, const std::vector<int8_t>& quantized_data, float scale) {
    float quantization_error = 0;

    std::vector<float> dequantized_data = dequantize_symmetric_int8(quantized_data, scale);

    for (size_t i = 0; i < original_data.size(); ++i) {
        quantization_error += pow(original_data[i] - static_cast<float>(dequantized_data[i]), 2);
    }

    return quantization_error / original_data.size();
}
