#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <vector>
#include <cstdint>

// Range clipping might be useful to implement here in the future --> minimize the effect of outliers on the quantization
// Useful when there is a large dataset of (e.g.) tensors to check through and generate ranges from (a calibration process)
std::pair<std::vector<int8_t>, float> symmetric_quantize_int8(const std::vector<float>& data);

std::vector<float> dequantize_symmetric_int8(const std::vector<int8_t>& quantized_data, float scale);

float calculate_quant_error(const std::vector<float>& original_data, const std::vector<int8_t>& quantized_data, float scale);

#endif // QUANTIZATION_H