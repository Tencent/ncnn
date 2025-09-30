// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_UTILS_H
#define PNNX_UTILS_H

#include <string>
#include <vector>

namespace pnnx {

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

std::string float_to_string(float f);
std::string double_to_string(double d);

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size);

} // namespace pnnx

#endif // PNNX_UTILS_H
