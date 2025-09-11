// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_UTILS_H
#define PNNX_UTILS_H

#include <vector>
#include <string>

namespace pnnx {

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size);

//int load_npy_shape_and_dtype(const std::string& filename, std::vector<int64_t>& shape, std::string& dtype);
int load_npy_tensor(const std::string& npy_path,
                   std::vector<int64_t>& out_shape,
                   std::string& out_dtype,
                   std::vector<float>& out_data);

} // namespace pnnx

#endif // PNNX_UTILS_H
