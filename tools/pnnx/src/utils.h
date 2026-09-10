// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_UTILS_H
#define PNNX_UTILS_H

#include <stdint.h>

#include <string>
#include <vector>

namespace pnnx {

uint16_t read_le16(const unsigned char* p);
uint32_t read_le32(const unsigned char* p);

bool is_little_endian();

std::string trim_ascii_whitespace(const std::string& value);
std::string trim_ascii_whitespace(const std::vector<unsigned char>& value);

int scalar_type_to_pnnx(int type);

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

std::string float_to_string(float f);
std::string double_to_string(double d);

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size);

struct NumpyArray
{
    std::vector<int64_t> shape;
    std::string type;
    std::vector<char> data;
};

bool load_numpy_file(const char* path, NumpyArray& array, bool load_data = true);

} // namespace pnnx

#endif // PNNX_UTILS_H
