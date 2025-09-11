// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_UTILS_H
#define PNNX_UTILS_H

#include <string>
#include <vector>
namespace pnnx {

unsigned short float32_to_float16(float value);

float float16_to_float32(unsigned short value);

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size);

void parse_dtype(char* dtype, std::vector<std::string>& types, char& endian);

void parse_numpy_header(char* header_str,
                        std::vector<std::vector<int64_t> >& shapes,
                        std::vector<std::string>& types,
                        bool& fortran_order,
                        char& endian);

char get_system_endian();

void swap_bytes(void* buffer, size_t type_size, size_t content_len);

size_t get_type_size_from_input_type(const char* str);

void convert_to_c_order(void* src, const std::vector<int64_t>& shape, size_t type_size, size_t content_len);

void parse_numpy_file(const char* path,
                      std::vector<std::vector<int64_t> >& shapes,
                      std::vector<std::string>& types,
                      std::vector<std::vector<char> >& contents);

} // namespace pnnx

#endif // PNNX_UTILS_H
