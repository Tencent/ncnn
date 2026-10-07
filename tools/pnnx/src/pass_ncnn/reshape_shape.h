// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NCNN_RESHAPE_SHAPE_H
#define PNNX_NCNN_RESHAPE_SHAPE_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

// dimensions are in logical order, before removing the native batch axis
std::string logical_dim_reference(const Operand* operand, int reference_index, int axis);
std::vector<std::string> logical_shape(const Operand* operand, int reference_index);
std::vector<std::string> split_shape_expression(const std::string& expression);
std::string shape_product(const std::vector<std::string>& dimensions);
std::string shape_quotient(const std::string& numerator, const std::string& denominator);
void write_reshape_shape(Operator* op, std::vector<std::string> shape);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_NCNN_RESHAPE_SHAPE_H
