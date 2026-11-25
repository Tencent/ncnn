// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NCNN_RESHAPE_SHAPE_H
#define PNNX_NCNN_RESHAPE_SHAPE_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

int get_ncnn_batch_axis(const Operand* operand);

// dimensions are in logical order, before removing the native batch axis
std::string get_logical_dim_expr(const Operand* operand, int reference_index, int axis);
// describe a runtime shape without creating a graph operand
std::vector<std::string> get_logical_shape_expr(int rank, int native_batch_axis, int reference_index);
std::vector<std::string> get_logical_shape_expr(const Operand* operand, int reference_index);
std::vector<std::string> split_shape_expression(const std::string& expression);
std::string make_shape_product_expr(const std::vector<std::string>& dimensions);
bool resolve_reshape_params(const std::vector<std::string>& input_shape, int input_axis, std::vector<std::string> shape, int output_axis, int input_count, std::map<std::string, Parameter>& params);
bool resolve_reshape_params(const Operator* op, const std::vector<std::string>& shape, std::map<std::string, Parameter>& params);
// write parameters and remove unused shape reference inputs
void write_reshape_params(Operator* op, const std::map<std::string, Parameter>& params);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_NCNN_RESHAPE_SHAPE_H
