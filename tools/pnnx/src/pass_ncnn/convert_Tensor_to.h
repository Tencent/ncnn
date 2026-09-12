// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_CONVERT_TENSOR_TO_H
#define PNNX_PASS_NCNN_CONVERT_TENSOR_TO_H

#include "ir.h"

namespace pnnx {

namespace ncnn {

void convert_Tensor_to(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_CONVERT_TENSOR_TO_H
