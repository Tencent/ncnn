// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_F_LINEAR_H
#define PNNX_PASS_NCNN_F_LINEAR_H

#include "ir.h"

namespace pnnx {

namespace ncnn {

void convert_aten_F_linear(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_F_LINEAR_H
