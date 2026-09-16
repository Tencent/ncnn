// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_NN_LINEAR_H
#define PNNX_PASS_NCNN_NN_LINEAR_H

#include "ir.h"

namespace pnnx {

namespace ncnn {

void convert_nn_Linear_3d_flatten(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_NN_LINEAR_H
