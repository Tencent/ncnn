// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_LINALG_VECTOR_NORM_H
#define PNNX_PASS_NCNN_LINALG_VECTOR_NORM_H

#include "ir.h"

namespace pnnx {

namespace ncnn {

void convert_aten_linalg_vector_norm(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_LINALG_VECTOR_NORM_H
