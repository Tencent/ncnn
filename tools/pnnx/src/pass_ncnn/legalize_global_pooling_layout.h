// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_LEGALIZE_GLOBAL_POOLING_LAYOUT_H
#define PNNX_PASS_NCNN_LEGALIZE_GLOBAL_POOLING_LAYOUT_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void legalize_global_pooling_layout(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_LEGALIZE_GLOBAL_POOLING_LAYOUT_H
