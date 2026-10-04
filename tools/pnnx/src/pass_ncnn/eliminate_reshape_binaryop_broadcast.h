// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_NCNN_ELIMINATE_RESHAPE_BINARYOP_BROADCAST_H
#define PNNX_PASS_NCNN_ELIMINATE_RESHAPE_BINARYOP_BROADCAST_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void eliminate_reshape_binaryop_broadcast(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_PASS_NCNN_ELIMINATE_RESHAPE_BINARYOP_BROADCAST_H
