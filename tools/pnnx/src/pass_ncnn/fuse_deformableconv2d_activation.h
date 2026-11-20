// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FUSE_DEFORMABLECONV2D_ACTIVATION_H
#define FUSE_DEFORMABLECONV2D_ACTIVATION_H

#include "ir.h"

namespace pnnx {

namespace ncnn {

void fuse_deformableconv2d_activation(Graph& graph);

} // namespace ncnn

} // namespace pnnx

#endif // FUSE_DEFORMABLECONV2D_ACTIVATION_H
