// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL5_FUSE_RESHAPE_ACTIVATION_RESHAPE_H
#define PNNX_PASS_LEVEL5_FUSE_RESHAPE_ACTIVATION_RESHAPE_H

#include "ir.h"

namespace pnnx {

void fuse_reshape_activation_reshape(Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL5_FUSE_RESHAPE_ACTIVATION_RESHAPE_H
