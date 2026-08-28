// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_FUSE_RECURRENT_H
#define PNNX_PASS_LEVEL2_FUSE_RECURRENT_H

#include "ir.h"

namespace pnnx {

void fuse_recurrent(Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_FUSE_RECURRENT_H
