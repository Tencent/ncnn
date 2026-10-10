// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_TORCH_WEIGHT_NORM_H
#define PNNX_PASS_LEVEL2_TORCH_WEIGHT_NORM_H

#include "ir.h"

namespace pnnx {

void fold_static_weight_norm(Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_TORCH_WEIGHT_NORM_H
