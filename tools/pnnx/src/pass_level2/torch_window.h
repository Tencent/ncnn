// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_TORCH_WINDOW_H
#define PNNX_PASS_LEVEL2_TORCH_WINDOW_H

#include "ir.h"

namespace pnnx {

void fold_static_windows(Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_TORCH_WINDOW_H
