// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NORMALIZE_PT2_DTYPE_H
#define PNNX_NORMALIZE_PT2_DTYPE_H

#include "ir.h"

namespace pnnx {

int normalize_pt2_scalar_type(int type);
void normalize_pt2_tensor_dtype(Graph& graph);

int normalize_pt2_dtype(Graph& graph);

} // namespace pnnx

#endif // PNNX_NORMALIZE_PT2_DTYPE_H
