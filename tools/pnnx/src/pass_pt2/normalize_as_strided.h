// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NORMALIZE_PT2_AS_STRIDED_H
#define PNNX_NORMALIZE_PT2_AS_STRIDED_H

#include "pt2_tensor.h"

namespace pnnx {

int normalize_pt2_layout_query(Operator* op, const Pt2TensorMeta& source);

int normalize_pt2_as_strided(Graph& graph, Operator* op, const Pt2TensorMeta& source);

} // namespace pnnx

#endif // PNNX_NORMALIZE_PT2_AS_STRIDED_H
