// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_PT2_H
#define PNNX_PASS_PT2_H

#include "ir.h"
#include "pass_pt2/pt2_tensor.h"

namespace pnnx {

int pass_pt2(Graph& graph, const std::map<Operator*, Pt2TensorMeta>& layout_sources);

} // namespace pnnx

#endif // PNNX_PASS_PT2_H
