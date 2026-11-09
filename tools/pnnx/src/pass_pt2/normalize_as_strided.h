// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NORMALIZE_PT2_AS_STRIDED_H
#define PNNX_NORMALIZE_PT2_AS_STRIDED_H

#include "load_pt2/pt2_tensor.h"

namespace pnnx {

bool normalize_pt2_layout_query(Operator* op, const Pt2TensorMeta& source, std::string& error);

bool normalize_pt2_as_strided(Graph& graph, Operator* op, const Pt2TensorMeta& source, std::string& error);

} // namespace pnnx

#endif // PNNX_NORMALIZE_PT2_AS_STRIDED_H
