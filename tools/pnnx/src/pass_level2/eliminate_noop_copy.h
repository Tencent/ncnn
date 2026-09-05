// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

namespace pnnx {

// eliminate functionalization artifacts that are pure identity copies
//  aten::lift_fresh_copy / aten::detach_
void eliminate_noop_copy(Graph& graph);

} // namespace pnnx
