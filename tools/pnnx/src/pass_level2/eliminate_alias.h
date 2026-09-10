// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL2_ELIMINATE_ALIAS_H
#define PNNX_PASS_LEVEL2_ELIMINATE_ALIAS_H

#include "ir.h"

namespace pnnx {

void eliminate_alias(Graph& graph);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL2_ELIMINATE_ALIAS_H
