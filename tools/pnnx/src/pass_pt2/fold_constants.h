// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_FOLD_PT2_CONSTANTS_H
#define PNNX_FOLD_PT2_CONSTANTS_H

#include "ir.h"

namespace pnnx {

int fold_pt2_constants(Graph& graph, const std::map<Operator*, std::string>& targets, const std::map<const Operand*, double>& floating_constants);

} // namespace pnnx

#endif // PNNX_FOLD_PT2_CONSTANTS_H
