// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

namespace pnnx {

void fold_constants(Graph& graph, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath);

} // namespace pnnx
