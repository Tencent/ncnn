// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"

namespace pnnx {

void fuse_expression(Graph& graph, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath);

} // namespace pnnx
