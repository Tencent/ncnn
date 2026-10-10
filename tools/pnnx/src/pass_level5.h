// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL5_H
#define PNNX_PASS_LEVEL5_H

#include "ir.h"

namespace pnnx {

void pass_level5(Graph& g, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL5_H
