// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL1_H
#define PNNX_PASS_LEVEL1_H

#include "ir.h"

namespace pnnx {

void pass_level1(const torch::jit::Module& mod, const std::shared_ptr<torch::jit::Graph>& g, const std::vector<std::string>& module_operators, Graph& pg);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL1_H
