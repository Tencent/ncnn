// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_PASS_LEVEL0_H
#define PNNX_PASS_LEVEL0_H

#include <torch/script.h>
#include "ir.h"

namespace pnnx {

void pass_level0(const torch::jit::Module& mod, std::shared_ptr<torch::jit::Graph>& g, const std::vector<at::Tensor>& input_tensors, const std::vector<at::Tensor>& input_tensors2, const std::vector<std::string>& module_operators, const std::string& ptpath, const std::string& device, std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath);

} // namespace pnnx

#endif // PNNX_PASS_LEVEL0_H
