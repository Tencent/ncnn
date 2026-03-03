// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void inline_block(std::shared_ptr<torch::jit::Graph>& graph, const std::vector<std::string>& module_operators);

} // namespace pnnx
