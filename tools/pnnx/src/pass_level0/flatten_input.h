// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void flatten_input(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace pnnx
