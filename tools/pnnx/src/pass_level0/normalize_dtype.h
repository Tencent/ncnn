// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

int normalize_dtype(std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod, const std::vector<at::Tensor>& input_tensors);

} // namespace pnnx
