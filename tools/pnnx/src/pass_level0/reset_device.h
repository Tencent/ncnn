// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

bool has_mutable_tensor_cast(const std::shared_ptr<torch::jit::Graph>& graph);

void reset_device(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device, bool preserve_dtype);

} // namespace pnnx
