// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void reset_device(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device);

} // namespace pnnx
