// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/script.h>

namespace pnnx {

void constant_unpooling(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace pnnx
