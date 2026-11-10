// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reset_device.h"

namespace pnnx {

void reset_device(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device)
{
    for (torch::jit::Node* n : graph->nodes())
    {
        if (n->kind().is_aten() && n->hasNamedInput("device"))
        {
            torch::jit::Node* device_node = n->namedInput("device")->node();
            device_node->s_(torch::jit::attr::value, (device == "gpu") ? "cuda" : "cpu");
        }
    }
}

} // namespace pnnx
