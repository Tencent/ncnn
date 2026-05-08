// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reset_device.h"

namespace pnnx {

void reset_device(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device)
{
    for (torch::jit::Node* n : graph->nodes())
    {
        if (n->kind().is_aten())
        {
            if (n->hasNamedInput("dtype"))
            {
                torch::jit::Node* dtype_node = n->namedInput("dtype")->node();

                if (dtype_node->hasAttribute(torch::jit::attr::value))
                {
                    // change dtype=half to dtype=float
                    if (dtype_node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::i && dtype_node->i(torch::jit::attr::value) == 5)
                    {
                        dtype_node->i_(torch::jit::attr::value, 6);
                    }
                    // change dtype=bfloat16 to dtype=float
                    if (dtype_node->kindOf(torch::jit::attr::value) == torch::jit::AttributeKind::i && dtype_node->i(torch::jit::attr::value) == 15)
                    {
                        dtype_node->i_(torch::jit::attr::value, 6);
                    }
                }
            }

            if (n->hasNamedInput("device"))
            {
                torch::jit::Node* device_node = n->namedInput("device")->node();

                device_node->s_(torch::jit::attr::value, (device == "gpu") ? "cuda" : "cpu");
            }
        }
    }
}

} // namespace pnnx
