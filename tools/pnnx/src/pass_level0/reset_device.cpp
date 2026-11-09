// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reset_device.h"

#include "torch_alias.h"

#include <map>
#include <set>

namespace pnnx {

bool has_mutable_tensor_cast(const std::shared_ptr<torch::jit::Graph>& graph)
{
    std::map<const torch::jit::Value*, const torch::jit::Value*> aliases;
    std::set<const torch::jit::Value*> cast_roots;
    auto root = [&](const torch::jit::Value* value) {
        while (aliases.count(value))
            value = aliases.at(value);
        return value;
    };
    for (const torch::jit::Node* n : graph->nodes())
    {
        const std::string type = n->kind().toDisplayString();
        if (n->inputs().empty())
            continue;
        const bool mutation = torch_mutation_kind(type) != TorchMutationNone;
        const bool cast = type == "aten::to" || type == "aten::type_as";
        const bool container = type == "prim::ListConstruct" || type == "prim::TupleConstruct"
                               || type == "prim::ListUnpack" || type == "prim::TupleUnpack"
                               || type == "aten::__getitem__" || type == "prim::TupleIndex";
        // Only later writes can observe the aliasing changed by a cast,
        // either through the converted value or through its source.
        if (mutation && cast_roots.count(root(n->input(0))))
            return true;
        if (!container && !mutation && torch_alias_kind(n) == TorchAliasNone)
            continue;

        const torch::jit::Value* parent = root(torch_alias_input(n));
        if (cast)
            cast_roots.insert(parent);
        if (container)
        {
            for (const torch::jit::Value* input : n->inputs())
            {
                const torch::jit::Value* other = root(input);
                if (other != parent)
                {
                    aliases[other] = parent;
                    if (cast_roots.erase(other))
                        cast_roots.insert(parent);
                }
            }
        }
        for (const torch::jit::Value* output : n->outputs())
            aliases[output] = parent;
    }
    return false;
}

void reset_device(std::shared_ptr<torch::jit::Graph>& graph, const std::string& device, bool preserve_dtype)
{
    for (torch::jit::Node* n : graph->nodes())
    {
        if (n->kind().is_aten())
        {
            if (!preserve_dtype && n->hasNamedInput("dtype"))
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
