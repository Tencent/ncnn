// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_alias.h"

#include "fuse_module_pass.h"

#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

TorchAliasKind torch_alias_kind(const torch::jit::Node* node)
{
    if (node->kind() != c10::prim::CallMethod)
        return torch_alias_kind(node->kind().toDisplayString());

    const auto class_type = node->input(0)->type()->cast<torch::jit::ClassType>();
    if (!class_type)
        return TorchAliasNone;
    const std::string type = torch::jit::removeTorchMangle(class_type->str());
    for (const auto& pass : get_global_pnnx_fuse_module_passes())
        if (type == pass->match_type_str())
            return torch_alias_kind(std::string(pass->type_str()));
    return TorchAliasNone;
}

torch::jit::Value* torch_alias_input(const torch::jit::Node* node)
{
    return node->input(node->kind() == c10::prim::CallMethod ? 1 : 0);
}

} // namespace pnnx
