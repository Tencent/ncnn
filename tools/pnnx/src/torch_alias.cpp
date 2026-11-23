// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_alias.h"

namespace pnnx {

static bool is_dropout(const std::string& type)
{
    return type == "aten::dropout" || type == "aten::dropout_"
           || type == "aten::feature_dropout" || type == "aten::feature_dropout_"
           || type == "aten::alpha_dropout" || type == "aten::alpha_dropout_"
           || type == "aten::feature_alpha_dropout" || type == "aten::feature_alpha_dropout_";
}

TorchAliasKind torch_alias_kind(const std::string& type)
{
    if (type == "aten::detach_" || type == "nn.Dropout" || type == "nn.AlphaDropout")
        return TorchAliasIdentity;

    if (type == "aten::slice" || type == "aten::select" || type == "aten::view"
            || type == "aten::transpose" || type == "aten::t" || type == "aten::permute"
            || type == "aten::squeeze" || type == "aten::unsqueeze"
            || type == "aten::detach" || type == "aten::alias")
        return TorchAliasView;

    if (type == "aten::reshape" || type == "aten::reshape_as" || type == "aten::flatten" || type == "aten::view_as"
            || type == "aten::expand" || type == "aten::expand_as" || type == "aten::as_strided"
            || type == "aten::diagonal" || type == "aten::narrow" || type == "aten::unfold"
            || type == "aten::split" || type == "aten::split_with_sizes" || type == "aten::chunk" || type == "aten::unbind"
            || type == "aten::view_as_real" || type == "aten::view_as_complex" || type == "aten::real" || type == "aten::imag"
            || type == "aten::contiguous" || type == "aten::to" || type == "aten::type_as" || is_dropout(type)
            || type == "nn.Dropout2d" || type == "nn.Dropout3d")
        return TorchAliasUntracked;

    return TorchAliasNone;
}

static const Parameter* constant_value(const Operand* r)
{
    const Operator* op = r->producer;
    if (op->type != "prim::Constant" || !op->has_param("value"))
        return 0;
    return &op->params.at("value");
}

TorchAliasKind torch_alias_kind(const std::string& type, const std::vector<Operand*>& inputs, int output_type)
{
    const TorchAliasKind kind = torch_alias_kind(type);
    if (kind != TorchAliasUntracked || inputs.empty())
        return kind;

    if (is_dropout(type) && inputs.size() == 3)
    {
        const Parameter* training = constant_value(inputs[2]);
        const Parameter* probability = constant_value(inputs[1]);
        if ((training && training->type == 1 && !training->b)
                || (probability && ((probability->type == 3 && probability->f == 0.f) || (probability->type == 2 && probability->i == 0))))
            return TorchAliasIdentity;
        if (training && training->type == 1 && training->b)
            return TorchAliasNone;
    }

    if (type == "aten::type_as" && inputs.size() == 2)
    {
        // Both frontends normalize tensors to the selected inference device.
        if (inputs[0]->type > 0 && inputs[1]->type > 0)
            return inputs[0]->type == inputs[1]->type ? TorchAliasIdentity : TorchAliasNone;
    }

    if (type == "aten::to")
    {
        if (inputs[0]->type > 0 && output_type > 0 && inputs[0]->type != output_type)
            return TorchAliasNone;

        // dtype/other, device+dtype, and dtype+layout overloads respectively.
        if (inputs.size() != 5 && inputs.size() != 6 && inputs.size() != 8)
            return TorchAliasUntracked;
        const Parameter* copy = constant_value(inputs[inputs.size() - 2]);
        const Parameter* format = constant_value(inputs.back());
        if (!copy || copy->type != 1 || !format || (format->type != 0 && format->type != 2))
            return TorchAliasUntracked;
        if (copy->b)
            return TorchAliasNone;
        if (inputs[0]->type > 0 && inputs[0]->type == output_type
                && (format->type == 0 || format->i == 1))
            return TorchAliasIdentity;
    }

    return kind;
}

TorchAliasKind torch_alias_kind(const Operator* op)
{
    return torch_alias_kind(op->type, op->inputs, op->outputs.empty() ? 0 : op->outputs[0]->type);
}

TorchMutationKind torch_mutation_kind(const std::string& type)
{
    if (type == "aten::transpose_" || type == "aten::t_" || type == "aten::squeeze_" || type == "aten::unsqueeze_")
        return TorchMutationMetadata;
    if (type == "aten::resize_" || type == "aten::resize_as_" || type == "aten::set_" || type == "aten::as_strided_")
        return TorchMutationStorage;
    if (type.size() > 2 && type[type.size() - 2] != '_' && type.back() == '_')
        return TorchMutationData;
    return TorchMutationNone;
}

} // namespace pnnx
