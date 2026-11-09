// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_TORCH_ALIAS_H
#define PNNX_TORCH_ALIAS_H

#include "ir.h"

namespace pnnx {

enum TorchAliasKind
{
    TorchAliasNone,
    TorchAliasIdentity,
    TorchAliasView,
    TorchAliasUntracked
};

enum TorchMutationKind
{
    TorchMutationNone,
    TorchMutationData,
    TorchMutationMetadata,
    TorchMutationStorage
};

TorchAliasKind torch_alias_kind(const std::string& type);
TorchAliasKind torch_alias_kind(const std::string& type, const std::vector<Operand*>& inputs, int output_type);
TorchAliasKind torch_alias_kind(const Operator* op);
TorchMutationKind torch_mutation_kind(const std::string& type);

#if BUILD_TORCH2PNNX
TorchAliasKind torch_alias_kind(const torch::jit::Node* node);
torch::jit::Value* torch_alias_input(const torch::jit::Node* node);
#endif

} // namespace pnnx

#endif // PNNX_TORCH_ALIAS_H
