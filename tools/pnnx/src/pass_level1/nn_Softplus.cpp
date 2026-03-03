// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Softplus : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Softplus";
    }

    const char* type_str() const
    {
        return "nn.Softplus";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* softplus = graph.find_node_by_kind("aten::softplus");

        op->params["beta"] = softplus->namedInput("beta");
        op->params["threshold"] = softplus->namedInput("threshold");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Softplus)

} // namespace pnnx
