// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Softshrink : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Softshrink";
    }

    const char* type_str() const
    {
        return "nn.Softshrink";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* softshrink = graph.find_node_by_kind("aten::softshrink");

        op->params["lambd"] = softshrink->namedInput("lambd");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Softshrink)

} // namespace pnnx
