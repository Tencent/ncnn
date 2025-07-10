// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Linear : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.linear.Linear";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    void write(Operator* op, const TorchGraphProxy& /*graph*/, const TorchModuleProxy& mod) const
    {
        // const TorchNodeProxy* addmm = graph.find_node_by_kind("aten::addmm");

        const TorchTensorProxy& weight = mod.attr("weight");

        op->params["in_features"] = weight.size(1);
        op->params["out_features"] = weight.size(0);
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Linear)

} // namespace pnnx
