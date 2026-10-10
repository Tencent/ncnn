// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class GroupNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.GroupNorm";
    }

    const char* type_str() const
    {
        return "nn.GroupNorm";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        //         graph->dump();

        const TorchNodeProxy* gn = graph.find_node_by_kind("aten::group_norm");

        //         for (auto aa : gn->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        op->params["num_groups"] = gn->namedInput("num_groups");
        op->params["eps"] = gn->namedInput("eps");
        op->params["affine"] = mod.hasattr("weight") && mod.hasattr("bias");

        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            const auto& weight = mod.attr("weight");

            op->params["num_channels"] = weight.size(0);

            op->attrs["weight"] = weight;
            op->attrs["bias"] = mod.attr("bias");
        }
        else
        {
            fprintf(stderr, "Cannot resolve GroupNorm num_channels when affine=False\n");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(GroupNorm)

} // namespace pnnx
