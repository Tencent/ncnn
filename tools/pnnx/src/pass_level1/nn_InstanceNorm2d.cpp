// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class InstanceNorm2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.instancenorm.InstanceNorm2d";
    }

    const char* type_str() const
    {
        return "nn.InstanceNorm2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        //         graph->dump();

        const TorchNodeProxy* in = graph.find_node_by_kind("aten::instance_norm");

        //         for (auto aa : in->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        op->params["eps"] = in->namedInput("eps");
        op->params["affine"] = mod.hasattr("weight") && mod.hasattr("bias");
        op->params["track_running_stats"] = mod.hasattr("running_mean") && mod.hasattr("running_var");

        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            const TorchTensorProxy& weight = mod.attr("weight");

            op->params["num_features"] = weight.size(0);

            op->attrs["weight"] = weight;
            op->attrs["bias"] = mod.attr("bias");
        }

        if (mod.hasattr("running_mean") && mod.hasattr("running_var"))
        {
            const TorchTensorProxy& running_mean = mod.attr("running_mean");

            op->params["num_features"] = running_mean.size(0);

            op->attrs["running_mean"] = running_mean;
            op->attrs["running_var"] = mod.attr("running_var");
        }

        // take num_features from input shape
        if (!op->has_param("num_features") && !op->inputs[0]->shape.empty())
        {
            op->params["num_features"] = op->inputs[0]->shape[op->inputs[0]->shape.size() - 3];
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(InstanceNorm2d)

} // namespace pnnx
