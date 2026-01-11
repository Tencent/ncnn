// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class BatchNorm1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.batchnorm.BatchNorm1d";
    }

    const char* type_str() const
    {
        return "nn.BatchNorm1d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* bn = graph.find_node_by_kind("aten::batch_norm");

        const TorchTensorProxy& running_mean = mod.attr("running_mean");
        const TorchTensorProxy& running_var = mod.attr("running_var");

        op->params["num_features"] = running_mean.size(0);
        op->params["eps"] = bn->namedInput("eps");
        op->params["affine"] = mod.hasattr("weight") && mod.hasattr("bias");

        op->attrs["running_mean"] = running_mean;
        op->attrs["running_var"] = running_var;
        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            op->attrs["weight"] = mod.attr("weight");
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(BatchNorm1d)

} // namespace pnnx
