// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LayerNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.LayerNorm";
    }

    const char* type_str() const
    {
        return "nn.LayerNorm";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* ln = graph.find_node_by_kind("aten::layer_norm");

        op->params["normalized_shape"] = ln->namedInput("normalized_shape");
        op->params["eps"] = ln->namedInput("eps");
        op->params["elementwise_affine"] = mod.hasattr("weight") && mod.hasattr("bias");

        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            op->attrs["weight"] = mod.attr("weight");
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LayerNorm)

} // namespace pnnx
