// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class RMSNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.RMSNorm";
    }

    const char* type_str() const
    {
        return "nn.RMSNorm";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* rmsn = graph.find_node_by_kind("aten::rms_norm");

        op->params["normalized_shape"] = rmsn->namedInput("normalized_shape");
        op->params["eps"] = rmsn->namedInput("eps");
        op->params["elementwise_affine"] = mod.hasattr("weight");

        if (mod.hasattr("weight"))
        {
            op->attrs["weight"] = mod.attr("weight");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RMSNorm)

} // namespace pnnx
