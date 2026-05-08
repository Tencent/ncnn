// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class UpsamplingNearest2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.upsampling.UpsamplingNearest2d";
    }

    const char* type_str() const
    {
        return "nn.UpsamplingNearest2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* upsample = graph.find_node_by_kind("aten::upsample_nearest2d");

        if (upsample->hasNamedInput("output_size"))
        {
            op->params["size"] = upsample->namedInput("output_size");
        }

        if (upsample->hasNamedInput("scale_factors"))
        {
            op->params["scale_factor"] = upsample->namedInput("scale_factors");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(UpsamplingNearest2d)

} // namespace pnnx
