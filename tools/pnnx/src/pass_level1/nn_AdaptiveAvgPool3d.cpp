// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class AdaptiveAvgPool3d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool3d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool3d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* adaptive_avg_pool3d = graph.find_node_by_kind("aten::adaptive_avg_pool3d");

        op->params["output_size"] = adaptive_avg_pool3d->namedInput("output_size");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool3d)

} // namespace pnnx
