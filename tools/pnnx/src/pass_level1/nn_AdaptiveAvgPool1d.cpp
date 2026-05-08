// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class AdaptiveAvgPool1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.AdaptiveAvgPool1d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool1d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* adaptive_avg_pool1d = graph.find_node_by_kind("aten::adaptive_avg_pool1d");

        op->params["output_size"] = adaptive_avg_pool1d->namedInput("output_size");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(AdaptiveAvgPool1d)

} // namespace pnnx
