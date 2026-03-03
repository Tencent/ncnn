// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class AvgPool1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.AvgPool1d";
    }

    const char* type_str() const
    {
        return "nn.AvgPool1d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* avg_pool1d = graph.find_node_by_kind("aten::avg_pool1d");

        op->params["kernel_size"] = avg_pool1d->namedInput("kernel_size");
        op->params["stride"] = avg_pool1d->namedInput("stride");
        op->params["padding"] = avg_pool1d->namedInput("padding");
        op->params["ceil_mode"] = avg_pool1d->namedInput("ceil_mode");
        op->params["count_include_pad"] = avg_pool1d->namedInput("count_include_pad");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(AvgPool1d)

} // namespace pnnx
