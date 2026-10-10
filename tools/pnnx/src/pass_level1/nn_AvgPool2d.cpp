// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class AvgPool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.AvgPool2d";
    }

    const char* type_str() const
    {
        return "nn.AvgPool2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* avg_pool2d = graph.find_node_by_kind("aten::avg_pool2d");

        op->params["kernel_size"] = avg_pool2d->namedInput("kernel_size");
        op->params["stride"] = avg_pool2d->namedInput("stride");
        op->params["padding"] = avg_pool2d->namedInput("padding");
        op->params["ceil_mode"] = avg_pool2d->namedInput("ceil_mode");
        op->params["count_include_pad"] = avg_pool2d->namedInput("count_include_pad");
        op->params["divisor_override"] = avg_pool2d->namedInput("divisor_override");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(AvgPool2d)

} // namespace pnnx
