// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LPPool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.LPPool2d";
    }

    const char* type_str() const
    {
        return "nn.LPPool2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pow = graph.find_node_by_kind("aten::pow");
        op->params["norm_type"] = pow->input(1);

        const TorchNodeProxy* avg_pool2d = graph.find_node_by_kind("aten::avg_pool2d");

        const TorchNodeProxy* stride = graph.find_producer_node_by_value(avg_pool2d->namedInput("stride"));

        op->params["kernel_size"] = avg_pool2d->namedInput("kernel_size");
        if (stride->input_count() == 0)
        {
            op->params["stride"] = op->params["kernel_size"];
        }
        else
        {
            op->params["stride"] = avg_pool2d->namedInput("stride");
        }
        op->params["ceil_mode"] = avg_pool2d->namedInput("ceil_mode");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LPPool2d)

} // namespace pnnx
