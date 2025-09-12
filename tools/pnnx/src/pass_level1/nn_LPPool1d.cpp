// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LPPool1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.LPPool1d";
    }

    const char* type_str() const
    {
        return "nn.LPPool1d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pow = graph.find_node_by_kind("aten::pow");
        op->params["norm_type"] = pow->input(1);

        const TorchNodeProxy* avg_pool1d = graph.find_node_by_kind("aten::avg_pool1d");

        const TorchNodeProxy* kernel_size = graph.find_producer_node_by_value(avg_pool1d->namedInput("kernel_size"));
        const TorchNodeProxy* stride = graph.find_producer_node_by_value(avg_pool1d->namedInput("stride"));

        op->params["kernel_size"] = kernel_size->input(0);
        if (stride->input_count() == 0)
        {
            op->params["stride"] = op->params["kernel_size"];
        }
        else
        {
            op->params["stride"] = stride->input(0);
        }
        op->params["ceil_mode"] = avg_pool1d->namedInput("ceil_mode");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LPPool1d)

} // namespace pnnx
