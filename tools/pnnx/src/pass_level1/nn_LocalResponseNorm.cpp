// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LocalResponseNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.LocalResponseNorm";
    }

    const char* type_str() const
    {
        return "nn.LocalResponseNorm";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* avg_pool = graph.find_node_by_kind("aten::avg_pool2d");
        const TorchNodeProxy* avg_pool3d = graph.find_node_by_kind("aten::avg_pool3d");

        if (avg_pool3d)
        {
            avg_pool = avg_pool3d;
        }

        const TorchNodeProxy* kernel_size = graph.find_producer_node_by_value(avg_pool->namedInput("kernel_size"));
        op->params["size"] = kernel_size->input(0);

        const TorchNodeProxy* pow = graph.find_node_by_kind("aten::pow");
        op->params["beta"] = pow->input(1);

        const TorchNodeProxy* add = graph.find_producer_node_by_value(pow->input(0));
        op->params["k"] = add->input(1);

        const TorchNodeProxy* mul = graph.find_producer_node_by_value(add->input(0));
        op->params["alpha"] = mul->input(1);
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LocalResponseNorm)

} // namespace pnnx
