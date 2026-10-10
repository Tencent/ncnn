// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LeakyReLU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.LeakyReLU";
    }

    const char* type_str() const
    {
        return "nn.LeakyReLU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* leaky_relu = graph.find_node_by_kind("aten::leaky_relu");
        const TorchNodeProxy* leaky_relu_ = graph.find_node_by_kind("aten::leaky_relu_");

        if (leaky_relu_)
        {
            leaky_relu = leaky_relu_;
        }

        op->params["negative_slope"] = leaky_relu->namedInput("negative_slope");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LeakyReLU)

} // namespace pnnx
