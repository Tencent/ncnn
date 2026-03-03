// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class RReLU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.RReLU";
    }

    const char* type_str() const
    {
        return "nn.RReLU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* rrelu = graph.find_node_by_kind("aten::rrelu");

        op->params["lower"] = rrelu->namedInput("lower");
        op->params["upper"] = rrelu->namedInput("upper");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RReLU)

} // namespace pnnx
