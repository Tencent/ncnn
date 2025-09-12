// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Hardshrink : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Hardshrink";
    }

    const char* type_str() const
    {
        return "nn.Hardshrink";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* hardshrink = graph.find_node_by_kind("aten::hardshrink");

        op->params["lambd"] = hardshrink->namedInput("lambd");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Hardshrink)

} // namespace pnnx
