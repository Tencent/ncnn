// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Threshold : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Threshold";
    }

    const char* type_str() const
    {
        return "nn.Threshold";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* threshold = graph.find_node_by_kind("aten::threshold");

        op->params["threshold"] = threshold->namedInput("threshold");
        op->params["value"] = threshold->namedInput("value");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Threshold)

} // namespace pnnx
