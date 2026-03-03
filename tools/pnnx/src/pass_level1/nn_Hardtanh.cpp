// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Hardtanh : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Hardtanh";
    }

    const char* type_str() const
    {
        return "nn.Hardtanh";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* hardtanh = graph.find_node_by_kind("aten::hardtanh");

        op->params["min_val"] = hardtanh->namedInput("min_val");
        op->params["max_val"] = hardtanh->namedInput("max_val");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Hardtanh)

} // namespace pnnx
