// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class GELU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.GELU";
    }

    const char* type_str() const
    {
        return "nn.GELU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* gelu = graph.find_node_by_kind("aten::gelu");

        if (gelu->hasNamedInput("approximate"))
        {
            op->params["approximate"] = gelu->namedInput("approximate");
            if (op->params["approximate"].s == "none")
                op->params.clear();
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(GELU)

} // namespace pnnx
