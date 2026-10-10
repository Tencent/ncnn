// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class CELU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.CELU";
    }

    const char* type_str() const
    {
        return "nn.CELU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* celu = graph.find_node_by_kind("aten::celu");

        op->params["alpha"] = celu->namedInput("alpha");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(CELU)

} // namespace pnnx
