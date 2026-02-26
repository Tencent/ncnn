// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ELU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.ELU";
    }

    const char* type_str() const
    {
        return "nn.ELU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* elu = graph.find_node_by_kind("aten::elu");

        op->params["alpha"] = elu->namedInput("alpha");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ELU)

} // namespace pnnx
