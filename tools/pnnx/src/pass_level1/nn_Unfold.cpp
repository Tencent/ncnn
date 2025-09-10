// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Unfold : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.fold.Unfold";
    }

    const char* type_str() const
    {
        return "nn.Unfold";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* im2col = graph.find_node_by_kind("aten::im2col");

        op->params["kernel_size"] = im2col->namedInput("kernel_size");
        op->params["stride"] = im2col->namedInput("stride");
        op->params["padding"] = im2col->namedInput("padding");
        op->params["dilation"] = im2col->namedInput("dilation");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Unfold)

} // namespace pnnx
