// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Fold : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.fold.Fold";
    }

    const char* type_str() const
    {
        return "nn.Fold";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* col2im = graph.find_node_by_kind("aten::col2im");

        op->params["output_size"] = col2im->namedInput("output_size");
        op->params["kernel_size"] = col2im->namedInput("kernel_size");
        op->params["stride"] = col2im->namedInput("stride");
        op->params["padding"] = col2im->namedInput("padding");
        op->params["dilation"] = col2im->namedInput("dilation");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Fold)

} // namespace pnnx
