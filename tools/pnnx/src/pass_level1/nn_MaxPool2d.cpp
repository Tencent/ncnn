// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class MaxPool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.MaxPool2d";
    }

    const char* type_str() const
    {
        return "nn.MaxPool2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* max_pool2d = graph.find_node_by_kind("aten::max_pool2d");
        const TorchNodeProxy* max_pool2d_with_indices = graph.find_node_by_kind("aten::max_pool2d_with_indices");

        if (max_pool2d_with_indices)
        {
            max_pool2d = max_pool2d_with_indices;
        }

        op->params["kernel_size"] = max_pool2d->namedInput("kernel_size");
        op->params["stride"] = max_pool2d->namedInput("stride");
        op->params["padding"] = max_pool2d->namedInput("padding");
        op->params["dilation"] = max_pool2d->namedInput("dilation");
        op->params["ceil_mode"] = max_pool2d->namedInput("ceil_mode");
        op->params["return_indices"] = max_pool2d_with_indices ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MaxPool2d)

} // namespace pnnx
