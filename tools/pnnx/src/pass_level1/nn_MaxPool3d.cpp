// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class MaxPool3d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.MaxPool3d";
    }

    const char* type_str() const
    {
        return "nn.MaxPool3d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* max_pool3d = graph.find_node_by_kind("aten::max_pool3d");
        const TorchNodeProxy* max_pool3d_with_indices = graph.find_node_by_kind("aten::max_pool3d_with_indices");

        if (max_pool3d_with_indices)
        {
            max_pool3d = max_pool3d_with_indices;
        }

        op->params["kernel_size"] = max_pool3d->namedInput("kernel_size");
        op->params["stride"] = max_pool3d->namedInput("stride");
        op->params["padding"] = max_pool3d->namedInput("padding");
        op->params["dilation"] = max_pool3d->namedInput("dilation");
        op->params["ceil_mode"] = max_pool3d->namedInput("ceil_mode");
        op->params["return_indices"] = max_pool3d_with_indices ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MaxPool3d)

} // namespace pnnx
