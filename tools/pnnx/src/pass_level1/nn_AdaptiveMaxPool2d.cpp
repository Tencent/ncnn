// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class AdaptiveMaxPool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.AdaptiveMaxPool2d";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveMaxPool2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* adaptive_max_pool2d = graph.find_node_by_kind("aten::adaptive_max_pool2d");

        const TorchNodeProxy* graph_out = graph.find_producer_node_by_value(graph.output(0));

        op->params["output_size"] = adaptive_max_pool2d->namedInput("output_size");
        op->params["return_indices"] = graph_out->kind() == "prim::TupleConstruct" ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(AdaptiveMaxPool2d)

} // namespace pnnx
