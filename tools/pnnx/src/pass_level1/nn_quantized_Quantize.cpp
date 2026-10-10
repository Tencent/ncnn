// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Quantize : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.quantized.modules.Quantize";
    }

    const char* type_str() const
    {
        return "nn.quantized.Quantize";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        //         mod.dump(true, false, false);

        //         graph->dump();

        const TorchNodeProxy* quantize_per_tensor = graph.find_node_by_kind("aten::quantize_per_tensor");

        //         for (auto aa : quantize_per_tensor->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        // scale, zero_point
        op->params["scale"] = quantize_per_tensor->namedInput("scale");
        op->params["zero_point"] = quantize_per_tensor->namedInput("zero_point");
        op->params["dtype"] = "torch.qint8";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Quantize)

} // namespace pnnx
