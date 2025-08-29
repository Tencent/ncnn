// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class DeQuantize : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.quantized.modules.DeQuantize";
    }

    const char* type_str() const
    {
        return "nn.quantized.DeQuantize";
    }

    // void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    // {
    //     //         mod.dump(true, false, false);
    //
    //     //         graph->dump();
    //
    //     const torch::jit::Node* dequantize = find_node_by_kind(graph, "aten::dequantize");
    //
    //     //         for (auto aa : dequantize->schema().arguments())
    //     //         {
    //     //             fprintf(stderr, "arg %s\n", aa.name().c_str());
    //     //         }
    // }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(DeQuantize)

} // namespace pnnx
