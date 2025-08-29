// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Softmax : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Softmax";
    }

    const char* type_str() const
    {
        return "nn.Softmax";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* softmax = graph.find_node_by_kind("aten::softmax");

        op->params["dim"] = softmax->namedInput("dim");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Softmax)

} // namespace pnnx
