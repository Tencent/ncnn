// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class LogSoftmax : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.LogSoftmax";
    }

    const char* type_str() const
    {
        return "nn.LogSoftmax";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* log_softmax = graph.find_node_by_kind("aten::log_softmax");

        op->params["dim"] = log_softmax->namedInput("dim");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LogSoftmax)

} // namespace pnnx
