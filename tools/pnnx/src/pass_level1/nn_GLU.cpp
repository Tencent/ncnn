// Copyright 2021 Tencent
// Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class GLU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.GLU";
    }

    const char* type_str() const
    {
        return "nn.GLU";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* glu = graph.find_node_by_kind("aten::glu");

        op->params["dim"] = glu->namedInput("dim");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(GLU)

} // namespace pnnx
