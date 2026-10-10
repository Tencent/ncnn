// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ZeroPad2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.padding.ZeroPad2d";
    }

    const char* type_str() const
    {
        return "nn.ZeroPad2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pad = graph.find_node_by_kind("aten::pad");
        const TorchNodeProxy* constant_pad_nd = graph.find_node_by_kind("aten::constant_pad_nd");

        if (!pad)
        {
            pad = constant_pad_nd;
        }

        op->params["padding"] = pad->namedInput("pad");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ZeroPad2d)

} // namespace pnnx
