// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ReplicationPad2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.padding.ReplicationPad2d";
    }

    const char* type_str() const
    {
        return "nn.ReplicationPad2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pad = graph.find_node_by_kind("aten::pad");
        const TorchNodeProxy* replication_pad2d = graph.find_node_by_kind("aten::replication_pad2d");

        if (pad)
        {
            op->params["padding"] = pad->namedInput("pad");
        }
        else
        {
            op->params["padding"] = replication_pad2d->namedInput("padding");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ReplicationPad2d)

} // namespace pnnx
