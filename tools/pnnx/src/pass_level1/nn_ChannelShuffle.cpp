// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ChannelShuffle : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.channelshuffle.ChannelShuffle";
    }

    const char* type_str() const
    {
        return "nn.ChannelShuffle";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* channel_shuffle = graph.find_node_by_kind("aten::channel_shuffle");

        op->params["groups"] = channel_shuffle->namedInput("groups");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ChannelShuffle)

} // namespace pnnx
