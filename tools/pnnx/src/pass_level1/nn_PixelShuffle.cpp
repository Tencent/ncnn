// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class PixelShuffle : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pixelshuffle.PixelShuffle";
    }

    const char* type_str() const
    {
        return "nn.PixelShuffle";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pixel_shuffle = graph.find_node_by_kind("aten::pixel_shuffle");

        op->params["upscale_factor"] = pixel_shuffle->namedInput("upscale_factor");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(PixelShuffle)

} // namespace pnnx
