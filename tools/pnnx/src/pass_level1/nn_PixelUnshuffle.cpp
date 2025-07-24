// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class PixelUnshuffle : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pixelshuffle.PixelUnshuffle";
    }

    const char* type_str() const
    {
        return "nn.PixelUnshuffle";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* pixel_unshuffle = graph.find_node_by_kind("aten::pixel_unshuffle");

        op->params["downscale_factor"] = pixel_unshuffle->namedInput("downscale_factor");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(PixelUnshuffle)

} // namespace pnnx
