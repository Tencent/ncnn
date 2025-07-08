// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class RoIAlign : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torchvision.ops.roi_align.RoIAlign";
    }

    const char* type_str() const
    {
        return "torchvision.ops.RoIAlign";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* roi_align = graph.find_node_by_kind("torchvision::roi_align");

        if (roi_align->input(0) == graph.input(2) && roi_align->input(1) == graph.input(1))
        {
            fprintf(stderr, "roi_align inputs swapped detected !\n");
            std::swap(op->inputs[0], op->inputs[1]);
        }

        const Parameter pooled_height = roi_align->namedInput("pooled_height");
        const Parameter pooled_width = roi_align->namedInput("pooled_width");

        op->params["spatial_scale"] = roi_align->namedInput("spatial_scale");
        op->params["sampling_ratio"] = roi_align->namedInput("sampling_ratio");
        op->params["aligned"] = roi_align->namedInput("aligned");
        op->params["output_size"] = {pooled_height.i, pooled_width.i};
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RoIAlign)

} // namespace pnnx
