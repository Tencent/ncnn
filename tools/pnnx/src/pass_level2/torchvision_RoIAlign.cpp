// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torchvision_RoIAlign_export : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
pnnx.Input              rois        0 1 rois
prim::Constant          spatial_scale 0 1 spatial_scale value=%spatial_scale
prim::Constant          pooled_height 0 1 pooled_height value=%pooled_height
prim::Constant          pooled_width 0 1 pooled_width value=%pooled_width
prim::Constant          sampling_ratio 0 1 sampling_ratio value=%sampling_ratio
prim::Constant          aligned     0 1 aligned value=%aligned
torchvision::roi_align  op_0        7 1 input rois spatial_scale pooled_height pooled_width sampling_ratio aligned out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchvision.ops.RoIAlign";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["output_size"] = Parameter{captured_params.at("pooled_height").i, captured_params.at("pooled_width").i};
        op->params["spatial_scale"] = captured_params.at("spatial_scale");
        op->params["sampling_ratio"] = captured_params.at("sampling_ratio");
        op->params["aligned"] = captured_params.at("aligned");
        op->inputnames = {"input", "rois"};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchvision_RoIAlign_export, 110)

} // namespace pnnx
