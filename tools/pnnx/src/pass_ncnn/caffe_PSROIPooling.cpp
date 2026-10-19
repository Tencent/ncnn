// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class caffe_PSROIPooling : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 feature
pnnx.Input              input_1     0 1 roi
caffe.PSROIPooling          op_0        2 1 feature roi out pooled_h=%pooled_h pooled_w=%pooled_w spatial_scale=%spatial_scale output_dim=%output_dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 feature
pnnx.Input              input_1     0 1 roi
Crop                    crop        1 1 roi coordinates 9=(1) 10=(5) 11=(0)
PSROIPooling              pool        2 1 feature coordinates out 0=%pooled_w 1=%pooled_h 2=%spatial_scale 3=%output_dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(caffe_PSROIPooling, 20)


} // namespace ncnn

} // namespace pnnx
