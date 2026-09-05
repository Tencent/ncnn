// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torchvision_RoIAlign : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              rois        0 1 rois
torchvision.ops.RoIAlign op_0       2 1 input rois out output_size=%output_size spatial_scale=%spatial_scale sampling_ratio=%sampling_ratio aligned=%aligned
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ROIAlign";
    }

    const char* name_str() const
    {
        return "roialign";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators,
               const std::map<std::string, Parameter>& /*captured_params*/,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // ncnn ROIAlign reads exactly four coordinates and produces one output;
        // it cannot express torchvision's (K, 5) rois layout nor a per-row
        // batch index. only lower a single constant ROI with batch index 0.
        const Operator* op0 = matched_operators.at("op_0");
        if (op0->inputs.size() < 2)
            return false;

        Operator* rois_op = op0->inputs[1]->producer;
        if (!rois_op || rois_op->type != "pnnx.Attribute" || rois_op->attrs.empty())
            return false; // dynamic rois: keep the torchvision op untouched

        const Attribute& a = rois_op->attrs.begin()->second;
        if (a.elemcount() != 5)
            return false; // only a single ROI is supported

        const std::vector<float> f = a.get_float32_data();
        if (f.size() < 5 || f[0] != 0.f)
            return false; // batch index must be zero

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const std::vector<int>& output_size = captured_params.at("output_size").ai;
        op->params["0"] = output_size[1]; // pooled_width
        op->params["1"] = output_size[0]; // pooled_height
        op->params["2"] = captured_params.at("spatial_scale").f;
        op->params["3"] = captured_params.at("sampling_ratio").i;
        op->params["4"] = captured_params.at("aligned").b ? 1 : 0;
        op->params["5"] = 0; // version

        // torchvision rois format is (num_rois, 5) = [batch_idx, x1, y1, x2, y2];
        // ncnn ROIAlign expects rois as [x1, y1, x2, y2]. match() guarantees a
        // single constant ROI with batch index 0, so drop the leading entry.
        Operator* rois_op = op->inputs[1]->producer;
        if (rois_op && rois_op->type == "pnnx.Attribute" && !rois_op->attrs.empty())
        {
            Attribute& a = rois_op->attrs.begin()->second;
            std::vector<float> f = a.get_float32_data();
            if (f.size() == 5)
            {
                std::vector<float> f4(f.begin() + 1, f.end());
                a.set_float32_data(f4);
                a.shape = std::vector<int>{4};
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchvision_RoIAlign, 20)

} // namespace ncnn

} // namespace pnnx
