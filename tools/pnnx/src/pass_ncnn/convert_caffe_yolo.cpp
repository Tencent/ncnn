// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convert_caffe_yolo.h"

namespace pnnx {

namespace ncnn {

void convert_caffe_yolo(Graph& graph)
{
    for (Operator* op : graph.ops)
    {
        if (op->type != "caffe.YoloDetectionOutput" && op->type != "caffe.Yolov3DetectionOutput")
            continue;
        const bool v3 = op->type == "caffe.Yolov3DetectionOutput";
        const std::map<std::string, Parameter> params = op->params;
        op->type = v3 ? "Yolov3DetectionOutput" : "YoloDetectionOutput";
        op->params.clear();
        op->params["0"] = params.at("num_classes");
        op->params["1"] = params.at("num_box");
        op->params["2"] = params.at("confidence_threshold");
        op->params["3"] = params.at("nms_threshold");
        op->params["4"] = params.at("biases");
        if (v3)
        {
            const std::vector<int>& mask = params.at("mask").ai;
            const std::vector<int>& scales = params.at("anchors_scale").ai;
            op->params["5"] = std::vector<float>(mask.begin(), mask.end());
            op->params["6"] = std::vector<float>(scales.begin(), scales.end());
        }
    }
}

} // namespace ncnn

} // namespace pnnx
