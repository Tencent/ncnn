// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_yolo.h"

#include "parameter.h"

#include <stdio.h>

#include <cmath>

namespace pnnx {

namespace caffe2pnnx {

int prepare_yolo(Operator* op)
{
    if (op->inputs.empty() || op->outputs.size() != 1 || !op->attrs.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected at least one input, one output and no blobs; got inputs=%zu outputs=%zu blobs=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size());
        return -1;
    }
    for (Operator* consumer : op->outputs[0]->consumers)
        if (consumer->type != "pnnx.Output")
        {
            fprintf(stderr, "caffe layer %s (%s): detection must be terminal; downstream layer %s (%s) is unsupported\n", op->name.c_str(), op->type.c_str(), consumer->name.c_str(), consumer->type.c_str());
            return -1;
        }
    const bool v3 = op->type == "caffe.Yolov3DetectionOutput";
    const int classes = get_int(op, "num_classes", 20);
    const int boxes = get_int(op, "num_box", v3 ? 3 : 5);
    const float confidence = get_float(op, "confidence_threshold", 0.01f);
    const float nms = get_float(op, "nms_threshold", 0.45f);
    std::vector<float> biases = get_parameter(op, "biases", std::vector<float>()).af;
    if (classes <= 0 || classes > INT_MAX - 5 || boxes <= 0 || boxes > INT_MAX / (classes + 5) || !std::isfinite(confidence) || !std::isfinite(nms) || biases.empty() || (v3 && biases.size() % 2))
    {
        fprintf(stderr, "caffe layer %s (%s): invalid classes=%d boxes=%d confidence=%g nms=%g biases count=%zu; require positive bounded channel counts, finite thresholds and nonempty anchor pairs for v3\n", op->name.c_str(), op->type.c_str(), classes, boxes, confidence, nms, biases.size());
        return -1;
    }
    if (!v3)
    {
        if (biases.size() < (size_t)boxes * 2)
        {
            fprintf(stderr, "caffe layer %s (%s): biases count=%zu is smaller than required anchor values=%lld\n", op->name.c_str(), op->type.c_str(), biases.size(), (long long)boxes * 2);
            return -1;
        }
        biases.resize(boxes * 2);
    }
    for (float bias : biases)
        if (!std::isfinite(bias) || bias <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): anchor bias=%g must be finite and positive\n", op->name.c_str(), op->type.c_str(), bias);
            return -1;
        }
    for (Operand* r : op->inputs)
    {
        if (r->shape.size() != 4 || r->shape[0] != 1 || r->shape[1] != boxes * (classes + 5) || r->shape[2] <= 0 || r->shape[3] <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): input %s requires shape [1,%d,H>0,W>0]; got rank=%zu N=%d C=%d H=%d W=%d\n", op->name.c_str(), op->type.c_str(), r->name.c_str(), boxes * (classes + 5), r->shape.size(), r->shape.size() == 4 ? r->shape[0] : 0, r->shape.size() == 4 ? r->shape[1] : 0, r->shape.size() == 4 ? r->shape[2] : 0, r->shape.size() == 4 ? r->shape[3] : 0);
            return -1;
        }
        r->params["__batch_index"] = 0;
    }
    std::vector<int> mask;
    std::vector<int> scales;
    if (v3)
    {
        mask = get_parameter(op, "mask", std::vector<int>()).ai;
        scales = get_parameter(op, "anchors_scale", std::vector<int>()).ai;
        if (mask.size() != op->inputs.size() * boxes || scales.size() != op->inputs.size())
        {
            fprintf(stderr, "caffe layer %s (%s): mask count=%zu anchors_scale count=%zu; expected %zu and %zu for %d boxes per head\n", op->name.c_str(), op->type.c_str(), mask.size(), scales.size(), op->inputs.size() * boxes, op->inputs.size(), boxes);
            return -1;
        }
        for (int index : mask)
            if (index < 0 || (size_t)index >= biases.size() / 2)
            {
                fprintf(stderr, "caffe layer %s (%s): anchor mask index=%d is outside [0,%zu)\n", op->name.c_str(), op->type.c_str(), index, biases.size() / 2);
                return -1;
            }
        for (size_t i = 0; i < scales.size(); i++)
        {
            if (scales[i] <= 0 || scales[i] > INT_MAX / op->inputs[i]->shape[2] || scales[i] > INT_MAX / op->inputs[i]->shape[3])
            {
                fprintf(stderr, "caffe layer %s (%s): head %zu anchors_scale=%d must be positive and scaled spatial shape (%d,%d) fit INT_MAX\n", op->name.c_str(), op->type.c_str(), i, scales[i], op->inputs[i]->shape[2], op->inputs[i]->shape[3]);
                return -1;
            }
            if ((float)scales[i] * op->inputs[i]->shape[2] >= 2147483648.f || (float)scales[i] * op->inputs[i]->shape[3] >= 2147483648.f)
            {
                fprintf(stderr, "caffe layer %s (%s): head %zu float scaled spatial extent overflows int32; anchors_scale=%d height=%d width=%d\n", op->name.c_str(), op->type.c_str(), i, scales[i], op->inputs[i]->shape[2], op->inputs[i]->shape[3]);
                return -1;
            }
        }
    }
    else if (get_int(op, "coords", 4) != 4)
    {
        fprintf(stderr, "caffe layer %s (%s): coords=%d is unsupported, expected 4\n", op->name.c_str(), op->type.c_str(), get_int(op, "coords", 4));
        return -1;
    }
    if (confidence >= 0.f)
        fprintf(stderr, "warning: caffe layer %s (%s): existing ncnn does not reliably represent an empty detection result; inputs producing no candidates are unsupported.\n", op->name.c_str(), op->type.c_str());
    op->params.clear();
    op->params["num_classes"] = classes;
    op->params["num_box"] = boxes;
    op->params["confidence_threshold"] = confidence;
    op->params["nms_threshold"] = nms;
    op->params["biases"] = biases;
    if (v3)
    {
        op->params["mask"] = mask;
        op->params["anchors_scale"] = scales;
    }
    op->outputs[0]->shape = {-1, 6};
    op->outputs[0]->params["__batch_index"] = 233;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
