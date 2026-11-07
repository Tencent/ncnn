// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_roi.h"

#include "parameter.h"

#include <cmath>

namespace pnnx {

namespace caffe2pnnx {

int lower_roi(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 2 || op->outputs.size() != 1 || !op->attrs.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected feature and ROI inputs, one output and no blobs; got %zu inputs, %zu outputs and %zu blobs\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size());
        return -1;
    }
    Operand* feature = op->inputs[0];
    Operand* rois = op->inputs[1];
    const bool dynamic = rois->producer && rois->producer->type == "caffe.Proposal" && rois == rois->producer->outputs[0];
    if (dynamic)
    {
        fprintf(stderr, "caffe layer %s (%s): dynamic Proposal-to-ROI requires runtime multi-ROI support; the existing ncnn ROI layer consumes only one box\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    fprintf(stderr, "warning: caffe layer %s (%s): existing ncnn ROI runtime is preserved; out-of-range ROIAlign samples, clipped ROIPooling bins and negative-half-integer PSROI endpoints are not guaranteed equivalent to Caffe\n", op->name.c_str(), op->type.c_str());
    if (feature->shape.size() != 4 || feature->shape[0] != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): feature input requires rank 4 with batch 1; rank=%zu batch=%d\n", op->name.c_str(), op->type.c_str(), feature->shape.size(), feature->shape.empty() ? -1 : feature->shape[0]);
        return -1;
    }
    if (rois->shape.size() < 2 || rois->shape[0] <= 0 || rois->shape[1] != 5)
    {
        fprintf(stderr, "caffe layer %s (%s): ROI input requires [R,5] with positive static R; rank=%zu R=%d coordinates=%d\n", op->name.c_str(), op->type.c_str(), rois->shape.size(), rois->shape.empty() ? -1 : rois->shape[0], rois->shape.size() < 2 ? -1 : rois->shape[1]);
        return -1;
    }
    for (size_t i = 2; i < rois->shape.size(); i++)
        if (rois->shape[i] != 1)
        {
            fprintf(stderr, "caffe layer %s (%s): trailing ROI dimension %zu must be 1, got %d\n", op->name.c_str(), op->type.c_str(), i, rois->shape[i]);
            return -1;
        }

    int pooled_h;
    int pooled_w;
    int channels = feature->shape[1];
    float spatial_scale;
    if (op->type == "caffe.PSROIPooling")
    {
        pooled_h = pooled_w = get_int(op, "group_size", 0);
        channels = get_int(op, "output_dim", 0);
        spatial_scale = get_float(op, "spatial_scale", 0.f);
        if ((int64_t)channels * pooled_h * pooled_w != feature->shape[1])
        {
            fprintf(stderr, "caffe layer %s (%s): output_dim %d * group_size %d * %d must equal feature channels %d\n", op->name.c_str(), op->type.c_str(), channels, pooled_h, pooled_w, feature->shape[1]);
            return -1;
        }
    }
    else
    {
        pooled_h = get_int(op, "pooled_h", 0);
        pooled_w = get_int(op, "pooled_w", 0);
        spatial_scale = get_float(op, "spatial_scale", 1.f);
    }
    if (pooled_h <= 0 || pooled_w <= 0 || channels <= 0 || !std::isfinite(spatial_scale) || spatial_scale <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): pooled dimensions/channels must be positive and spatial_scale finite and positive; pooled=%dx%d channels=%d spatial_scale=%g\n", op->name.c_str(), op->type.c_str(), pooled_h, pooled_w, channels, spatial_scale);
        return -1;
    }

    const int count = rois->shape[0];
    feature->params["__batch_index"] = 0;
    rois->params["__batch_index"] = 233;
    feature->remove_consumer(op);
    rois->remove_consumer(op);
    if (rois->shape.size() != 2)
    {
        Operator* reshape = graph.new_operator_before("Tensor.reshape", op->name + "_rois", op);
        Operand* out = graph.new_operand(op->name + "_rois");
        reshape->inputs.push_back(rois);
        rois->consumers.push_back(reshape);
        reshape->outputs.push_back(out);
        reshape->params["shape"] = std::vector<int> {count, 5};
        out->shape = {count, 5};
        out->type = 1;
        out->params["__batch_index"] = 233;
        out->producer = reshape;
        rois = out;
    }
    const std::string type = op->type;
    op->inputs.clear();
    op->params.clear();
    op->type = "torch.stack";
    op->params["dim"] = 0;
    for (int i = 0; i < count; i++)
    {
        const std::string name = op->name + "_roi" + std::to_string(i);
        Operator* select = graph.new_operator_before("Tensor.select", name + "_select", op);
        Operand* roi = graph.new_operand(name + "_coordinates");
        select->inputs.push_back(rois);
        rois->consumers.push_back(select);
        select->outputs.push_back(roi);
        select->params["dim"] = 0;
        select->params["index"] = i;
        roi->shape = {5};
        roi->type = 1;
        roi->params["__batch_index"] = 233;
        roi->producer = select;

        Operator* pool = graph.new_operator_before(type, name, op);
        Operand* out = graph.new_operand(name + "_out");
        pool->inputs = {feature, roi};
        feature->consumers.push_back(pool);
        roi->consumers.push_back(pool);
        pool->outputs.push_back(out);
        pool->params["pooled_h"] = pooled_h;
        pool->params["pooled_w"] = pooled_w;
        pool->params["spatial_scale"] = spatial_scale;
        if (type == "caffe.PSROIPooling")
            pool->params["output_dim"] = channels;
        out->shape = {channels, pooled_h, pooled_w};
        out->type = 1;
        out->params["__batch_index"] = 233;
        out->producer = pool;
        out->consumers.push_back(op);
        op->inputs.push_back(out);
    }
    op->outputs[0]->shape = {count, channels, pooled_h, pooled_w};
    op->outputs[0]->params["__batch_index"] = 233;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
