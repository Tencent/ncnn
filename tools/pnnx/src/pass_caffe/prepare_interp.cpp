// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_interp.h"

#include "parameter.h"

#include <stdio.h>

#include <limits.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_interp(Graph& graph, Operator* op)
{
    if (op->inputs.size() != 1 || op->outputs.size() != 1 || op->inputs[0]->shape.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): expected one input/output with known nonempty shape; inputs=%zu outputs=%zu rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size());
        return -1;
    }
    std::vector<int> shape = op->inputs[0]->shape;
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): expected rank4 input with positive spatial dimensions; rank=%zu height=%d width=%d\n", op->name.c_str(), op->type.c_str(), shape.size(), shape.size() == 4 ? shape[2] : 0, shape.size() == 4 ? shape[3] : 0);
        return -1;
    }
    const int height = get_int(op, "height", 0);
    const int width = get_int(op, "width", 0);
    const int zoom = get_int(op, "zoom_factor", 1);
    const int shrink = get_int(op, "shrink_factor", 1);
    const int pad_beg = get_int(op, "pad_beg", 0);
    const int pad_end = get_int(op, "pad_end", 0);
    if (height < 0 || width < 0 || zoom <= 0 || (height == 0) != (width == 0))
    {
        fprintf(stderr, "caffe layer %s (%s): invalid height=%d width=%d zoom_factor=%d; sizes must both be positive or both zero, zoom must be positive\n", op->name.c_str(), op->type.c_str(), height, width, zoom);
        return -1;
    }
    if (shrink <= 0 || pad_beg > 0 || pad_end > 0)
    {
        fprintf(stderr, "caffe layer %s (%s): require shrink_factor>0 and nonpositive padding; shrink_factor=%d pad_beg=%d pad_end=%d\n", op->name.c_str(), op->type.c_str(), shrink, pad_beg, pad_end);
        return -1;
    }
    const int64_t cropped_h = (int64_t)shape[2] + pad_beg + pad_end;
    const int64_t cropped_w = (int64_t)shape[3] + pad_beg + pad_end;
    if (cropped_h <= 0 || cropped_w <= 0)
    {
        fprintf(stderr, "caffe layer %s (%s): padding removes the spatial input; cropped height=%lld width=%lld\n", op->name.c_str(), op->type.c_str(), (long long)cropped_h, (long long)cropped_w);
        return -1;
    }

    const bool has_zoom = op->has_param("zoom_factor");
    const bool has_shrink = op->has_param("shrink_factor");
    int64_t out_h;
    int64_t out_w;
    if (has_shrink && !has_zoom)
    {
        out_h = (cropped_h - 1) / shrink + 1;
        out_w = (cropped_w - 1) / shrink + 1;
    }
    else if (has_zoom && !has_shrink)
    {
        out_h = (cropped_h - 1) * zoom + 1;
        out_w = (cropped_w - 1) * zoom + 1;
    }
    else if (height && width)
    {
        out_h = height;
        out_w = width;
    }
    else if (has_zoom && has_shrink)
    {
        out_h = (cropped_h - 1) / shrink * zoom + 1;
        out_w = (cropped_w - 1) / shrink * zoom + 1;
    }
    else
    {
        fprintf(stderr, "caffe layer %s (%s): no output size rule: height=%d width=%d zoom_factor present=%d shrink_factor present=%d\n", op->name.c_str(), op->type.c_str(), height, width, has_zoom, has_shrink);
        return -1;
    }
    if (out_h > INT_MAX || out_w > INT_MAX)
    {
        fprintf(stderr, "caffe layer %s (%s): computed output height=%lld width=%lld exceeds INT_MAX\n", op->name.c_str(), op->type.c_str(), (long long)out_h, (long long)out_w);
        return -1;
    }

    if (pad_beg || pad_end)
    {
        Operator* crop = graph.new_operator_before("Tensor.slice", op->name + "_crop", op);
        crop->params["dims"] = std::vector<int> {2, 3};
        crop->params["starts"] = std::vector<int> {-pad_beg, -pad_beg};
        crop->params["ends"] = std::vector<int> {shape[2] + pad_end, shape[3] + pad_end};
        crop->params["steps"] = std::vector<int> {1, 1};
        Operand* input = op->inputs[0];
        Operand* r = graph.new_operand(op->name + "_crop");
        input->remove_consumer(op);
        input->consumers.push_back(crop);
        crop->inputs.push_back(input);
        crop->outputs.push_back(r);
        r->producer = crop;
        r->consumers.push_back(op);
        r->type = 1;
        r->shape = {shape[0], shape[1], (int)cropped_h, (int)cropped_w};
        r->params["__batch_index"] = 0;
        op->inputs[0] = r;
    }
    // caffe's Interp aligns the first and last input and output pixels
    shape[2] = (int)out_h;
    shape[3] = (int)out_w;
    op->params.clear();
    op->params["size"] = std::vector<int> {shape[2], shape[3]};
    op->params["mode"] = "bilinear";
    op->params["align_corners"] = true;
    op->outputs[0]->shape = shape;
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
