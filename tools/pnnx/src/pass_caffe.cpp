// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_caffe.h"

#include <stdio.h>
#include <algorithm>

#include "pass_caffe/lower_argmax.h"
#include "pass_caffe/lower_bnll.h"
#include "pass_caffe/lower_concat.h"
#include "pass_caffe/lower_crop.h"
#include "pass_caffe/lower_eltwise.h"
#include "pass_caffe/lower_exp_log.h"
#include "pass_caffe/lower_lrn.h"
#include "pass_caffe/lower_lstm.h"
#include "pass_caffe/lower_normalization.h"
#include "pass_caffe/lower_power.h"
#include "pass_caffe/lower_proposal.h"
#include "pass_caffe/lower_reduction.h"
#include "pass_caffe/lower_roi.h"
#include "pass_caffe/lower_scale.h"
#include "pass_caffe/lower_threshold.h"
#include "pass_caffe/parameter.h"
#include "pass_caffe/prepare_batch_norm.h"
#include "pass_caffe/prepare_convolution.h"
#include "pass_caffe/prepare_detection.h"
#include "pass_caffe/prepare_dropout.h"
#include "pass_caffe/prepare_embed.h"
#include "pass_caffe/prepare_flatten.h"
#include "pass_caffe/prepare_inner_product.h"
#include "pass_caffe/prepare_interp.h"
#include "pass_caffe/prepare_permute.h"
#include "pass_caffe/prepare_pooling.h"
#include "pass_caffe/prepare_prelu.h"
#include "pass_caffe/prepare_reorg.h"
#include "pass_caffe/prepare_reshape.h"
#include "pass_caffe/prepare_shuffle_channel.h"
#include "pass_caffe/prepare_slice.h"
#include "pass_caffe/prepare_softmax.h"
#include "pass_caffe/prepare_tile.h"
#include "pass_caffe/prepare_yolo.h"

namespace pnnx {

int check_caffe_operators(const Graph& graph)
{
    for (const Operator* op : graph.ops)
    {
        if (op->type.compare(0, 6, "caffe.") != 0)
            continue;

        // these extensions have both Python helpers and ncnn exporters
        if (op->type == "caffe.PriorBox" || op->type == "caffe.DetectionOutput"
                || op->type == "caffe.ROIPooling" || op->type == "caffe.PSROIPooling" || op->type == "caffe.ROIAlign"
                || op->type == "caffe.Proposal" || op->type == "caffe.ArgMax" || op->type == "caffe.Power"
                || op->type == "caffe.YoloDetectionOutput" || op->type == "caffe.Yolov3DetectionOutput")
            continue;

        fprintf(stderr, "caffe layer %s (%s): operator was not converted by level2\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    return 0;
}

int pass_caffe(Graph& graph)
{
    if (caffe2pnnx::prepare_embed_indices(graph) != 0)
        return -1;
    // lower in execution order so each layer sees its input shapes
    const std::vector<Operator*> ops = graph.ops;
    for (Operator* op : ops)
    {
        if (op->type.compare(0, 6, "caffe.") != 0)
            continue;

        const std::string type = op->type;
        if (type == "caffe.Split")
        {
            if (op->inputs.size() != 1 || op->outputs.empty())
            {
                fprintf(stderr, "caffe layer %s (%s): Split requires one input and at least one output, got %zu and %zu\n", op->name.c_str(), type.c_str(), op->inputs.size(), op->outputs.size());
                return -1;
            }
            for (Operand* output : op->outputs)
            {
                Operator* identity = graph.new_operator_before("F.dropout", op->name + "_" + output->name, op);
                identity->params["p"] = 0.f;
                identity->params["training"] = false;
                identity->inputs = op->inputs;
                identity->outputs.push_back(output);
                op->inputs[0]->consumers.push_back(identity);
                output->producer = identity;
                output->shape = op->inputs[0]->shape;
                output->type = op->inputs[0]->type;
                if (caffe2pnnx::has_known_shape(op->inputs[0]))
                    output->params["__caffe_shape_known"] = true;
            }
            op->inputs[0]->remove_consumer(op);
            op->inputs.clear();
            op->outputs.clear();
            graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
            delete op;
            continue;
        }
        int ret;
        if (type == "caffe.Convolution" || type == "caffe.ConvolutionDepthwise" || type == "caffe.DepthwiseConvolution" || type == "caffe.Deconvolution")
            ret = caffe2pnnx::prepare_convolution(op);
        else if (type == "caffe.Pooling")
            ret = caffe2pnnx::prepare_pooling(op);
        else if (type == "caffe.PriorBox" || type == "caffe.DetectionOutput")
            ret = caffe2pnnx::prepare_detection(op);
        else if (type == "caffe.MVN" || type == "caffe.Normalize")
            ret = caffe2pnnx::lower_normalization(graph, op);
        else if (type == "caffe.Crop")
            ret = caffe2pnnx::lower_crop(op);
        else if (type == "caffe.Embed")
            ret = caffe2pnnx::prepare_embed(op);
        else if (type == "caffe.Interp")
            ret = caffe2pnnx::prepare_interp(graph, op);
        else if (type == "caffe.InnerProduct")
            ret = caffe2pnnx::prepare_inner_product(graph, op);
        else if (type == "caffe.BatchNorm")
            ret = caffe2pnnx::prepare_batch_norm(op);
        else if (type == "caffe.Scale" || type == "caffe.BN")
            ret = caffe2pnnx::lower_scale(graph, op);
        else if (type == "caffe.Tile")
            ret = caffe2pnnx::prepare_tile(op);
        else if (type == "caffe.Exp" || type == "caffe.Log")
            ret = caffe2pnnx::prepare_exp_log(op);
        else if (type == "caffe.Threshold")
            ret = caffe2pnnx::prepare_threshold(op);
        else if (type == "caffe.BNLL")
            ret = caffe2pnnx::prepare_bnll(op);
        else if (type == "caffe.ELU" || type == "caffe.ReLU6"
                 || type == "caffe.ReLU" || type == "caffe.Sigmoid" || type == "caffe.TanH" || type == "caffe.AbsVal")
        {
            if (op->inputs.size() != 1 || op->outputs.size() != 1)
            {
                fprintf(stderr, "caffe layer %s (%s): expected one input and one output, got %zu and %zu\n", op->name.c_str(), type.c_str(), op->inputs.size(), op->outputs.size());
                return -1;
            }
            if (!caffe2pnnx::has_known_shape(op->inputs[0]))
            {
                fprintf(stderr, "caffe layer %s (%s): input shape is unknown; provide inputshape\n", op->name.c_str(), type.c_str());
                return -1;
            }
            op->outputs[0]->shape = op->inputs[0]->shape;
            if (type == "caffe.AbsVal")
                op->type = "aten::abs";
            if (type == "caffe.Sigmoid")
                op->type = "aten::sigmoid";
            if (type == "caffe.TanH")
                op->type = "aten::tanh";
            if (type == "caffe.AbsVal" || type == "caffe.Sigmoid" || type == "caffe.TanH")
                op->params.clear();
            ret = 0;
        }
        else if (type == "caffe.ROIPooling" || type == "caffe.PSROIPooling" || type == "caffe.ROIAlign")
            ret = caffe2pnnx::lower_roi(graph, op);
        else if (type == "caffe.Python")
            ret = caffe2pnnx::lower_proposal(op);
        else if (type == "caffe.LSTM")
            ret = caffe2pnnx::lower_lstm(op);
        else if (type == "caffe.Reduction")
            ret = caffe2pnnx::lower_reduction(graph, op);
        else if (type == "caffe.ArgMax")
            ret = caffe2pnnx::lower_argmax(op);
        else if (type == "caffe.YoloDetectionOutput" || type == "caffe.Yolov3DetectionOutput")
            ret = caffe2pnnx::prepare_yolo(op);
        else if (type == "caffe.Power")
            ret = caffe2pnnx::prepare_power(op);
        else if (type == "caffe.LRN")
            ret = caffe2pnnx::lower_lrn(graph, op);
        else if (type == "caffe.Slice")
            ret = caffe2pnnx::prepare_slice(graph, op);
        else if (type == "caffe.Reorg")
            ret = caffe2pnnx::prepare_reorg(op);
        else if (type == "caffe.ShuffleChannel")
            ret = caffe2pnnx::prepare_shuffle_channel(op);
        else if (type == "caffe.Eltwise")
            ret = caffe2pnnx::prepare_eltwise(graph, op);
        else if (type == "caffe.Concat")
            ret = caffe2pnnx::lower_concat(graph, op);
        else if (type == "caffe.PReLU")
            ret = caffe2pnnx::prepare_prelu(op);
        else if (type == "caffe.Softmax")
            ret = caffe2pnnx::prepare_softmax(op);
        else if (type == "caffe.Dropout")
            ret = caffe2pnnx::prepare_dropout(op);
        else if (type == "caffe.Flatten")
            ret = caffe2pnnx::prepare_flatten(op);
        else if (type == "caffe.Permute")
            ret = caffe2pnnx::prepare_permute(op);
        else if (type == "caffe.Reshape")
            ret = caffe2pnnx::prepare_reshape(op);
        else
        {
            fprintf(stderr, "caffe layer %s (%s): unsupported operator type\n", op->name.c_str(), type.c_str());
            return -1;
        }

        if (ret != 0)
            return -1;
        for (Operand* r : op->outputs)
        {
            r->type = 1;
            r->params["__caffe_shape_known"] = true;
        }
    }
    // Shape validity is only needed while preparing the Caffe source graph.
    for (Operand* r : graph.operands)
        r->params.erase("__caffe_shape_known");
    caffe2pnnx::lower_exp_log(graph);
    caffe2pnnx::lower_threshold(graph);
    caffe2pnnx::lower_bnll(graph);
    caffe2pnnx::lower_power(graph);
    caffe2pnnx::lower_eltwise(graph);
    return 0;
}

} // namespace pnnx
