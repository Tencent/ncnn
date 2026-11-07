// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_proposal.h"

#include "parameter.h"

#include <ctype.h>
#include <stdlib.h>

namespace pnnx {

namespace caffe2pnnx {

static void skip_space(const char*& p)
{
    while (isspace((unsigned char)*p))
        p++;
}

static int parse_proposal_parameters(const std::string& text, int& feat_stride)
{
    // accept the small YAML mapping used by py-faster-rcnn without executing Python
    const char* p = text.c_str();
    skip_space(p);
    bool braces = *p == '{';
    if (braces)
        p++;
    std::set<std::string> keys;
    while (1)
    {
        skip_space(p);
        if (!*p || (braces && *p == '}'))
            break;
        const char quote = *p == '\'' || *p == '"' ? *p++ : 0;
        const char* start = p;
        while (isalnum((unsigned char)*p) || *p == '_')
            p++;
        const std::string key(start, p);
        if (quote && *p++ != quote)
            return -1;
        skip_space(p);
        if (*p++ != ':' || !keys.insert(key).second)
            return -1;
        skip_space(p);
        if (key == "feat_stride")
        {
            char* end;
            const long value = strtol(p, &end, 10);
            if (end == p || value <= 0 || value > INT_MAX)
                return -1;
            feat_stride = (int)value;
            p = end;
        }
        else if (key == "scales")
        {
            if (*p++ != '[')
                return -1;
            const int expected[] = {8, 16, 32};
            for (int i = 0; i < 3; i++)
            {
                skip_space(p);
                char* end;
                const long value = strtol(p, &end, 10);
                if (end == p || value != expected[i])
                    return -1;
                p = end;
                skip_space(p);
                if (*p++ != (i == 2 ? ']' : ','))
                    return -1;
            }
        }
        else
        {
            return -1;
        }
        skip_space(p);
        if (*p == ',')
            p++;
        else if (*p && (!braces || *p != '}'))
            return -1;
    }
    if (braces && *p++ != '}')
        return -1;
    skip_space(p);
    return *p ? -1 : 0;
}

int lower_proposal(Operator* op)
{
    if (get_parameter(op, "layer", "").s != "ProposalLayer")
    {
        fprintf(stderr, "caffe layer %s (%s): Python layer must be ProposalLayer, got %s\n", op->name.c_str(), op->type.c_str(), get_parameter(op, "layer", "").s.c_str());
        return -1;
    }
    if (op->inputs.size() != 3 || op->outputs.empty() || op->outputs.size() > 2 || !op->attrs.empty())
    {
        fprintf(stderr, "caffe layer %s (%s): Proposal requires three inputs, one or two outputs and no blobs; got %zu inputs, %zu outputs and %zu blobs\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->outputs.size(), op->attrs.size());
        return -1;
    }
    for (size_t i = 0; i < op->outputs.size(); i++)
    {
        Operand* r = op->outputs[i];
        for (Operator* consumer : r->consumers)
        {
            if (consumer->type == "pnnx.Output")
                continue;
            fprintf(stderr, "caffe layer %s (%s): dynamic Proposal output %zu cannot feed %s (%s); existing ncnn ROI layers consume only one box, so Proposal-to-ROI requires unsupported runtime multi-ROI support\n", op->name.c_str(), op->type.c_str(), i, consumer->name.c_str(), consumer->type.c_str());
            return -1;
        }
    }
    const std::vector<int>& scores = op->inputs[0]->shape;
    const std::vector<int>& deltas = op->inputs[1]->shape;
    const std::vector<int>& info = op->inputs[2]->shape;
    if (scores.size() != 4 || deltas.size() != 4)
    {
        fprintf(stderr, "caffe layer %s (%s): scores and deltas require rank 4, got %zu and %zu\n", op->name.c_str(), op->type.c_str(), scores.size(), deltas.size());
        return -1;
    }
    if (scores[0] != 1 || deltas[0] != 1 || scores[1] != 18 || deltas[1] != 36)
    {
        fprintf(stderr, "caffe layer %s (%s): scores/deltas require batch 1 and channels 18/36; got batch %d/%d channels %d/%d\n", op->name.c_str(), op->type.c_str(), scores[0], deltas[0], scores[1], deltas[1]);
        return -1;
    }
    if (scores[2] <= 0 || scores[3] <= 0 || scores[2] != deltas[2] || scores[3] != deltas[3])
    {
        fprintf(stderr, "caffe layer %s (%s): scores/deltas require equal positive spatial sizes, got %dx%d and %dx%d\n", op->name.c_str(), op->type.c_str(), scores[2], scores[3], deltas[2], deltas[3]);
        return -1;
    }
    if (info != std::vector<int> {3} && info != std::vector<int> {1, 3})
    {
        fprintf(stderr, "caffe layer %s (%s): image info shape must be [3] or [1,3]; rank=%zu first=%d last=%d\n", op->name.c_str(), op->type.c_str(), info.size(), info.empty() ? -1 : info.front(), info.empty() ? -1 : info.back());
        return -1;
    }
    int feat_stride = 16;
    if (parse_proposal_parameters(get_parameter(op, "param_str", "").s, feat_stride) != 0)
    {
        fprintf(stderr, "caffe layer %s (%s): invalid Proposal param_str; expected unique feat_stride (positive integer) and/or scales [8,16,32], with optional mapping braces\n", op->name.c_str(), op->type.c_str());
        return -1;
    }
    fprintf(stderr, "warning: caffe layer %s (%s): existing ncnn Proposal uses exclusive NMS, non-stable tie ordering and fixed min_size=16; empty candidates may fail, and these boundary cases are not guaranteed equivalent to Caffe\n", op->name.c_str(), op->type.c_str());
    op->type = "caffe.Proposal";
    op->params.clear();
    op->params["feat_stride"] = feat_stride;
    op->params["return_scores"] = op->outputs.size() == 2;
    op->inputs[0]->params["__batch_index"] = 0;
    op->inputs[1]->params["__batch_index"] = 0;
    op->inputs[2]->params["__batch_index"] = info.size() == 2 ? 0 : 233;
    op->outputs[0]->shape = {-1, 5};
    op->outputs[0]->params["__batch_index"] = 233;
    if (op->outputs.size() == 2)
    {
        op->outputs[1]->shape = {-1, 1};
        op->outputs[1]->params["__batch_index"] = 233;
    }
    return 0;
}

} // namespace caffe2pnnx

} // namespace pnnx
