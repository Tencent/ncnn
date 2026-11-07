// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prepare_detection.h"

#include "parameter.h"

#include <float.h>
#include <cmath>
#include <stdio.h>

namespace pnnx {

namespace caffe2pnnx {

int prepare_detection(Operator* op)
{
    if (!op->attrs.empty() || op->outputs.size() != 1)
    {
        fprintf(stderr, "caffe layer %s (%s): expected no weight blobs and one output, got %zu blobs and %zu outputs\n", op->name.c_str(), op->type.c_str(), op->attrs.size(), op->outputs.size());
        return -1;
    }

    if (op->type == "caffe.PriorBox")
    {
        if (op->inputs.size() != 2 || op->inputs[0]->shape.size() != 4 || op->inputs[1]->shape.size() != 4)
        {
            fprintf(stderr, "caffe layer %s (%s): PriorBox requires two rank-4 inputs; inputs=%zu feature_rank=%zu image_rank=%zu\n", op->name.c_str(), op->type.c_str(), op->inputs.size(), op->inputs.empty() ? 0u : op->inputs[0]->shape.size(), op->inputs.size() < 2 ? 0u : op->inputs[1]->shape.size());
            return -1;
        }
        const std::vector<float> min_sizes = get_parameter(op, "min_size", std::vector<float>()).af;
        const std::vector<float> max_sizes = get_parameter(op, "max_size", std::vector<float>()).af;
        if (min_sizes.empty() || (!max_sizes.empty() && max_sizes.size() != min_sizes.size()))
        {
            fprintf(stderr, "caffe layer %s (%s): min_size must be nonempty and max_size empty or have matching length; counts=%zu/%zu\n", op->name.c_str(), op->type.c_str(), min_sizes.size(), max_sizes.size());
            return -1;
        }
        for (size_t i = 0; i < min_sizes.size(); i++)
            if (!std::isfinite(min_sizes[i]) || min_sizes[i] <= 0 || (!max_sizes.empty() && (!std::isfinite(max_sizes[i]) || max_sizes[i] <= min_sizes[i])))
            {
                fprintf(stderr, "caffe layer %s (%s): size pair %zu requires finite positive min_size and finite max_size greater than min_size; min=%g max=%g (0 if absent)\n", op->name.c_str(), op->type.c_str(), i, min_sizes[i], max_sizes.empty() ? 0.f : max_sizes[i]);
                return -1;
            }

        const bool flip = get_bool(op, "flip", true);
        std::vector<float> ratios = {1.f};
        for (float ar : get_parameter(op, "aspect_ratio", std::vector<float>()).af)
        {
            if (!std::isfinite(ar) || ar <= 0)
            {
                fprintf(stderr, "caffe layer %s (%s): aspect_ratio must be finite and positive, got %g\n", op->name.c_str(), op->type.c_str(), ar);
                return -1;
            }
            bool exists = false;
            for (float r : ratios)
                exists |= fabsf(r - ar) < 1e-6f;
            if (exists)
                continue;
            ratios.push_back(ar);
            if (flip)
                ratios.push_back(1.f / ar);
        }
        ratios.erase(ratios.begin());
        std::vector<float> variance = get_parameter(op, "variance", std::vector<float>()).af;
        if (variance.empty())
            variance = {0.1f, 0.1f, 0.1f, 0.1f};
        else if (variance.size() == 1)
            variance.resize(4, variance[0]);
        else if (variance.size() != 4)
        {
            fprintf(stderr, "caffe layer %s (%s): variance must contain 0, 1 or 4 values, got %zu\n", op->name.c_str(), op->type.c_str(), variance.size());
            return -1;
        }
        for (float v : variance)
            if (!std::isfinite(v) || v <= 0)
            {
                fprintf(stderr, "caffe layer %s (%s): variance must be finite and positive, got %g\n", op->name.c_str(), op->type.c_str(), v);
                return -1;
            }

        int image_h = get_int(op, "img_h", 0);
        int image_w = get_int(op, "img_w", 0);
        const int image_size = get_int(op, "img_size", 0);
        if ((image_size && (image_h || image_w)) || (image_h == 0) != (image_w == 0))
        {
            fprintf(stderr, "caffe layer %s (%s): img_size and img_h/img_w are mutually exclusive, and img_h/img_w must be supplied together; img_size=%d img_h=%d img_w=%d\n", op->name.c_str(), op->type.c_str(), image_size, image_h, image_w);
            return -1;
        }
        if (image_size)
            image_h = image_w = image_size;
        if (!image_h)
        {
            image_h = op->inputs[1]->shape[2];
            image_w = op->inputs[1]->shape[3];
        }
        float step_h = get_float(op, "step_h", 0.f);
        float step_w = get_float(op, "step_w", 0.f);
        const float step = get_float(op, "step", 0.f);
        if ((step && (step_h || step_w)) || (step_h == 0) != (step_w == 0))
        {
            fprintf(stderr, "caffe layer %s (%s): step and step_h/step_w are mutually exclusive, and step_h/step_w must be supplied together; step=%g step_h=%g step_w=%g\n", op->name.c_str(), op->type.c_str(), step, step_h, step_w);
            return -1;
        }
        if (step)
            step_h = step_w = step;
        const int height = op->inputs[0]->shape[2];
        const int width = op->inputs[0]->shape[3];
        if (image_h <= 0 || image_w <= 0 || height <= 0 || width <= 0)
        {
            fprintf(stderr, "caffe layer %s (%s): image and feature spatial sizes must be positive; image=%dx%d feature=%dx%d\n", op->name.c_str(), op->type.c_str(), image_h, image_w, height, width);
            return -1;
        }
        if (!step_h)
        {
            step_h = (float)image_h / height;
            step_w = (float)image_w / width;
        }
        const float offset = get_float(op, "offset", 0.5f);
        if (!std::isfinite(step_h) || !std::isfinite(step_w) || step_h <= 0 || step_w <= 0 || !std::isfinite(offset))
        {
            fprintf(stderr, "caffe layer %s (%s): steps must be finite and positive and offset finite; step_h=%g step_w=%g offset=%g\n", op->name.c_str(), op->type.c_str(), step_h, step_w, offset);
            return -1;
        }
        const bool clip = get_bool(op, "clip", false);
        const int64_t count = (int64_t)height * width * ((int64_t)min_sizes.size() * (ratios.size() + 1) + max_sizes.size()) * 4;
        if (count > INT_MAX)
        {
            fprintf(stderr, "caffe layer %s (%s): PriorBox coordinate count %lld exceeds INT_MAX\n", op->name.c_str(), op->type.c_str(), (long long)count);
            return -1;
        }
        op->params.clear();
        op->params["min_sizes"] = min_sizes;
        op->params["max_sizes"] = max_sizes;
        op->params["aspect_ratios"] = ratios;
        op->params["variance"] = variance;
        op->params["clip"] = clip;
        op->params["image_h"] = image_h;
        op->params["image_w"] = image_w;
        op->params["step_h"] = step_h;
        op->params["step_w"] = step_w;
        op->params["offset"] = offset;
        op->outputs[0]->shape = {1, 2, (int)count};
        op->outputs[0]->params["__batch_index"] = 0;
        return 0;
    }

    if (op->type == "caffe.DetectionOutput")
    {
        if (op->inputs.size() != 3)
        {
            fprintf(stderr, "caffe layer %s (%s): DetectionOutput requires location, confidence and prior inputs, got %zu inputs\n", op->name.c_str(), op->type.c_str(), op->inputs.size());
            return -1;
        }
        // the native ncnn detection representation is only valid at graph outputs
        for (const Operator* consumer : op->outputs[0]->consumers)
            if (consumer->type != "pnnx.Output")
            {
                fprintf(stderr, "caffe layer %s (%s): DetectionOutput must be terminal; consumer %s (%s) cannot use native detection layout\n", op->name.c_str(), op->type.c_str(), consumer->name.c_str(), consumer->type.c_str());
                return -1;
            }
        const int classes = get_int(op, "num_classes", 0);
        const int code_type = get_int(op, "code_type", 1);
        const float nms_threshold = get_float(op, "nms_param.nms_threshold", 0.3f);
        const float eta = get_float(op, "nms_param.eta", 1.f);
        const int top_k = get_int(op, "nms_param.top_k", -1);
        const int keep_top_k = get_int(op, "keep_top_k", -1);
        const float confidence_threshold = get_float(op, "confidence_threshold", -FLT_MAX);
        if (classes <= 1)
        {
            fprintf(stderr, "caffe layer %s (%s): num_classes must exceed 1, got %d\n", op->name.c_str(), op->type.c_str(), classes);
            return -1;
        }
        if (code_type != 2)
        {
            fprintf(stderr, "caffe layer %s (%s): code_type must be CENTER_SIZE (2), got %d\n", op->name.c_str(), op->type.c_str(), code_type);
            return -1;
        }
        if (!get_bool(op, "share_location", true))
        {
            fprintf(stderr, "caffe layer %s (%s): share_location=false is unsupported\n", op->name.c_str(), op->type.c_str());
            return -1;
        }
        if (get_int(op, "background_label_id", 0) != 0)
        {
            fprintf(stderr, "caffe layer %s (%s): background_label_id must be 0, got %d\n", op->name.c_str(), op->type.c_str(), get_int(op, "background_label_id", 0));
            return -1;
        }
        if (get_bool(op, "variance_encoded_in_target", false))
        {
            fprintf(stderr, "caffe layer %s (%s): variance_encoded_in_target=true is unsupported\n", op->name.c_str(), op->type.c_str());
            return -1;
        }
        if (eta != 1.f)
        {
            fprintf(stderr, "caffe layer %s (%s): adaptive NMS is unsupported; nms_param.eta must be 1, got %g\n", op->name.c_str(), op->type.c_str(), eta);
            return -1;
        }
        if (!std::isfinite(nms_threshold) || nms_threshold < 0)
        {
            fprintf(stderr, "caffe layer %s (%s): nms_param.nms_threshold must be finite and nonnegative, got %g\n", op->name.c_str(), op->type.c_str(), nms_threshold);
            return -1;
        }
        if (top_k < -1)
        {
            fprintf(stderr, "caffe layer %s (%s): nms_param.top_k must be -1 or nonnegative, got %d\n", op->name.c_str(), op->type.c_str(), top_k);
            return -1;
        }
        if (keep_top_k < -1)
        {
            fprintf(stderr, "caffe layer %s (%s): keep_top_k must be -1 or nonnegative, got %d\n", op->name.c_str(), op->type.c_str(), keep_top_k);
            return -1;
        }
        if (!std::isfinite(confidence_threshold))
        {
            fprintf(stderr, "caffe layer %s (%s): confidence_threshold must be finite, got %g\n", op->name.c_str(), op->type.c_str(), confidence_threshold);
            return -1;
        }
        for (int i = 0; i < 2; i++)
            if (op->inputs[i]->shape.empty() || op->inputs[i]->shape[0] != 1)
            {
                fprintf(stderr, "caffe layer %s (%s): input %d requires known singleton batch; rank=%zu batch=%d (provide inputshape if unknown)\n", op->name.c_str(), op->type.c_str(), i, op->inputs[i]->shape.size(), op->inputs[i]->shape.empty() ? -1 : op->inputs[i]->shape[0]);
                return -1;
            }
        const std::vector<int>& prior_shape = op->inputs[2]->shape;
        if (prior_shape.size() != 3 || prior_shape[0] != 1 || prior_shape[1] != 2 || prior_shape[2] <= 0 || prior_shape[2] % 4)
        {
            fprintf(stderr, "caffe layer %s (%s): prior input must be [1,2,4*num_prior] with positive num_prior; rank=%zu dimensions=%d,%d,%d\n", op->name.c_str(), op->type.c_str(), prior_shape.size(), prior_shape.empty() ? -1 : prior_shape[0], prior_shape.size() < 2 ? -1 : prior_shape[1], prior_shape.size() < 3 ? -1 : prior_shape[2]);
            return -1;
        }
        const int num_prior = prior_shape[2] / 4;
        for (int i = 0; i < 2; i++)
        {
            int64_t count = 1;
            for (int dim : op->inputs[i]->shape)
            {
                if (dim <= 0 || count > INT_MAX / dim)
                {
                    fprintf(stderr, "caffe layer %s (%s): input %d dimensions must be positive and element count fit INT_MAX; dimension=%d accumulated=%lld\n", op->name.c_str(), op->type.c_str(), i, dim, (long long)count);
                    return -1;
                }
                count *= dim;
            }
            if (count != (int64_t)num_prior * (i == 0 ? 4 : classes))
            {
                fprintf(stderr, "caffe layer %s (%s): input %d has %lld elements, expected %lld from %d priors and %d classes\n", op->name.c_str(), op->type.c_str(), i, (long long)count, (long long)num_prior * (i == 0 ? 4 : classes), num_prior, classes);
                return -1;
            }
        }
        for (Operand* r : op->inputs)
            r->params["__batch_index"] = 0;
        fprintf(stderr, "warning: caffe layer %s (%s): existing ncnn DetectionOutput assumes normalized class probabilities; unnormalized scores with background >= 1-confidence_threshold and foreground > confidence_threshold are unsupported.\n", op->name.c_str(), op->type.c_str());
        op->params.clear();
        op->params["num_classes"] = classes;
        op->params["nms_threshold"] = nms_threshold;
        op->params["top_k"] = top_k;
        op->params["keep_top_k"] = keep_top_k;
        op->params["confidence_threshold"] = confidence_threshold;
        op->outputs[0]->shape = {1, 1, -1, 7};
        op->outputs[0]->params["__batch_index"] = 0;
        return 0;
    }
    fprintf(stderr, "caffe layer %s (%s): expected PriorBox or DetectionOutput\n", op->name.c_str(), op->type.c_str());
    return -1;
}

} // namespace caffe2pnnx

} // namespace pnnx
