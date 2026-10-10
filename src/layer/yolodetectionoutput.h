// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_YOLODETECTIONOUTPUT_H
#define LAYER_YOLODETECTIONOUTPUT_H

#include "layer.h"

namespace ncnn {

class YoloDetectionOutput : public Layer
{
public:
    YoloDetectionOutput();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;

public:
    int num_class;
    int num_box;
    float confidence_threshold;
    float nms_threshold;
    Mat biases;

    ncnn::Layer* softmax;
};

} // namespace ncnn

#endif // LAYER_YOLODETECTIONOUTPUT_H
