// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NORMALIZE_H
#define LAYER_NORMALIZE_H

#include "layer.h"

namespace ncnn {

class Normalize : public Layer
{
public:
    Normalize();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int across_spatial;
    int across_channel;
    int channel_shared;
    float eps;
    int scale_data_size;

    // 0 = v / sqrt(v2 + eps) caffe/mxnet
    // 1 = v / max(sqrt(v2), eps) pytorch
    // 2 = v / sqrt(max(v2, eps)) tensorflow
    int eps_mode;

    Mat scale_data;
};

} // namespace ncnn

#endif // LAYER_NORMALIZE_H
