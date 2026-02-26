// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING_H
#define LAYER_POOLING_H

#include "layer.h"

namespace ncnn {

class Pooling : public Layer
{
public:
    Pooling();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    enum PoolMethod
    {
        PoolMethod_MAX = 0,
        PoolMethod_AVE = 1
    };

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;

public:
    // param
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int global_pooling;
    int pad_mode; // 0=full 1=valid 2=SAME_UPPER 3=SAME_LOWER
    int avgpool_count_include_pad;
    int adaptive_pooling;
    int out_w;
    int out_h;
};

} // namespace ncnn

#endif // LAYER_POOLING_H
