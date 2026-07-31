// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INPUT_H
#define LAYER_INPUT_H

#include "layer.h"

namespace ncnn {

class Input : public Layer
{
public:
    Input();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

public:
    int w;
    int h;
    int d;
    int c;
};

} // namespace ncnn

#endif // LAYER_INPUT_H
