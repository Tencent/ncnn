// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PIXELSHUFFLE_H
#define LAYER_PIXELSHUFFLE_H

#include "layer.h"

namespace ncnn {

class PixelShuffle : public Layer
{
public:
    PixelShuffle();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int upscale_factor;
    int mode;
};

} // namespace ncnn

#endif // LAYER_PIXELSHUFFLE_H
