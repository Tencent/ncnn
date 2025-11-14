// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_H
#define LAYER_ROTARYEMBED_H

#include "layer.h"

namespace ncnn {

class RotaryEmbed : public Layer
{
public:
    RotaryEmbed();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int interleaved;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_H
