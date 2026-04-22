// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MOD_H
#define LAYER_MOD_H

#include "layer.h"

namespace ncnn {

class Mod : public Layer
{
public:
    Mod();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int fmod; // 0 = remainder (Python-style), 1 = fmod (C-style)
};

} // namespace ncnn

#endif // LAYER_MOD_H
