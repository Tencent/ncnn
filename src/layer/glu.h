// Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GLU_H
#define LAYER_GLU_H

#include "layer.h"

namespace ncnn {

class GLU : public Layer
{
public:
    GLU();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob,
                        const Option& opt) const;

public:
    int axis;
};

} // namespace ncnn

#endif // LAYER_GLU_H
