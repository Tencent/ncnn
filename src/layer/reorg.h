// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REORG_H
#define LAYER_REORG_H

#include "layer.h"

namespace ncnn {

class Reorg : public Layer
{
public:
    Reorg();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int stride;
    int mode;
};

} // namespace ncnn

#endif // LAYER_REORG_H
