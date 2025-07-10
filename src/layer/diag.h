// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DIAG_H
#define LAYER_DIAG_H

#include "layer.h"

namespace ncnn {

class Diag : public Layer
{
public:
    Diag();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int diagonal;
};

} // namespace ncnn

#endif // LAYER_DIAG_H
