// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_H
#define LAYER_ERF_H

#include "layer.h"

namespace ncnn {

class Erf : public Layer
{
public:
    Erf();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ERF_H
