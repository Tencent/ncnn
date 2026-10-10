// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEEPCOPY_H
#define LAYER_DEEPCOPY_H

#include "layer.h"

namespace ncnn {

class DeepCopy : public Layer
{
public:
    DeepCopy();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DEEPCOPY_H
