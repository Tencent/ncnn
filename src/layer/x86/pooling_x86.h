// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING_X86_H
#define LAYER_POOLING_X86_H

#include "pooling.h"

namespace ncnn {

class Pooling_x86 : public Pooling
{
public:
    Pooling_x86();

    virtual int create_pipeline(const Option& opt);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob,
                        const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_POOLING_X86_H
