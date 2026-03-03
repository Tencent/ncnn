// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING_ARM_H
#define LAYER_POOLING_ARM_H

#include "pooling.h"

namespace ncnn {

class Pooling_arm : public Pooling
{
public:
    Pooling_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_POOLING_ARM_H
