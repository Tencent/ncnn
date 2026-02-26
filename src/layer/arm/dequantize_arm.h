// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEQUANTIZE_ARM_H
#define LAYER_DEQUANTIZE_ARM_H

#include "dequantize.h"

namespace ncnn {

class Dequantize_arm : public Dequantize
{
public:
    Dequantize_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_DEQUANTIZE_ARM_H
