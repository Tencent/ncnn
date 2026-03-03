// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_ARM_H
#define LAYER_PADDING_ARM_H

#include "padding.h"

namespace ncnn {

class Padding_arm : public Padding
{
public:
    Padding_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
#if NCNN_BF16
    // bf16
    unsigned short value_bf16;
    Mat per_channel_pad_data_bf16;
#endif

    // fp16
    unsigned short value_fp16;
    Mat per_channel_pad_data_fp16;
};

} // namespace ncnn

#endif // LAYER_PADDING_ARM_H
