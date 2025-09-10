// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHUFFLECHANNEL_RISCV_H
#define LAYER_SHUFFLECHANNEL_RISCV_H

#include "shufflechannel.h"

namespace ncnn {

class ShuffleChannel_riscv : virtual public ShuffleChannel
{
public:
    ShuffleChannel_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_RISCV_H
