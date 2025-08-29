// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_RISCV_H
#define LAYER_CAST_RISCV_H

#include "cast.h"

namespace ncnn {

class Cast_riscv : public Cast
{
public:
    Cast_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

#if NCNN_ZFH
    void cast_fp32_to_fp16(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    void cast_fp16_to_fp32(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_CAST_RISCV_H
