// Copyright 2021 Xavier Hsinyuan <thelastlinex@hotmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BINARYOP_RISCV_H
#define LAYER_BINARYOP_RISCV_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_riscv : public BinaryOp
{
public:
    BinaryOp_riscv();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BINARYOP_RISCV_H
