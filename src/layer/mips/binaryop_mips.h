// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BINARYOP_MIPS_H
#define LAYER_BINARYOP_MIPS_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_mips : public BinaryOp
{
public:
    BinaryOp_mips();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BINARYOP_MIPS_H
