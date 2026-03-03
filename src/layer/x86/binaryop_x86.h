// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BINARYOP_X86_H
#define LAYER_BINARYOP_X86_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_x86 : public BinaryOp
{
public:
    BinaryOp_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BINARYOP_X86_H
