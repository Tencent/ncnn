// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_X86_H
#define LAYER_RESHAPE_X86_H

#include "reshape.h"

namespace ncnn {

class Reshape_x86 : public Reshape
{
public:
    Reshape_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_X86_H
