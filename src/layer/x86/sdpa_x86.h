// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_X86_H
#define LAYER_SDPA_X86_H

#include "sdpa.h"

namespace ncnn {

class SDPA_x86 : public SDPA
{
public:
    SDPA_x86();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    Layer* qk_gemm;
    Layer* qkv_gemm;

    Layer* qk_softmax;
};

} // namespace ncnn

#endif // LAYER_SDPA_X86_H
