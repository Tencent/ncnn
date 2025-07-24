// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_RISCV_H
#define LAYER_INTERP_RISCV_H

#include "interp.h"

namespace ncnn {

class Interp_riscv : public Interp
{
public:
    Interp_riscv();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_INTERP_RISCV_H
