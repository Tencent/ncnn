// Copyright 2025 xiaofan <xiaofan@iscas.ac.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELTWISE_RISCV_H
#define LAYER_ELTWISE_RISCV_H

#include "eltwise.h"

namespace ncnn {

class Eltwise_riscv : public Eltwise
{
public:
    Eltwise_riscv();
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ZFH
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ELTWISE_RISCV_H
