// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_POWERPC_H
#define LAYER_BATCHNORM_POWERPC_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_powerpc : public BatchNorm
{
public:
    BatchNorm_powerpc();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_POWERPC_H
