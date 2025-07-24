// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_MIPS_H
#define LAYER_BATCHNORM_MIPS_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_mips : public BatchNorm
{
public:
    BatchNorm_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_MIPS_H
