// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_LOONGARCH_H
#define LAYER_ERF_LOONGARCH_H

#include "erf.h"

namespace ncnn {

class Erf_loongarch : public Erf
{
public:
    Erf_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ERF_LOONGARCH_H
