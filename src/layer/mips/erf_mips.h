// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_MIPS_H
#define LAYER_ERF_MIPS_H

#include "erf.h"

namespace ncnn {

class Erf_mips : public Erf
{
public:
    Erf_mips();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ERF_MIPS_H
