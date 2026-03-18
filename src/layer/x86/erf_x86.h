// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_X86_H
#define LAYER_ERF_X86_H

#include "erf.h"

namespace ncnn {

class Erf_x86 : public Erf
{
public:
    Erf_x86();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ERF_X86_H
