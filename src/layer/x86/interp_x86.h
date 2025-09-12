// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_X86_H
#define LAYER_INTERP_X86_H

#include "interp.h"

namespace ncnn {

class Interp_x86 : public Interp
{
public:
    Interp_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_INTERP_X86_H
