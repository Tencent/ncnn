// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_X86_H
#define LAYER_CONCAT_X86_H

#include "concat.h"

namespace ncnn {

class Concat_x86 : public Concat
{
public:
    Concat_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CONCAT_X86_H
