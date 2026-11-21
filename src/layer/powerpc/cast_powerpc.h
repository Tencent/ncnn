// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_POWERPC_H
#define LAYER_CAST_POWERPC_H

#include "cast.h"

namespace ncnn {

class Cast_powerpc : public Cast
{
public:
    Cast_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CAST_POWERPC_H
