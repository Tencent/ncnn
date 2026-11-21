// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PIXELSHUFFLE_POWERPC_H
#define LAYER_PIXELSHUFFLE_POWERPC_H

#include "pixelshuffle.h"

namespace ncnn {

class PixelShuffle_powerpc : public PixelShuffle
{
public:
    PixelShuffle_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_PIXELSHUFFLE_POWERPC_H
