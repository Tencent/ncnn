// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_LOONGARCH_H
#define LAYER_ROTARYEMBED_LOONGARCH_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_loongarch : public RotaryEmbed
{
public:
    RotaryEmbed_loongarch();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_BF16
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_LOONGARCH_H
