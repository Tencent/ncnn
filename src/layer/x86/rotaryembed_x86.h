// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_X86_H
#define LAYER_ROTARYEMBED_X86_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_x86 : public RotaryEmbed
{
public:
    RotaryEmbed_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs,
                        std::vector<Mat>& top_blobs,
                        const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_X86_H
