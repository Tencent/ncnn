// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHUFFLECHANNEL_POWERPC_H
#define LAYER_SHUFFLECHANNEL_POWERPC_H

#include "shufflechannel.h"

namespace ncnn {

class ShuffleChannel_powerpc : public ShuffleChannel
{
public:
    ShuffleChannel_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_POWERPC_H
