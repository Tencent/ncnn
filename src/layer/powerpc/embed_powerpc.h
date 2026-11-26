// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_EMBED_POWERPC_H
#define LAYER_EMBED_POWERPC_H

#include "embed.h"

namespace ncnn {

class Embed_powerpc : public Embed
{
public:
    Embed_powerpc();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_EMBED_POWERPC_H
