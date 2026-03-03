// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_LOONGARCH_H
#define LAYER_CLIP_LOONGARCH_H

#include "clip.h"

namespace ncnn {

class Clip_loongarch : public Clip
{
public:
    Clip_loongarch();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CLIP_LOONGARCH_H
