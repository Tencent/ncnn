// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CROP_LOONGARCH_H
#define LAYER_CROP_LOONGARCH_H

#include "crop.h"

namespace ncnn {

class Crop_loongarch : public Crop
{
public:
    Crop_loongarch();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CROP_LOONGARCH_H
