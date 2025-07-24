// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CROP_X86_H
#define LAYER_CROP_X86_H

#include "crop.h"

namespace ncnn {

class Crop_x86 : public Crop
{
public:
    Crop_x86();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_CROP_X86_H
