// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "layer.h"

namespace ncnn {

class Scale : public Layer
{
public:
    Scale();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
    Mat scale_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_SCALE_H
