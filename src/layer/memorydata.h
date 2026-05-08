// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MEMORYDATA_H
#define LAYER_MEMORYDATA_H

#include "layer.h"

namespace ncnn {

class MemoryData : public Layer
{
public:
    MemoryData();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int w;
    int h;
    int d;
    int c;
    int load_type;

    Mat data;
};

} // namespace ncnn

#endif // LAYER_MEMORYDATA_H
