// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_H
#define LAYER_LRN_H

#include "layer.h"

namespace ncnn {

class LRN : public Layer
{
public:
    LRN();

    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    enum NormRegionType
    {
        NormRegion_ACROSS_CHANNELS = 0,
        NormRegion_WITHIN_CHANNEL = 1
    };

public:
    // param
    int region_type;
    int local_size;
    float alpha;
    float beta;
    float bias;
};

} // namespace ncnn

#endif // LAYER_LRN_H
