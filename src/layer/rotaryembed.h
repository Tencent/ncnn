// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_H
#define LAYER_ROTARYEMBED_H

#include "layer.h"

namespace ncnn {

class RotaryEmbed : public Layer
{
public:
    RotaryEmbed();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int interleaved;
    /* for graph-level fusion (default 0, behavior completely unchanged): the
     * input tensor is interpreted as [w, c, h], i.e. the preceding
     * Permute(order_type=2, [w,h,c]->[w,c,h]) is absorbed into this layer.
     * Values and operation order are exactly the same, only the read indices
     * differ => the output is bit-identical. */
    int input_hc_swapped;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_H
