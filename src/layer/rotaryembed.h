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
    /* 图级融合用（默认 0，行为完全不变）：输入张量按 [w, c, h] 解释，
     * 即把前面那个 Permute(order_type=2, [w,h,c]->[w,c,h]) 吸收进本层。
     * 数值与运算顺序完全相同，只是读取下标不同 ⇒ 输出逐位一致。 */
    int input_hc_swapped;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_H
