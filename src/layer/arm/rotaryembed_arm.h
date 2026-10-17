// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_ARM_H
#define LAYER_ROTARYEMBED_ARM_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_arm : public RotaryEmbed
{
public:
    RotaryEmbed_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ARM82
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    /* 融合布局（input_hc_swapped）的 fp16 算术版：委托给通用实现会升级成
     * fp32 计算，改变请求的算术模式与输出位；此版本保持 fp16sa 语义。 */
    int forward_fp16sa_swapped(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_ARM_H
