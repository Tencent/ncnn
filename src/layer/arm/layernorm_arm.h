// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_LAYERNORM_ARM_H
#define LAYER_LAYERNORM_ARM_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_arm : virtual public LayerNorm
{
public:
    LayerNorm_arm();

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

    virtual int create_pipeline(const Option& opt);

protected:
#if NCNN_ARM82
    int create_pipeline_fp16s(const Option& opt);

    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack1_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack4_fp16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack8_fp16s(Mat& bottom_top_blob, const Option& opt) const;

    int forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack1_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack4_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack8_fp16sa(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int create_pipeline_bf16s(const Option& opt);

    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack1_bf16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_pack4_bf16s(Mat& bottom_top_blob, const Option& opt) const;
    int forward_inplace_naive_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif

public:

    // fp16
    Mat gamma_data_fp16;
    Mat beta_data_fp16;

    // bf16
    Mat gamma_data_bf16;
    Mat beta_data_bf16;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_ARM_H
