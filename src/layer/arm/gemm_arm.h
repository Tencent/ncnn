// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_GEMM_ARM_H
#define LAYER_GEMM_ARM_H

#include "gemm.h"

namespace ncnn {

class Gemm_arm : public Gemm
{
public:
    Gemm_arm();

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_VFPV4
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#if NCNN_ARM82
    int create_pipeline_fp16sa(const Option& opt);
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#endif
#if NCNN_BF16
    int create_pipeline_bf16s(const Option& opt);
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;
};

} // namespace ncnn

#endif // LAYER_GEMM_ARM_H
