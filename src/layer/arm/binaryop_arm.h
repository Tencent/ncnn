// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_BINARYOP_ARM_H
#define LAYER_BINARYOP_ARM_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_arm : virtual public BinaryOp
{
public:
    BinaryOp_arm();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

protected:
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_BINARYOP_ARM_H
