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

#ifndef LAYER_PADDING_ARM_H
#define LAYER_PADDING_ARM_H

#include "padding.h"

namespace ncnn {

class Padding_arm : virtual public Padding
{
public:
    Padding_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    int forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    // bf16
    unsigned short value_bf16;
    Mat per_channel_pad_data_bf16;

    // fp16
    Mat per_channel_pad_data_fp16;
};

} // namespace ncnn

#endif // LAYER_PADDING_ARM_H
