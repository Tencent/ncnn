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

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "cast_fp16.h"

void cast_fp32_to_fp16_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp32_to_fp16_sse(bottom_blob, top_blob, opt);
}

void cast_fp16_to_fp32_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp16_to_fp32_sse(bottom_blob, top_blob, opt);
}

} // namespace ncnn
