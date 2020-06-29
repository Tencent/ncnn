// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#ifndef LAYER_MIPS_COMMON_H
#define LAYER_MIPS_COMMON_H

#include <msa.h>
#include <stdint.h>

namespace ncnn {

typedef union
{
    int32_t i;
    float f;
} FloatInt;

/* declare some mips constants with union */
#define _MIPS_FLOAT_CONST(Name, Val) \
    static const ncnn::FloatInt Name = {.f = Val}

} // namespace ncnn

/* float type data load instructions */
inline v4f32 __msa_fill_w_f32(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (v4f32)__msa_fill_w(fi_tmpval.i);
}

#endif // LAYER_MIPS_COMMON_H
