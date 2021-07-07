// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#ifndef MIPS_USABILITY_H
#define MIPS_USABILITY_H

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include <stdint.h>

namespace ncnn {

typedef union
{
    int32_t i;
    float f;
} FloatInt;

} // namespace ncnn

#if __mips_msa
/* declare some mips constants with union */
#define _MIPS_FLOAT_CONST(Name, Val) \
    static const ncnn::FloatInt Name = {.f = Val}

/* float type data load instructions */
static inline v4f32 __msa_fill_w_f32(float val)
{
    ncnn::FloatInt fi_tmpval = {.f = val};
    return (v4f32)__msa_fill_w(fi_tmpval.i);
}

static inline float __msa_fhadd_w(v4f32 _v)
{
    // TODO find a more efficient way
    return _v[0] + _v[1] + _v[2] + _v[3];
}

static inline int __msa_cfcmsa_msacsr()
{
    int v;
    asm volatile("cfcmsa %0, $1 \n"
                 : "=r"(v)
                 :
                 :);
    return v;
}

static inline void __msa_ctcmsa_msacsr(int v)
{
    asm volatile("ctcmsa $1, %0 \n"
                 :
                 : "r"(v)
                 :);
}
#endif // __mips_msa

#endif // MIPS_USABILITY_H
