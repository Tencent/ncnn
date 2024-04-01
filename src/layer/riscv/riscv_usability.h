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

#ifndef RISCV_USABILITY_H
#define RISCV_USABILITY_H

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#if __riscv_vector
static inline int csrr_vl()
{
    int a = 0;
    asm volatile("csrr %0, vl"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vtype()
{
    int a = 0;
    asm volatile("csrr %0, vtype"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline vfloat32m8_t vle32_v_f32m8_f32m1(const float* ptr)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m8(packn * 8);

    // NOTE vloxei8_v_f32m8 gets illegal instruction on d1  --- nihui

    // 128bit
    static const uint32_t index_128bit[32] = {
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12,
        0, 4, 8, 12
    };

    // 256bit
    static const uint32_t index_256bit[64] = {
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28,
        0, 4, 8, 12, 16, 20, 24, 28
    };

    const uint32_t* index = packn == 4 ? index_128bit : index_256bit;
    vuint32m8_t bindex = vle32_v_u32m8(index, vl);
    return vloxei32_v_f32m8(ptr, bindex, vl);
}

#if __riscv_zfh
static inline vfloat16m8_t vle16_v_f16m8_f16m1(const __fp16* ptr)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m8(packn * 8);

    // NOTE vloxei8_v_f16m8 gets illegal instruction on d1  --- nihui

    // 128bit
    static const uint16_t index_128bit[64] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14
    };

    // 256bit
    static const uint16_t index_256bit[128] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    };

    const uint16_t* index = packn == 8 ? index_128bit : index_256bit;
    vuint16m8_t bindex = vle16_v_u16m8(index, vl);
    return vloxei16_v_f16m8(ptr, bindex, vl);
}
#endif // __riscv_zfh
#endif // __riscv_vector

#if __riscv_vector && __rvv_tuple

// f32m1, vsseg.v
static inline void vsseg8e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e32_v_f32m1x8(base, _tmp, vl);
}

static inline void vsseg4e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = vcreate_f32m1x4(v0, v1, v2, v3);
    vsseg4e32_v_f32m1x4(base, _tmp, vl);
}

static inline void vsseg2e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = vcreate_f32m1x2(v0, v1);
    vsseg2e32_v_f32m1x2(base, _tmp, vl);
}

// f32m1, vssseg.v, 8/4/2
static inline void vssseg8e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vssseg8e32_v_f32m1x8(base, bstride, _tmp, vl);
}

static inline void vssseg4e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = vcreate_f32m1x4(v0, v1, v2, v3);
    vssseg4e32_v_f32m1x4(base, bstride, _tmp, vl);
}

static inline void vssseg2e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = vcreate_f32m1x2(v0, v1);
    vssseg2e32_v_f32m1x2(base, bstride, _tmp, vl);
}

// f32m2, vsseg.v, 4/2
static inline void vsseg4e32_v_f32m2(float32_t* base, vfloat32m2_t v0, vfloat32m2_t v1, vfloat32m2_t v2, vfloat32m2_t v3, size_t vl)
{
    vfloat32m2x4_t _tmp = vcreate_f32m2x4(v0, v1, v2, v3);
    vsseg4e32_v_f32m2x4(base, _tmp, vl);
}

static inline void vsseg2e32_v_f32m2(float32_t* base, vfloat32m2_t v0, vfloat32m2_t v1, size_t vl)
{
    vfloat32m2x2_t _tmp = vcreate_f32m2x2(v0, v1);
    vsseg2e32_v_f32m2x2(base, _tmp, vl);
}

// u16m1, vsseg.v, 8/4
static inline void vsseg8e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, vuint16m1_t v4, vuint16m1_t v5, vuint16m1_t v6, vuint16m1_t v7, size_t vl)
{
    vuint16m1x8_t _tmp = vcreate_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e16_v_u16m1x8(base, _tmp, vl);
}

static inline void vsseg4e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, size_t vl)
{
    vuint16m1x4_t _tmp = vcreate_u16m1x4(v0, v1, v2, v3);
    vsseg4e16_v_u16m1x4(base, _tmp, vl);
}

// u16m2, vsseg.v, 4/2
static inline void vsseg4e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, vuint16m2_t v2, vuint16m2_t v3, size_t vl)
{
    vuint16m2x4_t _tmp = vcreate_u16m2x4(v0, v1, v2, v3);
    vsseg4e16_v_u16m2x4(base, _tmp, vl);
}

static inline void vsseg2e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, size_t vl)
{
    vuint16m2x2_t _tmp = vcreate_u16m2x2(v0, v1);
    vsseg2e16_v_u16m2x2(base, _tmp, vl);
}

// f32m1, vlseg.v 8/4/2
static inline void vlseg8e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, vfloat32m1_t* v4, vfloat32m1_t* v5, vfloat32m1_t* v6, vfloat32m1_t* v7, const float32_t* base, size_t vl)
{
    vfloat32m1x8_t _tmp = vlseg8e32_v_f32m1x8(base, vl);
    *v0 = vget_f32m1x8_f32m1(_tmp, 0);
    *v1 = vget_f32m1x8_f32m1(_tmp, 1);
    *v2 = vget_f32m1x8_f32m1(_tmp, 2);
    *v3 = vget_f32m1x8_f32m1(_tmp, 3);
    *v4 = vget_f32m1x8_f32m1(_tmp, 4);
    *v5 = vget_f32m1x8_f32m1(_tmp, 5);
    *v6 = vget_f32m1x8_f32m1(_tmp, 6);
    *v7 = vget_f32m1x8_f32m1(_tmp, 7);
}

static inline void vlseg4e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, const float32_t* base, size_t vl)
{
    vfloat32m1x4_t _tmp = vlseg4e32_v_f32m1x4(base, vl);
    *v0 = vget_f32m1x4_f32m1(_tmp, 0);
    *v1 = vget_f32m1x4_f32m1(_tmp, 1);
    *v2 = vget_f32m1x4_f32m1(_tmp, 2);
    *v3 = vget_f32m1x4_f32m1(_tmp, 3);
}

static inline void vlseg2e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, const float32_t* base, size_t vl)
{
    vfloat32m1x2_t _tmp = vlseg2e32_v_f32m1x2(base, vl);
    *v0 = vget_f32m1x2_f32m1(_tmp, 0);
    *v1 = vget_f32m1x2_f32m1(_tmp, 1);
}

// f32m2, vlseg.v, 4
static inline void vlseg4e32_v_f32m2(vfloat32m2_t* v0, vfloat32m2_t* v1, vfloat32m2_t* v2, vfloat32m2_t* v3, const float32_t* base, size_t vl)
{
    vfloat32m2x4_t _tmp = vlseg4e32_v_f32m2x4(base, vl);
    *v0 = vget_f32m2x4_f32m2(_tmp, 0);
    *v1 = vget_f32m2x4_f32m2(_tmp, 1);
    *v2 = vget_f32m2x4_f32m2(_tmp, 2);
    *v3 = vget_f32m2x4_f32m2(_tmp, 3);
}

// f32m4, vlseg.v, 2
static inline void vlseg2e32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float32_t* base, size_t vl)
{
    vfloat32m4x2_t _tmp = vlseg2e32_v_f32m4x2(base, vl);
    *v0 = vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = vget_f32m4x2_f32m4(_tmp, 1);
}

// f32m4, vloxseg.v
static inline void vloxseg2ei32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float32_t* base, vuint32m4_t bindex, size_t vl)
{
    vfloat32m4x2_t _tmp = vloxseg2ei32_v_f32m4x2(base, bindex, vl);
    *v0 = vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = vget_f32m4x2_f32m4(_tmp, 1);
}

// u16m1, vlseg.v 8/4/2
static inline void vlseg8e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, vuint16m1_t* v4, vuint16m1_t* v5, vuint16m1_t* v6, vuint16m1_t* v7, const uint16_t* base, size_t vl)
{
    vuint16m1x8_t _tmp = vlseg8e16_v_u16m1x8(base, vl);
    *v0 = vget_u16m1x8_u16m1(_tmp, 0);
    *v1 = vget_u16m1x8_u16m1(_tmp, 1);
    *v2 = vget_u16m1x8_u16m1(_tmp, 2);
    *v3 = vget_u16m1x8_u16m1(_tmp, 3);
    *v4 = vget_u16m1x8_u16m1(_tmp, 4);
    *v5 = vget_u16m1x8_u16m1(_tmp, 5);
    *v6 = vget_u16m1x8_u16m1(_tmp, 6);
    *v7 = vget_u16m1x8_u16m1(_tmp, 7);
}

static inline void vlseg4e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m1x4_t _tmp = vlseg4e16_v_u16m1x4(base, vl);
    *v0 = vget_u16m1x4_u16m1(_tmp, 0);
    *v1 = vget_u16m1x4_u16m1(_tmp, 1);
    *v2 = vget_u16m1x4_u16m1(_tmp, 2);
    *v3 = vget_u16m1x4_u16m1(_tmp, 3);
}

static inline void vlseg2e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m1x2_t _tmp = vlseg2e16_v_u16m1x2(base, vl);
    *v0 = vget_u16m1x2_u16m1(_tmp, 0);
    *v1 = vget_u16m1x2_u16m1(_tmp, 1);
}

// u16m2, vlseg.v, 4
static inline void vlseg4e16_v_u16m2(vuint16m2_t* v0, vuint16m2_t* v1, vuint16m2_t* v2, vuint16m2_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m2x4_t _tmp = vlseg4e16_v_u16m2x4(base, vl);
    *v0 = vget_u16m2x4_u16m2(_tmp, 0);
    *v1 = vget_u16m2x4_u16m2(_tmp, 1);
    *v2 = vget_u16m2x4_u16m2(_tmp, 2);
    *v3 = vget_u16m2x4_u16m2(_tmp, 3);
}

// u16m4, vlseg.v, 2
static inline void vlseg2e16_v_u16m4(vuint16m4_t* v0, vuint16m4_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m4x2_t _tmp = vlseg2e16_v_u16m4x2(base, vl);
    *v0 = vget_u16m4x2_u16m4(_tmp, 0);
    *v1 = vget_u16m4x2_u16m4(_tmp, 1);
}

#if __riscv_zfh

// f16m1, vsseg.v, 8/4/2
static inline void vsseg8e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e16_v_f16m1x8(base, _tmp, vl);
}

static inline void vsseg4e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = vcreate_f16m1x4(v0, v1, v2, v3);
    vsseg4e16_v_f16m1x4(base, _tmp, vl);
}

static inline void vsseg2e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = vcreate_f16m1x2(v0, v1);
    vsseg2e16_v_f16m1x2(base, _tmp, vl);
}

// f16m1, vssseg.v, 8/4/2
static inline void vssseg8e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vssseg8e16_v_f16m1x8(base, bstride, _tmp, vl);
}

static inline void vssseg4e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = vcreate_f16m1x4(v0, v1, v2, v3);
    vssseg4e16_v_f16m1x4(base, bstride, _tmp, vl);
}

static inline void vssseg2e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = vcreate_f16m1x2(v0, v1);
    vssseg2e16_v_f16m1x2(base, bstride, _tmp, vl);
}

// f16m1, vlseg.v 8/4/2
static inline void vlseg8e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, vfloat16m1_t* v4, vfloat16m1_t* v5, vfloat16m1_t* v6, vfloat16m1_t* v7, const float16_t* base, size_t vl)
{
    vfloat16m1x8_t _tmp = vlseg8e16_v_f16m1x8(base, vl);
    *v0 = vget_f16m1x8_f16m1(_tmp, 0);
    *v1 = vget_f16m1x8_f16m1(_tmp, 1);
    *v2 = vget_f16m1x8_f16m1(_tmp, 2);
    *v3 = vget_f16m1x8_f16m1(_tmp, 3);
    *v4 = vget_f16m1x8_f16m1(_tmp, 4);
    *v5 = vget_f16m1x8_f16m1(_tmp, 5);
    *v6 = vget_f16m1x8_f16m1(_tmp, 6);
    *v7 = vget_f16m1x8_f16m1(_tmp, 7);
}

static inline void vlseg4e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m1x4_t _tmp = vlseg4e16_v_f16m1x4(base, vl);
    *v0 = vget_f16m1x4_f16m1(_tmp, 0);
    *v1 = vget_f16m1x4_f16m1(_tmp, 1);
    *v2 = vget_f16m1x4_f16m1(_tmp, 2);
    *v3 = vget_f16m1x4_f16m1(_tmp, 3);
}

static inline void vlseg2e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m1x2_t _tmp = vlseg2e16_v_f16m1x2(base, vl);
    *v0 = vget_f16m1x2_f16m1(_tmp, 0);
    *v1 = vget_f16m1x2_f16m1(_tmp, 1);
}

// f16m2, vlseg.v, 4
static inline void vlseg4e16_v_f16m2(vfloat16m2_t* v0, vfloat16m2_t* v1, vfloat16m2_t* v2, vfloat16m2_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m2x4_t _tmp = vlseg4e16_v_f16m2x4(base, vl);
    *v0 = vget_f16m2x4_f16m2(_tmp, 0);
    *v1 = vget_f16m2x4_f16m2(_tmp, 1);
    *v2 = vget_f16m2x4_f16m2(_tmp, 2);
    *v3 = vget_f16m2x4_f16m2(_tmp, 3);
}

// f16m4, vlseg.v, 2
static inline void vlseg2e16_v_f16m4(vfloat16m4_t* v0, vfloat16m4_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m4x2_t _tmp = vlseg2e16_v_f16m4x2(base, vl);
    *v0 = vget_f16m4x2_f16m4(_tmp, 0);
    *v1 = vget_f16m4x2_f16m4(_tmp, 1);
}

#endif // __riscv_zfh
#endif // __riscv_vector

#ifdef __riscv_vector

static inline void transpose8x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                   vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                   vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                   vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                   vfloat32m1_t& _r7l, vfloat32m1_t& _r7h, size_t vl)
{
    float tmp[64];
    vsseg8e32_v_f32m1(&tmp[0], _r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l, vl);
    vsseg8e32_v_f32m1(&tmp[32], _r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 15 * 4, vl);
}

static inline void transpose4x4_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, size_t vl)
{
    float tmp[16];
    vsseg4e32_v_f32m1(&tmp[0], _r0, _r1, _r2, _r3, vl);
    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
}

static inline void transpose8x12_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                    vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                    vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                    vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                    vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                    vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                    vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                    vfloat32m1_t& _r7l, vfloat32m1_t& _r7h,
                                    vfloat32m1_t& _r8l, vfloat32m1_t& _r8h,
                                    vfloat32m1_t& _r9l, vfloat32m1_t& _r9h,
                                    vfloat32m1_t& _ral, vfloat32m1_t& _rah,
                                    vfloat32m1_t& _rbl, vfloat32m1_t& _rbh, size_t vl)
{
    float tmp[8][12];

    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0l, vl);
    vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 12, _r0h, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1l, vl);
    vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 12, _r1h, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2l, vl);
    vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 12, _r2h, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3l, vl);
    vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 12, _r3h, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4l, vl);
    vsse32_v_f32m1(&tmp[4][4], sizeof(float) * 12, _r4h, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5l, vl);
    vsse32_v_f32m1(&tmp[4][5], sizeof(float) * 12, _r5h, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6l, vl);
    vsse32_v_f32m1(&tmp[4][6], sizeof(float) * 12, _r6h, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7l, vl);
    vsse32_v_f32m1(&tmp[4][7], sizeof(float) * 12, _r7h, vl);
    vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8l, vl);
    vsse32_v_f32m1(&tmp[4][8], sizeof(float) * 12, _r8h, vl);
    vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9l, vl);
    vsse32_v_f32m1(&tmp[4][9], sizeof(float) * 12, _r9h, vl);
    vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ral, vl);
    vsse32_v_f32m1(&tmp[4][10], sizeof(float) * 12, _rah, vl);
    vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rbl, vl);
    vsse32_v_f32m1(&tmp[4][11], sizeof(float) * 12, _rbh, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 15 * 4, vl);
    _r8l = vle32_v_f32m1(ptr + 16 * 4, vl);
    _r8h = vle32_v_f32m1(ptr + 17 * 4, vl);
    _r9l = vle32_v_f32m1(ptr + 18 * 4, vl);
    _r9h = vle32_v_f32m1(ptr + 19 * 4, vl);
    _ral = vle32_v_f32m1(ptr + 20 * 4, vl);
    _rah = vle32_v_f32m1(ptr + 21 * 4, vl);
    _rbl = vle32_v_f32m1(ptr + 22 * 4, vl);
    _rbh = vle32_v_f32m1(ptr + 23 * 4, vl);
}

static inline void transpose12x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0m, vfloat32m1_t& _r0h,
                                    vfloat32m1_t& _r1l, vfloat32m1_t& _r1m, vfloat32m1_t& _r1h,
                                    vfloat32m1_t& _r2l, vfloat32m1_t& _r2m, vfloat32m1_t& _r2h,
                                    vfloat32m1_t& _r3l, vfloat32m1_t& _r3m, vfloat32m1_t& _r3h,
                                    vfloat32m1_t& _r4l, vfloat32m1_t& _r4m, vfloat32m1_t& _r4h,
                                    vfloat32m1_t& _r5l, vfloat32m1_t& _r5m, vfloat32m1_t& _r5h,
                                    vfloat32m1_t& _r6l, vfloat32m1_t& _r6m, vfloat32m1_t& _r6h,
                                    vfloat32m1_t& _r7l, vfloat32m1_t& _r7m, vfloat32m1_t& _r7h, size_t vl)
{
    float tmp[96];
    vsseg8e32_v_f32m1(&tmp[0], _r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l, vl);
    vsseg8e32_v_f32m1(&tmp[32], _r0m, _r1m, _r2m, _r3m, _r4m, _r5m, _r6m, _r7m, vl);
    vsseg8e32_v_f32m1(&tmp[64], _r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h, vl);

    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0m = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r1m = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r2m = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r3m = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r4m = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 15 * 4, vl);
    _r5m = vle32_v_f32m1(ptr + 16 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 17 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 18 * 4, vl);
    _r6m = vle32_v_f32m1(ptr + 19 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 20 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 21 * 4, vl);
    _r7m = vle32_v_f32m1(ptr + 22 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 23 * 4, vl);
}

static inline void transpose4x8_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, size_t vl)
{
    float tmp[32];
    vsseg8e32_v_f32m1(&tmp[0], _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = vle32_v_f32m1(ptr + 7 * 4, vl);
}

static inline void transpose4x12_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, vfloat32m1_t& _r8, vfloat32m1_t& _r9, vfloat32m1_t& _ra, vfloat32m1_t& _rb, size_t vl)
{
    float tmp[4][12];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7, vl);
    vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8, vl);
    vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9, vl);
    vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ra, vl);
    vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rb, vl);
    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r8 = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r9 = vle32_v_f32m1(ptr + 9 * 4, vl);
    _ra = vle32_v_f32m1(ptr + 10 * 4, vl);
    _rb = vle32_v_f32m1(ptr + 11 * 4, vl);
}

static inline void transpose8x4_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h, size_t vl)
{
    float tmp[32];
    vsseg4e32_v_f32m1(&tmp[0], _r0l, _r1l, _r2l, _r3l, vl);
    vsseg4e32_v_f32m1(&tmp[16], _r0h, _r1h, _r2h, _r3h, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
}
#endif

#endif // RISCV_USABILITY_H
