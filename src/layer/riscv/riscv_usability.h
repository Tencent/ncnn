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

static inline vfloat32m8_t __riscv_vle32_v_f32m8_f32m1(const float* ptr)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m8(packn * 8);

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
    vuint32m8_t bindex = __riscv_vle32_v_u32m8(index, vl);
    return __riscv_vloxei32_v_f32m8(ptr, bindex, vl);
}

#if __riscv_zvfh
static inline vfloat16m8_t __riscv_vle16_v_f16m8_f16m1(const __fp16* ptr)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m8(packn * 8);

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
    vuint16m8_t bindex = __riscv_vle16_v_u16m8(index, vl);
    return __riscv_vloxei16_v_f16m8(ptr, bindex, vl);
}
#endif // __riscv_zvfh
#endif // __riscv_vector

#if 0 //__riscv_vector && __rvv_tuple

// f32m1, vsseg.v
static inline void __riscv_vsseg8e32_v_f32m1(float* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = __riscv_vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vsseg8e32_v_f32m1x8(base, _tmp, vl);
}

static inline void __riscv_vsseg4e32_v_f32m1(float* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = __riscv_vcreate_f32m1x4(v0, v1, v2, v3);
    __riscv_vsseg4e32_v_f32m1x4(base, _tmp, vl);
}

static inline void __riscv_vsseg2e32_v_f32m1(float* base, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = __riscv_vcreate_f32m1x2(v0, v1);
    __riscv_vsseg2e32_v_f32m1x2(base, _tmp, vl);
}

// f32m1, vssseg.v, 8/4/2
static inline void __riscv_vssseg8e32_v_f32m1(float* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = __riscv_vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vssseg8e32_v_f32m1x8(base, bstride, _tmp, vl);
}

static inline void __riscv_vssseg4e32_v_f32m1(float* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = __riscv_vcreate_f32m1x4(v0, v1, v2, v3);
    __riscv_vssseg4e32_v_f32m1x4(base, bstride, _tmp, vl);
}

static inline void __riscv_vssseg2e32_v_f32m1(float* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = __riscv_vcreate_f32m1x2(v0, v1);
    __riscv_vssseg2e32_v_f32m1x2(base, bstride, _tmp, vl);
}

// f32m2, vsseg.v, 4/2
static inline void __riscv_vsseg4e32_v_f32m2(float* base, vfloat32m2_t v0, vfloat32m2_t v1, vfloat32m2_t v2, vfloat32m2_t v3, size_t vl)
{
    vfloat32m2x4_t _tmp = __riscv_vcreate_f32m2x4(v0, v1, v2, v3);
    __riscv_vsseg4e32_v_f32m2x4(base, _tmp, vl);
}

static inline void __riscv_vsseg2e32_v_f32m2(float* base, vfloat32m2_t v0, vfloat32m2_t v1, size_t vl)
{
    vfloat32m2x2_t _tmp = __riscv_vcreate_f32m2x2(v0, v1);
    __riscv_vsseg2e32_v_f32m2x2(base, _tmp, vl);
}

// u16m1, vsseg.v, 8/4
static inline void __riscv_vsseg8e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, vuint16m1_t v4, vuint16m1_t v5, vuint16m1_t v6, vuint16m1_t v7, size_t vl)
{
    vuint16m1x8_t _tmp = __riscv_vcreate_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vsseg8e16_v_u16m1x8(base, _tmp, vl);
}

static inline void __riscv_vsseg4e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, size_t vl)
{
    vuint16m1x4_t _tmp = __riscv_vcreate_u16m1x4(v0, v1, v2, v3);
    __riscv_vsseg4e16_v_u16m1x4(base, _tmp, vl);
}

// u16m2, vsseg.v, 4/2
static inline void __riscv_vsseg4e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, vuint16m2_t v2, vuint16m2_t v3, size_t vl)
{
    vuint16m2x4_t _tmp = __riscv_vcreate_u16m2x4(v0, v1, v2, v3);
    __riscv_vsseg4e16_v_u16m2x4(base, _tmp, vl);
}

static inline void __riscv_vsseg2e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, size_t vl)
{
    vuint16m2x2_t _tmp = __riscv_vcreate_u16m2x2(v0, v1);
    __riscv_vsseg2e16_v_u16m2x2(base, _tmp, vl);
}

// f32m1, vlseg.v 8/4/2
static inline void __riscv_vlseg8e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, vfloat32m1_t* v4, vfloat32m1_t* v5, vfloat32m1_t* v6, vfloat32m1_t* v7, const float* base, size_t vl)
{
    vfloat32m1x8_t _tmp = __riscv_vlseg8e32_v_f32m1x8(base, vl);
    *v0 = __riscv_vget_f32m1x8_f32m1(_tmp, 0);
    *v1 = __riscv_vget_f32m1x8_f32m1(_tmp, 1);
    *v2 = __riscv_vget_f32m1x8_f32m1(_tmp, 2);
    *v3 = __riscv_vget_f32m1x8_f32m1(_tmp, 3);
    *v4 = __riscv_vget_f32m1x8_f32m1(_tmp, 4);
    *v5 = __riscv_vget_f32m1x8_f32m1(_tmp, 5);
    *v6 = __riscv_vget_f32m1x8_f32m1(_tmp, 6);
    *v7 = __riscv_vget_f32m1x8_f32m1(_tmp, 7);
}

static inline void __riscv_vlseg4e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, const float* base, size_t vl)
{
    vfloat32m1x4_t _tmp = __riscv_vlseg4e32_v_f32m1x4(base, vl);
    *v0 = __riscv_vget_f32m1x4_f32m1(_tmp, 0);
    *v1 = __riscv_vget_f32m1x4_f32m1(_tmp, 1);
    *v2 = __riscv_vget_f32m1x4_f32m1(_tmp, 2);
    *v3 = __riscv_vget_f32m1x4_f32m1(_tmp, 3);
}

static inline void __riscv_vlseg2e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, const float* base, size_t vl)
{
    vfloat32m1x2_t _tmp = __riscv_vlseg2e32_v_f32m1x2(base, vl);
    *v0 = __riscv_vget_f32m1x2_f32m1(_tmp, 0);
    *v1 = __riscv_vget_f32m1x2_f32m1(_tmp, 1);
}

// f32m2, vlseg.v, 4
static inline void __riscv_vlseg4e32_v_f32m2(vfloat32m2_t* v0, vfloat32m2_t* v1, vfloat32m2_t* v2, vfloat32m2_t* v3, const float* base, size_t vl)
{
    vfloat32m2x4_t _tmp = __riscv_vlseg4e32_v_f32m2x4(base, vl);
    *v0 = __riscv_vget_f32m2x4_f32m2(_tmp, 0);
    *v1 = __riscv_vget_f32m2x4_f32m2(_tmp, 1);
    *v2 = __riscv_vget_f32m2x4_f32m2(_tmp, 2);
    *v3 = __riscv_vget_f32m2x4_f32m2(_tmp, 3);
}

// f32m4, vlseg.v, 2
static inline void __riscv_vlseg2e32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float* base, size_t vl)
{
    vfloat32m4x2_t _tmp = __riscv_vlseg2e32_v_f32m4x2(base, vl);
    *v0 = __riscv_vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = __riscv_vget_f32m4x2_f32m4(_tmp, 1);
}

// f32m4, vloxseg.v
static inline void __riscv_vloxseg2ei32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float* base, vuint32m4_t bindex, size_t vl)
{
    vfloat32m4x2_t _tmp = __riscv_vloxseg2ei32_v_f32m4x2(base, bindex, vl);
    *v0 = __riscv_vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = __riscv_vget_f32m4x2_f32m4(_tmp, 1);
}

// u16m1, vlseg.v 8/4/2
static inline void __riscv_vlseg8e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, vuint16m1_t* v4, vuint16m1_t* v5, vuint16m1_t* v6, vuint16m1_t* v7, const uint16_t* base, size_t vl)
{
    vuint16m1x8_t _tmp = __riscv_vlseg8e16_v_u16m1x8(base, vl);
    *v0 = __riscv_vget_u16m1x8_u16m1(_tmp, 0);
    *v1 = __riscv_vget_u16m1x8_u16m1(_tmp, 1);
    *v2 = __riscv_vget_u16m1x8_u16m1(_tmp, 2);
    *v3 = __riscv_vget_u16m1x8_u16m1(_tmp, 3);
    *v4 = __riscv_vget_u16m1x8_u16m1(_tmp, 4);
    *v5 = __riscv_vget_u16m1x8_u16m1(_tmp, 5);
    *v6 = __riscv_vget_u16m1x8_u16m1(_tmp, 6);
    *v7 = __riscv_vget_u16m1x8_u16m1(_tmp, 7);
}

static inline void __riscv_vlseg4e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m1x4_t _tmp = __riscv_vlseg4e16_v_u16m1x4(base, vl);
    *v0 = __riscv_vget_u16m1x4_u16m1(_tmp, 0);
    *v1 = __riscv_vget_u16m1x4_u16m1(_tmp, 1);
    *v2 = __riscv_vget_u16m1x4_u16m1(_tmp, 2);
    *v3 = __riscv_vget_u16m1x4_u16m1(_tmp, 3);
}

static inline void __riscv_vlseg2e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m1x2_t _tmp = __riscv_vlseg2e16_v_u16m1x2(base, vl);
    *v0 = __riscv_vget_u16m1x2_u16m1(_tmp, 0);
    *v1 = __riscv_vget_u16m1x2_u16m1(_tmp, 1);
}

// u16m2, vlseg.v, 4
static inline void __riscv_vlseg4e16_v_u16m2(vuint16m2_t* v0, vuint16m2_t* v1, vuint16m2_t* v2, vuint16m2_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m2x4_t _tmp = __riscv_vlseg4e16_v_u16m2x4(base, vl);
    *v0 = __riscv_vget_u16m2x4_u16m2(_tmp, 0);
    *v1 = __riscv_vget_u16m2x4_u16m2(_tmp, 1);
    *v2 = __riscv_vget_u16m2x4_u16m2(_tmp, 2);
    *v3 = __riscv_vget_u16m2x4_u16m2(_tmp, 3);
}

// u16m4, vlseg.v, 2
static inline void __riscv_vlseg2e16_v_u16m4(vuint16m4_t* v0, vuint16m4_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m4x2_t _tmp = __riscv_vlseg2e16_v_u16m4x2(base, vl);
    *v0 = __riscv_vget_u16m4x2_u16m4(_tmp, 0);
    *v1 = __riscv_vget_u16m4x2_u16m4(_tmp, 1);
}

#if __riscv_zvfh

// f16m1, vsseg.v, 8/4/2
static inline void __riscv_vsseg8e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = __riscv_vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vsseg8e16_v_f16m1x8(base, _tmp, vl);
}

static inline void __riscv_vsseg4e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = __riscv_vcreate_f16m1x4(v0, v1, v2, v3);
    __riscv_vsseg4e16_v_f16m1x4(base, _tmp, vl);
}

static inline void __riscv_vsseg2e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = __riscv_vcreate_f16m1x2(v0, v1);
    __riscv_vsseg2e16_v_f16m1x2(base, _tmp, vl);
}

// f16m1, vssseg.v, 8/4/2
static inline void __riscv_vssseg8e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = __riscv_vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vssseg8e16_v_f16m1x8(base, bstride, _tmp, vl);
}

static inline void __riscv_vssseg4e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = __riscv_vcreate_f16m1x4(v0, v1, v2, v3);
    __riscv_vssseg4e16_v_f16m1x4(base, bstride, _tmp, vl);
}

static inline void __riscv_vssseg2e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = __riscv_vcreate_f16m1x2(v0, v1);
    __riscv_vssseg2e16_v_f16m1x2(base, bstride, _tmp, vl);
}

// f16m1, vlseg.v 8/4/2
static inline void __riscv_vlseg8e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, vfloat16m1_t* v4, vfloat16m1_t* v5, vfloat16m1_t* v6, vfloat16m1_t* v7, const float16_t* base, size_t vl)
{
    vfloat16m1x8_t _tmp = __riscv_vlseg8e16_v_f16m1x8(base, vl);
    *v0 = __riscv_vget_f16m1x8_f16m1(_tmp, 0);
    *v1 = __riscv_vget_f16m1x8_f16m1(_tmp, 1);
    *v2 = __riscv_vget_f16m1x8_f16m1(_tmp, 2);
    *v3 = __riscv_vget_f16m1x8_f16m1(_tmp, 3);
    *v4 = __riscv_vget_f16m1x8_f16m1(_tmp, 4);
    *v5 = __riscv_vget_f16m1x8_f16m1(_tmp, 5);
    *v6 = __riscv_vget_f16m1x8_f16m1(_tmp, 6);
    *v7 = __riscv_vget_f16m1x8_f16m1(_tmp, 7);
}

static inline void __riscv_vlseg4e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m1x4_t _tmp = __riscv_vlseg4e16_v_f16m1x4(base, vl);
    *v0 = __riscv_vget_f16m1x4_f16m1(_tmp, 0);
    *v1 = __riscv_vget_f16m1x4_f16m1(_tmp, 1);
    *v2 = __riscv_vget_f16m1x4_f16m1(_tmp, 2);
    *v3 = __riscv_vget_f16m1x4_f16m1(_tmp, 3);
}

static inline void __riscv_vlseg2e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m1x2_t _tmp = __riscv_vlseg2e16_v_f16m1x2(base, vl);
    *v0 = __riscv_vget_f16m1x2_f16m1(_tmp, 0);
    *v1 = __riscv_vget_f16m1x2_f16m1(_tmp, 1);
}

// f16m2, vlseg.v, 4
static inline void __riscv_vlseg4e16_v_f16m2(vfloat16m2_t* v0, vfloat16m2_t* v1, vfloat16m2_t* v2, vfloat16m2_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m2x4_t _tmp = __riscv_vlseg4e16_v_f16m2x4(base, vl);
    *v0 = __riscv_vget_f16m2x4_f16m2(_tmp, 0);
    *v1 = __riscv_vget_f16m2x4_f16m2(_tmp, 1);
    *v2 = __riscv_vget_f16m2x4_f16m2(_tmp, 2);
    *v3 = __riscv_vget_f16m2x4_f16m2(_tmp, 3);
}

// f16m4, vlseg.v, 2
static inline void __riscv_vlseg2e16_v_f16m4(vfloat16m4_t* v0, vfloat16m4_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m4x2_t _tmp = __riscv_vlseg2e16_v_f16m4x2(base, vl);
    *v0 = __riscv_vget_f16m4x2_f16m4(_tmp, 0);
    *v1 = __riscv_vget_f16m4x2_f16m4(_tmp, 1);
}

#endif // __riscv_zvfh
#endif // __riscv_vector

#if __riscv_vector

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
#if __riscv_vector
#if 1 //__riscv_v_intrinsic > 12000
#warning A
    vfloat32m1x8_t _rl = __riscv_vcreate_v_f32m1x8(_r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l);
    vfloat32m1x8_t _rh = __riscv_vcreate_v_f32m1x8(_r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h);
#else
#warning B
    vfloat32m1x8_t _rl = vfloat32m1x8_t();
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 0, _r0l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 1, _r1l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 2, _r2l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 3, _r3l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 4, _r4l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 5, _r5l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 6, _r6l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 7, _r7l);
    vfloat32m1x8_t _rh = vfloat32m1x8_t();
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 0, _r0h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 1, _r1h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 2, _r2h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 3, _r3h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 4, _r4h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 5, _r5h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 6, _r6h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 7, _r7h);
#endif
    __riscv_vsseg8e32_v_f32m1x8(&tmp[0], _rl, vl);
    __riscv_vsseg8e32_v_f32m1x8(&tmp[32], _rh, vl);
#elif __riscv_xtheadvector
    vfloat32m1x8_t _rl = vfloat32m1x8_t();
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 0, _r0l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 1, _r1l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 2, _r2l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 3, _r3l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 4, _r4l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 5, _r5l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 6, _r6l);
    // _rl = __riscv_th_vset_v_f32m1_f32m1x8(_rl, 7, _r7l);
    vfloat32m1x8_t _rh = vfloat32m1x8_t();
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 0, _r0h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 1, _r1h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 2, _r2h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 3, _r3h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 4, _r4h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 5, _r5h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 6, _r6h);
    // _rh = __riscv_th_vset_v_f32m1_f32m1x8(_rh, 7, _r7h);
    __riscv_th_vsseg8e32_v_f32m1x8(&tmp[0], _rl, vl);
    __riscv_th_vsseg8e32_v_f32m1x8(&tmp[32], _rh, vl);
#endif

    float* ptr = (float*)tmp;
    _r0l = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = __riscv_vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = __riscv_vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = __riscv_vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = __riscv_vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = __riscv_vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = __riscv_vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = __riscv_vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = __riscv_vle32_v_f32m1(ptr + 15 * 4, vl);
}

static inline void transpose4x4_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, size_t vl)
{
    float tmp[16];
#if __riscv_vector && __riscv_v_intrinsic > 12000
    vfloat32m1x4_t _r = __riscv_vcreate_v_f32m1x4(_r0, _r1, _r2, _r3);
#else
    vfloat32m1x4_t _r = vfloat32m1x4_t();
    _r = __riscv_vset_v_f32m1_f32m1x4(_r, 0, _r0);
    _r = __riscv_vset_v_f32m1_f32m1x4(_r, 1, _r1);
    _r = __riscv_vset_v_f32m1_f32m1x4(_r, 2, _r2);
    _r = __riscv_vset_v_f32m1_f32m1x4(_r, 3, _r3);
#endif
    __riscv_vsseg4e32_v_f32m1x4(&tmp[0], _r, vl);
    float* ptr = (float*)tmp;
    _r0 = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
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

    __riscv_vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 12, _r0h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 12, _r1h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 12, _r2h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 12, _r3h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][4], sizeof(float) * 12, _r4h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][5], sizeof(float) * 12, _r5h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][6], sizeof(float) * 12, _r6h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][7], sizeof(float) * 12, _r7h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][8], sizeof(float) * 12, _r8h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9l, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][9], sizeof(float) * 12, _r9h, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ral, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][10], sizeof(float) * 12, _rah, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rbl, vl);
    __riscv_vsse32_v_f32m1(&tmp[4][11], sizeof(float) * 12, _rbh, vl);
    float* ptr = (float*)tmp;
    _r0l = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = __riscv_vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = __riscv_vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = __riscv_vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = __riscv_vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = __riscv_vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = __riscv_vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = __riscv_vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = __riscv_vle32_v_f32m1(ptr + 15 * 4, vl);
    _r8l = __riscv_vle32_v_f32m1(ptr + 16 * 4, vl);
    _r8h = __riscv_vle32_v_f32m1(ptr + 17 * 4, vl);
    _r9l = __riscv_vle32_v_f32m1(ptr + 18 * 4, vl);
    _r9h = __riscv_vle32_v_f32m1(ptr + 19 * 4, vl);
    _ral = __riscv_vle32_v_f32m1(ptr + 20 * 4, vl);
    _rah = __riscv_vle32_v_f32m1(ptr + 21 * 4, vl);
    _rbl = __riscv_vle32_v_f32m1(ptr + 22 * 4, vl);
    _rbh = __riscv_vle32_v_f32m1(ptr + 23 * 4, vl);
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
#if __riscv_vector && __riscv_v_intrinsic > 12000
    vfloat32m1x8_t _rl = __riscv_vcreate_v_f32m1x8(_r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l);
    vfloat32m1x8_t _rh = __riscv_vcreate_v_f32m1x8(_r0m, _r1m, _r2m, _r3m, _r4m, _r5m, _r6m, _r7m);
    vfloat32m1x8_t _rh = __riscv_vcreate_v_f32m1x8(_r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h);
#else
    vfloat32m1x8_t _rl = vfloat32m1x8_t();
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 0, _r0l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 1, _r1l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 2, _r2l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 3, _r3l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 4, _r4l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 5, _r5l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 6, _r6l);
    _rl = __riscv_vset_v_f32m1_f32m1x8(_rl, 7, _r7l);
    vfloat32m1x8_t _rm = vfloat32m1x8_t();
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 0, _r0m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 1, _r1m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 2, _r2m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 3, _r3m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 4, _r4m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 5, _r5m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 6, _r6m);
    _rm = __riscv_vset_v_f32m1_f32m1x8(_rm, 7, _r7m);
    vfloat32m1x8_t _rh = vfloat32m1x8_t();
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 0, _r0h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 1, _r1h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 2, _r2h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 3, _r3h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 4, _r4h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 5, _r5h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 6, _r6h);
    _rh = __riscv_vset_v_f32m1_f32m1x8(_rh, 7, _r7h);
#endif
    __riscv_vsseg8e32_v_f32m1x8(&tmp[0], _rl, vl);
    __riscv_vsseg8e32_v_f32m1x8(&tmp[32], _rm, vl);
    __riscv_vsseg8e32_v_f32m1x8(&tmp[64], _rh, vl);

    float* ptr = (float*)tmp;
    _r0l = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0m = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r0h = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1l = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r1m = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r1h = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r2l = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r2m = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
    _r2h = __riscv_vle32_v_f32m1(ptr + 8 * 4, vl);
    _r3l = __riscv_vle32_v_f32m1(ptr + 9 * 4, vl);
    _r3m = __riscv_vle32_v_f32m1(ptr + 10 * 4, vl);
    _r3h = __riscv_vle32_v_f32m1(ptr + 11 * 4, vl);
    _r4l = __riscv_vle32_v_f32m1(ptr + 12 * 4, vl);
    _r4m = __riscv_vle32_v_f32m1(ptr + 13 * 4, vl);
    _r4h = __riscv_vle32_v_f32m1(ptr + 14 * 4, vl);
    _r5l = __riscv_vle32_v_f32m1(ptr + 15 * 4, vl);
    _r5m = __riscv_vle32_v_f32m1(ptr + 16 * 4, vl);
    _r5h = __riscv_vle32_v_f32m1(ptr + 17 * 4, vl);
    _r6l = __riscv_vle32_v_f32m1(ptr + 18 * 4, vl);
    _r6m = __riscv_vle32_v_f32m1(ptr + 19 * 4, vl);
    _r6h = __riscv_vle32_v_f32m1(ptr + 20 * 4, vl);
    _r7l = __riscv_vle32_v_f32m1(ptr + 21 * 4, vl);
    _r7m = __riscv_vle32_v_f32m1(ptr + 22 * 4, vl);
    _r7h = __riscv_vle32_v_f32m1(ptr + 23 * 4, vl);
}

static inline void transpose4x8_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, size_t vl)
{
    float tmp[32];
#if __riscv_vector && __riscv_v_intrinsic > 12000
    vfloat32m1x8_t _r = __riscv_vcreate_v_f32m1x8(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
#else
    vfloat32m1x8_t _r = vfloat32m1x8_t();
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 0, _r0);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 1, _r1);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 2, _r2);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 3, _r3);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 4, _r4);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 5, _r5);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 6, _r6);
    _r = __riscv_vset_v_f32m1_f32m1x8(_r, 7, _r7);
#endif
    __riscv_vsseg8e32_v_f32m1x8(&tmp[0], _r, vl);

    float* ptr = (float*)tmp;
    _r0 = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
}

static inline void transpose4x12_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, vfloat32m1_t& _r8, vfloat32m1_t& _r9, vfloat32m1_t& _ra, vfloat32m1_t& _rb, size_t vl)
{
    float tmp[4][12];
    __riscv_vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ra, vl);
    __riscv_vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rb, vl);
    float* ptr = (float*)tmp;
    _r0 = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
    _r8 = __riscv_vle32_v_f32m1(ptr + 8 * 4, vl);
    _r9 = __riscv_vle32_v_f32m1(ptr + 9 * 4, vl);
    _ra = __riscv_vle32_v_f32m1(ptr + 10 * 4, vl);
    _rb = __riscv_vle32_v_f32m1(ptr + 11 * 4, vl);
}

static inline void transpose8x4_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h, size_t vl)
{
    float tmp[32];
#if __riscv_vector && __riscv_v_intrinsic > 12000
    vfloat32m1x4_t _rl = __riscv_vcreate_v_f32m1x4(_r0l, _r1l, _r2l, _r3l);
    vfloat32m1x4_t _rh = __riscv_vcreate_v_f32m1x4(_r0h, _r1h, _r2h, _r3h);
#else
    vfloat32m1x4_t _rl = vfloat32m1x4_t();
    _rl = __riscv_vset_v_f32m1_f32m1x4(_rl, 0, _r0l);
    _rl = __riscv_vset_v_f32m1_f32m1x4(_rl, 1, _r1l);
    _rl = __riscv_vset_v_f32m1_f32m1x4(_rl, 2, _r2l);
    _rl = __riscv_vset_v_f32m1_f32m1x4(_rl, 3, _r3l);
    vfloat32m1x4_t _rh = vfloat32m1x4_t();
    _rh = __riscv_vset_v_f32m1_f32m1x4(_rh, 0, _r0h);
    _rh = __riscv_vset_v_f32m1_f32m1x4(_rh, 1, _r1h);
    _rh = __riscv_vset_v_f32m1_f32m1x4(_rh, 2, _r2h);
    _rh = __riscv_vset_v_f32m1_f32m1x4(_rh, 3, _r3h);
#endif
    __riscv_vsseg4e32_v_f32m1x4(&tmp[0], _rl, vl);
    __riscv_vsseg4e32_v_f32m1x4(&tmp[16], _rh, vl);
    float* ptr = (float*)tmp;
    _r0l = __riscv_vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = __riscv_vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = __riscv_vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = __riscv_vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = __riscv_vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = __riscv_vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = __riscv_vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = __riscv_vle32_v_f32m1(ptr + 7 * 4, vl);
}
#endif

#endif // RISCV_USABILITY_H
