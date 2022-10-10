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

#endif // RISCV_USABILITY_H
