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

static inline vfloat32m1_t vfmaq_laneq_f32_riscv(vfloat32m1_t sum, vfloat32m1_t a, vfloat32m1_t b, int lane) {
    float t[4];
    vse32_v_f32m1(t, b, 4);
    vfloat32m1_t ret = vfmadd_vf_f32m1(
            a, t[lane], sum, 4);
    return ret;     
}

static inline vfloat32m1_t vdupq_n_f32_riscv(float32_t f) {
    return vfmv_v_f_f32m1(f, 4);
}

#define VL 4

static inline vfloat32m1x2_t vzip_f32(vfloat32m1_t vector1, vfloat32m1_t vector2) {
    vfloat32m2_t d = vundefined_f32m2();
    d = vset_v_f32m1_f32m2(d, 0, vector1);
    d = vset_v_f32m1_f32m2(d, 1, vector2);
    vuint32m2_t index;
    uint32_t index_128[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    index = vle32_v_u32m2(index_128, 8);
    vfloat32m2_t g_d = vrgather_vv_f32m2(d, index, 8);
    vfloat32m1_t v0 = vget_v_f32m2_f32m1(g_d, 0);
    vfloat32m1_t v1 = vget_v_f32m2_f32m1(g_d, 1);
    return vcreate_f32m1x2(v0, v1);
}

static inline vfloat32m1x4_t vzip_f32_v4(vfloat32m1_t vector1, vfloat32m1_t vector2, vfloat32m1_t vector3, vfloat32m1_t vector4) {
    vfloat32m4_t d = vundefined_f32m4();
    d = vset_v_f32m1_f32m4(d, 0, vector1);
    d = vset_v_f32m1_f32m4(d, 1, vector2);
    d = vset_v_f32m1_f32m4(d, 2, vector3);
    d = vset_v_f32m1_f32m4(d, 3, vector4);
    
    vuint32m4_t index;
    uint32_t index_128[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    index = vle32_v_u32m4(index_128, 16);
    vfloat32m4_t g_d = vrgather_vv_f32m4(d, index, 16);
    vfloat32m1_t v0 = vget_v_f32m4_f32m1(g_d, 0);
    vfloat32m1_t v1 = vget_v_f32m4_f32m1(g_d, 1);
    vfloat32m1_t v2 = vget_v_f32m4_f32m1(g_d, 2);
    vfloat32m1_t v3 = vget_v_f32m4_f32m1(g_d, 3);
    return vcreate_f32m1x4(v0, v1, v2, v3);
}

static inline vfloat32m1_t vget_low_f32(vfloat32m1_t a) {
    return vmv_v_v_f32m1(a, 2);
}

static inline vfloat32m1_t vget_high_f32(vfloat32m1_t a) 
{
    float t[4];
    vse32_v_f32m1(t, a, 4);
    return vle32_v_f32m1(t + 2, 2);
}

static inline vfloat32m1_t vcombine_f32(vfloat32m1_t a, vfloat32m1_t b) {
    float t[4];
    vse32_v_f32m1(t, a, 2);
    vse32_v_f32m1(t + 2, b, 2);
    return vle32_v_f32m1(t, 4);
}

static inline void transpose8x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                   vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                   vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                   vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                   vfloat32m1_t& _r7l, vfloat32m1_t& _r7h)
{
    vfloat32m1x2_t _r01lz = vzip_f32(_r0l, _r1l);
    vfloat32m1x2_t _r23lz = vzip_f32(_r2l, _r3l);
    vfloat32m1x2_t _r01hz = vzip_f32(_r0h, _r1h);
    vfloat32m1x2_t _r23hz = vzip_f32(_r2h, _r3h);
    vfloat32m1x2_t _r45lz = vzip_f32(_r4l, _r5l);
    vfloat32m1x2_t _r67lz = vzip_f32(_r6l, _r7l);
    vfloat32m1x2_t _r45hz = vzip_f32(_r4h, _r5h);
    vfloat32m1x2_t _r67hz = vzip_f32(_r6h, _r7h); 
    _r0l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01lz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r23lz, 0)));
    _r0h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45lz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r67lz, 0)));
    _r1l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01lz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r23lz, 0)));
    _r1h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45lz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r67lz, 0)));
    _r2l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01lz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r23lz, 1)));
    _r2h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45lz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r67lz, 1)));
    _r3l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01lz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r23lz, 1)));
    _r3h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45lz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r67lz, 1)));
    _r4l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01hz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r23hz, 0)));
    _r4h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45hz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r67hz, 0)));
    _r5l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01hz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r23hz, 0)));
    _r5h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45hz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r67hz, 0)));
    _r6l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01hz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r23hz, 1)));
    _r6h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45hz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r67hz, 1)));
    _r7l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01hz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r23hz, 1)));
    _r7h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45hz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r67hz, 1)));
}

static inline void transpose4x4_ps(vfloat32m1_t &_r0, vfloat32m1_t &_r1, vfloat32m1_t &_r2, vfloat32m1_t &_r3)
{
    vfloat32m1x2_t _r01z = vzip_f32(_r0, _r1);
    vfloat32m1x2_t _r23z = vzip_f32(_r2, _r3);
    _r0 = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01z, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r23z, 0)));
    _r1 = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01z, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r23z, 0)));
    _r2 = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01z, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r23z, 1)));
    _r3 = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01z, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r23z, 1)));
}

static inline void store_float32_v2(vfloat32m1_t vector1, vfloat32m1_t vector2, float *buf) 
{
    vfloat32m1x2_t zip = vzip_f32(vector1, vector2);
    vse32_v_f32m1(buf, vget_f32m1x2_f32m1(zip, 0), VL);
    vse32_v_f32m1(buf + 4, vget_f32m1x2_f32m1(zip, 1), VL);
}

static inline void store_float_v4(vfloat32m1_t vector1, vfloat32m1_t vector2, vfloat32m1_t vector3, vfloat32m1_t vector4, float *buf) 
{
    vfloat32m1x4_t zip = vzip_f32_v4(vector1, vector2, vector3, vector4);
    vse32_v_f32m1(buf, vget_f32m1x4_f32m1(zip, 0), VL);
    vse32_v_f32m1(buf + 4, vget_f32m1x4_f32m1(zip, 1), VL);
    vse32_v_f32m1(buf + 8, vget_f32m1x4_f32m1(zip, 2), VL);
    vse32_v_f32m1(buf + 12, vget_f32m1x4_f32m1(zip, 3), VL);
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
