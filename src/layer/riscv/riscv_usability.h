// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
    vfloat32m1x8_t _rl = __riscv_vcreate_v_f32m1x8(_r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l);
    vfloat32m1x8_t _rh = __riscv_vcreate_v_f32m1x8(_r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h);
    __riscv_vsseg8e32_v_f32m1x8(&tmp[0], _rl, vl);
    __riscv_vsseg8e32_v_f32m1x8(&tmp[32], _rh, vl);
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
    vfloat32m1x4_t _r = __riscv_vcreate_v_f32m1x4(_r0, _r1, _r2, _r3);
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
    vfloat32m1x8_t _rl = __riscv_vcreate_v_f32m1x8(_r0l, _r1l, _r2l, _r3l, _r4l, _r5l, _r6l, _r7l);
    vfloat32m1x8_t _rm = __riscv_vcreate_v_f32m1x8(_r0m, _r1m, _r2m, _r3m, _r4m, _r5m, _r6m, _r7m);
    vfloat32m1x8_t _rh = __riscv_vcreate_v_f32m1x8(_r0h, _r1h, _r2h, _r3h, _r4h, _r5h, _r6h, _r7h);
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
    vfloat32m1x8_t _r = __riscv_vcreate_v_f32m1x8(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
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
    vfloat32m1x4_t _rl = __riscv_vcreate_v_f32m1x4(_r0l, _r1l, _r2l, _r3l);
    vfloat32m1x4_t _rh = __riscv_vcreate_v_f32m1x4(_r0h, _r1h, _r2h, _r3h);
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
