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

#ifndef RISCV_V_071_FIX_H
#define RISCV_V_071_FIX_H

#if __riscv_vector
#include <riscv-vector.h>

typedef unsigned int word_type;

typedef e32xm1_t vbool32_t;
typedef e32xm2_t vbool16_t;
typedef e32xm4_t vbool8_t;
typedef e32xm8_t vbool4_t;

// typedef e16xm1_t vbool8_t;
// typedef e16xm2_t vbool6_t;
// typedef e16xm4_t vbool4_t;
typedef e16xm8_t vbool2_t;

typedef float32xm1_t vfloat32m1_t;
typedef float32xm2_t vfloat32m2_t;
typedef float32xm4_t vfloat32m4_t;
typedef float32xm8_t vfloat32m8_t;

typedef float32x2xm1_t vfloat32m1x2_t;
typedef float32x4xm1_t vfloat32m1x4_t;
typedef float32x8xm1_t vfloat32m1x8_t;
typedef float32x4xm2_t vfloat32m2x4_t;
typedef float32x2xm4_t vfloat32m4x2_t;

typedef float16xm1_t vfloat16m1_t;
typedef float16xm2_t vfloat16m2_t;
typedef float16xm4_t vfloat16m4_t;
typedef float16xm8_t vfloat16m8_t;

typedef float16x2xm1_t vfloat16m1x2_t;
typedef float16x4xm1_t vfloat16m1x4_t;
typedef float16x8xm1_t vfloat16m1x8_t;
typedef float16x4xm2_t vfloat16m2x4_t;

typedef int32xm1_t vint32m1_t;
typedef int32xm2_t vint32m2_t;
typedef int32xm4_t vint32m4_t;
typedef int32xm8_t vint32m8_t;

typedef int32x4xm1_t vint32m1x4_t;
typedef int32x4xm2_t vint32m2x4_t;
typedef int32x8xm1_t vint32m1x8_t;

typedef int16xm1_t vint16m1_t;
typedef int16xm2_t vint16m2_t;
typedef int16xm4_t vint16m4_t;
typedef int16xm8_t vint16m8_t;

typedef int16x4xm1_t vint16m1x4_t;
typedef int16x4xm2_t vint16m2x4_t;
typedef int16x8xm1_t vint16m1x8_t;

typedef int8xm1_t vint8m1_t;
typedef int8xm2_t vint8m2_t;
typedef int8xm4_t vint8m4_t;
typedef int8xm8_t vint8m8_t;

typedef int8x4xm1_t vint8m1x4_t;
typedef int8x4xm2_t vint8m2x4_t;
typedef int8x8xm1_t vint8m1x8_t;

typedef uint32xm1_t vuint32m1_t;
typedef uint32xm2_t vuint32m2_t;
typedef uint32xm4_t vuint32m4_t;
typedef uint32xm8_t vuint32m8_t;

typedef uint32x4xm1_t vuint32m1x4_t;
typedef uint32x4xm2_t vuint32m2x4_t;
typedef uint32x8xm1_t vuint32m1x8_t;

typedef uint16xm1_t vuint16m1_t;
typedef uint16xm2_t vuint16m2_t;
typedef uint16xm4_t vuint16m4_t;
typedef uint16xm8_t vuint16m8_t;

typedef uint16x4xm1_t vuint16m1x4_t;
typedef uint16x4xm2_t vuint16m2x4_t;
typedef uint16x8xm1_t vuint16m1x8_t;

#define vsetvl_e32m1(n) vsetvli(n, RVV_E32, RVV_M1)
#define vsetvl_e32m2(n) vsetvli(n, RVV_E32, RVV_M2)
#define vsetvl_e32m4(n) vsetvli(n, RVV_E32, RVV_M4)
#define vsetvl_e32m8(n) vsetvli(n, RVV_E32, RVV_M8)

#define vsetvl_e16m1(n) vsetvli(n, RVV_E16, RVV_M1)
#define vsetvl_e16m2(n) vsetvli(n, RVV_E16, RVV_M2)
#define vsetvl_e16m4(n) vsetvli(n, RVV_E16, RVV_M4)
#define vsetvl_e16m8(n) vsetvli(n, RVV_E16, RVV_M8)

#define vsetvl_e8m1(n) vsetvli(n, RVV_E8, RVV_M1)
#define vsetvl_e8m2(n) vsetvli(n, RVV_E8, RVV_M2)
#define vsetvl_e8m4(n) vsetvli(n, RVV_E8, RVV_M4)
#define vsetvl_e8m8(n) vsetvli(n, RVV_E8, RVV_M8)

/******************************** float32 ********************************/
#define vle32_v_f32m1 vlev_float32xm1
#define vle32_v_f32m2 vlev_float32xm2
#define vle32_v_f32m4 vlev_float32xm4
#define vle32_v_f32m8 vlev_float32xm8

#define vse32_v_f32m1 vsev_float32xm1
#define vse32_v_f32m2 vsev_float32xm2
#define vse32_v_f32m4 vsev_float32xm4
#define vse32_v_f32m8 vsev_float32xm8

#define vlse32_v_f32m1 vlsev_float32xm1
#define vlse32_v_f32m2 vlsev_float32xm2
#define vlse32_v_f32m4 vlsev_float32xm4
#define vlse32_v_f32m8 vlsev_float32xm8

#define vsse32_v_f32m1 vssev_float32xm1
#define vsse32_v_f32m2 vssev_float32xm2
#define vsse32_v_f32m4 vssev_float32xm4
#define vsse32_v_f32m8 vssev_float32xm8

#define vlseg2e32_v_f32m1x2 vlseg2ev_float32x2xm1
#define vsseg2e32_v_f32m1x2 vsseg2ev_float32x2xm1

#define vlseg4e32_v_f32m1x4 vlseg4ev_float32x4xm1
#define vsseg4e32_v_f32m1x4 vsseg4ev_float32x4xm1

#define vlseg8e32_v_f32m1x8 vlseg8ev_float32x8xm1
#define vsseg8e32_v_f32m1x8 vsseg8ev_float32x8xm1

#define vlseg4e32_v_f32m2x4 vlseg4ev_float32x4xm2
#define vsseg4e32_v_f32m2x4 vsseg4ev_float32x4xm2

#define vlseg2e32_v_f32m4x2 vlseg2ev_float32x2xm4
#define vsseg2e32_v_f32m4x2 vsseg2ev_float32x2xm4

#define vssseg8e32_v_f32m1x8 vssseg8ev_float32x8xm1
#define vssseg4e32_v_f32m1x4 vssseg4ev_float32x4xm1
#define vssseg2e32_v_f32m1x2 vssseg2ev_float32x2xm1

#define vloxseg2ei32_v_f32m4x2(a, i, vl) vlxseg2ev_float32x2xm4(a, reinterpret_cast<int32xm4_t>(i), vl)

#define vset_f32m1x2       vseg_element_set_float32x2xm1
#define vset_f32m1x4       vseg_element_set_float32x4xm1
#define vset_f32m1x8       vseg_element_set_float32x8xm1
#define vset_f32m2x4       vseg_element_set_float32x4xm2
#define vset_f32m4x2       vseg_element_set_float32x2xm4
#define vget_f32m1x2_f32m1 vseg_element_get_float32x2xm1
#define vget_f32m1x4_f32m1 vseg_element_get_float32x4xm1
#define vget_f32m1x8_f32m1 vseg_element_get_float32x8xm1
#define vget_f32m2x4_f32m2 vseg_element_get_float32x4xm2
#define vget_f32m4x2_f32m4 vseg_element_get_float32x2xm4

static inline vfloat32m1x2_t vcreate_f32m1x2(vfloat32m1_t v0, vfloat32m1_t v1)
{
    vfloat32m1x2_t p;
    p = vset_f32m1x2(p, 0, v0);
    p = vset_f32m1x2(p, 1, v1);
    return p;
}

static inline vfloat32m1x4_t vcreate_f32m1x4(vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3)
{
    vfloat32m1x4_t p;
    p = vset_f32m1x4(p, 0, v0);
    p = vset_f32m1x4(p, 1, v1);
    p = vset_f32m1x4(p, 2, v2);
    p = vset_f32m1x4(p, 3, v3);
    return p;
}

static inline vfloat32m2x4_t vcreate_f32m2x4(vfloat32m2_t v0, vfloat32m2_t v1, vfloat32m2_t v2, vfloat32m2_t v3)
{
    vfloat32m2x4_t p;
    p = vset_f32m2x4(p, 0, v0);
    p = vset_f32m2x4(p, 1, v1);
    p = vset_f32m2x4(p, 2, v2);
    p = vset_f32m2x4(p, 3, v3);
    return p;
}

static inline vfloat32m1x8_t vcreate_f32m1x8(vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7)
{
    vfloat32m1x8_t p;
    p = vset_f32m1x8(p, 0, v0);
    p = vset_f32m1x8(p, 1, v1);
    p = vset_f32m1x8(p, 2, v2);
    p = vset_f32m1x8(p, 3, v3);
    p = vset_f32m1x8(p, 4, v4);
    p = vset_f32m1x8(p, 5, v5);
    p = vset_f32m1x8(p, 6, v6);
    p = vset_f32m1x8(p, 7, v7);
    return p;
}

#define vfmv_s_f_f32m1(a, b, vl) vfmvsf_float32xm1(b, vl)
#define vfmv_s_f_f32m2(a, b, vl) vfmvsf_float32xm2(b, vl)
#define vfmv_s_f_f32m4(a, b, vl) vfmvsf_float32xm4(b, vl)
#define vfmv_s_f_f32m8(a, b, vl) vfmvsf_float32xm8(b, vl)

#define vfmv_f_s_f32m1_f32(x) vfmvfs_float32xm1(x, vl)
#define vfmv_f_s_f32m2_f32(x) vfmvfs_float32xm2(x, vl)
#define vfmv_f_s_f32m4_f32(x) vfmvfs_float32xm4(x, vl)
#define vfmv_f_s_f32m8_f32(x) vfmvfs_float32xm8(x, vl)

#define vfmv_v_f_f32m1 vfmvvf_float32xm1
#define vfmv_v_f_f32m2 vfmvvf_float32xm2
#define vfmv_v_f_f32m4 vfmvvf_float32xm4
#define vfmv_v_f_f32m8 vfmvvf_float32xm8

#define vfadd_vv_f32m1                           vfaddvv_float32xm1
#define vfadd_vv_f32m2                           vfaddvv_float32xm2
#define vfadd_vv_f32m4                           vfaddvv_float32xm4
#define vfadd_vv_f32m8                           vfaddvv_float32xm8
#define vfadd_vf_f32m1                           vfaddvf_float32xm1
#define vfadd_vf_f32m2                           vfaddvf_float32xm2
#define vfadd_vf_f32m4                           vfaddvf_float32xm4
#define vfadd_vf_f32m8                           vfaddvf_float32xm8
#define vfadd_vv_f32m1_m(mask, merged, a, b, vl) vfaddvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfadd_vv_f32m2_m(mask, merged, a, b, vl) vfaddvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfadd_vv_f32m4_m(mask, merged, a, b, vl) vfaddvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfadd_vv_f32m8_m(mask, merged, a, b, vl) vfaddvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfadd_vf_f32m1_m(mask, merged, a, b, vl) vfaddvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfadd_vf_f32m2_m(mask, merged, a, b, vl) vfaddvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfadd_vf_f32m4_m(mask, merged, a, b, vl) vfaddvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfadd_vf_f32m8_m(mask, merged, a, b, vl) vfaddvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfsub_vv_f32m1                           vfsubvv_float32xm1
#define vfsub_vv_f32m2                           vfsubvv_float32xm2
#define vfsub_vv_f32m4                           vfsubvv_float32xm4
#define vfsub_vv_f32m8                           vfsubvv_float32xm8
#define vfsub_vf_f32m1                           vfsubvf_float32xm1
#define vfsub_vf_f32m2                           vfsubvf_float32xm2
#define vfsub_vf_f32m4                           vfsubvf_float32xm4
#define vfsub_vf_f32m8                           vfsubvf_float32xm8
#define vfsub_vv_f32m1_m(mask, merged, a, b, vl) vfsubvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfsub_vv_f32m2_m(mask, merged, a, b, vl) vfsubvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfsub_vv_f32m4_m(mask, merged, a, b, vl) vfsubvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfsub_vv_f32m8_m(mask, merged, a, b, vl) vfsubvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfsub_vf_f32m1_m(mask, merged, a, b, vl) vfsubvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfsub_vf_f32m2_m(mask, merged, a, b, vl) vfsubvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfsub_vf_f32m4_m(mask, merged, a, b, vl) vfsubvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfsub_vf_f32m8_m(mask, merged, a, b, vl) vfsubvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfmul_vv_f32m1                           vfmulvv_float32xm1
#define vfmul_vv_f32m2                           vfmulvv_float32xm2
#define vfmul_vv_f32m4                           vfmulvv_float32xm4
#define vfmul_vv_f32m8                           vfmulvv_float32xm8
#define vfmul_vf_f32m1                           vfmulvf_float32xm1
#define vfmul_vf_f32m2                           vfmulvf_float32xm2
#define vfmul_vf_f32m4                           vfmulvf_float32xm4
#define vfmul_vf_f32m8                           vfmulvf_float32xm8
#define vfmul_vv_f32m1_m(mask, merged, a, b, vl) vfmulvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfmul_vv_f32m2_m(mask, merged, a, b, vl) vfmulvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfmul_vv_f32m4_m(mask, merged, a, b, vl) vfmulvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfmul_vv_f32m8_m(mask, merged, a, b, vl) vfmulvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfmul_vf_f32m1_m(mask, merged, a, b, vl) vfmulvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfmul_vf_f32m2_m(mask, merged, a, b, vl) vfmulvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfmul_vf_f32m4_m(mask, merged, a, b, vl) vfmulvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfmul_vf_f32m8_m(mask, merged, a, b, vl) vfmulvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfdiv_vv_f32m1                           vfdivvv_float32xm1
#define vfdiv_vv_f32m2                           vfdivvv_float32xm2
#define vfdiv_vv_f32m4                           vfdivvv_float32xm4
#define vfdiv_vv_f32m8                           vfdivvv_float32xm8
#define vfdiv_vf_f32m1                           vfdivvf_float32xm1
#define vfdiv_vf_f32m2                           vfdivvf_float32xm2
#define vfdiv_vf_f32m4                           vfdivvf_float32xm4
#define vfdiv_vf_f32m8                           vfdivvf_float32xm8
#define vfdiv_vv_f32m1_m(mask, merged, a, b, vl) vfdivvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfdiv_vv_f32m2_m(mask, merged, a, b, vl) vfdivvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfdiv_vv_f32m4_m(mask, merged, a, b, vl) vfdivvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfdiv_vv_f32m8_m(mask, merged, a, b, vl) vfdivvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfdiv_vf_f32m1_m(mask, merged, a, b, vl) vfdivvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfdiv_vf_f32m2_m(mask, merged, a, b, vl) vfdivvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfdiv_vf_f32m4_m(mask, merged, a, b, vl) vfdivvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfdiv_vf_f32m8_m(mask, merged, a, b, vl) vfdivvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfmax_vv_f32m1                           vfmaxvv_float32xm1
#define vfmax_vv_f32m2                           vfmaxvv_float32xm2
#define vfmax_vv_f32m4                           vfmaxvv_float32xm4
#define vfmax_vv_f32m8                           vfmaxvv_float32xm8
#define vfmax_vf_f32m1                           vfmaxvf_float32xm1
#define vfmax_vf_f32m2                           vfmaxvf_float32xm2
#define vfmax_vf_f32m4                           vfmaxvf_float32xm4
#define vfmax_vf_f32m8                           vfmaxvf_float32xm8
#define vfmax_vv_f32m1_m(mask, merged, a, b, vl) vfmaxvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfmax_vv_f32m2_m(mask, merged, a, b, vl) vfmaxvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfmax_vv_f32m4_m(mask, merged, a, b, vl) vfmaxvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfmax_vv_f32m8_m(mask, merged, a, b, vl) vfmaxvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfmax_vf_f32m1_m(mask, merged, a, b, vl) vfmaxvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfmax_vf_f32m2_m(mask, merged, a, b, vl) vfmaxvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfmax_vf_f32m4_m(mask, merged, a, b, vl) vfmaxvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfmax_vf_f32m8_m(mask, merged, a, b, vl) vfmaxvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfmin_vv_f32m1                           vfminvv_float32xm1
#define vfmin_vv_f32m2                           vfminvv_float32xm2
#define vfmin_vv_f32m4                           vfminvv_float32xm4
#define vfmin_vv_f32m8                           vfminvv_float32xm8
#define vfmin_vf_f32m1                           vfminvf_float32xm1
#define vfmin_vf_f32m2                           vfminvf_float32xm2
#define vfmin_vf_f32m4                           vfminvf_float32xm4
#define vfmin_vf_f32m8                           vfminvf_float32xm8
#define vfmin_vv_f32m1_m(mask, merged, a, b, vl) vfminvv_mask_float32xm1(merged, a, b, mask, vl)
#define vfmin_vv_f32m2_m(mask, merged, a, b, vl) vfminvv_mask_float32xm2(merged, a, b, mask, vl)
#define vfmin_vv_f32m4_m(mask, merged, a, b, vl) vfminvv_mask_float32xm4(merged, a, b, mask, vl)
#define vfmin_vv_f32m8_m(mask, merged, a, b, vl) vfminvv_mask_float32xm8(merged, a, b, mask, vl)
#define vfmin_vf_f32m1_m(mask, merged, a, b, vl) vfminvf_mask_float32xm1(merged, a, b, mask, vl)
#define vfmin_vf_f32m2_m(mask, merged, a, b, vl) vfminvf_mask_float32xm2(merged, a, b, mask, vl)
#define vfmin_vf_f32m4_m(mask, merged, a, b, vl) vfminvf_mask_float32xm4(merged, a, b, mask, vl)
#define vfmin_vf_f32m8_m(mask, merged, a, b, vl) vfminvf_mask_float32xm8(merged, a, b, mask, vl)

#define vfrsub_vv_f32m1 vfrsubvv_float32xm1
#define vfrsub_vv_f32m2 vfrsubvv_float32xm2
#define vfrsub_vv_f32m4 vfrsubvv_float32xm4
#define vfrsub_vv_f32m8 vfrsubvv_float32xm8
#define vfrsub_vf_f32m1 vfrsubvf_float32xm1
#define vfrsub_vf_f32m2 vfrsubvf_float32xm2
#define vfrsub_vf_f32m4 vfrsubvf_float32xm4
#define vfrsub_vf_f32m8 vfrsubvf_float32xm8

#define vfrdiv_vv_f32m1 vfrdivvv_float32xm1
#define vfrdiv_vv_f32m2 vfrdivvv_float32xm2
#define vfrdiv_vv_f32m4 vfrdivvv_float32xm4
#define vfrdiv_vv_f32m8 vfrdivvv_float32xm8
#define vfrdiv_vf_f32m1 vfrdivvf_float32xm1
#define vfrdiv_vf_f32m2 vfrdivvf_float32xm2
#define vfrdiv_vf_f32m4 vfrdivvf_float32xm4
#define vfrdiv_vf_f32m8 vfrdivvf_float32xm8

#define vfsgnj_vv_f32m1 vfsgnjvv_float32xm1
#define vfsgnj_vv_f32m2 vfsgnjvv_float32xm2
#define vfsgnj_vv_f32m4 vfsgnjvv_float32xm4
#define vfsgnj_vv_f32m8 vfsgnjvv_float32xm8
#define vfsgnj_vf_f32m1 vfsgnjvf_float32xm1
#define vfsgnj_vf_f32m2 vfsgnjvf_float32xm2
#define vfsgnj_vf_f32m4 vfsgnjvf_float32xm4
#define vfsgnj_vf_f32m8 vfsgnjvf_float32xm8

#define vfsgnjn_vv_f32m1 vfsgnjnvv_float32xm1
#define vfsgnjn_vv_f32m2 vfsgnjnvv_float32xm2
#define vfsgnjn_vv_f32m4 vfsgnjnvv_float32xm4
#define vfsgnjn_vv_f32m8 vfsgnjnvv_float32xm8
#define vfsgnjn_vf_f32m1 vfsgnjnvf_float32xm1
#define vfsgnjn_vf_f32m2 vfsgnjnvf_float32xm2
#define vfsgnjn_vf_f32m4 vfsgnjnvf_float32xm4
#define vfsgnjn_vf_f32m8 vfsgnjnvf_float32xm8

#define vfsgnjx_vv_f32m1 vfsgnjxvv_float32xm1
#define vfsgnjx_vv_f32m2 vfsgnjxvv_float32xm2
#define vfsgnjx_vv_f32m4 vfsgnjxvv_float32xm4
#define vfsgnjx_vv_f32m8 vfsgnjxvv_float32xm8
#define vfsgnjx_vf_f32m1 vfsgnjxvf_float32xm1
#define vfsgnjx_vf_f32m2 vfsgnjxvf_float32xm2
#define vfsgnjx_vf_f32m4 vfsgnjxvf_float32xm4
#define vfsgnjx_vf_f32m8 vfsgnjxvf_float32xm8

#define vfsqrt_v_f32m1 vfsqrtv_float32xm1
#define vfsqrt_v_f32m2 vfsqrtv_float32xm2
#define vfsqrt_v_f32m4 vfsqrtv_float32xm4
#define vfsqrt_v_f32m8 vfsqrtv_float32xm8

#define vfneg_v_f32m1(x, vl) vfrsubvf_float32xm1(x, 0.f, vl)
#define vfneg_v_f32m2(x, vl) vfrsubvf_float32xm2(x, 0.f, vl)
#define vfneg_v_f32m4(x, vl) vfrsubvf_float32xm4(x, 0.f, vl)
#define vfneg_v_f32m8(x, vl) vfrsubvf_float32xm8(x, 0.f, vl)

#define vfrec7_v_f32m1(x, vl) vfrdivvf_float32xm1(x, 1.f, vl)
#define vfrec7_v_f32m2(x, vl) vfrdivvf_float32xm2(x, 1.f, vl)
#define vfrec7_v_f32m4(x, vl) vfrdivvf_float32xm4(x, 1.f, vl)
#define vfrec7_v_f32m8(x, vl) vfrdivvf_float32xm8(x, 1.f, vl)

#define vfrsqrt7_v_f32m1(x, vl) vfrdivvf_float32xm1(vfsqrtv_float32xm1(x, vl), 1.f, vl)
#define vfrsqrt7_v_f32m2(x, vl) vfrdivvf_float32xm2(vfsqrtv_float32xm2(x, vl), 1.f, vl)
#define vfrsqrt7_v_f32m4(x, vl) vfrdivvf_float32xm4(vfsqrtv_float32xm4(x, vl), 1.f, vl)
#define vfrsqrt7_v_f32m8(x, vl) vfrdivvf_float32xm8(vfsqrtv_float32xm8(x, vl), 1.f, vl)

#define vfmacc_vv_f32m1 vfmaccvv_float32xm1
#define vfmacc_vv_f32m2 vfmaccvv_float32xm2
#define vfmacc_vv_f32m4 vfmaccvv_float32xm4
#define vfmacc_vv_f32m8 vfmaccvv_float32xm8
#define vfmacc_vf_f32m1 vfmaccvf_float32xm1
#define vfmacc_vf_f32m2 vfmaccvf_float32xm2
#define vfmacc_vf_f32m4 vfmaccvf_float32xm4
#define vfmacc_vf_f32m8 vfmaccvf_float32xm8

#define vfnmsac_vv_f32m1 vfnmsacvv_float32xm1
#define vfnmsac_vv_f32m2 vfnmsacvv_float32xm2
#define vfnmsac_vv_f32m4 vfnmsacvv_float32xm4
#define vfnmsac_vv_f32m8 vfnmsacvv_float32xm8
#define vfnmsac_vf_f32m1 vfnmsacvf_float32xm1
#define vfnmsac_vf_f32m2 vfnmsacvf_float32xm2
#define vfnmsac_vf_f32m4 vfnmsacvf_float32xm4
#define vfnmsac_vf_f32m8 vfnmsacvf_float32xm8

#define vfwmul_vv_f32m2 vfwmulvv_float32xm2_float16xm1
#define vfwmul_vv_f32m4 vfwmulvv_float32xm4_float16xm2
#define vfwmul_vv_f32m8 vfwmulvv_float32xm8_float16xm4
#define vfwmul_vf_f32m2 vfwmulvf_float32xm2_float16xm1
#define vfwmul_vf_f32m4 vfwmulvf_float32xm4_float16xm2
#define vfwmul_vf_f32m8 vfwmulvf_float32xm8_float16xm4

#define vfwmacc_vv_f32m2 vfwmaccvv_float32xm2_float16xm1
#define vfwmacc_vv_f32m4 vfwmaccvv_float32xm4_float16xm2
#define vfwmacc_vv_f32m8 vfwmaccvv_float32xm8_float16xm4
#define vfwmacc_vf_f32m2 vfwmaccvf_float32xm2_float16xm1
#define vfwmacc_vf_f32m4 vfwmaccvf_float32xm4_float16xm2
#define vfwmacc_vf_f32m8 vfwmaccvf_float32xm8_float16xm4

static inline vfloat32m1_t vfredsum_vs_f32m1_f32m1(vfloat32m1_t dst, vfloat32m1_t a, vfloat32m1_t b, word_type vl)
{
    return vfredsumvs_float32xm1(a, b, vl);
}
static inline vfloat32m1_t vfredsum_vs_f32m2_f32m1(vfloat32m1_t dst, vfloat32m2_t a, vfloat32m1_t b, word_type vl)
{
    float32xm2_u b2;
    b2.m1[0] = b;
    b2.m1[1] = vfmvvf_float32xm1(0.f, vl);
    b2.v = vfredsumvs_float32xm2(a, b2.v, vl);
    return vfaddvv_float32xm1(b2.m1[0], b2.m1[1], vl);
}
static inline vfloat32m1_t vfredsum_vs_f32m4_f32m1(vfloat32m1_t dst, vfloat32m4_t a, vfloat32m1_t b, word_type vl)
{
    float32xm4_u b4;
    b4.m1[0] = b;
    b4.m1[1] = vfmvvf_float32xm1(0.f, vl);
    b4.m1[2] = vfmvvf_float32xm1(0.f, vl);
    b4.m1[3] = vfmvvf_float32xm1(0.f, vl);
    b4.v = vfredsumvs_float32xm4(a, b4.v, vl);
    return vfaddvv_float32xm1(vfaddvv_float32xm1(b4.m1[0], b4.m1[1], vl), vfaddvv_float32xm1(b4.m1[2], b4.m1[3], vl), vl);
}
static inline vfloat32m1_t vfredsum_vs_f32m8_f32m1(vfloat32m1_t dst, vfloat32m8_t a, vfloat32m1_t b, word_type vl)
{
    float32xm8_u b8;
    b8.m1[0] = b;
    b8.m1[1] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[2] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[3] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[4] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[5] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[6] = vfmvvf_float32xm1(0.f, vl);
    b8.m1[7] = vfmvvf_float32xm1(0.f, vl);
    b8.v = vfredsumvs_float32xm8(a, b8.v, vl);
    return vfaddvv_float32xm1(vfaddvv_float32xm1(vfaddvv_float32xm1(b8.m1[0], b8.m1[1], vl), vfaddvv_float32xm1(b8.m1[2], b8.m1[3], vl), vl), vfaddvv_float32xm1(vfaddvv_float32xm1(b8.m1[4], b8.m1[5], vl), vfaddvv_float32xm1(b8.m1[6], b8.m1[7], vl), vl), vl);
}

#define vmfeq_vv_f32m1_b32 vmfeqvv_e32xm1_float32xm1
#define vmfeq_vv_f32m2_b16 vmfeqvv_e32xm2_float32xm2
#define vmfeq_vv_f32m4_b8  vmfeqvv_e32xm4_float32xm4
#define vmfeq_vv_f32m8_b4  vmfeqvv_e32xm8_float32xm8
#define vmfeq_vf_f32m1_b32 vmfeqvf_e32xm1_float32xm1
#define vmfeq_vf_f32m2_b16 vmfeqvf_e32xm2_float32xm2
#define vmfeq_vf_f32m4_b8  vmfeqvf_e32xm4_float32xm4
#define vmfeq_vf_f32m8_b4  vmfeqvf_e32xm8_float32xm8

#define vmfne_vv_f32m1_b32 vmfnevv_e32xm1_float32xm1
#define vmfne_vv_f32m2_b16 vmfnevv_e32xm2_float32xm2
#define vmfne_vv_f32m4_b8  vmfnevv_e32xm4_float32xm4
#define vmfne_vv_f32m8_b4  vmfnevv_e32xm8_float32xm8
#define vmfne_vf_f32m1_b32 vmfnevf_e32xm1_float32xm1
#define vmfne_vf_f32m2_b16 vmfnevf_e32xm2_float32xm2
#define vmfne_vf_f32m4_b8  vmfnevf_e32xm4_float32xm4
#define vmfne_vf_f32m8_b4  vmfnevf_e32xm8_float32xm8

#define vmfgt_vv_f32m1_b32 vmfgtvv_e32xm1_float32xm1
#define vmfgt_vv_f32m2_b16 vmfgtvv_e32xm2_float32xm2
#define vmfgt_vv_f32m4_b8  vmfgtvv_e32xm4_float32xm4
#define vmfgt_vv_f32m8_b4  vmfgtvv_e32xm8_float32xm8
#define vmfgt_vf_f32m1_b32 vmfgtvf_e32xm1_float32xm1
#define vmfgt_vf_f32m2_b16 vmfgtvf_e32xm2_float32xm2
#define vmfgt_vf_f32m4_b8  vmfgtvf_e32xm4_float32xm4
#define vmfgt_vf_f32m8_b4  vmfgtvf_e32xm8_float32xm8

#define vmfge_vv_f32m1_b32 vmfgevv_e32xm1_float32xm1
#define vmfge_vv_f32m2_b16 vmfgevv_e32xm2_float32xm2
#define vmfge_vv_f32m4_b8  vmfgevv_e32xm4_float32xm4
#define vmfge_vv_f32m8_b4  vmfgevv_e32xm8_float32xm8
#define vmfge_vf_f32m1_b32 vmfgevf_e32xm1_float32xm1
#define vmfge_vf_f32m2_b16 vmfgevf_e32xm2_float32xm2
#define vmfge_vf_f32m4_b8  vmfgevf_e32xm4_float32xm4
#define vmfge_vf_f32m8_b4  vmfgevf_e32xm8_float32xm8

#define vmflt_vv_f32m1_b32 vmfltvv_e32xm1_float32xm1
#define vmflt_vv_f32m2_b16 vmfltvv_e32xm2_float32xm2
#define vmflt_vv_f32m4_b8  vmfltvv_e32xm4_float32xm4
#define vmflt_vv_f32m8_b4  vmfltvv_e32xm8_float32xm8
#define vmflt_vf_f32m1_b32 vmfltvf_e32xm1_float32xm1
#define vmflt_vf_f32m2_b16 vmfltvf_e32xm2_float32xm2
#define vmflt_vf_f32m4_b8  vmfltvf_e32xm4_float32xm4
#define vmflt_vf_f32m8_b4  vmfltvf_e32xm8_float32xm8

#define vmfle_vv_f32m1_b32 vmflevv_e32xm1_float32xm1
#define vmfle_vv_f32m2_b16 vmflevv_e32xm2_float32xm2
#define vmfle_vv_f32m4_b8  vmflevv_e32xm4_float32xm4
#define vmfle_vv_f32m8_b4  vmflevv_e32xm8_float32xm8
#define vmfle_vf_f32m1_b32 vmflevf_e32xm1_float32xm1
#define vmfle_vf_f32m2_b16 vmflevf_e32xm2_float32xm2
#define vmfle_vf_f32m4_b8  vmflevf_e32xm4_float32xm4
#define vmfle_vf_f32m8_b4  vmflevf_e32xm8_float32xm8

#define vfcvt_x_f_v_i32m1 vfcvtxfv_int32xm1_float32xm1
#define vfcvt_x_f_v_i32m2 vfcvtxfv_int32xm2_float32xm2
#define vfcvt_x_f_v_i32m4 vfcvtxfv_int32xm4_float32xm4
#define vfcvt_x_f_v_i32m8 vfcvtxfv_int32xm8_float32xm8

#define vfcvt_f_x_v_f32m1 vfcvtfxv_float32xm1_int32xm1
#define vfcvt_f_x_v_f32m2 vfcvtfxv_float32xm2_int32xm2
#define vfcvt_f_x_v_f32m4 vfcvtfxv_float32xm4_int32xm4
#define vfcvt_f_x_v_f32m8 vfcvtfxv_float32xm8_int32xm8

#define vfcvt_xu_f_v_u32m1 vfcvtxufv_uint32xm1_float32xm1
#define vfcvt_xu_f_v_u32m2 vfcvtxufv_uint32xm2_float32xm2
#define vfcvt_xu_f_v_u32m4 vfcvtxufv_uint32xm4_float32xm4
#define vfcvt_xu_f_v_u32m8 vfcvtxufv_uint32xm8_float32xm8

#define vfcvt_f_xu_v_f32m1 vfcvtfxuv_float32xm1_uint32xm1
#define vfcvt_f_xu_v_f32m2 vfcvtfxuv_float32xm2_uint32xm2
#define vfcvt_f_xu_v_f32m4 vfcvtfxuv_float32xm4_uint32xm4
#define vfcvt_f_xu_v_f32m8 vfcvtfxuv_float32xm8_uint32xm8

#define vfwcvt_f_f_v_f32m2 vfwcvtffv_float32xm2_float16xm1
#define vfwcvt_f_f_v_f32m4 vfwcvtffv_float32xm4_float16xm2
#define vfwcvt_f_f_v_f32m8 vfwcvtffv_float32xm8_float16xm4

#define vfncvt_f_f_w_f16m1 vfncvtffv_float16xm1_float32xm2
#define vfncvt_f_f_w_f16m2 vfncvtffv_float16xm2_float32xm4
#define vfncvt_f_f_w_f16m4 vfncvtffv_float16xm4_float32xm8

#define vmerge_vvm_f32m1(mask, a, b, vl) reinterpret_cast<vfloat32m1_t>(vmergevvm_mask_uint32xm1(reinterpret_cast<vuint32m1_t>(a), reinterpret_cast<vuint32m1_t>(b), mask, vl))
#define vmerge_vvm_f32m2(mask, a, b, vl) reinterpret_cast<vfloat32m2_t>(vmergevvm_mask_uint32xm2(reinterpret_cast<vuint32m2_t>(a), reinterpret_cast<vuint32m2_t>(b), mask, vl))
#define vmerge_vvm_f32m4(mask, a, b, vl) reinterpret_cast<vfloat32m4_t>(vmergevvm_mask_uint32xm4(reinterpret_cast<vuint32m4_t>(a), reinterpret_cast<vuint32m4_t>(b), mask, vl))
#define vmerge_vvm_f32m8(mask, a, b, vl) reinterpret_cast<vfloat32m8_t>(vmergevvm_mask_uint32xm8(reinterpret_cast<vuint32m8_t>(a), reinterpret_cast<vuint32m8_t>(b), mask, vl))

#define vfmerge_vfm_f32m1(mask, a, b, vl) vfmergevfm_mask_float32xm1(a, b, mask, vl)
#define vfmerge_vfm_f32m2(mask, a, b, vl) vfmergevfm_mask_float32xm2(a, b, mask, vl)
#define vfmerge_vfm_f32m4(mask, a, b, vl) vfmergevfm_mask_float32xm4(a, b, mask, vl)
#define vfmerge_vfm_f32m8(mask, a, b, vl) vfmergevfm_mask_float32xm8(a, b, mask, vl)

#define vreinterpret_v_i32m1_f32m1(x) reinterpret_cast<vfloat32m1_t>(x)
#define vreinterpret_v_i32m2_f32m2(x) reinterpret_cast<vfloat32m2_t>(x)
#define vreinterpret_v_i32m4_f32m4(x) reinterpret_cast<vfloat32m4_t>(x)
#define vreinterpret_v_i32m8_f32m8(x) reinterpret_cast<vfloat32m8_t>(x)

#define vreinterpret_v_u32m1_f32m1(x) reinterpret_cast<vfloat32m1_t>(x)
#define vreinterpret_v_u32m2_f32m2(x) reinterpret_cast<vfloat32m2_t>(x)
#define vreinterpret_v_u32m4_f32m4(x) reinterpret_cast<vfloat32m4_t>(x)
#define vreinterpret_v_u32m8_f32m8(x) reinterpret_cast<vfloat32m8_t>(x)

/******************************** float16 ********************************/
#define vle16_v_f16m1 vlev_float16xm1
#define vle16_v_f16m2 vlev_float16xm2
#define vle16_v_f16m4 vlev_float16xm4
#define vle16_v_f16m8 vlev_float16xm8

#define vse16_v_f16m1 vsev_float16xm1
#define vse16_v_f16m2 vsev_float16xm2
#define vse16_v_f16m4 vsev_float16xm4
#define vse16_v_f16m8 vsev_float16xm8

#define vlse16_v_f16m1 vlsev_float16xm1
#define vlse16_v_f16m2 vlsev_float16xm2
#define vlse16_v_f16m4 vlsev_float16xm4
#define vlse16_v_f16m8 vlsev_float16xm8

#define vsse16_v_f16m1 vssev_float16xm1
#define vsse16_v_f16m2 vssev_float16xm2
#define vsse16_v_f16m4 vssev_float16xm4
#define vsse16_v_f16m8 vssev_float16xm8

#define vlseg2e16_v_f16m1x2 vlseg2ev_float16x2xm1
#define vsseg2e16_v_f16m1x2 vsseg2ev_float16x2xm1

#define vlseg4e16_v_f16m1x4 vlseg4ev_float16x4xm1
#define vsseg4e16_v_f16m1x4 vsseg4ev_float16x4xm1

#define vlseg8e16_v_f16m1x8 vlseg8ev_float16x8xm1
#define vsseg8e16_v_f16m1x8 vsseg8ev_float16x8xm1

#define vlseg4e16_v_f16m2x4 vlseg4ev_float16x4xm2
#define vsseg4e16_v_f16m2x4 vsseg4ev_float16x4xm2

#define vssseg8e16_v_f16m1x8 vssseg8ev_float16x8xm1
#define vssseg4e16_v_f16m1x4 vssseg4ev_float16x4xm1
#define vssseg2e16_v_f16m1x2 vssseg2ev_float16x2xm1

#define vset_f16m1x2       vseg_element_set_float16x2xm1
#define vset_f16m1x4       vseg_element_set_float16x4xm1
#define vset_f16m1x8       vseg_element_set_float16x8xm1
#define vget_f16m1x2_f16m1 vseg_element_get_float16x2xm1
#define vget_f16m1x4_f16m1 vseg_element_get_float16x4xm1
#define vget_f16m1x8_f16m1 vseg_element_get_float16x8xm1

static inline vfloat16m1x2_t vcreate_f16m1x2(vfloat16m1_t v0, vfloat16m1_t v1)
{
    vfloat16m1x2_t p;
    p = vset_f16m1x2(p, 0, v0);
    p = vset_f16m1x2(p, 1, v1);
    return p;
}

static inline vfloat16m1x4_t vcreate_f16m1x4(vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3)
{
    vfloat16m1x4_t p;
    p = vset_f16m1x4(p, 0, v0);
    p = vset_f16m1x4(p, 1, v1);
    p = vset_f16m1x4(p, 2, v2);
    p = vset_f16m1x4(p, 3, v3);
    return p;
}

static inline vfloat16m1x8_t vcreate_f16m1x8(vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7)
{
    vfloat16m1x8_t p;
    p = vset_f16m1x8(p, 0, v0);
    p = vset_f16m1x8(p, 1, v1);
    p = vset_f16m1x8(p, 2, v2);
    p = vset_f16m1x8(p, 3, v3);
    p = vset_f16m1x8(p, 4, v4);
    p = vset_f16m1x8(p, 5, v5);
    p = vset_f16m1x8(p, 6, v6);
    p = vset_f16m1x8(p, 7, v7);
    return p;
}

#define vfmv_s_f_f16m1(a, b, vl) vfmvsf_float16xm1(b, vl)
#define vfmv_s_f_f16m2(a, b, vl) vfmvsf_float16xm2(b, vl)
#define vfmv_s_f_f16m4(a, b, vl) vfmvsf_float16xm4(b, vl)
#define vfmv_s_f_f16m8(a, b, vl) vfmvsf_float16xm8(b, vl)

#define vfmv_f_s_f16m1_f16(x) vfmvfs_float16xm1(x, vl)
#define vfmv_f_s_f16m2_f16(x) vfmvfs_float16xm2(x, vl)
#define vfmv_f_s_f16m4_f16(x) vfmvfs_float16xm4(x, vl)
#define vfmv_f_s_f16m8_f16(x) vfmvfs_float16xm8(x, vl)

#define vfmv_v_f_f16m1 vfmvvf_float16xm1
#define vfmv_v_f_f16m2 vfmvvf_float16xm2
#define vfmv_v_f_f16m4 vfmvvf_float16xm4
#define vfmv_v_f_f16m8 vfmvvf_float16xm8

#define vfadd_vv_f16m1                           vfaddvv_float16xm1
#define vfadd_vv_f16m2                           vfaddvv_float16xm2
#define vfadd_vv_f16m4                           vfaddvv_float16xm4
#define vfadd_vv_f16m8                           vfaddvv_float16xm8
#define vfadd_vf_f16m1                           vfaddvf_float16xm1
#define vfadd_vf_f16m2                           vfaddvf_float16xm2
#define vfadd_vf_f16m4                           vfaddvf_float16xm4
#define vfadd_vf_f16m8                           vfaddvf_float16xm8
#define vfadd_vv_f16m1_m(mask, merged, a, b, vl) vfaddvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfadd_vv_f16m2_m(mask, merged, a, b, vl) vfaddvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfadd_vv_f16m4_m(mask, merged, a, b, vl) vfaddvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfadd_vv_f16m8_m(mask, merged, a, b, vl) vfaddvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfadd_vf_f16m1_m(mask, merged, a, b, vl) vfaddvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfadd_vf_f16m2_m(mask, merged, a, b, vl) vfaddvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfadd_vf_f16m4_m(mask, merged, a, b, vl) vfaddvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfadd_vf_f16m8_m(mask, merged, a, b, vl) vfaddvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfsub_vv_f16m1                           vfsubvv_float16xm1
#define vfsub_vv_f16m2                           vfsubvv_float16xm2
#define vfsub_vv_f16m4                           vfsubvv_float16xm4
#define vfsub_vv_f16m8                           vfsubvv_float16xm8
#define vfsub_vf_f16m1                           vfsubvf_float16xm1
#define vfsub_vf_f16m2                           vfsubvf_float16xm2
#define vfsub_vf_f16m4                           vfsubvf_float16xm4
#define vfsub_vf_f16m8                           vfsubvf_float16xm8
#define vfsub_vv_f16m1_m(mask, merged, a, b, vl) vfsubvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfsub_vv_f16m2_m(mask, merged, a, b, vl) vfsubvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfsub_vv_f16m4_m(mask, merged, a, b, vl) vfsubvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfsub_vv_f16m8_m(mask, merged, a, b, vl) vfsubvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfsub_vf_f16m1_m(mask, merged, a, b, vl) vfsubvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfsub_vf_f16m2_m(mask, merged, a, b, vl) vfsubvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfsub_vf_f16m4_m(mask, merged, a, b, vl) vfsubvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfsub_vf_f16m8_m(mask, merged, a, b, vl) vfsubvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfmul_vv_f16m1                           vfmulvv_float16xm1
#define vfmul_vv_f16m2                           vfmulvv_float16xm2
#define vfmul_vv_f16m4                           vfmulvv_float16xm4
#define vfmul_vv_f16m8                           vfmulvv_float16xm8
#define vfmul_vf_f16m1                           vfmulvf_float16xm1
#define vfmul_vf_f16m2                           vfmulvf_float16xm2
#define vfmul_vf_f16m4                           vfmulvf_float16xm4
#define vfmul_vf_f16m8                           vfmulvf_float16xm8
#define vfmul_vv_f16m1_m(mask, merged, a, b, vl) vfmulvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfmul_vv_f16m2_m(mask, merged, a, b, vl) vfmulvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfmul_vv_f16m4_m(mask, merged, a, b, vl) vfmulvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfmul_vv_f16m8_m(mask, merged, a, b, vl) vfmulvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfmul_vf_f16m1_m(mask, merged, a, b, vl) vfmulvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfmul_vf_f16m2_m(mask, merged, a, b, vl) vfmulvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfmul_vf_f16m4_m(mask, merged, a, b, vl) vfmulvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfmul_vf_f16m8_m(mask, merged, a, b, vl) vfmulvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfdiv_vv_f16m1                           vfdivvv_float16xm1
#define vfdiv_vv_f16m2                           vfdivvv_float16xm2
#define vfdiv_vv_f16m4                           vfdivvv_float16xm4
#define vfdiv_vv_f16m8                           vfdivvv_float16xm8
#define vfdiv_vf_f16m1                           vfdivvf_float16xm1
#define vfdiv_vf_f16m2                           vfdivvf_float16xm2
#define vfdiv_vf_f16m4                           vfdivvf_float16xm4
#define vfdiv_vf_f16m8                           vfdivvf_float16xm8
#define vfdiv_vv_f16m1_m(mask, merged, a, b, vl) vfdivvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfdiv_vv_f16m2_m(mask, merged, a, b, vl) vfdivvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfdiv_vv_f16m4_m(mask, merged, a, b, vl) vfdivvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfdiv_vv_f16m8_m(mask, merged, a, b, vl) vfdivvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfdiv_vf_f16m1_m(mask, merged, a, b, vl) vfdivvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfdiv_vf_f16m2_m(mask, merged, a, b, vl) vfdivvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfdiv_vf_f16m4_m(mask, merged, a, b, vl) vfdivvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfdiv_vf_f16m8_m(mask, merged, a, b, vl) vfdivvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfmax_vv_f16m1                           vfmaxvv_float16xm1
#define vfmax_vv_f16m2                           vfmaxvv_float16xm2
#define vfmax_vv_f16m4                           vfmaxvv_float16xm4
#define vfmax_vv_f16m8                           vfmaxvv_float16xm8
#define vfmax_vf_f16m1                           vfmaxvf_float16xm1
#define vfmax_vf_f16m2                           vfmaxvf_float16xm2
#define vfmax_vf_f16m4                           vfmaxvf_float16xm4
#define vfmax_vf_f16m8                           vfmaxvf_float16xm8
#define vfmax_vv_f16m1_m(mask, merged, a, b, vl) vfmaxvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfmax_vv_f16m2_m(mask, merged, a, b, vl) vfmaxvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfmax_vv_f16m4_m(mask, merged, a, b, vl) vfmaxvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfmax_vv_f16m8_m(mask, merged, a, b, vl) vfmaxvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfmax_vf_f16m1_m(mask, merged, a, b, vl) vfmaxvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfmax_vf_f16m2_m(mask, merged, a, b, vl) vfmaxvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfmax_vf_f16m4_m(mask, merged, a, b, vl) vfmaxvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfmax_vf_f16m8_m(mask, merged, a, b, vl) vfmaxvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfmin_vv_f16m1                           vfminvv_float16xm1
#define vfmin_vv_f16m2                           vfminvv_float16xm2
#define vfmin_vv_f16m4                           vfminvv_float16xm4
#define vfmin_vv_f16m8                           vfminvv_float16xm8
#define vfmin_vf_f16m1                           vfminvf_float16xm1
#define vfmin_vf_f16m2                           vfminvf_float16xm2
#define vfmin_vf_f16m4                           vfminvf_float16xm4
#define vfmin_vf_f16m8                           vfminvf_float16xm8
#define vfmin_vv_f16m1_m(mask, merged, a, b, vl) vfminvv_mask_float16xm1(merged, a, b, mask, vl)
#define vfmin_vv_f16m2_m(mask, merged, a, b, vl) vfminvv_mask_float16xm2(merged, a, b, mask, vl)
#define vfmin_vv_f16m4_m(mask, merged, a, b, vl) vfminvv_mask_float16xm4(merged, a, b, mask, vl)
#define vfmin_vv_f16m8_m(mask, merged, a, b, vl) vfminvv_mask_float16xm8(merged, a, b, mask, vl)
#define vfmin_vf_f16m1_m(mask, merged, a, b, vl) vfminvf_mask_float16xm1(merged, a, b, mask, vl)
#define vfmin_vf_f16m2_m(mask, merged, a, b, vl) vfminvf_mask_float16xm2(merged, a, b, mask, vl)
#define vfmin_vf_f16m4_m(mask, merged, a, b, vl) vfminvf_mask_float16xm4(merged, a, b, mask, vl)
#define vfmin_vf_f16m8_m(mask, merged, a, b, vl) vfminvf_mask_float16xm8(merged, a, b, mask, vl)

#define vfrsub_vv_f16m1 vfrsubvv_float16xm1
#define vfrsub_vv_f16m2 vfrsubvv_float16xm2
#define vfrsub_vv_f16m4 vfrsubvv_float16xm4
#define vfrsub_vv_f16m8 vfrsubvv_float16xm8
#define vfrsub_vf_f16m1 vfrsubvf_float16xm1
#define vfrsub_vf_f16m2 vfrsubvf_float16xm2
#define vfrsub_vf_f16m4 vfrsubvf_float16xm4
#define vfrsub_vf_f16m8 vfrsubvf_float16xm8

#define vfrdiv_vv_f16m1 vfrdivvv_float16xm1
#define vfrdiv_vv_f16m2 vfrdivvv_float16xm2
#define vfrdiv_vv_f16m4 vfrdivvv_float16xm4
#define vfrdiv_vv_f16m8 vfrdivvv_float16xm8
#define vfrdiv_vf_f16m1 vfrdivvf_float16xm1
#define vfrdiv_vf_f16m2 vfrdivvf_float16xm2
#define vfrdiv_vf_f16m4 vfrdivvf_float16xm4
#define vfrdiv_vf_f16m8 vfrdivvf_float16xm8

#define vfsgnj_vv_f16m1 vfsgnjvv_float16xm1
#define vfsgnj_vv_f16m2 vfsgnjvv_float16xm2
#define vfsgnj_vv_f16m4 vfsgnjvv_float16xm4
#define vfsgnj_vv_f16m8 vfsgnjvv_float16xm8
#define vfsgnj_vf_f16m1 vfsgnjvf_float16xm1
#define vfsgnj_vf_f16m2 vfsgnjvf_float16xm2
#define vfsgnj_vf_f16m4 vfsgnjvf_float16xm4
#define vfsgnj_vf_f16m8 vfsgnjvf_float16xm8

#define vfsgnjn_vv_f16m1 vfsgnjnvv_float16xm1
#define vfsgnjn_vv_f16m2 vfsgnjnvv_float16xm2
#define vfsgnjn_vv_f16m4 vfsgnjnvv_float16xm4
#define vfsgnjn_vv_f16m8 vfsgnjnvv_float16xm8
#define vfsgnjn_vf_f16m1 vfsgnjnvf_float16xm1
#define vfsgnjn_vf_f16m2 vfsgnjnvf_float16xm2
#define vfsgnjn_vf_f16m4 vfsgnjnvf_float16xm4
#define vfsgnjn_vf_f16m8 vfsgnjnvf_float16xm8

#define vfsgnjx_vv_f16m1 vfsgnjxvv_float16xm1
#define vfsgnjx_vv_f16m2 vfsgnjxvv_float16xm2
#define vfsgnjx_vv_f16m4 vfsgnjxvv_float16xm4
#define vfsgnjx_vv_f16m8 vfsgnjxvv_float16xm8
#define vfsgnjx_vf_f16m1 vfsgnjxvf_float16xm1
#define vfsgnjx_vf_f16m2 vfsgnjxvf_float16xm2
#define vfsgnjx_vf_f16m4 vfsgnjxvf_float16xm4
#define vfsgnjx_vf_f16m8 vfsgnjxvf_float16xm8

#define vfsqrt_v_f16m1 vfsqrtv_float16xm1
#define vfsqrt_v_f16m2 vfsqrtv_float16xm2
#define vfsqrt_v_f16m4 vfsqrtv_float16xm4
#define vfsqrt_v_f16m8 vfsqrtv_float16xm8

#define vfneg_v_f16m1(x, vl) vfrsubvf_float16xm1(x, 0.f, vl)
#define vfneg_v_f16m2(x, vl) vfrsubvf_float16xm2(x, 0.f, vl)
#define vfneg_v_f16m4(x, vl) vfrsubvf_float16xm4(x, 0.f, vl)
#define vfneg_v_f16m8(x, vl) vfrsubvf_float16xm8(x, 0.f, vl)

#define vfrec7_v_f16m1(x, vl) vfrdivvf_float16xm1(x, 1.f, vl)
#define vfrec7_v_f16m2(x, vl) vfrdivvf_float16xm2(x, 1.f, vl)
#define vfrec7_v_f16m4(x, vl) vfrdivvf_float16xm4(x, 1.f, vl)
#define vfrec7_v_f16m8(x, vl) vfrdivvf_float16xm8(x, 1.f, vl)

#define vfrsqrt7_v_f16m1(x, vl) vfrdivvf_float16xm1(vfsqrtv_float16xm1(x, vl), 1.f, vl)
#define vfrsqrt7_v_f16m2(x, vl) vfrdivvf_float16xm2(vfsqrtv_float16xm2(x, vl), 1.f, vl)
#define vfrsqrt7_v_f16m4(x, vl) vfrdivvf_float16xm4(vfsqrtv_float16xm4(x, vl), 1.f, vl)
#define vfrsqrt7_v_f16m8(x, vl) vfrdivvf_float16xm8(vfsqrtv_float16xm8(x, vl), 1.f, vl)

#define vfmacc_vv_f16m1 vfmaccvv_float16xm1
#define vfmacc_vv_f16m2 vfmaccvv_float16xm2
#define vfmacc_vv_f16m4 vfmaccvv_float16xm4
#define vfmacc_vv_f16m8 vfmaccvv_float16xm8
#define vfmacc_vf_f16m1 vfmaccvf_float16xm1
#define vfmacc_vf_f16m2 vfmaccvf_float16xm2
#define vfmacc_vf_f16m4 vfmaccvf_float16xm4
#define vfmacc_vf_f16m8 vfmaccvf_float16xm8

#define vfnmsac_vv_f16m1 vfnmsacvv_float16xm1
#define vfnmsac_vv_f16m2 vfnmsacvv_float16xm2
#define vfnmsac_vv_f16m4 vfnmsacvv_float16xm4
#define vfnmsac_vv_f16m8 vfnmsacvv_float16xm8
#define vfnmsac_vf_f16m1 vfnmsacvf_float16xm1
#define vfnmsac_vf_f16m2 vfnmsacvf_float16xm2
#define vfnmsac_vf_f16m4 vfnmsacvf_float16xm4
#define vfnmsac_vf_f16m8 vfnmsacvf_float16xm8

static inline vfloat16m1_t vfredsum_vs_f16m1_f16m1(vfloat16m1_t dst, vfloat16m1_t a, vfloat16m1_t b, word_type vl)
{
    return vfredsumvs_float16xm1(a, b, vl);
}
static inline vfloat16m1_t vfredsum_vs_f16m2_f16m1(vfloat16m1_t dst, vfloat16m2_t a, vfloat16m1_t b, word_type vl)
{
    float16xm2_u b2;
    b2.m1[0] = b;
    b2.m1[1] = vfmvvf_float16xm1(0.f, vl);
    b2.v = vfredsumvs_float16xm2(a, b2.v, vl);
    return vfaddvv_float16xm1(b2.m1[0], b2.m1[1], vl);
}
static inline vfloat16m1_t vfredsum_vs_f16m4_f16m1(vfloat16m1_t dst, vfloat16m4_t a, vfloat16m1_t b, word_type vl)
{
    float16xm4_u b4;
    b4.m1[0] = b;
    b4.m1[1] = vfmvvf_float16xm1(0.f, vl);
    b4.m1[2] = vfmvvf_float16xm1(0.f, vl);
    b4.m1[3] = vfmvvf_float16xm1(0.f, vl);
    b4.v = vfredsumvs_float16xm4(a, b4.v, vl);
    return vfaddvv_float16xm1(vfaddvv_float16xm1(b4.m1[0], b4.m1[1], vl), vfaddvv_float16xm1(b4.m1[2], b4.m1[3], vl), vl);
}
static inline vfloat16m1_t vfredsum_vs_f16m8_f16m1(vfloat16m1_t dst, vfloat16m8_t a, vfloat16m1_t b, word_type vl)
{
    float16xm8_u b8;
    b8.m1[0] = b;
    b8.m1[1] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[2] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[3] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[4] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[5] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[6] = vfmvvf_float16xm1(0.f, vl);
    b8.m1[7] = vfmvvf_float16xm1(0.f, vl);
    b8.v = vfredsumvs_float16xm8(a, b8.v, vl);
    return vfaddvv_float16xm1(vfaddvv_float16xm1(vfaddvv_float16xm1(b8.m1[0], b8.m1[1], vl), vfaddvv_float16xm1(b8.m1[2], b8.m1[3], vl), vl), vfaddvv_float16xm1(vfaddvv_float16xm1(b8.m1[4], b8.m1[5], vl), vfaddvv_float16xm1(b8.m1[6], b8.m1[7], vl), vl), vl);
}

#define vmfeq_vv_f16m1_b16 vmfeqvv_e16xm1_float16xm1
#define vmfeq_vv_f16m2_b8  vmfeqvv_e16xm2_float16xm2
#define vmfeq_vv_f16m4_b4  vmfeqvv_e16xm4_float16xm4
#define vmfeq_vv_f16m8_b2  vmfeqvv_e16xm8_float16xm8
#define vmfeq_vf_f16m1_b16 vmfeqvf_e16xm1_float16xm1
#define vmfeq_vf_f16m2_b8  vmfeqvf_e16xm2_float16xm2
#define vmfeq_vf_f16m4_b4  vmfeqvf_e16xm4_float16xm4
#define vmfeq_vf_f16m8_b2  vmfeqvf_e16xm8_float16xm8

#define vmfne_vv_f16m1_b16 vmfnevv_e16xm1_float16xm1
#define vmfne_vv_f16m2_b8  vmfnevv_e16xm2_float16xm2
#define vmfne_vv_f16m4_b4  vmfnevv_e16xm4_float16xm4
#define vmfne_vv_f16m8_b2  vmfnevv_e16xm8_float16xm8
#define vmfne_vf_f16m1_b16 vmfnevf_e16xm1_float16xm1
#define vmfne_vf_f16m2_b8  vmfnevf_e16xm2_float16xm2
#define vmfne_vf_f16m4_b4  vmfnevf_e16xm4_float16xm4
#define vmfne_vf_f16m8_b2  vmfnevf_e16xm8_float16xm8

#define vmfgt_vv_f16m1_b16 vmfgtvv_e16xm1_float16xm1
#define vmfgt_vv_f16m2_b8  vmfgtvv_e16xm2_float16xm2
#define vmfgt_vv_f16m4_b4  vmfgtvv_e16xm4_float16xm4
#define vmfgt_vv_f16m8_b2  vmfgtvv_e16xm8_float16xm8
#define vmfgt_vf_f16m1_b16 vmfgtvf_e16xm1_float16xm1
#define vmfgt_vf_f16m2_b8  vmfgtvf_e16xm2_float16xm2
#define vmfgt_vf_f16m4_b4  vmfgtvf_e16xm4_float16xm4
#define vmfgt_vf_f16m8_b2  vmfgtvf_e16xm8_float16xm8

#define vmfge_vv_f16m1_b16 vmfgevv_e16xm1_float16xm1
#define vmfge_vv_f16m2_b8  vmfgevv_e16xm2_float16xm2
#define vmfge_vv_f16m4_b4  vmfgevv_e16xm4_float16xm4
#define vmfge_vv_f16m8_b2  vmfgevv_e16xm8_float16xm8
#define vmfge_vf_f16m1_b16 vmfgevf_e16xm1_float16xm1
#define vmfge_vf_f16m2_b8  vmfgevf_e16xm2_float16xm2
#define vmfge_vf_f16m4_b4  vmfgevf_e16xm4_float16xm4
#define vmfge_vf_f16m8_b2  vmfgevf_e16xm8_float16xm8

#define vmflt_vv_f16m1_b16 vmfltvv_e16xm1_float16xm1
#define vmflt_vv_f16m2_b8  vmfltvv_e16xm2_float16xm2
#define vmflt_vv_f16m4_b4  vmfltvv_e16xm4_float16xm4
#define vmflt_vv_f16m8_b2  vmfltvv_e16xm8_float16xm8
#define vmflt_vf_f16m1_b16 vmfltvf_e16xm1_float16xm1
#define vmflt_vf_f16m2_b8  vmfltvf_e16xm2_float16xm2
#define vmflt_vf_f16m4_b4  vmfltvf_e16xm4_float16xm4
#define vmflt_vf_f16m8_b2  vmfltvf_e16xm8_float16xm8

#define vmfle_vv_f16m1_b16 vmflevv_e16xm1_float16xm1
#define vmfle_vv_f16m2_b8  vmflevv_e16xm2_float16xm2
#define vmfle_vv_f16m4_b4  vmflevv_e16xm4_float16xm4
#define vmfle_vv_f16m8_b2  vmflevv_e16xm8_float16xm8
#define vmfle_vf_f16m1_b16 vmflevf_e16xm1_float16xm1
#define vmfle_vf_f16m2_b8  vmflevf_e16xm2_float16xm2
#define vmfle_vf_f16m4_b4  vmflevf_e16xm4_float16xm4
#define vmfle_vf_f16m8_b2  vmflevf_e16xm8_float16xm8

#define vfcvt_x_f_v_i16m1 vfcvtxfv_int16xm1_float16xm1
#define vfcvt_x_f_v_i16m2 vfcvtxfv_int16xm2_float16xm2
#define vfcvt_x_f_v_i16m4 vfcvtxfv_int16xm4_float16xm4
#define vfcvt_x_f_v_i16m8 vfcvtxfv_int16xm8_float16xm8

#define vfcvt_f_x_v_f16m1 vfcvtfxv_float16xm1_int16xm1
#define vfcvt_f_x_v_f16m2 vfcvtfxv_float16xm2_int16xm2
#define vfcvt_f_x_v_f16m4 vfcvtfxv_float16xm4_int16xm4
#define vfcvt_f_x_v_f16m8 vfcvtfxv_float16xm8_int16xm8

#define vfcvt_xu_f_v_u16m1 vfcvtxufv_uint16xm1_float16xm1
#define vfcvt_xu_f_v_u16m2 vfcvtxufv_uint16xm2_float16xm2
#define vfcvt_xu_f_v_u16m4 vfcvtxufv_uint16xm4_float16xm4
#define vfcvt_xu_f_v_u16m8 vfcvtxufv_uint16xm8_float16xm8

#define vfcvt_f_xu_v_f16m1 vfcvtfxuv_float16xm1_uint16xm1
#define vfcvt_f_xu_v_f16m2 vfcvtfxuv_float16xm2_uint16xm2
#define vfcvt_f_xu_v_f16m4 vfcvtfxuv_float16xm4_uint16xm4
#define vfcvt_f_xu_v_f16m8 vfcvtfxuv_float16xm8_uint16xm8

#define vmerge_vvm_f16m1(mask, a, b, vl) reinterpret_cast<vfloat16m1_t>(vmergevvm_mask_uint16xm1(reinterpret_cast<vuint16m1_t>(a), reinterpret_cast<vuint16m1_t>(b), mask, vl))
#define vmerge_vvm_f16m2(mask, a, b, vl) reinterpret_cast<vfloat16m2_t>(vmergevvm_mask_uint16xm2(reinterpret_cast<vuint16m2_t>(a), reinterpret_cast<vuint16m2_t>(b), mask, vl))
#define vmerge_vvm_f16m4(mask, a, b, vl) reinterpret_cast<vfloat16m4_t>(vmergevvm_mask_uint16xm4(reinterpret_cast<vuint16m4_t>(a), reinterpret_cast<vuint16m4_t>(b), mask, vl))
#define vmerge_vvm_f16m8(mask, a, b, vl) reinterpret_cast<vfloat16m8_t>(vmergevvm_mask_uint16xm8(reinterpret_cast<vuint16m8_t>(a), reinterpret_cast<vuint16m8_t>(b), mask, vl))

#define vfmerge_vfm_f16m1(mask, a, b, vl) vfmergevfm_mask_float16xm1(a, b, mask, vl)
#define vfmerge_vfm_f16m2(mask, a, b, vl) vfmergevfm_mask_float16xm2(a, b, mask, vl)
#define vfmerge_vfm_f16m4(mask, a, b, vl) vfmergevfm_mask_float16xm4(a, b, mask, vl)
#define vfmerge_vfm_f16m8(mask, a, b, vl) vfmergevfm_mask_float16xm8(a, b, mask, vl)

#define vreinterpret_v_i16m1_f16m1(x) reinterpret_cast<vfloat16m1_t>(x)
#define vreinterpret_v_i16m2_f16m2(x) reinterpret_cast<vfloat16m2_t>(x)
#define vreinterpret_v_i16m4_f16m4(x) reinterpret_cast<vfloat16m4_t>(x)
#define vreinterpret_v_i16m8_f16m8(x) reinterpret_cast<vfloat16m8_t>(x)

#define vreinterpret_v_u16m1_f16m1(x) reinterpret_cast<vfloat16m1_t>(x)
#define vreinterpret_v_u16m2_f16m2(x) reinterpret_cast<vfloat16m2_t>(x)
#define vreinterpret_v_u16m4_f16m4(x) reinterpret_cast<vfloat16m4_t>(x)
#define vreinterpret_v_u16m8_f16m8(x) reinterpret_cast<vfloat16m8_t>(x)

/******************************** int32 ********************************/
#define vadd_vv_i32m1                           vaddvv_int32xm1
#define vadd_vv_i32m2                           vaddvv_int32xm2
#define vadd_vv_i32m4                           vaddvv_int32xm4
#define vadd_vv_i32m8                           vaddvv_int32xm8
#define vadd_vx_i32m1                           vaddvx_int32xm1
#define vadd_vx_i32m2                           vaddvx_int32xm2
#define vadd_vx_i32m4                           vaddvx_int32xm4
#define vadd_vx_i32m8                           vaddvx_int32xm8
#define vadd_vv_i32m1_m(mask, merged, a, b, vl) vaddvv_mask_int32xm1(merged, a, b, mask, vl)
#define vadd_vv_i32m2_m(mask, merged, a, b, vl) vaddvv_mask_int32xm2(merged, a, b, mask, vl)
#define vadd_vv_i32m4_m(mask, merged, a, b, vl) vaddvv_mask_int32xm4(merged, a, b, mask, vl)
#define vadd_vv_i32m8_m(mask, merged, a, b, vl) vaddvv_mask_int32xm8(merged, a, b, mask, vl)
#define vadd_vx_i32m1_m(mask, merged, a, b, vl) vaddvx_mask_int32xm1(merged, a, b, mask, vl)
#define vadd_vx_i32m2_m(mask, merged, a, b, vl) vaddvx_mask_int32xm2(merged, a, b, mask, vl)
#define vadd_vx_i32m4_m(mask, merged, a, b, vl) vaddvx_mask_int32xm4(merged, a, b, mask, vl)
#define vadd_vx_i32m8_m(mask, merged, a, b, vl) vaddvx_mask_int32xm8(merged, a, b, mask, vl)

#define vsub_vv_i32m1                           vsubvv_int32xm1
#define vsub_vv_i32m2                           vsubvv_int32xm2
#define vsub_vv_i32m4                           vsubvv_int32xm4
#define vsub_vv_i32m8                           vsubvv_int32xm8
#define vsub_vx_i32m1                           vsubvx_int32xm1
#define vsub_vx_i32m2                           vsubvx_int32xm2
#define vsub_vx_i32m4                           vsubvx_int32xm4
#define vsub_vx_i32m8                           vsubvx_int32xm8
#define vsub_vv_i32m1_m(mask, merged, a, b, vl) vsubvv_mask_int32xm1(merged, a, b, mask, vl)
#define vsub_vv_i32m2_m(mask, merged, a, b, vl) vsubvv_mask_int32xm2(merged, a, b, mask, vl)
#define vsub_vv_i32m4_m(mask, merged, a, b, vl) vsubvv_mask_int32xm4(merged, a, b, mask, vl)
#define vsub_vv_i32m8_m(mask, merged, a, b, vl) vsubvv_mask_int32xm8(merged, a, b, mask, vl)
#define vsub_vx_i32m1_m(mask, merged, a, b, vl) vsubvx_mask_int32xm1(merged, a, b, mask, vl)
#define vsub_vx_i32m2_m(mask, merged, a, b, vl) vsubvx_mask_int32xm2(merged, a, b, mask, vl)
#define vsub_vx_i32m4_m(mask, merged, a, b, vl) vsubvx_mask_int32xm4(merged, a, b, mask, vl)
#define vsub_vx_i32m8_m(mask, merged, a, b, vl) vsubvx_mask_int32xm8(merged, a, b, mask, vl)

#define vsll_vv_i32m1                           vsllvv_int32xm1
#define vsll_vv_i32m2                           vsllvv_int32xm2
#define vsll_vv_i32m4                           vsllvv_int32xm4
#define vsll_vv_i32m8                           vsllvv_int32xm8
#define vsll_vx_i32m1                           vsllvx_int32xm1
#define vsll_vx_i32m2                           vsllvx_int32xm2
#define vsll_vx_i32m4                           vsllvx_int32xm4
#define vsll_vx_i32m8                           vsllvx_int32xm8
#define vsll_vv_i32m1_m(mask, merged, a, b, vl) vsllvv_mask_int32xm1(merged, a, b, mask, vl)
#define vsll_vv_i32m2_m(mask, merged, a, b, vl) vsllvv_mask_int32xm2(merged, a, b, mask, vl)
#define vsll_vv_i32m4_m(mask, merged, a, b, vl) vsllvv_mask_int32xm4(merged, a, b, mask, vl)
#define vsll_vv_i32m8_m(mask, merged, a, b, vl) vsllvv_mask_int32xm8(merged, a, b, mask, vl)
#define vsll_vx_i32m1_m(mask, merged, a, b, vl) vsllvx_mask_int32xm1(merged, a, b, mask, vl)
#define vsll_vx_i32m2_m(mask, merged, a, b, vl) vsllvx_mask_int32xm2(merged, a, b, mask, vl)
#define vsll_vx_i32m4_m(mask, merged, a, b, vl) vsllvx_mask_int32xm4(merged, a, b, mask, vl)
#define vsll_vx_i32m8_m(mask, merged, a, b, vl) vsllvx_mask_int32xm8(merged, a, b, mask, vl)

#define vsra_vv_i32m1                           vsravv_int32xm1
#define vsra_vv_i32m2                           vsravv_int32xm2
#define vsra_vv_i32m4                           vsravv_int32xm4
#define vsra_vv_i32m8                           vsravv_int32xm8
#define vsra_vx_i32m1                           vsravx_int32xm1
#define vsra_vx_i32m2                           vsravx_int32xm2
#define vsra_vx_i32m4                           vsravx_int32xm4
#define vsra_vx_i32m8                           vsravx_int32xm8
#define vsra_vv_i32m1_m(mask, merged, a, b, vl) vsravv_mask_int32xm1(merged, a, b, mask, vl)
#define vsra_vv_i32m2_m(mask, merged, a, b, vl) vsravv_mask_int32xm2(merged, a, b, mask, vl)
#define vsra_vv_i32m4_m(mask, merged, a, b, vl) vsravv_mask_int32xm4(merged, a, b, mask, vl)
#define vsra_vv_i32m8_m(mask, merged, a, b, vl) vsravv_mask_int32xm8(merged, a, b, mask, vl)
#define vsra_vx_i32m1_m(mask, merged, a, b, vl) vsravx_mask_int32xm1(merged, a, b, mask, vl)
#define vsra_vx_i32m2_m(mask, merged, a, b, vl) vsravx_mask_int32xm2(merged, a, b, mask, vl)
#define vsra_vx_i32m4_m(mask, merged, a, b, vl) vsravx_mask_int32xm4(merged, a, b, mask, vl)
#define vsra_vx_i32m8_m(mask, merged, a, b, vl) vsravx_mask_int32xm8(merged, a, b, mask, vl)

#define vand_vv_i32m1                           vandvv_int32xm1
#define vand_vv_i32m2                           vandvv_int32xm2
#define vand_vv_i32m4                           vandvv_int32xm4
#define vand_vv_i32m8                           vandvv_int32xm8
#define vand_vx_i32m1                           vandvx_int32xm1
#define vand_vx_i32m2                           vandvx_int32xm2
#define vand_vx_i32m4                           vandvx_int32xm4
#define vand_vx_i32m8                           vandvx_int32xm8
#define vand_vv_i32m1_m(mask, merged, a, b, vl) vandvv_mask_int32xm1(merged, a, b, mask, vl)
#define vand_vv_i32m2_m(mask, merged, a, b, vl) vandvv_mask_int32xm2(merged, a, b, mask, vl)
#define vand_vv_i32m4_m(mask, merged, a, b, vl) vandvv_mask_int32xm4(merged, a, b, mask, vl)
#define vand_vv_i32m8_m(mask, merged, a, b, vl) vandvv_mask_int32xm8(merged, a, b, mask, vl)
#define vand_vx_i32m1_m(mask, merged, a, b, vl) vandvx_mask_int32xm1(merged, a, b, mask, vl)
#define vand_vx_i32m2_m(mask, merged, a, b, vl) vandvx_mask_int32xm2(merged, a, b, mask, vl)
#define vand_vx_i32m4_m(mask, merged, a, b, vl) vandvx_mask_int32xm4(merged, a, b, mask, vl)
#define vand_vx_i32m8_m(mask, merged, a, b, vl) vandvx_mask_int32xm8(merged, a, b, mask, vl)

#define vor_vv_i32m1                           vorvv_int32xm1
#define vor_vv_i32m2                           vorvv_int32xm2
#define vor_vv_i32m4                           vorvv_int32xm4
#define vor_vv_i32m8                           vorvv_int32xm8
#define vor_vx_i32m1                           vorvx_int32xm1
#define vor_vx_i32m2                           vorvx_int32xm2
#define vor_vx_i32m4                           vorvx_int32xm4
#define vor_vx_i32m8                           vorvx_int32xm8
#define vor_vv_i32m1_m(mask, merged, a, b, vl) vorvv_mask_int32xm1(merged, a, b, mask, vl)
#define vor_vv_i32m2_m(mask, merged, a, b, vl) vorvv_mask_int32xm2(merged, a, b, mask, vl)
#define vor_vv_i32m4_m(mask, merged, a, b, vl) vorvv_mask_int32xm4(merged, a, b, mask, vl)
#define vor_vv_i32m8_m(mask, merged, a, b, vl) vorvv_mask_int32xm8(merged, a, b, mask, vl)
#define vor_vx_i32m1_m(mask, merged, a, b, vl) vorvx_mask_int32xm1(merged, a, b, mask, vl)
#define vor_vx_i32m2_m(mask, merged, a, b, vl) vorvx_mask_int32xm2(merged, a, b, mask, vl)
#define vor_vx_i32m4_m(mask, merged, a, b, vl) vorvx_mask_int32xm4(merged, a, b, mask, vl)
#define vor_vx_i32m8_m(mask, merged, a, b, vl) vorvx_mask_int32xm8(merged, a, b, mask, vl)

#define vreinterpret_v_f32m1_i32m1(x) reinterpret_cast<vint32m1_t>(x)
#define vreinterpret_v_f32m2_i32m2(x) reinterpret_cast<vint32m2_t>(x)
#define vreinterpret_v_f32m4_i32m4(x) reinterpret_cast<vint32m4_t>(x)
#define vreinterpret_v_f32m8_i32m8(x) reinterpret_cast<vint32m8_t>(x)

/******************************** int16 ********************************/
#define vadd_vv_i16m1                           vaddvv_int16xm1
#define vadd_vv_i16m2                           vaddvv_int16xm2
#define vadd_vv_i16m4                           vaddvv_int16xm4
#define vadd_vv_i16m8                           vaddvv_int16xm8
#define vadd_vx_i16m1                           vaddvx_int16xm1
#define vadd_vx_i16m2                           vaddvx_int16xm2
#define vadd_vx_i16m4                           vaddvx_int16xm4
#define vadd_vx_i16m8                           vaddvx_int16xm8
#define vadd_vv_i16m1_m(mask, merged, a, b, vl) vaddvv_mask_int16xm1(merged, a, b, mask, vl)
#define vadd_vv_i16m2_m(mask, merged, a, b, vl) vaddvv_mask_int16xm2(merged, a, b, mask, vl)
#define vadd_vv_i16m4_m(mask, merged, a, b, vl) vaddvv_mask_int16xm4(merged, a, b, mask, vl)
#define vadd_vv_i16m8_m(mask, merged, a, b, vl) vaddvv_mask_int16xm8(merged, a, b, mask, vl)
#define vadd_vx_i16m1_m(mask, merged, a, b, vl) vaddvx_mask_int16xm1(merged, a, b, mask, vl)
#define vadd_vx_i16m2_m(mask, merged, a, b, vl) vaddvx_mask_int16xm2(merged, a, b, mask, vl)
#define vadd_vx_i16m4_m(mask, merged, a, b, vl) vaddvx_mask_int16xm4(merged, a, b, mask, vl)
#define vadd_vx_i16m8_m(mask, merged, a, b, vl) vaddvx_mask_int16xm8(merged, a, b, mask, vl)

#define vsub_vv_i16m1                           vsubvv_int16xm1
#define vsub_vv_i16m2                           vsubvv_int16xm2
#define vsub_vv_i16m4                           vsubvv_int16xm4
#define vsub_vv_i16m8                           vsubvv_int16xm8
#define vsub_vx_i16m1                           vsubvx_int16xm1
#define vsub_vx_i16m2                           vsubvx_int16xm2
#define vsub_vx_i16m4                           vsubvx_int16xm4
#define vsub_vx_i16m8                           vsubvx_int16xm8
#define vsub_vv_i16m1_m(mask, merged, a, b, vl) vsubvv_mask_int16xm1(merged, a, b, mask, vl)
#define vsub_vv_i16m2_m(mask, merged, a, b, vl) vsubvv_mask_int16xm2(merged, a, b, mask, vl)
#define vsub_vv_i16m4_m(mask, merged, a, b, vl) vsubvv_mask_int16xm4(merged, a, b, mask, vl)
#define vsub_vv_i16m8_m(mask, merged, a, b, vl) vsubvv_mask_int16xm8(merged, a, b, mask, vl)
#define vsub_vx_i16m1_m(mask, merged, a, b, vl) vsubvx_mask_int16xm1(merged, a, b, mask, vl)
#define vsub_vx_i16m2_m(mask, merged, a, b, vl) vsubvx_mask_int16xm2(merged, a, b, mask, vl)
#define vsub_vx_i16m4_m(mask, merged, a, b, vl) vsubvx_mask_int16xm4(merged, a, b, mask, vl)
#define vsub_vx_i16m8_m(mask, merged, a, b, vl) vsubvx_mask_int16xm8(merged, a, b, mask, vl)

#define vsll_vv_i16m1                           vsllvv_int16xm1
#define vsll_vv_i16m2                           vsllvv_int16xm2
#define vsll_vv_i16m4                           vsllvv_int16xm4
#define vsll_vv_i16m8                           vsllvv_int16xm8
#define vsll_vx_i16m1                           vsllvx_int16xm1
#define vsll_vx_i16m2                           vsllvx_int16xm2
#define vsll_vx_i16m4                           vsllvx_int16xm4
#define vsll_vx_i16m8                           vsllvx_int16xm8
#define vsll_vv_i16m1_m(mask, merged, a, b, vl) vsllvv_mask_int16xm1(merged, a, b, mask, vl)
#define vsll_vv_i16m2_m(mask, merged, a, b, vl) vsllvv_mask_int16xm2(merged, a, b, mask, vl)
#define vsll_vv_i16m4_m(mask, merged, a, b, vl) vsllvv_mask_int16xm4(merged, a, b, mask, vl)
#define vsll_vv_i16m8_m(mask, merged, a, b, vl) vsllvv_mask_int16xm8(merged, a, b, mask, vl)
#define vsll_vx_i16m1_m(mask, merged, a, b, vl) vsllvx_mask_int16xm1(merged, a, b, mask, vl)
#define vsll_vx_i16m2_m(mask, merged, a, b, vl) vsllvx_mask_int16xm2(merged, a, b, mask, vl)
#define vsll_vx_i16m4_m(mask, merged, a, b, vl) vsllvx_mask_int16xm4(merged, a, b, mask, vl)
#define vsll_vx_i16m8_m(mask, merged, a, b, vl) vsllvx_mask_int16xm8(merged, a, b, mask, vl)

#define vsra_vv_i16m1                           vsravv_int16xm1
#define vsra_vv_i16m2                           vsravv_int16xm2
#define vsra_vv_i16m4                           vsravv_int16xm4
#define vsra_vv_i16m8                           vsravv_int16xm8
#define vsra_vx_i16m1                           vsravx_int16xm1
#define vsra_vx_i16m2                           vsravx_int16xm2
#define vsra_vx_i16m4                           vsravx_int16xm4
#define vsra_vx_i16m8                           vsravx_int16xm8
#define vsra_vv_i16m1_m(mask, merged, a, b, vl) vsravv_mask_int16xm1(merged, a, b, mask, vl)
#define vsra_vv_i16m2_m(mask, merged, a, b, vl) vsravv_mask_int16xm2(merged, a, b, mask, vl)
#define vsra_vv_i16m4_m(mask, merged, a, b, vl) vsravv_mask_int16xm4(merged, a, b, mask, vl)
#define vsra_vv_i16m8_m(mask, merged, a, b, vl) vsravv_mask_int16xm8(merged, a, b, mask, vl)
#define vsra_vx_i16m1_m(mask, merged, a, b, vl) vsravx_mask_int16xm1(merged, a, b, mask, vl)
#define vsra_vx_i16m2_m(mask, merged, a, b, vl) vsravx_mask_int16xm2(merged, a, b, mask, vl)
#define vsra_vx_i16m4_m(mask, merged, a, b, vl) vsravx_mask_int16xm4(merged, a, b, mask, vl)
#define vsra_vx_i16m8_m(mask, merged, a, b, vl) vsravx_mask_int16xm8(merged, a, b, mask, vl)

#define vand_vv_i16m1                           vandvv_int16xm1
#define vand_vv_i16m2                           vandvv_int16xm2
#define vand_vv_i16m4                           vandvv_int16xm4
#define vand_vv_i16m8                           vandvv_int16xm8
#define vand_vx_i16m1                           vandvx_int16xm1
#define vand_vx_i16m2                           vandvx_int16xm2
#define vand_vx_i16m4                           vandvx_int16xm4
#define vand_vx_i16m8                           vandvx_int16xm8
#define vand_vv_i16m1_m(mask, merged, a, b, vl) vandvv_mask_int16xm1(merged, a, b, mask, vl)
#define vand_vv_i16m2_m(mask, merged, a, b, vl) vandvv_mask_int16xm2(merged, a, b, mask, vl)
#define vand_vv_i16m4_m(mask, merged, a, b, vl) vandvv_mask_int16xm4(merged, a, b, mask, vl)
#define vand_vv_i16m8_m(mask, merged, a, b, vl) vandvv_mask_int16xm8(merged, a, b, mask, vl)
#define vand_vx_i16m1_m(mask, merged, a, b, vl) vandvx_mask_int16xm1(merged, a, b, mask, vl)
#define vand_vx_i16m2_m(mask, merged, a, b, vl) vandvx_mask_int16xm2(merged, a, b, mask, vl)
#define vand_vx_i16m4_m(mask, merged, a, b, vl) vandvx_mask_int16xm4(merged, a, b, mask, vl)
#define vand_vx_i16m8_m(mask, merged, a, b, vl) vandvx_mask_int16xm8(merged, a, b, mask, vl)

#define vor_vv_i16m1                           vorvv_int16xm1
#define vor_vv_i16m2                           vorvv_int16xm2
#define vor_vv_i16m4                           vorvv_int16xm4
#define vor_vv_i16m8                           vorvv_int16xm8
#define vor_vx_i16m1                           vorvx_int16xm1
#define vor_vx_i16m2                           vorvx_int16xm2
#define vor_vx_i16m4                           vorvx_int16xm4
#define vor_vx_i16m8                           vorvx_int16xm8
#define vor_vv_i16m1_m(mask, merged, a, b, vl) vorvv_mask_int16xm1(merged, a, b, mask, vl)
#define vor_vv_i16m2_m(mask, merged, a, b, vl) vorvv_mask_int16xm2(merged, a, b, mask, vl)
#define vor_vv_i16m4_m(mask, merged, a, b, vl) vorvv_mask_int16xm4(merged, a, b, mask, vl)
#define vor_vv_i16m8_m(mask, merged, a, b, vl) vorvv_mask_int16xm8(merged, a, b, mask, vl)
#define vor_vx_i16m1_m(mask, merged, a, b, vl) vorvx_mask_int16xm1(merged, a, b, mask, vl)
#define vor_vx_i16m2_m(mask, merged, a, b, vl) vorvx_mask_int16xm2(merged, a, b, mask, vl)
#define vor_vx_i16m4_m(mask, merged, a, b, vl) vorvx_mask_int16xm4(merged, a, b, mask, vl)
#define vor_vx_i16m8_m(mask, merged, a, b, vl) vorvx_mask_int16xm8(merged, a, b, mask, vl)

#define vreinterpret_v_f16m1_i16m1(x) reinterpret_cast<vint16m1_t>(x)
#define vreinterpret_v_f16m2_i16m2(x) reinterpret_cast<vint16m2_t>(x)
#define vreinterpret_v_f16m4_i16m4(x) reinterpret_cast<vint16m4_t>(x)
#define vreinterpret_v_f16m8_i16m8(x) reinterpret_cast<vint16m8_t>(x)

/******************************** int8 ********************************/
#define vle8_v_i8m1 vlev_int8xm1
#define vle8_v_i8m2 vlev_int8xm2
#define vle8_v_i8m4 vlev_int8xm4
#define vle8_v_i8m8 vlev_int8xm8

#define vse8_v_i8m1 vsev_int8xm1
#define vse8_v_i8m2 vsev_int8xm2
#define vse8_v_i8m4 vsev_int8xm4
#define vse8_v_i8m8 vsev_int8xm8

#define vlse8_v_i8m1 vlsev_int8xm1
#define vlse8_v_i8m2 vlsev_int8xm2
#define vlse8_v_i8m4 vlsev_int8xm4
#define vlse8_v_i8m8 vlsev_int8xm8

#define vsse8_v_i8m1 vssev_int8xm1
#define vsse8_v_i8m2 vssev_int8xm2
#define vsse8_v_i8m4 vssev_int8xm4
#define vsse8_v_i8m8 vssev_int8xm8

#define vmv_v_x_i8m1 vmvvx_int8xm1
#define vmv_v_x_i8m2 vmvvx_int8xm2
#define vmv_v_x_i8m4 vmvvx_int8xm4
#define vmv_v_x_i8m8 vmvvx_int8xm8

/******************************** uint32 ********************************/
#define vle32_v_u32m1 vlev_uint32xm1
#define vle32_v_u32m2 vlev_uint32xm2
#define vle32_v_u32m4 vlev_uint32xm4
#define vle32_v_u32m8 vlev_uint32xm8

#define vse32_v_u32m1 vsev_uint32xm1
#define vse32_v_u32m2 vsev_uint32xm2
#define vse32_v_u32m4 vsev_uint32xm4
#define vse32_v_u32m8 vsev_uint32xm8

#define vadd_vv_u32m1                           vaddvv_uint32xm1
#define vadd_vv_u32m2                           vaddvv_uint32xm2
#define vadd_vv_u32m4                           vaddvv_uint32xm4
#define vadd_vv_u32m8                           vaddvv_uint32xm8
#define vadd_vx_u32m1                           vaddvx_uint32xm1
#define vadd_vx_u32m2                           vaddvx_uint32xm2
#define vadd_vx_u32m4                           vaddvx_uint32xm4
#define vadd_vx_u32m8                           vaddvx_uint32xm8
#define vadd_vv_u32m1_m(mask, merged, a, b, vl) vaddvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vadd_vv_u32m2_m(mask, merged, a, b, vl) vaddvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vadd_vv_u32m4_m(mask, merged, a, b, vl) vaddvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vadd_vv_u32m8_m(mask, merged, a, b, vl) vaddvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vadd_vx_u32m1_m(mask, merged, a, b, vl) vaddvx_mask_uint32xm1(merged, a, b, mask, vl)
#define vadd_vx_u32m2_m(mask, merged, a, b, vl) vaddvx_mask_uint32xm2(merged, a, b, mask, vl)
#define vadd_vx_u32m4_m(mask, merged, a, b, vl) vaddvx_mask_uint32xm4(merged, a, b, mask, vl)
#define vadd_vx_u32m8_m(mask, merged, a, b, vl) vaddvx_mask_uint32xm8(merged, a, b, mask, vl)

#define vsub_vv_u32m1                           vsubvv_uint32xm1
#define vsub_vv_u32m2                           vsubvv_uint32xm2
#define vsub_vv_u32m4                           vsubvv_uint32xm4
#define vsub_vv_u32m8                           vsubvv_uint32xm8
#define vsub_vx_u32m1                           vsubvx_uint32xm1
#define vsub_vx_u32m2                           vsubvx_uint32xm2
#define vsub_vx_u32m4                           vsubvx_uint32xm4
#define vsub_vx_u32m8                           vsubvx_uint32xm8
#define vsub_vv_u32m1_m(mask, merged, a, b, vl) vsubvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vsub_vv_u32m2_m(mask, merged, a, b, vl) vsubvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vsub_vv_u32m4_m(mask, merged, a, b, vl) vsubvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vsub_vv_u32m8_m(mask, merged, a, b, vl) vsubvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vsub_vx_u32m1_m(mask, merged, a, b, vl) vsubvx_mask_uint32xm1(merged, a, b, mask, vl)
#define vsub_vx_u32m2_m(mask, merged, a, b, vl) vsubvx_mask_uint32xm2(merged, a, b, mask, vl)
#define vsub_vx_u32m4_m(mask, merged, a, b, vl) vsubvx_mask_uint32xm4(merged, a, b, mask, vl)
#define vsub_vx_u32m8_m(mask, merged, a, b, vl) vsubvx_mask_uint32xm8(merged, a, b, mask, vl)

#define vmul_vv_u32m1                           vmulvv_uint32xm1
#define vmul_vv_u32m2                           vmulvv_uint32xm2
#define vmul_vv_u32m4                           vmulvv_uint32xm4
#define vmul_vv_u32m8                           vmulvv_uint32xm8
#define vmul_vx_u32m1(a, b, vl)                 vmulvv_uint32xm1(a, vmvvx_uint32xm4(b, vl), vl)
#define vmul_vx_u32m2(a, b, vl)                 vmulvv_uint32xm2(a, vmvvx_uint32xm4(b, vl), vl)
#define vmul_vx_u32m4(a, b, vl)                 vmulvv_uint32xm4(a, vmvvx_uint32xm4(b, vl), vl)
#define vmul_vx_u32m8(a, b, vl)                 vmulvv_uint32xm8(a, vmvvx_uint32xm4(b, vl), vl)
#define vmul_vv_u32m1_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vmul_vv_u32m2_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vmul_vv_u32m4_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vmul_vv_u32m8_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vmul_vx_u32m1_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm1(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vmul_vx_u32m2_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm2(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vmul_vx_u32m4_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm4(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vmul_vx_u32m8_m(mask, merged, a, b, vl) vmulvv_mask_uint32xm8(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)

#define vdiv_vv_u32m1                           vdivvv_uint32xm1
#define vdiv_vv_u32m2                           vdivvv_uint32xm2
#define vdiv_vv_u32m4                           vdivvv_uint32xm4
#define vdiv_vv_u32m8                           vdivvv_uint32xm8
#define vdiv_vx_u32m1(a, b, vl)                 vdivvv_uint32xm1(a, vmvvx_uint32xm4(b, vl), vl)
#define vdiv_vx_u32m2(a, b, vl)                 vdivvv_uint32xm2(a, vmvvx_uint32xm4(b, vl), vl)
#define vdiv_vx_u32m4(a, b, vl)                 vdivvv_uint32xm4(a, vmvvx_uint32xm4(b, vl), vl)
#define vdiv_vx_u32m8(a, b, vl)                 vdivvv_uint32xm8(a, vmvvx_uint32xm4(b, vl), vl)
#define vdiv_vv_u32m1_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vdiv_vv_u32m2_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vdiv_vv_u32m4_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vdiv_vv_u32m8_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vdiv_vx_u32m1_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm1(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vdiv_vx_u32m2_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm2(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vdiv_vx_u32m4_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm4(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)
#define vdiv_vx_u32m8_m(mask, merged, a, b, vl) vdivvv_mask_uint32xm8(merged, a, vmvvx_uint32xm4(b, vl), mask, vl)

#define vsll_vv_u32m1                           vsllvv_uint32xm1
#define vsll_vv_u32m2                           vsllvv_uint32xm2
#define vsll_vv_u32m4                           vsllvv_uint32xm4
#define vsll_vv_u32m8                           vsllvv_uint32xm8
#define vsll_vx_u32m1                           vsllvx_uint32xm1
#define vsll_vx_u32m2                           vsllvx_uint32xm2
#define vsll_vx_u32m4                           vsllvx_uint32xm4
#define vsll_vx_u32m8                           vsllvx_uint32xm8
#define vsll_vv_u32m1_m(mask, merged, a, b, vl) vsllvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vsll_vv_u32m2_m(mask, merged, a, b, vl) vsllvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vsll_vv_u32m4_m(mask, merged, a, b, vl) vsllvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vsll_vv_u32m8_m(mask, merged, a, b, vl) vsllvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vsll_vx_u32m1_m(mask, merged, a, b, vl) vsllvx_mask_uint32xm1(merged, a, b, mask, vl)
#define vsll_vx_u32m2_m(mask, merged, a, b, vl) vsllvx_mask_uint32xm2(merged, a, b, mask, vl)
#define vsll_vx_u32m4_m(mask, merged, a, b, vl) vsllvx_mask_uint32xm4(merged, a, b, mask, vl)
#define vsll_vx_u32m8_m(mask, merged, a, b, vl) vsllvx_mask_uint32xm8(merged, a, b, mask, vl)

#define vsra_vv_u32m1                           vsravv_uint32xm1
#define vsra_vv_u32m2                           vsravv_uint32xm2
#define vsra_vv_u32m4                           vsravv_uint32xm4
#define vsra_vv_u32m8                           vsravv_uint32xm8
#define vsra_vx_u32m1                           vsravx_uint32xm1
#define vsra_vx_u32m2                           vsravx_uint32xm2
#define vsra_vx_u32m4                           vsravx_uint32xm4
#define vsra_vx_u32m8                           vsravx_uint32xm8
#define vsra_vv_u32m1_m(mask, merged, a, b, vl) vsravv_mask_uint32xm1(merged, a, b, mask, vl)
#define vsra_vv_u32m2_m(mask, merged, a, b, vl) vsravv_mask_uint32xm2(merged, a, b, mask, vl)
#define vsra_vv_u32m4_m(mask, merged, a, b, vl) vsravv_mask_uint32xm4(merged, a, b, mask, vl)
#define vsra_vv_u32m8_m(mask, merged, a, b, vl) vsravv_mask_uint32xm8(merged, a, b, mask, vl)
#define vsra_vx_u32m1_m(mask, merged, a, b, vl) vsravx_mask_uint32xm1(merged, a, b, mask, vl)
#define vsra_vx_u32m2_m(mask, merged, a, b, vl) vsravx_mask_uint32xm2(merged, a, b, mask, vl)
#define vsra_vx_u32m4_m(mask, merged, a, b, vl) vsravx_mask_uint32xm4(merged, a, b, mask, vl)
#define vsra_vx_u32m8_m(mask, merged, a, b, vl) vsravx_mask_uint32xm8(merged, a, b, mask, vl)

#define vand_vv_u32m1                           vandvv_uint32xm1
#define vand_vv_u32m2                           vandvv_uint32xm2
#define vand_vv_u32m4                           vandvv_uint32xm4
#define vand_vv_u32m8                           vandvv_uint32xm8
#define vand_vx_u32m1                           vandvx_uint32xm1
#define vand_vx_u32m2                           vandvx_uint32xm2
#define vand_vx_u32m4                           vandvx_uint32xm4
#define vand_vx_u32m8                           vandvx_uint32xm8
#define vand_vv_u32m1_m(mask, merged, a, b, vl) vandvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vand_vv_u32m2_m(mask, merged, a, b, vl) vandvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vand_vv_u32m4_m(mask, merged, a, b, vl) vandvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vand_vv_u32m8_m(mask, merged, a, b, vl) vandvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vand_vx_u32m1_m(mask, merged, a, b, vl) vandvx_mask_uint32xm1(merged, a, b, mask, vl)
#define vand_vx_u32m2_m(mask, merged, a, b, vl) vandvx_mask_uint32xm2(merged, a, b, mask, vl)
#define vand_vx_u32m4_m(mask, merged, a, b, vl) vandvx_mask_uint32xm4(merged, a, b, mask, vl)
#define vand_vx_u32m8_m(mask, merged, a, b, vl) vandvx_mask_uint32xm8(merged, a, b, mask, vl)

#define vor_vv_u32m1                           vorvv_uint32xm1
#define vor_vv_u32m2                           vorvv_uint32xm2
#define vor_vv_u32m4                           vorvv_uint32xm4
#define vor_vv_u32m8                           vorvv_uint32xm8
#define vor_vx_u32m1                           vorvx_uint32xm1
#define vor_vx_u32m2                           vorvx_uint32xm2
#define vor_vx_u32m4                           vorvx_uint32xm4
#define vor_vx_u32m8                           vorvx_uint32xm8
#define vor_vv_u32m1_m(mask, merged, a, b, vl) vorvv_mask_uint32xm1(merged, a, b, mask, vl)
#define vor_vv_u32m2_m(mask, merged, a, b, vl) vorvv_mask_uint32xm2(merged, a, b, mask, vl)
#define vor_vv_u32m4_m(mask, merged, a, b, vl) vorvv_mask_uint32xm4(merged, a, b, mask, vl)
#define vor_vv_u32m8_m(mask, merged, a, b, vl) vorvv_mask_uint32xm8(merged, a, b, mask, vl)
#define vor_vx_u32m1_m(mask, merged, a, b, vl) vorvx_mask_uint32xm1(merged, a, b, mask, vl)
#define vor_vx_u32m2_m(mask, merged, a, b, vl) vorvx_mask_uint32xm2(merged, a, b, mask, vl)
#define vor_vx_u32m4_m(mask, merged, a, b, vl) vorvx_mask_uint32xm4(merged, a, b, mask, vl)
#define vor_vx_u32m8_m(mask, merged, a, b, vl) vorvx_mask_uint32xm8(merged, a, b, mask, vl)

#define vmseq_vv_u32m1_b32 vmseqvv_e32xm1_uint32xm1
#define vmseq_vv_u32m2_b16 vmseqvv_e32xm2_uint32xm2
#define vmseq_vv_u32m4_b8  vmseqvv_e32xm4_uint32xm4
#define vmseq_vv_u32m8_b4  vmseqvv_e32xm8_uint32xm8
#define vmseq_vx_u32m1_b32 vmseqvx_e32xm1_uint32xm1
#define vmseq_vx_u32m2_b16 vmseqvx_e32xm2_uint32xm2
#define vmseq_vx_u32m4_b8  vmseqvx_e32xm4_uint32xm4
#define vmseq_vx_u32m8_b4  vmseqvx_e32xm8_uint32xm8

#define vmsne_vv_u32m1_b32 vmsnevv_e32xm1_uint32xm1
#define vmsne_vv_u32m2_b16 vmsnevv_e32xm2_uint32xm2
#define vmsne_vv_u32m4_b8  vmsnevv_e32xm4_uint32xm4
#define vmsne_vv_u32m8_b4  vmsnevv_e32xm8_uint32xm8
#define vmsne_vx_u32m1_b32 vmsnevx_e32xm1_uint32xm1
#define vmsne_vx_u32m2_b16 vmsnevx_e32xm2_uint32xm2
#define vmsne_vx_u32m4_b8  vmsnevx_e32xm4_uint32xm4
#define vmsne_vx_u32m8_b4  vmsnevx_e32xm8_uint32xm8

#define vmsgt_vv_u32m1_b32 vmsgtvv_e32xm1_uint32xm1
#define vmsgt_vv_u32m2_b16 vmsgtvv_e32xm2_uint32xm2
#define vmsgt_vv_u32m4_b8  vmsgtvv_e32xm4_uint32xm4
#define vmsgt_vv_u32m8_b4  vmsgtvv_e32xm8_uint32xm8
#define vmsgt_vx_u32m1_b32 vmsgtvx_e32xm1_uint32xm1
#define vmsgt_vx_u32m2_b16 vmsgtvx_e32xm2_uint32xm2
#define vmsgt_vx_u32m4_b8  vmsgtvx_e32xm4_uint32xm4
#define vmsgt_vx_u32m8_b4  vmsgtvx_e32xm8_uint32xm8

#define vmsge_vv_u32m1_b32 vmsgevv_e32xm1_uint32xm1
#define vmsge_vv_u32m2_b16 vmsgevv_e32xm2_uint32xm2
#define vmsge_vv_u32m4_b8  vmsgevv_e32xm4_uint32xm4
#define vmsge_vv_u32m8_b4  vmsgevv_e32xm8_uint32xm8
#define vmsge_vx_u32m1_b32 vmsgevx_e32xm1_uint32xm1
#define vmsge_vx_u32m2_b16 vmsgevx_e32xm2_uint32xm2
#define vmsge_vx_u32m4_b8  vmsgevx_e32xm4_uint32xm4
#define vmsge_vx_u32m8_b4  vmsgevx_e32xm8_uint32xm8

#define vmslt_vv_u32m1_b32 vmsltvv_e32xm1_uint32xm1
#define vmslt_vv_u32m2_b16 vmsltvv_e32xm2_uint32xm2
#define vmslt_vv_u32m4_b8  vmsltvv_e32xm4_uint32xm4
#define vmslt_vv_u32m8_b4  vmsltvv_e32xm8_uint32xm8
#define vmslt_vx_u32m1_b32 vmsltvx_e32xm1_uint32xm1
#define vmslt_vx_u32m2_b16 vmsltvx_e32xm2_uint32xm2
#define vmslt_vx_u32m4_b8  vmsltvx_e32xm4_uint32xm4
#define vmslt_vx_u32m8_b4  vmsltvx_e32xm8_uint32xm8

#define vmsle_vv_u32m1_b32 vmslevv_e32xm1_uint32xm1
#define vmsle_vv_u32m2_b16 vmslevv_e32xm2_uint32xm2
#define vmsle_vv_u32m4_b8  vmslevv_e32xm4_uint32xm4
#define vmsle_vv_u32m8_b4  vmslevv_e32xm8_uint32xm8
#define vmsle_vx_u32m1_b32 vmslevx_e32xm1_uint32xm1
#define vmsle_vx_u32m2_b16 vmslevx_e32xm2_uint32xm2
#define vmsle_vx_u32m4_b8  vmslevx_e32xm4_uint32xm4
#define vmsle_vx_u32m8_b4  vmslevx_e32xm8_uint32xm8

#define vreinterpret_v_f32m1_u32m1(x) reinterpret_cast<vuint32m1_t>(x)
#define vreinterpret_v_f32m2_u32m2(x) reinterpret_cast<vuint32m2_t>(x)
#define vreinterpret_v_f32m4_u32m4(x) reinterpret_cast<vuint32m4_t>(x)
#define vreinterpret_v_f32m8_u32m8(x) reinterpret_cast<vuint32m8_t>(x)

/******************************** uint16 ********************************/
#define vle16_v_u16m1 vlev_uint16xm1
#define vle16_v_u16m2 vlev_uint16xm2
#define vle16_v_u16m4 vlev_uint16xm4
#define vle16_v_u16m8 vlev_uint16xm8

#define vse16_v_u16m1 vsev_uint16xm1
#define vse16_v_u16m2 vsev_uint16xm2
#define vse16_v_u16m4 vsev_uint16xm4
#define vse16_v_u16m8 vsev_uint16xm8

#define vlse16_v_u16m1 vlsev_uint16xm1
#define vlse16_v_u16m2 vlsev_uint16xm2
#define vlse16_v_u16m4 vlsev_uint16xm4
#define vlse16_v_u16m8 vlsev_uint16xm8

#define vsse16_v_u16m1 vssev_uint16xm1
#define vsse16_v_u16m2 vssev_uint16xm2
#define vsse16_v_u16m4 vssev_uint16xm4
#define vsse16_v_u16m8 vssev_uint16xm8

#define vlseg8e16_v_u16m1x8 vlseg8ev_uint16x8xm1
#define vsseg8e16_v_u16m1x8 vsseg8ev_uint16x8xm1

#define vlseg4e16_v_u16m2x4 vlseg4ev_uint16x4xm2
#define vsseg4e16_v_u16m2x4 vsseg4ev_uint16x4xm2

#define vlseg4e16_v_u16m1x4 vlseg4ev_uint16x4xm1
#define vsseg4e16_v_u16m1x4 vsseg4ev_uint16x4xm1

#define vset_u16m2x4       vseg_element_set_uint16x4xm2
#define vget_u16m2x4_u16m2 vseg_element_get_uint16x4xm2

#define vset_u16m1x8       vseg_element_set_uint16x8xm1
#define vget_u16m1x8_u16m1 vseg_element_get_uint16x8xm1

#define vset_u16m1x4       vseg_element_set_uint16x4xm1
#define vget_u16m1x4_u16m1 vseg_element_get_uint16x4xm1

static inline vuint16m1x4_t vcreate_u16m1x4(vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3)
{
    vuint16m1x4_t p;
    p = vset_u16m1x4(p, 0, v0);
    p = vset_u16m1x4(p, 1, v1);
    p = vset_u16m1x4(p, 2, v2);
    p = vset_u16m1x4(p, 3, v3);
    return p;
}

static inline vuint16m2x4_t vcreate_u16m2x4(vuint16m2_t v0, vuint16m2_t v1, vuint16m2_t v2, vuint16m2_t v3)
{
    vuint16m2x4_t p;
    p = vset_u16m2x4(p, 0, v0);
    p = vset_u16m2x4(p, 1, v1);
    p = vset_u16m2x4(p, 2, v2);
    p = vset_u16m2x4(p, 3, v3);
    return p;
}

static inline vuint16m1x8_t vcreate_u16m1x8(vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, vuint16m1_t v4, vuint16m1_t v5, vuint16m1_t v6, vuint16m1_t v7)
{
    vuint16m1x8_t p;
    p = vset_u16m1x8(p, 0, v0);
    p = vset_u16m1x8(p, 1, v1);
    p = vset_u16m1x8(p, 2, v2);
    p = vset_u16m1x8(p, 3, v3);
    p = vset_u16m1x8(p, 4, v4);
    p = vset_u16m1x8(p, 5, v5);
    p = vset_u16m1x8(p, 6, v6);
    p = vset_u16m1x8(p, 7, v7);
    return p;
}

#define vmv_v_x_u16m1 vmvvx_uint16xm1
#define vmv_v_x_u16m2 vmvvx_uint16xm2
#define vmv_v_x_u16m4 vmvvx_uint16xm4
#define vmv_v_x_u16m8 vmvvx_uint16xm8

#define vadd_vv_u16m1                           vaddvv_uint16xm1
#define vadd_vv_u16m2                           vaddvv_uint16xm2
#define vadd_vv_u16m4                           vaddvv_uint16xm4
#define vadd_vv_u16m8                           vaddvv_uint16xm8
#define vadd_vx_u16m1                           vaddvx_uint16xm1
#define vadd_vx_u16m2                           vaddvx_uint16xm2
#define vadd_vx_u16m4                           vaddvx_uint16xm4
#define vadd_vx_u16m8                           vaddvx_uint16xm8
#define vadd_vv_u16m1_m(mask, merged, a, b, vl) vaddvv_mask_uint16xm1(merged, a, b, mask, vl)
#define vadd_vv_u16m2_m(mask, merged, a, b, vl) vaddvv_mask_uint16xm2(merged, a, b, mask, vl)
#define vadd_vv_u16m4_m(mask, merged, a, b, vl) vaddvv_mask_uint16xm4(merged, a, b, mask, vl)
#define vadd_vv_u16m8_m(mask, merged, a, b, vl) vaddvv_mask_uint16xm8(merged, a, b, mask, vl)
#define vadd_vx_u16m1_m(mask, merged, a, b, vl) vaddvx_mask_uint16xm1(merged, a, b, mask, vl)
#define vadd_vx_u16m2_m(mask, merged, a, b, vl) vaddvx_mask_uint16xm2(merged, a, b, mask, vl)
#define vadd_vx_u16m4_m(mask, merged, a, b, vl) vaddvx_mask_uint16xm4(merged, a, b, mask, vl)
#define vadd_vx_u16m8_m(mask, merged, a, b, vl) vaddvx_mask_uint16xm8(merged, a, b, mask, vl)

#define vsub_vv_u16m1                           vsubvv_uint16xm1
#define vsub_vv_u16m2                           vsubvv_uint16xm2
#define vsub_vv_u16m4                           vsubvv_uint16xm4
#define vsub_vv_u16m8                           vsubvv_uint16xm8
#define vsub_vx_u16m1                           vsubvx_uint16xm1
#define vsub_vx_u16m2                           vsubvx_uint16xm2
#define vsub_vx_u16m4                           vsubvx_uint16xm4
#define vsub_vx_u16m8                           vsubvx_uint16xm8
#define vsub_vv_u16m1_m(mask, merged, a, b, vl) vsubvv_mask_uint16xm1(merged, a, b, mask, vl)
#define vsub_vv_u16m2_m(mask, merged, a, b, vl) vsubvv_mask_uint16xm2(merged, a, b, mask, vl)
#define vsub_vv_u16m4_m(mask, merged, a, b, vl) vsubvv_mask_uint16xm4(merged, a, b, mask, vl)
#define vsub_vv_u16m8_m(mask, merged, a, b, vl) vsubvv_mask_uint16xm8(merged, a, b, mask, vl)
#define vsub_vx_u16m1_m(mask, merged, a, b, vl) vsubvx_mask_uint16xm1(merged, a, b, mask, vl)
#define vsub_vx_u16m2_m(mask, merged, a, b, vl) vsubvx_mask_uint16xm2(merged, a, b, mask, vl)
#define vsub_vx_u16m4_m(mask, merged, a, b, vl) vsubvx_mask_uint16xm4(merged, a, b, mask, vl)
#define vsub_vx_u16m8_m(mask, merged, a, b, vl) vsubvx_mask_uint16xm8(merged, a, b, mask, vl)

#define vsll_vv_u16m1                           vsllvv_uint16xm1
#define vsll_vv_u16m2                           vsllvv_uint16xm2
#define vsll_vv_u16m4                           vsllvv_uint16xm4
#define vsll_vv_u16m8                           vsllvv_uint16xm8
#define vsll_vx_u16m1                           vsllvx_uint16xm1
#define vsll_vx_u16m2                           vsllvx_uint16xm2
#define vsll_vx_u16m4                           vsllvx_uint16xm4
#define vsll_vx_u16m8                           vsllvx_uint16xm8
#define vsll_vv_u16m1_m(mask, merged, a, b, vl) vsllvv_mask_uint16xm1(merged, a, b, mask, vl)
#define vsll_vv_u16m2_m(mask, merged, a, b, vl) vsllvv_mask_uint16xm2(merged, a, b, mask, vl)
#define vsll_vv_u16m4_m(mask, merged, a, b, vl) vsllvv_mask_uint16xm4(merged, a, b, mask, vl)
#define vsll_vv_u16m8_m(mask, merged, a, b, vl) vsllvv_mask_uint16xm8(merged, a, b, mask, vl)
#define vsll_vx_u16m1_m(mask, merged, a, b, vl) vsllvx_mask_uint16xm1(merged, a, b, mask, vl)
#define vsll_vx_u16m2_m(mask, merged, a, b, vl) vsllvx_mask_uint16xm2(merged, a, b, mask, vl)
#define vsll_vx_u16m4_m(mask, merged, a, b, vl) vsllvx_mask_uint16xm4(merged, a, b, mask, vl)
#define vsll_vx_u16m8_m(mask, merged, a, b, vl) vsllvx_mask_uint16xm8(merged, a, b, mask, vl)

#define vsra_vv_u16m1                           vsravv_uint16xm1
#define vsra_vv_u16m2                           vsravv_uint16xm2
#define vsra_vv_u16m4                           vsravv_uint16xm4
#define vsra_vv_u16m8                           vsravv_uint16xm8
#define vsra_vx_u16m1                           vsravx_uint16xm1
#define vsra_vx_u16m2                           vsravx_uint16xm2
#define vsra_vx_u16m4                           vsravx_uint16xm4
#define vsra_vx_u16m8                           vsravx_uint16xm8
#define vsra_vv_u16m1_m(mask, merged, a, b, vl) vsravv_mask_uint16xm1(merged, a, b, mask, vl)
#define vsra_vv_u16m2_m(mask, merged, a, b, vl) vsravv_mask_uint16xm2(merged, a, b, mask, vl)
#define vsra_vv_u16m4_m(mask, merged, a, b, vl) vsravv_mask_uint16xm4(merged, a, b, mask, vl)
#define vsra_vv_u16m8_m(mask, merged, a, b, vl) vsravv_mask_uint16xm8(merged, a, b, mask, vl)
#define vsra_vx_u16m1_m(mask, merged, a, b, vl) vsravx_mask_uint16xm1(merged, a, b, mask, vl)
#define vsra_vx_u16m2_m(mask, merged, a, b, vl) vsravx_mask_uint16xm2(merged, a, b, mask, vl)
#define vsra_vx_u16m4_m(mask, merged, a, b, vl) vsravx_mask_uint16xm4(merged, a, b, mask, vl)
#define vsra_vx_u16m8_m(mask, merged, a, b, vl) vsravx_mask_uint16xm8(merged, a, b, mask, vl)

#define vand_vv_u16m1                           vandvv_uint16xm1
#define vand_vv_u16m2                           vandvv_uint16xm2
#define vand_vv_u16m4                           vandvv_uint16xm4
#define vand_vv_u16m8                           vandvv_uint16xm8
#define vand_vx_u16m1                           vandvx_uint16xm1
#define vand_vx_u16m2                           vandvx_uint16xm2
#define vand_vx_u16m4                           vandvx_uint16xm4
#define vand_vx_u16m8                           vandvx_uint16xm8
#define vand_vv_u16m1_m(mask, merged, a, b, vl) vandvv_mask_uint16xm1(merged, a, b, mask, vl)
#define vand_vv_u16m2_m(mask, merged, a, b, vl) vandvv_mask_uint16xm2(merged, a, b, mask, vl)
#define vand_vv_u16m4_m(mask, merged, a, b, vl) vandvv_mask_uint16xm4(merged, a, b, mask, vl)
#define vand_vv_u16m8_m(mask, merged, a, b, vl) vandvv_mask_uint16xm8(merged, a, b, mask, vl)
#define vand_vx_u16m1_m(mask, merged, a, b, vl) vandvx_mask_uint16xm1(merged, a, b, mask, vl)
#define vand_vx_u16m2_m(mask, merged, a, b, vl) vandvx_mask_uint16xm2(merged, a, b, mask, vl)
#define vand_vx_u16m4_m(mask, merged, a, b, vl) vandvx_mask_uint16xm4(merged, a, b, mask, vl)
#define vand_vx_u16m8_m(mask, merged, a, b, vl) vandvx_mask_uint16xm8(merged, a, b, mask, vl)

#define vor_vv_u16m1                           vorvv_uint16xm1
#define vor_vv_u16m2                           vorvv_uint16xm2
#define vor_vv_u16m4                           vorvv_uint16xm4
#define vor_vv_u16m8                           vorvv_uint16xm8
#define vor_vx_u16m1                           vorvx_uint16xm1
#define vor_vx_u16m2                           vorvx_uint16xm2
#define vor_vx_u16m4                           vorvx_uint16xm4
#define vor_vx_u16m8                           vorvx_uint16xm8
#define vor_vv_u16m1_m(mask, merged, a, b, vl) vorvv_mask_uint16xm1(merged, a, b, mask, vl)
#define vor_vv_u16m2_m(mask, merged, a, b, vl) vorvv_mask_uint16xm2(merged, a, b, mask, vl)
#define vor_vv_u16m4_m(mask, merged, a, b, vl) vorvv_mask_uint16xm4(merged, a, b, mask, vl)
#define vor_vv_u16m8_m(mask, merged, a, b, vl) vorvv_mask_uint16xm8(merged, a, b, mask, vl)
#define vor_vx_u16m1_m(mask, merged, a, b, vl) vorvx_mask_uint16xm1(merged, a, b, mask, vl)
#define vor_vx_u16m2_m(mask, merged, a, b, vl) vorvx_mask_uint16xm2(merged, a, b, mask, vl)
#define vor_vx_u16m4_m(mask, merged, a, b, vl) vorvx_mask_uint16xm4(merged, a, b, mask, vl)
#define vor_vx_u16m8_m(mask, merged, a, b, vl) vorvx_mask_uint16xm8(merged, a, b, mask, vl)

#define vmseq_vv_u16m1_b16 vmseqvv_e16xm1_uint16xm1
#define vmseq_vv_u16m2_b8  vmseqvv_e16xm2_uint16xm2
#define vmseq_vv_u16m4_b4  vmseqvv_e16xm4_uint16xm4
#define vmseq_vv_u16m8_b2  vmseqvv_e16xm8_uint16xm8
#define vmseq_vx_u16m1_b16 vmseqvx_e16xm1_uint16xm1
#define vmseq_vx_u16m2_b8  vmseqvx_e16xm2_uint16xm2
#define vmseq_vx_u16m4_b4  vmseqvx_e16xm4_uint16xm4
#define vmseq_vx_u16m8_b2  vmseqvx_e16xm8_uint16xm8

#define vmsne_vv_u16m1_b16 vmsnevv_e16xm1_uint16xm1
#define vmsne_vv_u16m2_b8  vmsnevv_e16xm2_uint16xm2
#define vmsne_vv_u16m4_b4  vmsnevv_e16xm4_uint16xm4
#define vmsne_vv_u16m8_b2  vmsnevv_e16xm8_uint16xm8
#define vmsne_vx_u16m1_b16 vmsnevx_e16xm1_uint16xm1
#define vmsne_vx_u16m2_b8  vmsnevx_e16xm2_uint16xm2
#define vmsne_vx_u16m4_b4  vmsnevx_e16xm4_uint16xm4
#define vmsne_vx_u16m8_b2  vmsnevx_e16xm8_uint16xm8

#define vmsgt_vv_u16m1_b16 vmsgtvv_e16xm1_uint16xm1
#define vmsgt_vv_u16m2_b8  vmsgtvv_e16xm2_uint16xm2
#define vmsgt_vv_u16m4_b4  vmsgtvv_e16xm4_uint16xm4
#define vmsgt_vv_u16m8_b2  vmsgtvv_e16xm8_uint16xm8
#define vmsgt_vx_u16m1_b16 vmsgtvx_e16xm1_uint16xm1
#define vmsgt_vx_u16m2_b8  vmsgtvx_e16xm2_uint16xm2
#define vmsgt_vx_u16m4_b4  vmsgtvx_e16xm4_uint16xm4
#define vmsgt_vx_u16m8_b2  vmsgtvx_e16xm8_uint16xm8

#define vmsge_vv_u16m1_b16 vmsgevv_e16xm1_uint16xm1
#define vmsge_vv_u16m2_b8  vmsgevv_e16xm2_uint16xm2
#define vmsge_vv_u16m4_b4  vmsgevv_e16xm4_uint16xm4
#define vmsge_vv_u16m8_b2  vmsgevv_e16xm8_uint16xm8
#define vmsge_vx_u16m1_b16 vmsgevx_e16xm1_uint16xm1
#define vmsge_vx_u16m2_b8  vmsgevx_e16xm2_uint16xm2
#define vmsge_vx_u16m4_b4  vmsgevx_e16xm4_uint16xm4
#define vmsge_vx_u16m8_b2  vmsgevx_e16xm8_uint16xm8

#define vmslt_vv_u16m1_b16 vmsltvv_e16xm1_uint16xm1
#define vmslt_vv_u16m2_b8  vmsltvv_e16xm2_uint16xm2
#define vmslt_vv_u16m4_b4  vmsltvv_e16xm4_uint16xm4
#define vmslt_vv_u16m8_b2  vmsltvv_e16xm8_uint16xm8
#define vmslt_vx_u16m1_b16 vmsltvx_e16xm1_uint16xm1
#define vmslt_vx_u16m2_b8  vmsltvx_e16xm2_uint16xm2
#define vmslt_vx_u16m4_b4  vmsltvx_e16xm4_uint16xm4
#define vmslt_vx_u16m8_b2  vmsltvx_e16xm8_uint16xm8

#define vmsle_vv_u16m1_b16 vmslevv_e16xm1_uint16xm1
#define vmsle_vv_u16m2_b8  vmslevv_e16xm2_uint16xm2
#define vmsle_vv_u16m4_b4  vmslevv_e16xm4_uint16xm4
#define vmsle_vv_u16m8_b2  vmslevv_e16xm8_uint16xm8
#define vmsle_vx_u16m1_b16 vmslevx_e16xm1_uint16xm1
#define vmsle_vx_u16m2_b8  vmslevx_e16xm2_uint16xm2
#define vmsle_vx_u16m4_b4  vmslevx_e16xm4_uint16xm4
#define vmsle_vx_u16m8_b2  vmslevx_e16xm8_uint16xm8

#define vreinterpret_v_f16m1_u16m1(x) reinterpret_cast<vuint16m1_t>(x)
#define vreinterpret_v_f16m2_u16m2(x) reinterpret_cast<vuint16m2_t>(x)
#define vreinterpret_v_f16m4_u16m4(x) reinterpret_cast<vuint16m4_t>(x)
#define vreinterpret_v_f16m8_u16m8(x) reinterpret_cast<vuint16m8_t>(x)

/******************************** mask ********************************/
#define vmxor_mm_b32 vmxormm_e32xm1
#define vmxor_mm_b16 vmxormm_e32xm2
#define vmxor_mm_b8  vmxormm_e32xm4
#define vmxor_mm_b4  vmxormm_e32xm8
#define vmxor_mm_b2  vmxormm_e16xm8

#endif // __riscv_vector

#endif // RISCV_V_071_FIX_H
