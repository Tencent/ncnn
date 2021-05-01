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

typedef float32xm8_t vfloat32m8_t;
typedef float32xm4_t vfloat32m4_t;
typedef float32xm2_t vfloat32m2_t;
typedef float32xm1_t vfloat32m1_t;

typedef float32x4xm2_t vfloat32m2x4_t;
typedef float32x4xm1_t vfloat32m1x4_t;

typedef uint16xm8_t vuint16m8_t;
typedef uint16xm4_t vuint16m4_t;
typedef uint16xm2_t vuint16m2_t;
typedef uint16xm1_t vuint16m1_t;

typedef uint16x8xm1_t vuint16m1x8_t;
typedef uint16x4xm2_t vuint16m2x4_t;
typedef uint16x4xm1_t vuint16m1x4_t;

#define vsetvl_e32m8(n) vsetvli(n, RVV_E32, RVV_M8)
#define vsetvl_e32m4(n) vsetvli(n, RVV_E32, RVV_M4)
#define vsetvl_e32m2(n) vsetvli(n, RVV_E32, RVV_M2)
#define vsetvl_e32m1(n) vsetvli(n, RVV_E32, RVV_M1)

#define vsetvl_e16m2(n) vsetvli(n, RVV_E16, RVV_M2)
#define vsetvl_e16m1(n) vsetvli(n, RVV_E16, RVV_M1)

#define vle32_v_f32m8 vlev_float32xm8
#define vse32_v_f32m8 vsev_float32xm8

#define vle32_v_f32m4 vlev_float32xm4
#define vse32_v_f32m4 vsev_float32xm4

#define vle32_v_f32m2 vlev_float32xm2
#define vse32_v_f32m2 vsev_float32xm2

#define vle32_v_f32m1 vlev_float32xm1
#define vse32_v_f32m1 vsev_float32xm1

#define vlseg4e32_v_f32m2x4 vlseg4ev_float32x4xm2
#define vsseg4e32_v_f32m2x4 vsseg4ev_float32x4xm2

#define vlseg4e32_v_f32m1x4 vlseg4ev_float32x4xm1
#define vsseg4e32_v_f32m1x4 vsseg4ev_float32x4xm1

#define vle16_v_u16m8 vlev_uint16xm8
#define vse16_v_u16m8 vsev_uint16xm8

#define vle16_v_u16m4 vlev_uint16xm4
#define vse16_v_u16m4 vsev_uint16xm4

#define vle16_v_u16m2 vlev_uint16xm2
#define vse16_v_u16m2 vsev_uint16xm2

#define vle16_v_u16m1 vlev_uint16xm1
#define vse16_v_u16m1 vsev_uint16xm1

#define vlseg8e16_v_u16m1x8 vlseg8ev_uint16x8xm1
#define vsseg8e16_v_u16m1x8 vsseg8ev_uint16x8xm1

#define vlseg4e16_v_u16m2x4 vlseg4ev_uint16x4xm2
#define vsseg4e16_v_u16m2x4 vsseg4ev_uint16x4xm2

#define vlseg4e16_v_u16m1x4 vlseg4ev_uint16x4xm1
#define vsseg4e16_v_u16m1x4 vsseg4ev_uint16x4xm1

#define vset_f32m2x4       vseg_element_set_float32x4xm2
#define vget_f32m2x4_f32m2 vseg_element_get_float32x4xm2

#define vset_u16m2x4       vseg_element_set_uint16x4xm2
#define vget_u16m2x4_u16m2 vseg_element_get_uint16x4xm2

#define vset_u16m1x8       vseg_element_set_uint16x8xm1
#define vget_u16m1x8_u16m1 vseg_element_get_uint16x8xm1

#define vset_u16m1x4       vseg_element_set_uint16x4xm1
#define vget_u16m1x4_u16m1 vseg_element_get_uint16x4xm1

#define vfmax_vf_f32m8 vfmaxvf_float32xm8
#define vfmin_vf_f32m8 vfminvf_float32xm8

#endif // __riscv_vector

#endif // RISCV_V_071_FIX_H
