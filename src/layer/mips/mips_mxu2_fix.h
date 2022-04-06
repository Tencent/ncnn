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

#ifndef MIPS_MXU2_FIX_H
#define MIPS_MXU2_FIX_H

#if __mips_mxu2
#include <mxu2.h>

#define v4i32_w v4i32

#define __msa_ld_b(p, i) (v16i8) _mx128_lu1q((void*)(p), i)
#define __msa_ld_h(p, i) (v8i16) _mx128_lu1q((void*)(p), i)
#define __msa_ld_w(p, i) (v4i32) _mx128_lu1q((void*)(p), i)

#define __msa_st_b(a, p, i) _mx128_su1q((v16i8)(a), p, i)
#define __msa_st_h(a, p, i) _mx128_su1q((v16i8)(a), p, i)
#define __msa_st_w(a, p, i) _mx128_su1q((v16i8)(a), p, i)

#define __msa_fill_h     _mx128_mfcpu_h
#define __msa_fill_w     _mx128_mfcpu_w
#define __msa_fill_w_f32 _mx128_mffpu_w

#define __msa_addv_h _mx128_add_h
#define __msa_addv_w _mx128_add_w
#define __msa_subv_h _mx128_sub_h
#define __msa_subv_w _mx128_sub_w
#define __msa_mulv_h _mx128_mul_h
#define __msa_mulv_w _mx128_mul_w

#define __msa_maddv_w _mx128_madd_w

// only a==b supported
#define __msa_hadd_s_w(a, b) _mx128_dotps_w(a, (v8i16){1, 1, 1, 1, 1, 1, 1, 1})

#define __msa_and_v(a, b) _mx128_andv((v16i8)(a), (v16i8)(b))
#define __msa_or_v(a, b)  _mx128_orv((v16i8)(a), (v16i8)(b))

#define __msa_clti_s_b(a, i) _mx128_clts_b(a, _mx128_mfcpu_b(i))
#define __msa_clti_s_h(a, i) _mx128_clts_h(a, _mx128_mfcpu_h(i))

#define __msa_bsel_v(a, b, c) _mx128_bselv((v16i8)(a), (v16i8)(b), (v16i8)(c))

#define __msa_sll_h  _mx128_sll_h
#define __msa_sll_w  _mx128_sll_w
#define __msa_slli_h _mx128_slli_h
#define __msa_slli_w _mx128_slli_w
#define __msa_srl_h  _mx128_srl_h
#define __msa_srl_w  _mx128_srl_w
#define __msa_srli_h _mx128_srli_h
#define __msa_srli_w _mx128_srli_w

#define __msa_ffint_s_w  _mx128_vcvtssw
#define __msa_ftint_s_w  _mx128_vcvtsws
#define __msa_ftrunc_s_w _mx128_vtruncsws

#define __msa_fadd_w  _mx128_fadd_w
#define __msa_fsub_w  _mx128_fsub_w
#define __msa_fmul_w  _mx128_fmul_w
#define __msa_fdiv_w  _mx128_fdiv_w
#define __msa_fmax_w  _mx128_fmax_w
#define __msa_fmin_w  _mx128_fmin_w
#define __msa_fsqrt_w _mx128_fsqrt_w
#define __msa_fmadd_w _mx128_fmadd_w

#define __msa_frsqrt_w(a) _mx128_fdiv_w(_mx128_mffpu_w(1.f), _mx128_fsqrt_w(a))
#define __msa_frcp_w(a)   _mx128_fdiv_w(_mx128_mffpu_w(1.f), a)

#define __msa_fclt_w _mx128_fclt_w
#define __msa_fcle_w _mx128_fcle_w

#define __msa_fslt_w _mx128_fclt_w
#define __msa_fsle_w _mx128_fcle_w

#define __msa_bclri_w(a, i)     _mx128_andv((v16i8)(a), (v16i8)_mx128_mfcpu_w(~(1u << i)))
#define __msa_bnegi_w(a, i)     _mx128_bselv((v16i8)(a), _mx128_norv((v16i8)(a), (v16i8)(a)), (v16i8)_mx128_mfcpu_w(1u << i))
#define __msa_binsli_w(a, b, i) _mx128_bselv((v16i8)(a), (v16i8)(b), (v16i8)_mx128_mfcpu_w(1u << i))

#define __msa_ilvev_b(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29})
#define __msa_ilvev_h(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 1, 3, 8, 10, 9, 11, 16, 18, 17, 19, 24, 26, 25, 27})
#define __msa_ilvev_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 4, 6, 1, 3, 5, 7, 16, 18, 20, 22, 17, 19, 21, 23})
#define __msa_ilvev_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15})
#define __msa_ilvod_b(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31})
#define __msa_ilvod_h(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){4, 6, 5, 7, 12, 14, 13, 15, 20, 22, 21, 23, 28, 30, 29, 31})
#define __msa_ilvod_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){8, 10, 12, 14, 9, 11, 13, 15, 24, 26, 28, 30, 25, 27, 29, 31})
#define __msa_ilvod_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){16, 18, 20, 22, 24, 26, 28, 30, 17, 19, 21, 23, 25, 27, 29, 31})

#define __msa_ilvl_b(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
#define __msa_ilvl_h(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15})
#define __msa_ilvl_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15})
#define __msa_ilvl_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15})
#define __msa_ilvr_b(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
#define __msa_ilvr_h(a, b) (v8i16) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 30, 29, 31})
#define __msa_ilvr_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31})
#define __msa_ilvr_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8){16, 18, 20, 22, 24, 26, 28, 30, 17, 19, 21, 23, 25, 27, 29, 31})

#define __msa_splati_w _mx128_repi_w

#endif // __mips_mxu2

#endif // MIPS_MXU2_FIX_H
