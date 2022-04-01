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

#define __msa_ld_w(p, i)    (v4i32) _mx128_lu1q((void*)(p), i)
#define __msa_st_w(a, p, i) _mx128_su1q((v16i8)(a), p, i)

#define __msa_fill_w     _mx128_mfcpu_w
#define __msa_fill_w_f32 _mx128_mffpu_w

#define __msa_addv_w _mx128_add_w
#define __msa_subv_w _mx128_sub_w

#define __msa_and_v(a, b) _mx128_andv((v16i8)(a), (v16i8)(b))
#define __msa_or_v(a, b)  _mx128_orv((v16i8)(a), (v16i8)(b))

#define __msa_bsel_v(a, b, c) _mx128_bselv((v16i8)(a), (v16i8)(b), (v16i8)(c))

#define __msa_sll_w _mx128_sll_w
#define __msa_srl_w _mx128_srl_w

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

#define __msa_ilvl_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8)_mx128_mfcpu_w(0xf0f0))
#define __msa_ilvr_w(a, b) (v4i32) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8)_mx128_mfcpu_w(0x0f0f))
#define __msa_ilvl_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8)_mx128_mfcpu_w(0xff00))
#define __msa_ilvr_d(a, b) (v2i64) _mx128_shufv((v16i8)(a), (v16i8)(b), (v16i8)_mx128_mfcpu_w(0x00ff))

#define __msa_splati_w _mx128_repi_w

#endif // __mips_mxu2

#endif // MIPS_MXU2_FIX_H
