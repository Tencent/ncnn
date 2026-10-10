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
#include <math.h>
#include <stdint.h>
#include <string.h>

// MSA is a source-level interface here; MXU2 never executes MSA instructions.
// Select this header before the compiler's <msa.h>.
#define v4i32_w v4i32

// MXU2 load/store offsets, like MSA offsets, are measured in bytes.
#define __msa_ld_b(p, i)    (v16i8) _mx128_lu1q((const void*)(p), (i))
#define __msa_ld_h(p, i)    (v8i16) _mx128_lu1q((const void*)(p), (i))
#define __msa_ld_w(p, i)    (v4i32) _mx128_lu1q((const void*)(p), (i))
#define __msa_ld_d(p, i)    (v2i64) _mx128_lu1q((const void*)(p), (i))
#define __msa_st_b(v, p, i) _mx128_su1q((v16i8)(v), (void*)(p), (i))
#define __msa_st_h(v, p, i) _mx128_su1q((v16i8)(v), (void*)(p), (i))
#define __msa_st_w(v, p, i) _mx128_su1q((v16i8)(v), (void*)(p), (i))
#define __msa_st_d(v, p, i) _mx128_su1q((v16i8)(v), (void*)(p), (i))

#define __msa_fill_b _mx128_mfcpu_b
#define __msa_fill_h _mx128_mfcpu_h
#define __msa_fill_w _mx128_mfcpu_w
#define __msa_fill_d _mx128_mfcpu_d

#define __msa_addv_b               _mx128_add_b
#define __msa_addv_h               _mx128_add_h
#define __msa_addv_w               _mx128_add_w
#define __msa_addv_d               _mx128_add_d
#define __msa_subv_b               _mx128_sub_b
#define __msa_subv_h               _mx128_sub_h
#define __msa_subv_w               _mx128_sub_w
#define __msa_subv_d               _mx128_sub_d
#define __msa_mulv_b               _mx128_mul_b
#define __msa_mulv_h               _mx128_mul_h
#define __msa_mulv_w               _mx128_mul_w
#define __msa_mulv_d               _mx128_mul_d
#define __msa_maddv_b              _mx128_madd_b
#define __msa_maddv_h              _mx128_madd_h
#define __msa_maddv_w              _mx128_madd_w
#define __msa_maddv_d              _mx128_madd_d
#define __msa_msubv_b              _mx128_msub_b
#define __msa_msubv_h              _mx128_msub_h
#define __msa_msubv_w              _mx128_msub_w
#define __msa_msubv_d              _mx128_msub_d
#define __msa_max_s_h              _mx128_maxs_h
#define __msa_maxi_s_h(a, i)       _mx128_maxs_h((a), _mx128_mfcpu_h(i))
#define __msa_sat_s_h              _mx128_sats_h
#define __msa_sat_s_w              _mx128_sats_w
#define __msa_dotp_s_h             _mx128_dotps_h
#define __msa_dotp_s_w             _mx128_dotps_w
#define __msa_dpadd_s_w(acc, a, b) _mx128_add_w((acc), _mx128_dotps_w((a), (b)))

#define __msa_and_v(a, b)        (v16u8) _mx128_andv((v16i8)(a), (v16i8)(b))
#define __msa_or_v(a, b)         (v16u8) _mx128_orv((v16i8)(a), (v16i8)(b))
#define __msa_xor_v(a, b)        (v16u8) _mx128_xorv((v16i8)(a), (v16i8)(b))
#define __msa_bsel_v(mask, a, b) (v16u8) _mx128_orv(_mx128_andv((v16i8)(mask), (v16i8)(b)), _mx128_andv(_mx128_norv((v16i8)(mask), (v16i8)(mask)), (v16i8)(a)))
#define __msa_bclri_w(a, i)      (v4u32) _mx128_andv((v16i8)(a), (v16i8)_mx128_mfcpu_w(~(1u << (i))))
#define __msa_bnegi_w(a, i)      (v4u32) _mx128_xorv((v16i8)(a), (v16i8)_mx128_mfcpu_w(1u << (i)))

static inline v4u32 ncnn_mxu2_binsli_w(v4u32 a, v4u32 b, int i)
{
    // MSA BINSLI copies the (i + 1) most significant bits, not one bit.
    const uint32_t mask = ~0u << (31 - i);
    v16i8 m = (v16i8)_mx128_mfcpu_w(mask);
    return (v4u32)_mx128_orv(_mx128_andv(m, (v16i8)b), _mx128_andv(_mx128_norv(m, m), (v16i8)a));
}
#define __msa_binsli_w(a, b, i) ncnn_mxu2_binsli_w((v4u32)(a), (v4u32)(b), (i))

#define __msa_ceqi_w(a, i)   _mx128_ceq_w((a), _mx128_mfcpu_w(i))
#define __msa_clti_s_b(a, i) _mx128_clts_b((a), _mx128_mfcpu_b(i))
#define __msa_clti_s_h(a, i) _mx128_clts_h((a), _mx128_mfcpu_h(i))

#define __msa_sll_h   _mx128_sll_h
#define __msa_sll_w   _mx128_sll_w
#define __msa_slli_h  _mx128_slli_h
#define __msa_slli_w  _mx128_slli_w
#define __msa_srl_h   _mx128_srl_h
#define __msa_srl_w   _mx128_srl_w
#define __msa_srli_h  _mx128_srli_h
#define __msa_srli_w  _mx128_srli_w
#define __msa_srlri_w _mx128_srlri_w
#define __msa_srai_b  _mx128_srai_b
#define __msa_srai_h  _mx128_srai_h
#define __msa_srai_w  _mx128_srai_w

#define __msa_ffint_s_w    _mx128_vcvtssw
#define __msa_ftint_s_w    _mx128_vcvtsws
#define __msa_ftrunc_s_w   _mx128_vtruncsws
#define __msa_fadd_w       _mx128_fadd_w
#define __msa_fsub_w       _mx128_fsub_w
#define __msa_fmul_w       _mx128_fmul_w
#define __msa_fdiv_w       _mx128_fdiv_w
#define __msa_fmax_w       _mx128_fmax_w
#define __msa_fmin_w       _mx128_fmin_w
#define __msa_fsqrt_w      _mx128_fsqrt_w
#define __msa_fmadd_w      _mx128_fmadd_w
#define __msa_fmadd_d      _mx128_fmadd_d
#define __msa_fmsub_w      _mx128_fmsub_w
#define __msa_fmsub_d      _mx128_fmsub_d
#define __msa_fclt_w       _mx128_fclt_w
#define __msa_fcle_w       _mx128_fcle_w
#define __msa_fceq_w       _mx128_fceq_w
#define __msa_fslt_w       _mx128_fclt_w
#define __msa_fsle_w       _mx128_fcle_w
#define __msa_frcp_w(a)    _mx128_fdiv_w(_mx128_mffpu_w(1.f), (a))
#define __msa_frsqrt_w(a)  _mx128_fdiv_w(_mx128_mffpu_w(1.f), _mx128_fsqrt_w(a))
#define __msa_fcne_w(a, b) (v4i32) _mx128_andv((v16i8)_mx128_fcor_w((a), (b)), _mx128_norv((v16i8)_mx128_fceq_w((a), (b)), (v16i8)_mx128_fceq_w((a), (b))))

#define __msa_insert_h(v, i, val) _mx128_insfcpu_h((v), (i), (val))
#define __msa_insert_w(v, i, val) _mx128_insfcpu_w((v), (i), (val))
#define __msa_copy_s_h(v, i)      _mx128_mtcpus_h((v), (i))
#define __msa_copy_s_w(v, i)      _mx128_mtcpus_w((v), (i))
#define __msa_copy_s_d(v, i)      _mx128_mtcpus_d((v), (i))
#define __msa_splati_w(v, i)      _mx128_repi_w((v), (i))

// MXU2 shufv takes the control vector FIRST. MSA interleaves use b before a.
// Each control byte selects one byte from concatenated (b, a), indices 0..31.
#define __msa_ilvr_b(a, b)  (v16i8) _mx128_shufv((v16i8){0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvr_h(a, b)  (v8i16) _mx128_shufv((v16i8){0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvr_w(a, b)  (v4i32) _mx128_shufv((v16i8){0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvr_d(a, b)  (v2i64) _mx128_shufv((v16i8){0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvl_b(a, b)  (v16i8) _mx128_shufv((v16i8){8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvl_h(a, b)  (v8i16) _mx128_shufv((v16i8){8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvl_w(a, b)  (v4i32) _mx128_shufv((v16i8){8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvl_d(a, b)  (v2i64) _mx128_shufv((v16i8){8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvev_b(a, b) (v16i8) _mx128_shufv((v16i8){0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvev_h(a, b) (v8i16) _mx128_shufv((v16i8){0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvev_w(a, b) (v4i32) _mx128_shufv((v16i8){0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvev_d(a, b) (v2i64) _mx128_shufv((v16i8){0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvod_b(a, b) (v16i8) _mx128_shufv((v16i8){1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvod_h(a, b) (v8i16) _mx128_shufv((v16i8){2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvod_w(a, b) (v4i32) _mx128_shufv((v16i8){4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_ilvod_d(a, b) (v2i64) _mx128_shufv((v16i8){8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_pckev_b(a, b) (v16i8) _mx128_shufv((v16i8){0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}, (v16i8)(b), (v16i8)(a))
#define __msa_pckev_h(a, b) (v8i16) _mx128_shufv((v16i8){0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29}, (v16i8)(b), (v16i8)(a))
#define __msa_pckev_w(a, b) (v4i32) _mx128_shufv((v16i8){0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27}, (v16i8)(b), (v16i8)(a))
#define __msa_pckev_d(a, b) (v2i64) _mx128_shufv((v16i8){0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23}, (v16i8)(b), (v16i8)(a))
#define __msa_pckod_b(a, b) (v16i8) _mx128_shufv((v16i8){1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_pckod_h(a, b) (v8i16) _mx128_shufv((v16i8){2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_pckod_w(a, b) (v4i32) _mx128_shufv((v16i8){4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))
#define __msa_pckod_d(a, b) (v2i64) _mx128_shufv((v16i8){8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31}, (v16i8)(b), (v16i8)(a))

// Dynamic and immediate MSA shuffles absent from MXU2 are expressed in lanes.
// They preserve MSA lane order; the compiler may lower them to scalar moves.
static inline v4i32 ncnn_mxu2_shf_w(v4i32 a, int imm)
{
    v4i32 r;
    for (int i = 0; i < 4; i++) r[i] = a[(imm >> (2 * i)) & 3];
    return r;
}
#define __msa_shf_w(a, imm) ncnn_mxu2_shf_w((v4i32)(a), (imm))

static inline v8i16 ncnn_mxu2_shf_h(v8i16 a, int imm)
{
    v8i16 r;
    for (int i = 0; i < 8; i++) r[i] = a[(i & 4) + ((imm >> (2 * (i & 3))) & 3)];
    return r;
}
#define __msa_shf_h(a, imm) ncnn_mxu2_shf_h((v8i16)(a), (imm))

static inline v16i8 ncnn_mxu2_sldi_b(v16i8 a, v16i8 b, int shift)
{
    v16i8 r;
    for (int i = 0; i < 16; i++)
    {
        int j = i + shift;
        r[i] = j < 16 ? b[j] : a[j - 16];
    }
    return r;
}
#define __msa_sldi_b(a, b, shift) ncnn_mxu2_sldi_b((v16i8)(a), (v16i8)(b), (shift))

static inline v4i32 ncnn_mxu2_vshf_w(v4i32 mask, v4i32 a, v4i32 b)
{
    v4i32 r;
    for (int i = 0; i < 4; i++)
    {
        unsigned int j = (unsigned int)mask[i] & 7u;
        r[i] = j < 4 ? b[j] : a[j - 4];
    }
    return r;
}
#define __msa_vshf_w(mask, a, b) ncnn_mxu2_vshf_w((v4i32)(mask), (v4i32)(a), (v4i32)(b))

static inline v8i16 ncnn_mxu2_vshf_h(v8i16 mask, v8i16 a, v8i16 b)
{
    v8i16 r;
    for (int i = 0; i < 8; i++)
    {
        unsigned int j = (unsigned short)mask[i] & 15u;
        r[i] = j < 8 ? b[j] : a[j - 8];
    }
    return r;
}
#define __msa_vshf_h(mask, a, b) ncnn_mxu2_vshf_h((v8i16)(mask), (v8i16)(a), (v8i16)(b))

static inline v8i16 ncnn_mxu2_mulhi_s_h(v8i16 a, v8i16 b)
{
    v8i16 r;
    for (int i = 0; i < 8; i++) r[i] = (int(a[i]) * int(b[i])) >> 16;
    return r;
}
#define __msa_mulhi_s_h(a, b) ncnn_mxu2_mulhi_s_h((v8i16)(a), (v8i16)(b))

static inline v4f32 ncnn_mxu2_frint_w(v4f32 a)
{
    v4f32 r;
    for (int i = 0; i < 4; i++) r[i] = nearbyintf(a[i]);
    return r;
}
#define __msa_frint_w(a) ncnn_mxu2_frint_w((v4f32)(a))

// FP16 pack/unpack fallback for MSA fexdo/fexupl/fexupr.
// Convert with round-to-nearest-even; these paths are not native MXU2 SIMD.
static inline float ncnn_mxu2_half_to_float(unsigned short h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t mant = h & 0x3ff;
    unsigned int exp = (h >> 10) & 31;
    uint32_t x;
    if (exp == 0)
    {
        if (mant == 0)
            x = sign;
        else
        {
            int e = -14;
            while ((mant & 0x400) == 0)
            {
                mant <<= 1;
                e--;
            }
            x = sign | ((uint32_t)(e + 127) << 23) | ((mant & 0x3ff) << 13);
        }
    }
    else if (exp == 31)
        x = sign | 0x7f800000u | (mant << 13);
    else
        x = sign | ((exp + 112u) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, sizeof(f));
    return f;
}

static inline unsigned short ncnn_mxu2_float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mant = x & 0x7fffff;
    unsigned int exp = (x >> 23) & 255;
    if (exp == 255)
        return (unsigned short)(sign | (mant ? (0x7e00 | (mant >> 13)) : 0x7c00));
    int e = (int)exp - 112;
    if (e >= 31) return (unsigned short)(sign | 0x7c00);
    if (e <= 0)
    {
        if (e < -10) return (unsigned short)sign;
        mant |= 0x800000;
        int shift = 14 - e;
        uint32_t q = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        q += rem > half || (rem == half && (q & 1u));
        return (unsigned short)(sign | q);
    }
    uint32_t q = mant >> 13;
    uint32_t rem = mant & 0x1fff;
    q |= (uint32_t)e << 10;
    q += rem > 0x1000 || (rem == 0x1000 && (q & 1u));
    return (unsigned short)(sign | q);
}

static inline v4f32 ncnn_mxu2_fexupl_w(v8i16 h)
{
    v4f32 r;
    for (int i = 0; i < 4; i++) r[i] = ncnn_mxu2_half_to_float((unsigned short)h[i]);
    return r;
}
static inline v4f32 ncnn_mxu2_fexupr_w(v8i16 h)
{
    v4f32 r;
    for (int i = 0; i < 4; i++) r[i] = ncnn_mxu2_half_to_float((unsigned short)h[i + 4]);
    return r;
}
static inline v8i16 ncnn_mxu2_fexdo_h(v4f32 a, v4f32 b)
{
    v8i16 r;
    for (int i = 0; i < 4; i++)
    {
        r[i] = (short)ncnn_mxu2_float_to_half(b[i]);
        r[i + 4] = (short)ncnn_mxu2_float_to_half(a[i]);
    }
    return r;
}
#define __msa_fexupl_w(a)   ncnn_mxu2_fexupl_w((v8i16)(a))
#define __msa_fexupr_w(a)   ncnn_mxu2_fexupr_w((v8i16)(a))
#define __msa_fexdo_h(a, b) ncnn_mxu2_fexdo_h((v4f32)(a), (v4f32)(b))

#endif // __mips_mxu2

#endif // MIPS_MXU2_FIX_H
