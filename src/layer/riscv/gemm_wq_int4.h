// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdint.h>

static inline unsigned char get_weight_wq_int4(const unsigned char* p, int k)
{
    return (p[k / 2] >> ((k & 1) * 4)) & 15;
}

static inline signed char get_packed_weight_wq_int4(const unsigned char* p, size_t index)
{
    return (signed char)(p[index / 2] << ((index & 1) ? 0 : 4) & 0xf0);
}

static inline uint16_t riscv_wq_int4_load_u16(const unsigned char* ptr)
{
    return (uint16_t)(unsigned char)get_packed_weight_wq_int4(ptr, 0)
           | (uint16_t)(unsigned char)get_packed_weight_wq_int4(ptr, 1) << 8;
}

static inline uint32_t riscv_wq_int4_load_u32(const unsigned char* ptr)
{
    return (uint32_t)(unsigned char)get_packed_weight_wq_int4(ptr, 0)
           | (uint32_t)(unsigned char)get_packed_weight_wq_int4(ptr, 1) << 8
           | (uint32_t)(unsigned char)get_packed_weight_wq_int4(ptr, 2) << 16
           | (uint32_t)(unsigned char)get_packed_weight_wq_int4(ptr, 3) << 24;
}

static inline uint64_t riscv_wq_int4_load_u64(const unsigned char* ptr)
{
    return (uint64_t)riscv_wq_int4_load_u32(ptr)
           | (uint64_t)riscv_wq_int4_load_u32(ptr + 2) << 32;
}

#if __riscv_vector
static inline vint8m1_t riscv_wq_int4_load(const unsigned char* ptr, size_t vl)
{
    const size_t vlp = (vl + 1) / 2;
    vuint8m1_t _p = __riscv_vle8_v_u8m1(ptr, vlp);
    vuint8m1_t _index = __riscv_vid_v_u8m1(vl);
    vuint8m1_t _q = __riscv_vrgather_vv_u8m1(_p, __riscv_vsrl_vx_u8m1(_index, 1, vl), vl);
    vuint8m1_t _lo = __riscv_vsll_vx_u8m1(_q, 4, vl);
    vuint8m1_t _hi = __riscv_vand_vx_u8m1(_q, 0xf0, vl);
    vbool8_t _odd = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(_index, 1, vl), 0, vl);
    _q = __riscv_vmerge_vvm_u8m1(_lo, _hi, _odd, vl);
    return __riscv_vreinterpret_v_u8m1_i8m1(_q);
}

#endif // __riscv_vector

// group-major, output-major within each K4/K1 fragment
static void pack_B_tile_wq_int4(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
#if __riscv_vector
    const size_t B_hstep = B.w;
    const ptrdiff_t B_stride = (ptrdiff_t)B_hstep;
    const size_t vl8 = __riscv_vsetvl_e8m1(8);
    const size_t vl4 = __riscv_vsetvl_e8m1(4);
    const size_t vl2 = __riscv_vsetvl_e8m1(2);
    const vuint8m1_t _pack_index4 = __riscv_vmul_vx_u8m1(__riscv_vid_v_u8m1(vl4), 2, vl4);
    const vuint8m1_t _pack_index2 = __riscv_vmul_vx_u8m1(__riscv_vid_v_u8m1(vl2), 2, vl2);
#endif // __riscv_vector
    const int block_count = (K + block_size - 1) / block_size;
    unsigned char* pp = BT_tile;
    float* pd = BT_descales_tile;

    int jj = 0;
#if __riscv_vector
    const bool use_nr8 = csrr_vlenb() >= 32;
    for (; use_nr8 && jj + 7 < max_jj; jj += 8)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);
        const float* ps4 = B_scales.row(j + jj + 4);
        const float* ps5 = B_scales.row(j + jj + 5);
        const float* ps6 = B_scales.row(j + jj + 6);
        const float* ps7 = B_scales.row(j + jj + 7);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl8);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl8);
                vuint8m1_t _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl8), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl8), 4, vl8), vl8);
                __riscv_vse8_v_u8m1(pp, __riscv_vrgather_vv_u8m1(_p, _pack_index4, vl4), vl4);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl8), __riscv_vand_vx_u8m1(_qn, 240, vl8), vl8);
                __riscv_vse8_v_u8m1(pp + 4, __riscv_vrgather_vv_u8m1(_p, _pack_index4, vl4), vl4);

                _q = __riscv_vlse8_v_u8m1(p0 + kk / 2 + 1, B_stride, vl8);
                _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl8);
                _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl8), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl8), 4, vl8), vl8);
                __riscv_vse8_v_u8m1(pp + 8, __riscv_vrgather_vv_u8m1(_p, _pack_index4, vl4), vl4);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl8), __riscv_vand_vx_u8m1(_qn, 240, vl8), vl8);
                __riscv_vse8_v_u8m1(pp + 12, __riscv_vrgather_vv_u8m1(_p, _pack_index4, vl4), vl4);
                pp += 16;
            }
            for (; kk < max_kk; kk++)
            {
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl8);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl8);
                vuint8m1_t _p;
                if (kk & 1)
                    _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl8), __riscv_vand_vx_u8m1(_qn, 240, vl8), vl8);
                else
                    _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl8), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl8), 4, vl8), vl8);
                __riscv_vse8_v_u8m1(pp, __riscv_vrgather_vv_u8m1(_p, _pack_index4, vl4), vl4);
                pp += 4;
            }
            const size_t consumed = ((size_t)max_kk + 1) / 2;
            p0 += consumed;
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
            *pd++ = (1.f / *ps2++) * 0.0625f;
            *pd++ = (1.f / *ps3++) * 0.0625f;
            *pd++ = (1.f / *ps4++) * 0.0625f;
            *pd++ = (1.f / *ps5++) * 0.0625f;
            *pd++ = (1.f / *ps6++) * 0.0625f;
            *pd++ = (1.f / *ps7++) * 0.0625f;
        }
    }
#endif // __riscv_vector
    for (; jj + 3 < max_jj; jj += 4)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
#if !__riscv_vector
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
        const unsigned char* p2 = B.row<const unsigned char>(j + jj + 2);
        const unsigned char* p3 = B.row<const unsigned char>(j + jj + 3);
#endif // !__riscv_vector
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);
        const float* ps2 = B_scales.row(j + jj + 2);
        const float* ps3 = B_scales.row(j + jj + 3);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl4);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl4);
                vuint8m1_t _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl4), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl4), 4, vl4), vl4);
                __riscv_vse8_v_u8m1(pp, __riscv_vrgather_vv_u8m1(_p, _pack_index2, vl2), vl2);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl4), __riscv_vand_vx_u8m1(_qn, 240, vl4), vl4);
                __riscv_vse8_v_u8m1(pp + 2, __riscv_vrgather_vv_u8m1(_p, _pack_index2, vl2), vl2);

                _q = __riscv_vlse8_v_u8m1(p0 + kk / 2 + 1, B_stride, vl4);
                _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl4);
                _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl4), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl4), 4, vl4), vl4);
                __riscv_vse8_v_u8m1(pp + 4, __riscv_vrgather_vv_u8m1(_p, _pack_index2, vl2), vl2);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl4), __riscv_vand_vx_u8m1(_qn, 240, vl4), vl4);
                __riscv_vse8_v_u8m1(pp + 6, __riscv_vrgather_vv_u8m1(_p, _pack_index2, vl2), vl2);
                pp += 8;
            }
#else
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[kk / 2];
                pp[1] = p0[kk / 2 + 1];
                pp[2] = p1[kk / 2];
                pp[3] = p1[kk / 2 + 1];
                pp[4] = p2[kk / 2];
                pp[5] = p2[kk / 2 + 1];
                pp[6] = p3[kk / 2];
                pp[7] = p3[kk / 2 + 1];
                pp += 8;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
#if __riscv_vector
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl4);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl4);
                vuint8m1_t _p;
                if (kk & 1)
                    _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl4), __riscv_vand_vx_u8m1(_qn, 240, vl4), vl4);
                else
                    _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl4), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl4), 4, vl4), vl4);
                __riscv_vse8_v_u8m1(pp, __riscv_vrgather_vv_u8m1(_p, _pack_index2, vl2), vl2);
#else
                pp[0] = get_weight_wq_int4(p0, kk) | get_weight_wq_int4(p1, kk) << 4;
                pp[1] = get_weight_wq_int4(p2, kk) | get_weight_wq_int4(p3, kk) << 4;
#endif // __riscv_vector
                pp += 2;
            }
            p0 += ((size_t)max_kk + 1) / 2;
#if !__riscv_vector
            p1 += ((size_t)max_kk + 1) / 2;
            p2 += ((size_t)max_kk + 1) / 2;
            p3 += ((size_t)max_kk + 1) / 2;
#endif // !__riscv_vector
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
            *pd++ = (1.f / *ps2++) * 0.0625f;
            *pd++ = (1.f / *ps3++) * 0.0625f;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
#if !__riscv_vector
        const unsigned char* p1 = B.row<const unsigned char>(j + jj + 1);
#endif // !__riscv_vector
        const float* ps0 = B_scales.row(j + jj);
        const float* ps1 = B_scales.row(j + jj + 1);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            int kk = 0;
#if __riscv_vector
            for (; kk + 3 < max_kk; kk += 4)
            {
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl2);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl2);
                vuint8m1_t _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl2), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl2), 4, vl2), vl2);
                pp[0] = __riscv_vmv_x_s_u8m1_u8(_p);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl2), __riscv_vand_vx_u8m1(_qn, 240, vl2), vl2);
                pp[1] = __riscv_vmv_x_s_u8m1_u8(_p);

                _q = __riscv_vlse8_v_u8m1(p0 + kk / 2 + 1, B_stride, vl2);
                _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl2);
                _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl2), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl2), 4, vl2), vl2);
                pp[2] = __riscv_vmv_x_s_u8m1_u8(_p);
                _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl2), __riscv_vand_vx_u8m1(_qn, 240, vl2), vl2);
                pp[3] = __riscv_vmv_x_s_u8m1_u8(_p);
                pp += 4;
            }
#else
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[kk / 2];
                pp[1] = p0[kk / 2 + 1];
                pp[2] = p1[kk / 2];
                pp[3] = p1[kk / 2 + 1];
                pp += 4;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
#if __riscv_vector
                vuint8m1_t _q = __riscv_vlse8_v_u8m1(p0 + kk / 2, B_stride, vl2);
                vuint8m1_t _qn = __riscv_vslidedown_vx_u8m1(_q, 1, vl2);
                vuint8m1_t _p;
                if (kk & 1)
                    _p = __riscv_vor_vv_u8m1(__riscv_vsrl_vx_u8m1(_q, 4, vl2), __riscv_vand_vx_u8m1(_qn, 240, vl2), vl2);
                else
                    _p = __riscv_vor_vv_u8m1(__riscv_vand_vx_u8m1(_q, 15, vl2), __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(_qn, 15, vl2), 4, vl2), vl2);
                *pp++ = __riscv_vmv_x_s_u8m1_u8(_p);
#else
                *pp++ = get_weight_wq_int4(p0, kk) | get_weight_wq_int4(p1, kk) << 4;
#endif // __riscv_vector
            }
            p0 += ((size_t)max_kk + 1) / 2;
#if !__riscv_vector
            p1 += ((size_t)max_kk + 1) / 2;
#endif // !__riscv_vector
            *pd++ = (1.f / *ps0++) * 0.0625f;
            *pd++ = (1.f / *ps1++) * 0.0625f;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const unsigned char* p0 = B.row<const unsigned char>(j + jj);
        const float* ps0 = B_scales.row(j + jj);
        for (int g = 0; g < block_count; g++)
        {
            const int max_kk = std::min(K - g * block_size, block_size);
            const int bytes = (max_kk + 1) / 2;
            for (int kk = 0; kk < bytes; kk++)
                *pp++ = *p0++;
            *pd++ = (1.f / *ps0++) * 0.0625f;
        }
    }
}

static void gemm_transB_packed_tile_wq_int4(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
    const signed char* pAT = AT_tile;
    const float* pAT_descales = AT_descales_tile;
    const unsigned char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;
    float* outptr = topT_tile;
    const int A_hstep = AT_tile.w;
    const int A_descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const int block_start = k / block_size;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const bool use_nr8 = csrr_vlenb() >= 32;
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;
        const size_t vl = __riscv_vsetvl_e32m1(packn);

        int jj = 0;
        for (; use_nr8 && jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            vfloat32m1_t _fsum2;
            vfloat32m1_t _fsum3;
            vfloat32m1_t _fsum4;
            vfloat32m1_t _fsum5;
            vfloat32m1_t _fsum6;
            vfloat32m1_t _fsum7;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                _fsum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _fsum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _fsum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _fsum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
                _fsum4 = __riscv_vle32_v_f32m1(outptr + packn * 4, vl);
                _fsum5 = __riscv_vle32_v_f32m1(outptr + packn * 5, vl);
                _fsum6 = __riscv_vle32_v_f32m1(outptr + packn * 6, vl);
                _fsum7 = __riscv_vle32_v_f32m1(outptr + packn * 7, vl);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum4 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum5 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum6 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum7 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint8m1_t _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint64_t b = riscv_wq_int4_load_u64(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)(b >> 32), _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b >> 40), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b >> 48), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b >> 56), _a, vl);
                    pA += packn;
                    pB += 4;

                    _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u64(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)(b >> 32), _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b >> 40), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b >> 48), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b >> 56), _a, vl);
                    pA += packn;
                    pB += 4;

                    _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u64(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)(b >> 32), _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b >> 40), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b >> 48), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b >> 56), _a, vl);
                    pA += packn;
                    pB += 4;

                    _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u64(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)(b >> 32), _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b >> 40), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b >> 48), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b >> 56), _a, vl);
                    pA += packn;
                    pB += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint8m1_t _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint32_t b0 = riscv_wq_int4_load_u32(pB);
                    uint32_t b1 = riscv_wq_int4_load_u32(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += packn;
                    pB += 4;

                    _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b0 = riscv_wq_int4_load_u32(pB);
                    b1 = riscv_wq_int4_load_u32(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += packn;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint8m1_t _a8 = __riscv_vle8_v_i8m1(pA, vl);
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(_a8, 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    const uint32_t b0 = riscv_wq_int4_load_u32(pB);
                    const uint32_t b1 = riscv_wq_int4_load_u32(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b0, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b0 >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b0 >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b0 >> 24), _a, vl);
                    _sum4 = __riscv_vmacc_vx_i32m1(_sum4, (signed char)b1, _a, vl);
                    _sum5 = __riscv_vmacc_vx_i32m1(_sum5, (signed char)(b1 >> 8), _a, vl);
                    _sum6 = __riscv_vmacc_vx_i32m1(_sum6, (signed char)(b1 >> 16), _a, vl);
                    _sum7 = __riscv_vmacc_vx_i32m1(_sum7, (signed char)(b1 >> 24), _a, vl);
                    pA += packn;
                    pB += 4;
                }

                vfloat32m1_t _descaleA = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _descaleA, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _descaleA, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum2, vl), _descaleA, vl);
                _fsum2 = __riscv_vfmacc_vf_f32m1(_fsum2, pB_descales[2], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum3, vl), _descaleA, vl);
                _fsum3 = __riscv_vfmacc_vf_f32m1(_fsum3, pB_descales[3], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum4, vl), _descaleA, vl);
                _fsum4 = __riscv_vfmacc_vf_f32m1(_fsum4, pB_descales[4], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum5, vl), _descaleA, vl);
                _fsum5 = __riscv_vfmacc_vf_f32m1(_fsum5, pB_descales[5], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum6, vl), _descaleA, vl);
                _fsum6 = __riscv_vfmacc_vf_f32m1(_fsum6, pB_descales[6], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum7, vl), _descaleA, vl);
                _fsum7 = __riscv_vfmacc_vf_f32m1(_fsum7, pB_descales[7], _v, vl);
                pA_descales += packn;
                pB_descales += 8;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + packn, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 3, _fsum3, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 4, _fsum4, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 5, _fsum5, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 6, _fsum6, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 7, _fsum7, vl);
            outptr += packn * 8;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            vfloat32m1_t _fsum2;
            vfloat32m1_t _fsum3;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                _fsum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _fsum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
                _fsum2 = __riscv_vle32_v_f32m1(outptr + packn * 2, vl);
                _fsum3 = __riscv_vle32_v_f32m1(outptr + packn * 3, vl);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum2 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum3 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint32_t b = riscv_wq_int4_load_u32(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u32(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u32(pB + 4);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u32(pB + 6);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += packn * 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    const uint32_t b = riscv_wq_int4_load_u32(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _sum2 = __riscv_vmacc_vx_i32m1(_sum2, (signed char)(b >> 16), _a, vl);
                    _sum3 = __riscv_vmacc_vx_i32m1(_sum3, (signed char)(b >> 24), _a, vl);
                    pA += packn;
                    pB += 2;
                }

                vfloat32m1_t _descaleA = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _descaleA, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _descaleA, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum2, vl), _descaleA, vl);
                _fsum2 = __riscv_vfmacc_vf_f32m1(_fsum2, pB_descales[2], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum3, vl), _descaleA, vl);
                _fsum3 = __riscv_vfmacc_vf_f32m1(_fsum3, pB_descales[3], _v, vl);
                pA_descales += packn;
                pB_descales += 4;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + packn, _fsum1, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 2, _fsum2, vl);
            __riscv_vse32_v_f32m1(outptr + packn * 3, _fsum3, vl);
            outptr += packn * 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                _fsum0 = __riscv_vle32_v_f32m1(outptr, vl);
                _fsum1 = __riscv_vle32_v_f32m1(outptr + packn, vl);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    uint16_t b = riscv_wq_int4_load_u16(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u16(pB + 1);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u16(pB + 2);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    b = riscv_wq_int4_load_u16(pB + 3);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += packn * 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    const uint16_t b = riscv_wq_int4_load_u16(pB);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, (signed char)b, _a, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, (signed char)(b >> 8), _a, vl);
                    pA += packn;
                    pB += 1;
                }

                vfloat32m1_t _descaleA = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), _descaleA, vl);
                _fsum0 = __riscv_vfmacc_vf_f32m1(_fsum0, pB_descales[0], _v, vl);
                _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), _descaleA, vl);
                _fsum1 = __riscv_vfmacc_vf_f32m1(_fsum1, pB_descales[1], _v, vl);
                pA_descales += packn;
                pB_descales += 2;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum0, vl);
            __riscv_vse32_v_f32m1(outptr + packn, _fsum1, vl);
            outptr += packn * 2;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const uint32_t b = (unsigned char)get_packed_weight_wq_int4(pB, 0) | ((unsigned char)get_packed_weight_wq_int4(pB, 1) << 8) | ((unsigned char)get_packed_weight_wq_int4(pB, 2) << 16) | ((uint32_t)(unsigned char)get_packed_weight_wq_int4(pB, 3) << 24);
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)b, _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)(b >> 8), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 2, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)(b >> 16), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn * 3, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, (signed char)(b >> 24), _a, vl);
                    pA += packn * 4;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, get_packed_weight_wq_int4(pB, 0), _a, vl);
                    _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA + packn, vl), 0, vl);
                    _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, get_packed_weight_wq_int4(pB, 1), _a, vl);
                    pA += packn * 2;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vint32m4_t _a32 = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m1_t _a = __riscv_vget_v_i32m4_i32m1(_a32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, get_packed_weight_wq_int4(pB, 0), _a, vl);
                    pA += packn;
                    pB++;
                }

                vfloat32m1_t _descaleA = __riscv_vle32_v_f32m1(pA_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vv_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), _descaleA, vl);
                _fsum = __riscv_vfmacc_vf_f32m1(_fsum, pB_descales[0], _v, vl);
                pA_descales += packn;
                pB_descales++;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += packn;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * packn;
        pAT_descales += A_descales_hstep * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __riscv_vector
        for (; use_nr8 && jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(8);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 8, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 12, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b, vl);
                    pA += 8;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB += 4;
                }

                vfloat32m1_t _descaleB = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _descaleB, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _descaleB, _v, vl);
                pA_descales += 2;
                pB_descales += 8;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
            outptr += 16;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 6, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB += 2;
                }

                vfloat32m1_t _descaleB = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _descaleB, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _descaleB, _v, vl);
                pA_descales += 2;
                pB_descales += 4;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
#else
            float sum00;
            float sum01;
            float sum02;
            float sum03;
            float sum10;
            float sum11;
            float sum12;
            float sum13;
            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum02 = 0.f;
                sum03 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;
                sum12 = 0.f;
                sum13 = 0.f;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
                sum02 = outptr[4];
                sum12 = outptr[5];
                sum03 = outptr[6];
                sum13 = outptr[7];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s00 = 0, s01 = 0, s02 = 0, s03 = 0;
                int s10 = 0, s11 = 0, s12 = 0, s13 = 0;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned char* b0 = pB;
                    const unsigned char* b1 = b0 + 2;
                    const unsigned char* b2 = b1 + 2;
                    const unsigned char* b3 = b2 + 2;
                    s00 += pA[0] * get_packed_weight_wq_int4(b0, 0) + pA[2] * get_packed_weight_wq_int4(b0, 1) + pA[4] * get_packed_weight_wq_int4(b0, 2) + pA[6] * get_packed_weight_wq_int4(b0, 3);
                    s01 += pA[0] * get_packed_weight_wq_int4(b1, 0) + pA[2] * get_packed_weight_wq_int4(b1, 1) + pA[4] * get_packed_weight_wq_int4(b1, 2) + pA[6] * get_packed_weight_wq_int4(b1, 3);
                    s02 += pA[0] * get_packed_weight_wq_int4(b2, 0) + pA[2] * get_packed_weight_wq_int4(b2, 1) + pA[4] * get_packed_weight_wq_int4(b2, 2) + pA[6] * get_packed_weight_wq_int4(b2, 3);
                    s03 += pA[0] * get_packed_weight_wq_int4(b3, 0) + pA[2] * get_packed_weight_wq_int4(b3, 1) + pA[4] * get_packed_weight_wq_int4(b3, 2) + pA[6] * get_packed_weight_wq_int4(b3, 3);
                    s10 += pA[1] * get_packed_weight_wq_int4(b0, 0) + pA[3] * get_packed_weight_wq_int4(b0, 1) + pA[5] * get_packed_weight_wq_int4(b0, 2) + pA[7] * get_packed_weight_wq_int4(b0, 3);
                    s11 += pA[1] * get_packed_weight_wq_int4(b1, 0) + pA[3] * get_packed_weight_wq_int4(b1, 1) + pA[5] * get_packed_weight_wq_int4(b1, 2) + pA[7] * get_packed_weight_wq_int4(b1, 3);
                    s12 += pA[1] * get_packed_weight_wq_int4(b2, 0) + pA[3] * get_packed_weight_wq_int4(b2, 1) + pA[5] * get_packed_weight_wq_int4(b2, 2) + pA[7] * get_packed_weight_wq_int4(b2, 3);
                    s13 += pA[1] * get_packed_weight_wq_int4(b3, 0) + pA[3] * get_packed_weight_wq_int4(b3, 1) + pA[5] * get_packed_weight_wq_int4(b3, 2) + pA[7] * get_packed_weight_wq_int4(b3, 3);
                    pA += 8;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    s00 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s01 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    s02 += pA[0] * get_packed_weight_wq_int4(pB, 2);
                    s03 += pA[0] * get_packed_weight_wq_int4(pB, 3);
                    s10 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    s11 += pA[1] * get_packed_weight_wq_int4(pB, 1);
                    s12 += pA[1] * get_packed_weight_wq_int4(pB, 2);
                    s13 += pA[1] * get_packed_weight_wq_int4(pB, 3);
                    pA += 2;
                    pB += 2;
                }

                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                const float* bd = pB_descales;
                sum00 += s00 * ad0 * bd[0];
                sum01 += s01 * ad0 * bd[1];
                sum02 += s02 * ad0 * bd[2];
                sum03 += s03 * ad0 * bd[3];
                sum10 += s10 * ad1 * bd[0];
                sum11 += s11 * ad1 * bd[1];
                sum12 += s12 * ad1 * bd[2];
                sum13 += s13 * ad1 * bd[3];
                pA_descales += 2;
                pB_descales += 4;
            }

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
            outptr[4] = sum02;
            outptr[5] = sum12;
            outptr[6] = sum03;
            outptr[7] = sum13;
#endif // __riscv_vector
            outptr += 8;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e32m1(2);
            vfloat32m1_t _fsum0;
            vfloat32m1_t _fsum1;
            if (k == 0)
            {
                _fsum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
                _fsum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            }
            else
            {
                vfloat32m1x2_t _s = __riscv_vlseg2e32_v_f32m1x2(outptr, vl);
                _fsum0 = __riscv_vget_v_f32m1x2_f32m1(_s, 0);
                _fsum1 = __riscv_vget_v_f32m1x2_f32m1(_s, 1);
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t _sum1 = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 1, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[2], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[3], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[4], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[5], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 3, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[6], _b0, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[7], _b0, vl);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum0 = __riscv_vmacc_vx_i32m1(_sum0, pA[0], _b, vl);
                    _sum1 = __riscv_vmacc_vx_i32m1(_sum1, pA[1], _b, vl);
                    pA += 2;
                    pB += 1;
                }

                vfloat32m1_t _descaleB = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum0, vl), pA_descales[0], vl);
                _fsum0 = __riscv_vfmacc_vv_f32m1(_fsum0, _descaleB, _v, vl);
                _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum1, vl), pA_descales[1], vl);
                _fsum1 = __riscv_vfmacc_vv_f32m1(_fsum1, _descaleB, _v, vl);
                pA_descales += 2;
                pB_descales += 2;
            }

            __riscv_vsseg2e32_v_f32m1x2(outptr, __riscv_vcreate_v_f32m1x2(_fsum0, _fsum1), vl);
#else
            float sum00;
            float sum01;
            float sum10;
            float sum11;
            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s00 = 0, s01 = 0, s10 = 0, s11 = 0;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned char* b0 = pB;
                    const unsigned char* b1 = b0 + 2;
                    s00 += pA[0] * get_packed_weight_wq_int4(b0, 0) + pA[2] * get_packed_weight_wq_int4(b0, 1) + pA[4] * get_packed_weight_wq_int4(b0, 2) + pA[6] * get_packed_weight_wq_int4(b0, 3);
                    s01 += pA[0] * get_packed_weight_wq_int4(b1, 0) + pA[2] * get_packed_weight_wq_int4(b1, 1) + pA[4] * get_packed_weight_wq_int4(b1, 2) + pA[6] * get_packed_weight_wq_int4(b1, 3);
                    s10 += pA[1] * get_packed_weight_wq_int4(b0, 0) + pA[3] * get_packed_weight_wq_int4(b0, 1) + pA[5] * get_packed_weight_wq_int4(b0, 2) + pA[7] * get_packed_weight_wq_int4(b0, 3);
                    s11 += pA[1] * get_packed_weight_wq_int4(b1, 0) + pA[3] * get_packed_weight_wq_int4(b1, 1) + pA[5] * get_packed_weight_wq_int4(b1, 2) + pA[7] * get_packed_weight_wq_int4(b1, 3);
                    pA += 8;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    s00 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s01 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    s10 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    s11 += pA[1] * get_packed_weight_wq_int4(pB, 1);
                    pA += 2;
                    pB += 1;
                }

                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                const float* bd = pB_descales;
                sum00 += s00 * ad0 * bd[0];
                sum01 += s01 * ad0 * bd[1];
                sum10 += s10 * ad1 * bd[0];
                sum11 += s11 * ad1 * bd[1];
                pA_descales += 2;
                pB_descales += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;
#endif // __riscv_vector
            outptr += 4;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            float sum0;
            float sum1;
            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                int s0 = 0;
                int s1 = 0;
#if __riscv_vector
                const int vlmax = packn * 4;
                const size_t vl = __riscv_vsetvl_e32m4(std::min(max_kk0, vlmax));
                vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                for (; kk + (int)vl <= max_kk0; kk += (int)vl)
                {
                    vint16m2_t _a016 = __riscv_vwadd_vx_i16m2(__riscv_vlse8_v_i8m1(pA, 2, vl), 0, vl);
                    vint16m2_t _a116 = __riscv_vwadd_vx_i16m2(__riscv_vlse8_v_i8m1(pA + 1, 2, vl), 0, vl);
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _a0 = __riscv_vwadd_vx_i32m4(_a016, 0, vl);
                    vint32m4_t _a1 = __riscv_vwadd_vx_i32m4(_a116, 0, vl);
                    vint32m4_t _b = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _sum0 = __riscv_vmacc_vv_i32m4(_sum0, _a0, _b, vl);
                    _sum1 = __riscv_vmacc_vv_i32m4(_sum1, _a1, _b, vl);
                    pA += vl * 2;
                    pB += (vl + 1) / 2;
                }
                vint32m1_t _zero = __riscv_vmv_v_x_i32m1(0, 1);
                s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(_sum0, _zero, vl));
                s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(_sum1, _zero, vl));
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    s0 += pA[2] * get_packed_weight_wq_int4(pB, 1);
                    s1 += pA[3] * get_packed_weight_wq_int4(pB, 1);
                    pA += 4;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    pA += 2;
                    pB++;
                }
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[2] * get_packed_weight_wq_int4(pB, 1) + pA[4] * get_packed_weight_wq_int4(pB, 2) + pA[6] * get_packed_weight_wq_int4(pB, 3);
                    s1 += pA[1] * get_packed_weight_wq_int4(pB, 0) + pA[3] * get_packed_weight_wq_int4(pB, 1) + pA[5] * get_packed_weight_wq_int4(pB, 2) + pA[7] * get_packed_weight_wq_int4(pB, 3);
                    pA += 8;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    s0 += pA[2] * get_packed_weight_wq_int4(pB, 1);
                    s1 += pA[3] * get_packed_weight_wq_int4(pB, 1);
                    pA += 4;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[1] * get_packed_weight_wq_int4(pB, 0);
                    pA += 2;
                    pB++;
                }
#endif // __riscv_vector

                const float bd = pB_descales[0];
                sum0 += s0 * pA_descales[0] * bd;
                sum1 += s1 * pA_descales[1] * bd;
                pA_descales += 2;
                pB_descales++;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep * 2;
        pAT_descales += A_descales_hstep * 2;
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned char* pB_panel = pBT;
        const float* pB_descales_panel = pBT_descales;

        int jj = 0;
#if __riscv_vector
        for (; use_nr8 && jj + 7 < max_jj; jj += 8)
        {
            const unsigned char* pB = pB_panel + (size_t)8 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)8 * block_start;
            const size_t vl = __riscv_vsetvl_e32m1(8);
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 8, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[2], _b, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 12, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[3], _b, vl);
                    pA += 4;
                    pB += 16;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 4;
                }

                vfloat32m1_t _descaleB = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _descaleB, _v, vl);
                pA_descales++;
                pB_descales += 8;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
            outptr += 8;
            pB_panel += ((size_t)8 * K + 1) / 2;
            pB_descales_panel += (size_t)8 * block_count;
        }
#endif // __riscv_vector
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned char* pB = pB_panel + (size_t)4 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)4 * block_start;
#if __riscv_vector
            const size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _fsum;
            if (k == 0)
                _fsum = __riscv_vfmv_v_f_f32m1(0.f, vl);
            else
                _fsum = __riscv_vle32_v_f32m1(outptr, vl);

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                vint32m1_t _sum = __riscv_vmv_v_x_i32m1(0, vl);

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 2, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[1], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 4, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[2], _b0, vl);
                    _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB + 6, vl), 0, vl);
                    _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    _b0 = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[3], _b0, vl);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    vint16m2_t _b16 = __riscv_vwadd_vx_i16m2(riscv_wq_int4_load(pB, vl), 0, vl);
                    vint32m4_t _b32 = __riscv_vwadd_vx_i32m4(_b16, 0, vl);
                    vint32m1_t _b = __riscv_vget_v_i32m4_i32m1(_b32, 0);
                    _sum = __riscv_vmacc_vx_i32m1(_sum, pA[0], _b, vl);
                    pA++;
                    pB += 2;
                }

                vfloat32m1_t _descaleB = __riscv_vle32_v_f32m1(pB_descales, vl);
                vfloat32m1_t _v = __riscv_vfmul_vf_f32m1(__riscv_vfcvt_f_x_v_f32m1(_sum, vl), pA_descales[0], vl);
                _fsum = __riscv_vfmacc_vv_f32m1(_fsum, _descaleB, _v, vl);
                pA_descales++;
                pB_descales += 4;
            }

            __riscv_vse32_v_f32m1(outptr, _fsum, vl);
#else
            float sum0;
            float sum1;
            float sum2;
            float sum3;
            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
                sum2 = 0.f;
                sum3 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
                sum2 = outptr[2];
                sum3 = outptr[3];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s0 = 0, s1 = 0, s2 = 0, s3 = 0;

                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned char* b0 = pB;
                    const unsigned char* b1 = b0 + 2;
                    const unsigned char* b2 = b1 + 2;
                    const unsigned char* b3 = b2 + 2;
                    s0 += pA[0] * get_packed_weight_wq_int4(b0, 0) + pA[1] * get_packed_weight_wq_int4(b0, 1) + pA[2] * get_packed_weight_wq_int4(b0, 2) + pA[3] * get_packed_weight_wq_int4(b0, 3);
                    s1 += pA[0] * get_packed_weight_wq_int4(b1, 0) + pA[1] * get_packed_weight_wq_int4(b1, 1) + pA[2] * get_packed_weight_wq_int4(b1, 2) + pA[3] * get_packed_weight_wq_int4(b1, 3);
                    s2 += pA[0] * get_packed_weight_wq_int4(b2, 0) + pA[1] * get_packed_weight_wq_int4(b2, 1) + pA[2] * get_packed_weight_wq_int4(b2, 2) + pA[3] * get_packed_weight_wq_int4(b2, 3);
                    s3 += pA[0] * get_packed_weight_wq_int4(b3, 0) + pA[1] * get_packed_weight_wq_int4(b3, 1) + pA[2] * get_packed_weight_wq_int4(b3, 2) + pA[3] * get_packed_weight_wq_int4(b3, 3);
                    pA += 4;
                    pB += 8;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    s2 += pA[0] * get_packed_weight_wq_int4(pB, 2);
                    s3 += pA[0] * get_packed_weight_wq_int4(pB, 3);
                    pA++;
                    pB += 2;
                }

                const float ad = pA_descales[0];
                const float* bd = pB_descales;
                sum0 += s0 * ad * bd[0];
                sum1 += s1 * ad * bd[1];
                sum2 += s2 * ad * bd[2];
                sum3 += s3 * ad * bd[3];
                pA_descales++;
                pB_descales += 4;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
#endif // __riscv_vector
            outptr += 4;
            pB_panel += ((size_t)4 * K + 1) / 2;
            pB_descales_panel += (size_t)4 * block_count;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned char* pB = pB_panel + (size_t)2 * k / 2;
            const float* pB_descales = pB_descales_panel + (size_t)2 * block_start;
            float sum0;
            float sum1;
            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int kk = 0;
                int s0 = 0;
                int s1 = 0;
#if __riscv_vector
                const int vlmax = packn * 4;
                const size_t vl = __riscv_vsetvl_e32m4(std::min(max_kk0, vlmax));
                vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                for (; kk + (int)vl <= max_kk0; kk += (int)vl)
                {
                    vint16m2_t _a16 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(pA, vl), 0, vl);
                    vuint8m1_t _q = __riscv_vle8_v_u8m1(pB, vl);
                    vuint8m1_t _q0 = __riscv_vsll_vx_u8m1(_q, 4, vl);
                    vuint8m1_t _q1 = __riscv_vand_vx_u8m1(_q, 0xf0, vl);
                    vint16m2_t _b016 = __riscv_vwadd_vx_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(_q0), 0, vl);
                    vint16m2_t _b116 = __riscv_vwadd_vx_i16m2(__riscv_vreinterpret_v_u8m1_i8m1(_q1), 0, vl);
                    vint32m4_t _a = __riscv_vwadd_vx_i32m4(_a16, 0, vl);
                    vint32m4_t _b0 = __riscv_vwadd_vx_i32m4(_b016, 0, vl);
                    vint32m4_t _b1 = __riscv_vwadd_vx_i32m4(_b116, 0, vl);
                    _sum0 = __riscv_vmacc_vv_i32m4(_sum0, _a, _b0, vl);
                    _sum1 = __riscv_vmacc_vv_i32m4(_sum1, _a, _b1, vl);
                    pA += vl;
                    pB += vl;
                }
                vint32m1_t _zero = __riscv_vmv_v_x_i32m1(0, 1);
                s0 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(_sum0, _zero, vl));
                s1 = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(_sum1, _zero, vl));
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    pA++;
                    pB += 1;
                }
#else
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    const unsigned char* b0 = pB;
                    const unsigned char* b1 = b0 + 2;
                    s0 += pA[0] * get_packed_weight_wq_int4(b0, 0) + pA[1] * get_packed_weight_wq_int4(b0, 1) + pA[2] * get_packed_weight_wq_int4(b0, 2) + pA[3] * get_packed_weight_wq_int4(b0, 3);
                    s1 += pA[0] * get_packed_weight_wq_int4(b1, 0) + pA[1] * get_packed_weight_wq_int4(b1, 1) + pA[2] * get_packed_weight_wq_int4(b1, 2) + pA[3] * get_packed_weight_wq_int4(b1, 3);
                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk0; kk++)
                {
                    s0 += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    s1 += pA[0] * get_packed_weight_wq_int4(pB, 1);
                    pA++;
                    pB += 1;
                }
#endif // __riscv_vector

                const float ad = pA_descales[0];
                sum0 += s0 * ad * pB_descales[0];
                sum1 += s1 * ad * pB_descales[1];
                pA_descales++;
                pB_descales += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
            pB_panel += ((size_t)2 * K + 1) / 2;
            pB_descales_panel += (size_t)2 * block_count;
        }
        for (; jj < max_jj; jj++)
        {
            const unsigned char* pB = pB_panel + k / 2;
            const float* pB_descales = pB_descales_panel + block_start;
            float sum;
            if (k == 0)
                sum = 0.f;
            else
                sum = outptr[0];

            const signed char* pA = pAT;
            const float* pA_descales = pAT_descales;
            for (int kk0 = 0; kk0 < max_kk; kk0 += block_size)
            {
                const int max_kk0 = std::min(max_kk - kk0, block_size);
                int s = 0;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    s += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1) + pA[2] * get_packed_weight_wq_int4(pB, 2) + pA[3] * get_packed_weight_wq_int4(pB, 3);
                    pA += 4;
                    pB += 2;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    s += pA[0] * get_packed_weight_wq_int4(pB, 0) + pA[1] * get_packed_weight_wq_int4(pB, 1);
                    pA += 2;
                    pB++;
                }
                for (; kk < max_kk0; kk++)
                {
                    s += pA[0] * get_packed_weight_wq_int4(pB, 0);
                    pA++;
                    pB++;
                }
                sum += s * pA_descales[0] * pB_descales[0];
                pA_descales++;
                pB_descales++;
            }

            outptr[0] = sum;
            outptr++;
            pB_panel += ((size_t)K + 1) / 2;
            pB_descales_panel += block_count;
        }

        pAT += A_hstep;
        pAT_descales += A_descales_hstep;
    }
}

static void get_optimal_tile_mnk_wq_int4(int M, int N, int K, int block_size, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (sizeof(signed char) + 0.5f + sizeof(float) + 8.f / block_size));
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const int nr = csrr_vlenb() >= 32 ? 8 : 4;
#else
    const int packn = 2;
    const int nr = 4;
#endif // __riscv_vector

    TILE_M = std::max(packn, tile_size / packn * packn);
    TILE_N = std::max(nr, tile_size / nr * nr);
    TILE_K = std::max(block_size, tile_size / block_size * block_size);
    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + block_size - 1) / block_size * block_size);
        TILE_K = std::min(TILE_K, K);

        if (nn_K == 1)
        {
            const float linear_footprint = (1.5f + 8.f / block_size) * TILE_K;
            tile_size = std::max(1, (int)((sqrtf(linear_footprint * linear_footprint + 16.f * l2_cache_size) - linear_footprint) / 8.f));
            TILE_M = std::max(packn, tile_size / packn * packn);
            TILE_N = std::max(nr, tile_size / nr * nr);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + packn - 1) / packn * packn);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + nr - 1) / nr * nr);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + packn - 1) / packn * packn);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + packn - 1) / packn * packn;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + nr - 1) / nr * nr;
    }
    if (constant_TILE_K > 0)
    {
        TILE_K = std::max(block_size, (constant_TILE_K + block_size - 1) / block_size * block_size);
        if (K > 0)
            TILE_K = std::min(TILE_K, K);
    }
}
