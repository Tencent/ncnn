// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_fp32_bf16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    const int A_elempack = A.elempack;
#if __riscv_vector
    if (A_elempack != 1 && A_elempack != packn)
#else
    if (A_elempack != 1)
#endif
    {
        float* pp = AT;
        int ii = 0;
#if __riscv_vector
        for (; ii + (packn - 1) < max_ii; ii += packn)
        {
            for (int kk = 0; kk < max_kk; kk++)
            {
                for (int iii = 0; iii < packn; iii++)
                {
                    const int row = i + ii + iii;
                    pp[iii] = ((const float*)A)[(row / A_elempack * A_elempack) * A_hstep + (k + kk) * A_elempack + row % A_elempack];
                }
                pp += packn;
            }
        }
#endif
        for (; ii + 1 < max_ii; ii += 2)
        {
            const int row0 = i + ii;
            const int row1 = i + ii + 1;
            const float* p0 = (const float*)A + (row0 / A_elempack * A_elempack) * A_hstep + k * A_elempack + row0 % A_elempack;
            const float* p1 = (const float*)A + (row1 / A_elempack * A_elempack) * A_hstep + k * A_elempack + row1 % A_elempack;
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0 += A_elempack;
                p1 += A_elempack;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const int row = i + ii;
            const float* p0 = (const float*)A + (row / A_elempack * A_elempack) * A_hstep + k * A_elempack + row % A_elempack;
            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp++;
                p0 += A_elempack;
            }
        }
        return;
    }

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
    const int elempack = A.elempack;

    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (elempack == packn)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * packn;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m2(pp, __riscv_vle32_v_f32m2(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse32_v_f32m2(pp, __riscv_vlse32_v_f32m2(p0, A_hstep * sizeof(float), vl), vl);
                pp += packn;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vfloat32m2_t v0 = __riscv_vle32_v_f32m2(p0, vl);
                vfloat32m2_t v1 = __riscv_vle32_v_f32m2(p1, vl);
                __riscv_vsseg2e32_v_f32m2x2(pp, __riscv_vcreate_v_f32m2x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse32_v_f32m2(pp, __riscv_vle32_v_f32m2(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void pack_A_tile_fp32_to_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vlse32_v_f32m2(p0, A_hstep * sizeof(float), vl), vl), vl);
            pp += packn;
            p0++;
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vle32_v_f32m2(p0, vl), vl), vl);
            pp += packn;
            p0 += A_hstep;
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_fp32_to_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const size_t vl8 = __riscv_vsetvl_e16m1(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
#endif

    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vlse32_v_f32m2(p0, B_hstep * sizeof(float), vl8), vl8), vl8);
            pp += 8;
            p0++;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vlse32_v_f32m2(p0, B_hstep * sizeof(float), vl4), vl4), vl4);
            pp += 4;
            p0++;
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const size_t vl8 = __riscv_vsetvl_e32m2(8);
    const size_t vl4 = __riscv_vsetvl_e32m2(4);
#endif

    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vle32_v_f32m2(p0, vl8), vl8), vl8);
            pp += 8;
            p0 += B_hstep;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_u16m1(pp, float2bfloat_rvv_roundaway(__riscv_vle32_v_f32m2(p0, vl4), vl4), vl4);
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp[1] = float32_to_bfloat16(p0[1]);
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_bfloat16(p0[0]);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFWMA && __riscv_vector && !__riscv_zvfbfwma && !__riscv_xtheadvector
void gemm_transB_packed_tile_bf16s_zvfbfwma(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end);
#endif

static void gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFWMA && __riscv_vector && !__riscv_zvfbfwma && !__riscv_xtheadvector
    if (cpu_support_riscv_zvfbfwma())
    {
        gemm_transB_packed_tile_bf16s_zvfbfwma(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        return;
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl2 = __riscv_vsetvl_e32m2(packn);
    const size_t vl8 = __riscv_vsetvl_e16m2(8);
    const size_t vl8w = __riscv_vsetvl_e32m4(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
    const size_t vl4w = __riscv_vsetvl_e32m2(4);
#endif

#if __riscv_vector
    const int out_elempack = top_blob.elempack;
#endif
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;

    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vfloat32m2_t _sum4;
            vfloat32m2_t _sum5;
            vfloat32m2_t _sum6;
            vfloat32m2_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = __riscv_vle32_v_f32m2(pC + packn, vl2);
                        _sum2 = __riscv_vle32_v_f32m2(pC + packn * 2, vl2);
                        _sum3 = __riscv_vle32_v_f32m2(pC + packn * 3, vl2);
                        _sum4 = __riscv_vle32_v_f32m2(pC + packn * 4, vl2);
                        _sum5 = __riscv_vle32_v_f32m2(pC + packn * 5, vl2);
                        _sum6 = __riscv_vle32_v_f32m2(pC + packn * 6, vl2);
                        _sum7 = __riscv_vle32_v_f32m2(pC + packn * 7, vl2);
                        pC += packn * 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = __riscv_vfmv_v_f_f32m2(pC[1], vl2);
                        _sum2 = __riscv_vfmv_v_f_f32m2(pC[2], vl2);
                        _sum3 = __riscv_vfmv_v_f_f32m2(pC[3], vl2);
                        _sum4 = __riscv_vfmv_v_f_f32m2(pC[4], vl2);
                        _sum5 = __riscv_vfmv_v_f_f32m2(pC[5], vl2);
                        _sum6 = __riscv_vfmv_v_f_f32m2(pC[6], vl2);
                        _sum7 = __riscv_vfmv_v_f_f32m2(pC[7], vl2);
                        pC += 8;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum2 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum3 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum4 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum5 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum6 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum7 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
                _sum1 = __riscv_vle32_v_f32m2(outptr + packn, vl2);
                _sum2 = __riscv_vle32_v_f32m2(outptr + packn * 2, vl2);
                _sum3 = __riscv_vle32_v_f32m2(outptr + packn * 3, vl2);
                _sum4 = __riscv_vle32_v_f32m2(outptr + packn * 4, vl2);
                _sum5 = __riscv_vle32_v_f32m2(outptr + packn * 5, vl2);
                _sum6 = __riscv_vle32_v_f32m2(outptr + packn * 6, vl2);
                _sum7 = __riscv_vle32_v_f32m2(outptr + packn * 7, vl2);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pA = __riscv_vle16_v_bf16m1((const __bf16*)pA, vl);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m2(_sum0, ((const __bf16*)pB)[0], _pA, vl);
                _sum1 = __riscv_vfwmaccbf16_vf_f32m2(_sum1, ((const __bf16*)pB)[1], _pA, vl);
                _sum2 = __riscv_vfwmaccbf16_vf_f32m2(_sum2, ((const __bf16*)pB)[2], _pA, vl);
                _sum3 = __riscv_vfwmaccbf16_vf_f32m2(_sum3, ((const __bf16*)pB)[3], _pA, vl);
                _sum4 = __riscv_vfwmaccbf16_vf_f32m2(_sum4, ((const __bf16*)pB)[4], _pA, vl);
                _sum5 = __riscv_vfwmaccbf16_vf_f32m2(_sum5, ((const __bf16*)pB)[5], _pA, vl);
                _sum6 = __riscv_vfwmaccbf16_vf_f32m2(_sum6, ((const __bf16*)pB)[6], _pA, vl);
                _sum7 = __riscv_vfwmaccbf16_vf_f32m2(_sum7, ((const __bf16*)pB)[7], _pA, vl);
#else
                vfloat32m2_t _pA = bfloat2float_rvv(__riscv_vle16_v_u16m1(pA, vl), vl);
                _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, bfloat16_to_float32(pB[0]), _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, bfloat16_to_float32(pB[1]), _pA, vl);
                _sum2 = __riscv_vfmacc_vf_f32m2(_sum2, bfloat16_to_float32(pB[2]), _pA, vl);
                _sum3 = __riscv_vfmacc_vf_f32m2(_sum3, bfloat16_to_float32(pB[3]), _pA, vl);
                _sum4 = __riscv_vfmacc_vf_f32m2(_sum4, bfloat16_to_float32(pB[4]), _pA, vl);
                _sum5 = __riscv_vfmacc_vf_f32m2(_sum5, bfloat16_to_float32(pB[5]), _pA, vl);
                _sum6 = __riscv_vfmacc_vf_f32m2(_sum6, bfloat16_to_float32(pB[6]), _pA, vl);
                _sum7 = __riscv_vfmacc_vf_f32m2(_sum7, bfloat16_to_float32(pB[7]), _pA, vl);
#endif
                pA += packn;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl2);
                _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl2);
                _sum2 = __riscv_vfmul_vf_f32m2(_sum2, alpha, vl2);
                _sum3 = __riscv_vfmul_vf_f32m2(_sum3, alpha, vl2);
                _sum4 = __riscv_vfmul_vf_f32m2(_sum4, alpha, vl2);
                _sum5 = __riscv_vfmul_vf_f32m2(_sum5, alpha, vl2);
                _sum6 = __riscv_vfmul_vf_f32m2(_sum6, alpha, vl2);
                _sum7 = __riscv_vfmul_vf_f32m2(_sum7, alpha, vl2);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum0, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn, float2bfloat_rvv_roundaway(_sum1, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 2, float2bfloat_rvv_roundaway(_sum2, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 3, float2bfloat_rvv_roundaway(_sum3, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 4, float2bfloat_rvv_roundaway(_sum4, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 5, float2bfloat_rvv_roundaway(_sum5, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 6, float2bfloat_rvv_roundaway(_sum6, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 7, float2bfloat_rvv_roundaway(_sum7, vl), vl);
                    outptr0 += packn * 8;
                }
                if (out_elempack == 1)
                {
                    vuint16m1x8_t _sum_bf16 = __riscv_vcreate_v_u16m1x8(
                                                  float2bfloat_rvv_roundaway(_sum0, vl),
                                                  float2bfloat_rvv_roundaway(_sum1, vl),
                                                  float2bfloat_rvv_roundaway(_sum2, vl),
                                                  float2bfloat_rvv_roundaway(_sum3, vl),
                                                  float2bfloat_rvv_roundaway(_sum4, vl),
                                                  float2bfloat_rvv_roundaway(_sum5, vl),
                                                  float2bfloat_rvv_roundaway(_sum6, vl),
                                                  float2bfloat_rvv_roundaway(_sum7, vl));
                    __riscv_vssseg8e16_v_u16m1x8(outptr0, out_hstep * sizeof(unsigned short), _sum_bf16, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum0, vl2);
                __riscv_vse32_v_f32m2(outptr + packn, _sum1, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 2, _sum2, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 3, _sum3, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 4, _sum4, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 5, _sum5, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 6, _sum6, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 7, _sum7, vl2);
            }

            outptr += packn * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = __riscv_vle32_v_f32m2(pC + packn, vl2);
                        _sum2 = __riscv_vle32_v_f32m2(pC + packn * 2, vl2);
                        _sum3 = __riscv_vle32_v_f32m2(pC + packn * 3, vl2);
                        pC += packn * 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = __riscv_vfmv_v_f_f32m2(pC[1], vl2);
                        _sum2 = __riscv_vfmv_v_f_f32m2(pC[2], vl2);
                        _sum3 = __riscv_vfmv_v_f_f32m2(pC[3], vl2);
                        pC += 4;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum2 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum3 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
                _sum1 = __riscv_vle32_v_f32m2(outptr + packn, vl2);
                _sum2 = __riscv_vle32_v_f32m2(outptr + packn * 2, vl2);
                _sum3 = __riscv_vle32_v_f32m2(outptr + packn * 3, vl2);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pA = __riscv_vle16_v_bf16m1((const __bf16*)pA, vl);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m2(_sum0, ((const __bf16*)pB)[0], _pA, vl);
                _sum1 = __riscv_vfwmaccbf16_vf_f32m2(_sum1, ((const __bf16*)pB)[1], _pA, vl);
                _sum2 = __riscv_vfwmaccbf16_vf_f32m2(_sum2, ((const __bf16*)pB)[2], _pA, vl);
                _sum3 = __riscv_vfwmaccbf16_vf_f32m2(_sum3, ((const __bf16*)pB)[3], _pA, vl);
#else
                vfloat32m2_t _pA = bfloat2float_rvv(__riscv_vle16_v_u16m1(pA, vl), vl);
                _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, bfloat16_to_float32(pB[0]), _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, bfloat16_to_float32(pB[1]), _pA, vl);
                _sum2 = __riscv_vfmacc_vf_f32m2(_sum2, bfloat16_to_float32(pB[2]), _pA, vl);
                _sum3 = __riscv_vfmacc_vf_f32m2(_sum3, bfloat16_to_float32(pB[3]), _pA, vl);
#endif
                pA += packn;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl2);
                _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl2);
                _sum2 = __riscv_vfmul_vf_f32m2(_sum2, alpha, vl2);
                _sum3 = __riscv_vfmul_vf_f32m2(_sum3, alpha, vl2);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum0, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn, float2bfloat_rvv_roundaway(_sum1, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 2, float2bfloat_rvv_roundaway(_sum2, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn * 3, float2bfloat_rvv_roundaway(_sum3, vl), vl);
                    outptr0 += packn * 4;
                }
                if (out_elempack == 1)
                {
                    vuint16m1x4_t _sum_bf16 = __riscv_vcreate_v_u16m1x4(
                                                  float2bfloat_rvv_roundaway(_sum0, vl),
                                                  float2bfloat_rvv_roundaway(_sum1, vl),
                                                  float2bfloat_rvv_roundaway(_sum2, vl),
                                                  float2bfloat_rvv_roundaway(_sum3, vl));
                    __riscv_vssseg4e16_v_u16m1x4(outptr0, out_hstep * sizeof(unsigned short), _sum_bf16, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum0, vl2);
                __riscv_vse32_v_f32m2(outptr + packn, _sum1, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 2, _sum2, vl2);
                __riscv_vse32_v_f32m2(outptr + packn * 3, _sum3, vl2);
            }

            outptr += packn * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = __riscv_vle32_v_f32m2(pC + packn, vl2);
                        pC += packn * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = __riscv_vfmv_v_f_f32m2(pC[1], vl2);
                        pC += 2;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
                _sum1 = __riscv_vle32_v_f32m2(outptr + packn, vl2);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pA = __riscv_vle16_v_bf16m1((const __bf16*)pA, vl);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m2(_sum0, ((const __bf16*)pB)[0], _pA, vl);
                _sum1 = __riscv_vfwmaccbf16_vf_f32m2(_sum1, ((const __bf16*)pB)[1], _pA, vl);
#else
                vfloat32m2_t _pA = bfloat2float_rvv(__riscv_vle16_v_u16m1(pA, vl), vl);
                _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, bfloat16_to_float32(pB[0]), _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, bfloat16_to_float32(pB[1]), _pA, vl);
#endif
                pA += packn;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl2);
                _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl2);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum0, vl), vl);
                    __riscv_vse16_v_u16m1(outptr0 + packn, float2bfloat_rvv_roundaway(_sum1, vl), vl);
                    outptr0 += packn * 2;
                }
                if (out_elempack == 1)
                {
                    vuint16m1x2_t _sum_bf16 = __riscv_vcreate_v_u16m1x2(
                                                  float2bfloat_rvv_roundaway(_sum0, vl),
                                                  float2bfloat_rvv_roundaway(_sum1, vl));
                    __riscv_vssseg2e16_v_u16m1x2(outptr0, out_hstep * sizeof(unsigned short), _sum_bf16, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum0, vl2);
                __riscv_vse32_v_f32m2(outptr + packn, _sum1, vl2);
            }

            outptr += packn * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m2_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        pC += packn;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        pC += 1;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pA = __riscv_vle16_v_bf16m1((const __bf16*)pA, vl);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m2(_sum0, ((const __bf16*)pB)[0], _pA, vl);
#else
                vfloat32m2_t _pA = bfloat2float_rvv(__riscv_vle16_v_u16m1(pA, vl), vl);
                _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, bfloat16_to_float32(pB[0]), _pA, vl);
#endif
                pA += packn;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl2);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum0, vl), vl);
                    outptr0 += packn;
                }
                if (out_elempack == 1)
                {
                    __riscv_vsse16_v_u16m1(outptr0, out_hstep * sizeof(unsigned short), float2bfloat_rvv_roundaway(_sum0, vl), vl);
                    outptr0++;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum0, vl2);
            }

            outptr += packn;
        }

        pAT += max_kk * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m4_t _sum0;
            vfloat32m4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m4(pC[0], vl8w);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m4(pC[0], vl8w);
                        _sum1 = __riscv_vfmv_v_f_f32m4(pC[1], vl8w);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m4x2_t _s0 = __riscv_vlseg2e32_v_f32m4x2(pC, vl8w);
                        _sum0 = __riscv_vget_v_f32m4x2_f32m4(_s0, 0);
                        _sum1 = __riscv_vget_v_f32m4x2_f32m4(_s0, 1);
                        pC += 8 * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle32_v_f32m4(pC, vl8w);
                        _sum1 = _sum0;
                        pC += 8;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m4(0.f, vl8w);
                    _sum1 = __riscv_vfmv_v_f_f32m4(0.f, vl8w);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m4(outptr, vl8w);
                _sum1 = __riscv_vle32_v_f32m4(outptr + 8, vl8w);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m2_t _pB = __riscv_vle16_v_bf16m2((const __bf16*)pB, vl8);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m4(_sum0, ((const __bf16*)pA)[0], _pB, vl8);
                _sum1 = __riscv_vfwmaccbf16_vf_f32m4(_sum1, ((const __bf16*)pA)[1], _pB, vl8);
#else
                vfloat32m4_t _pB = bfloat2float_rvv(__riscv_vle16_v_u16m2(pB, vl8), vl8);
                _sum0 = __riscv_vfmacc_vf_f32m4(_sum0, bfloat16_to_float32(pA[0]), _pB, vl8);
                _sum1 = __riscv_vfmacc_vf_f32m4(_sum1, bfloat16_to_float32(pA[1]), _pB, vl8);
#endif
                pA += 2;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m4(_sum0, alpha, vl8w);
                _sum1 = __riscv_vfmul_vf_f32m4(_sum1, alpha, vl8w);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_u16m2(outptr0, float2bfloat_rvv_roundaway(_sum0, vl8), vl8);
                    __riscv_vse16_v_u16m2(outptr0 + out_hstep, float2bfloat_rvv_roundaway(_sum1, vl8), vl8);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse32_v_f32m4(outptr, _sum0, vl8w);
                __riscv_vse32_v_f32m4(outptr + 8, _sum1, vl8w);
            }

            outptr += 8 * 2;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl4w);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl4w);
                        _sum1 = __riscv_vfmv_v_f_f32m2(pC[1], vl4w);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m2x2_t _s0 = __riscv_vlseg2e32_v_f32m2x2(pC, vl4w);
                        _sum0 = __riscv_vget_v_f32m2x2_f32m2(_s0, 0);
                        _sum1 = __riscv_vget_v_f32m2x2_f32m2(_s0, 1);
                        pC += 4 * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl4w);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl4w);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl4w);
                }
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl4w);
                _sum1 = __riscv_vle32_v_f32m2(outptr + 4, vl4w);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pB = __riscv_vle16_v_bf16m1((const __bf16*)pB, vl4);
                _sum0 = __riscv_vfwmaccbf16_vf_f32m2(_sum0, ((const __bf16*)pA)[0], _pB, vl4);
                _sum1 = __riscv_vfwmaccbf16_vf_f32m2(_sum1, ((const __bf16*)pA)[1], _pB, vl4);
#else
                vfloat32m2_t _pB = bfloat2float_rvv(__riscv_vle16_v_u16m1(pB, vl4), vl4);
                _sum0 = __riscv_vfmacc_vf_f32m2(_sum0, bfloat16_to_float32(pA[0]), _pB, vl4);
                _sum1 = __riscv_vfmacc_vf_f32m2(_sum1, bfloat16_to_float32(pA[1]), _pB, vl4);
#endif
                pA += 2;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl4w);
                _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl4w);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum0, vl4), vl4);
                    __riscv_vse16_v_u16m1(outptr0 + out_hstep, float2bfloat_rvv_roundaway(_sum1, vl4), vl4);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum0, vl4w);
                __riscv_vse32_v_f32m2(outptr + 4, _sum1, vl4w);
            }

            outptr += 4 * 2;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
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

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum01 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                sum10 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);
                sum11 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[1]);
                pA += 2;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum01 *= alpha;
                sum10 *= alpha;
                sum11 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum00);
                    outptr0[1] = float32_to_bfloat16(sum10);
                    outptr0[out_hstep] = float32_to_bfloat16(sum01);
                    outptr0[out_hstep + 1] = float32_to_bfloat16(sum11);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[out_hstep] = float32_to_bfloat16(sum1);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m4_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f32m4(pC[0], vl8w);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle32_v_f32m4(pC, vl8w);
                        pC += 8;
                    }
                }
                else
                {
                    _sum = __riscv_vfmv_v_f_f32m4(0.f, vl8w);
                }
            }
            else
            {
                _sum = __riscv_vle32_v_f32m4(outptr, vl8w);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m2_t _pB = __riscv_vle16_v_bf16m2((const __bf16*)pB, vl8);
                _sum = __riscv_vfwmaccbf16_vf_f32m4(_sum, ((const __bf16*)pA)[0], _pB, vl8);
#else
                vfloat32m4_t _pB = bfloat2float_rvv(__riscv_vle16_v_u16m2(pB, vl8), vl8);
                _sum = __riscv_vfmacc_vf_f32m4(_sum, bfloat16_to_float32(pA[0]), _pB, vl8);
#endif
                pA += 1;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f32m4(_sum, alpha, vl8w);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_u16m2(outptr0, float2bfloat_rvv_roundaway(_sum, vl8), vl8);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse32_v_f32m4(outptr, _sum, vl8w);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f32m2(pC[0], vl4w);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle32_v_f32m2(pC, vl4w);
                        pC += 4;
                    }
                }
                else
                {
                    _sum = __riscv_vfmv_v_f_f32m2(0.f, vl4w);
                }
            }
            else
            {
                _sum = __riscv_vle32_v_f32m2(outptr, vl4w);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __riscv_zvfbfwma
                vbfloat16m1_t _pB = __riscv_vle16_v_bf16m1((const __bf16*)pB, vl4);
                _sum = __riscv_vfwmaccbf16_vf_f32m2(_sum, ((const __bf16*)pA)[0], _pB, vl4);
#else
                vfloat32m2_t _pB = bfloat2float_rvv(__riscv_vle16_v_u16m1(pB, vl4), vl4);
                _sum = __riscv_vfmacc_vf_f32m2(_sum, bfloat16_to_float32(pA[0]), _pB, vl4);
#endif
                pA += 1;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f32m2(_sum, alpha, vl4w);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_u16m1(outptr0, float2bfloat_rvv_roundaway(_sum, vl4), vl4);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum, vl4w);
            }

            outptr += 4;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);
                pA += 1;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                sum0 *= alpha;
                sum1 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[1] = float32_to_bfloat16(sum1);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                sum = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                pA += 1;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                sum *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void unpack_output_tile_bf16s(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, int output_transpose)
{
    const int out_elempack = top_blob.elempack;
    const int out_elembits = top_blob.elembits();
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        for (int jj = 0; jj < max_jj; jj++)
        {
            if (output_transpose)
            {
                const int out_j = j + jj;
                const size_t vl = __riscv_vsetvl_e32m2(packn);
                vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp, vl);
                if (out_elembits == 16)
                {
                    unsigned short* p0 = (unsigned short*)top_blob + (out_j / out_elempack * out_elempack) * out_hstep + out_j % out_elempack + (i + ii) * out_elempack;
                    __riscv_vsse16_v_u16m1(p0, out_elempack * sizeof(unsigned short), float2bfloat_rvv_roundaway(_sum, vl), vl);
                }
                else
                {
                    float* p0 = (float*)top_blob + (out_j / out_elempack * out_elempack) * out_hstep + out_j % out_elempack + (i + ii) * out_elempack;
                    __riscv_vsse32_v_f32m2(p0, out_elempack * sizeof(float), _sum, vl);
                }
            }
            else if (out_elempack == 1)
            {
                const size_t vl = __riscv_vsetvl_e32m2(packn);
                vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp, vl);
                if (out_elembits == 16)
                {
                    unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j + jj;
                    __riscv_vsse16_v_u16m1(p0, out_hstep * sizeof(unsigned short), float2bfloat_rvv_roundaway(_sum, vl), vl);
                }
                else
                {
                    float* p0 = (float*)top_blob + (i + ii) * out_hstep + j + jj;
                    __riscv_vsse32_v_f32m2(p0, out_hstep * sizeof(float), _sum, vl);
                }
            }
            else
            {
                for (int iii = 0; iii < packn;)
                {
                    const int out_i = i + ii + iii;
                    const int n = std::min(packn - iii, out_elempack - out_i % out_elempack);
                    const size_t vl = __riscv_vsetvl_e32m2(n);
                    vfloat32m2_t _sum = __riscv_vle32_v_f32m2(pp + iii, vl);
                    if (out_elembits == 16)
                    {
                        unsigned short* p0 = (unsigned short*)top_blob + (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + (j + jj) * out_elempack;
                        __riscv_vse16_v_u16m1(p0, float2bfloat_rvv_roundaway(_sum, vl), vl);
                    }
                    else
                    {
                        float* p0 = (float*)top_blob + (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + (j + jj) * out_elempack;
                        __riscv_vse32_v_f32m2(p0, _sum, vl);
                    }
                    iii += n;
                }
            }
            pp += packn;
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        int jj = 0;
#if __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            for (int jjj = 0; jjj < 8; jjj++)
            {
                const int out_i = output_transpose ? j + jj + jjj : i + ii;
                const int out_j = output_transpose ? i + ii : j + jj + jjj;
                const int out_i1 = output_transpose ? out_i : out_i + 1;
                const int out_j1 = output_transpose ? out_j + 1 : out_j;
                const size_t offset0 = (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + out_j * out_elempack;
                const size_t offset1 = (out_i1 / out_elempack * out_elempack) * out_hstep + out_i1 % out_elempack + out_j1 * out_elempack;
                if (out_elembits == 16)
                {
                    ((unsigned short*)top_blob)[offset0] = float32_to_bfloat16(pp[jjj]);
                    ((unsigned short*)top_blob)[offset1] = float32_to_bfloat16(pp[8 + jjj]);
                }
                else
                {
                    ((float*)top_blob)[offset0] = pp[jjj];
                    ((float*)top_blob)[offset1] = pp[8 + jjj];
                }
            }
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            for (int jjj = 0; jjj < 4; jjj++)
            {
                const int out_i = output_transpose ? j + jj + jjj : i + ii;
                const int out_j = output_transpose ? i + ii : j + jj + jjj;
                const int out_i1 = output_transpose ? out_i : out_i + 1;
                const int out_j1 = output_transpose ? out_j + 1 : out_j;
                const size_t offset0 = (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + out_j * out_elempack;
                const size_t offset1 = (out_i1 / out_elempack * out_elempack) * out_hstep + out_i1 % out_elempack + out_j1 * out_elempack;
                if (out_elembits == 16)
                {
                    ((unsigned short*)top_blob)[offset0] = float32_to_bfloat16(pp[jjj]);
                    ((unsigned short*)top_blob)[offset1] = float32_to_bfloat16(pp[4 + jjj]);
                }
                else
                {
                    ((float*)top_blob)[offset0] = pp[jjj];
                    ((float*)top_blob)[offset1] = pp[4 + jjj];
                }
            }
            pp += 8;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
            for (int jjj = 0; jjj < 2; jjj++)
            {
                const int out_i = output_transpose ? j + jj + jjj : i + ii;
                const int out_j = output_transpose ? i + ii : j + jj + jjj;
                const int out_i1 = output_transpose ? out_i : out_i + 1;
                const int out_j1 = output_transpose ? out_j + 1 : out_j;
                const size_t offset0 = (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + out_j * out_elempack;
                const size_t offset1 = (out_i1 / out_elempack * out_elempack) * out_hstep + out_i1 % out_elempack + out_j1 * out_elempack;
                if (out_elembits == 16)
                {
                    ((unsigned short*)top_blob)[offset0] = float32_to_bfloat16(pp[jjj]);
                    ((unsigned short*)top_blob)[offset1] = float32_to_bfloat16(pp[2 + jjj]);
                }
                else
                {
                    ((float*)top_blob)[offset0] = pp[jjj];
                    ((float*)top_blob)[offset1] = pp[2 + jjj];
                }
            }
            pp += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            for (int jjj = 0; jjj < 1; jjj++)
            {
                const int out_i = output_transpose ? j + jj + jjj : i + ii;
                const int out_j = output_transpose ? i + ii : j + jj + jjj;
                const int out_i1 = output_transpose ? out_i : out_i + 1;
                const int out_j1 = output_transpose ? out_j + 1 : out_j;
                const size_t offset0 = (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + out_j * out_elempack;
                const size_t offset1 = (out_i1 / out_elempack * out_elempack) * out_hstep + out_i1 % out_elempack + out_j1 * out_elempack;
                if (out_elembits == 16)
                {
                    ((unsigned short*)top_blob)[offset0] = float32_to_bfloat16(pp[jjj]);
                    ((unsigned short*)top_blob)[offset1] = float32_to_bfloat16(pp[1 + jjj]);
                }
                else
                {
                    ((float*)top_blob)[offset0] = pp[jjj];
                    ((float*)top_blob)[offset1] = pp[1 + jjj];
                }
            }
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        for (int jj = 0; jj < max_jj; jj++)
        {
            const int out_i = output_transpose ? j + jj : i + ii;
            const int out_j = output_transpose ? i + ii : j + jj;
            const size_t offset = (out_i / out_elempack * out_elempack) * out_hstep + out_i % out_elempack + out_j * out_elempack;
            if (out_elembits == 16)
                ((unsigned short*)top_blob)[offset] = float32_to_bfloat16(pp[0]);
            else
                ((float*)top_blob)[offset] = pp[0];
            pp++;
        }
    }
}
