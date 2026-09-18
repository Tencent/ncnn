// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_A_tile_fp16s(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const int elempack = A.elempack;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    float* pp = AT;

    int ii = 0;
#if __riscv_vector
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

static void pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_f16m1(pp, __riscv_vfncvt_f_f_w_f16m1(__riscv_vlse32_v_f32m2(p0, A_hstep * sizeof(float), vl), vl), vl);
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
            pp[0] = (__fp16)p0[0];
            pp[1] = (__fp16)p1[0];
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
            pp[0] = (__fp16)p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_f16m1(pp, __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(p0, vl), vl), vl);
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
            pp[0] = (__fp16)p0[0];
            pp[1] = (__fp16)p0[1];
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
            pp[0] = (__fp16)p0[0];
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_f16m1(pp, __riscv_vfncvt_f_f_w_f16m1(__riscv_vlse32_v_f32m2(p0, B_hstep * sizeof(float), vl), vl), vl);
            pp += packn;
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
            pp[0] = (__fp16)p0[0];
            pp[1] = (__fp16)p1[0];
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
            pp[0] = (__fp16)p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + (packn - 1) < max_jj; jj += packn)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            __riscv_vse16_v_f16m1(pp, __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(p0, vl), vl), vl);
            pp += packn;
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
            pp[0] = (__fp16)p0[0];
            pp[1] = (__fp16)p0[1];
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
            pp[0] = (__fp16)p0[0];
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void transpose_unpack_output_tile_fp32_to_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e32m2(packn);
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (out_elempack == packn)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vsse16_v_f16m1(p0 + l, packn * sizeof(__fp16), __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(pp, vl), vl), vl);
                    pp += packn;
                }
                p0 += out_hstep * packn;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                __riscv_vse16_v_f16m1(p0, __riscv_vfncvt_f_f_w_f16m1(__riscv_vle32_v_f32m2(pp, vl), vl), vl);
                pp += packn;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                for (int l = 0; l < packn; l++)
                {
                    p0[l] = (__fp16)pp[l * 2 + 0];
                    p0[packn + l] = (__fp16)pp[l * 2 + 1];
                }
                pp += packn * 2;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = (__fp16)(pp[0]);
                p0[1] = (__fp16)(pp[1]);
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * packn;

            for (int jj = 0; jj + (packn - 1) < max_jj; jj += packn)
            {
                for (int l = 0; l < packn; l++)
                {
                    p0[l] = (__fp16)pp[l];
                }
                pp += packn;
                p0 += out_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = (__fp16)(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_fp16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl2 = __riscv_vsetvl_e32m2(packn);
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;

    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

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
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            if (packn == 16)
            {
                vfloat32m2_t _sum0;
                vfloat32m2_t _sum1;
                vfloat32m2_t _sum2;
                vfloat32m2_t _sum3;
                vfloat32m2_t _sum4;
                vfloat32m2_t _sum5;
                vfloat32m2_t _sum6;
                vfloat32m2_t _sum7;
                vfloat32m2_t _sum8;
                vfloat32m2_t _sum9;
                vfloat32m2_t _suma;
                vfloat32m2_t _sumb;
                vfloat32m2_t _sumc;
                vfloat32m2_t _sumd;
                vfloat32m2_t _sume;
                vfloat32m2_t _sumf;

                if (k == 0)
                {
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum2 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum3 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum4 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum5 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum6 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum7 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum8 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum9 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _suma = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sumb = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sumc = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sumd = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sume = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sumf = __riscv_vfmv_v_f_f32m2(0.f, vl2);

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
                            _sum8 = _sum0;
                            _sum9 = _sum0;
                            _suma = _sum0;
                            _sumb = _sum0;
                            _sumc = _sum0;
                            _sumd = _sum0;
                            _sume = _sum0;
                            _sumf = _sum0;
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
                            _sum8 = _sum0;
                            _sum9 = _sum0;
                            _suma = _sum0;
                            _sumb = _sum0;
                            _sumc = _sum0;
                            _sumd = _sum0;
                            _sume = _sum0;
                            _sumf = _sum0;
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
                            _sum8 = __riscv_vle32_v_f32m2(pC + packn * 8, vl2);
                            _sum9 = __riscv_vle32_v_f32m2(pC + packn * 9, vl2);
                            _suma = __riscv_vle32_v_f32m2(pC + packn * 10, vl2);
                            _sumb = __riscv_vle32_v_f32m2(pC + packn * 11, vl2);
                            _sumc = __riscv_vle32_v_f32m2(pC + packn * 12, vl2);
                            _sumd = __riscv_vle32_v_f32m2(pC + packn * 13, vl2);
                            _sume = __riscv_vle32_v_f32m2(pC + packn * 14, vl2);
                            _sumf = __riscv_vle32_v_f32m2(pC + packn * 15, vl2);
                            pC += packn * 16;
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
                            _sum8 = __riscv_vfmv_v_f_f32m2(pC[8], vl2);
                            _sum9 = __riscv_vfmv_v_f_f32m2(pC[9], vl2);
                            _suma = __riscv_vfmv_v_f_f32m2(pC[10], vl2);
                            _sumb = __riscv_vfmv_v_f_f32m2(pC[11], vl2);
                            _sumc = __riscv_vfmv_v_f_f32m2(pC[12], vl2);
                            _sumd = __riscv_vfmv_v_f_f32m2(pC[13], vl2);
                            _sume = __riscv_vfmv_v_f_f32m2(pC[14], vl2);
                            _sumf = __riscv_vfmv_v_f_f32m2(pC[15], vl2);
                            pC += 16;
                        }
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
                    _sum8 = __riscv_vle32_v_f32m2(outptr + packn * 8, vl2);
                    _sum9 = __riscv_vle32_v_f32m2(outptr + packn * 9, vl2);
                    _suma = __riscv_vle32_v_f32m2(outptr + packn * 10, vl2);
                    _sumb = __riscv_vle32_v_f32m2(outptr + packn * 11, vl2);
                    _sumc = __riscv_vle32_v_f32m2(outptr + packn * 12, vl2);
                    _sumd = __riscv_vle32_v_f32m2(outptr + packn * 13, vl2);
                    _sume = __riscv_vle32_v_f32m2(outptr + packn * 14, vl2);
                    _sumf = __riscv_vle32_v_f32m2(outptr + packn * 15, vl2);
                }

                const __fp16* pA = pAT;
                int kk = 0;
                for (; kk < max_kk; kk += 1)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                    _sum2 = __riscv_vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                    _sum3 = __riscv_vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                    _sum4 = __riscv_vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                    _sum5 = __riscv_vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                    _sum6 = __riscv_vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                    _sum7 = __riscv_vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);
                    _sum8 = __riscv_vfwmacc_vf_f32m2(_sum8, pB[8], _pA, vl);
                    _sum9 = __riscv_vfwmacc_vf_f32m2(_sum9, pB[9], _pA, vl);
                    _suma = __riscv_vfwmacc_vf_f32m2(_suma, pB[10], _pA, vl);
                    _sumb = __riscv_vfwmacc_vf_f32m2(_sumb, pB[11], _pA, vl);
                    _sumc = __riscv_vfwmacc_vf_f32m2(_sumc, pB[12], _pA, vl);
                    _sumd = __riscv_vfwmacc_vf_f32m2(_sumd, pB[13], _pA, vl);
                    _sume = __riscv_vfwmacc_vf_f32m2(_sume, pB[14], _pA, vl);
                    _sumf = __riscv_vfwmacc_vf_f32m2(_sumf, pB[15], _pA, vl);
                    pA += packn;
                    pB += 16;
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
                    _sum8 = __riscv_vfmul_vf_f32m2(_sum8, alpha, vl2);
                    _sum9 = __riscv_vfmul_vf_f32m2(_sum9, alpha, vl2);
                    _suma = __riscv_vfmul_vf_f32m2(_suma, alpha, vl2);
                    _sumb = __riscv_vfmul_vf_f32m2(_sumb, alpha, vl2);
                    _sumc = __riscv_vfmul_vf_f32m2(_sumc, alpha, vl2);
                    _sumd = __riscv_vfmul_vf_f32m2(_sumd, alpha, vl2);
                    _sume = __riscv_vfmul_vf_f32m2(_sume, alpha, vl2);
                    _sumf = __riscv_vfmul_vf_f32m2(_sumf, alpha, vl2);
                }

                if (k_end)
                {
                    if (out_elempack == packn)
                    {
                        __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn, __riscv_vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 2, __riscv_vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 3, __riscv_vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 4, __riscv_vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 5, __riscv_vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 6, __riscv_vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 7, __riscv_vfncvt_f_f_w_f16m1(_sum7, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 8, __riscv_vfncvt_f_f_w_f16m1(_sum8, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 9, __riscv_vfncvt_f_f_w_f16m1(_sum9, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 10, __riscv_vfncvt_f_f_w_f16m1(_suma, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 11, __riscv_vfncvt_f_f_w_f16m1(_sumb, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 12, __riscv_vfncvt_f_f_w_f16m1(_sumc, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 13, __riscv_vfncvt_f_f_w_f16m1(_sumd, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 14, __riscv_vfncvt_f_f_w_f16m1(_sume, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 15, __riscv_vfncvt_f_f_w_f16m1(_sumf, vl), vl);
                        outptr0 += packn * 16;
                    }
                    if (out_elempack == 1)
                    {
                        vfloat16m1x8_t _sum0_f16 = __riscv_vcreate_v_f16m1x8(
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum0, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum1, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum2, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum3, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum4, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum5, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum6, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum7, vl));
                        vfloat16m1x8_t _sum1_f16 = __riscv_vcreate_v_f16m1x8(
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum8, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sum9, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_suma, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sumb, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sumc, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sumd, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sume, vl),
                                                       __riscv_vfncvt_f_f_w_f16m1(_sumf, vl));
                        __riscv_vssseg8e16_v_f16m1x8(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                        __riscv_vssseg8e16_v_f16m1x8(outptr0 + 8, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                        outptr0 += 16;
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
                    __riscv_vse32_v_f32m2(outptr + packn * 8, _sum8, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 9, _sum9, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 10, _suma, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 11, _sumb, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 12, _sumc, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 13, _sumd, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 14, _sume, vl2);
                    __riscv_vse32_v_f32m2(outptr + packn * 15, _sumf, vl2);
                }

                outptr += packn * 16;
            }
            else if (packn == 8)
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
                    _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum2 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum3 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum4 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum5 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum6 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                    _sum7 = __riscv_vfmv_v_f_f32m2(0.f, vl2);

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

                const __fp16* pA = pAT;
                int kk = 0;
                for (; kk < max_kk; kk += 1)
                {
                    vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                    _sum0 = __riscv_vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                    _sum1 = __riscv_vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                    _sum2 = __riscv_vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                    _sum3 = __riscv_vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                    _sum4 = __riscv_vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                    _sum5 = __riscv_vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                    _sum6 = __riscv_vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                    _sum7 = __riscv_vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);
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
                        __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn, __riscv_vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 2, __riscv_vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 3, __riscv_vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 4, __riscv_vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 5, __riscv_vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 6, __riscv_vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                        __riscv_vse16_v_f16m1(outptr0 + packn * 7, __riscv_vfncvt_f_f_w_f16m1(_sum7, vl), vl);
                        outptr0 += packn * 8;
                    }
                    if (out_elempack == 1)
                    {
                        vfloat16m1x8_t _sum_f16 = __riscv_vcreate_v_f16m1x8(
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum0, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum1, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum2, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum3, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum4, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum5, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum6, vl),
                                                      __riscv_vfncvt_f_f_w_f16m1(_sum7, vl));
                        __riscv_vssseg8e16_v_f16m1x8(outptr0, out_hstep * sizeof(__fp16), _sum_f16, vl);
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
            else
            {
                NCNN_LOGE("unsupported vector length");
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            if (k == 0)
            {
                _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);

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
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
                _sum1 = __riscv_vle32_v_f32m2(outptr + packn, vl2);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = __riscv_vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
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
                    __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, __riscv_vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    outptr0 += packn * 2;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1x2_t _sum_f16 = __riscv_vcreate_v_f16m1x2(
                                                  __riscv_vfncvt_f_f_w_f16m1(_sum0, vl),
                                                  __riscv_vfncvt_f_f_w_f16m1(_sum1, vl));
                    __riscv_vssseg2e16_v_f16m1x2(outptr0, out_hstep * sizeof(__fp16), _sum_f16, vl);
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
                _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);

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
            }
            else
            {
                _sum0 = __riscv_vle32_v_f32m2(outptr, vl2);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
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
                    __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    outptr0 += packn;
                }
                if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
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
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

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
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            if (k == 0)
            {
                _sum0 = __riscv_vfmv_v_f_f32m2(0.f, vl2);
                _sum1 = __riscv_vfmv_v_f_f32m2(0.f, vl2);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                        _sum1 = __riscv_vfmv_v_f_f32m2(pC[1], vl2);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m2x2_t _s0 = __riscv_vlseg2e32_v_f32m2x2(pC, vl2);
                        _sum0 = __riscv_vget_v_f32m2x2_f32m2(_s0, 0);
                        _sum1 = __riscv_vget_v_f32m2x2_f32m2(_s0, 1);
                        pC += packn * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle32_v_f32m2(pC, vl2);
                        _sum1 = _sum0;
                        pC += packn;
                    }
                }
            }
            else
            {
                vfloat32m2x2_t _s0 = __riscv_vlseg2e32_v_f32m2x2(outptr, vl2);
                _sum0 = __riscv_vget_v_f32m2x2_f32m2(_s0, 0);
                _sum1 = __riscv_vget_v_f32m2x2_f32m2(_s0, 1);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl);
                _sum0 = __riscv_vfwmacc_vf_f32m2(_sum0, pA[0], _pB, vl);
                _sum1 = __riscv_vfwmacc_vf_f32m2(_sum1, pA[1], _pB, vl);
                pA += 2;
                pB += packn;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f32m2(_sum0, alpha, vl2);
                _sum1 = __riscv_vfmul_vf_f32m2(_sum1, alpha, vl2);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    __riscv_vse16_v_f16m1(outptr0 + out_hstep, __riscv_vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    outptr0 += packn;
                }
            }
            else
            {
                __riscv_vsseg2e32_v_f32m2x2(outptr, __riscv_vcreate_v_f32m2x2(_sum0, _sum1), vl2);
            }

            outptr += packn * 2;
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
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += (float)pA[0] * (float)pB[0];
                sum01 += (float)pA[1] * (float)pB[0];
                sum10 += (float)pA[0] * (float)pB[1];
                sum11 += (float)pA[1] * (float)pB[1];
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
                    outptr0[0] = (__fp16)sum00;
                    outptr0[1] = (__fp16)sum10;
                    outptr0[out_hstep] = (__fp16)sum01;
                    outptr0[out_hstep + 1] = (__fp16)sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
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

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += (float)pA[0] * (float)pB[0];
                sum1 += (float)pA[1] * (float)pB[0];
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
                    outptr0[0] = (__fp16)sum0;
                    outptr0[out_hstep] = (__fp16)sum1;
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
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

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
        for (; jj + (packn - 1) < max_jj; jj += packn)
        {
            vfloat32m2_t _sum;

            if (k == 0)
            {
                _sum = __riscv_vfmv_v_f_f32m2(0.f, vl2);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f32m2(pC[0], vl2);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle32_v_f32m2(pC, vl2);
                        pC += packn;
                    }
                }
            }
            else
            {
                _sum = __riscv_vle32_v_f32m2(outptr, vl2);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl);
                _sum = __riscv_vfwmacc_vf_f32m2(_sum, pA[0], _pB, vl);
                pA += 1;
                pB += packn;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f32m2(_sum, alpha, vl2);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __riscv_vse16_v_f16m1(outptr0, __riscv_vfncvt_f_f_w_f16m1(_sum, vl), vl);
                    outptr0 += packn;
                }
            }
            else
            {
                __riscv_vse32_v_f32m2(outptr, _sum, vl2);
            }

            outptr += packn;
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

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += (float)pA[0] * (float)pB[0];
                sum1 += (float)pA[0] * (float)pB[1];
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
                    outptr0[0] = (__fp16)sum0;
                    outptr0[1] = (__fp16)sum1;
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

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += (float)pA[0] * (float)pB[0];
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
                    outptr0[0] = (__fp16)sum;
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
