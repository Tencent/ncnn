// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void gemm_transB_packed_tile_fma(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void gemm_transB_packed_tile_fma4(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end);
#endif

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        gemm_transB_packed_tile_fma(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        gemm_transB_packed_tile_fma4(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

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
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;
            __m512 _sum4;
            __m512 _sum5;
            __m512 _sum6;
            __m512 _sum7;
            __m512 _sum8;
            __m512 _sum9;
            __m512 _suma;
            __m512 _sumb;
            __m512 _sumc;
            __m512 _sumd;
            __m512 _sume;
            __m512 _sumf;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();
                _sum4 = _mm512_setzero_ps();
                _sum5 = _mm512_setzero_ps();
                _sum6 = _mm512_setzero_ps();
                _sum7 = _mm512_setzero_ps();
                _sum8 = _mm512_setzero_ps();
                _sum9 = _mm512_setzero_ps();
                _suma = _mm512_setzero_ps();
                _sumb = _mm512_setzero_ps();
                _sumc = _mm512_setzero_ps();
                _sumd = _mm512_setzero_ps();
                _sume = _mm512_setzero_ps();
                _sumf = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                        _sum4 = _mm512_set1_ps(pC[0]);
                        _sum5 = _mm512_set1_ps(pC[0]);
                        _sum6 = _mm512_set1_ps(pC[0]);
                        _sum7 = _mm512_set1_ps(pC[0]);
                        _sum8 = _mm512_set1_ps(pC[0]);
                        _sum9 = _mm512_set1_ps(pC[0]);
                        _suma = _mm512_set1_ps(pC[0]);
                        _sumb = _mm512_set1_ps(pC[0]);
                        _sumc = _mm512_set1_ps(pC[0]);
                        _sumd = _mm512_set1_ps(pC[0]);
                        _sume = _mm512_set1_ps(pC[0]);
                        _sumf = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
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
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 16 * 2);
                        _sum3 = _mm512_loadu_ps(pC + 16 * 3);
                        _sum4 = _mm512_loadu_ps(pC + 16 * 4);
                        _sum5 = _mm512_loadu_ps(pC + 16 * 5);
                        _sum6 = _mm512_loadu_ps(pC + 16 * 6);
                        _sum7 = _mm512_loadu_ps(pC + 16 * 7);
                        _sum8 = _mm512_loadu_ps(pC + 16 * 8);
                        _sum9 = _mm512_loadu_ps(pC + 16 * 9);
                        _suma = _mm512_loadu_ps(pC + 16 * 10);
                        _sumb = _mm512_loadu_ps(pC + 16 * 11);
                        _sumc = _mm512_loadu_ps(pC + 16 * 12);
                        _sumd = _mm512_loadu_ps(pC + 16 * 13);
                        _sume = _mm512_loadu_ps(pC + 16 * 14);
                        _sumf = _mm512_loadu_ps(pC + 16 * 15);
                        pC += 256;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        _sum4 = _mm512_set1_ps(pC[4]);
                        _sum5 = _mm512_set1_ps(pC[5]);
                        _sum6 = _mm512_set1_ps(pC[6]);
                        _sum7 = _mm512_set1_ps(pC[7]);
                        _sum8 = _mm512_set1_ps(pC[8]);
                        _sum9 = _mm512_set1_ps(pC[9]);
                        _suma = _mm512_set1_ps(pC[10]);
                        _sumb = _mm512_set1_ps(pC[11]);
                        _sumc = _mm512_set1_ps(pC[12]);
                        _sumd = _mm512_set1_ps(pC[13]);
                        _sume = _mm512_set1_ps(pC[14]);
                        _sumf = _mm512_set1_ps(pC[15]);
                        pC += 16;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
                _sum4 = _mm512_load_ps(outptr + 16 * 4);
                _sum5 = _mm512_load_ps(outptr + 16 * 5);
                _sum6 = _mm512_load_ps(outptr + 16 * 6);
                _sum7 = _mm512_load_ps(outptr + 16 * 7);
                _sum8 = _mm512_load_ps(outptr + 16 * 8);
                _sum9 = _mm512_load_ps(outptr + 16 * 9);
                _suma = _mm512_load_ps(outptr + 16 * 10);
                _sumb = _mm512_load_ps(outptr + 16 * 11);
                _sumc = _mm512_load_ps(outptr + 16 * 12);
                _sumd = _mm512_load_ps(outptr + 16 * 13);
                _sume = _mm512_load_ps(outptr + 16 * 14);
                _sumf = _mm512_load_ps(outptr + 16 * 15);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);
                _sum8 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[8]), _sum8);
                _sum9 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[9]), _sum9);
                _suma = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[10]), _suma);
                _sumb = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[11]), _sumb);
                _sumc = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[12]), _sumc);
                _sumd = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[13]), _sumd);
                _sume = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[14]), _sume);
                _sumf = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[15]), _sumf);

                pA += 16;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    _mm512_store_ps(outptr0 + 16 * 8, _sum8);
                    _mm512_store_ps(outptr0 + 16 * 9, _sum9);
                    _mm512_store_ps(outptr0 + 16 * 10, _suma);
                    _mm512_store_ps(outptr0 + 16 * 11, _sumb);
                    _mm512_store_ps(outptr0 + 16 * 12, _sumc);
                    _mm512_store_ps(outptr0 + 16 * 13, _sumd);
                    _mm512_store_ps(outptr0 + 16 * 14, _sume);
                    _mm512_store_ps(outptr0 + 16 * 15, _sumf);
                    outptr0 += 256;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);
                    _mm512_storeu_ps(outptr0 + 16 * 4, _tmp4);
                    _mm512_storeu_ps(outptr0 + 16 * 5, _tmp5);
                    _mm512_storeu_ps(outptr0 + 16 * 6, _tmp6);
                    _mm512_storeu_ps(outptr0 + 16 * 7, _tmp7);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmpa);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmpb);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 4, _tmpc);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 5, _tmpd);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 6, _tmpe);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 7, _tmpf);

                    outptr0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp8 = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_sum8, _sum9, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_suma, _sumb, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_sumc, _sumd, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_sume, _sumf, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _suma = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _sumb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                    _sumc = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _sumd = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _sume = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                    _sumf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + 32, _sum8);
                    _mm512_storeu_ps(outptr0 + 48, _sumc);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 32, _sum9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 48, _sumd);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 32, _suma);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 48, _sume);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 32, _sumb);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 48, _sumf);

                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 7, _sum7);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum8);
                    _mm512_storeu_ps(outptr0 + out_hstep * 9, _sum9);
                    _mm512_storeu_ps(outptr0 + out_hstep * 10, _suma);
                    _mm512_storeu_ps(outptr0 + out_hstep * 11, _sumb);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sumc);
                    _mm512_storeu_ps(outptr0 + out_hstep * 13, _sumd);
                    _mm512_storeu_ps(outptr0 + out_hstep * 14, _sume);
                    _mm512_storeu_ps(outptr0 + out_hstep * 15, _sumf);

                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
                _mm512_store_ps(outptr + 16 * 8, _sum8);
                _mm512_store_ps(outptr + 16 * 9, _sum9);
                _mm512_store_ps(outptr + 16 * 10, _suma);
                _mm512_store_ps(outptr + 16 * 11, _sumb);
                _mm512_store_ps(outptr + 16 * 12, _sumc);
                _mm512_store_ps(outptr + 16 * 13, _sumd);
                _mm512_store_ps(outptr + 16 * 14, _sume);
                _mm512_store_ps(outptr + 16 * 15, _sumf);
            }

            outptr += 256;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;
            __m512 _sum4;
            __m512 _sum5;
            __m512 _sum6;
            __m512 _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();
                _sum4 = _mm512_setzero_ps();
                _sum5 = _mm512_setzero_ps();
                _sum6 = _mm512_setzero_ps();
                _sum7 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                        _sum4 = _mm512_set1_ps(pC[0]);
                        _sum5 = _mm512_set1_ps(pC[0]);
                        _sum6 = _mm512_set1_ps(pC[0]);
                        _sum7 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
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
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 16 * 2);
                        _sum3 = _mm512_loadu_ps(pC + 16 * 3);
                        _sum4 = _mm512_loadu_ps(pC + 16 * 4);
                        _sum5 = _mm512_loadu_ps(pC + 16 * 5);
                        _sum6 = _mm512_loadu_ps(pC + 16 * 6);
                        _sum7 = _mm512_loadu_ps(pC + 16 * 7);
                        pC += 128;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        _sum4 = _mm512_set1_ps(pC[4]);
                        _sum5 = _mm512_set1_ps(pC[5]);
                        _sum6 = _mm512_set1_ps(pC[6]);
                        _sum7 = _mm512_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
                _sum4 = _mm512_load_ps(outptr + 16 * 4);
                _sum5 = _mm512_load_ps(outptr + 16 * 5);
                _sum6 = _mm512_load_ps(outptr + 16 * 6);
                _sum7 = _mm512_load_ps(outptr + 16 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);
                _sum4 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[4]), _sum4);
                _sum5 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[5]), _sum5);
                _sum6 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[6]), _sum6);
                _sum7 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[7]), _sum7);

                pA += 16;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 2, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16 * 3, _tmp7);

                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12 + 16, _sum7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _mm512_extractf32x8_ps(_sum0, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _mm512_extractf32x8_ps(_sum1, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _mm512_extractf32x8_ps(_sum1, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _mm512_extractf32x8_ps(_sum2, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _mm512_extractf32x8_ps(_sum2, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _mm512_extractf32x8_ps(_sum3, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _mm512_extractf32x8_ps(_sum3, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum4, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 9, _mm512_extractf32x8_ps(_sum4, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 10, _mm512_extractf32x8_ps(_sum5, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 11, _mm512_extractf32x8_ps(_sum5, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 12, _mm512_extractf32x8_ps(_sum6, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 13, _mm512_extractf32x8_ps(_sum6, 1));
                    _mm256_storeu_ps(outptr0 + out_hstep * 14, _mm512_extractf32x8_ps(_sum7, 0));
                    _mm256_storeu_ps(outptr0 + out_hstep * 15, _mm512_extractf32x8_ps(_sum7, 1));

                    outptr0 += 8;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
            }

            outptr += 128;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 32);
                        _sum3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        _sum2 = _mm512_set1_ps(pC[2]);
                        _sum3 = _mm512_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16 * 1);
                _sum2 = _mm512_load_ps(outptr + 16 * 2);
                _sum3 = _mm512_load_ps(outptr + 16 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);
                _sum2 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[2]), _sum2);
                _sum3 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[3]), _sum3);

                pA += 16;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);

                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8 + 16, _tmp3);

                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 2, 3, 2));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 12, _sum3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm512_extractf32x4_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm512_extractf32x4_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm512_extractf32x4_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm512_extractf32x4_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm512_extractf32x4_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm512_extractf32x4_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm512_extractf32x4_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm512_extractf32x4_ps(_sum3, 1);
                    __m128 _sum0_2 = _mm512_extractf32x4_ps(_sum0, 2);
                    __m128 _sum1_2 = _mm512_extractf32x4_ps(_sum1, 2);
                    __m128 _sum2_2 = _mm512_extractf32x4_ps(_sum2, 2);
                    __m128 _sum3_2 = _mm512_extractf32x4_ps(_sum3, 2);
                    __m128 _sum0_3 = _mm512_extractf32x4_ps(_sum0, 3);
                    __m128 _sum1_3 = _mm512_extractf32x4_ps(_sum1, 3);
                    __m128 _sum2_3 = _mm512_extractf32x4_ps(_sum2, 3);
                    __m128 _sum3_3 = _mm512_extractf32x4_ps(_sum3, 3);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);
                    _MM_TRANSPOSE4_PS(_sum0_2, _sum1_2, _sum2_2, _sum3_2);
                    _MM_TRANSPOSE4_PS(_sum0_3, _sum1_3, _sum2_3, _sum3_3);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 8, _sum0_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 9, _sum1_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 10, _sum2_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 11, _sum3_2);
                    _mm_storeu_ps(outptr0 + out_hstep * 12, _sum0_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 13, _sum1_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 14, _sum2_3);
                    _mm_storeu_ps(outptr0 + out_hstep * 15, _sum3_3);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
            }

            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _sum0;
            __m512 _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
                _sum1 = _mm512_load_ps(outptr + 16);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);
                _sum1 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[1]), _sum1);

                pA += 16;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16, _sum1);
                    outptr0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 8, _tmp1);

                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + 4, _mm512_extractf32x4_ps(_sum1, 0));

                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_sum1, 1));

                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_sum1, 2));

                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    _mm_store_ps(outptr0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_sum1, 3));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(64))
#else
                    __attribute__((aligned(64)))
#endif
                    float sumbuf[32];
                    float* sum0 = sumbuf;
                    float* sum1 = sumbuf + 16;
                    _mm512_store_ps(sum0, _sum0);
                    _mm512_store_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 8 + 1] = sum1[8];
                    outptr0[out_hstep * 9 + 1] = sum1[9];
                    outptr0[out_hstep * 10 + 1] = sum1[10];
                    outptr0[out_hstep * 11 + 1] = sum1[11];
                    outptr0[out_hstep * 12 + 1] = sum1[12];
                    outptr0[out_hstep * 13 + 1] = sum1[13];
                    outptr0[out_hstep * 14 + 1] = sum1[14];
                    outptr0[out_hstep * 15 + 1] = sum1[15];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16, _sum1);
            }

            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m512 _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pA = _mm512_load_ps(pA);

                _sum0 = _mm512_fmadd_ps(_pA, _mm512_set1_ps(pB[0]), _sum0);

                pA += 16;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    outptr0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _mm512_extractf32x8_ps(_sum0, 0));
                    _mm256_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x8_ps(_sum0, 1));
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm512_extractf32x4_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm512_extractf32x4_ps(_sum0, 1));
                    _mm_store_ps(outptr0 + out_hstep * 8, _mm512_extractf32x4_ps(_sum0, 2));
                    _mm_store_ps(outptr0 + out_hstep * 12, _mm512_extractf32x4_ps(_sum0, 3));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(64))
#else
                    __attribute__((aligned(64)))
#endif
                    float sum0[16];
                    _mm512_store_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0[out_hstep * 8] = sum0[8];
                    outptr0[out_hstep * 9] = sum0[9];
                    outptr0[out_hstep * 10] = sum0[10];
                    outptr0[out_hstep * 11] = sum0[11];
                    outptr0[out_hstep * 12] = sum0[12];
                    outptr0[out_hstep * 13] = sum0[13];
                    outptr0[out_hstep * 14] = sum0[14];
                    outptr0[out_hstep * 15] = sum0[15];
                    outptr0++;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
            }

            outptr += 16;
        }

        pAT += max_kk * 16;
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;
            __m512 _sum4;
            __m512 _sum5;
            __m512 _sum6;
            __m512 _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();
                _sum4 = _mm512_setzero_ps();
                _sum5 = _mm512_setzero_ps();
                _sum6 = _mm512_setzero_ps();
                _sum7 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                        _sum4 = _mm512_set1_ps(pC[0]);
                        _sum5 = _mm512_set1_ps(pC[0]);
                        _sum6 = _mm512_set1_ps(pC[0]);
                        _sum7 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m256 _tmp = _mm256_loadu_ps(pC);
                        _sum0 = combine8x2_ps(_tmp, _tmp);
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
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 16 * 2);
                        _sum3 = _mm512_loadu_ps(pC + 16 * 3);
                        _sum4 = _mm512_loadu_ps(pC + 16 * 4);
                        _sum5 = _mm512_loadu_ps(pC + 16 * 5);
                        _sum6 = _mm512_loadu_ps(pC + 16 * 6);
                        _sum7 = _mm512_loadu_ps(pC + 16 * 7);
                        pC += 128;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = combine8x2_ps(_mm256_set1_ps(pC[0]), _mm256_set1_ps(pC[1]));
                        _sum1 = combine8x2_ps(_mm256_set1_ps(pC[2]), _mm256_set1_ps(pC[3]));
                        _sum2 = combine8x2_ps(_mm256_set1_ps(pC[4]), _mm256_set1_ps(pC[5]));
                        _sum3 = combine8x2_ps(_mm256_set1_ps(pC[6]), _mm256_set1_ps(pC[7]));
                        _sum4 = combine8x2_ps(_mm256_set1_ps(pC[8]), _mm256_set1_ps(pC[9]));
                        _sum5 = combine8x2_ps(_mm256_set1_ps(pC[10]), _mm256_set1_ps(pC[11]));
                        _sum6 = combine8x2_ps(_mm256_set1_ps(pC[12]), _mm256_set1_ps(pC[13]));
                        _sum7 = combine8x2_ps(_mm256_set1_ps(pC[14]), _mm256_set1_ps(pC[15]));
                        pC += 16;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_loadu_ps(outptr);
                _sum1 = _mm512_loadu_ps(outptr + 16 * 1);
                _sum2 = _mm512_loadu_ps(outptr + 16 * 2);
                _sum3 = _mm512_loadu_ps(outptr + 16 * 3);
                _sum4 = _mm512_loadu_ps(outptr + 16 * 4);
                _sum5 = _mm512_loadu_ps(outptr + 16 * 5);
                _sum6 = _mm512_loadu_ps(outptr + 16 * 6);
                _sum7 = _mm512_loadu_ps(outptr + 16 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);
                __m512 _pAA = combine8x2_ps(_pA, _pA);
                __m512 _pB0 = combine8x2_ps(_mm256_set1_ps(pB[0]), _mm256_set1_ps(pB[1]));
                __m512 _pB1 = combine8x2_ps(_mm256_set1_ps(pB[2]), _mm256_set1_ps(pB[3]));
                __m512 _pB2 = combine8x2_ps(_mm256_set1_ps(pB[4]), _mm256_set1_ps(pB[5]));
                __m512 _pB3 = combine8x2_ps(_mm256_set1_ps(pB[6]), _mm256_set1_ps(pB[7]));
                __m512 _pB4 = combine8x2_ps(_mm256_set1_ps(pB[8]), _mm256_set1_ps(pB[9]));
                __m512 _pB5 = combine8x2_ps(_mm256_set1_ps(pB[10]), _mm256_set1_ps(pB[11]));
                __m512 _pB6 = combine8x2_ps(_mm256_set1_ps(pB[12]), _mm256_set1_ps(pB[13]));
                __m512 _pB7 = combine8x2_ps(_mm256_set1_ps(pB[14]), _mm256_set1_ps(pB[15]));
                _sum0 = _mm512_fmadd_ps(_pAA, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pAA, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pAA, _pB2, _sum2);
                _sum3 = _mm512_fmadd_ps(_pAA, _pB3, _sum3);
                _sum4 = _mm512_fmadd_ps(_pAA, _pB4, _sum4);
                _sum5 = _mm512_fmadd_ps(_pAA, _pB5, _sum5);
                _sum6 = _mm512_fmadd_ps(_pAA, _pB6, _sum6);
                _sum7 = _mm512_fmadd_ps(_pAA, _pB7, _sum7);

                pA += 8;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm512_store_ps(outptr0, _sum0);
                    _mm512_store_ps(outptr0 + 16 * 1, _sum1);
                    _mm512_store_ps(outptr0 + 16 * 2, _sum2);
                    _mm512_store_ps(outptr0 + 16 * 3, _sum3);
                    _mm512_store_ps(outptr0 + 16 * 4, _sum4);
                    _mm512_store_ps(outptr0 + 16 * 5, _sum5);
                    _mm512_store_ps(outptr0 + 16 * 6, _sum6);
                    _mm512_store_ps(outptr0 + 16 * 7, _sum7);
                    outptr0 += 128;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(outptr0, _tmp0);
                    _mm512_storeu_ps(outptr0 + 16, _tmp1);
                    _mm512_storeu_ps(outptr0 + 16 * 2, _tmp2);
                    _mm512_storeu_ps(outptr0 + 16 * 3, _tmp3);

                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _tmp4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16, _tmp5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16 * 2, _tmp6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4 + 16 * 3, _tmp7);

                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_unpacklo_ps(_sum0, _sum1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_sum2, _sum3);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_sum4, _sum5);
                    __m512 _tmp3 = _mm512_unpacklo_ps(_sum6, _sum7);
                    __m512 _tmp4 = _mm512_unpackhi_ps(_sum0, _sum1);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_sum2, _sum3);
                    __m512 _tmp6 = _mm512_unpackhi_ps(_sum4, _sum5);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_sum6, _sum7);

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp2 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp4 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp5 = _mm512_shuffle_f32x4(_sum4, _sum5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp7 = _mm512_shuffle_f32x4(_sum6, _sum7, _MM_SHUFFLE(3, 1, 3, 1));

                    _sum0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
                    _sum1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
                    _sum2 = _mm512_unpacklo_ps(_tmp2, _tmp3);
                    _sum3 = _mm512_unpackhi_ps(_tmp2, _tmp3);
                    _sum4 = _mm512_unpacklo_ps(_tmp4, _tmp5);
                    _sum5 = _mm512_unpackhi_ps(_tmp4, _tmp5);
                    _sum6 = _mm512_unpacklo_ps(_tmp6, _tmp7);
                    _sum7 = _mm512_unpackhi_ps(_tmp6, _tmp7);

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm512_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm512_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm512_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm512_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_store_ps(outptr, _sum0);
                _mm512_store_ps(outptr + 16 * 1, _sum1);
                _mm512_store_ps(outptr + 16 * 2, _sum2);
                _mm512_store_ps(outptr + 16 * 3, _sum3);
                _mm512_store_ps(outptr + 16 * 4, _sum4);
                _mm512_store_ps(outptr + 16 * 5, _sum5);
                _mm512_store_ps(outptr + 16 * 6, _sum6);
                _mm512_store_ps(outptr + 16 * 7, _sum7);
            }

            outptr += 128;
        }
#else  // __AVX512F__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;
            __m256 _sum8;
            __m256 _sum9;
            __m256 _suma;
            __m256 _sumb;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();
                _sum4 = _mm256_setzero_ps();
                _sum5 = _mm256_setzero_ps();
                _sum6 = _mm256_setzero_ps();
                _sum7 = _mm256_setzero_ps();
                _sum8 = _mm256_setzero_ps();
                _sum9 = _mm256_setzero_ps();
                _suma = _mm256_setzero_ps();
                _sumb = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                        _sum4 = _mm256_set1_ps(pC[0]);
                        _sum5 = _mm256_set1_ps(pC[0]);
                        _sum6 = _mm256_set1_ps(pC[0]);
                        _sum7 = _mm256_set1_ps(pC[0]);
                        _sum8 = _mm256_set1_ps(pC[0]);
                        _sum9 = _mm256_set1_ps(pC[0]);
                        _suma = _mm256_set1_ps(pC[0]);
                        _sumb = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
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
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 8 * 2);
                        _sum3 = _mm256_loadu_ps(pC + 8 * 3);
                        _sum4 = _mm256_loadu_ps(pC + 8 * 4);
                        _sum5 = _mm256_loadu_ps(pC + 8 * 5);
                        _sum6 = _mm256_loadu_ps(pC + 8 * 6);
                        _sum7 = _mm256_loadu_ps(pC + 8 * 7);
                        _sum8 = _mm256_loadu_ps(pC + 8 * 8);
                        _sum9 = _mm256_loadu_ps(pC + 8 * 9);
                        _suma = _mm256_loadu_ps(pC + 8 * 10);
                        _sumb = _mm256_loadu_ps(pC + 8 * 11);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        _sum4 = _mm256_set1_ps(pC[4]);
                        _sum5 = _mm256_set1_ps(pC[5]);
                        _sum6 = _mm256_set1_ps(pC[6]);
                        _sum7 = _mm256_set1_ps(pC[7]);
                        _sum8 = _mm256_set1_ps(pC[8]);
                        _sum9 = _mm256_set1_ps(pC[9]);
                        _suma = _mm256_set1_ps(pC[10]);
                        _sumb = _mm256_set1_ps(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
                _sum8 = _mm256_load_ps(outptr + 8 * 8);
                _sum9 = _mm256_load_ps(outptr + 8 * 9);
                _suma = _mm256_load_ps(outptr + 8 * 10);
                _sumb = _mm256_load_ps(outptr + 8 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);
                _sum8 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[8]), _sum8);
                _sum9 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[9]), _sum9);
                _suma = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[10]), _suma);
                _sumb = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[11]), _sumb);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    _mm256_store_ps(outptr0 + 8 * 8, _sum8);
                    _mm256_store_ps(outptr0 + 8 * 9, _sum9);
                    _mm256_store_ps(outptr0 + 8 * 10, _suma);
                    _mm256_store_ps(outptr0 + 8 * 11, _sumb);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp8 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp9 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpa = _mm256_permute2f128_ps(_sum8, _sum9, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmpb = _mm256_permute2f128_ps(_suma, _sumb, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);
                    _mm256_storeu_ps(outptr0 + 8 * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + 8 * 5, _tmp5);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp7);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp8);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp9);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 4, _tmpa);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 5, _tmpb);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    __m128 _sum8_0 = _mm256_extractf128_ps(_sum8, 0);
                    __m128 _sum9_0 = _mm256_extractf128_ps(_sum9, 0);
                    __m128 _suma_0 = _mm256_extractf128_ps(_suma, 0);
                    __m128 _sumb_0 = _mm256_extractf128_ps(_sumb, 0);
                    __m128 _sum8_1 = _mm256_extractf128_ps(_sum8, 1);
                    __m128 _sum9_1 = _mm256_extractf128_ps(_sum9, 1);
                    __m128 _suma_1 = _mm256_extractf128_ps(_suma, 1);
                    __m128 _sumb_1 = _mm256_extractf128_ps(_sumb, 1);

                    _MM_TRANSPOSE4_PS(_sum8_0, _sum9_0, _suma_0, _sumb_0);
                    _MM_TRANSPOSE4_PS(_sum8_1, _sum9_1, _suma_1, _sumb_1);

                    _mm_storeu_ps(outptr0 + 8, _sum8_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4 + 8, _sum8_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5 + 8, _sum9_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6 + 8, _suma_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7 + 8, _sumb_1);

                    outptr0 += 12;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
                _mm256_store_ps(outptr + 8 * 8, _sum8);
                _mm256_store_ps(outptr + 8 * 9, _sum9);
                _mm256_store_ps(outptr + 8 * 10, _suma);
                _mm256_store_ps(outptr + 8 * 11, _sumb);
            }

            outptr += 96;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;
            __m256 _sum4;
            __m256 _sum5;
            __m256 _sum6;
            __m256 _sum7;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();
                _sum4 = _mm256_setzero_ps();
                _sum5 = _mm256_setzero_ps();
                _sum6 = _mm256_setzero_ps();
                _sum7 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                        _sum4 = _mm256_set1_ps(pC[0]);
                        _sum5 = _mm256_set1_ps(pC[0]);
                        _sum6 = _mm256_set1_ps(pC[0]);
                        _sum7 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
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
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 8 * 2);
                        _sum3 = _mm256_loadu_ps(pC + 8 * 3);
                        _sum4 = _mm256_loadu_ps(pC + 8 * 4);
                        _sum5 = _mm256_loadu_ps(pC + 8 * 5);
                        _sum6 = _mm256_loadu_ps(pC + 8 * 6);
                        _sum7 = _mm256_loadu_ps(pC + 8 * 7);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        _sum4 = _mm256_set1_ps(pC[4]);
                        _sum5 = _mm256_set1_ps(pC[5]);
                        _sum6 = _mm256_set1_ps(pC[6]);
                        _sum7 = _mm256_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
                _sum4 = _mm256_load_ps(outptr + 8 * 4);
                _sum5 = _mm256_load_ps(outptr + 8 * 5);
                _sum6 = _mm256_load_ps(outptr + 8 * 6);
                _sum7 = _mm256_load_ps(outptr + 8 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);
                _sum4 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[4]), _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[5]), _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[6]), _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[7]), _sum7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                    _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                    _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                    _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_sum4, _sum5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_sum6, _sum7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);
                    _mm256_storeu_ps(outptr0 + 8 * 2, _tmp2);
                    _mm256_storeu_ps(outptr0 + 8 * 3, _tmp3);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 2, _tmp6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8 * 3, _tmp7);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm256_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _sum4);
                    _mm256_storeu_ps(outptr0 + out_hstep * 5, _sum5);
                    _mm256_storeu_ps(outptr0 + out_hstep * 6, _sum6);
                    _mm256_storeu_ps(outptr0 + out_hstep * 7, _sum7);

                    outptr0 += 8;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
                _mm256_store_ps(outptr + 8 * 4, _sum4);
                _mm256_store_ps(outptr + 8 * 5, _sum5);
                _mm256_store_ps(outptr + 8 * 6, _sum6);
                _mm256_store_ps(outptr + 8 * 7, _sum7);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();
                _sum2 = _mm256_setzero_ps();
                _sum3 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                        _sum2 = _mm256_set1_ps(pC[0]);
                        _sum3 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        _sum2 = _mm256_loadu_ps(pC + 16);
                        _sum3 = _mm256_loadu_ps(pC + 24);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        _sum2 = _mm256_set1_ps(pC[2]);
                        _sum3 = _mm256_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8 * 1);
                _sum2 = _mm256_load_ps(outptr + 8 * 2);
                _sum3 = _mm256_load_ps(outptr + 8 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[2]), _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[3]), _sum3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8 * 1, _sum1);
                    _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                    _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_sum2, _sum3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + 8, _tmp1);

                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp2);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4 + 8, _tmp3);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    __m128 _sum0_0 = _mm256_extractf128_ps(_sum0, 0);
                    __m128 _sum1_0 = _mm256_extractf128_ps(_sum1, 0);
                    __m128 _sum2_0 = _mm256_extractf128_ps(_sum2, 0);
                    __m128 _sum3_0 = _mm256_extractf128_ps(_sum3, 0);
                    __m128 _sum0_1 = _mm256_extractf128_ps(_sum0, 1);
                    __m128 _sum1_1 = _mm256_extractf128_ps(_sum1, 1);
                    __m128 _sum2_1 = _mm256_extractf128_ps(_sum2, 1);
                    __m128 _sum3_1 = _mm256_extractf128_ps(_sum3, 1);

                    _MM_TRANSPOSE4_PS(_sum0_0, _sum1_0, _sum2_0, _sum3_0);
                    _MM_TRANSPOSE4_PS(_sum0_1, _sum1_1, _sum2_1, _sum3_1);

                    _mm_storeu_ps(outptr0, _sum0_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3_0);
                    _mm_storeu_ps(outptr0 + out_hstep * 4, _sum0_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 5, _sum1_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 6, _sum2_1);
                    _mm_storeu_ps(outptr0 + out_hstep * 7, _sum3_1);

                    outptr0 += 4;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8 * 1, _sum1);
                _mm256_store_ps(outptr + 8 * 2, _sum2);
                _mm256_store_ps(outptr + 8 * 3, _sum3);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();
                _sum1 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        _sum1 = _mm256_loadu_ps(pC + 8);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        _sum1 = _mm256_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
                _sum1 = _mm256_load_ps(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[1]), _sum1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    _mm256_store_ps(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_sum0, _sum1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(outptr0, _tmp0);
                    _mm256_storeu_ps(outptr0 + out_hstep * 4, _tmp1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(32))
#else
                    __attribute__((aligned(32)))
#endif
                    float sumbuf[16];
                    float* sum0 = sumbuf;
                    float* sum1 = sumbuf + 8;
                    _mm256_store_ps(sum0, _sum0);
                    _mm256_store_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
                _mm256_store_ps(outptr + 8, _sum1);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m256 _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm256_loadu_ps(pC);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm256_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm256_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = _mm256_load_ps(pA);

                _sum0 = _mm256_comp_fmadd_ps(_pA, _mm256_set1_ps(pB[0]), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    _mm256_store_ps(outptr0, _sum0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum0, 0));
                    _mm_store_ps(outptr0 + out_hstep * 4, _mm256_extractf128_ps(_sum0, 1));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(32))
#else
                    __attribute__((aligned(32)))
#endif
                    float sum0[8];
                    _mm256_store_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                _mm256_store_ps(outptr, _sum0);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

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
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0;
            __m512 _sum1;
            __m512 _sum2;
            __m512 _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();
                _sum2 = _mm512_setzero_ps();
                _sum3 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                        _sum2 = _mm512_set1_ps(pC[0]);
                        _sum3 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        __m128 _tmp = _mm_loadu_ps(pC);
                        _sum0 = _mm512_broadcast_f32x4(_tmp);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        _sum2 = _mm512_loadu_ps(pC + 32);
                        _sum3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        __m512 _tmp = _mm512_loadu_ps(pC);
                        _sum0 = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3), _tmp);
                        _sum1 = _mm512_permutexvar_ps(_mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7), _tmp);
                        _sum2 = _mm512_permutexvar_ps(_mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11), _tmp);
                        _sum3 = _mm512_permutexvar_ps(_mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15), _tmp);
                        pC += 16;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_loadu_ps(outptr);
                _sum1 = _mm512_loadu_ps(outptr + 16);
                _sum2 = _mm512_loadu_ps(outptr + 32);
                _sum3 = _mm512_loadu_ps(outptr + 48);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);
                __m512 _pB = _mm512_loadu_ps(pB);

                __m512 _pAAAA = _mm512_broadcast_f32x4(_pA);

                __m512 _pB0 = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3), _pB);
                __m512 _pB1 = _mm512_permutexvar_ps(_mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7), _pB);
                __m512 _pB2 = _mm512_permutexvar_ps(_mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11), _pB);
                __m512 _pB3 = _mm512_permutexvar_ps(_mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15), _pB);
                _sum0 = _mm512_fmadd_ps(_pAAAA, _pB0, _sum0);
                _sum1 = _mm512_fmadd_ps(_pAAAA, _pB1, _sum1);
                _sum2 = _mm512_fmadd_ps(_pAAAA, _pB2, _sum2);
                _sum3 = _mm512_fmadd_ps(_pAAAA, _pB3, _sum3);

                pA += 4;
                pB += 16;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + 16, _sum1);
                    _mm512_storeu_ps(outptr0 + 32, _sum2);
                    _mm512_storeu_ps(outptr0 + 48, _sum3);
                    outptr0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_sum2, _sum3, _MM_SHUFFLE(3, 1, 3, 1));

                    _sum0 = _mm512_shuffle_f32x4(_tmp0, _tmp2, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum1 = _mm512_shuffle_f32x4(_tmp1, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _sum2 = _mm512_shuffle_f32x4(_tmp0, _tmp2, _MM_SHUFFLE(3, 1, 3, 1));
                    _sum3 = _mm512_shuffle_f32x4(_tmp1, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _tmp0 = _mm512_unpacklo_ps(_sum0, _sum1);
                    _tmp1 = _mm512_unpacklo_ps(_sum2, _sum3);
                    _tmp2 = _mm512_unpackhi_ps(_sum0, _sum1);
                    _tmp3 = _mm512_unpackhi_ps(_sum2, _sum3);

                    _sum0 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(1, 0, 1, 0));
                    _sum1 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 2, 3, 2));
                    _sum2 = _mm512_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    _sum3 = _mm512_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm512_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm512_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_storeu_ps(outptr, _sum0);
                _mm512_storeu_ps(outptr + 16, _sum1);
                _mm512_storeu_ps(outptr + 32, _sum2);
                _mm512_storeu_ps(outptr + 48, _sum3);
            }

            outptr += 64;
        }
#else  // __AVX512F__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;
            __m128 _sum8;
            __m128 _sum9;
            __m128 _suma;
            __m128 _sumb;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();
                _sum4 = _mm_setzero_ps();
                _sum5 = _mm_setzero_ps();
                _sum6 = _mm_setzero_ps();
                _sum7 = _mm_setzero_ps();
                _sum8 = _mm_setzero_ps();
                _sum9 = _mm_setzero_ps();
                _suma = _mm_setzero_ps();
                _sumb = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                        _sum4 = _mm_set1_ps(pC[0]);
                        _sum5 = _mm_set1_ps(pC[0]);
                        _sum6 = _mm_set1_ps(pC[0]);
                        _sum7 = _mm_set1_ps(pC[0]);
                        _sum8 = _mm_set1_ps(pC[0]);
                        _sum9 = _mm_set1_ps(pC[0]);
                        _suma = _mm_set1_ps(pC[0]);
                        _sumb = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
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
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        _sum4 = _mm_loadu_ps(pC + 16);
                        _sum5 = _mm_loadu_ps(pC + 20);
                        _sum6 = _mm_loadu_ps(pC + 24);
                        _sum7 = _mm_loadu_ps(pC + 28);
                        _sum8 = _mm_loadu_ps(pC + 32);
                        _sum9 = _mm_loadu_ps(pC + 36);
                        _suma = _mm_loadu_ps(pC + 40);
                        _sumb = _mm_loadu_ps(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        _sum4 = _mm_set1_ps(pC[4]);
                        _sum5 = _mm_set1_ps(pC[5]);
                        _sum6 = _mm_set1_ps(pC[6]);
                        _sum7 = _mm_set1_ps(pC[7]);
                        _sum8 = _mm_set1_ps(pC[8]);
                        _sum9 = _mm_set1_ps(pC[9]);
                        _suma = _mm_set1_ps(pC[10]);
                        _sumb = _mm_set1_ps(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
                _sum8 = _mm_load_ps(outptr + 4 * 8);
                _sum9 = _mm_load_ps(outptr + 4 * 9);
                _suma = _mm_load_ps(outptr + 4 * 10);
                _sumb = _mm_load_ps(outptr + 4 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);
                _sum8 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[8]), _sum8);
                _sum9 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[9]), _sum9);
                _suma = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[10]), _suma);
                _sumb = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[11]), _sumb);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    _mm_store_ps(outptr0 + 4 * 4, _sum4);
                    _mm_store_ps(outptr0 + 4 * 5, _sum5);
                    _mm_store_ps(outptr0 + 4 * 6, _sum6);
                    _mm_store_ps(outptr0 + 4 * 7, _sum7);
                    _mm_store_ps(outptr0 + 4 * 8, _sum8);
                    _mm_store_ps(outptr0 + 4 * 9, _sum9);
                    _mm_store_ps(outptr0 + 4 * 10, _suma);
                    _mm_store_ps(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);
                    _MM_TRANSPOSE4_PS(_sum8, _sum9, _suma, _sumb);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    _mm_storeu_ps(outptr0 + 8, _sum8);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 8, _sum9);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 8, _suma);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 8, _sumb);
                    outptr0 += 12;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
                _mm_store_ps(outptr + 4 * 8, _sum8);
                _mm_store_ps(outptr + 4 * 9, _sum9);
                _mm_store_ps(outptr + 4 * 10, _suma);
                _mm_store_ps(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;
            __m128 _sum4;
            __m128 _sum5;
            __m128 _sum6;
            __m128 _sum7;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();
                _sum4 = _mm_setzero_ps();
                _sum5 = _mm_setzero_ps();
                _sum6 = _mm_setzero_ps();
                _sum7 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                        _sum4 = _mm_set1_ps(pC[0]);
                        _sum5 = _mm_set1_ps(pC[0]);
                        _sum6 = _mm_set1_ps(pC[0]);
                        _sum7 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
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
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        _sum4 = _mm_loadu_ps(pC + 16);
                        _sum5 = _mm_loadu_ps(pC + 20);
                        _sum6 = _mm_loadu_ps(pC + 24);
                        _sum7 = _mm_loadu_ps(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        _sum4 = _mm_set1_ps(pC[4]);
                        _sum5 = _mm_set1_ps(pC[5]);
                        _sum6 = _mm_set1_ps(pC[6]);
                        _sum7 = _mm_set1_ps(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
                _sum4 = _mm_load_ps(outptr + 4 * 4);
                _sum5 = _mm_load_ps(outptr + 4 * 5);
                _sum6 = _mm_load_ps(outptr + 4 * 6);
                _sum7 = _mm_load_ps(outptr + 4 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);
                _sum4 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[4]), _sum4);
                _sum5 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[5]), _sum5);
                _sum6 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[6]), _sum6);
                _sum7 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[7]), _sum7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    _mm_store_ps(outptr0 + 4 * 4, _sum4);
                    _mm_store_ps(outptr0 + 4 * 5, _sum5);
                    _mm_store_ps(outptr0 + 4 * 6, _sum6);
                    _mm_store_ps(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);
                    _MM_TRANSPOSE4_PS(_sum4, _sum5, _sum6, _sum7);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    _mm_storeu_ps(outptr0 + 4, _sum4);
                    _mm_storeu_ps(outptr0 + out_hstep * 1 + 4, _sum5);
                    _mm_storeu_ps(outptr0 + out_hstep * 2 + 4, _sum6);
                    _mm_storeu_ps(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
                _mm_store_ps(outptr + 4 * 4, _sum4);
                _mm_store_ps(outptr + 4 * 5, _sum5);
                _mm_store_ps(outptr + 4 * 6, _sum6);
                _mm_store_ps(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();
                _sum3 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                        _sum3 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        _sum3 = _mm_loadu_ps(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        _sum2 = _mm_set1_ps(pC[2]);
                        _sum3 = _mm_set1_ps(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4 * 1);
                _sum2 = _mm_load_ps(outptr + 4 * 2);
                _sum3 = _mm_load_ps(outptr + 4 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[2]), _sum2);
                _sum3 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[3]), _sum3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4 * 1, _sum1);
                    _mm_store_ps(outptr0 + 4 * 2, _sum2);
                    _mm_store_ps(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep * 1, _sum1);
                    _mm_storeu_ps(outptr0 + out_hstep * 2, _sum2);
                    _mm_storeu_ps(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4 * 1, _sum1);
                _mm_store_ps(outptr + 4 * 2, _sum2);
                _mm_store_ps(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
                _sum1 = _mm_load_ps(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[1]), _sum1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(16))
#else
                    __attribute__((aligned(16)))
#endif
                    float sumbuf[8];
                    float* sum0 = sumbuf;
                    float* sum1 = sumbuf + 4;
                    _mm_store_ps(sum0, _sum0);
                    _mm_store_ps(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
                _mm_store_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            __m128 _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = _mm_load_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = _mm_load_ps(pA);

                _sum0 = _mm_comp_fmadd_ps(_pA, _mm_set1_ps(pB[0]), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    _mm_store_ps(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
#ifdef _MSC_VER
                    __declspec(align(16))
#else
                    __attribute__((aligned(16)))
#endif
                    float sum0[4];
                    _mm_store_ps(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                _mm_store_ps(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

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
#if __SSE2__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0;
            __m512 _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();
                _sum1 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                        _sum1 = _mm512_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum0 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum1 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        _sum1 = _sum0;
                        pC += 16;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_loadu_ps(outptr);
                _sum1 = _mm512_loadu_ps(outptr + 16);
                __m512 _tmp0 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(2, 0, 2, 0));
                __m512 _tmp1 = _mm512_shuffle_f32x4(_sum0, _sum1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum0 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm512_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pB = _mm512_loadu_ps(pB);
                _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(pA[0]), _pB, _sum0);
                _sum1 = _mm512_fmadd_ps(_mm512_set1_ps(pA[1]), _pB, _sum1);

                pA += 2;
                pB += 16;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm512_storeu_ps(outptr0, _sum0);
                    _mm512_storeu_ps(outptr0 + out_hstep, _sum1);
                    outptr0 += 16;
                }
            }
            else
            {
                transpose16x2_ps(_sum0, _sum1);
                _mm512_storeu_ps(outptr, _sum0);
                _mm512_storeu_ps(outptr + 16, _sum1);
            }

            outptr += 32;
        }
#else  // __AVX512F__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum02;
            __m128 _sum10;
            __m128 _sum11;
            __m128 _sum12;

            if (k == 0)
            {
                _sum00 = _mm_setzero_ps();
                _sum01 = _mm_setzero_ps();
                _sum02 = _mm_setzero_ps();
                _sum10 = _mm_setzero_ps();
                _sum11 = _mm_setzero_ps();
                _sum12 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum02 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[0]);
                        _sum11 = _mm_set1_ps(pC[0]);
                        _sum12 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum02 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[1]);
                        _sum11 = _mm_set1_ps(pC[1]);
                        _sum12 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        __m128 _tmp2 = _mm_loadu_ps(pC + 8);
                        __m128 _tmp3 = _mm_loadu_ps(pC + 12);
                        __m128 _tmp4 = _mm_loadu_ps(pC + 16);
                        __m128 _tmp5 = _mm_loadu_ps(pC + 20);
                        _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum02 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum12 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = _mm_loadu_ps(pC);
                        _sum01 = _mm_loadu_ps(pC + 4);
                        _sum02 = _mm_loadu_ps(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                __m128 _tmp4 = _mm_loadu_ps(outptr + 16);
                __m128 _tmp5 = _mm_loadu_ps(outptr + 20);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum02 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                _sum12 = _mm_shuffle_ps(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);
                __m128 _pB2 = _mm_load_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                _sum02 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum02);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);
                _sum12 = _mm_comp_fmadd_ps(_pA1, _pB2, _sum12);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + 8, _sum02);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    _mm_storeu_ps(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                __m128 _tmp4 = _mm_unpacklo_ps(_sum02, _sum12);
                __m128 _tmp5 = _mm_unpackhi_ps(_sum02, _sum12);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
                _mm_store_ps(outptr + 16, _tmp4);
                _mm_store_ps(outptr + 20, _tmp5);
            }

            outptr += 24;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                _sum00 = _mm_setzero_ps();
                _sum01 = _mm_setzero_ps();
                _sum10 = _mm_setzero_ps();
                _sum11 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[0]);
                        _sum11 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = _mm_set1_ps(pC[0]);
                        _sum01 = _mm_set1_ps(pC[0]);
                        _sum10 = _mm_set1_ps(pC[1]);
                        _sum11 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        __m128 _tmp2 = _mm_loadu_ps(pC + 8);
                        __m128 _tmp3 = _mm_loadu_ps(pC + 12);
                        _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = _mm_loadu_ps(pC);
                        _sum01 = _mm_loadu_ps(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                __m128 _tmp2 = _mm_loadu_ps(outptr + 8);
                __m128 _tmp3 = _mm_loadu_ps(outptr + 12);
                _sum00 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum01 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _sum10 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                _sum11 = _mm_shuffle_ps(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum00);
                _sum01 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum01);
                __m128 _pA1 = _mm_set1_ps(pA[1]);
                _sum10 = _mm_comp_fmadd_ps(_pA1, _pB0, _sum10);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB1, _sum11);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum00);
                    _mm_storeu_ps(outptr0 + 4, _sum01);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum10);
                    _mm_storeu_ps(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum00, _sum10);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum00, _sum10);
                __m128 _tmp2 = _mm_unpacklo_ps(_sum01, _sum11);
                __m128 _tmp3 = _mm_unpackhi_ps(_sum01, _sum11);
                _mm_store_ps(outptr, _tmp0);
                _mm_store_ps(outptr + 4, _tmp1);
                _mm_store_ps(outptr + 8, _tmp2);
                _mm_store_ps(outptr + 12, _tmp3);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        __m128 _tmp0 = _mm_loadu_ps(pC);
                        __m128 _tmp1 = _mm_loadu_ps(pC + 4);
                        _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                __m128 _tmp0 = _mm_loadu_ps(outptr);
                __m128 _tmp1 = _mm_loadu_ps(outptr + 4);
                _sum0 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _sum1 = _mm_shuffle_ps(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_load_ps(pB);
                _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[1]), _pB, _sum1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                __m128 _tmp0 = _mm_unpacklo_ps(_sum0, _sum1);
                __m128 _tmp1 = _mm_unpackhi_ps(_sum0, _sum1);
                _mm_storeu_ps(outptr, _tmp0);
                _mm_storeu_ps(outptr + 4, _tmp1);
            }

            outptr += 8;
        }
#endif // __SSE2__
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
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

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
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
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

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
#if __SSE2__
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm512_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                }
            }
            else
            {
                _sum0 = _mm512_loadu_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            __m512 _sum00 = _mm512_setzero_ps();
            __m512 _sum01 = _mm512_setzero_ps();
            __m512 _sum02 = _mm512_setzero_ps();
            __m512 _sum03 = _mm512_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512 _pB0 = _mm512_loadu_ps(pB);
                __m512 _pB1 = _mm512_loadu_ps(pB + 16);
                __m512 _pB2 = _mm512_loadu_ps(pB + 32);
                __m512 _pB3 = _mm512_loadu_ps(pB + 48);
                _sum00 = _mm512_fmadd_ps(_mm512_set1_ps(pA[0]), _pB0, _sum00);
                _sum01 = _mm512_fmadd_ps(_mm512_set1_ps(pA[1]), _pB1, _sum01);
                _sum02 = _mm512_fmadd_ps(_mm512_set1_ps(pA[2]), _pB2, _sum02);
                _sum03 = _mm512_fmadd_ps(_mm512_set1_ps(pA[3]), _pB3, _sum03);

                pA += 4;
                pB += 64;
            }
            _sum00 = _mm512_add_ps(_sum00, _sum01);
            _sum02 = _mm512_add_ps(_sum02, _sum03);
            _sum00 = _mm512_add_ps(_sum00, _sum02);
            _sum0 = _mm512_add_ps(_sum0, _sum00);
            for (; kk < max_kk; kk += 1)
            {
                __m512 _pB = _mm512_loadu_ps(pB);
                _sum0 = _mm512_fmadd_ps(_mm512_set1_ps(pA[0]), _pB, _sum0);

                pA += 1;
                pB += 16;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm512_storeu_ps(outptr0, _sum0);
                    outptr0 += 16;
                }
            }
            else
            {
                _mm512_storeu_ps(outptr, _sum0);
            }

            outptr += 16;
        }
#else  // __AVX512F__
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();
                _sum2 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                        _sum2 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        _sum2 = _mm_loadu_ps(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
                _sum2 = _mm_loadu_ps(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            __m128 _sum00 = _mm_setzero_ps();
            __m128 _sum01 = _mm_setzero_ps();
            __m128 _sum02 = _mm_setzero_ps();
            __m128 _sum03 = _mm_setzero_ps();
            __m128 _sum10 = _mm_setzero_ps();
            __m128 _sum11 = _mm_setzero_ps();
            __m128 _sum12 = _mm_setzero_ps();
            __m128 _sum13 = _mm_setzero_ps();
            __m128 _sum20 = _mm_setzero_ps();
            __m128 _sum21 = _mm_setzero_ps();
            __m128 _sum22 = _mm_setzero_ps();
            __m128 _sum23 = _mm_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _pB00 = _mm_load_ps(pB);
                __m128 _pB10 = _mm_load_ps(pB + 4);
                __m128 _pB20 = _mm_load_ps(pB + 8);
                __m128 _pA0 = _mm_set1_ps(pA[0]);

                __m128 _pB01 = _mm_load_ps(pB + 12);
                __m128 _pB11 = _mm_load_ps(pB + 16);
                __m128 _pB21 = _mm_load_ps(pB + 20);
                __m128 _pA1 = _mm_set1_ps(pA[1]);

                __m128 _pB02 = _mm_load_ps(pB + 24);
                __m128 _pB12 = _mm_load_ps(pB + 28);
                __m128 _pB22 = _mm_load_ps(pB + 32);
                __m128 _pA2 = _mm_set1_ps(pA[2]);

                __m128 _pB03 = _mm_load_ps(pB + 36);
                __m128 _pB13 = _mm_load_ps(pB + 40);
                __m128 _pB23 = _mm_load_ps(pB + 44);
                __m128 _pA3 = _mm_set1_ps(pA[3]);

                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB00, _sum00);
                _sum10 = _mm_comp_fmadd_ps(_pA0, _pB10, _sum10);
                _sum20 = _mm_comp_fmadd_ps(_pA0, _pB20, _sum20);
                _sum01 = _mm_comp_fmadd_ps(_pA1, _pB01, _sum01);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB11, _sum11);
                _sum21 = _mm_comp_fmadd_ps(_pA1, _pB21, _sum21);
                _sum02 = _mm_comp_fmadd_ps(_pA2, _pB02, _sum02);
                _sum12 = _mm_comp_fmadd_ps(_pA2, _pB12, _sum12);
                _sum22 = _mm_comp_fmadd_ps(_pA2, _pB22, _sum22);
                _sum03 = _mm_comp_fmadd_ps(_pA3, _pB03, _sum03);
                _sum13 = _mm_comp_fmadd_ps(_pA3, _pB13, _sum13);
                _sum23 = _mm_comp_fmadd_ps(_pA3, _pB23, _sum23);

                pA += 4;
                pB += 48;
            }
            _sum00 = _mm_add_ps(_sum00, _sum01);
            _sum02 = _mm_add_ps(_sum02, _sum03);
            _sum10 = _mm_add_ps(_sum10, _sum11);
            _sum12 = _mm_add_ps(_sum12, _sum13);
            _sum20 = _mm_add_ps(_sum20, _sum21);
            _sum22 = _mm_add_ps(_sum22, _sum23);
            _sum00 = _mm_add_ps(_sum00, _sum02);
            _sum10 = _mm_add_ps(_sum10, _sum12);
            _sum20 = _mm_add_ps(_sum20, _sum22);
            _sum0 = _mm_add_ps(_sum0, _sum00);
            _sum1 = _mm_add_ps(_sum1, _sum10);
            _sum2 = _mm_add_ps(_sum2, _sum20);
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);
                __m128 _pB2 = _mm_load_ps(pB + 8);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_pA0, _pB2, _sum2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    _mm_storeu_ps(outptr0 + 8, _sum2);
                    outptr0 += 12;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
                _mm_storeu_ps(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_ps();
                _sum1 = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = _mm_set1_ps(pC[0]);
                        _sum1 = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = _mm_loadu_ps(pC);
                        _sum1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = _mm_loadu_ps(outptr);
                _sum1 = _mm_loadu_ps(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            __m128 _sum00 = _mm_setzero_ps();
            __m128 _sum01 = _mm_setzero_ps();
            __m128 _sum02 = _mm_setzero_ps();
            __m128 _sum03 = _mm_setzero_ps();
            __m128 _sum10 = _mm_setzero_ps();
            __m128 _sum11 = _mm_setzero_ps();
            __m128 _sum12 = _mm_setzero_ps();
            __m128 _sum13 = _mm_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _pB00 = _mm_load_ps(pB);
                __m128 _pB10 = _mm_load_ps(pB + 4);
                __m128 _pA0 = _mm_set1_ps(pA[0]);

                __m128 _pB01 = _mm_load_ps(pB + 8);
                __m128 _pB11 = _mm_load_ps(pB + 12);
                __m128 _pA1 = _mm_set1_ps(pA[1]);

                __m128 _pB02 = _mm_load_ps(pB + 16);
                __m128 _pB12 = _mm_load_ps(pB + 20);
                __m128 _pA2 = _mm_set1_ps(pA[2]);

                __m128 _pB03 = _mm_load_ps(pB + 24);
                __m128 _pB13 = _mm_load_ps(pB + 28);
                __m128 _pA3 = _mm_set1_ps(pA[3]);

                _sum00 = _mm_comp_fmadd_ps(_pA0, _pB00, _sum00);
                _sum10 = _mm_comp_fmadd_ps(_pA0, _pB10, _sum10);
                _sum01 = _mm_comp_fmadd_ps(_pA1, _pB01, _sum01);
                _sum11 = _mm_comp_fmadd_ps(_pA1, _pB11, _sum11);
                _sum02 = _mm_comp_fmadd_ps(_pA2, _pB02, _sum02);
                _sum12 = _mm_comp_fmadd_ps(_pA2, _pB12, _sum12);
                _sum03 = _mm_comp_fmadd_ps(_pA3, _pB03, _sum03);
                _sum13 = _mm_comp_fmadd_ps(_pA3, _pB13, _sum13);

                pA += 4;
                pB += 32;
            }
            _sum00 = _mm_add_ps(_sum00, _sum01);
            _sum02 = _mm_add_ps(_sum02, _sum03);
            _sum10 = _mm_add_ps(_sum10, _sum11);
            _sum12 = _mm_add_ps(_sum12, _sum13);
            _sum00 = _mm_add_ps(_sum00, _sum02);
            _sum10 = _mm_add_ps(_sum10, _sum12);
            _sum0 = _mm_add_ps(_sum0, _sum00);
            _sum1 = _mm_add_ps(_sum1, _sum10);
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);

                __m128 _pA0 = _mm_set1_ps(pA[0]);
                _sum0 = _mm_comp_fmadd_ps(_pA0, _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_pA0, _pB1, _sum1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum0);
                    _mm_storeu_ps(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum0);
                _mm_storeu_ps(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum;

            if (k == 0)
            {
                _sum = _mm_setzero_ps();

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = _mm_set1_ps(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = _mm_loadu_ps(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            __m128 _sum0 = _mm_setzero_ps();
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128 _pB0 = _mm_load_ps(pB);
                __m128 _pB1 = _mm_load_ps(pB + 4);
                __m128 _pB2 = _mm_load_ps(pB + 8);
                __m128 _pB3 = _mm_load_ps(pB + 12);
                _sum0 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[1]), _pB1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[2]), _pB2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_mm_set1_ps(pA[3]), _pB3, _sum3);

                pA += 4;
                pB += 16;
            }
            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _sum0 = _mm_add_ps(_sum0, _sum2);
            _sum = _mm_add_ps(_sum, _sum0);
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = _mm_load_ps(pB);
                _sum = _mm_comp_fmadd_ps(_mm_set1_ps(pA[0]), _pB, _sum);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    _mm_storeu_ps(outptr0, _sum);
                    outptr0 += 4;
                }
            }
            else
            {
                _mm_storeu_ps(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __SSE2__
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

            const float* pA = pAT;
            int kk = 0;
            float sum00 = 0.f;
            float sum01 = 0.f;
            float sum02 = 0.f;
            float sum03 = 0.f;
            float sum10 = 0.f;
            float sum11 = 0.f;
            float sum12 = 0.f;
            float sum13 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[0] * pB[1];
                sum01 += pA[1] * pB[2];
                sum11 += pA[1] * pB[3];
                sum02 += pA[2] * pB[4];
                sum12 += pA[2] * pB[5];
                sum03 += pA[3] * pB[6];
                sum13 += pA[3] * pB[7];

                pA += 4;
                pB += 8;
            }
            sum00 += sum01;
            sum02 += sum03;
            sum10 += sum11;
            sum12 += sum13;
            sum0 += sum00 + sum02;
            sum1 += sum10 + sum12;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
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

            const float* pA = pAT;
            int kk = 0;
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[1];
                sum2 += pA[2] * pB[2];
                sum3 += pA[3] * pB[3];

                pA += 4;
                pB += 4;
            }
            sum0 += sum1;
            sum2 += sum3;
            sum += sum0 + sum2;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
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

