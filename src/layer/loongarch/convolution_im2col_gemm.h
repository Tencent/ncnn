// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_im2col_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    float* pp = AT;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            __m256 _r0 = (__m256)__lasx_xvld(p0, 0);
            __m256 _r1 = (__m256)__lasx_xvld(p1, 0);
            __m256 _r2 = (__m256)__lasx_xvld(p2, 0);
            __m256 _r3 = (__m256)__lasx_xvld(p3, 0);
            __m256 _r4 = (__m256)__lasx_xvld(p4, 0);
            __m256 _r5 = (__m256)__lasx_xvld(p5, 0);
            __m256 _r6 = (__m256)__lasx_xvld(p6, 0);
            __m256 _r7 = (__m256)__lasx_xvld(p7, 0);
            transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __lasx_xvst((__m256i)_r0, pp, 0);
            __lasx_xvst((__m256i)_r1, pp + 8, 0);
            __lasx_xvst((__m256i)_r2, pp + 8 * 2, 0);
            __lasx_xvst((__m256i)_r3, pp + 8 * 3, 0);
            __lasx_xvst((__m256i)_r4, pp + 8 * 4, 0);
            __lasx_xvst((__m256i)_r5, pp + 8 * 5, 0);
            __lasx_xvst((__m256i)_r6, pp + 8 * 6, 0);
            __lasx_xvst((__m256i)_r7, pp + 8 * 7, 0);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __m128 _r1 = (__m128)__lsx_vld(p1, 0);
            __m128 _r2 = (__m128)__lsx_vld(p2, 0);
            __m128 _r3 = (__m128)__lsx_vld(p3, 0);
            transpose4x4_ps(_r0, _r1, _r2, _r3);
            __lsx_vst((__m128i)_r0, pp, 0);
            __lsx_vst((__m128i)_r1, pp + 4, 0);
            __lsx_vst((__m128i)_r2, pp + 8, 0);
            __lsx_vst((__m128i)_r3, pp + 12, 0);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __m128 _r0 = (__m128)__lsx_vld(p0, 0);
            __m128 _r1 = (__m128)__lsx_vld(p1, 0);
            __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
            __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
            __lsx_vst((__m128i)_tmp0, pp, 0);
            __lsx_vst((__m128i)_tmp1, pp + 4, 0);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __loongarch_sx
        for (; kk + 3 < max_kk; kk += 4)
        {
            __lsx_vst((__m128i)(__m128)__lsx_vld(p0, 0), pp, 0);
            pp += 4;
            p0 += 4;
        }
#endif // __loongarch_sx
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const float* pA = pAT;

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
            __m256 _sum10;
            __m256 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _sum10 = _sum0;
                    _sum11 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                    _sum4 = (__m256)__lasx_xvldi(0);
                    _sum5 = (__m256)__lasx_xvldi(0);
                    _sum6 = (__m256)__lasx_xvldi(0);
                    _sum7 = (__m256)__lasx_xvldi(0);
                    _sum8 = (__m256)__lasx_xvldi(0);
                    _sum9 = (__m256)__lasx_xvldi(0);
                    _sum10 = (__m256)__lasx_xvldi(0);
                    _sum11 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
                _sum8 = (__m256)__lasx_xvld(outptr + 64, 0);
                _sum9 = (__m256)__lasx_xvld(outptr + 72, 0);
                _sum10 = (__m256)__lasx_xvld(outptr + 80, 0);
                _sum11 = (__m256)__lasx_xvld(outptr + 88, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[3]), _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[4]), _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[5]), _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[6]), _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[7]), _sum7);
                _sum8 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[8]), _sum8);
                _sum9 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[9]), _sum9);
                _sum10 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[10]), _sum10);
                _sum11 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[11]), _sum11);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr0 + 0, 0);
                    __lasx_xvst((__m256i)_sum1, outptr0 + 8, 0);
                    __lasx_xvst((__m256i)_sum2, outptr0 + 16, 0);
                    __lasx_xvst((__m256i)_sum3, outptr0 + 24, 0);
                    __lasx_xvst((__m256i)_sum4, outptr0 + 32, 0);
                    __lasx_xvst((__m256i)_sum5, outptr0 + 40, 0);
                    __lasx_xvst((__m256i)_sum6, outptr0 + 48, 0);
                    __lasx_xvst((__m256i)_sum7, outptr0 + 56, 0);
                    __lasx_xvst((__m256i)_sum8, outptr0 + 64, 0);
                    __lasx_xvst((__m256i)_sum9, outptr0 + 72, 0);
                    __lasx_xvst((__m256i)_sum10, outptr0 + 80, 0);
                    __lasx_xvst((__m256i)_sum11, outptr0 + 88, 0);
                    outptr0 += 96;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0l = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x20);
                    __m256i _tmp0h = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x31);
                    __m256i _tmp2l = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x20);
                    __m256i _tmp2h = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x31);
                    __m256i _tmp4l = __lasx_xvpermi_q((__m256i)_sum5, (__m256i)_sum4, 0x20);
                    __m256i _tmp4h = __lasx_xvpermi_q((__m256i)_sum5, (__m256i)_sum4, 0x31);
                    __m256i _tmp6l = __lasx_xvpermi_q((__m256i)_sum7, (__m256i)_sum6, 0x20);
                    __m256i _tmp6h = __lasx_xvpermi_q((__m256i)_sum7, (__m256i)_sum6, 0x31);
                    __m256i _tmp8l = __lasx_xvpermi_q((__m256i)_sum9, (__m256i)_sum8, 0x20);
                    __m256i _tmp8h = __lasx_xvpermi_q((__m256i)_sum9, (__m256i)_sum8, 0x31);
                    __m256i _tmp10l = __lasx_xvpermi_q((__m256i)_sum11, (__m256i)_sum10, 0x20);
                    __m256i _tmp10h = __lasx_xvpermi_q((__m256i)_sum11, (__m256i)_sum10, 0x31);
                    __lasx_xvst(_tmp0l, outptr0 + 0, 0);
                    __lasx_xvst(_tmp2l, outptr0 + 8, 0);
                    __lasx_xvst(_tmp4l, outptr0 + 16, 0);
                    __lasx_xvst(_tmp6l, outptr0 + 24, 0);
                    __lasx_xvst(_tmp8l, outptr0 + 32, 0);
                    __lasx_xvst(_tmp10l, outptr0 + 40, 0);
                    __lasx_xvst(_tmp0h, outptr0 + out_hstep * 4 + 0, 0);
                    __lasx_xvst(_tmp2h, outptr0 + out_hstep * 4 + 8, 0);
                    __lasx_xvst(_tmp4h, outptr0 + out_hstep * 4 + 16, 0);
                    __lasx_xvst(_tmp6h, outptr0 + out_hstep * 4 + 24, 0);
                    __lasx_xvst(_tmp8h, outptr0 + out_hstep * 4 + 32, 0);
                    __lasx_xvst(_tmp10h, outptr0 + out_hstep * 4 + 40, 0);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    float sum4[8];
                    __lasx_xvst((__m256i)_sum4, sum4, 0);
                    float sum5[8];
                    __lasx_xvst((__m256i)_sum5, sum5, 0);
                    float sum6[8];
                    __lasx_xvst((__m256i)_sum6, sum6, 0);
                    float sum7[8];
                    __lasx_xvst((__m256i)_sum7, sum7, 0);
                    float sum8[8];
                    __lasx_xvst((__m256i)_sum8, sum8, 0);
                    float sum9[8];
                    __lasx_xvst((__m256i)_sum9, sum9, 0);
                    float sum10[8];
                    __lasx_xvst((__m256i)_sum10, sum10, 0);
                    float sum11[8];
                    __lasx_xvst((__m256i)_sum11, sum11, 0);
                    outptr0[out_hstep * 0 + 0] = sum0[0];
                    outptr0[out_hstep * 0 + 1] = sum1[0];
                    outptr0[out_hstep * 0 + 2] = sum2[0];
                    outptr0[out_hstep * 0 + 3] = sum3[0];
                    outptr0[out_hstep * 0 + 4] = sum4[0];
                    outptr0[out_hstep * 0 + 5] = sum5[0];
                    outptr0[out_hstep * 0 + 6] = sum6[0];
                    outptr0[out_hstep * 0 + 7] = sum7[0];
                    outptr0[out_hstep * 0 + 8] = sum8[0];
                    outptr0[out_hstep * 0 + 9] = sum9[0];
                    outptr0[out_hstep * 0 + 10] = sum10[0];
                    outptr0[out_hstep * 0 + 11] = sum11[0];
                    outptr0[out_hstep * 1 + 0] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 1 + 2] = sum2[1];
                    outptr0[out_hstep * 1 + 3] = sum3[1];
                    outptr0[out_hstep * 1 + 4] = sum4[1];
                    outptr0[out_hstep * 1 + 5] = sum5[1];
                    outptr0[out_hstep * 1 + 6] = sum6[1];
                    outptr0[out_hstep * 1 + 7] = sum7[1];
                    outptr0[out_hstep * 1 + 8] = sum8[1];
                    outptr0[out_hstep * 1 + 9] = sum9[1];
                    outptr0[out_hstep * 1 + 10] = sum10[1];
                    outptr0[out_hstep * 1 + 11] = sum11[1];
                    outptr0[out_hstep * 2 + 0] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 2 + 2] = sum2[2];
                    outptr0[out_hstep * 2 + 3] = sum3[2];
                    outptr0[out_hstep * 2 + 4] = sum4[2];
                    outptr0[out_hstep * 2 + 5] = sum5[2];
                    outptr0[out_hstep * 2 + 6] = sum6[2];
                    outptr0[out_hstep * 2 + 7] = sum7[2];
                    outptr0[out_hstep * 2 + 8] = sum8[2];
                    outptr0[out_hstep * 2 + 9] = sum9[2];
                    outptr0[out_hstep * 2 + 10] = sum10[2];
                    outptr0[out_hstep * 2 + 11] = sum11[2];
                    outptr0[out_hstep * 3 + 0] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 3 + 2] = sum2[3];
                    outptr0[out_hstep * 3 + 3] = sum3[3];
                    outptr0[out_hstep * 3 + 4] = sum4[3];
                    outptr0[out_hstep * 3 + 5] = sum5[3];
                    outptr0[out_hstep * 3 + 6] = sum6[3];
                    outptr0[out_hstep * 3 + 7] = sum7[3];
                    outptr0[out_hstep * 3 + 8] = sum8[3];
                    outptr0[out_hstep * 3 + 9] = sum9[3];
                    outptr0[out_hstep * 3 + 10] = sum10[3];
                    outptr0[out_hstep * 3 + 11] = sum11[3];
                    outptr0[out_hstep * 4 + 0] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 4 + 2] = sum2[4];
                    outptr0[out_hstep * 4 + 3] = sum3[4];
                    outptr0[out_hstep * 4 + 4] = sum4[4];
                    outptr0[out_hstep * 4 + 5] = sum5[4];
                    outptr0[out_hstep * 4 + 6] = sum6[4];
                    outptr0[out_hstep * 4 + 7] = sum7[4];
                    outptr0[out_hstep * 4 + 8] = sum8[4];
                    outptr0[out_hstep * 4 + 9] = sum9[4];
                    outptr0[out_hstep * 4 + 10] = sum10[4];
                    outptr0[out_hstep * 4 + 11] = sum11[4];
                    outptr0[out_hstep * 5 + 0] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 5 + 2] = sum2[5];
                    outptr0[out_hstep * 5 + 3] = sum3[5];
                    outptr0[out_hstep * 5 + 4] = sum4[5];
                    outptr0[out_hstep * 5 + 5] = sum5[5];
                    outptr0[out_hstep * 5 + 6] = sum6[5];
                    outptr0[out_hstep * 5 + 7] = sum7[5];
                    outptr0[out_hstep * 5 + 8] = sum8[5];
                    outptr0[out_hstep * 5 + 9] = sum9[5];
                    outptr0[out_hstep * 5 + 10] = sum10[5];
                    outptr0[out_hstep * 5 + 11] = sum11[5];
                    outptr0[out_hstep * 6 + 0] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 6 + 2] = sum2[6];
                    outptr0[out_hstep * 6 + 3] = sum3[6];
                    outptr0[out_hstep * 6 + 4] = sum4[6];
                    outptr0[out_hstep * 6 + 5] = sum5[6];
                    outptr0[out_hstep * 6 + 6] = sum6[6];
                    outptr0[out_hstep * 6 + 7] = sum7[6];
                    outptr0[out_hstep * 6 + 8] = sum8[6];
                    outptr0[out_hstep * 6 + 9] = sum9[6];
                    outptr0[out_hstep * 6 + 10] = sum10[6];
                    outptr0[out_hstep * 6 + 11] = sum11[6];
                    outptr0[out_hstep * 7 + 0] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 7 + 2] = sum2[7];
                    outptr0[out_hstep * 7 + 3] = sum3[7];
                    outptr0[out_hstep * 7 + 4] = sum4[7];
                    outptr0[out_hstep * 7 + 5] = sum5[7];
                    outptr0[out_hstep * 7 + 6] = sum6[7];
                    outptr0[out_hstep * 7 + 7] = sum7[7];
                    outptr0[out_hstep * 7 + 8] = sum8[7];
                    outptr0[out_hstep * 7 + 9] = sum9[7];
                    outptr0[out_hstep * 7 + 10] = sum10[7];
                    outptr0[out_hstep * 7 + 11] = sum11[7];
                    outptr0 += 12;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
                __lasx_xvst((__m256i)_sum4, outptr + 32, 0);
                __lasx_xvst((__m256i)_sum5, outptr + 40, 0);
                __lasx_xvst((__m256i)_sum6, outptr + 48, 0);
                __lasx_xvst((__m256i)_sum7, outptr + 56, 0);
                __lasx_xvst((__m256i)_sum8, outptr + 64, 0);
                __lasx_xvst((__m256i)_sum9, outptr + 72, 0);
                __lasx_xvst((__m256i)_sum10, outptr + 80, 0);
                __lasx_xvst((__m256i)_sum11, outptr + 88, 0);
                outptr += 96;
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pA = pAT;

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
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                    _sum4 = (__m256)__lasx_xvldi(0);
                    _sum5 = (__m256)__lasx_xvldi(0);
                    _sum6 = (__m256)__lasx_xvldi(0);
                    _sum7 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
                _sum4 = (__m256)__lasx_xvld(outptr + 32, 0);
                _sum5 = (__m256)__lasx_xvld(outptr + 40, 0);
                _sum6 = (__m256)__lasx_xvld(outptr + 48, 0);
                _sum7 = (__m256)__lasx_xvld(outptr + 56, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[3]), _sum3);
                _sum4 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[4]), _sum4);
                _sum5 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[5]), _sum5);
                _sum6 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[6]), _sum6);
                _sum7 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[7]), _sum7);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr0 + 0, 0);
                    __lasx_xvst((__m256i)_sum1, outptr0 + 8, 0);
                    __lasx_xvst((__m256i)_sum2, outptr0 + 16, 0);
                    __lasx_xvst((__m256i)_sum3, outptr0 + 24, 0);
                    __lasx_xvst((__m256i)_sum4, outptr0 + 32, 0);
                    __lasx_xvst((__m256i)_sum5, outptr0 + 40, 0);
                    __lasx_xvst((__m256i)_sum6, outptr0 + 48, 0);
                    __lasx_xvst((__m256i)_sum7, outptr0 + 56, 0);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0l = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x20);
                    __m256i _tmp0h = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x31);
                    __m256i _tmp2l = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x20);
                    __m256i _tmp2h = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x31);
                    __m256i _tmp4l = __lasx_xvpermi_q((__m256i)_sum5, (__m256i)_sum4, 0x20);
                    __m256i _tmp4h = __lasx_xvpermi_q((__m256i)_sum5, (__m256i)_sum4, 0x31);
                    __m256i _tmp6l = __lasx_xvpermi_q((__m256i)_sum7, (__m256i)_sum6, 0x20);
                    __m256i _tmp6h = __lasx_xvpermi_q((__m256i)_sum7, (__m256i)_sum6, 0x31);

                    __lasx_xvst(_tmp0l, outptr0 + 0, 0);
                    __lasx_xvst(_tmp2l, outptr0 + 8, 0);
                    __lasx_xvst(_tmp4l, outptr0 + 16, 0);
                    __lasx_xvst(_tmp6l, outptr0 + 24, 0);
                    __lasx_xvst(_tmp0h, outptr0 + out_hstep * 4 + 0, 0);
                    __lasx_xvst(_tmp2h, outptr0 + out_hstep * 4 + 8, 0);
                    __lasx_xvst(_tmp4h, outptr0 + out_hstep * 4 + 16, 0);
                    __lasx_xvst(_tmp6h, outptr0 + out_hstep * 4 + 24, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    float sum4[8];
                    __lasx_xvst((__m256i)_sum4, sum4, 0);
                    float sum5[8];
                    __lasx_xvst((__m256i)_sum5, sum5, 0);
                    float sum6[8];
                    __lasx_xvst((__m256i)_sum6, sum6, 0);
                    float sum7[8];
                    __lasx_xvst((__m256i)_sum7, sum7, 0);
                    outptr0[out_hstep * 0 + 0] = sum0[0];
                    outptr0[out_hstep * 0 + 1] = sum1[0];
                    outptr0[out_hstep * 0 + 2] = sum2[0];
                    outptr0[out_hstep * 0 + 3] = sum3[0];
                    outptr0[out_hstep * 0 + 4] = sum4[0];
                    outptr0[out_hstep * 0 + 5] = sum5[0];
                    outptr0[out_hstep * 0 + 6] = sum6[0];
                    outptr0[out_hstep * 0 + 7] = sum7[0];
                    outptr0[out_hstep * 1 + 0] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 1 + 2] = sum2[1];
                    outptr0[out_hstep * 1 + 3] = sum3[1];
                    outptr0[out_hstep * 1 + 4] = sum4[1];
                    outptr0[out_hstep * 1 + 5] = sum5[1];
                    outptr0[out_hstep * 1 + 6] = sum6[1];
                    outptr0[out_hstep * 1 + 7] = sum7[1];
                    outptr0[out_hstep * 2 + 0] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 2 + 2] = sum2[2];
                    outptr0[out_hstep * 2 + 3] = sum3[2];
                    outptr0[out_hstep * 2 + 4] = sum4[2];
                    outptr0[out_hstep * 2 + 5] = sum5[2];
                    outptr0[out_hstep * 2 + 6] = sum6[2];
                    outptr0[out_hstep * 2 + 7] = sum7[2];
                    outptr0[out_hstep * 3 + 0] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 3 + 2] = sum2[3];
                    outptr0[out_hstep * 3 + 3] = sum3[3];
                    outptr0[out_hstep * 3 + 4] = sum4[3];
                    outptr0[out_hstep * 3 + 5] = sum5[3];
                    outptr0[out_hstep * 3 + 6] = sum6[3];
                    outptr0[out_hstep * 3 + 7] = sum7[3];
                    outptr0[out_hstep * 4 + 0] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 4 + 2] = sum2[4];
                    outptr0[out_hstep * 4 + 3] = sum3[4];
                    outptr0[out_hstep * 4 + 4] = sum4[4];
                    outptr0[out_hstep * 4 + 5] = sum5[4];
                    outptr0[out_hstep * 4 + 6] = sum6[4];
                    outptr0[out_hstep * 4 + 7] = sum7[4];
                    outptr0[out_hstep * 5 + 0] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 5 + 2] = sum2[5];
                    outptr0[out_hstep * 5 + 3] = sum3[5];
                    outptr0[out_hstep * 5 + 4] = sum4[5];
                    outptr0[out_hstep * 5 + 5] = sum5[5];
                    outptr0[out_hstep * 5 + 6] = sum6[5];
                    outptr0[out_hstep * 5 + 7] = sum7[5];
                    outptr0[out_hstep * 6 + 0] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 6 + 2] = sum2[6];
                    outptr0[out_hstep * 6 + 3] = sum3[6];
                    outptr0[out_hstep * 6 + 4] = sum4[6];
                    outptr0[out_hstep * 6 + 5] = sum5[6];
                    outptr0[out_hstep * 6 + 6] = sum6[6];
                    outptr0[out_hstep * 6 + 7] = sum7[6];
                    outptr0[out_hstep * 7 + 0] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 7 + 2] = sum2[7];
                    outptr0[out_hstep * 7 + 3] = sum3[7];
                    outptr0[out_hstep * 7 + 4] = sum4[7];
                    outptr0[out_hstep * 7 + 5] = sum5[7];
                    outptr0[out_hstep * 7 + 6] = sum6[7];
                    outptr0[out_hstep * 7 + 7] = sum7[7];
                    outptr0 += 8;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
                __lasx_xvst((__m256i)_sum4, outptr + 32, 0);
                __lasx_xvst((__m256i)_sum5, outptr + 40, 0);
                __lasx_xvst((__m256i)_sum6, outptr + 48, 0);
                __lasx_xvst((__m256i)_sum7, outptr + 56, 0);
                outptr += 64;
            }
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;
            __m256 _sum2;
            __m256 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                    _sum2 = (__m256)__lasx_xvldi(0);
                    _sum3 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
                _sum2 = (__m256)__lasx_xvld(outptr + 16, 0);
                _sum3 = (__m256)__lasx_xvld(outptr + 24, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[3]), _sum3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr0 + 0, 0);
                    __lasx_xvst((__m256i)_sum1, outptr0 + 8, 0);
                    __lasx_xvst((__m256i)_sum2, outptr0 + 16, 0);
                    __lasx_xvst((__m256i)_sum3, outptr0 + 24, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0l = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x20);
                    __m256i _tmp0h = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x31);
                    __m256i _tmp2l = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x20);
                    __m256i _tmp2h = __lasx_xvpermi_q((__m256i)_sum3, (__m256i)_sum2, 0x31);

                    __lasx_xvst(_tmp0l, outptr0 + 0, 0);
                    __lasx_xvst(_tmp2l, outptr0 + 8, 0);
                    __lasx_xvst(_tmp0h, outptr0 + out_hstep * 4 + 0, 0);
                    __lasx_xvst(_tmp2h, outptr0 + out_hstep * 4 + 8, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    float sum2[8];
                    __lasx_xvst((__m256i)_sum2, sum2, 0);
                    float sum3[8];
                    __lasx_xvst((__m256i)_sum3, sum3, 0);
                    outptr0[out_hstep * 0 + 0] = sum0[0];
                    outptr0[out_hstep * 0 + 1] = sum1[0];
                    outptr0[out_hstep * 0 + 2] = sum2[0];
                    outptr0[out_hstep * 0 + 3] = sum3[0];
                    outptr0[out_hstep * 1 + 0] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 1 + 2] = sum2[1];
                    outptr0[out_hstep * 1 + 3] = sum3[1];
                    outptr0[out_hstep * 2 + 0] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 2 + 2] = sum2[2];
                    outptr0[out_hstep * 2 + 3] = sum3[2];
                    outptr0[out_hstep * 3 + 0] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 3 + 2] = sum2[3];
                    outptr0[out_hstep * 3 + 3] = sum3[3];
                    outptr0[out_hstep * 4 + 0] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 4 + 2] = sum2[4];
                    outptr0[out_hstep * 4 + 3] = sum3[4];
                    outptr0[out_hstep * 5 + 0] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 5 + 2] = sum2[5];
                    outptr0[out_hstep * 5 + 3] = sum3[5];
                    outptr0[out_hstep * 6 + 0] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 6 + 2] = sum2[6];
                    outptr0[out_hstep * 6 + 3] = sum3[6];
                    outptr0[out_hstep * 7 + 0] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0[out_hstep * 7 + 2] = sum2[7];
                    outptr0[out_hstep * 7 + 3] = sum3[7];
                    outptr0 += 4;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                __lasx_xvst((__m256i)_sum2, outptr + 16, 0);
                __lasx_xvst((__m256i)_sum3, outptr + 24, 0);
                outptr += 32;
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            __m256 _sum0;
            __m256 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                    _sum1 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
                _sum1 = (__m256)__lasx_xvld(outptr + 8, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[1]), _sum1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr0 + 0, 0);
                    __lasx_xvst((__m256i)_sum1, outptr0 + 8, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256i _tmp0l = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x20);
                    __m256i _tmp0h = __lasx_xvpermi_q((__m256i)_sum1, (__m256i)_sum0, 0x31);

                    __lasx_xvst(_tmp0l, outptr0 + 0, 0);
                    __lasx_xvst(_tmp0h, outptr0 + out_hstep * 4 + 0, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    float sum1[8];
                    __lasx_xvst((__m256i)_sum1, sum1, 0);
                    outptr0[out_hstep * 0 + 0] = sum0[0];
                    outptr0[out_hstep * 0 + 1] = sum1[0];
                    outptr0[out_hstep * 1 + 0] = sum0[1];
                    outptr0[out_hstep * 1 + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 0] = sum0[2];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 0] = sum0[3];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 0] = sum0[4];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 0] = sum0[5];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 0] = sum0[6];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 0] = sum0[7];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                __lasx_xvst((__m256i)_sum1, outptr + 8, 0);
                outptr += 16;
            }
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            __m256 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m256)__lasx_xvld(pC, 0);
                }
                else
                {
                    _sum0 = (__m256)__lasx_xvldi(0);
                }
            }
            else
            {
                _sum0 = (__m256)__lasx_xvld(outptr + 0, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m256 _pA = (__m256)__lasx_xvld(pA, 0);

                _sum0 = __lasx_xvfmadd_s(_pA, (__m256)__lasx_xvreplfr2vr_s(pB[0]), _sum0);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    __lasx_xvst((__m256i)_sum0, outptr0 + 0, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    __m128i _sum0l = __lasx_extract_lo128((__m256i)_sum0);
                    __m128i _sum0h = __lasx_extract_hi128((__m256i)_sum0);
                    __lsx_vst(_sum0l, outptr0, 0);
                    __lsx_vst(_sum0h, outptr0 + out_hstep * 4, 0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    __lasx_xvst((__m256i)_sum0, sum0, 0);
                    outptr0[out_hstep * 0 + 0] = sum0[0];
                    outptr0[out_hstep * 1 + 0] = sum0[1];
                    outptr0[out_hstep * 2 + 0] = sum0[2];
                    outptr0[out_hstep * 3 + 0] = sum0[3];
                    outptr0[out_hstep * 4 + 0] = sum0[4];
                    outptr0[out_hstep * 5 + 0] = sum0[5];
                    outptr0[out_hstep * 6 + 0] = sum0[6];
                    outptr0[out_hstep * 7 + 0] = sum0[7];
                    outptr0 += 1;
                }
            }
            else
            {
                __lasx_xvst((__m256i)_sum0, outptr + 0, 0);
                outptr += 8;
            }
        }

        pAT += max_kk * 8;
    }
#endif // __loongarch_asx
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const float* pA = pAT;

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
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
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
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum4 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum5 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum6 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum7 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum8 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum9 = (__m128)__lsx_vreplgr2vr_w(0);
                    _suma = (__m128)__lsx_vreplgr2vr_w(0);
                    _sumb = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4 * 1, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 4 * 2, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 4 * 3, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 4 * 4, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 4 * 5, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 4 * 6, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 4 * 7, 0);
                _sum8 = (__m128)__lsx_vld(outptr + 4 * 8, 0);
                _sum9 = (__m128)__lsx_vld(outptr + 4 * 9, 0);
                _suma = (__m128)__lsx_vld(outptr + 4 * 10, 0);
                _sumb = (__m128)__lsx_vld(outptr + 4 * 11, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[3]), _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[4]), _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[5]), _sum5);
                _sum6 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[6]), _sum6);
                _sum7 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[7]), _sum7);
                _sum8 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[8]), _sum8);
                _sum9 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[9]), _sum9);
                _suma = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[10]), _suma);
                _sumb = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[11]), _sumb);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + 4 * 3, 0);
                    __lsx_vst((__m128i)_sum4, outptr0 + 4 * 4, 0);
                    __lsx_vst((__m128i)_sum5, outptr0 + 4 * 5, 0);
                    __lsx_vst((__m128i)_sum6, outptr0 + 4 * 6, 0);
                    __lsx_vst((__m128i)_sum7, outptr0 + 4 * 7, 0);
                    __lsx_vst((__m128i)_sum8, outptr0 + 4 * 8, 0);
                    __lsx_vst((__m128i)_sum9, outptr0 + 4 * 9, 0);
                    __lsx_vst((__m128i)_suma, outptr0 + 4 * 10, 0);
                    __lsx_vst((__m128i)_sumb, outptr0 + 4 * 11, 0);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                    transpose4x4_ps(_sum8, _sum9, _suma, _sumb);

                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + out_hstep * 1, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + out_hstep * 3, 0);
                    __lsx_vst((__m128i)_sum4, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum5, outptr0 + out_hstep * 1 + 4, 0);
                    __lsx_vst((__m128i)_sum6, outptr0 + out_hstep * 2 + 4, 0);
                    __lsx_vst((__m128i)_sum7, outptr0 + out_hstep * 3 + 4, 0);
                    __lsx_vst((__m128i)_sum8, outptr0 + 8, 0);
                    __lsx_vst((__m128i)_sum9, outptr0 + out_hstep * 1 + 8, 0);
                    __lsx_vst((__m128i)_suma, outptr0 + out_hstep * 2 + 8, 0);
                    __lsx_vst((__m128i)_sumb, outptr0 + out_hstep * 3 + 8, 0);
                    outptr0 += 12;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
                __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
                __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
                __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
                __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);
                __lsx_vst((__m128i)_sum8, outptr + 4 * 8, 0);
                __lsx_vst((__m128i)_sum9, outptr + 4 * 9, 0);
                __lsx_vst((__m128i)_suma, outptr + 4 * 10, 0);
                __lsx_vst((__m128i)_sumb, outptr + 4 * 11, 0);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const float* pA = pAT;

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
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum4 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum5 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum6 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum7 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4 * 1, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 4 * 2, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 4 * 3, 0);
                _sum4 = (__m128)__lsx_vld(outptr + 4 * 4, 0);
                _sum5 = (__m128)__lsx_vld(outptr + 4 * 5, 0);
                _sum6 = (__m128)__lsx_vld(outptr + 4 * 6, 0);
                _sum7 = (__m128)__lsx_vld(outptr + 4 * 7, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[3]), _sum3);
                _sum4 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[4]), _sum4);
                _sum5 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[5]), _sum5);
                _sum6 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[6]), _sum6);
                _sum7 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[7]), _sum7);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + 4 * 3, 0);
                    __lsx_vst((__m128i)_sum4, outptr0 + 4 * 4, 0);
                    __lsx_vst((__m128i)_sum5, outptr0 + 4 * 5, 0);
                    __lsx_vst((__m128i)_sum6, outptr0 + 4 * 6, 0);
                    __lsx_vst((__m128i)_sum7, outptr0 + 4 * 7, 0);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                    transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + out_hstep * 1, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + out_hstep * 3, 0);
                    __lsx_vst((__m128i)_sum4, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum5, outptr0 + out_hstep * 1 + 4, 0);
                    __lsx_vst((__m128i)_sum6, outptr0 + out_hstep * 2 + 4, 0);
                    __lsx_vst((__m128i)_sum7, outptr0 + out_hstep * 3 + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
                __lsx_vst((__m128i)_sum4, outptr + 4 * 4, 0);
                __lsx_vst((__m128i)_sum5, outptr + 4 * 5, 0);
                __lsx_vst((__m128i)_sum6, outptr + 4 * 6, 0);
                __lsx_vst((__m128i)_sum7, outptr + 4 * 7, 0);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const float* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;
            __m128 _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4 * 1, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 4 * 2, 0);
                _sum3 = (__m128)__lsx_vld(outptr + 4 * 3, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);
                _sum2 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[2]), _sum2);
                _sum3 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[3]), _sum3);

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 4 * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + 4 * 3, 0);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + out_hstep * 1, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + out_hstep * 2, 0);
                    __lsx_vst((__m128i)_sum3, outptr0 + out_hstep * 3, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 4 * 2, 0);
                __lsx_vst((__m128i)_sum3, outptr + 4 * 3, 0);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const float* pA = pAT;

            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);
                _sum1 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[1]), _sum1);

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    __lsx_vst((__m128i)_sum0, sum0, 0);
                    __lsx_vst((__m128i)_sum1, sum1, 0);

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
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const float* pA = pAT;

            __m128 _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vld(pC, 0);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pA = (__m128)__lsx_vld(pA, 0);

                _sum0 = __lsx_vfmadd_s(_pA, (__m128)__lsx_vreplfr2vr_s(pB[0]), _sum0);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    __lsx_vst((__m128i)_sum0, sum0, 0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __loongarch_sx
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __loongarch_sx
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
                if (pC)
                {
                    _sum00 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum01 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum02 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum10 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                    _sum11 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                    _sum12 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                }
                else
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum02 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum12 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                __m128 _tmp2 = (__m128)__lsx_vld(outptr + 8, 0);
                __m128 _tmp3 = (__m128)__lsx_vld(outptr + 12, 0);
                __m128 _tmp4 = (__m128)__lsx_vld(outptr + 16, 0);
                __m128 _tmp5 = (__m128)__lsx_vld(outptr + 20, 0);
                _sum00 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum01 = (__m128)__lsx_vpickev_w((__m128i)_tmp3, (__m128i)_tmp2);
                _sum02 = (__m128)__lsx_vpickev_w((__m128i)_tmp5, (__m128i)_tmp4);
                _sum10 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vpickod_w((__m128i)_tmp3, (__m128i)_tmp2);
                _sum12 = (__m128)__lsx_vpickod_w((__m128i)_tmp5, (__m128i)_tmp4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                __m128 _pB2 = (__m128)__lsx_vld(pB + 8, 0);

                __m128 _pA0 = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA0, _pB1, _sum01);
                _sum02 = __lsx_vfmadd_s(_pA0, _pB2, _sum02);
                __m128 _pA1 = (__m128)__lsx_vreplfr2vr_s(pA[1]);
                _sum10 = __lsx_vfmadd_s(_pA1, _pB0, _sum10);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);
                _sum12 = __lsx_vfmadd_s(_pA1, _pB2, _sum12);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum00, outptr0, 0);
                    __lsx_vst((__m128i)_sum01, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum02, outptr0 + 8, 0);
                    __lsx_vst((__m128i)_sum10, outptr0 + out_hstep, 0);
                    __lsx_vst((__m128i)_sum11, outptr0 + out_hstep + 4, 0);
                    __lsx_vst((__m128i)_sum12, outptr0 + out_hstep + 8, 0);
                    outptr0 += 12;
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum00);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum00);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum01);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum01);
                __m128 _tmp4 = (__m128)__lsx_vilvl_w((__m128i)_sum12, (__m128i)_sum02);
                __m128 _tmp5 = (__m128)__lsx_vilvh_w((__m128i)_sum12, (__m128i)_sum02);
                __lsx_vst((__m128i)_tmp0, outptr, 0);
                __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
                __lsx_vst((__m128i)_tmp2, outptr + 8, 0);
                __lsx_vst((__m128i)_tmp3, outptr + 12, 0);
                __lsx_vst((__m128i)_tmp4, outptr + 16, 0);
                __lsx_vst((__m128i)_tmp5, outptr + 20, 0);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum00;
            __m128 _sum01;
            __m128 _sum10;
            __m128 _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum01 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum10 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                    _sum11 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                }
                else
                {
                    _sum00 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum01 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum10 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum11 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                __m128 _tmp2 = (__m128)__lsx_vld(outptr + 8, 0);
                __m128 _tmp3 = (__m128)__lsx_vld(outptr + 12, 0);
                _sum00 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum01 = (__m128)__lsx_vpickev_w((__m128i)_tmp3, (__m128i)_tmp2);
                _sum10 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum11 = (__m128)__lsx_vpickod_w((__m128i)_tmp3, (__m128i)_tmp2);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);

                __m128 _pA0 = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                _sum00 = __lsx_vfmadd_s(_pA0, _pB0, _sum00);
                _sum01 = __lsx_vfmadd_s(_pA0, _pB1, _sum01);
                __m128 _pA1 = (__m128)__lsx_vreplfr2vr_s(pA[1]);
                _sum10 = __lsx_vfmadd_s(_pA1, _pB0, _sum10);
                _sum11 = __lsx_vfmadd_s(_pA1, _pB1, _sum11);

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum00, outptr0, 0);
                    __lsx_vst((__m128i)_sum01, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum10, outptr0 + out_hstep, 0);
                    __lsx_vst((__m128i)_sum11, outptr0 + out_hstep + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum10, (__m128i)_sum00);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum10, (__m128i)_sum00);
                __m128 _tmp2 = (__m128)__lsx_vilvl_w((__m128i)_sum11, (__m128i)_sum01);
                __m128 _tmp3 = (__m128)__lsx_vilvh_w((__m128i)_sum11, (__m128i)_sum01);
                __lsx_vst((__m128i)_tmp0, outptr, 0);
                __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
                __lsx_vst((__m128i)_tmp2, outptr + 8, 0);
                __lsx_vst((__m128i)_tmp3, outptr + 12, 0);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum1 = (__m128)__lsx_vreplfr2vr_s(pC[1]);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vld(outptr, 0);
                __m128 _tmp1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum0 = (__m128)__lsx_vpickev_w((__m128i)_tmp1, (__m128i)_tmp0);
                _sum1 = (__m128)__lsx_vpickod_w((__m128i)_tmp1, (__m128i)_tmp0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = (__m128)__lsx_vld(pB, 0);

                _sum0 = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(pA[0]), _pB, _sum0);
                _sum1 = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(pA[1]), _pB, _sum1);

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + out_hstep, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_sum1, (__m128i)_sum0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_sum1, (__m128i)_sum0);
                __lsx_vst((__m128i)_tmp0, outptr, 0);
                __lsx_vst((__m128i)_tmp1, outptr + 4, 0);
            }

            outptr += 8;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
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
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
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
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __loongarch_sx
        for (; jj + 11 < max_jj; jj += 12)
        {
            __m128 _sum0;
            __m128 _sum1;
            __m128 _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum1 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum2 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
                _sum2 = (__m128)__lsx_vld(outptr + 8, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);
                __m128 _pB2 = (__m128)__lsx_vld(pB + 8, 0);

                __m128 _pA0 = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);
                _sum2 = __lsx_vfmadd_s(_pA0, _pB2, _sum2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    __lsx_vst((__m128i)_sum2, outptr0 + 8, 0);
                    outptr0 += 12;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
                __lsx_vst((__m128i)_sum2, outptr + 8, 0);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _sum0;
            __m128 _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                    _sum1 = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                }
                else
                {
                    _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum0 = (__m128)__lsx_vld(outptr, 0);
                _sum1 = (__m128)__lsx_vld(outptr + 4, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB0 = (__m128)__lsx_vld(pB, 0);
                __m128 _pB1 = (__m128)__lsx_vld(pB + 4, 0);

                __m128 _pA0 = (__m128)__lsx_vreplfr2vr_s(pA[0]);
                _sum0 = __lsx_vfmadd_s(_pA0, _pB0, _sum0);
                _sum1 = __lsx_vfmadd_s(_pA0, _pB1, _sum1);

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum0, outptr0, 0);
                    __lsx_vst((__m128i)_sum1, outptr0 + 4, 0);
                    outptr0 += 8;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum0, outptr, 0);
                __lsx_vst((__m128i)_sum1, outptr + 4, 0);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = (__m128)__lsx_vreplfr2vr_s(pC[0]);
                }
                else
                {
                    _sum = (__m128)__lsx_vreplgr2vr_w(0);
                }
            }
            else
            {
                _sum = (__m128)__lsx_vld(outptr, 0);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __m128 _pB = (__m128)__lsx_vld(pB, 0);

                _sum = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(pA[0]), _pB, _sum);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    __lsx_vst((__m128i)_sum, outptr0, 0);
                    outptr0 += 4;
                }
            }
            else
            {
                __lsx_vst((__m128i)_sum, outptr, 0);
            }

            outptr += 4;
        }
#endif // __loongarch_sx
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
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
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
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

static void convolution_im2col_gemm_get_optimal_tile_mnk(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_fp32 = (int)(get_cpu_level2_cache_size() / sizeof(float));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __loongarch_sx
#if __loongarch_asx
        int tile_size = (l2_cache_size_fp32 - 32) / 16;
#else
        int tile_size = (l2_cache_size_fp32 - 16) / 8;
#endif
#else
        int tile_size = (l2_cache_size_fp32 - 2) / 3;
#endif

#if __loongarch_sx
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __loongarch_sx
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __loongarch_sx
#if __loongarch_asx
        int nn_M = (M + 7) / 8;
#else
        int nn_M = (M + 3) / 4;
#endif
#else
        int nn_M = (M + 3) / 4;
#endif

#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __loongarch_sx
#if __loongarch_asx
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#endif
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __loongarch_sx
#if __loongarch_asx
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#endif
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_fp32 - TILE_M * TILE_K) / (TILE_M + TILE_K);
        }

#if __loongarch_sx
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __loongarch_sx
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif

#if __loongarch_sx
        TILE_N = std::max(4, TILE_N);
#else
        TILE_N = std::max(1, TILE_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    float* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 4 * 2, 0);
                __m128 _r3 = (__m128)__lsx_vld(p0 + 4 * 3, 0);
                __m128 _r4 = (__m128)__lsx_vld(p0 + 4 * 4, 0);
                __m128 _r5 = (__m128)__lsx_vld(p0 + 4 * 5, 0);
                __m128 _r6 = (__m128)__lsx_vld(p0 + 4 * 6, 0);
                __m128 _r7 = (__m128)__lsx_vld(p0 + 4 * 7, 0);
                __m128 _r8 = (__m128)__lsx_vld(p0 + 4 * 8, 0);
                __m128 _r9 = (__m128)__lsx_vld(p0 + 4 * 9, 0);
                __m128 _ra = (__m128)__lsx_vld(p0 + 4 * 10, 0);
                __m128 _rb = (__m128)__lsx_vld(p0 + 4 * 11, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                transpose4x4_ps(_r8, _r9, _ra, _rb);
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                __lsx_vst((__m128i)_r8, pp + 4 * 2, 0);
                __lsx_vst((__m128i)_r1, pp + 4 * 3, 0);
                __lsx_vst((__m128i)_r5, pp + 4 * 4, 0);
                __lsx_vst((__m128i)_r9, pp + 4 * 5, 0);
                __lsx_vst((__m128i)_r2, pp + 4 * 6, 0);
                __lsx_vst((__m128i)_r6, pp + 4 * 7, 0);
                __lsx_vst((__m128i)_ra, pp + 4 * 8, 0);
                __lsx_vst((__m128i)_r3, pp + 4 * 9, 0);
                __lsx_vst((__m128i)_r7, pp + 4 * 10, 0);
                __lsx_vst((__m128i)_rb, pp + 4 * 11, 0);
                pp += 48;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 8, 0);
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r1, pp + 4, 0);
                __lsx_vst((__m128i)_r2, pp + 8, 0);
                pp += 12;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 4 * 2, 0);
                __m128 _r3 = (__m128)__lsx_vld(p0 + 4 * 3, 0);
                __m128 _r4 = (__m128)__lsx_vld(p0 + 4 * 4, 0);
                __m128 _r5 = (__m128)__lsx_vld(p0 + 4 * 5, 0);
                __m128 _r6 = (__m128)__lsx_vld(p0 + 4 * 6, 0);
                __m128 _r7 = (__m128)__lsx_vld(p0 + 4 * 7, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                __lsx_vst((__m128i)_r1, pp + 4 * 2, 0);
                __lsx_vst((__m128i)_r5, pp + 4 * 3, 0);
                __lsx_vst((__m128i)_r2, pp + 4 * 4, 0);
                __lsx_vst((__m128i)_r6, pp + 4 * 5, 0);
                __lsx_vst((__m128i)_r3, pp + 4 * 6, 0);
                __lsx_vst((__m128i)_r7, pp + 4 * 7, 0);
                pp += 32;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r1, pp + 4, 0);
                pp += 8;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _r2 = (__m128)__lsx_vld(p0 + 4 * 2, 0);
                __m128 _r3 = (__m128)__lsx_vld(p0 + 4 * 3, 0);
                transpose4x4_ps(_r0, _r1, _r2, _r3);
                __lsx_vst((__m128i)_r0, pp, 0);
                __lsx_vst((__m128i)_r1, pp + 4 * 1, 0);
                __lsx_vst((__m128i)_r2, pp + 4 * 2, 0);
                __lsx_vst((__m128i)_r3, pp + 4 * 3, 0);
                pp += 16;
                p0 += bottom_blob.cstep * 4;
            }
        }

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __lsx_vst((__m128i)(__m128)__lsx_vld(p0, 0), pp, 0);
                pp += 4;
                p0 += bottom_blob.cstep;
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                // transpose4x2
                __m128 _r0 = (__m128)__lsx_vld(p0, 0);
                __m128 _r1 = (__m128)__lsx_vld(p0 + 4, 0);
                __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                __lsx_vst((__m128i)_tmp0, pp, 0);
                __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                pp += 8;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __loongarch_sx

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += bottom_blob.cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __loongarch_sx
        if (elempack == 4)
        {
            const float* p0 = (const float*)bottom_blob.channel(k / 4) + (j + jj) * 4;

            int kk = 0;
            for (; kk < max_kk / 4; kk++)
            {
                __lsx_vst((__m128i)(__m128)__lsx_vld(p0, 0), pp, 0);
                pp += 4;
                p0 += bottom_blob.cstep * 4;
            }
        }
#endif // __loongarch_sx

        if (elempack == 1)
        {
            const float* p0 = (const float*)bottom_blob.channel(k) + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += bottom_blob.cstep;
            }
        }
    }
}

static inline void convolution_im2col_input_tile_impl(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    float* pp = B;

    int jj = 0;
#if __loongarch_sx
    for (; jj + 11 < max_jj; jj += 12)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dy8 = (j + jj + 8) / outw;
        int dy9 = (j + jj + 9) / outw;
        int dya = (j + jj + 10) / outw;
        int dyb = (j + jj + 11) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;
        int dx8 = (j + jj + 8) % outw;
        int dx9 = (j + jj + 9) % outw;
        int dxa = (j + jj + 10) % outw;
        int dxb = (j + jj + 11) % outw;

        if (dy0 == dyb)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr + stride_w * 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr + stride_w * 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr + stride_w * 12, 0);
                    __m128 _r4 = (__m128)__lsx_vld(sptr + stride_w * 16, 0);
                    __m128 _r5 = (__m128)__lsx_vld(sptr + stride_w * 20, 0);
                    __m128 _r6 = (__m128)__lsx_vld(sptr + stride_w * 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(sptr + stride_w * 28, 0);
                    __m128 _r8 = (__m128)__lsx_vld(sptr + stride_w * 32, 0);
                    __m128 _r9 = (__m128)__lsx_vld(sptr + stride_w * 36, 0);
                    __m128 _ra = (__m128)__lsx_vld(sptr + stride_w * 40, 0);
                    __m128 _rb = (__m128)__lsx_vld(sptr + stride_w * 44, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r8, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 3, 0);
                    __lsx_vst((__m128i)_r5, pp + 4 * 4, 0);
                    __lsx_vst((__m128i)_r9, pp + 4 * 5, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 6, 0);
                    __lsx_vst((__m128i)_r6, pp + 4 * 7, 0);
                    __lsx_vst((__m128i)_ra, pp + 4 * 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 9, 0);
                    __lsx_vst((__m128i)_r7, pp + 4 * 10, 0);
                    __lsx_vst((__m128i)_rb, pp + 4 * 11, 0);
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp[8] = sptr[stride_w * 8];
                    pp[9] = sptr[stride_w * 9];
                    pp[10] = sptr[stride_w * 10];
                    pp[11] = sptr[stride_w * 11];
                    pp += 12;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int x8 = stride_w * dx8 + dilation_w * v;
                int x9 = stride_w * dx9 + dilation_w * v;
                int xa = stride_w * dxa + dilation_w * v;
                int xb = stride_w * dxb + dilation_w * v;

                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;
                int y8 = stride_h * dy8 + dilation_h * u;
                int y9 = stride_h * dy9 + dilation_h * u;
                int ya = stride_h * dya + dilation_h * u;
                int yb = stride_h * dyb + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;
                const float* sptr4 = img.row(y4) + x4 * elempack;
                const float* sptr5 = img.row(y5) + x5 * elempack;
                const float* sptr6 = img.row(y6) + x6 * elempack;
                const float* sptr7 = img.row(y7) + x7 * elempack;
                const float* sptr8 = img.row(y8) + x8 * elempack;
                const float* sptr9 = img.row(y9) + x9 * elempack;
                const float* sptra = img.row(ya) + xa * elempack;
                const float* sptrb = img.row(yb) + xb * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr3, 0);
                    __m128 _r4 = (__m128)__lsx_vld(sptr4, 0);
                    __m128 _r5 = (__m128)__lsx_vld(sptr5, 0);
                    __m128 _r6 = (__m128)__lsx_vld(sptr6, 0);
                    __m128 _r7 = (__m128)__lsx_vld(sptr7, 0);
                    __m128 _r8 = (__m128)__lsx_vld(sptr8, 0);
                    __m128 _r9 = (__m128)__lsx_vld(sptr9, 0);
                    __m128 _ra = (__m128)__lsx_vld(sptra, 0);
                    __m128 _rb = (__m128)__lsx_vld(sptrb, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    transpose4x4_ps(_r8, _r9, _ra, _rb);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r8, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 3, 0);
                    __lsx_vst((__m128i)_r5, pp + 4 * 4, 0);
                    __lsx_vst((__m128i)_r9, pp + 4 * 5, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 6, 0);
                    __lsx_vst((__m128i)_r6, pp + 4 * 7, 0);
                    __lsx_vst((__m128i)_ra, pp + 4 * 8, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 9, 0);
                    __lsx_vst((__m128i)_r7, pp + 4 * 10, 0);
                    __lsx_vst((__m128i)_rb, pp + 4 * 11, 0);
                    pp += 48;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr8[0];
                    pp[9] = sptr9[0];
                    pp[10] = sptra[0];
                    pp[11] = sptrb[0];
                    pp += 12;
                }
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dy4 = (j + jj + 4) / outw;
        int dy5 = (j + jj + 5) / outw;
        int dy6 = (j + jj + 6) / outw;
        int dy7 = (j + jj + 7) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;
        int dx4 = (j + jj + 4) % outw;
        int dx5 = (j + jj + 5) % outw;
        int dx6 = (j + jj + 6) % outw;
        int dx7 = (j + jj + 7) % outw;

        if (dy0 == dy7)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr + stride_w * 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr + stride_w * 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr + stride_w * 12, 0);
                    __m128 _r4 = (__m128)__lsx_vld(sptr + stride_w * 16, 0);
                    __m128 _r5 = (__m128)__lsx_vld(sptr + stride_w * 20, 0);
                    __m128 _r6 = (__m128)__lsx_vld(sptr + stride_w * 24, 0);
                    __m128 _r7 = (__m128)__lsx_vld(sptr + stride_w * 28, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r5, pp + 4 * 3, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 4, 0);
                    __lsx_vst((__m128i)_r6, pp + 4 * 5, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 6, 0);
                    __lsx_vst((__m128i)_r7, pp + 4 * 7, 0);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp[4] = sptr[stride_w * 4];
                    pp[5] = sptr[stride_w * 5];
                    pp[6] = sptr[stride_w * 6];
                    pp[7] = sptr[stride_w * 7];
                    pp += 8;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int x4 = stride_w * dx4 + dilation_w * v;
                int x5 = stride_w * dx5 + dilation_w * v;
                int x6 = stride_w * dx6 + dilation_w * v;
                int x7 = stride_w * dx7 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;
                int y4 = stride_h * dy4 + dilation_h * u;
                int y5 = stride_h * dy5 + dilation_h * u;
                int y6 = stride_h * dy6 + dilation_h * u;
                int y7 = stride_h * dy7 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;
                const float* sptr4 = img.row(y4) + x4 * elempack;
                const float* sptr5 = img.row(y5) + x5 * elempack;
                const float* sptr6 = img.row(y6) + x6 * elempack;
                const float* sptr7 = img.row(y7) + x7 * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr3, 0);
                    __m128 _r4 = (__m128)__lsx_vld(sptr4, 0);
                    __m128 _r5 = (__m128)__lsx_vld(sptr5, 0);
                    __m128 _r6 = (__m128)__lsx_vld(sptr6, 0);
                    __m128 _r7 = (__m128)__lsx_vld(sptr7, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r4, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r5, pp + 4 * 3, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 4, 0);
                    __lsx_vst((__m128i)_r6, pp + 4 * 5, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 6, 0);
                    __lsx_vst((__m128i)_r7, pp + 4 * 7, 0);
                    pp += 32;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp += 8;
                }
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dy2 = (j + jj + 2) / outw;
        int dy3 = (j + jj + 3) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;
        int dx2 = (j + jj + 2) % outw;
        int dx3 = (j + jj + 3) % outw;

        if (dy0 == dy3)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr + stride_w * 4, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr + stride_w * 8, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr + stride_w * 12, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 3, 0);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp[2] = sptr[stride_w * 2];
                    pp[3] = sptr[stride_w * 3];
                    pp += 4;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int x2 = stride_w * dx2 + dilation_w * v;
                int x3 = stride_w * dx3 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;
                int y2 = stride_h * dy2 + dilation_h * u;
                int y3 = stride_h * dy3 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;
                const float* sptr2 = img.row(y2) + x2 * elempack;
                const float* sptr3 = img.row(y3) + x3 * elempack;

                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr1, 0);
                    __m128 _r2 = (__m128)__lsx_vld(sptr2, 0);
                    __m128 _r3 = (__m128)__lsx_vld(sptr3, 0);
                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    __lsx_vst((__m128i)_r0, pp, 0);
                    __lsx_vst((__m128i)_r1, pp + 4 * 1, 0);
                    __lsx_vst((__m128i)_r2, pp + 4 * 2, 0);
                    __lsx_vst((__m128i)_r3, pp + 4 * 3, 0);
                    pp += 16;
                }
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp += 4;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw;
        int dy1 = (j + jj + 1) / outw;
        int dx0 = (j + jj) % outw;
        int dx1 = (j + jj + 1) % outw;

        if (dy0 == dy1)
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;

                const float* sptr = img.row(y0) + x0 * elempack;

#if __loongarch_sx
                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr + stride_w * 4, 0);
                    __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __lsx_vst((__m128i)_tmp0, pp, 0);
                    __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                    pp += 8;
                }
#endif // __loongarch_sx
                if (elempack == 1)
                {
                    pp[0] = sptr[0];
                    pp[1] = sptr[stride_w];
                    pp += 2;
                }
            }
        }
        else
        {
            int kk = 0;
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = stride_w * dx0 + dilation_w * v;
                int x1 = stride_w * dx1 + dilation_w * v;
                int y0 = stride_h * dy0 + dilation_h * u;
                int y1 = stride_h * dy1 + dilation_h * u;

                const float* sptr0 = img.row(y0) + x0 * elempack;
                const float* sptr1 = img.row(y1) + x1 * elempack;

#if __loongarch_sx
                if (elempack == 4)
                {
                    __m128 _r0 = (__m128)__lsx_vld(sptr0, 0);
                    __m128 _r1 = (__m128)__lsx_vld(sptr1, 0);
                    __m128 _tmp0 = (__m128)__lsx_vilvl_w((__m128i)_r1, (__m128i)_r0);
                    __m128 _tmp1 = (__m128)__lsx_vilvh_w((__m128i)_r1, (__m128i)_r0);
                    __lsx_vst((__m128i)_tmp0, pp, 0);
                    __lsx_vst((__m128i)_tmp1, pp + 4, 0);
                    pp += 8;
                }
#endif // __loongarch_sx
                if (elempack == 1)
                {
                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp += 2;
                }
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
        int dy = (j + jj) / outw;
        int dx = (j + jj) % outw;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = stride_w * dx + dilation_w * v;
            int y = stride_h * dy + dilation_h * u;

            const float* sptr = img.row(y) + x * elempack;

#if __loongarch_sx
            if (elempack == 4)
            {
                __lsx_vst((__m128i)(__m128)__lsx_vld(sptr, 0), pp, 0);
                pp += 4;
            }
#endif // __loongarch_sx
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

template void convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);

static void convolution_im2col_input_tile(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        convolution_im2col_input_tile<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    convolution_im2col_input_tile_impl(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

static void convolution_im2col_gemm_transform_kernel(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch);

        for (int q = 0; q < outch; q += 1)
        {
            float* g00 = A_data.row(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static int convolution_im2col_gemm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
    {
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
        if (topT_tileX.empty())
            return -100;
    }

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }

    return 0;
}
