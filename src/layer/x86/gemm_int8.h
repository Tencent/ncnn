// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_avx512vnni(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void pack_A_tile_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avxvnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_avxvnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_avxvnni(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_avx2(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void unpack_output_tile_int32_to_fp32_avx2(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose);
void gemm_transB_packed_tile_int8_avx2(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void gemm_transB_packed_tile_int8_xop(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if __AVX512F__
static void print(__m512 x)
{
    float a[16];
    _mm512_storeu_ps(a, x);
    for (int i = 0; i < 16; i++)
    {
        fprintf(stderr, "%.0f ", a[i]);
    }
    fprintf(stderr, "\n");
}
#endif

#if __AVX__
static void print(__m256 x)
{
    float a[8];
    _mm256_storeu_ps(a, x);
    for (int i = 0; i < 8; i++)
    {
        fprintf(stderr, "%.0f ", a[i]);
    }
    fprintf(stderr, "\n");
}
#endif

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_A_tile_int8_avx512vnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_A_tile_int8_avxvnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_A_tile_int8_avx2(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    signed char* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;
        const signed char* p4 = A.row<const signed char>(i + ii + 4) + k;
        const signed char* p5 = A.row<const signed char>(i + ii + 5) + k;
        const signed char* p6 = A.row<const signed char>(i + ii + 6) + k;
        const signed char* p7 = A.row<const signed char>(i + ii + 7) + k;
        const signed char* p8 = A.row<const signed char>(i + ii + 8) + k;
        const signed char* p9 = A.row<const signed char>(i + ii + 9) + k;
        const signed char* pa = A.row<const signed char>(i + ii + 10) + k;
        const signed char* pb = A.row<const signed char>(i + ii + 11) + k;
        const signed char* pc = A.row<const signed char>(i + ii + 12) + k;
        const signed char* pd = A.row<const signed char>(i + ii + 13) + k;
        const signed char* pe = A.row<const signed char>(i + ii + 14) + k;
        const signed char* pf = A.row<const signed char>(i + ii + 15) + k;

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        int w_shift4 = 0;
        int w_shift5 = 0;
        int w_shift6 = 0;
        int w_shift7 = 0;
        int w_shift8 = 0;
        int w_shift9 = 0;
        int w_shifta = 0;
        int w_shiftb = 0;
        int w_shiftc = 0;
        int w_shiftd = 0;
        int w_shifte = 0;
        int w_shiftf = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp[8] = p2[0];
            pp[9] = p2[1];
            pp[10] = p2[2];
            pp[11] = p2[3];
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];

            pp[16 + 0] = p4[0];
            pp[16 + 1] = p4[1];
            pp[16 + 2] = p4[2];
            pp[16 + 3] = p4[3];
            pp[16 + 4] = p5[0];
            pp[16 + 5] = p5[1];
            pp[16 + 6] = p5[2];
            pp[16 + 7] = p5[3];
            pp[16 + 8] = p6[0];
            pp[16 + 9] = p6[1];
            pp[16 + 10] = p6[2];
            pp[16 + 11] = p6[3];
            pp[16 + 12] = p7[0];
            pp[16 + 13] = p7[1];
            pp[16 + 14] = p7[2];
            pp[16 + 15] = p7[3];

            pp[32 + 0] = p8[0];
            pp[32 + 1] = p8[1];
            pp[32 + 2] = p8[2];
            pp[32 + 3] = p8[3];
            pp[32 + 4] = p9[0];
            pp[32 + 5] = p9[1];
            pp[32 + 6] = p9[2];
            pp[32 + 7] = p9[3];
            pp[32 + 8] = pa[0];
            pp[32 + 9] = pa[1];
            pp[32 + 10] = pa[2];
            pp[32 + 11] = pa[3];
            pp[32 + 12] = pb[0];
            pp[32 + 13] = pb[1];
            pp[32 + 14] = pb[2];
            pp[32 + 15] = pb[3];

            pp[48 + 0] = pc[0];
            pp[48 + 1] = pc[1];
            pp[48 + 2] = pc[2];
            pp[48 + 3] = pc[3];
            pp[48 + 4] = pd[0];
            pp[48 + 5] = pd[1];
            pp[48 + 6] = pd[2];
            pp[48 + 7] = pd[3];
            pp[48 + 8] = pe[0];
            pp[48 + 9] = pe[1];
            pp[48 + 10] = pe[2];
            pp[48 + 11] = pe[3];
            pp[48 + 12] = pf[0];
            pp[48 + 13] = pf[1];
            pp[48 + 14] = pf[2];
            pp[48 + 15] = pf[3];

            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            w_shift4 += pp[16];
            w_shift4 += pp[17];
            w_shift4 += pp[18];
            w_shift4 += pp[19];
            w_shift5 += pp[20];
            w_shift5 += pp[21];
            w_shift5 += pp[22];
            w_shift5 += pp[23];
            w_shift6 += pp[24];
            w_shift6 += pp[25];
            w_shift6 += pp[26];
            w_shift6 += pp[27];
            w_shift7 += pp[28];
            w_shift7 += pp[29];
            w_shift7 += pp[30];
            w_shift7 += pp[31];

            w_shift8 += pp[32 + 0];
            w_shift8 += pp[32 + 1];
            w_shift8 += pp[32 + 2];
            w_shift8 += pp[32 + 3];
            w_shift9 += pp[32 + 4];
            w_shift9 += pp[32 + 5];
            w_shift9 += pp[32 + 6];
            w_shift9 += pp[32 + 7];
            w_shifta += pp[32 + 8];
            w_shifta += pp[32 + 9];
            w_shifta += pp[32 + 10];
            w_shifta += pp[32 + 11];
            w_shiftb += pp[32 + 12];
            w_shiftb += pp[32 + 13];
            w_shiftb += pp[32 + 14];
            w_shiftb += pp[32 + 15];
            w_shiftc += pp[32 + 16];
            w_shiftc += pp[32 + 17];
            w_shiftc += pp[32 + 18];
            w_shiftc += pp[32 + 19];
            w_shiftd += pp[32 + 20];
            w_shiftd += pp[32 + 21];
            w_shiftd += pp[32 + 22];
            w_shiftd += pp[32 + 23];
            w_shifte += pp[32 + 24];
            w_shifte += pp[32 + 25];
            w_shifte += pp[32 + 26];
            w_shifte += pp[32 + 27];
            w_shiftf += pp[32 + 28];
            w_shiftf += pp[32 + 29];
            w_shiftf += pp[32 + 30];
            w_shiftf += pp[32 + 31];

            pp += 64;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
            p8 += 4;
            p9 += 4;
            pa += 4;
            pb += 4;
            pc += 4;
            pd += 4;
            pe += 4;
            pf += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            ((int*)pp)[4] = w_shift4 * 127;
            ((int*)pp)[5] = w_shift5 * 127;
            ((int*)pp)[6] = w_shift6 * 127;
            ((int*)pp)[7] = w_shift7 * 127;
            ((int*)pp)[8] = w_shift8 * 127;
            ((int*)pp)[9] = w_shift9 * 127;
            ((int*)pp)[10] = w_shifta * 127;
            ((int*)pp)[11] = w_shiftb * 127;
            ((int*)pp)[12] = w_shiftc * 127;
            ((int*)pp)[13] = w_shiftd * 127;
            ((int*)pp)[14] = w_shifte * 127;
            ((int*)pp)[15] = w_shiftf * 127;
            pp += 64;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];

            pp[16 + 0] = p8[0];
            pp[16 + 1] = p8[1];
            pp[16 + 2] = p9[0];
            pp[16 + 3] = p9[1];
            pp[16 + 4] = pa[0];
            pp[16 + 5] = pa[1];
            pp[16 + 6] = pb[0];
            pp[16 + 7] = pb[1];
            pp[16 + 8] = pc[0];
            pp[16 + 9] = pc[1];
            pp[16 + 10] = pd[0];
            pp[16 + 11] = pd[1];
            pp[16 + 12] = pe[0];
            pp[16 + 13] = pe[1];
            pp[16 + 14] = pf[0];
            pp[16 + 15] = pf[1];

            pp += 32;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
            p8 += 2;
            p9 += 2;
            pa += 2;
            pb += 2;
            pc += 2;
            pd += 2;
            pe += 2;
            pf += 2;
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
            pp[8] = p8[0];
            pp[9] = p9[0];
            pp[10] = pa[0];
            pp[11] = pb[0];
            pp[12] = pc[0];
            pp[13] = pd[0];
            pp[14] = pe[0];
            pp[15] = pf[0];
            pp += 16;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
            pc++;
            pd++;
            pe++;
            pf++;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;
        const signed char* p4 = A.row<const signed char>(i + ii + 4) + k;
        const signed char* p5 = A.row<const signed char>(i + ii + 5) + k;
        const signed char* p6 = A.row<const signed char>(i + ii + 6) + k;
        const signed char* p7 = A.row<const signed char>(i + ii + 7) + k;

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        int w_shift4 = 0;
        int w_shift5 = 0;
        int w_shift6 = 0;
        int w_shift7 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp[8] = p2[0];
            pp[9] = p2[1];
            pp[10] = p2[2];
            pp[11] = p2[3];
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            pp[16] = p4[0];
            pp[17] = p4[1];
            pp[18] = p4[2];
            pp[19] = p4[3];
            pp[20] = p5[0];
            pp[21] = p5[1];
            pp[22] = p5[2];
            pp[23] = p5[3];
            pp[24] = p6[0];
            pp[25] = p6[1];
            pp[26] = p6[2];
            pp[27] = p6[3];
            pp[28] = p7[0];
            pp[29] = p7[1];
            pp[30] = p7[2];
            pp[31] = p7[3];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            w_shift4 += pp[16];
            w_shift4 += pp[17];
            w_shift4 += pp[18];
            w_shift4 += pp[19];
            w_shift5 += pp[20];
            w_shift5 += pp[21];
            w_shift5 += pp[22];
            w_shift5 += pp[23];
            w_shift6 += pp[24];
            w_shift6 += pp[25];
            w_shift6 += pp[26];
            w_shift6 += pp[27];
            w_shift7 += pp[28];
            w_shift7 += pp[29];
            w_shift7 += pp[30];
            w_shift7 += pp[31];
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            ((int*)pp)[4] = w_shift4 * 127;
            ((int*)pp)[5] = w_shift5 * 127;
            ((int*)pp)[6] = w_shift6 * 127;
            ((int*)pp)[7] = w_shift7 * 127;
            pp += 32;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
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
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp[8] = p2[0];
            pp[9] = p2[1];
            pp[10] = p2[2];
            pp[11] = p2[3];
            pp[12] = p3[0];
            pp[13] = p3[1];
            pp[14] = p3[2];
            pp[15] = p3[3];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            pp += 16;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
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
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            pp += 8;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#endif // __SSE2__
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
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

        int kk = 0;
#if __AVX512VNNI__
        int w_shift = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            w_shift += pp[0];
            w_shift += pp[1];
            w_shift += pp[2];
            w_shift += pp[3];
            pp += 4;
            p0 += 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift * 127;
            pp += 4;
        }
#endif // __AVX512VNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_A_tile_int8_avx512vnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_A_tile_int8_avxvnni(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_A_tile_int8_avx2(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        int w_shift4 = 0;
        int w_shift5 = 0;
        int w_shift6 = 0;
        int w_shift7 = 0;
        int w_shift8 = 0;
        int w_shift9 = 0;
        int w_shifta = 0;
        int w_shiftb = 0;
        int w_shiftc = 0;
        int w_shiftd = 0;
        int w_shifte = 0;
        int w_shiftf = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            pp[8] = p0[2];
            pp[9] = p0[A_hstep + 2];
            pp[10] = p0[A_hstep * 2 + 2];
            pp[11] = p0[A_hstep * 3 + 2];
            pp[12] = p0[3];
            pp[13] = p0[A_hstep + 3];
            pp[14] = p0[A_hstep * 2 + 3];
            pp[15] = p0[A_hstep * 3 + 3];
            pp[16] = p0[4];
            pp[17] = p0[A_hstep + 4];
            pp[18] = p0[A_hstep * 2 + 4];
            pp[19] = p0[A_hstep * 3 + 4];
            pp[20] = p0[5];
            pp[21] = p0[A_hstep + 5];
            pp[22] = p0[A_hstep * 2 + 5];
            pp[23] = p0[A_hstep * 3 + 5];
            pp[24] = p0[6];
            pp[25] = p0[A_hstep + 6];
            pp[26] = p0[A_hstep * 2 + 6];
            pp[27] = p0[A_hstep * 3 + 6];
            pp[28] = p0[7];
            pp[29] = p0[A_hstep + 7];
            pp[30] = p0[A_hstep * 2 + 7];
            pp[31] = p0[A_hstep * 3 + 7];

            pp[32 + 0] = p0[8];
            pp[32 + 1] = p0[A_hstep + 8];
            pp[32 + 2] = p0[A_hstep * 2 + 8];
            pp[32 + 3] = p0[A_hstep * 3 + 8];
            pp[32 + 4] = p0[9];
            pp[32 + 5] = p0[A_hstep + 9];
            pp[32 + 6] = p0[A_hstep * 2 + 9];
            pp[32 + 7] = p0[A_hstep * 3 + 9];
            pp[32 + 8] = p0[10];
            pp[32 + 9] = p0[A_hstep + 10];
            pp[32 + 10] = p0[A_hstep * 2 + 10];
            pp[32 + 11] = p0[A_hstep * 3 + 10];
            pp[32 + 12] = p0[11];
            pp[32 + 13] = p0[A_hstep + 11];
            pp[32 + 14] = p0[A_hstep * 2 + 11];
            pp[32 + 15] = p0[A_hstep * 3 + 11];
            pp[32 + 16] = p0[12];
            pp[32 + 17] = p0[A_hstep + 12];
            pp[32 + 18] = p0[A_hstep * 2 + 12];
            pp[32 + 19] = p0[A_hstep * 3 + 12];
            pp[32 + 20] = p0[13];
            pp[32 + 21] = p0[A_hstep + 13];
            pp[32 + 22] = p0[A_hstep * 2 + 13];
            pp[32 + 23] = p0[A_hstep * 3 + 13];
            pp[32 + 24] = p0[14];
            pp[32 + 25] = p0[A_hstep + 14];
            pp[32 + 26] = p0[A_hstep * 2 + 14];
            pp[32 + 27] = p0[A_hstep * 3 + 14];
            pp[32 + 28] = p0[15];
            pp[32 + 29] = p0[A_hstep + 15];
            pp[32 + 30] = p0[A_hstep * 2 + 15];
            pp[32 + 31] = p0[A_hstep * 3 + 15];

            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            w_shift4 += pp[16];
            w_shift4 += pp[17];
            w_shift4 += pp[18];
            w_shift4 += pp[19];
            w_shift5 += pp[20];
            w_shift5 += pp[21];
            w_shift5 += pp[22];
            w_shift5 += pp[23];
            w_shift6 += pp[24];
            w_shift6 += pp[25];
            w_shift6 += pp[26];
            w_shift6 += pp[27];
            w_shift7 += pp[28];
            w_shift7 += pp[29];
            w_shift7 += pp[30];
            w_shift7 += pp[31];

            w_shift8 += pp[32 + 0];
            w_shift8 += pp[32 + 1];
            w_shift8 += pp[32 + 2];
            w_shift8 += pp[32 + 3];
            w_shift9 += pp[32 + 4];
            w_shift9 += pp[32 + 5];
            w_shift9 += pp[32 + 6];
            w_shift9 += pp[32 + 7];
            w_shifta += pp[32 + 8];
            w_shifta += pp[32 + 9];
            w_shifta += pp[32 + 10];
            w_shifta += pp[32 + 11];
            w_shiftb += pp[32 + 12];
            w_shiftb += pp[32 + 13];
            w_shiftb += pp[32 + 14];
            w_shiftb += pp[32 + 15];
            w_shiftc += pp[32 + 16];
            w_shiftc += pp[32 + 17];
            w_shiftc += pp[32 + 18];
            w_shiftc += pp[32 + 19];
            w_shiftd += pp[32 + 20];
            w_shiftd += pp[32 + 21];
            w_shiftd += pp[32 + 22];
            w_shiftd += pp[32 + 23];
            w_shifte += pp[32 + 24];
            w_shifte += pp[32 + 25];
            w_shifte += pp[32 + 26];
            w_shifte += pp[32 + 27];
            w_shiftf += pp[32 + 28];
            w_shiftf += pp[32 + 29];
            w_shiftf += pp[32 + 30];
            w_shiftf += pp[32 + 31];

            pp += 64;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            ((int*)pp)[4] = w_shift4 * 127;
            ((int*)pp)[5] = w_shift5 * 127;
            ((int*)pp)[6] = w_shift6 * 127;
            ((int*)pp)[7] = w_shift7 * 127;
            ((int*)pp)[8] = w_shift8 * 127;
            ((int*)pp)[9] = w_shift9 * 127;
            ((int*)pp)[10] = w_shifta * 127;
            ((int*)pp)[11] = w_shiftb * 127;
            ((int*)pp)[12] = w_shiftc * 127;
            ((int*)pp)[13] = w_shiftd * 127;
            ((int*)pp)[14] = w_shifte * 127;
            ((int*)pp)[15] = w_shiftf * 127;
            pp += 64;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[A_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[A_hstep + 3];
            pp[8] = p0[4];
            pp[9] = p0[A_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[A_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[A_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[A_hstep + 7];

            pp[16 + 0] = p0[8];
            pp[16 + 1] = p0[A_hstep + 8];
            pp[16 + 2] = p0[9];
            pp[16 + 3] = p0[A_hstep + 9];
            pp[16 + 4] = p0[10];
            pp[16 + 5] = p0[A_hstep + 10];
            pp[16 + 6] = p0[11];
            pp[16 + 7] = p0[A_hstep + 11];
            pp[16 + 8] = p0[12];
            pp[16 + 9] = p0[A_hstep + 12];
            pp[16 + 10] = p0[13];
            pp[16 + 11] = p0[A_hstep + 13];
            pp[16 + 12] = p0[14];
            pp[16 + 13] = p0[A_hstep + 14];
            pp[16 + 14] = p0[15];
            pp[16 + 15] = p0[A_hstep + 15];
            pp += 32;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp[8] = p0[8];
            pp[9] = p0[9];
            pp[10] = p0[10];
            pp[11] = p0[11];
            pp[12] = p0[12];
            pp[13] = p0[13];
            pp[14] = p0[14];
            pp[15] = p0[15];
            pp += 16;
            p0 += A_hstep;
        }
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        int w_shift4 = 0;
        int w_shift5 = 0;
        int w_shift6 = 0;
        int w_shift7 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            pp[8] = p0[2];
            pp[9] = p0[A_hstep + 2];
            pp[10] = p0[A_hstep * 2 + 2];
            pp[11] = p0[A_hstep * 3 + 2];
            pp[12] = p0[3];
            pp[13] = p0[A_hstep + 3];
            pp[14] = p0[A_hstep * 2 + 3];
            pp[15] = p0[A_hstep * 3 + 3];
            pp[16] = p0[4];
            pp[17] = p0[A_hstep + 4];
            pp[18] = p0[A_hstep * 2 + 4];
            pp[19] = p0[A_hstep * 3 + 4];
            pp[20] = p0[5];
            pp[21] = p0[A_hstep + 5];
            pp[22] = p0[A_hstep * 2 + 5];
            pp[23] = p0[A_hstep * 3 + 5];
            pp[24] = p0[6];
            pp[25] = p0[A_hstep + 6];
            pp[26] = p0[A_hstep * 2 + 6];
            pp[27] = p0[A_hstep * 3 + 6];
            pp[28] = p0[7];
            pp[29] = p0[A_hstep + 7];
            pp[30] = p0[A_hstep * 2 + 7];
            pp[31] = p0[A_hstep * 3 + 7];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            w_shift4 += pp[16];
            w_shift4 += pp[17];
            w_shift4 += pp[18];
            w_shift4 += pp[19];
            w_shift5 += pp[20];
            w_shift5 += pp[21];
            w_shift5 += pp[22];
            w_shift5 += pp[23];
            w_shift6 += pp[24];
            w_shift6 += pp[25];
            w_shift6 += pp[26];
            w_shift6 += pp[27];
            w_shift7 += pp[28];
            w_shift7 += pp[29];
            w_shift7 += pp[30];
            w_shift7 += pp[31];
            pp += 32;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            ((int*)pp)[4] = w_shift4 * 127;
            ((int*)pp)[5] = w_shift5 * 127;
            ((int*)pp)[6] = w_shift6 * 127;
            ((int*)pp)[7] = w_shift7 * 127;
            pp += 32;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[A_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[A_hstep + 3];
            pp[8] = p0[4];
            pp[9] = p0[A_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[A_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[A_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[A_hstep + 7];
            pp += 16;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += A_hstep;
        }
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        int w_shift2 = 0;
        int w_shift3 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            pp[8] = p0[2];
            pp[9] = p0[A_hstep + 2];
            pp[10] = p0[A_hstep * 2 + 2];
            pp[11] = p0[A_hstep * 3 + 2];
            pp[12] = p0[3];
            pp[13] = p0[A_hstep + 3];
            pp[14] = p0[A_hstep * 2 + 3];
            pp[15] = p0[A_hstep * 3 + 3];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            w_shift2 += pp[8];
            w_shift2 += pp[9];
            w_shift2 += pp[10];
            w_shift2 += pp[11];
            w_shift3 += pp[12];
            w_shift3 += pp[13];
            w_shift3 += pp[14];
            w_shift3 += pp[15];
            pp += 16;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            ((int*)pp)[2] = w_shift2 * 127;
            ((int*)pp)[3] = w_shift3 * 127;
            pp += 16;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[A_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[A_hstep + 3];
            pp += 8;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
        int w_shift0 = 0;
        int w_shift1 = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[A_hstep + 1];
            pp[6] = p0[A_hstep * 2 + 1];
            pp[7] = p0[A_hstep * 3 + 1];
            w_shift0 += pp[0];
            w_shift0 += pp[1];
            w_shift0 += pp[2];
            w_shift0 += pp[3];
            w_shift1 += pp[4];
            w_shift1 += pp[5];
            w_shift1 += pp[6];
            w_shift1 += pp[7];
            pp += 8;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift0 * 127;
            ((int*)pp)[1] = w_shift1 * 127;
            pp += 8;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp += 4;
            p0 += A_hstep * 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __AVX512VNNI__
        int w_shift = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            w_shift += pp[0];
            w_shift += pp[1];
            w_shift += pp[2];
            w_shift += pp[3];
            pp += 4;
            p0 += A_hstep * 4;
        }
        if (max_kk >= 4)
        {
            ((int*)pp)[0] = w_shift * 127;
            pp += 4;
        }
#endif // __AVX512VNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_int8_avx512vnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_int8_avxvnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
        const signed char* p4 = B.row<const signed char>(j + jj + 4) + k;
        const signed char* p5 = B.row<const signed char>(j + jj + 5) + k;
        const signed char* p6 = B.row<const signed char>(j + jj + 6) + k;
        const signed char* p7 = B.row<const signed char>(j + jj + 7) + k;
        const signed char* p8 = B.row<const signed char>(j + jj + 8) + k;
        const signed char* p9 = B.row<const signed char>(j + jj + 9) + k;
        const signed char* pa = B.row<const signed char>(j + jj + 10) + k;
        const signed char* pb = B.row<const signed char>(j + jj + 11) + k;
        const signed char* pc = B.row<const signed char>(j + jj + 12) + k;
        const signed char* pd = B.row<const signed char>(j + jj + 13) + k;
        const signed char* pe = B.row<const signed char>(j + jj + 14) + k;
        const signed char* pf = B.row<const signed char>(j + jj + 15) + k;

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp[4] = p1[0] + 127;
            pp[5] = p1[1] + 127;
            pp[6] = p1[2] + 127;
            pp[7] = p1[3] + 127;
            pp[8] = p2[0] + 127;
            pp[9] = p2[1] + 127;
            pp[10] = p2[2] + 127;
            pp[11] = p2[3] + 127;
            pp[12] = p3[0] + 127;
            pp[13] = p3[1] + 127;
            pp[14] = p3[2] + 127;
            pp[15] = p3[3] + 127;

            pp[16 + 0] = p4[0] + 127;
            pp[16 + 1] = p4[1] + 127;
            pp[16 + 2] = p4[2] + 127;
            pp[16 + 3] = p4[3] + 127;
            pp[16 + 4] = p5[0] + 127;
            pp[16 + 5] = p5[1] + 127;
            pp[16 + 6] = p5[2] + 127;
            pp[16 + 7] = p5[3] + 127;
            pp[16 + 8] = p6[0] + 127;
            pp[16 + 9] = p6[1] + 127;
            pp[16 + 10] = p6[2] + 127;
            pp[16 + 11] = p6[3] + 127;
            pp[16 + 12] = p7[0] + 127;
            pp[16 + 13] = p7[1] + 127;
            pp[16 + 14] = p7[2] + 127;
            pp[16 + 15] = p7[3] + 127;

            pp[32 + 0] = p8[0] + 127;
            pp[32 + 1] = p8[1] + 127;
            pp[32 + 2] = p8[2] + 127;
            pp[32 + 3] = p8[3] + 127;
            pp[32 + 4] = p9[0] + 127;
            pp[32 + 5] = p9[1] + 127;
            pp[32 + 6] = p9[2] + 127;
            pp[32 + 7] = p9[3] + 127;
            pp[32 + 8] = pa[0] + 127;
            pp[32 + 9] = pa[1] + 127;
            pp[32 + 10] = pa[2] + 127;
            pp[32 + 11] = pa[3] + 127;
            pp[32 + 12] = pb[0] + 127;
            pp[32 + 13] = pb[1] + 127;
            pp[32 + 14] = pb[2] + 127;
            pp[32 + 15] = pb[3] + 127;

            pp[48 + 0] = pc[0] + 127;
            pp[48 + 1] = pc[1] + 127;
            pp[48 + 2] = pc[2] + 127;
            pp[48 + 3] = pc[3] + 127;
            pp[48 + 4] = pd[0] + 127;
            pp[48 + 5] = pd[1] + 127;
            pp[48 + 6] = pd[2] + 127;
            pp[48 + 7] = pd[3] + 127;
            pp[48 + 8] = pe[0] + 127;
            pp[48 + 9] = pe[1] + 127;
            pp[48 + 10] = pe[2] + 127;
            pp[48 + 11] = pe[3] + 127;
            pp[48 + 12] = pf[0] + 127;
            pp[48 + 13] = pf[1] + 127;
            pp[48 + 14] = pf[2] + 127;
            pp[48 + 15] = pf[3] + 127;

            pp += 64;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
            p8 += 4;
            p9 += 4;
            pa += 4;
            pb += 4;
            pc += 4;
            pd += 4;
            pe += 4;
            pf += 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];

            pp[16 + 0] = p8[0];
            pp[16 + 1] = p8[1];
            pp[16 + 2] = p9[0];
            pp[16 + 3] = p9[1];
            pp[16 + 4] = pa[0];
            pp[16 + 5] = pa[1];
            pp[16 + 6] = pb[0];
            pp[16 + 7] = pb[1];
            pp[16 + 8] = pc[0];
            pp[16 + 9] = pc[1];
            pp[16 + 10] = pd[0];
            pp[16 + 11] = pd[1];
            pp[16 + 12] = pe[0];
            pp[16 + 13] = pe[1];
            pp[16 + 14] = pf[0];
            pp[16 + 15] = pf[1];

            pp += 32;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
            p8 += 2;
            p9 += 2;
            pa += 2;
            pb += 2;
            pc += 2;
            pd += 2;
            pe += 2;
            pf += 2;
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
            pp[8] = p8[0];
            pp[9] = p9[0];
            pp[10] = pa[0];
            pp[11] = pb[0];
            pp[12] = pc[0];
            pp[13] = pd[0];
            pp[14] = pe[0];
            pp[15] = pf[0];
            pp += 16;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
            p8++;
            p9++;
            pa++;
            pb++;
            pc++;
            pd++;
            pe++;
            pf++;
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;
        const signed char* p4 = B.row<const signed char>(j + jj + 4) + k;
        const signed char* p5 = B.row<const signed char>(j + jj + 5) + k;
        const signed char* p6 = B.row<const signed char>(j + jj + 6) + k;
        const signed char* p7 = B.row<const signed char>(j + jj + 7) + k;

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp[4] = p1[0] + 127;
            pp[5] = p1[1] + 127;
            pp[6] = p1[2] + 127;
            pp[7] = p1[3] + 127;
            pp[8] = p2[0] + 127;
            pp[9] = p2[1] + 127;
            pp[10] = p2[2] + 127;
            pp[11] = p2[3] + 127;
            pp[12] = p3[0] + 127;
            pp[13] = p3[1] + 127;
            pp[14] = p3[2] + 127;
            pp[15] = p3[3] + 127;
            pp[16 + 0] = p4[0] + 127;
            pp[16 + 1] = p4[1] + 127;
            pp[16 + 2] = p4[2] + 127;
            pp[16 + 3] = p4[3] + 127;
            pp[16 + 4] = p5[0] + 127;
            pp[16 + 5] = p5[1] + 127;
            pp[16 + 6] = p5[2] + 127;
            pp[16 + 7] = p5[3] + 127;
            pp[16 + 8] = p6[0] + 127;
            pp[16 + 9] = p6[1] + 127;
            pp[16 + 10] = p6[2] + 127;
            pp[16 + 11] = p6[3] + 127;
            pp[16 + 12] = p7[0] + 127;
            pp[16 + 13] = p7[1] + 127;
            pp[16 + 14] = p7[2] + 127;
            pp[16 + 15] = p7[3] + 127;
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p4[0];
            pp[9] = p4[1];
            pp[10] = p5[0];
            pp[11] = p5[1];
            pp[12] = p6[0];
            pp[13] = p6[1];
            pp[14] = p7[0];
            pp[15] = p7[1];
            pp += 16;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
            p4 += 2;
            p5 += 2;
            p6 += 2;
            p7 += 2;
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
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp[4] = p1[0] + 127;
            pp[5] = p1[1] + 127;
            pp[6] = p1[2] + 127;
            pp[7] = p1[3] + 127;
            pp[8] = p2[0] + 127;
            pp[9] = p2[1] + 127;
            pp[10] = p2[2] + 127;
            pp[11] = p2[3] + 127;
            pp[12] = p3[0] + 127;
            pp[13] = p3[1] + 127;
            pp[14] = p3[2] + 127;
            pp[15] = p3[3] + 127;
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp += 8;
            p0 += 2;
            p1 += 2;
            p2 += 2;
            p3 += 2;
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
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp[4] = p1[0] + 127;
            pp[5] = p1[1] + 127;
            pp[6] = p1[2] + 127;
            pp[7] = p1[3] + 127;
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp += 4;
            p0 += 2;
            p1 += 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[1] + 127;
            pp[2] = p0[2] + 127;
            pp[3] = p0[3] + 127;
            pp += 4;
            p0 += 4;
        }
#endif // __AVX512VNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_B_tile_int8_avx512vnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_B_tile_int8_avxvnni(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    const int B_hstep = B.w;

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp[4] = p0[1] + 127;
            pp[5] = p0[B_hstep + 1] + 127;
            pp[6] = p0[B_hstep * 2 + 1] + 127;
            pp[7] = p0[B_hstep * 3 + 1] + 127;
            pp[8] = p0[2] + 127;
            pp[9] = p0[B_hstep + 2] + 127;
            pp[10] = p0[B_hstep * 2 + 2] + 127;
            pp[11] = p0[B_hstep * 3 + 2] + 127;
            pp[12] = p0[3] + 127;
            pp[13] = p0[B_hstep + 3] + 127;
            pp[14] = p0[B_hstep * 2 + 3] + 127;
            pp[15] = p0[B_hstep * 3 + 3] + 127;
            pp[16] = p0[4] + 127;
            pp[17] = p0[B_hstep + 4] + 127;
            pp[18] = p0[B_hstep * 2 + 4] + 127;
            pp[19] = p0[B_hstep * 3 + 4] + 127;
            pp[20] = p0[5] + 127;
            pp[21] = p0[B_hstep + 5] + 127;
            pp[22] = p0[B_hstep * 2 + 5] + 127;
            pp[23] = p0[B_hstep * 3 + 5] + 127;
            pp[24] = p0[6] + 127;
            pp[25] = p0[B_hstep + 6] + 127;
            pp[26] = p0[B_hstep * 2 + 6] + 127;
            pp[27] = p0[B_hstep * 3 + 6] + 127;
            pp[28] = p0[7] + 127;
            pp[29] = p0[B_hstep + 7] + 127;
            pp[30] = p0[B_hstep * 2 + 7] + 127;
            pp[31] = p0[B_hstep * 3 + 7] + 127;

            pp[32 + 0] = p0[8] + 127;
            pp[32 + 1] = p0[B_hstep + 8] + 127;
            pp[32 + 2] = p0[B_hstep * 2 + 8] + 127;
            pp[32 + 3] = p0[B_hstep * 3 + 8] + 127;
            pp[32 + 4] = p0[9] + 127;
            pp[32 + 5] = p0[B_hstep + 9] + 127;
            pp[32 + 6] = p0[B_hstep * 2 + 9] + 127;
            pp[32 + 7] = p0[B_hstep * 3 + 9] + 127;
            pp[32 + 8] = p0[10] + 127;
            pp[32 + 9] = p0[B_hstep + 10] + 127;
            pp[32 + 10] = p0[B_hstep * 2 + 10] + 127;
            pp[32 + 11] = p0[B_hstep * 3 + 10] + 127;
            pp[32 + 12] = p0[11] + 127;
            pp[32 + 13] = p0[B_hstep + 11] + 127;
            pp[32 + 14] = p0[B_hstep * 2 + 11] + 127;
            pp[32 + 15] = p0[B_hstep * 3 + 11] + 127;
            pp[32 + 16] = p0[12] + 127;
            pp[32 + 17] = p0[B_hstep + 12] + 127;
            pp[32 + 18] = p0[B_hstep * 2 + 12] + 127;
            pp[32 + 19] = p0[B_hstep * 3 + 12] + 127;
            pp[32 + 20] = p0[13] + 127;
            pp[32 + 21] = p0[B_hstep + 13] + 127;
            pp[32 + 22] = p0[B_hstep * 2 + 13] + 127;
            pp[32 + 23] = p0[B_hstep * 3 + 13] + 127;
            pp[32 + 24] = p0[14] + 127;
            pp[32 + 25] = p0[B_hstep + 14] + 127;
            pp[32 + 26] = p0[B_hstep * 2 + 14] + 127;
            pp[32 + 27] = p0[B_hstep * 3 + 14] + 127;
            pp[32 + 28] = p0[15] + 127;
            pp[32 + 29] = p0[B_hstep + 15] + 127;
            pp[32 + 30] = p0[B_hstep * 2 + 15] + 127;
            pp[32 + 31] = p0[B_hstep * 3 + 15] + 127;
            pp += 64;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[B_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[B_hstep + 3];
            pp[8] = p0[4];
            pp[9] = p0[B_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[B_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[B_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[B_hstep + 7];

            pp[16 + 0] = p0[8];
            pp[16 + 1] = p0[B_hstep + 8];
            pp[16 + 2] = p0[9];
            pp[16 + 3] = p0[B_hstep + 9];
            pp[16 + 4] = p0[10];
            pp[16 + 5] = p0[B_hstep + 10];
            pp[16 + 6] = p0[11];
            pp[16 + 7] = p0[B_hstep + 11];
            pp[16 + 8] = p0[12];
            pp[16 + 9] = p0[B_hstep + 12];
            pp[16 + 10] = p0[13];
            pp[16 + 11] = p0[B_hstep + 13];
            pp[16 + 12] = p0[14];
            pp[16 + 13] = p0[B_hstep + 14];
            pp[16 + 14] = p0[15];
            pp[16 + 15] = p0[B_hstep + 15];
            pp += 32;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp[8] = p0[8];
            pp[9] = p0[9];
            pp[10] = p0[10];
            pp[11] = p0[11];
            pp[12] = p0[12];
            pp[13] = p0[13];
            pp[14] = p0[14];
            pp[15] = p0[15];
            pp += 16;
            p0 += B_hstep;
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp[4] = p0[1] + 127;
            pp[5] = p0[B_hstep + 1] + 127;
            pp[6] = p0[B_hstep * 2 + 1] + 127;
            pp[7] = p0[B_hstep * 3 + 1] + 127;
            pp[8] = p0[2] + 127;
            pp[9] = p0[B_hstep + 2] + 127;
            pp[10] = p0[B_hstep * 2 + 2] + 127;
            pp[11] = p0[B_hstep * 3 + 2] + 127;
            pp[12] = p0[3] + 127;
            pp[13] = p0[B_hstep + 3] + 127;
            pp[14] = p0[B_hstep * 2 + 3] + 127;
            pp[15] = p0[B_hstep * 3 + 3] + 127;
            pp[16] = p0[4] + 127;
            pp[17] = p0[B_hstep + 4] + 127;
            pp[18] = p0[B_hstep * 2 + 4] + 127;
            pp[19] = p0[B_hstep * 3 + 4] + 127;
            pp[20] = p0[5] + 127;
            pp[21] = p0[B_hstep + 5] + 127;
            pp[22] = p0[B_hstep * 2 + 5] + 127;
            pp[23] = p0[B_hstep * 3 + 5] + 127;
            pp[24] = p0[6] + 127;
            pp[25] = p0[B_hstep + 6] + 127;
            pp[26] = p0[B_hstep * 2 + 6] + 127;
            pp[27] = p0[B_hstep * 3 + 6] + 127;
            pp[28] = p0[7] + 127;
            pp[29] = p0[B_hstep + 7] + 127;
            pp[30] = p0[B_hstep * 2 + 7] + 127;
            pp[31] = p0[B_hstep * 3 + 7] + 127;
            pp += 32;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[B_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[B_hstep + 3];
            pp[8] = p0[4];
            pp[9] = p0[B_hstep + 4];
            pp[10] = p0[5];
            pp[11] = p0[B_hstep + 5];
            pp[12] = p0[6];
            pp[13] = p0[B_hstep + 6];
            pp[14] = p0[7];
            pp[15] = p0[B_hstep + 7];
            pp += 16;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += B_hstep;
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp[4] = p0[1] + 127;
            pp[5] = p0[B_hstep + 1] + 127;
            pp[6] = p0[B_hstep * 2 + 1] + 127;
            pp[7] = p0[B_hstep * 3 + 1] + 127;
            pp[8] = p0[2] + 127;
            pp[9] = p0[B_hstep + 2] + 127;
            pp[10] = p0[B_hstep * 2 + 2] + 127;
            pp[11] = p0[B_hstep * 3 + 2] + 127;
            pp[12] = p0[3] + 127;
            pp[13] = p0[B_hstep + 3] + 127;
            pp[14] = p0[B_hstep * 2 + 3] + 127;
            pp[15] = p0[B_hstep * 3 + 3] + 127;
            pp += 16;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp[4] = p0[2];
            pp[5] = p0[B_hstep + 2];
            pp[6] = p0[3];
            pp[7] = p0[B_hstep + 3];
            pp += 8;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp[4] = p0[1] + 127;
            pp[5] = p0[B_hstep + 1] + 127;
            pp[6] = p0[B_hstep * 2 + 1] + 127;
            pp[7] = p0[B_hstep * 3 + 1] + 127;
            pp += 8;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp += 4;
            p0 += B_hstep * 2;
        }
#endif // __SSE2__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __AVX512VNNI__
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0] + 127;
            pp[1] = p0[B_hstep] + 127;
            pp[2] = p0[B_hstep * 2] + 127;
            pp[3] = p0[B_hstep * 3] + 127;
            pp += 4;
            p0 += B_hstep * 4;
        }
#endif // __AVX512VNNI__
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void compute_A_tile_fp32_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    const int K = A.w;

    // NCNN_LOGE("compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        // __m512 _v127 = _mm512_set1_ps(127.f);
        // __m512 _v127_B_scale = _mm512_set1_ps(v127_B_scale);
        for (int ii = 0; ii + 15 < max_ii; ii += 16)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            __m512 _absmax0 = _mm512_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m512 _p = _mm512_loadu_ps(p0);
                _absmax0 = _mm512_max_ps(_absmax0, abs512_ps(_p));
                p0 += 16;
            }

            // __m512 _scale = _mm512_div_ps(_v127, _absmax0);
            // __m512 _out_descale = _mm512_div_ps(_absmax0, _v127_B_scale);

            // _mm512_store_ps(ps, _scale);
            // _mm512_store_ps(pods, _out_descale);

            float absmax[16];
            _mm512_storeu_ps(absmax, _absmax0);

            ps[0] = 127.f / absmax[0];
            ps[1] = 127.f / absmax[1];
            ps[2] = 127.f / absmax[2];
            ps[3] = 127.f / absmax[3];
            ps[4] = 127.f / absmax[4];
            ps[5] = 127.f / absmax[5];
            ps[6] = 127.f / absmax[6];
            ps[7] = 127.f / absmax[7];
            ps[8] = 127.f / absmax[8];
            ps[9] = 127.f / absmax[9];
            ps[10] = 127.f / absmax[10];
            ps[11] = 127.f / absmax[11];
            ps[12] = 127.f / absmax[12];
            ps[13] = 127.f / absmax[13];
            ps[14] = 127.f / absmax[14];
            ps[15] = 127.f / absmax[15];
            pods[0] = absmax[0] / v127_B_scale;
            pods[1] = absmax[1] / v127_B_scale;
            pods[2] = absmax[2] / v127_B_scale;
            pods[3] = absmax[3] / v127_B_scale;
            pods[4] = absmax[4] / v127_B_scale;
            pods[5] = absmax[5] / v127_B_scale;
            pods[6] = absmax[6] / v127_B_scale;
            pods[7] = absmax[7] / v127_B_scale;
            pods[8] = absmax[8] / v127_B_scale;
            pods[9] = absmax[9] / v127_B_scale;
            pods[10] = absmax[10] / v127_B_scale;
            pods[11] = absmax[11] / v127_B_scale;
            pods[12] = absmax[12] / v127_B_scale;
            pods[13] = absmax[13] / v127_B_scale;
            pods[14] = absmax[14] / v127_B_scale;
            pods[15] = absmax[15] / v127_B_scale;

            ps += 16;
            pods += 16;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        // __m256 _v127 = _mm256_set1_ps(127.f);
        // __m256 _v127_B_scale = _mm256_set1_ps(v127_B_scale);
        for (int ii = 0; ii + 7 < max_ii; ii += 8)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            __m256 _absmax0 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0);
                _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p));
                p0 += 8;
            }

            // __m256 _scale = _mm256_div_ps(_v127, _absmax0);
            // __m256 _out_descale = _mm256_div_ps(_absmax0, _v127_B_scale);

            // _mm256_store_ps(ps, _scale);
            // _mm256_store_ps(pods, _out_descale);

            float absmax[8];
            _mm256_storeu_ps(absmax, _absmax0);

            ps[0] = 127.f / absmax[0];
            ps[1] = 127.f / absmax[1];
            ps[2] = 127.f / absmax[2];
            ps[3] = 127.f / absmax[3];
            ps[4] = 127.f / absmax[4];
            ps[5] = 127.f / absmax[5];
            ps[6] = 127.f / absmax[6];
            ps[7] = 127.f / absmax[7];
            pods[0] = absmax[0] / v127_B_scale;
            pods[1] = absmax[1] / v127_B_scale;
            pods[2] = absmax[2] / v127_B_scale;
            pods[3] = absmax[3] / v127_B_scale;
            pods[4] = absmax[4] / v127_B_scale;
            pods[5] = absmax[5] / v127_B_scale;
            pods[6] = absmax[6] / v127_B_scale;
            pods[7] = absmax[7] / v127_B_scale;

            ps += 8;
            pods += 8;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        // __m128 _v127 = _mm_set1_ps(127.f);
        // __m128 _v127_B_scale = _mm_set1_ps(v127_B_scale);
        for (int ii = 0; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            __m128 _absmax0 = _mm_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p));
                p0 += 4;
            }

            // __m128 _scale = _mm_div_ps(_v127, _absmax0);
            // __m128 _out_descale = _mm_div_ps(_absmax0, _v127_B_scale);

            // _mm_store_ps(ps, _scale);
            // _mm_store_ps(pods, _out_descale);

            float absmax[4];
            _mm_storeu_ps(absmax, _absmax0);

            ps[0] = 127.f / absmax[0];
            ps[1] = 127.f / absmax[1];
            ps[2] = 127.f / absmax[2];
            ps[3] = 127.f / absmax[3];
            pods[0] = absmax[0] / v127_B_scale;
            pods[1] = absmax[1] / v127_B_scale;
            pods[2] = absmax[2] / v127_B_scale;
            pods[3] = absmax[3] / v127_B_scale;

            ps += 4;
            pods += 4;
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        for (int ii = 0; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            float absmax = 0.f;
            int kk = 0;
            for (; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(p0[0]));
                p0++;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_A_tile_fp32_to_int8_avx512vnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_A_tile_fp32_to_int8_avxvnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d %d", max_ii, max_kk, elempack);

    signed char* pp = (signed char*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        const float scale8 = scales[i + ii + 8];
        const float scale9 = scales[i + ii + 9];
        const float scalea = scales[i + ii + 10];
        const float scaleb = scales[i + ii + 11];
        const float scalec = scales[i + ii + 12];
        const float scaled = scales[i + ii + 13];
        const float scalee = scales[i + ii + 14];
        const float scalef = scales[i + ii + 15];

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[16] * scale0);
                pp[2] = float2int8(p0[32] * scale0);
                pp[3] = float2int8(p0[48] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[17] * scale1);
                pp[6] = float2int8(p0[33] * scale1);
                pp[7] = float2int8(p0[49] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[18] * scale2);
                pp[10] = float2int8(p0[34] * scale2);
                pp[11] = float2int8(p0[50] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[19] * scale3);
                pp[14] = float2int8(p0[35] * scale3);
                pp[15] = float2int8(p0[51] * scale3);
                pp[16] = float2int8(p0[4] * scale4);
                pp[17] = float2int8(p0[20] * scale4);
                pp[18] = float2int8(p0[36] * scale4);
                pp[19] = float2int8(p0[52] * scale4);
                pp[20] = float2int8(p0[5] * scale5);
                pp[21] = float2int8(p0[21] * scale5);
                pp[22] = float2int8(p0[37] * scale5);
                pp[23] = float2int8(p0[53] * scale5);
                pp[24] = float2int8(p0[6] * scale6);
                pp[25] = float2int8(p0[22] * scale6);
                pp[26] = float2int8(p0[38] * scale6);
                pp[27] = float2int8(p0[54] * scale6);
                pp[28] = float2int8(p0[7] * scale7);
                pp[29] = float2int8(p0[23] * scale7);
                pp[30] = float2int8(p0[39] * scale7);
                pp[31] = float2int8(p0[55] * scale7);
                pp[32] = float2int8(p0[8] * scale8);
                pp[33] = float2int8(p0[24] * scale8);
                pp[34] = float2int8(p0[40] * scale8);
                pp[35] = float2int8(p0[56] * scale8);
                pp[36] = float2int8(p0[9] * scale9);
                pp[37] = float2int8(p0[25] * scale9);
                pp[38] = float2int8(p0[41] * scale9);
                pp[39] = float2int8(p0[57] * scale9);
                pp[40] = float2int8(p0[10] * scalea);
                pp[41] = float2int8(p0[26] * scalea);
                pp[42] = float2int8(p0[42] * scalea);
                pp[43] = float2int8(p0[58] * scalea);
                pp[44] = float2int8(p0[11] * scaleb);
                pp[45] = float2int8(p0[27] * scaleb);
                pp[46] = float2int8(p0[43] * scaleb);
                pp[47] = float2int8(p0[59] * scaleb);
                pp[48] = float2int8(p0[12] * scalec);
                pp[49] = float2int8(p0[28] * scalec);
                pp[50] = float2int8(p0[44] * scalec);
                pp[51] = float2int8(p0[60] * scalec);
                pp[52] = float2int8(p0[13] * scaled);
                pp[53] = float2int8(p0[29] * scaled);
                pp[54] = float2int8(p0[45] * scaled);
                pp[55] = float2int8(p0[61] * scaled);
                pp[56] = float2int8(p0[14] * scalee);
                pp[57] = float2int8(p0[30] * scalee);
                pp[58] = float2int8(p0[46] * scalee);
                pp[59] = float2int8(p0[62] * scalee);
                pp[60] = float2int8(p0[15] * scalef);
                pp[61] = float2int8(p0[31] * scalef);
                pp[62] = float2int8(p0[47] * scalef);
                pp[63] = float2int8(p0[63] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += 64;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[16] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[17] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[18] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[19] * scale3);
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[20] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[21] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[22] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[23] * scale7);
                pp[16 + 0] = float2int8(p0[8] * scale8);
                pp[16 + 1] = float2int8(p0[24] * scale8);
                pp[16 + 2] = float2int8(p0[9] * scale9);
                pp[16 + 3] = float2int8(p0[25] * scale9);
                pp[16 + 4] = float2int8(p0[10] * scalea);
                pp[16 + 5] = float2int8(p0[26] * scalea);
                pp[16 + 6] = float2int8(p0[11] * scaleb);
                pp[16 + 7] = float2int8(p0[27] * scaleb);
                pp[16 + 8] = float2int8(p0[12] * scalec);
                pp[16 + 9] = float2int8(p0[28] * scalec);
                pp[16 + 10] = float2int8(p0[13] * scaled);
                pp[16 + 11] = float2int8(p0[29] * scaled);
                pp[16 + 12] = float2int8(p0[14] * scalee);
                pp[16 + 13] = float2int8(p0[30] * scalee);
                pp[16 + 14] = float2int8(p0[15] * scalef);
                pp[16 + 15] = float2int8(p0[31] * scalef);
                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp[8] = float2int8(p0[8] * scale8);
                pp[9] = float2int8(p0[9] * scale9);
                pp[10] = float2int8(p0[10] * scalea);
                pp[11] = float2int8(p0[11] * scaleb);
                pp[12] = float2int8(p0[12] * scalec);
                pp[13] = float2int8(p0[13] * scaled);
                pp[14] = float2int8(p0[14] * scalee);
                pp[15] = float2int8(p0[15] * scalef);
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[8] * scale0);
                pp[2] = float2int8(p0[16] * scale0);
                pp[3] = float2int8(p0[24] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[17] * scale1);
                pp[7] = float2int8(p0[25] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[10] * scale2);
                pp[10] = float2int8(p0[18] * scale2);
                pp[11] = float2int8(p0[26] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[11] * scale3);
                pp[14] = float2int8(p0[19] * scale3);
                pp[15] = float2int8(p0[27] * scale3);
                pp[16] = float2int8(p0[4] * scale4);
                pp[17] = float2int8(p0[12] * scale4);
                pp[18] = float2int8(p0[20] * scale4);
                pp[19] = float2int8(p0[28] * scale4);
                pp[20] = float2int8(p0[5] * scale5);
                pp[21] = float2int8(p0[13] * scale5);
                pp[22] = float2int8(p0[21] * scale5);
                pp[23] = float2int8(p0[29] * scale5);
                pp[24] = float2int8(p0[6] * scale6);
                pp[25] = float2int8(p0[14] * scale6);
                pp[26] = float2int8(p0[22] * scale6);
                pp[27] = float2int8(p0[30] * scale6);
                pp[28] = float2int8(p0[7] * scale7);
                pp[29] = float2int8(p0[15] * scale7);
                pp[30] = float2int8(p0[23] * scale7);
                pp[31] = float2int8(p0[31] * scale7);

                pp[32 + 0] = float2int8(p0[A_hstep * 8 + 0] * scale8);
                pp[32 + 1] = float2int8(p0[A_hstep * 8 + 8] * scale8);
                pp[32 + 2] = float2int8(p0[A_hstep * 8 + 16] * scale8);
                pp[32 + 3] = float2int8(p0[A_hstep * 8 + 24] * scale8);
                pp[32 + 4] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[32 + 5] = float2int8(p0[A_hstep * 8 + 9] * scale9);
                pp[32 + 6] = float2int8(p0[A_hstep * 8 + 17] * scale9);
                pp[32 + 7] = float2int8(p0[A_hstep * 8 + 25] * scale9);
                pp[32 + 8] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[32 + 9] = float2int8(p0[A_hstep * 8 + 10] * scalea);
                pp[32 + 10] = float2int8(p0[A_hstep * 8 + 18] * scalea);
                pp[32 + 11] = float2int8(p0[A_hstep * 8 + 26] * scalea);
                pp[32 + 12] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[32 + 13] = float2int8(p0[A_hstep * 8 + 11] * scaleb);
                pp[32 + 14] = float2int8(p0[A_hstep * 8 + 19] * scaleb);
                pp[32 + 15] = float2int8(p0[A_hstep * 8 + 27] * scaleb);
                pp[32 + 16] = float2int8(p0[A_hstep * 8 + 4] * scalec);
                pp[32 + 17] = float2int8(p0[A_hstep * 8 + 12] * scalec);
                pp[32 + 18] = float2int8(p0[A_hstep * 8 + 20] * scalec);
                pp[32 + 19] = float2int8(p0[A_hstep * 8 + 28] * scalec);
                pp[32 + 20] = float2int8(p0[A_hstep * 8 + 5] * scaled);
                pp[32 + 21] = float2int8(p0[A_hstep * 8 + 13] * scaled);
                pp[32 + 22] = float2int8(p0[A_hstep * 8 + 21] * scaled);
                pp[32 + 23] = float2int8(p0[A_hstep * 8 + 29] * scaled);
                pp[32 + 24] = float2int8(p0[A_hstep * 8 + 6] * scalee);
                pp[32 + 25] = float2int8(p0[A_hstep * 8 + 14] * scalee);
                pp[32 + 26] = float2int8(p0[A_hstep * 8 + 22] * scalee);
                pp[32 + 27] = float2int8(p0[A_hstep * 8 + 30] * scalee);
                pp[32 + 28] = float2int8(p0[A_hstep * 8 + 7] * scalef);
                pp[32 + 29] = float2int8(p0[A_hstep * 8 + 15] * scalef);
                pp[32 + 30] = float2int8(p0[A_hstep * 8 + 23] * scalef);
                pp[32 + 31] = float2int8(p0[A_hstep * 8 + 31] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += 32;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[8] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[10] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[11] * scale3);
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[12] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[13] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[14] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[15] * scale7);

                pp[16 + 0] = float2int8(p0[A_hstep * 8 + 0] * scale8);
                pp[16 + 1] = float2int8(p0[A_hstep * 8 + 8] * scale8);
                pp[16 + 2] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[16 + 3] = float2int8(p0[A_hstep * 8 + 9] * scale9);
                pp[16 + 4] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[16 + 5] = float2int8(p0[A_hstep * 8 + 10] * scalea);
                pp[16 + 6] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[16 + 7] = float2int8(p0[A_hstep * 8 + 11] * scaleb);
                pp[16 + 8] = float2int8(p0[A_hstep * 8 + 4] * scalec);
                pp[16 + 9] = float2int8(p0[A_hstep * 8 + 12] * scalec);
                pp[16 + 10] = float2int8(p0[A_hstep * 8 + 5] * scaled);
                pp[16 + 11] = float2int8(p0[A_hstep * 8 + 13] * scaled);
                pp[16 + 12] = float2int8(p0[A_hstep * 8 + 6] * scalee);
                pp[16 + 13] = float2int8(p0[A_hstep * 8 + 14] * scalee);
                pp[16 + 14] = float2int8(p0[A_hstep * 8 + 7] * scalef);
                pp[16 + 15] = float2int8(p0[A_hstep * 8 + 15] * scalef);
                pp += 32;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp[8] = float2int8(p0[A_hstep * 8 + 0] * scale8);
                pp[9] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[10] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[11] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[12] = float2int8(p0[A_hstep * 8 + 4] * scalec);
                pp[13] = float2int8(p0[A_hstep * 8 + 5] * scaled);
                pp[14] = float2int8(p0[A_hstep * 8 + 6] * scalee);
                pp[15] = float2int8(p0[A_hstep * 8 + 7] * scalef);
                pp += 16;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[8] * scale0);
                pp[3] = float2int8(p0[12] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[9] * scale1);
                pp[7] = float2int8(p0[13] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[6] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[14] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[7] * scale3);
                pp[14] = float2int8(p0[11] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                pp[16 + 0] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp[16 + 1] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp[16 + 2] = float2int8(p0[A_hstep * 4 + 8] * scale4);
                pp[16 + 3] = float2int8(p0[A_hstep * 4 + 12] * scale4);
                pp[16 + 4] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[16 + 5] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp[16 + 6] = float2int8(p0[A_hstep * 4 + 9] * scale5);
                pp[16 + 7] = float2int8(p0[A_hstep * 4 + 13] * scale5);
                pp[16 + 8] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[16 + 9] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp[16 + 10] = float2int8(p0[A_hstep * 4 + 10] * scale6);
                pp[16 + 11] = float2int8(p0[A_hstep * 4 + 14] * scale6);
                pp[16 + 12] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[16 + 13] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp[16 + 14] = float2int8(p0[A_hstep * 4 + 11] * scale7);
                pp[16 + 15] = float2int8(p0[A_hstep * 4 + 15] * scale7);

                pp[32 + 0] = float2int8(p0[A_hstep * 8 + 0] * scale8);
                pp[32 + 1] = float2int8(p0[A_hstep * 8 + 4] * scale8);
                pp[32 + 2] = float2int8(p0[A_hstep * 8 + 8] * scale8);
                pp[32 + 3] = float2int8(p0[A_hstep * 8 + 12] * scale8);
                pp[32 + 4] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[32 + 5] = float2int8(p0[A_hstep * 8 + 5] * scale9);
                pp[32 + 6] = float2int8(p0[A_hstep * 8 + 9] * scale9);
                pp[32 + 7] = float2int8(p0[A_hstep * 8 + 13] * scale9);
                pp[32 + 8] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[32 + 9] = float2int8(p0[A_hstep * 8 + 6] * scalea);
                pp[32 + 10] = float2int8(p0[A_hstep * 8 + 10] * scalea);
                pp[32 + 11] = float2int8(p0[A_hstep * 8 + 14] * scalea);
                pp[32 + 12] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[32 + 13] = float2int8(p0[A_hstep * 8 + 7] * scaleb);
                pp[32 + 14] = float2int8(p0[A_hstep * 8 + 11] * scaleb);
                pp[32 + 15] = float2int8(p0[A_hstep * 8 + 15] * scaleb);

                pp[48 + 0] = float2int8(p0[A_hstep * 12 + 0] * scalec);
                pp[48 + 1] = float2int8(p0[A_hstep * 12 + 4] * scalec);
                pp[48 + 2] = float2int8(p0[A_hstep * 12 + 8] * scalec);
                pp[48 + 3] = float2int8(p0[A_hstep * 12 + 12] * scalec);
                pp[48 + 4] = float2int8(p0[A_hstep * 12 + 1] * scaled);
                pp[48 + 5] = float2int8(p0[A_hstep * 12 + 5] * scaled);
                pp[48 + 6] = float2int8(p0[A_hstep * 12 + 9] * scaled);
                pp[48 + 7] = float2int8(p0[A_hstep * 12 + 13] * scaled);
                pp[48 + 8] = float2int8(p0[A_hstep * 12 + 2] * scalee);
                pp[48 + 9] = float2int8(p0[A_hstep * 12 + 6] * scalee);
                pp[48 + 10] = float2int8(p0[A_hstep * 12 + 10] * scalee);
                pp[48 + 11] = float2int8(p0[A_hstep * 12 + 14] * scalee);
                pp[48 + 12] = float2int8(p0[A_hstep * 12 + 3] * scalef);
                pp[48 + 13] = float2int8(p0[A_hstep * 12 + 7] * scalef);
                pp[48 + 14] = float2int8(p0[A_hstep * 12 + 11] * scalef);
                pp[48 + 15] = float2int8(p0[A_hstep * 12 + 15] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[6] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[7] * scale3);
                pp[8] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp[10] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[11] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp[12] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[13] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp[14] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[15] = float2int8(p0[A_hstep * 4 + 7] * scale7);

                pp[16 + 0] = float2int8(p0[A_hstep * 8 + 0] * scale8);
                pp[16 + 1] = float2int8(p0[A_hstep * 8 + 4] * scale8);
                pp[16 + 2] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[16 + 3] = float2int8(p0[A_hstep * 8 + 5] * scale9);
                pp[16 + 4] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[16 + 5] = float2int8(p0[A_hstep * 8 + 6] * scalea);
                pp[16 + 6] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[16 + 7] = float2int8(p0[A_hstep * 8 + 7] * scaleb);

                pp[16 + 8] = float2int8(p0[A_hstep * 12 + 0] * scalec);
                pp[16 + 9] = float2int8(p0[A_hstep * 12 + 4] * scalec);
                pp[16 + 10] = float2int8(p0[A_hstep * 12 + 1] * scaled);
                pp[16 + 11] = float2int8(p0[A_hstep * 12 + 5] * scaled);
                pp[16 + 12] = float2int8(p0[A_hstep * 12 + 2] * scalee);
                pp[16 + 13] = float2int8(p0[A_hstep * 12 + 6] * scalee);
                pp[16 + 14] = float2int8(p0[A_hstep * 12 + 3] * scalef);
                pp[16 + 15] = float2int8(p0[A_hstep * 12 + 7] * scalef);

                pp += 32;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[6] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[7] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[8] = float2int8(p0[A_hstep * 8] * scale8);
                pp[9] = float2int8(p0[A_hstep * 8 + 1] * scale9);
                pp[10] = float2int8(p0[A_hstep * 8 + 2] * scalea);
                pp[11] = float2int8(p0[A_hstep * 8 + 3] * scaleb);
                pp[12] = float2int8(p0[A_hstep * 12] * scalec);
                pp[13] = float2int8(p0[A_hstep * 12 + 1] * scaled);
                pp[14] = float2int8(p0[A_hstep * 12 + 2] * scalee);
                pp[15] = float2int8(p0[A_hstep * 12 + 3] * scalef);
                pp += 16;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[A_hstep] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep + 2] * scale1);
                pp[7] = float2int8(p0[A_hstep + 3] * scale1);
                pp[8] = float2int8(p0[A_hstep * 2] * scale2);
                pp[9] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 2 + 3] * scale2);
                pp[12] = float2int8(p0[A_hstep * 3] * scale3);
                pp[13] = float2int8(p0[A_hstep * 3 + 1] * scale3);
                pp[14] = float2int8(p0[A_hstep * 3 + 2] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                pp[16] = float2int8(p0[A_hstep * 4] * scale4);
                pp[17] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp[18] = float2int8(p0[A_hstep * 4 + 2] * scale4);
                pp[19] = float2int8(p0[A_hstep * 4 + 3] * scale4);
                pp[20] = float2int8(p0[A_hstep * 5] * scale5);
                pp[21] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp[22] = float2int8(p0[A_hstep * 5 + 2] * scale5);
                pp[23] = float2int8(p0[A_hstep * 5 + 3] * scale5);
                pp[24] = float2int8(p0[A_hstep * 6] * scale6);
                pp[25] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp[26] = float2int8(p0[A_hstep * 6 + 2] * scale6);
                pp[27] = float2int8(p0[A_hstep * 6 + 3] * scale6);
                pp[28] = float2int8(p0[A_hstep * 7] * scale7);
                pp[29] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp[30] = float2int8(p0[A_hstep * 7 + 2] * scale7);
                pp[31] = float2int8(p0[A_hstep * 7 + 3] * scale7);

                pp[32 + 0] = float2int8(p0[A_hstep * 8] * scale8);
                pp[32 + 1] = float2int8(p0[A_hstep * 8 + 1] * scale8);
                pp[32 + 2] = float2int8(p0[A_hstep * 8 + 2] * scale8);
                pp[32 + 3] = float2int8(p0[A_hstep * 8 + 3] * scale8);
                pp[32 + 4] = float2int8(p0[A_hstep * 9] * scale9);
                pp[32 + 5] = float2int8(p0[A_hstep * 9 + 1] * scale9);
                pp[32 + 6] = float2int8(p0[A_hstep * 9 + 2] * scale9);
                pp[32 + 7] = float2int8(p0[A_hstep * 9 + 3] * scale9);
                pp[32 + 8] = float2int8(p0[A_hstep * 10] * scalea);
                pp[32 + 9] = float2int8(p0[A_hstep * 10 + 1] * scalea);
                pp[32 + 10] = float2int8(p0[A_hstep * 10 + 2] * scalea);
                pp[32 + 11] = float2int8(p0[A_hstep * 10 + 3] * scalea);
                pp[32 + 12] = float2int8(p0[A_hstep * 11] * scaleb);
                pp[32 + 13] = float2int8(p0[A_hstep * 11 + 1] * scaleb);
                pp[32 + 14] = float2int8(p0[A_hstep * 11 + 2] * scaleb);
                pp[32 + 15] = float2int8(p0[A_hstep * 11 + 3] * scaleb);
                pp[32 + 16] = float2int8(p0[A_hstep * 12] * scalec);
                pp[32 + 17] = float2int8(p0[A_hstep * 12 + 1] * scalec);
                pp[32 + 18] = float2int8(p0[A_hstep * 12 + 2] * scalec);
                pp[32 + 19] = float2int8(p0[A_hstep * 12 + 3] * scalec);
                pp[32 + 20] = float2int8(p0[A_hstep * 13] * scaled);
                pp[32 + 21] = float2int8(p0[A_hstep * 13 + 1] * scaled);
                pp[32 + 22] = float2int8(p0[A_hstep * 13 + 2] * scaled);
                pp[32 + 23] = float2int8(p0[A_hstep * 13 + 3] * scaled);
                pp[32 + 24] = float2int8(p0[A_hstep * 14] * scalee);
                pp[32 + 25] = float2int8(p0[A_hstep * 14 + 1] * scalee);
                pp[32 + 26] = float2int8(p0[A_hstep * 14 + 2] * scalee);
                pp[32 + 27] = float2int8(p0[A_hstep * 14 + 3] * scalee);
                pp[32 + 28] = float2int8(p0[A_hstep * 15] * scalef);
                pp[32 + 29] = float2int8(p0[A_hstep * 15 + 1] * scalef);
                pp[32 + 30] = float2int8(p0[A_hstep * 15 + 2] * scalef);
                pp[32 + 31] = float2int8(p0[A_hstep * 15 + 3] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[A_hstep * 2] * scale2);
                pp[5] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[6] = float2int8(p0[A_hstep * 3] * scale3);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale3);
                pp[8] = float2int8(p0[A_hstep * 4] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp[10] = float2int8(p0[A_hstep * 5] * scale5);
                pp[11] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp[12] = float2int8(p0[A_hstep * 6] * scale6);
                pp[13] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp[14] = float2int8(p0[A_hstep * 7] * scale7);
                pp[15] = float2int8(p0[A_hstep * 7 + 1] * scale7);

                pp[16 + 0] = float2int8(p0[A_hstep * 8] * scale8);
                pp[16 + 1] = float2int8(p0[A_hstep * 8 + 1] * scale8);
                pp[16 + 2] = float2int8(p0[A_hstep * 9] * scale9);
                pp[16 + 3] = float2int8(p0[A_hstep * 9 + 1] * scale9);
                pp[16 + 4] = float2int8(p0[A_hstep * 10] * scalea);
                pp[16 + 5] = float2int8(p0[A_hstep * 10 + 1] * scalea);
                pp[16 + 6] = float2int8(p0[A_hstep * 11] * scaleb);
                pp[16 + 7] = float2int8(p0[A_hstep * 11 + 1] * scaleb);
                pp[16 + 8] = float2int8(p0[A_hstep * 12] * scalec);
                pp[16 + 9] = float2int8(p0[A_hstep * 12 + 1] * scalec);
                pp[16 + 10] = float2int8(p0[A_hstep * 13] * scaled);
                pp[16 + 11] = float2int8(p0[A_hstep * 13 + 1] * scaled);
                pp[16 + 12] = float2int8(p0[A_hstep * 14] * scalee);
                pp[16 + 13] = float2int8(p0[A_hstep * 14 + 1] * scalee);
                pp[16 + 14] = float2int8(p0[A_hstep * 15] * scalef);
                pp[16 + 15] = float2int8(p0[A_hstep * 15 + 1] * scalef);
                pp += 32;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp[2] = float2int8(p0[A_hstep * 2] * scale2);
                pp[3] = float2int8(p0[A_hstep * 3] * scale3);
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 5] * scale5);
                pp[6] = float2int8(p0[A_hstep * 6] * scale6);
                pp[7] = float2int8(p0[A_hstep * 7] * scale7);
                pp[8] = float2int8(p0[A_hstep * 8] * scale8);
                pp[9] = float2int8(p0[A_hstep * 9] * scale9);
                pp[10] = float2int8(p0[A_hstep * 10] * scalea);
                pp[11] = float2int8(p0[A_hstep * 11] * scaleb);
                pp[12] = float2int8(p0[A_hstep * 12] * scalec);
                pp[13] = float2int8(p0[A_hstep * 13] * scaled);
                pp[14] = float2int8(p0[A_hstep * 14] * scalee);
                pp[15] = float2int8(p0[A_hstep * 15] * scalef);
                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + max_kk * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];

        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[8] * scale0);
                pp[2] = float2int8(p0[16] * scale0);
                pp[3] = float2int8(p0[24] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[17] * scale1);
                pp[7] = float2int8(p0[25] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[10] * scale2);
                pp[10] = float2int8(p0[18] * scale2);
                pp[11] = float2int8(p0[26] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[11] * scale3);
                pp[14] = float2int8(p0[19] * scale3);
                pp[15] = float2int8(p0[27] * scale3);
                pp[16] = float2int8(p0[4] * scale4);
                pp[17] = float2int8(p0[12] * scale4);
                pp[18] = float2int8(p0[20] * scale4);
                pp[19] = float2int8(p0[28] * scale4);
                pp[20] = float2int8(p0[5] * scale5);
                pp[21] = float2int8(p0[13] * scale5);
                pp[22] = float2int8(p0[21] * scale5);
                pp[23] = float2int8(p0[29] * scale5);
                pp[24] = float2int8(p0[6] * scale6);
                pp[25] = float2int8(p0[14] * scale6);
                pp[26] = float2int8(p0[22] * scale6);
                pp[27] = float2int8(p0[30] * scale6);
                pp[28] = float2int8(p0[7] * scale7);
                pp[29] = float2int8(p0[15] * scale7);
                pp[30] = float2int8(p0[23] * scale7);
                pp[31] = float2int8(p0[31] * scale7);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];
                pp += 32;
                p0 += 32;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[8] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[10] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[11] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[12] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[13] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[14] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[15] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[12] * scale4);
                pp1[2] = float2int8(p0[5] * scale5);
                pp1[3] = float2int8(p0[13] * scale5);
                pp1[4] = float2int8(p0[6] * scale6);
                pp1[5] = float2int8(p0[14] * scale6);
                pp1[6] = float2int8(p0[7] * scale7);
                pp1[7] = float2int8(p0[15] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[5] * scale5);
                pp1[2] = float2int8(p0[6] * scale6);
                pp1[3] = float2int8(p0[7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[8] * scale0);
                pp[3] = float2int8(p0[12] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[9] * scale1);
                pp[7] = float2int8(p0[13] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[6] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[14] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[7] * scale3);
                pp[14] = float2int8(p0[11] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                pp[16] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp[17] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp[18] = float2int8(p0[A_hstep * 4 + 8] * scale4);
                pp[19] = float2int8(p0[A_hstep * 4 + 12] * scale4);
                pp[20] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[21] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp[22] = float2int8(p0[A_hstep * 4 + 9] * scale5);
                pp[23] = float2int8(p0[A_hstep * 4 + 13] * scale5);
                pp[24] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[25] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp[26] = float2int8(p0[A_hstep * 4 + 10] * scale6);
                pp[27] = float2int8(p0[A_hstep * 4 + 14] * scale6);
                pp[28] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[29] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp[30] = float2int8(p0[A_hstep * 4 + 11] * scale7);
                pp[31] = float2int8(p0[A_hstep * 4 + 15] * scale7);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];
                pp += 32;
                p0 += 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[6] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[7] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp[10] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[11] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp[12] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[13] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp[14] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp[15] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[A_hstep * 4 + 0] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 4] * scale4);
                pp1[2] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp1[3] = float2int8(p0[A_hstep * 4 + 5] * scale5);
                pp1[4] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp1[5] = float2int8(p0[A_hstep * 4 + 6] * scale6);
                pp1[6] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp1[7] = float2int8(p0[A_hstep * 4 + 7] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp[6] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp[7] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 1] * scale5);
                pp1[2] = float2int8(p0[A_hstep * 4 + 2] * scale6);
                pp1[3] = float2int8(p0[A_hstep * 4 + 3] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[A_hstep] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep + 2] * scale1);
                pp[7] = float2int8(p0[A_hstep + 3] * scale1);
                pp[8] = float2int8(p0[A_hstep * 2] * scale2);
                pp[9] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 2 + 3] * scale2);
                pp[12] = float2int8(p0[A_hstep * 3] * scale3);
                pp[13] = float2int8(p0[A_hstep * 3 + 1] * scale3);
                pp[14] = float2int8(p0[A_hstep * 3 + 2] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                pp[16] = float2int8(p0[A_hstep * 4] * scale4);
                pp[17] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp[18] = float2int8(p0[A_hstep * 4 + 2] * scale4);
                pp[19] = float2int8(p0[A_hstep * 4 + 3] * scale4);
                pp[20] = float2int8(p0[A_hstep * 5] * scale5);
                pp[21] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp[22] = float2int8(p0[A_hstep * 5 + 2] * scale5);
                pp[23] = float2int8(p0[A_hstep * 5 + 3] * scale5);
                pp[24] = float2int8(p0[A_hstep * 6] * scale6);
                pp[25] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp[26] = float2int8(p0[A_hstep * 6 + 2] * scale6);
                pp[27] = float2int8(p0[A_hstep * 6 + 3] * scale6);
                pp[28] = float2int8(p0[A_hstep * 7] * scale7);
                pp[29] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp[30] = float2int8(p0[A_hstep * 7 + 2] * scale7);
                pp[31] = float2int8(p0[A_hstep * 7 + 3] * scale7);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];
                pp += 32;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[A_hstep * 2] * scale2);
                pp[5] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[6] = float2int8(p0[A_hstep * 3] * scale3);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[A_hstep * 4] * scale4);
                pp[9] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp[10] = float2int8(p0[A_hstep * 5] * scale5);
                pp[11] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp[12] = float2int8(p0[A_hstep * 6] * scale6);
                pp[13] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp[14] = float2int8(p0[A_hstep * 7] * scale7);
                pp[15] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 4 + 1] * scale4);
                pp1[2] = float2int8(p0[A_hstep * 5] * scale5);
                pp1[3] = float2int8(p0[A_hstep * 5 + 1] * scale5);
                pp1[4] = float2int8(p0[A_hstep * 6] * scale6);
                pp1[5] = float2int8(p0[A_hstep * 6 + 1] * scale6);
                pp1[6] = float2int8(p0[A_hstep * 7] * scale7);
                pp1[7] = float2int8(p0[A_hstep * 7 + 1] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp[2] = float2int8(p0[A_hstep * 2] * scale2);
                pp[3] = float2int8(p0[A_hstep * 3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[A_hstep * 4] * scale4);
                pp[5] = float2int8(p0[A_hstep * 5] * scale5);
                pp[6] = float2int8(p0[A_hstep * 6] * scale6);
                pp[7] = float2int8(p0[A_hstep * 7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[A_hstep * 4] * scale4);
                pp1[1] = float2int8(p0[A_hstep * 5] * scale5);
                pp1[2] = float2int8(p0[A_hstep * 6] * scale6);
                pp1[3] = float2int8(p0[A_hstep * 7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0++;
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_kk * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];

        // NCNN_LOGE("scale %f %f %f %f", scale0, scale1, scale2, scale3);

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[8] * scale0);
                pp[3] = float2int8(p0[12] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[9] * scale1);
                pp[7] = float2int8(p0[13] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[6] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[14] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[7] * scale3);
                pp[14] = float2int8(p0[11] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                pp += 16;
                p0 += 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[4] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[6] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[7] * scale3);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[A_hstep] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep + 2] * scale1);
                pp[7] = float2int8(p0[A_hstep + 3] * scale1);
                pp[8] = float2int8(p0[A_hstep * 2] * scale2);
                pp[9] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 2 + 3] * scale2);
                pp[12] = float2int8(p0[A_hstep * 3] * scale3);
                pp[13] = float2int8(p0[A_hstep * 3 + 1] * scale3);
                pp[14] = float2int8(p0[A_hstep * 3 + 2] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                pp += 16;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[A_hstep * 2] * scale2);
                pp[5] = float2int8(p0[A_hstep * 2 + 1] * scale2);
                pp[6] = float2int8(p0[A_hstep * 3] * scale3);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale3);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp[2] = float2int8(p0[A_hstep * 2] * scale2);
                pp[3] = float2int8(p0[A_hstep * 3] * scale3);

                pp += 4;
                p0++;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[A_hstep] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep + 2] * scale1);
                pp[7] = float2int8(p0[A_hstep + 3] * scale1);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                pp += 8;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp += 4;
                p0 += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale1);
                pp += 2;
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale = scales[i + ii];

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
                pp += 4;
                p0 += 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_compute_A_tile_fp32_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    const int K = A.dims == 3 ? A.c : A.h;

    // NCNN_LOGE("transpose_compute_A_tile_int8_scales %d %d", max_ii, elempack);

    const float v127_B_scale = 127.f * B_scale;

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 16;

            __m512 _absmax0 = _mm512_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m512 _p = _mm512_loadu_ps(p0);
                _absmax0 = _mm512_max_ps(_absmax0, abs512_ps(_p));
                p0 += A_hstep * 16;
            }
            float absmax = _mm512_reduce_max_ps(_absmax0);

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 8;

            __m256 _absmax0 = _mm256_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m256 _p = _mm256_loadu_ps(p0);
                _absmax0 = _mm256_max_ps(_absmax0, abs256_ps(_p));
                p0 += A_hstep * 8;
            }
            float absmax = _mm256_reduce_max_ps(_absmax0);

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 4;

            __m128 _absmax0 = _mm_setzero_ps();
            int kk = 0;
            for (; kk < K; kk++)
            {
                __m128 _p = _mm_loadu_ps(p0);
                _absmax0 = _mm_max_ps(_absmax0, abs_ps(_p));
                p0 += A_hstep * 4;
            }
            float absmax = _mm_reduce_max_ps(_absmax0);

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        int ii = 0;
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii);

            float absmax = 0.f;
            for (int kk = 0; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(p0[0]));
                p0 += A_hstep;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_A_tile_fp32_to_int8_avx512vnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_A_tile_fp32_to_int8_avxvnni(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        transpose_pack_A_tile_fp32_to_int8_avx2(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = (signed char*)AT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        const float scale8 = scales[i + ii + 8];
        const float scale9 = scales[i + ii + 9];
        const float scalea = scales[i + ii + 10];
        const float scaleb = scales[i + ii + 11];
        const float scalec = scales[i + ii + 12];
        const float scaled = scales[i + ii + 13];
        const float scalee = scales[i + ii + 14];
        const float scalef = scales[i + ii + 15];

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2 + 0] * scale0);
                pp[3] = float2int8(p0[2 + 1] * scale0);
                pp[4] = float2int8(p0[16] * scale1);
                pp[5] = float2int8(p0[17] * scale1);
                pp[6] = float2int8(p0[2 + 16] * scale1);
                pp[7] = float2int8(p0[2 + 17] * scale1);
                pp[8] = float2int8(p0[32] * scale2);
                pp[9] = float2int8(p0[33] * scale2);
                pp[10] = float2int8(p0[2 + 32] * scale2);
                pp[11] = float2int8(p0[2 + 33] * scale2);
                pp[12] = float2int8(p0[48] * scale3);
                pp[13] = float2int8(p0[49] * scale3);
                pp[14] = float2int8(p0[2 + 48] * scale3);
                pp[15] = float2int8(p0[2 + 49] * scale3);
                pp[16] = float2int8(p0[64] * scale4);
                pp[17] = float2int8(p0[65] * scale4);
                pp[18] = float2int8(p0[2 + 64] * scale4);
                pp[19] = float2int8(p0[2 + 65] * scale4);
                pp[20] = float2int8(p0[80] * scale5);
                pp[21] = float2int8(p0[81] * scale5);
                pp[22] = float2int8(p0[2 + 80] * scale5);
                pp[23] = float2int8(p0[2 + 81] * scale5);
                pp[24] = float2int8(p0[96] * scale6);
                pp[25] = float2int8(p0[97] * scale6);
                pp[26] = float2int8(p0[2 + 96] * scale6);
                pp[27] = float2int8(p0[2 + 97] * scale6);
                pp[28] = float2int8(p0[112] * scale7);
                pp[29] = float2int8(p0[113] * scale7);
                pp[30] = float2int8(p0[2 + 112] * scale7);
                pp[31] = float2int8(p0[2 + 113] * scale7);

                pp[32 + 0] = float2int8(p0[128 + 0] * scale8);
                pp[32 + 1] = float2int8(p0[128 + 1] * scale8);
                pp[32 + 2] = float2int8(p0[2 + 128 + 0] * scale8);
                pp[32 + 3] = float2int8(p0[2 + 128 + 1] * scale8);
                pp[32 + 4] = float2int8(p0[128 + 16] * scale9);
                pp[32 + 5] = float2int8(p0[128 + 17] * scale9);
                pp[32 + 6] = float2int8(p0[2 + 128 + 16] * scale9);
                pp[32 + 7] = float2int8(p0[2 + 128 + 17] * scale9);
                pp[32 + 8] = float2int8(p0[128 + 32] * scalea);
                pp[32 + 9] = float2int8(p0[128 + 33] * scalea);
                pp[32 + 10] = float2int8(p0[2 + 128 + 32] * scalea);
                pp[32 + 11] = float2int8(p0[2 + 128 + 33] * scalea);
                pp[32 + 12] = float2int8(p0[128 + 48] * scaleb);
                pp[32 + 13] = float2int8(p0[128 + 49] * scaleb);
                pp[32 + 14] = float2int8(p0[2 + 128 + 48] * scaleb);
                pp[32 + 15] = float2int8(p0[2 + 128 + 49] * scaleb);
                pp[32 + 16] = float2int8(p0[128 + 64] * scalec);
                pp[32 + 17] = float2int8(p0[128 + 65] * scalec);
                pp[32 + 18] = float2int8(p0[2 + 128 + 64] * scalec);
                pp[32 + 19] = float2int8(p0[2 + 128 + 65] * scalec);
                pp[32 + 20] = float2int8(p0[128 + 80] * scaled);
                pp[32 + 21] = float2int8(p0[128 + 81] * scaled);
                pp[32 + 22] = float2int8(p0[2 + 128 + 80] * scaled);
                pp[32 + 23] = float2int8(p0[2 + 128 + 81] * scaled);
                pp[32 + 24] = float2int8(p0[128 + 96] * scalee);
                pp[32 + 25] = float2int8(p0[128 + 97] * scalee);
                pp[32 + 26] = float2int8(p0[2 + 128 + 96] * scalee);
                pp[32 + 27] = float2int8(p0[2 + 128 + 97] * scalee);
                pp[32 + 28] = float2int8(p0[128 + 112] * scalef);
                pp[32 + 29] = float2int8(p0[128 + 113] * scalef);
                pp[32 + 30] = float2int8(p0[2 + 128 + 112] * scalef);
                pp[32 + 31] = float2int8(p0[2 + 128 + 113] * scalef);

                pp[64 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[64 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[64 + 2] = float2int8(p0[6 + 0] * scale0);
                pp[64 + 3] = float2int8(p0[6 + 1] * scale0);
                pp[64 + 4] = float2int8(p0[4 + 16] * scale1);
                pp[64 + 5] = float2int8(p0[4 + 17] * scale1);
                pp[64 + 6] = float2int8(p0[6 + 16] * scale1);
                pp[64 + 7] = float2int8(p0[6 + 17] * scale1);
                pp[64 + 8] = float2int8(p0[4 + 32] * scale2);
                pp[64 + 9] = float2int8(p0[4 + 33] * scale2);
                pp[64 + 10] = float2int8(p0[6 + 32] * scale2);
                pp[64 + 11] = float2int8(p0[6 + 33] * scale2);
                pp[64 + 12] = float2int8(p0[4 + 48] * scale3);
                pp[64 + 13] = float2int8(p0[4 + 49] * scale3);
                pp[64 + 14] = float2int8(p0[6 + 48] * scale3);
                pp[64 + 15] = float2int8(p0[6 + 49] * scale3);
                pp[64 + 16] = float2int8(p0[4 + 64] * scale4);
                pp[64 + 17] = float2int8(p0[4 + 65] * scale4);
                pp[64 + 18] = float2int8(p0[6 + 64] * scale4);
                pp[64 + 19] = float2int8(p0[6 + 65] * scale4);
                pp[64 + 20] = float2int8(p0[4 + 80] * scale5);
                pp[64 + 21] = float2int8(p0[4 + 81] * scale5);
                pp[64 + 22] = float2int8(p0[6 + 80] * scale5);
                pp[64 + 23] = float2int8(p0[6 + 81] * scale5);
                pp[64 + 24] = float2int8(p0[4 + 96] * scale6);
                pp[64 + 25] = float2int8(p0[4 + 97] * scale6);
                pp[64 + 26] = float2int8(p0[6 + 96] * scale6);
                pp[64 + 27] = float2int8(p0[6 + 97] * scale6);
                pp[64 + 28] = float2int8(p0[4 + 112] * scale7);
                pp[64 + 29] = float2int8(p0[4 + 113] * scale7);
                pp[64 + 30] = float2int8(p0[6 + 112] * scale7);
                pp[64 + 31] = float2int8(p0[6 + 113] * scale7);

                pp[96 + 0] = float2int8(p0[4 + 128 + 0] * scale8);
                pp[96 + 1] = float2int8(p0[4 + 128 + 1] * scale8);
                pp[96 + 2] = float2int8(p0[6 + 128 + 0] * scale8);
                pp[96 + 3] = float2int8(p0[6 + 128 + 1] * scale8);
                pp[96 + 4] = float2int8(p0[4 + 128 + 16] * scale9);
                pp[96 + 5] = float2int8(p0[4 + 128 + 17] * scale9);
                pp[96 + 6] = float2int8(p0[6 + 128 + 16] * scale9);
                pp[96 + 7] = float2int8(p0[6 + 128 + 17] * scale9);
                pp[96 + 8] = float2int8(p0[4 + 128 + 32] * scalea);
                pp[96 + 9] = float2int8(p0[4 + 128 + 33] * scalea);
                pp[96 + 10] = float2int8(p0[6 + 128 + 32] * scalea);
                pp[96 + 11] = float2int8(p0[6 + 128 + 33] * scalea);
                pp[96 + 12] = float2int8(p0[4 + 128 + 48] * scaleb);
                pp[96 + 13] = float2int8(p0[4 + 128 + 49] * scaleb);
                pp[96 + 14] = float2int8(p0[6 + 128 + 48] * scaleb);
                pp[96 + 15] = float2int8(p0[6 + 128 + 49] * scaleb);
                pp[96 + 16] = float2int8(p0[4 + 128 + 64] * scalec);
                pp[96 + 17] = float2int8(p0[4 + 128 + 65] * scalec);
                pp[96 + 18] = float2int8(p0[6 + 128 + 64] * scalec);
                pp[96 + 19] = float2int8(p0[6 + 128 + 65] * scalec);
                pp[96 + 20] = float2int8(p0[4 + 128 + 80] * scaled);
                pp[96 + 21] = float2int8(p0[4 + 128 + 81] * scaled);
                pp[96 + 22] = float2int8(p0[6 + 128 + 80] * scaled);
                pp[96 + 23] = float2int8(p0[6 + 128 + 81] * scaled);
                pp[96 + 24] = float2int8(p0[4 + 128 + 96] * scalee);
                pp[96 + 25] = float2int8(p0[4 + 128 + 97] * scalee);
                pp[96 + 26] = float2int8(p0[6 + 128 + 96] * scalee);
                pp[96 + 27] = float2int8(p0[6 + 128 + 97] * scalee);
                pp[96 + 28] = float2int8(p0[4 + 128 + 112] * scalef);
                pp[96 + 29] = float2int8(p0[4 + 128 + 113] * scalef);
                pp[96 + 30] = float2int8(p0[6 + 128 + 112] * scalef);
                pp[96 + 31] = float2int8(p0[6 + 128 + 113] * scalef);

                pp[128 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[128 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[128 + 2] = float2int8(p0[10 + 0] * scale0);
                pp[128 + 3] = float2int8(p0[10 + 1] * scale0);
                pp[128 + 4] = float2int8(p0[8 + 16] * scale1);
                pp[128 + 5] = float2int8(p0[8 + 17] * scale1);
                pp[128 + 6] = float2int8(p0[10 + 16] * scale1);
                pp[128 + 7] = float2int8(p0[10 + 17] * scale1);
                pp[128 + 8] = float2int8(p0[8 + 32] * scale2);
                pp[128 + 9] = float2int8(p0[8 + 33] * scale2);
                pp[128 + 10] = float2int8(p0[10 + 32] * scale2);
                pp[128 + 11] = float2int8(p0[10 + 33] * scale2);
                pp[128 + 12] = float2int8(p0[8 + 48] * scale3);
                pp[128 + 13] = float2int8(p0[8 + 49] * scale3);
                pp[128 + 14] = float2int8(p0[10 + 48] * scale3);
                pp[128 + 15] = float2int8(p0[10 + 49] * scale3);
                pp[128 + 16] = float2int8(p0[8 + 64] * scale4);
                pp[128 + 17] = float2int8(p0[8 + 65] * scale4);
                pp[128 + 18] = float2int8(p0[10 + 64] * scale4);
                pp[128 + 19] = float2int8(p0[10 + 65] * scale4);
                pp[128 + 20] = float2int8(p0[8 + 80] * scale5);
                pp[128 + 21] = float2int8(p0[8 + 81] * scale5);
                pp[128 + 22] = float2int8(p0[10 + 80] * scale5);
                pp[128 + 23] = float2int8(p0[10 + 81] * scale5);
                pp[128 + 24] = float2int8(p0[8 + 96] * scale6);
                pp[128 + 25] = float2int8(p0[8 + 97] * scale6);
                pp[128 + 26] = float2int8(p0[10 + 96] * scale6);
                pp[128 + 27] = float2int8(p0[10 + 97] * scale6);
                pp[128 + 28] = float2int8(p0[8 + 112] * scale7);
                pp[128 + 29] = float2int8(p0[8 + 113] * scale7);
                pp[128 + 30] = float2int8(p0[10 + 112] * scale7);
                pp[128 + 31] = float2int8(p0[10 + 113] * scale7);

                pp[160 + 0] = float2int8(p0[8 + 128 + 0] * scale8);
                pp[160 + 1] = float2int8(p0[8 + 128 + 1] * scale8);
                pp[160 + 2] = float2int8(p0[10 + 128 + 0] * scale8);
                pp[160 + 3] = float2int8(p0[10 + 128 + 1] * scale8);
                pp[160 + 4] = float2int8(p0[8 + 128 + 16] * scale9);
                pp[160 + 5] = float2int8(p0[8 + 128 + 17] * scale9);
                pp[160 + 6] = float2int8(p0[10 + 128 + 16] * scale9);
                pp[160 + 7] = float2int8(p0[10 + 128 + 17] * scale9);
                pp[160 + 8] = float2int8(p0[8 + 128 + 32] * scalea);
                pp[160 + 9] = float2int8(p0[8 + 128 + 33] * scalea);
                pp[160 + 10] = float2int8(p0[10 + 128 + 32] * scalea);
                pp[160 + 11] = float2int8(p0[10 + 128 + 33] * scalea);
                pp[160 + 12] = float2int8(p0[8 + 128 + 48] * scaleb);
                pp[160 + 13] = float2int8(p0[8 + 128 + 49] * scaleb);
                pp[160 + 14] = float2int8(p0[10 + 128 + 48] * scaleb);
                pp[160 + 15] = float2int8(p0[10 + 128 + 49] * scaleb);
                pp[160 + 16] = float2int8(p0[8 + 128 + 64] * scalec);
                pp[160 + 17] = float2int8(p0[8 + 128 + 65] * scalec);
                pp[160 + 18] = float2int8(p0[10 + 128 + 64] * scalec);
                pp[160 + 19] = float2int8(p0[10 + 128 + 65] * scalec);
                pp[160 + 20] = float2int8(p0[8 + 128 + 80] * scaled);
                pp[160 + 21] = float2int8(p0[8 + 128 + 81] * scaled);
                pp[160 + 22] = float2int8(p0[10 + 128 + 80] * scaled);
                pp[160 + 23] = float2int8(p0[10 + 128 + 81] * scaled);
                pp[160 + 24] = float2int8(p0[8 + 128 + 96] * scalee);
                pp[160 + 25] = float2int8(p0[8 + 128 + 97] * scalee);
                pp[160 + 26] = float2int8(p0[10 + 128 + 96] * scalee);
                pp[160 + 27] = float2int8(p0[10 + 128 + 97] * scalee);
                pp[160 + 28] = float2int8(p0[8 + 128 + 112] * scalef);
                pp[160 + 29] = float2int8(p0[8 + 128 + 113] * scalef);
                pp[160 + 30] = float2int8(p0[10 + 128 + 112] * scalef);
                pp[160 + 31] = float2int8(p0[10 + 128 + 113] * scalef);

                pp[192 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[192 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[192 + 2] = float2int8(p0[14 + 0] * scale0);
                pp[192 + 3] = float2int8(p0[14 + 1] * scale0);
                pp[192 + 4] = float2int8(p0[12 + 16] * scale1);
                pp[192 + 5] = float2int8(p0[12 + 17] * scale1);
                pp[192 + 6] = float2int8(p0[14 + 16] * scale1);
                pp[192 + 7] = float2int8(p0[14 + 17] * scale1);
                pp[192 + 8] = float2int8(p0[12 + 32] * scale2);
                pp[192 + 9] = float2int8(p0[12 + 33] * scale2);
                pp[192 + 10] = float2int8(p0[14 + 32] * scale2);
                pp[192 + 11] = float2int8(p0[14 + 33] * scale2);
                pp[192 + 12] = float2int8(p0[12 + 48] * scale3);
                pp[192 + 13] = float2int8(p0[12 + 49] * scale3);
                pp[192 + 14] = float2int8(p0[14 + 48] * scale3);
                pp[192 + 15] = float2int8(p0[14 + 49] * scale3);
                pp[192 + 16] = float2int8(p0[12 + 64] * scale4);
                pp[192 + 17] = float2int8(p0[12 + 65] * scale4);
                pp[192 + 18] = float2int8(p0[14 + 64] * scale4);
                pp[192 + 19] = float2int8(p0[14 + 65] * scale4);
                pp[192 + 20] = float2int8(p0[12 + 80] * scale5);
                pp[192 + 21] = float2int8(p0[12 + 81] * scale5);
                pp[192 + 22] = float2int8(p0[14 + 80] * scale5);
                pp[192 + 23] = float2int8(p0[14 + 81] * scale5);
                pp[192 + 24] = float2int8(p0[12 + 96] * scale6);
                pp[192 + 25] = float2int8(p0[12 + 97] * scale6);
                pp[192 + 26] = float2int8(p0[14 + 96] * scale6);
                pp[192 + 27] = float2int8(p0[14 + 97] * scale6);
                pp[192 + 28] = float2int8(p0[12 + 112] * scale7);
                pp[192 + 29] = float2int8(p0[12 + 113] * scale7);
                pp[192 + 30] = float2int8(p0[14 + 112] * scale7);
                pp[192 + 31] = float2int8(p0[14 + 113] * scale7);

                pp[224 + 0] = float2int8(p0[12 + 128 + 0] * scale8);
                pp[224 + 1] = float2int8(p0[12 + 128 + 1] * scale8);
                pp[224 + 2] = float2int8(p0[14 + 128 + 0] * scale8);
                pp[224 + 3] = float2int8(p0[14 + 128 + 1] * scale8);
                pp[224 + 4] = float2int8(p0[12 + 128 + 16] * scale9);
                pp[224 + 5] = float2int8(p0[12 + 128 + 17] * scale9);
                pp[224 + 6] = float2int8(p0[14 + 128 + 16] * scale9);
                pp[224 + 7] = float2int8(p0[14 + 128 + 17] * scale9);
                pp[224 + 8] = float2int8(p0[12 + 128 + 32] * scalea);
                pp[224 + 9] = float2int8(p0[12 + 128 + 33] * scalea);
                pp[224 + 10] = float2int8(p0[14 + 128 + 32] * scalea);
                pp[224 + 11] = float2int8(p0[14 + 128 + 33] * scalea);
                pp[224 + 12] = float2int8(p0[12 + 128 + 48] * scaleb);
                pp[224 + 13] = float2int8(p0[12 + 128 + 49] * scaleb);
                pp[224 + 14] = float2int8(p0[14 + 128 + 48] * scaleb);
                pp[224 + 15] = float2int8(p0[14 + 128 + 49] * scaleb);
                pp[224 + 16] = float2int8(p0[12 + 128 + 64] * scalec);
                pp[224 + 17] = float2int8(p0[12 + 128 + 65] * scalec);
                pp[224 + 18] = float2int8(p0[14 + 128 + 64] * scalec);
                pp[224 + 19] = float2int8(p0[14 + 128 + 65] * scalec);
                pp[224 + 20] = float2int8(p0[12 + 128 + 80] * scaled);
                pp[224 + 21] = float2int8(p0[12 + 128 + 81] * scaled);
                pp[224 + 22] = float2int8(p0[14 + 128 + 80] * scaled);
                pp[224 + 23] = float2int8(p0[14 + 128 + 81] * scaled);
                pp[224 + 24] = float2int8(p0[12 + 128 + 96] * scalee);
                pp[224 + 25] = float2int8(p0[12 + 128 + 97] * scalee);
                pp[224 + 26] = float2int8(p0[14 + 128 + 96] * scalee);
                pp[224 + 27] = float2int8(p0[14 + 128 + 97] * scalee);
                pp[224 + 28] = float2int8(p0[12 + 128 + 112] * scalef);
                pp[224 + 29] = float2int8(p0[12 + 128 + 113] * scalef);
                pp[224 + 30] = float2int8(p0[14 + 128 + 112] * scalef);
                pp[224 + 31] = float2int8(p0[14 + 128 + 113] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                w_shift0 += pp[64 + 0];
                w_shift0 += pp[64 + 1];
                w_shift0 += pp[64 + 2];
                w_shift0 += pp[64 + 3];
                w_shift1 += pp[64 + 4];
                w_shift1 += pp[64 + 5];
                w_shift1 += pp[64 + 6];
                w_shift1 += pp[64 + 7];
                w_shift2 += pp[64 + 8];
                w_shift2 += pp[64 + 9];
                w_shift2 += pp[64 + 10];
                w_shift2 += pp[64 + 11];
                w_shift3 += pp[64 + 12];
                w_shift3 += pp[64 + 13];
                w_shift3 += pp[64 + 14];
                w_shift3 += pp[64 + 15];
                w_shift4 += pp[64 + 16];
                w_shift4 += pp[64 + 17];
                w_shift4 += pp[64 + 18];
                w_shift4 += pp[64 + 19];
                w_shift5 += pp[64 + 20];
                w_shift5 += pp[64 + 21];
                w_shift5 += pp[64 + 22];
                w_shift5 += pp[64 + 23];
                w_shift6 += pp[64 + 24];
                w_shift6 += pp[64 + 25];
                w_shift6 += pp[64 + 26];
                w_shift6 += pp[64 + 27];
                w_shift7 += pp[64 + 28];
                w_shift7 += pp[64 + 29];
                w_shift7 += pp[64 + 30];
                w_shift7 += pp[64 + 31];

                w_shift8 += pp[96 + 0];
                w_shift8 += pp[96 + 1];
                w_shift8 += pp[96 + 2];
                w_shift8 += pp[96 + 3];
                w_shift9 += pp[96 + 4];
                w_shift9 += pp[96 + 5];
                w_shift9 += pp[96 + 6];
                w_shift9 += pp[96 + 7];
                w_shifta += pp[96 + 8];
                w_shifta += pp[96 + 9];
                w_shifta += pp[96 + 10];
                w_shifta += pp[96 + 11];
                w_shiftb += pp[96 + 12];
                w_shiftb += pp[96 + 13];
                w_shiftb += pp[96 + 14];
                w_shiftb += pp[96 + 15];
                w_shiftc += pp[96 + 16];
                w_shiftc += pp[96 + 17];
                w_shiftc += pp[96 + 18];
                w_shiftc += pp[96 + 19];
                w_shiftd += pp[96 + 20];
                w_shiftd += pp[96 + 21];
                w_shiftd += pp[96 + 22];
                w_shiftd += pp[96 + 23];
                w_shifte += pp[96 + 24];
                w_shifte += pp[96 + 25];
                w_shifte += pp[96 + 26];
                w_shifte += pp[96 + 27];
                w_shiftf += pp[96 + 28];
                w_shiftf += pp[96 + 29];
                w_shiftf += pp[96 + 30];
                w_shiftf += pp[96 + 31];

                w_shift0 += pp[128 + 0];
                w_shift0 += pp[128 + 1];
                w_shift0 += pp[128 + 2];
                w_shift0 += pp[128 + 3];
                w_shift1 += pp[128 + 4];
                w_shift1 += pp[128 + 5];
                w_shift1 += pp[128 + 6];
                w_shift1 += pp[128 + 7];
                w_shift2 += pp[128 + 8];
                w_shift2 += pp[128 + 9];
                w_shift2 += pp[128 + 10];
                w_shift2 += pp[128 + 11];
                w_shift3 += pp[128 + 12];
                w_shift3 += pp[128 + 13];
                w_shift3 += pp[128 + 14];
                w_shift3 += pp[128 + 15];
                w_shift4 += pp[128 + 16];
                w_shift4 += pp[128 + 17];
                w_shift4 += pp[128 + 18];
                w_shift4 += pp[128 + 19];
                w_shift5 += pp[128 + 20];
                w_shift5 += pp[128 + 21];
                w_shift5 += pp[128 + 22];
                w_shift5 += pp[128 + 23];
                w_shift6 += pp[128 + 24];
                w_shift6 += pp[128 + 25];
                w_shift6 += pp[128 + 26];
                w_shift6 += pp[128 + 27];
                w_shift7 += pp[128 + 28];
                w_shift7 += pp[128 + 29];
                w_shift7 += pp[128 + 30];
                w_shift7 += pp[128 + 31];

                w_shift8 += pp[160 + 0];
                w_shift8 += pp[160 + 1];
                w_shift8 += pp[160 + 2];
                w_shift8 += pp[160 + 3];
                w_shift9 += pp[160 + 4];
                w_shift9 += pp[160 + 5];
                w_shift9 += pp[160 + 6];
                w_shift9 += pp[160 + 7];
                w_shifta += pp[160 + 8];
                w_shifta += pp[160 + 9];
                w_shifta += pp[160 + 10];
                w_shifta += pp[160 + 11];
                w_shiftb += pp[160 + 12];
                w_shiftb += pp[160 + 13];
                w_shiftb += pp[160 + 14];
                w_shiftb += pp[160 + 15];
                w_shiftc += pp[160 + 16];
                w_shiftc += pp[160 + 17];
                w_shiftc += pp[160 + 18];
                w_shiftc += pp[160 + 19];
                w_shiftd += pp[160 + 20];
                w_shiftd += pp[160 + 21];
                w_shiftd += pp[160 + 22];
                w_shiftd += pp[160 + 23];
                w_shifte += pp[160 + 24];
                w_shifte += pp[160 + 25];
                w_shifte += pp[160 + 26];
                w_shifte += pp[160 + 27];
                w_shiftf += pp[160 + 28];
                w_shiftf += pp[160 + 29];
                w_shiftf += pp[160 + 30];
                w_shiftf += pp[160 + 31];

                w_shift0 += pp[192 + 0];
                w_shift0 += pp[192 + 1];
                w_shift0 += pp[192 + 2];
                w_shift0 += pp[192 + 3];
                w_shift1 += pp[192 + 4];
                w_shift1 += pp[192 + 5];
                w_shift1 += pp[192 + 6];
                w_shift1 += pp[192 + 7];
                w_shift2 += pp[192 + 8];
                w_shift2 += pp[192 + 9];
                w_shift2 += pp[192 + 10];
                w_shift2 += pp[192 + 11];
                w_shift3 += pp[192 + 12];
                w_shift3 += pp[192 + 13];
                w_shift3 += pp[192 + 14];
                w_shift3 += pp[192 + 15];
                w_shift4 += pp[192 + 16];
                w_shift4 += pp[192 + 17];
                w_shift4 += pp[192 + 18];
                w_shift4 += pp[192 + 19];
                w_shift5 += pp[192 + 20];
                w_shift5 += pp[192 + 21];
                w_shift5 += pp[192 + 22];
                w_shift5 += pp[192 + 23];
                w_shift6 += pp[192 + 24];
                w_shift6 += pp[192 + 25];
                w_shift6 += pp[192 + 26];
                w_shift6 += pp[192 + 27];
                w_shift7 += pp[192 + 28];
                w_shift7 += pp[192 + 29];
                w_shift7 += pp[192 + 30];
                w_shift7 += pp[192 + 31];

                w_shift8 += pp[224 + 0];
                w_shift8 += pp[224 + 1];
                w_shift8 += pp[224 + 2];
                w_shift8 += pp[224 + 3];
                w_shift9 += pp[224 + 4];
                w_shift9 += pp[224 + 5];
                w_shift9 += pp[224 + 6];
                w_shift9 += pp[224 + 7];
                w_shifta += pp[224 + 8];
                w_shifta += pp[224 + 9];
                w_shifta += pp[224 + 10];
                w_shifta += pp[224 + 11];
                w_shiftb += pp[224 + 12];
                w_shiftb += pp[224 + 13];
                w_shiftb += pp[224 + 14];
                w_shiftb += pp[224 + 15];
                w_shiftc += pp[224 + 16];
                w_shiftc += pp[224 + 17];
                w_shiftc += pp[224 + 18];
                w_shiftc += pp[224 + 19];
                w_shiftd += pp[224 + 20];
                w_shiftd += pp[224 + 21];
                w_shiftd += pp[224 + 22];
                w_shiftd += pp[224 + 23];
                w_shifte += pp[224 + 24];
                w_shifte += pp[224 + 25];
                w_shifte += pp[224 + 26];
                w_shifte += pp[224 + 27];
                w_shiftf += pp[224 + 28];
                w_shiftf += pp[224 + 29];
                w_shiftf += pp[224 + 30];
                w_shiftf += pp[224 + 31];

                pp += 256;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[16] * scale1);
                pp[3] = float2int8(p0[17] * scale1);
                pp[4] = float2int8(p0[32] * scale2);
                pp[5] = float2int8(p0[33] * scale2);
                pp[6] = float2int8(p0[48] * scale3);
                pp[7] = float2int8(p0[49] * scale3);
                pp[8] = float2int8(p0[64] * scale4);
                pp[9] = float2int8(p0[65] * scale4);
                pp[10] = float2int8(p0[80] * scale5);
                pp[11] = float2int8(p0[81] * scale5);
                pp[12] = float2int8(p0[96] * scale6);
                pp[13] = float2int8(p0[97] * scale6);
                pp[14] = float2int8(p0[112] * scale7);
                pp[15] = float2int8(p0[113] * scale7);

                pp[16 + 0] = float2int8(p0[128 + 0] * scale8);
                pp[16 + 1] = float2int8(p0[128 + 1] * scale8);
                pp[16 + 2] = float2int8(p0[128 + 16] * scale9);
                pp[16 + 3] = float2int8(p0[128 + 17] * scale9);
                pp[16 + 4] = float2int8(p0[128 + 32] * scalea);
                pp[16 + 5] = float2int8(p0[128 + 33] * scalea);
                pp[16 + 6] = float2int8(p0[128 + 48] * scaleb);
                pp[16 + 7] = float2int8(p0[128 + 49] * scaleb);
                pp[16 + 8] = float2int8(p0[128 + 64] * scalec);
                pp[16 + 9] = float2int8(p0[128 + 65] * scalec);
                pp[16 + 10] = float2int8(p0[128 + 80] * scaled);
                pp[16 + 11] = float2int8(p0[128 + 81] * scaled);
                pp[16 + 12] = float2int8(p0[128 + 96] * scalee);
                pp[16 + 13] = float2int8(p0[128 + 97] * scalee);
                pp[16 + 14] = float2int8(p0[128 + 112] * scalef);
                pp[16 + 15] = float2int8(p0[128 + 113] * scalef);

                pp[32 + 0] = float2int8(p0[2 + 0] * scale0);
                pp[32 + 1] = float2int8(p0[2 + 1] * scale0);
                pp[32 + 2] = float2int8(p0[2 + 16] * scale1);
                pp[32 + 3] = float2int8(p0[2 + 17] * scale1);
                pp[32 + 4] = float2int8(p0[2 + 32] * scale2);
                pp[32 + 5] = float2int8(p0[2 + 33] * scale2);
                pp[32 + 6] = float2int8(p0[2 + 48] * scale3);
                pp[32 + 7] = float2int8(p0[2 + 49] * scale3);
                pp[32 + 8] = float2int8(p0[2 + 64] * scale4);
                pp[32 + 9] = float2int8(p0[2 + 65] * scale4);
                pp[32 + 10] = float2int8(p0[2 + 80] * scale5);
                pp[32 + 11] = float2int8(p0[2 + 81] * scale5);
                pp[32 + 12] = float2int8(p0[2 + 96] * scale6);
                pp[32 + 13] = float2int8(p0[2 + 97] * scale6);
                pp[32 + 14] = float2int8(p0[2 + 112] * scale7);
                pp[32 + 15] = float2int8(p0[2 + 113] * scale7);

                pp[48 + 0] = float2int8(p0[2 + 128 + 0] * scale8);
                pp[48 + 1] = float2int8(p0[2 + 128 + 1] * scale8);
                pp[48 + 2] = float2int8(p0[2 + 128 + 16] * scale9);
                pp[48 + 3] = float2int8(p0[2 + 128 + 17] * scale9);
                pp[48 + 4] = float2int8(p0[2 + 128 + 32] * scalea);
                pp[48 + 5] = float2int8(p0[2 + 128 + 33] * scalea);
                pp[48 + 6] = float2int8(p0[2 + 128 + 48] * scaleb);
                pp[48 + 7] = float2int8(p0[2 + 128 + 49] * scaleb);
                pp[48 + 8] = float2int8(p0[2 + 128 + 64] * scalec);
                pp[48 + 9] = float2int8(p0[2 + 128 + 65] * scalec);
                pp[48 + 10] = float2int8(p0[2 + 128 + 80] * scaled);
                pp[48 + 11] = float2int8(p0[2 + 128 + 81] * scaled);
                pp[48 + 12] = float2int8(p0[2 + 128 + 96] * scalee);
                pp[48 + 13] = float2int8(p0[2 + 128 + 97] * scalee);
                pp[48 + 14] = float2int8(p0[2 + 128 + 112] * scalef);
                pp[48 + 15] = float2int8(p0[2 + 128 + 113] * scalef);

                pp[64 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[64 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[64 + 2] = float2int8(p0[4 + 16] * scale1);
                pp[64 + 3] = float2int8(p0[4 + 17] * scale1);
                pp[64 + 4] = float2int8(p0[4 + 32] * scale2);
                pp[64 + 5] = float2int8(p0[4 + 33] * scale2);
                pp[64 + 6] = float2int8(p0[4 + 48] * scale3);
                pp[64 + 7] = float2int8(p0[4 + 49] * scale3);
                pp[64 + 8] = float2int8(p0[4 + 64] * scale4);
                pp[64 + 9] = float2int8(p0[4 + 65] * scale4);
                pp[64 + 10] = float2int8(p0[4 + 80] * scale5);
                pp[64 + 11] = float2int8(p0[4 + 81] * scale5);
                pp[64 + 12] = float2int8(p0[4 + 96] * scale6);
                pp[64 + 13] = float2int8(p0[4 + 97] * scale6);
                pp[64 + 14] = float2int8(p0[4 + 112] * scale7);
                pp[64 + 15] = float2int8(p0[4 + 113] * scale7);

                pp[80 + 0] = float2int8(p0[4 + 128 + 0] * scale8);
                pp[80 + 1] = float2int8(p0[4 + 128 + 1] * scale8);
                pp[80 + 2] = float2int8(p0[4 + 128 + 16] * scale9);
                pp[80 + 3] = float2int8(p0[4 + 128 + 17] * scale9);
                pp[80 + 4] = float2int8(p0[4 + 128 + 32] * scalea);
                pp[80 + 5] = float2int8(p0[4 + 128 + 33] * scalea);
                pp[80 + 6] = float2int8(p0[4 + 128 + 48] * scaleb);
                pp[80 + 7] = float2int8(p0[4 + 128 + 49] * scaleb);
                pp[80 + 8] = float2int8(p0[4 + 128 + 64] * scalec);
                pp[80 + 9] = float2int8(p0[4 + 128 + 65] * scalec);
                pp[80 + 10] = float2int8(p0[4 + 128 + 80] * scaled);
                pp[80 + 11] = float2int8(p0[4 + 128 + 81] * scaled);
                pp[80 + 12] = float2int8(p0[4 + 128 + 96] * scalee);
                pp[80 + 13] = float2int8(p0[4 + 128 + 97] * scalee);
                pp[80 + 14] = float2int8(p0[4 + 128 + 112] * scalef);
                pp[80 + 15] = float2int8(p0[4 + 128 + 113] * scalef);

                pp[96 + 0] = float2int8(p0[6 + 0] * scale0);
                pp[96 + 1] = float2int8(p0[6 + 1] * scale0);
                pp[96 + 2] = float2int8(p0[6 + 16] * scale1);
                pp[96 + 3] = float2int8(p0[6 + 17] * scale1);
                pp[96 + 4] = float2int8(p0[6 + 32] * scale2);
                pp[96 + 5] = float2int8(p0[6 + 33] * scale2);
                pp[96 + 6] = float2int8(p0[6 + 48] * scale3);
                pp[96 + 7] = float2int8(p0[6 + 49] * scale3);
                pp[96 + 8] = float2int8(p0[6 + 64] * scale4);
                pp[96 + 9] = float2int8(p0[6 + 65] * scale4);
                pp[96 + 10] = float2int8(p0[6 + 80] * scale5);
                pp[96 + 11] = float2int8(p0[6 + 81] * scale5);
                pp[96 + 12] = float2int8(p0[6 + 96] * scale6);
                pp[96 + 13] = float2int8(p0[6 + 97] * scale6);
                pp[96 + 14] = float2int8(p0[6 + 112] * scale7);
                pp[96 + 15] = float2int8(p0[6 + 113] * scale7);

                pp[112 + 0] = float2int8(p0[6 + 128 + 0] * scale8);
                pp[112 + 1] = float2int8(p0[6 + 128 + 1] * scale8);
                pp[112 + 2] = float2int8(p0[6 + 128 + 16] * scale9);
                pp[112 + 3] = float2int8(p0[6 + 128 + 17] * scale9);
                pp[112 + 4] = float2int8(p0[6 + 128 + 32] * scalea);
                pp[112 + 5] = float2int8(p0[6 + 128 + 33] * scalea);
                pp[112 + 6] = float2int8(p0[6 + 128 + 48] * scaleb);
                pp[112 + 7] = float2int8(p0[6 + 128 + 49] * scaleb);
                pp[112 + 8] = float2int8(p0[6 + 128 + 64] * scalec);
                pp[112 + 9] = float2int8(p0[6 + 128 + 65] * scalec);
                pp[112 + 10] = float2int8(p0[6 + 128 + 80] * scaled);
                pp[112 + 11] = float2int8(p0[6 + 128 + 81] * scaled);
                pp[112 + 12] = float2int8(p0[6 + 128 + 96] * scalee);
                pp[112 + 13] = float2int8(p0[6 + 128 + 97] * scalee);
                pp[112 + 14] = float2int8(p0[6 + 128 + 112] * scalef);
                pp[112 + 15] = float2int8(p0[6 + 128 + 113] * scalef);

                pp[128 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[128 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[128 + 2] = float2int8(p0[8 + 16] * scale1);
                pp[128 + 3] = float2int8(p0[8 + 17] * scale1);
                pp[128 + 4] = float2int8(p0[8 + 32] * scale2);
                pp[128 + 5] = float2int8(p0[8 + 33] * scale2);
                pp[128 + 6] = float2int8(p0[8 + 48] * scale3);
                pp[128 + 7] = float2int8(p0[8 + 49] * scale3);
                pp[128 + 8] = float2int8(p0[8 + 64] * scale4);
                pp[128 + 9] = float2int8(p0[8 + 65] * scale4);
                pp[128 + 10] = float2int8(p0[8 + 80] * scale5);
                pp[128 + 11] = float2int8(p0[8 + 81] * scale5);
                pp[128 + 12] = float2int8(p0[8 + 96] * scale6);
                pp[128 + 13] = float2int8(p0[8 + 97] * scale6);
                pp[128 + 14] = float2int8(p0[8 + 112] * scale7);
                pp[128 + 15] = float2int8(p0[8 + 113] * scale7);

                pp[16 + 128 + 0] = float2int8(p0[8 + 128 + 0] * scale8);
                pp[16 + 128 + 1] = float2int8(p0[8 + 128 + 1] * scale8);
                pp[16 + 128 + 2] = float2int8(p0[8 + 128 + 16] * scale9);
                pp[16 + 128 + 3] = float2int8(p0[8 + 128 + 17] * scale9);
                pp[16 + 128 + 4] = float2int8(p0[8 + 128 + 32] * scalea);
                pp[16 + 128 + 5] = float2int8(p0[8 + 128 + 33] * scalea);
                pp[16 + 128 + 6] = float2int8(p0[8 + 128 + 48] * scaleb);
                pp[16 + 128 + 7] = float2int8(p0[8 + 128 + 49] * scaleb);
                pp[16 + 128 + 8] = float2int8(p0[8 + 128 + 64] * scalec);
                pp[16 + 128 + 9] = float2int8(p0[8 + 128 + 65] * scalec);
                pp[16 + 128 + 10] = float2int8(p0[8 + 128 + 80] * scaled);
                pp[16 + 128 + 11] = float2int8(p0[8 + 128 + 81] * scaled);
                pp[16 + 128 + 12] = float2int8(p0[8 + 128 + 96] * scalee);
                pp[16 + 128 + 13] = float2int8(p0[8 + 128 + 97] * scalee);
                pp[16 + 128 + 14] = float2int8(p0[8 + 128 + 112] * scalef);
                pp[16 + 128 + 15] = float2int8(p0[8 + 128 + 113] * scalef);

                pp[32 + 128 + 0] = float2int8(p0[10 + 0] * scale0);
                pp[32 + 128 + 1] = float2int8(p0[10 + 1] * scale0);
                pp[32 + 128 + 2] = float2int8(p0[10 + 16] * scale1);
                pp[32 + 128 + 3] = float2int8(p0[10 + 17] * scale1);
                pp[32 + 128 + 4] = float2int8(p0[10 + 32] * scale2);
                pp[32 + 128 + 5] = float2int8(p0[10 + 33] * scale2);
                pp[32 + 128 + 6] = float2int8(p0[10 + 48] * scale3);
                pp[32 + 128 + 7] = float2int8(p0[10 + 49] * scale3);
                pp[32 + 128 + 8] = float2int8(p0[10 + 64] * scale4);
                pp[32 + 128 + 9] = float2int8(p0[10 + 65] * scale4);
                pp[32 + 128 + 10] = float2int8(p0[10 + 80] * scale5);
                pp[32 + 128 + 11] = float2int8(p0[10 + 81] * scale5);
                pp[32 + 128 + 12] = float2int8(p0[10 + 96] * scale6);
                pp[32 + 128 + 13] = float2int8(p0[10 + 97] * scale6);
                pp[32 + 128 + 14] = float2int8(p0[10 + 112] * scale7);
                pp[32 + 128 + 15] = float2int8(p0[10 + 113] * scale7);

                pp[48 + 128 + 0] = float2int8(p0[10 + 128 + 0] * scale8);
                pp[48 + 128 + 1] = float2int8(p0[10 + 128 + 1] * scale8);
                pp[48 + 128 + 2] = float2int8(p0[10 + 128 + 16] * scale9);
                pp[48 + 128 + 3] = float2int8(p0[10 + 128 + 17] * scale9);
                pp[48 + 128 + 4] = float2int8(p0[10 + 128 + 32] * scalea);
                pp[48 + 128 + 5] = float2int8(p0[10 + 128 + 33] * scalea);
                pp[48 + 128 + 6] = float2int8(p0[10 + 128 + 48] * scaleb);
                pp[48 + 128 + 7] = float2int8(p0[10 + 128 + 49] * scaleb);
                pp[48 + 128 + 8] = float2int8(p0[10 + 128 + 64] * scalec);
                pp[48 + 128 + 9] = float2int8(p0[10 + 128 + 65] * scalec);
                pp[48 + 128 + 10] = float2int8(p0[10 + 128 + 80] * scaled);
                pp[48 + 128 + 11] = float2int8(p0[10 + 128 + 81] * scaled);
                pp[48 + 128 + 12] = float2int8(p0[10 + 128 + 96] * scalee);
                pp[48 + 128 + 13] = float2int8(p0[10 + 128 + 97] * scalee);
                pp[48 + 128 + 14] = float2int8(p0[10 + 128 + 112] * scalef);
                pp[48 + 128 + 15] = float2int8(p0[10 + 128 + 113] * scalef);

                pp[64 + 128 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[64 + 128 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[64 + 128 + 2] = float2int8(p0[12 + 16] * scale1);
                pp[64 + 128 + 3] = float2int8(p0[12 + 17] * scale1);
                pp[64 + 128 + 4] = float2int8(p0[12 + 32] * scale2);
                pp[64 + 128 + 5] = float2int8(p0[12 + 33] * scale2);
                pp[64 + 128 + 6] = float2int8(p0[12 + 48] * scale3);
                pp[64 + 128 + 7] = float2int8(p0[12 + 49] * scale3);
                pp[64 + 128 + 8] = float2int8(p0[12 + 64] * scale4);
                pp[64 + 128 + 9] = float2int8(p0[12 + 65] * scale4);
                pp[64 + 128 + 10] = float2int8(p0[12 + 80] * scale5);
                pp[64 + 128 + 11] = float2int8(p0[12 + 81] * scale5);
                pp[64 + 128 + 12] = float2int8(p0[12 + 96] * scale6);
                pp[64 + 128 + 13] = float2int8(p0[12 + 97] * scale6);
                pp[64 + 128 + 14] = float2int8(p0[12 + 112] * scale7);
                pp[64 + 128 + 15] = float2int8(p0[12 + 113] * scale7);

                pp[80 + 128 + 0] = float2int8(p0[12 + 128 + 0] * scale8);
                pp[80 + 128 + 1] = float2int8(p0[12 + 128 + 1] * scale8);
                pp[80 + 128 + 2] = float2int8(p0[12 + 128 + 16] * scale9);
                pp[80 + 128 + 3] = float2int8(p0[12 + 128 + 17] * scale9);
                pp[80 + 128 + 4] = float2int8(p0[12 + 128 + 32] * scalea);
                pp[80 + 128 + 5] = float2int8(p0[12 + 128 + 33] * scalea);
                pp[80 + 128 + 6] = float2int8(p0[12 + 128 + 48] * scaleb);
                pp[80 + 128 + 7] = float2int8(p0[12 + 128 + 49] * scaleb);
                pp[80 + 128 + 8] = float2int8(p0[12 + 128 + 64] * scalec);
                pp[80 + 128 + 9] = float2int8(p0[12 + 128 + 65] * scalec);
                pp[80 + 128 + 10] = float2int8(p0[12 + 128 + 80] * scaled);
                pp[80 + 128 + 11] = float2int8(p0[12 + 128 + 81] * scaled);
                pp[80 + 128 + 12] = float2int8(p0[12 + 128 + 96] * scalee);
                pp[80 + 128 + 13] = float2int8(p0[12 + 128 + 97] * scalee);
                pp[80 + 128 + 14] = float2int8(p0[12 + 128 + 112] * scalef);
                pp[80 + 128 + 15] = float2int8(p0[12 + 128 + 113] * scalef);

                pp[96 + 128 + 0] = float2int8(p0[14 + 0] * scale0);
                pp[96 + 128 + 1] = float2int8(p0[14 + 1] * scale0);
                pp[96 + 128 + 2] = float2int8(p0[14 + 16] * scale1);
                pp[96 + 128 + 3] = float2int8(p0[14 + 17] * scale1);
                pp[96 + 128 + 4] = float2int8(p0[14 + 32] * scale2);
                pp[96 + 128 + 5] = float2int8(p0[14 + 33] * scale2);
                pp[96 + 128 + 6] = float2int8(p0[14 + 48] * scale3);
                pp[96 + 128 + 7] = float2int8(p0[14 + 49] * scale3);
                pp[96 + 128 + 8] = float2int8(p0[14 + 64] * scale4);
                pp[96 + 128 + 9] = float2int8(p0[14 + 65] * scale4);
                pp[96 + 128 + 10] = float2int8(p0[14 + 80] * scale5);
                pp[96 + 128 + 11] = float2int8(p0[14 + 81] * scale5);
                pp[96 + 128 + 12] = float2int8(p0[14 + 96] * scale6);
                pp[96 + 128 + 13] = float2int8(p0[14 + 97] * scale6);
                pp[96 + 128 + 14] = float2int8(p0[14 + 112] * scale7);
                pp[96 + 128 + 15] = float2int8(p0[14 + 113] * scale7);

                pp[112 + 128 + 0] = float2int8(p0[14 + 128 + 0] * scale8);
                pp[112 + 128 + 1] = float2int8(p0[14 + 128 + 1] * scale8);
                pp[112 + 128 + 2] = float2int8(p0[14 + 128 + 16] * scale9);
                pp[112 + 128 + 3] = float2int8(p0[14 + 128 + 17] * scale9);
                pp[112 + 128 + 4] = float2int8(p0[14 + 128 + 32] * scalea);
                pp[112 + 128 + 5] = float2int8(p0[14 + 128 + 33] * scalea);
                pp[112 + 128 + 6] = float2int8(p0[14 + 128 + 48] * scaleb);
                pp[112 + 128 + 7] = float2int8(p0[14 + 128 + 49] * scaleb);
                pp[112 + 128 + 8] = float2int8(p0[14 + 128 + 64] * scalec);
                pp[112 + 128 + 9] = float2int8(p0[14 + 128 + 65] * scalec);
                pp[112 + 128 + 10] = float2int8(p0[14 + 128 + 80] * scaled);
                pp[112 + 128 + 11] = float2int8(p0[14 + 128 + 81] * scaled);
                pp[112 + 128 + 12] = float2int8(p0[14 + 128 + 96] * scalee);
                pp[112 + 128 + 13] = float2int8(p0[14 + 128 + 97] * scalee);
                pp[112 + 128 + 14] = float2int8(p0[14 + 128 + 112] * scalef);
                pp[112 + 128 + 15] = float2int8(p0[14 + 128 + 113] * scalef);

                pp += 256;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[8] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[16] * scale2);
                pp[9] = float2int8(p0[17] * scale2);
                pp[10] = float2int8(p0[18] * scale2);
                pp[11] = float2int8(p0[19] * scale2);
                pp[12] = float2int8(p0[24] * scale3);
                pp[13] = float2int8(p0[25] * scale3);
                pp[14] = float2int8(p0[26] * scale3);
                pp[15] = float2int8(p0[27] * scale3);
                pp[16] = float2int8(p0[32] * scale4);
                pp[17] = float2int8(p0[33] * scale4);
                pp[18] = float2int8(p0[34] * scale4);
                pp[19] = float2int8(p0[35] * scale4);
                pp[20] = float2int8(p0[40] * scale5);
                pp[21] = float2int8(p0[41] * scale5);
                pp[22] = float2int8(p0[42] * scale5);
                pp[23] = float2int8(p0[43] * scale5);
                pp[24] = float2int8(p0[48] * scale6);
                pp[25] = float2int8(p0[49] * scale6);
                pp[26] = float2int8(p0[50] * scale6);
                pp[27] = float2int8(p0[51] * scale6);
                pp[28] = float2int8(p0[56] * scale7);
                pp[29] = float2int8(p0[57] * scale7);
                pp[30] = float2int8(p0[58] * scale7);
                pp[31] = float2int8(p0[59] * scale7);

                pp[32 + 0] = float2int8(p0[64 + 0] * scale8);
                pp[32 + 1] = float2int8(p0[64 + 1] * scale8);
                pp[32 + 2] = float2int8(p0[64 + 2] * scale8);
                pp[32 + 3] = float2int8(p0[64 + 3] * scale8);
                pp[32 + 4] = float2int8(p0[64 + 8] * scale9);
                pp[32 + 5] = float2int8(p0[64 + 9] * scale9);
                pp[32 + 6] = float2int8(p0[64 + 10] * scale9);
                pp[32 + 7] = float2int8(p0[64 + 11] * scale9);
                pp[32 + 8] = float2int8(p0[64 + 16] * scalea);
                pp[32 + 9] = float2int8(p0[64 + 17] * scalea);
                pp[32 + 10] = float2int8(p0[64 + 18] * scalea);
                pp[32 + 11] = float2int8(p0[64 + 19] * scalea);
                pp[32 + 12] = float2int8(p0[64 + 24] * scaleb);
                pp[32 + 13] = float2int8(p0[64 + 25] * scaleb);
                pp[32 + 14] = float2int8(p0[64 + 26] * scaleb);
                pp[32 + 15] = float2int8(p0[64 + 27] * scaleb);
                pp[32 + 16] = float2int8(p0[64 + 32] * scalec);
                pp[32 + 17] = float2int8(p0[64 + 33] * scalec);
                pp[32 + 18] = float2int8(p0[64 + 34] * scalec);
                pp[32 + 19] = float2int8(p0[64 + 35] * scalec);
                pp[32 + 20] = float2int8(p0[64 + 40] * scaled);
                pp[32 + 21] = float2int8(p0[64 + 41] * scaled);
                pp[32 + 22] = float2int8(p0[64 + 42] * scaled);
                pp[32 + 23] = float2int8(p0[64 + 43] * scaled);
                pp[32 + 24] = float2int8(p0[64 + 48] * scalee);
                pp[32 + 25] = float2int8(p0[64 + 49] * scalee);
                pp[32 + 26] = float2int8(p0[64 + 50] * scalee);
                pp[32 + 27] = float2int8(p0[64 + 51] * scalee);
                pp[32 + 28] = float2int8(p0[64 + 56] * scalef);
                pp[32 + 29] = float2int8(p0[64 + 57] * scalef);
                pp[32 + 30] = float2int8(p0[64 + 58] * scalef);
                pp[32 + 31] = float2int8(p0[64 + 59] * scalef);

                pp[64 + 0] = float2int8(p0[4] * scale0);
                pp[64 + 1] = float2int8(p0[5] * scale0);
                pp[64 + 2] = float2int8(p0[6] * scale0);
                pp[64 + 3] = float2int8(p0[7] * scale0);
                pp[64 + 4] = float2int8(p0[12] * scale1);
                pp[64 + 5] = float2int8(p0[13] * scale1);
                pp[64 + 6] = float2int8(p0[14] * scale1);
                pp[64 + 7] = float2int8(p0[15] * scale1);
                pp[64 + 8] = float2int8(p0[20] * scale2);
                pp[64 + 9] = float2int8(p0[21] * scale2);
                pp[64 + 10] = float2int8(p0[22] * scale2);
                pp[64 + 11] = float2int8(p0[23] * scale2);
                pp[64 + 12] = float2int8(p0[28] * scale3);
                pp[64 + 13] = float2int8(p0[29] * scale3);
                pp[64 + 14] = float2int8(p0[30] * scale3);
                pp[64 + 15] = float2int8(p0[31] * scale3);
                pp[64 + 16] = float2int8(p0[36] * scale4);
                pp[64 + 17] = float2int8(p0[37] * scale4);
                pp[64 + 18] = float2int8(p0[38] * scale4);
                pp[64 + 19] = float2int8(p0[39] * scale4);
                pp[64 + 20] = float2int8(p0[44] * scale5);
                pp[64 + 21] = float2int8(p0[45] * scale5);
                pp[64 + 22] = float2int8(p0[46] * scale5);
                pp[64 + 23] = float2int8(p0[47] * scale5);
                pp[64 + 24] = float2int8(p0[52] * scale6);
                pp[64 + 25] = float2int8(p0[53] * scale6);
                pp[64 + 26] = float2int8(p0[54] * scale6);
                pp[64 + 27] = float2int8(p0[55] * scale6);
                pp[64 + 28] = float2int8(p0[60] * scale7);
                pp[64 + 29] = float2int8(p0[61] * scale7);
                pp[64 + 30] = float2int8(p0[62] * scale7);
                pp[64 + 31] = float2int8(p0[63] * scale7);

                pp[96 + 0] = float2int8(p0[64 + 4] * scale8);
                pp[96 + 1] = float2int8(p0[64 + 5] * scale8);
                pp[96 + 2] = float2int8(p0[64 + 6] * scale8);
                pp[96 + 3] = float2int8(p0[64 + 7] * scale8);
                pp[96 + 4] = float2int8(p0[64 + 12] * scale9);
                pp[96 + 5] = float2int8(p0[64 + 13] * scale9);
                pp[96 + 6] = float2int8(p0[64 + 14] * scale9);
                pp[96 + 7] = float2int8(p0[64 + 15] * scale9);
                pp[96 + 8] = float2int8(p0[64 + 20] * scalea);
                pp[96 + 9] = float2int8(p0[64 + 21] * scalea);
                pp[96 + 10] = float2int8(p0[64 + 22] * scalea);
                pp[96 + 11] = float2int8(p0[64 + 23] * scalea);
                pp[96 + 12] = float2int8(p0[64 + 28] * scaleb);
                pp[96 + 13] = float2int8(p0[64 + 29] * scaleb);
                pp[96 + 14] = float2int8(p0[64 + 30] * scaleb);
                pp[96 + 15] = float2int8(p0[64 + 31] * scaleb);
                pp[96 + 16] = float2int8(p0[64 + 36] * scalec);
                pp[96 + 17] = float2int8(p0[64 + 37] * scalec);
                pp[96 + 18] = float2int8(p0[64 + 38] * scalec);
                pp[96 + 19] = float2int8(p0[64 + 39] * scalec);
                pp[96 + 20] = float2int8(p0[64 + 44] * scaled);
                pp[96 + 21] = float2int8(p0[64 + 45] * scaled);
                pp[96 + 22] = float2int8(p0[64 + 46] * scaled);
                pp[96 + 23] = float2int8(p0[64 + 47] * scaled);
                pp[96 + 24] = float2int8(p0[64 + 52] * scalee);
                pp[96 + 25] = float2int8(p0[64 + 53] * scalee);
                pp[96 + 26] = float2int8(p0[64 + 54] * scalee);
                pp[96 + 27] = float2int8(p0[64 + 55] * scalee);
                pp[96 + 28] = float2int8(p0[64 + 60] * scalef);
                pp[96 + 29] = float2int8(p0[64 + 61] * scalef);
                pp[96 + 30] = float2int8(p0[64 + 62] * scalef);
                pp[96 + 31] = float2int8(p0[64 + 63] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                w_shift0 += pp[64 + 0];
                w_shift0 += pp[64 + 1];
                w_shift0 += pp[64 + 2];
                w_shift0 += pp[64 + 3];
                w_shift1 += pp[64 + 4];
                w_shift1 += pp[64 + 5];
                w_shift1 += pp[64 + 6];
                w_shift1 += pp[64 + 7];
                w_shift2 += pp[64 + 8];
                w_shift2 += pp[64 + 9];
                w_shift2 += pp[64 + 10];
                w_shift2 += pp[64 + 11];
                w_shift3 += pp[64 + 12];
                w_shift3 += pp[64 + 13];
                w_shift3 += pp[64 + 14];
                w_shift3 += pp[64 + 15];
                w_shift4 += pp[64 + 16];
                w_shift4 += pp[64 + 17];
                w_shift4 += pp[64 + 18];
                w_shift4 += pp[64 + 19];
                w_shift5 += pp[64 + 20];
                w_shift5 += pp[64 + 21];
                w_shift5 += pp[64 + 22];
                w_shift5 += pp[64 + 23];
                w_shift6 += pp[64 + 24];
                w_shift6 += pp[64 + 25];
                w_shift6 += pp[64 + 26];
                w_shift6 += pp[64 + 27];
                w_shift7 += pp[64 + 28];
                w_shift7 += pp[64 + 29];
                w_shift7 += pp[64 + 30];
                w_shift7 += pp[64 + 31];

                w_shift8 += pp[96 + 0];
                w_shift8 += pp[96 + 1];
                w_shift8 += pp[96 + 2];
                w_shift8 += pp[96 + 3];
                w_shift9 += pp[96 + 4];
                w_shift9 += pp[96 + 5];
                w_shift9 += pp[96 + 6];
                w_shift9 += pp[96 + 7];
                w_shifta += pp[96 + 8];
                w_shifta += pp[96 + 9];
                w_shifta += pp[96 + 10];
                w_shifta += pp[96 + 11];
                w_shiftb += pp[96 + 12];
                w_shiftb += pp[96 + 13];
                w_shiftb += pp[96 + 14];
                w_shiftb += pp[96 + 15];
                w_shiftc += pp[96 + 16];
                w_shiftc += pp[96 + 17];
                w_shiftc += pp[96 + 18];
                w_shiftc += pp[96 + 19];
                w_shiftd += pp[96 + 20];
                w_shiftd += pp[96 + 21];
                w_shiftd += pp[96 + 22];
                w_shiftd += pp[96 + 23];
                w_shifte += pp[96 + 24];
                w_shifte += pp[96 + 25];
                w_shifte += pp[96 + 26];
                w_shifte += pp[96 + 27];
                w_shiftf += pp[96 + 28];
                w_shiftf += pp[96 + 29];
                w_shiftf += pp[96 + 30];
                w_shiftf += pp[96 + 31];

                pp += 128;
                p0 += A_hstep * 8;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[16] * scale2);
                pp[5] = float2int8(p0[17] * scale2);
                pp[6] = float2int8(p0[24] * scale3);
                pp[7] = float2int8(p0[25] * scale3);
                pp[8] = float2int8(p0[32] * scale4);
                pp[9] = float2int8(p0[33] * scale4);
                pp[10] = float2int8(p0[40] * scale5);
                pp[11] = float2int8(p0[41] * scale5);
                pp[12] = float2int8(p0[48] * scale6);
                pp[13] = float2int8(p0[49] * scale6);
                pp[14] = float2int8(p0[56] * scale7);
                pp[15] = float2int8(p0[57] * scale7);

                pp[16 + 0] = float2int8(p0[64 + 0] * scale8);
                pp[16 + 1] = float2int8(p0[64 + 1] * scale8);
                pp[16 + 2] = float2int8(p0[64 + 8] * scale9);
                pp[16 + 3] = float2int8(p0[64 + 9] * scale9);
                pp[16 + 4] = float2int8(p0[64 + 16] * scalea);
                pp[16 + 5] = float2int8(p0[64 + 17] * scalea);
                pp[16 + 6] = float2int8(p0[64 + 24] * scaleb);
                pp[16 + 7] = float2int8(p0[64 + 25] * scaleb);
                pp[16 + 8] = float2int8(p0[64 + 32] * scalec);
                pp[16 + 9] = float2int8(p0[64 + 33] * scalec);
                pp[16 + 10] = float2int8(p0[64 + 40] * scaled);
                pp[16 + 11] = float2int8(p0[64 + 41] * scaled);
                pp[16 + 12] = float2int8(p0[64 + 48] * scalee);
                pp[16 + 13] = float2int8(p0[64 + 49] * scalee);
                pp[16 + 14] = float2int8(p0[64 + 56] * scalef);
                pp[16 + 15] = float2int8(p0[64 + 57] * scalef);

                pp[32 + 0] = float2int8(p0[2] * scale0);
                pp[32 + 1] = float2int8(p0[3] * scale0);
                pp[32 + 2] = float2int8(p0[10] * scale1);
                pp[32 + 3] = float2int8(p0[11] * scale1);
                pp[32 + 4] = float2int8(p0[18] * scale2);
                pp[32 + 5] = float2int8(p0[19] * scale2);
                pp[32 + 6] = float2int8(p0[26] * scale3);
                pp[32 + 7] = float2int8(p0[27] * scale3);
                pp[32 + 8] = float2int8(p0[34] * scale4);
                pp[32 + 9] = float2int8(p0[35] * scale4);
                pp[32 + 10] = float2int8(p0[42] * scale5);
                pp[32 + 11] = float2int8(p0[43] * scale5);
                pp[32 + 12] = float2int8(p0[50] * scale6);
                pp[32 + 13] = float2int8(p0[51] * scale6);
                pp[32 + 14] = float2int8(p0[58] * scale7);
                pp[32 + 15] = float2int8(p0[59] * scale7);

                pp[48 + 0] = float2int8(p0[64 + 2] * scale8);
                pp[48 + 1] = float2int8(p0[64 + 3] * scale8);
                pp[48 + 2] = float2int8(p0[64 + 10] * scale9);
                pp[48 + 3] = float2int8(p0[64 + 11] * scale9);
                pp[48 + 4] = float2int8(p0[64 + 18] * scalea);
                pp[48 + 5] = float2int8(p0[64 + 19] * scalea);
                pp[48 + 6] = float2int8(p0[64 + 26] * scaleb);
                pp[48 + 7] = float2int8(p0[64 + 27] * scaleb);
                pp[48 + 8] = float2int8(p0[64 + 34] * scalec);
                pp[48 + 9] = float2int8(p0[64 + 35] * scalec);
                pp[48 + 10] = float2int8(p0[64 + 42] * scaled);
                pp[48 + 11] = float2int8(p0[64 + 43] * scaled);
                pp[48 + 12] = float2int8(p0[64 + 50] * scalee);
                pp[48 + 13] = float2int8(p0[64 + 51] * scalee);
                pp[48 + 14] = float2int8(p0[64 + 58] * scalef);
                pp[48 + 15] = float2int8(p0[64 + 59] * scalef);

                pp[64 + 0] = float2int8(p0[4] * scale0);
                pp[64 + 1] = float2int8(p0[5] * scale0);
                pp[64 + 2] = float2int8(p0[12] * scale1);
                pp[64 + 3] = float2int8(p0[13] * scale1);
                pp[64 + 4] = float2int8(p0[20] * scale2);
                pp[64 + 5] = float2int8(p0[21] * scale2);
                pp[64 + 6] = float2int8(p0[28] * scale3);
                pp[64 + 7] = float2int8(p0[29] * scale3);
                pp[64 + 8] = float2int8(p0[36] * scale4);
                pp[64 + 9] = float2int8(p0[37] * scale4);
                pp[64 + 10] = float2int8(p0[44] * scale5);
                pp[64 + 11] = float2int8(p0[45] * scale5);
                pp[64 + 12] = float2int8(p0[52] * scale6);
                pp[64 + 13] = float2int8(p0[53] * scale6);
                pp[64 + 14] = float2int8(p0[60] * scale7);
                pp[64 + 15] = float2int8(p0[61] * scale7);

                pp[80 + 0] = float2int8(p0[64 + 4] * scale8);
                pp[80 + 1] = float2int8(p0[64 + 5] * scale8);
                pp[80 + 2] = float2int8(p0[64 + 12] * scale9);
                pp[80 + 3] = float2int8(p0[64 + 13] * scale9);
                pp[80 + 4] = float2int8(p0[64 + 20] * scalea);
                pp[80 + 5] = float2int8(p0[64 + 21] * scalea);
                pp[80 + 6] = float2int8(p0[64 + 28] * scaleb);
                pp[80 + 7] = float2int8(p0[64 + 29] * scaleb);
                pp[80 + 8] = float2int8(p0[64 + 36] * scalec);
                pp[80 + 9] = float2int8(p0[64 + 37] * scalec);
                pp[80 + 10] = float2int8(p0[64 + 44] * scaled);
                pp[80 + 11] = float2int8(p0[64 + 45] * scaled);
                pp[80 + 12] = float2int8(p0[64 + 52] * scalee);
                pp[80 + 13] = float2int8(p0[64 + 53] * scalee);
                pp[80 + 14] = float2int8(p0[64 + 60] * scalef);
                pp[80 + 15] = float2int8(p0[64 + 61] * scalef);

                pp[96 + 0] = float2int8(p0[6] * scale0);
                pp[96 + 1] = float2int8(p0[7] * scale0);
                pp[96 + 2] = float2int8(p0[14] * scale1);
                pp[96 + 3] = float2int8(p0[15] * scale1);
                pp[96 + 4] = float2int8(p0[22] * scale2);
                pp[96 + 5] = float2int8(p0[23] * scale2);
                pp[96 + 6] = float2int8(p0[30] * scale3);
                pp[96 + 7] = float2int8(p0[31] * scale3);
                pp[96 + 8] = float2int8(p0[38] * scale4);
                pp[96 + 9] = float2int8(p0[39] * scale4);
                pp[96 + 10] = float2int8(p0[46] * scale5);
                pp[96 + 11] = float2int8(p0[47] * scale5);
                pp[96 + 12] = float2int8(p0[54] * scale6);
                pp[96 + 13] = float2int8(p0[55] * scale6);
                pp[96 + 14] = float2int8(p0[62] * scale7);
                pp[96 + 15] = float2int8(p0[63] * scale7);

                pp[112 + 0] = float2int8(p0[64 + 6] * scale8);
                pp[112 + 1] = float2int8(p0[64 + 7] * scale8);
                pp[112 + 2] = float2int8(p0[64 + 14] * scale9);
                pp[112 + 3] = float2int8(p0[64 + 15] * scale9);
                pp[112 + 4] = float2int8(p0[64 + 22] * scalea);
                pp[112 + 5] = float2int8(p0[64 + 23] * scalea);
                pp[112 + 6] = float2int8(p0[64 + 30] * scaleb);
                pp[112 + 7] = float2int8(p0[64 + 31] * scaleb);
                pp[112 + 8] = float2int8(p0[64 + 38] * scalec);
                pp[112 + 9] = float2int8(p0[64 + 39] * scalec);
                pp[112 + 10] = float2int8(p0[64 + 46] * scaled);
                pp[112 + 11] = float2int8(p0[64 + 47] * scaled);
                pp[112 + 12] = float2int8(p0[64 + 54] * scalee);
                pp[112 + 13] = float2int8(p0[64 + 55] * scalee);
                pp[112 + 14] = float2int8(p0[64 + 62] * scalef);
                pp[112 + 15] = float2int8(p0[64 + 63] * scalef);

                pp += 128;
                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[4] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);
                pp[8] = float2int8(p0[8] * scale2);
                pp[9] = float2int8(p0[9] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[11] * scale2);
                pp[12] = float2int8(p0[12] * scale3);
                pp[13] = float2int8(p0[13] * scale3);
                pp[14] = float2int8(p0[14] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                pp[16] = float2int8(p0[16] * scale4);
                pp[17] = float2int8(p0[17] * scale4);
                pp[18] = float2int8(p0[18] * scale4);
                pp[19] = float2int8(p0[19] * scale4);
                pp[20] = float2int8(p0[20] * scale5);
                pp[21] = float2int8(p0[21] * scale5);
                pp[22] = float2int8(p0[22] * scale5);
                pp[23] = float2int8(p0[23] * scale5);
                pp[24] = float2int8(p0[24] * scale6);
                pp[25] = float2int8(p0[25] * scale6);
                pp[26] = float2int8(p0[26] * scale6);
                pp[27] = float2int8(p0[27] * scale6);
                pp[28] = float2int8(p0[28] * scale7);
                pp[29] = float2int8(p0[29] * scale7);
                pp[30] = float2int8(p0[30] * scale7);
                pp[31] = float2int8(p0[31] * scale7);

                pp[32 + 0] = float2int8(p0[32 + 0] * scale8);
                pp[32 + 1] = float2int8(p0[32 + 1] * scale8);
                pp[32 + 2] = float2int8(p0[32 + 2] * scale8);
                pp[32 + 3] = float2int8(p0[32 + 3] * scale8);
                pp[32 + 4] = float2int8(p0[32 + 4] * scale9);
                pp[32 + 5] = float2int8(p0[32 + 5] * scale9);
                pp[32 + 6] = float2int8(p0[32 + 6] * scale9);
                pp[32 + 7] = float2int8(p0[32 + 7] * scale9);
                pp[32 + 8] = float2int8(p0[32 + 8] * scalea);
                pp[32 + 9] = float2int8(p0[32 + 9] * scalea);
                pp[32 + 10] = float2int8(p0[32 + 10] * scalea);
                pp[32 + 11] = float2int8(p0[32 + 11] * scalea);
                pp[32 + 12] = float2int8(p0[32 + 12] * scaleb);
                pp[32 + 13] = float2int8(p0[32 + 13] * scaleb);
                pp[32 + 14] = float2int8(p0[32 + 14] * scaleb);
                pp[32 + 15] = float2int8(p0[32 + 15] * scaleb);
                pp[32 + 16] = float2int8(p0[32 + 16] * scalec);
                pp[32 + 17] = float2int8(p0[32 + 17] * scalec);
                pp[32 + 18] = float2int8(p0[32 + 18] * scalec);
                pp[32 + 19] = float2int8(p0[32 + 19] * scalec);
                pp[32 + 20] = float2int8(p0[32 + 20] * scaled);
                pp[32 + 21] = float2int8(p0[32 + 21] * scaled);
                pp[32 + 22] = float2int8(p0[32 + 22] * scaled);
                pp[32 + 23] = float2int8(p0[32 + 23] * scaled);
                pp[32 + 24] = float2int8(p0[32 + 24] * scalee);
                pp[32 + 25] = float2int8(p0[32 + 25] * scalee);
                pp[32 + 26] = float2int8(p0[32 + 26] * scalee);
                pp[32 + 27] = float2int8(p0[32 + 27] * scalee);
                pp[32 + 28] = float2int8(p0[32 + 28] * scalef);
                pp[32 + 29] = float2int8(p0[32 + 29] * scalef);
                pp[32 + 30] = float2int8(p0[32 + 30] * scalef);
                pp[32 + 31] = float2int8(p0[32 + 31] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[8] * scale2);
                pp[5] = float2int8(p0[9] * scale2);
                pp[6] = float2int8(p0[12] * scale3);
                pp[7] = float2int8(p0[13] * scale3);
                pp[8] = float2int8(p0[16] * scale4);
                pp[9] = float2int8(p0[17] * scale4);
                pp[10] = float2int8(p0[20] * scale5);
                pp[11] = float2int8(p0[21] * scale5);
                pp[12] = float2int8(p0[24] * scale6);
                pp[13] = float2int8(p0[25] * scale6);
                pp[14] = float2int8(p0[28] * scale7);
                pp[15] = float2int8(p0[29] * scale7);

                pp[16 + 0] = float2int8(p0[32 + 0] * scale8);
                pp[16 + 1] = float2int8(p0[32 + 1] * scale8);
                pp[16 + 2] = float2int8(p0[32 + 4] * scale9);
                pp[16 + 3] = float2int8(p0[32 + 5] * scale9);
                pp[16 + 4] = float2int8(p0[32 + 8] * scalea);
                pp[16 + 5] = float2int8(p0[32 + 9] * scalea);
                pp[16 + 6] = float2int8(p0[32 + 12] * scaleb);
                pp[16 + 7] = float2int8(p0[32 + 13] * scaleb);
                pp[16 + 8] = float2int8(p0[32 + 16] * scalec);
                pp[16 + 9] = float2int8(p0[32 + 17] * scalec);
                pp[16 + 10] = float2int8(p0[32 + 20] * scaled);
                pp[16 + 11] = float2int8(p0[32 + 21] * scaled);
                pp[16 + 12] = float2int8(p0[32 + 24] * scalee);
                pp[16 + 13] = float2int8(p0[32 + 25] * scalee);
                pp[16 + 14] = float2int8(p0[32 + 28] * scalef);
                pp[16 + 15] = float2int8(p0[32 + 29] * scalef);

                pp[32 + 0] = float2int8(p0[2] * scale0);
                pp[32 + 1] = float2int8(p0[3] * scale0);
                pp[32 + 2] = float2int8(p0[6] * scale1);
                pp[32 + 3] = float2int8(p0[7] * scale1);
                pp[32 + 4] = float2int8(p0[10] * scale2);
                pp[32 + 5] = float2int8(p0[11] * scale2);
                pp[32 + 6] = float2int8(p0[14] * scale3);
                pp[32 + 7] = float2int8(p0[15] * scale3);
                pp[32 + 8] = float2int8(p0[18] * scale4);
                pp[32 + 9] = float2int8(p0[19] * scale4);
                pp[32 + 10] = float2int8(p0[22] * scale5);
                pp[32 + 11] = float2int8(p0[23] * scale5);
                pp[32 + 12] = float2int8(p0[26] * scale6);
                pp[32 + 13] = float2int8(p0[27] * scale6);
                pp[32 + 14] = float2int8(p0[30] * scale7);
                pp[32 + 15] = float2int8(p0[31] * scale7);

                pp[48 + 0] = float2int8(p0[32 + 2] * scale8);
                pp[48 + 1] = float2int8(p0[32 + 3] * scale8);
                pp[48 + 2] = float2int8(p0[32 + 6] * scale9);
                pp[48 + 3] = float2int8(p0[32 + 7] * scale9);
                pp[48 + 4] = float2int8(p0[32 + 10] * scalea);
                pp[48 + 5] = float2int8(p0[32 + 11] * scalea);
                pp[48 + 6] = float2int8(p0[32 + 14] * scaleb);
                pp[48 + 7] = float2int8(p0[32 + 15] * scaleb);
                pp[48 + 8] = float2int8(p0[32 + 18] * scalec);
                pp[48 + 9] = float2int8(p0[32 + 19] * scalec);
                pp[48 + 10] = float2int8(p0[32 + 22] * scaled);
                pp[48 + 11] = float2int8(p0[32 + 23] * scaled);
                pp[48 + 12] = float2int8(p0[32 + 26] * scalee);
                pp[48 + 13] = float2int8(p0[32 + 27] * scalee);
                pp[48 + 14] = float2int8(p0[32 + 30] * scalef);
                pp[48 + 15] = float2int8(p0[32 + 31] * scalef);

                pp += 64;
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            int w_shift8 = 0;
            int w_shift9 = 0;
            int w_shifta = 0;
            int w_shiftb = 0;
            int w_shiftc = 0;
            int w_shiftd = 0;
            int w_shifte = 0;
            int w_shiftf = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[A_hstep * 2] * scale0);
                pp[3] = float2int8(p0[A_hstep * 3] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep * 2 + 1] * scale1);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[A_hstep + 2] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 3 + 2] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[A_hstep + 3] * scale3);
                pp[14] = float2int8(p0[A_hstep * 2 + 3] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                pp[16] = float2int8(p0[4] * scale4);
                pp[17] = float2int8(p0[A_hstep + 4] * scale4);
                pp[18] = float2int8(p0[A_hstep * 2 + 4] * scale4);
                pp[19] = float2int8(p0[A_hstep * 3 + 4] * scale4);
                pp[20] = float2int8(p0[5] * scale5);
                pp[21] = float2int8(p0[A_hstep + 5] * scale5);
                pp[22] = float2int8(p0[A_hstep * 2 + 5] * scale5);
                pp[23] = float2int8(p0[A_hstep * 3 + 5] * scale5);
                pp[24] = float2int8(p0[6] * scale6);
                pp[25] = float2int8(p0[A_hstep + 6] * scale6);
                pp[26] = float2int8(p0[A_hstep * 2 + 6] * scale6);
                pp[27] = float2int8(p0[A_hstep * 3 + 6] * scale6);
                pp[28] = float2int8(p0[7] * scale7);
                pp[29] = float2int8(p0[A_hstep + 7] * scale7);
                pp[30] = float2int8(p0[A_hstep * 2 + 7] * scale7);
                pp[31] = float2int8(p0[A_hstep * 3 + 7] * scale7);

                pp[32 + 0] = float2int8(p0[8] * scale8);
                pp[32 + 1] = float2int8(p0[A_hstep + 8] * scale8);
                pp[32 + 2] = float2int8(p0[A_hstep * 2 + 8] * scale8);
                pp[32 + 3] = float2int8(p0[A_hstep * 3 + 8] * scale8);
                pp[32 + 4] = float2int8(p0[9] * scale9);
                pp[32 + 5] = float2int8(p0[A_hstep + 9] * scale9);
                pp[32 + 6] = float2int8(p0[A_hstep * 2 + 9] * scale9);
                pp[32 + 7] = float2int8(p0[A_hstep * 3 + 9] * scale9);
                pp[32 + 8] = float2int8(p0[10] * scalea);
                pp[32 + 9] = float2int8(p0[A_hstep + 10] * scalea);
                pp[32 + 10] = float2int8(p0[A_hstep * 2 + 10] * scalea);
                pp[32 + 11] = float2int8(p0[A_hstep * 3 + 10] * scalea);
                pp[32 + 12] = float2int8(p0[11] * scaleb);
                pp[32 + 13] = float2int8(p0[A_hstep + 11] * scaleb);
                pp[32 + 14] = float2int8(p0[A_hstep * 2 + 11] * scaleb);
                pp[32 + 15] = float2int8(p0[A_hstep * 3 + 11] * scaleb);
                pp[32 + 16] = float2int8(p0[12] * scalec);
                pp[32 + 17] = float2int8(p0[A_hstep + 12] * scalec);
                pp[32 + 18] = float2int8(p0[A_hstep * 2 + 12] * scalec);
                pp[32 + 19] = float2int8(p0[A_hstep * 3 + 12] * scalec);
                pp[32 + 20] = float2int8(p0[13] * scaled);
                pp[32 + 21] = float2int8(p0[A_hstep + 13] * scaled);
                pp[32 + 22] = float2int8(p0[A_hstep * 2 + 13] * scaled);
                pp[32 + 23] = float2int8(p0[A_hstep * 3 + 13] * scaled);
                pp[32 + 24] = float2int8(p0[14] * scalee);
                pp[32 + 25] = float2int8(p0[A_hstep + 14] * scalee);
                pp[32 + 26] = float2int8(p0[A_hstep * 2 + 14] * scalee);
                pp[32 + 27] = float2int8(p0[A_hstep * 3 + 14] * scalee);
                pp[32 + 28] = float2int8(p0[15] * scalef);
                pp[32 + 29] = float2int8(p0[A_hstep + 15] * scalef);
                pp[32 + 30] = float2int8(p0[A_hstep * 2 + 15] * scalef);
                pp[32 + 31] = float2int8(p0[A_hstep * 3 + 15] * scalef);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift8 += pp[32 + 0];
                w_shift8 += pp[32 + 1];
                w_shift8 += pp[32 + 2];
                w_shift8 += pp[32 + 3];
                w_shift9 += pp[32 + 4];
                w_shift9 += pp[32 + 5];
                w_shift9 += pp[32 + 6];
                w_shift9 += pp[32 + 7];
                w_shifta += pp[32 + 8];
                w_shifta += pp[32 + 9];
                w_shifta += pp[32 + 10];
                w_shifta += pp[32 + 11];
                w_shiftb += pp[32 + 12];
                w_shiftb += pp[32 + 13];
                w_shiftb += pp[32 + 14];
                w_shiftb += pp[32 + 15];
                w_shiftc += pp[32 + 16];
                w_shiftc += pp[32 + 17];
                w_shiftc += pp[32 + 18];
                w_shiftc += pp[32 + 19];
                w_shiftd += pp[32 + 20];
                w_shiftd += pp[32 + 21];
                w_shiftd += pp[32 + 22];
                w_shiftd += pp[32 + 23];
                w_shifte += pp[32 + 24];
                w_shifte += pp[32 + 25];
                w_shifte += pp[32 + 26];
                w_shifte += pp[32 + 27];
                w_shiftf += pp[32 + 28];
                w_shiftf += pp[32 + 29];
                w_shiftf += pp[32 + 30];
                w_shiftf += pp[32 + 31];

                pp += 64;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                ((int*)pp)[8] = w_shift8 * 127;
                ((int*)pp)[9] = w_shift9 * 127;
                ((int*)pp)[10] = w_shifta * 127;
                ((int*)pp)[11] = w_shiftb * 127;
                ((int*)pp)[12] = w_shiftc * 127;
                ((int*)pp)[13] = w_shiftd * 127;
                ((int*)pp)[14] = w_shifte * 127;
                ((int*)pp)[15] = w_shiftf * 127;
                pp += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[A_hstep + 2] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[A_hstep + 3] * scale3);
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[A_hstep + 4] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[A_hstep + 5] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[A_hstep + 6] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[A_hstep + 7] * scale7);

                pp[16 + 0] = float2int8(p0[8] * scale8);
                pp[16 + 1] = float2int8(p0[A_hstep + 8] * scale8);
                pp[16 + 2] = float2int8(p0[9] * scale9);
                pp[16 + 3] = float2int8(p0[A_hstep + 9] * scale9);
                pp[16 + 4] = float2int8(p0[10] * scalea);
                pp[16 + 5] = float2int8(p0[A_hstep + 10] * scalea);
                pp[16 + 6] = float2int8(p0[11] * scaleb);
                pp[16 + 7] = float2int8(p0[A_hstep + 11] * scaleb);
                pp[16 + 8] = float2int8(p0[12] * scalec);
                pp[16 + 9] = float2int8(p0[A_hstep + 12] * scalec);
                pp[16 + 10] = float2int8(p0[13] * scaled);
                pp[16 + 11] = float2int8(p0[A_hstep + 13] * scaled);
                pp[16 + 12] = float2int8(p0[14] * scalee);
                pp[16 + 13] = float2int8(p0[A_hstep + 14] * scalee);
                pp[16 + 14] = float2int8(p0[15] * scalef);
                pp[16 + 15] = float2int8(p0[A_hstep + 15] * scalef);
                pp += 32;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp[8] = float2int8(p0[8] * scale8);
                pp[9] = float2int8(p0[9] * scale9);
                pp[10] = float2int8(p0[10] * scalea);
                pp[11] = float2int8(p0[11] * scaleb);
                pp[12] = float2int8(p0[12] * scalec);
                pp[13] = float2int8(p0[13] * scaled);
                pp[14] = float2int8(p0[14] * scalee);
                pp[15] = float2int8(p0[15] * scalef);
                pp += 16;
                p0 += A_hstep;
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    signed char* pp1 = pp + max_kk * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];

#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2 + 0] * scale0);
                pp[3] = float2int8(p0[2 + 1] * scale0);
                pp[4] = float2int8(p0[16] * scale1);
                pp[5] = float2int8(p0[17] * scale1);
                pp[6] = float2int8(p0[2 + 16] * scale1);
                pp[7] = float2int8(p0[2 + 17] * scale1);
                pp[8] = float2int8(p0[32] * scale2);
                pp[9] = float2int8(p0[33] * scale2);
                pp[10] = float2int8(p0[2 + 32] * scale2);
                pp[11] = float2int8(p0[2 + 33] * scale2);
                pp[12] = float2int8(p0[48] * scale3);
                pp[13] = float2int8(p0[49] * scale3);
                pp[14] = float2int8(p0[2 + 48] * scale3);
                pp[15] = float2int8(p0[2 + 49] * scale3);
                pp[16] = float2int8(p0[64] * scale4);
                pp[17] = float2int8(p0[65] * scale4);
                pp[18] = float2int8(p0[2 + 64] * scale4);
                pp[19] = float2int8(p0[2 + 65] * scale4);
                pp[20] = float2int8(p0[80] * scale5);
                pp[21] = float2int8(p0[81] * scale5);
                pp[22] = float2int8(p0[2 + 80] * scale5);
                pp[23] = float2int8(p0[2 + 81] * scale5);
                pp[24] = float2int8(p0[96] * scale6);
                pp[25] = float2int8(p0[97] * scale6);
                pp[26] = float2int8(p0[2 + 96] * scale6);
                pp[27] = float2int8(p0[2 + 97] * scale6);
                pp[28] = float2int8(p0[112] * scale7);
                pp[29] = float2int8(p0[113] * scale7);
                pp[30] = float2int8(p0[2 + 112] * scale7);
                pp[31] = float2int8(p0[2 + 113] * scale7);

                pp[32 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[32 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[32 + 2] = float2int8(p0[6 + 0] * scale0);
                pp[32 + 3] = float2int8(p0[6 + 1] * scale0);
                pp[32 + 4] = float2int8(p0[4 + 16] * scale1);
                pp[32 + 5] = float2int8(p0[4 + 17] * scale1);
                pp[32 + 6] = float2int8(p0[6 + 16] * scale1);
                pp[32 + 7] = float2int8(p0[6 + 17] * scale1);
                pp[32 + 8] = float2int8(p0[4 + 32] * scale2);
                pp[32 + 9] = float2int8(p0[4 + 33] * scale2);
                pp[32 + 10] = float2int8(p0[6 + 32] * scale2);
                pp[32 + 11] = float2int8(p0[6 + 33] * scale2);
                pp[32 + 12] = float2int8(p0[4 + 48] * scale3);
                pp[32 + 13] = float2int8(p0[4 + 49] * scale3);
                pp[32 + 14] = float2int8(p0[6 + 48] * scale3);
                pp[32 + 15] = float2int8(p0[6 + 49] * scale3);
                pp[32 + 16] = float2int8(p0[4 + 64] * scale4);
                pp[32 + 17] = float2int8(p0[4 + 65] * scale4);
                pp[32 + 18] = float2int8(p0[6 + 64] * scale4);
                pp[32 + 19] = float2int8(p0[6 + 65] * scale4);
                pp[32 + 20] = float2int8(p0[4 + 80] * scale5);
                pp[32 + 21] = float2int8(p0[4 + 81] * scale5);
                pp[32 + 22] = float2int8(p0[6 + 80] * scale5);
                pp[32 + 23] = float2int8(p0[6 + 81] * scale5);
                pp[32 + 24] = float2int8(p0[4 + 96] * scale6);
                pp[32 + 25] = float2int8(p0[4 + 97] * scale6);
                pp[32 + 26] = float2int8(p0[6 + 96] * scale6);
                pp[32 + 27] = float2int8(p0[6 + 97] * scale6);
                pp[32 + 28] = float2int8(p0[4 + 112] * scale7);
                pp[32 + 29] = float2int8(p0[4 + 113] * scale7);
                pp[32 + 30] = float2int8(p0[6 + 112] * scale7);
                pp[32 + 31] = float2int8(p0[6 + 113] * scale7);

                pp[64 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[64 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[64 + 2] = float2int8(p0[10 + 0] * scale0);
                pp[64 + 3] = float2int8(p0[10 + 1] * scale0);
                pp[64 + 4] = float2int8(p0[8 + 16] * scale1);
                pp[64 + 5] = float2int8(p0[8 + 17] * scale1);
                pp[64 + 6] = float2int8(p0[10 + 16] * scale1);
                pp[64 + 7] = float2int8(p0[10 + 17] * scale1);
                pp[64 + 8] = float2int8(p0[8 + 32] * scale2);
                pp[64 + 9] = float2int8(p0[8 + 33] * scale2);
                pp[64 + 10] = float2int8(p0[10 + 32] * scale2);
                pp[64 + 11] = float2int8(p0[10 + 33] * scale2);
                pp[64 + 12] = float2int8(p0[8 + 48] * scale3);
                pp[64 + 13] = float2int8(p0[8 + 49] * scale3);
                pp[64 + 14] = float2int8(p0[10 + 48] * scale3);
                pp[64 + 15] = float2int8(p0[10 + 49] * scale3);
                pp[64 + 16] = float2int8(p0[8 + 64] * scale4);
                pp[64 + 17] = float2int8(p0[8 + 65] * scale4);
                pp[64 + 18] = float2int8(p0[10 + 64] * scale4);
                pp[64 + 19] = float2int8(p0[10 + 65] * scale4);
                pp[64 + 20] = float2int8(p0[8 + 80] * scale5);
                pp[64 + 21] = float2int8(p0[8 + 81] * scale5);
                pp[64 + 22] = float2int8(p0[10 + 80] * scale5);
                pp[64 + 23] = float2int8(p0[10 + 81] * scale5);
                pp[64 + 24] = float2int8(p0[8 + 96] * scale6);
                pp[64 + 25] = float2int8(p0[8 + 97] * scale6);
                pp[64 + 26] = float2int8(p0[10 + 96] * scale6);
                pp[64 + 27] = float2int8(p0[10 + 97] * scale6);
                pp[64 + 28] = float2int8(p0[8 + 112] * scale7);
                pp[64 + 29] = float2int8(p0[8 + 113] * scale7);
                pp[64 + 30] = float2int8(p0[10 + 112] * scale7);
                pp[64 + 31] = float2int8(p0[10 + 113] * scale7);

                pp[96 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[96 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[96 + 2] = float2int8(p0[14 + 0] * scale0);
                pp[96 + 3] = float2int8(p0[14 + 1] * scale0);
                pp[96 + 4] = float2int8(p0[12 + 16] * scale1);
                pp[96 + 5] = float2int8(p0[12 + 17] * scale1);
                pp[96 + 6] = float2int8(p0[14 + 16] * scale1);
                pp[96 + 7] = float2int8(p0[14 + 17] * scale1);
                pp[96 + 8] = float2int8(p0[12 + 32] * scale2);
                pp[96 + 9] = float2int8(p0[12 + 33] * scale2);
                pp[96 + 10] = float2int8(p0[14 + 32] * scale2);
                pp[96 + 11] = float2int8(p0[14 + 33] * scale2);
                pp[96 + 12] = float2int8(p0[12 + 48] * scale3);
                pp[96 + 13] = float2int8(p0[12 + 49] * scale3);
                pp[96 + 14] = float2int8(p0[14 + 48] * scale3);
                pp[96 + 15] = float2int8(p0[14 + 49] * scale3);
                pp[96 + 16] = float2int8(p0[12 + 64] * scale4);
                pp[96 + 17] = float2int8(p0[12 + 65] * scale4);
                pp[96 + 18] = float2int8(p0[14 + 64] * scale4);
                pp[96 + 19] = float2int8(p0[14 + 65] * scale4);
                pp[96 + 20] = float2int8(p0[12 + 80] * scale5);
                pp[96 + 21] = float2int8(p0[12 + 81] * scale5);
                pp[96 + 22] = float2int8(p0[14 + 80] * scale5);
                pp[96 + 23] = float2int8(p0[14 + 81] * scale5);
                pp[96 + 24] = float2int8(p0[12 + 96] * scale6);
                pp[96 + 25] = float2int8(p0[12 + 97] * scale6);
                pp[96 + 26] = float2int8(p0[14 + 96] * scale6);
                pp[96 + 27] = float2int8(p0[14 + 97] * scale6);
                pp[96 + 28] = float2int8(p0[12 + 112] * scale7);
                pp[96 + 29] = float2int8(p0[12 + 113] * scale7);
                pp[96 + 30] = float2int8(p0[14 + 112] * scale7);
                pp[96 + 31] = float2int8(p0[14 + 113] * scale7);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift0 += pp[32 + 0];
                w_shift0 += pp[32 + 1];
                w_shift0 += pp[32 + 2];
                w_shift0 += pp[32 + 3];
                w_shift1 += pp[32 + 4];
                w_shift1 += pp[32 + 5];
                w_shift1 += pp[32 + 6];
                w_shift1 += pp[32 + 7];
                w_shift2 += pp[32 + 8];
                w_shift2 += pp[32 + 9];
                w_shift2 += pp[32 + 10];
                w_shift2 += pp[32 + 11];
                w_shift3 += pp[32 + 12];
                w_shift3 += pp[32 + 13];
                w_shift3 += pp[32 + 14];
                w_shift3 += pp[32 + 15];
                w_shift4 += pp[32 + 16];
                w_shift4 += pp[32 + 17];
                w_shift4 += pp[32 + 18];
                w_shift4 += pp[32 + 19];
                w_shift5 += pp[32 + 20];
                w_shift5 += pp[32 + 21];
                w_shift5 += pp[32 + 22];
                w_shift5 += pp[32 + 23];
                w_shift6 += pp[32 + 24];
                w_shift6 += pp[32 + 25];
                w_shift6 += pp[32 + 26];
                w_shift6 += pp[32 + 27];
                w_shift7 += pp[32 + 28];
                w_shift7 += pp[32 + 29];
                w_shift7 += pp[32 + 30];
                w_shift7 += pp[32 + 31];

                w_shift0 += pp[64 + 0];
                w_shift0 += pp[64 + 1];
                w_shift0 += pp[64 + 2];
                w_shift0 += pp[64 + 3];
                w_shift1 += pp[64 + 4];
                w_shift1 += pp[64 + 5];
                w_shift1 += pp[64 + 6];
                w_shift1 += pp[64 + 7];
                w_shift2 += pp[64 + 8];
                w_shift2 += pp[64 + 9];
                w_shift2 += pp[64 + 10];
                w_shift2 += pp[64 + 11];
                w_shift3 += pp[64 + 12];
                w_shift3 += pp[64 + 13];
                w_shift3 += pp[64 + 14];
                w_shift3 += pp[64 + 15];
                w_shift4 += pp[64 + 16];
                w_shift4 += pp[64 + 17];
                w_shift4 += pp[64 + 18];
                w_shift4 += pp[64 + 19];
                w_shift5 += pp[64 + 20];
                w_shift5 += pp[64 + 21];
                w_shift5 += pp[64 + 22];
                w_shift5 += pp[64 + 23];
                w_shift6 += pp[64 + 24];
                w_shift6 += pp[64 + 25];
                w_shift6 += pp[64 + 26];
                w_shift6 += pp[64 + 27];
                w_shift7 += pp[64 + 28];
                w_shift7 += pp[64 + 29];
                w_shift7 += pp[64 + 30];
                w_shift7 += pp[64 + 31];

                w_shift0 += pp[96 + 0];
                w_shift0 += pp[96 + 1];
                w_shift0 += pp[96 + 2];
                w_shift0 += pp[96 + 3];
                w_shift1 += pp[96 + 4];
                w_shift1 += pp[96 + 5];
                w_shift1 += pp[96 + 6];
                w_shift1 += pp[96 + 7];
                w_shift2 += pp[96 + 8];
                w_shift2 += pp[96 + 9];
                w_shift2 += pp[96 + 10];
                w_shift2 += pp[96 + 11];
                w_shift3 += pp[96 + 12];
                w_shift3 += pp[96 + 13];
                w_shift3 += pp[96 + 14];
                w_shift3 += pp[96 + 15];
                w_shift4 += pp[96 + 16];
                w_shift4 += pp[96 + 17];
                w_shift4 += pp[96 + 18];
                w_shift4 += pp[96 + 19];
                w_shift5 += pp[96 + 20];
                w_shift5 += pp[96 + 21];
                w_shift5 += pp[96 + 22];
                w_shift5 += pp[96 + 23];
                w_shift6 += pp[96 + 24];
                w_shift6 += pp[96 + 25];
                w_shift6 += pp[96 + 26];
                w_shift6 += pp[96 + 27];
                w_shift7 += pp[96 + 28];
                w_shift7 += pp[96 + 29];
                w_shift7 += pp[96 + 30];
                w_shift7 += pp[96 + 31];
                pp += 128;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[16] * scale1);
                pp[3] = float2int8(p0[17] * scale1);
                pp[4] = float2int8(p0[32] * scale2);
                pp[5] = float2int8(p0[33] * scale2);
                pp[6] = float2int8(p0[48] * scale3);
                pp[7] = float2int8(p0[49] * scale3);
                pp[8] = float2int8(p0[64] * scale4);
                pp[9] = float2int8(p0[65] * scale4);
                pp[10] = float2int8(p0[80] * scale5);
                pp[11] = float2int8(p0[81] * scale5);
                pp[12] = float2int8(p0[96] * scale6);
                pp[13] = float2int8(p0[97] * scale6);
                pp[14] = float2int8(p0[112] * scale7);
                pp[15] = float2int8(p0[113] * scale7);

                pp[16 + 0] = float2int8(p0[2 + 0] * scale0);
                pp[16 + 1] = float2int8(p0[2 + 1] * scale0);
                pp[16 + 2] = float2int8(p0[2 + 16] * scale1);
                pp[16 + 3] = float2int8(p0[2 + 17] * scale1);
                pp[16 + 4] = float2int8(p0[2 + 32] * scale2);
                pp[16 + 5] = float2int8(p0[2 + 33] * scale2);
                pp[16 + 6] = float2int8(p0[2 + 48] * scale3);
                pp[16 + 7] = float2int8(p0[2 + 49] * scale3);
                pp[16 + 8] = float2int8(p0[2 + 64] * scale4);
                pp[16 + 9] = float2int8(p0[2 + 65] * scale4);
                pp[16 + 10] = float2int8(p0[2 + 80] * scale5);
                pp[16 + 11] = float2int8(p0[2 + 81] * scale5);
                pp[16 + 12] = float2int8(p0[2 + 96] * scale6);
                pp[16 + 13] = float2int8(p0[2 + 97] * scale6);
                pp[16 + 14] = float2int8(p0[2 + 112] * scale7);
                pp[16 + 15] = float2int8(p0[2 + 113] * scale7);

                pp[32 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[32 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[32 + 2] = float2int8(p0[4 + 16] * scale1);
                pp[32 + 3] = float2int8(p0[4 + 17] * scale1);
                pp[32 + 4] = float2int8(p0[4 + 32] * scale2);
                pp[32 + 5] = float2int8(p0[4 + 33] * scale2);
                pp[32 + 6] = float2int8(p0[4 + 48] * scale3);
                pp[32 + 7] = float2int8(p0[4 + 49] * scale3);
                pp[32 + 8] = float2int8(p0[4 + 64] * scale4);
                pp[32 + 9] = float2int8(p0[4 + 65] * scale4);
                pp[32 + 10] = float2int8(p0[4 + 80] * scale5);
                pp[32 + 11] = float2int8(p0[4 + 81] * scale5);
                pp[32 + 12] = float2int8(p0[4 + 96] * scale6);
                pp[32 + 13] = float2int8(p0[4 + 97] * scale6);
                pp[32 + 14] = float2int8(p0[4 + 112] * scale7);
                pp[32 + 15] = float2int8(p0[4 + 113] * scale7);

                pp[48 + 0] = float2int8(p0[6 + 0] * scale0);
                pp[48 + 1] = float2int8(p0[6 + 1] * scale0);
                pp[48 + 2] = float2int8(p0[6 + 16] * scale1);
                pp[48 + 3] = float2int8(p0[6 + 17] * scale1);
                pp[48 + 4] = float2int8(p0[6 + 32] * scale2);
                pp[48 + 5] = float2int8(p0[6 + 33] * scale2);
                pp[48 + 6] = float2int8(p0[6 + 48] * scale3);
                pp[48 + 7] = float2int8(p0[6 + 49] * scale3);
                pp[48 + 8] = float2int8(p0[6 + 64] * scale4);
                pp[48 + 9] = float2int8(p0[6 + 65] * scale4);
                pp[48 + 10] = float2int8(p0[6 + 80] * scale5);
                pp[48 + 11] = float2int8(p0[6 + 81] * scale5);
                pp[48 + 12] = float2int8(p0[6 + 96] * scale6);
                pp[48 + 13] = float2int8(p0[6 + 97] * scale6);
                pp[48 + 14] = float2int8(p0[6 + 112] * scale7);
                pp[48 + 15] = float2int8(p0[6 + 113] * scale7);

                pp[64 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[64 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[64 + 2] = float2int8(p0[8 + 16] * scale1);
                pp[64 + 3] = float2int8(p0[8 + 17] * scale1);
                pp[64 + 4] = float2int8(p0[8 + 32] * scale2);
                pp[64 + 5] = float2int8(p0[8 + 33] * scale2);
                pp[64 + 6] = float2int8(p0[8 + 48] * scale3);
                pp[64 + 7] = float2int8(p0[8 + 49] * scale3);
                pp[64 + 8] = float2int8(p0[8 + 64] * scale4);
                pp[64 + 9] = float2int8(p0[8 + 65] * scale4);
                pp[64 + 10] = float2int8(p0[8 + 80] * scale5);
                pp[64 + 11] = float2int8(p0[8 + 81] * scale5);
                pp[64 + 12] = float2int8(p0[8 + 96] * scale6);
                pp[64 + 13] = float2int8(p0[8 + 97] * scale6);
                pp[64 + 14] = float2int8(p0[8 + 112] * scale7);
                pp[64 + 15] = float2int8(p0[8 + 113] * scale7);

                pp[80 + 0] = float2int8(p0[10 + 0] * scale0);
                pp[80 + 1] = float2int8(p0[10 + 1] * scale0);
                pp[80 + 2] = float2int8(p0[10 + 16] * scale1);
                pp[80 + 3] = float2int8(p0[10 + 17] * scale1);
                pp[80 + 4] = float2int8(p0[10 + 32] * scale2);
                pp[80 + 5] = float2int8(p0[10 + 33] * scale2);
                pp[80 + 6] = float2int8(p0[10 + 48] * scale3);
                pp[80 + 7] = float2int8(p0[10 + 49] * scale3);
                pp[80 + 8] = float2int8(p0[10 + 64] * scale4);
                pp[80 + 9] = float2int8(p0[10 + 65] * scale4);
                pp[80 + 10] = float2int8(p0[10 + 80] * scale5);
                pp[80 + 11] = float2int8(p0[10 + 81] * scale5);
                pp[80 + 12] = float2int8(p0[10 + 96] * scale6);
                pp[80 + 13] = float2int8(p0[10 + 97] * scale6);
                pp[80 + 14] = float2int8(p0[10 + 112] * scale7);
                pp[80 + 15] = float2int8(p0[10 + 113] * scale7);

                pp[96 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[96 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[96 + 2] = float2int8(p0[12 + 16] * scale1);
                pp[96 + 3] = float2int8(p0[12 + 17] * scale1);
                pp[96 + 4] = float2int8(p0[12 + 32] * scale2);
                pp[96 + 5] = float2int8(p0[12 + 33] * scale2);
                pp[96 + 6] = float2int8(p0[12 + 48] * scale3);
                pp[96 + 7] = float2int8(p0[12 + 49] * scale3);
                pp[96 + 8] = float2int8(p0[12 + 64] * scale4);
                pp[96 + 9] = float2int8(p0[12 + 65] * scale4);
                pp[96 + 10] = float2int8(p0[12 + 80] * scale5);
                pp[96 + 11] = float2int8(p0[12 + 81] * scale5);
                pp[96 + 12] = float2int8(p0[12 + 96] * scale6);
                pp[96 + 13] = float2int8(p0[12 + 97] * scale6);
                pp[96 + 14] = float2int8(p0[12 + 112] * scale7);
                pp[96 + 15] = float2int8(p0[12 + 113] * scale7);

                pp[112 + 0] = float2int8(p0[14 + 0] * scale0);
                pp[112 + 1] = float2int8(p0[14 + 1] * scale0);
                pp[112 + 2] = float2int8(p0[14 + 16] * scale1);
                pp[112 + 3] = float2int8(p0[14 + 17] * scale1);
                pp[112 + 4] = float2int8(p0[14 + 32] * scale2);
                pp[112 + 5] = float2int8(p0[14 + 33] * scale2);
                pp[112 + 6] = float2int8(p0[14 + 48] * scale3);
                pp[112 + 7] = float2int8(p0[14 + 49] * scale3);
                pp[112 + 8] = float2int8(p0[14 + 64] * scale4);
                pp[112 + 9] = float2int8(p0[14 + 65] * scale4);
                pp[112 + 10] = float2int8(p0[14 + 80] * scale5);
                pp[112 + 11] = float2int8(p0[14 + 81] * scale5);
                pp[112 + 12] = float2int8(p0[14 + 96] * scale6);
                pp[112 + 13] = float2int8(p0[14 + 97] * scale6);
                pp[112 + 14] = float2int8(p0[14 + 112] * scale7);
                pp[112 + 15] = float2int8(p0[14 + 113] * scale7);

                pp += 128;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[8] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[16] * scale2);
                pp[9] = float2int8(p0[17] * scale2);
                pp[10] = float2int8(p0[18] * scale2);
                pp[11] = float2int8(p0[19] * scale2);
                pp[12] = float2int8(p0[24] * scale3);
                pp[13] = float2int8(p0[25] * scale3);
                pp[14] = float2int8(p0[26] * scale3);
                pp[15] = float2int8(p0[27] * scale3);
                pp[16] = float2int8(p0[32] * scale4);
                pp[17] = float2int8(p0[33] * scale4);
                pp[18] = float2int8(p0[34] * scale4);
                pp[19] = float2int8(p0[35] * scale4);
                pp[20] = float2int8(p0[40] * scale5);
                pp[21] = float2int8(p0[41] * scale5);
                pp[22] = float2int8(p0[42] * scale5);
                pp[23] = float2int8(p0[43] * scale5);
                pp[24] = float2int8(p0[48] * scale6);
                pp[25] = float2int8(p0[49] * scale6);
                pp[26] = float2int8(p0[50] * scale6);
                pp[27] = float2int8(p0[51] * scale6);
                pp[28] = float2int8(p0[56] * scale7);
                pp[29] = float2int8(p0[57] * scale7);
                pp[30] = float2int8(p0[58] * scale7);
                pp[31] = float2int8(p0[59] * scale7);

                pp[32 + 0] = float2int8(p0[4] * scale0);
                pp[32 + 1] = float2int8(p0[5] * scale0);
                pp[32 + 2] = float2int8(p0[6] * scale0);
                pp[32 + 3] = float2int8(p0[7] * scale0);
                pp[32 + 4] = float2int8(p0[12] * scale1);
                pp[32 + 5] = float2int8(p0[13] * scale1);
                pp[32 + 6] = float2int8(p0[14] * scale1);
                pp[32 + 7] = float2int8(p0[15] * scale1);
                pp[32 + 8] = float2int8(p0[20] * scale2);
                pp[32 + 9] = float2int8(p0[21] * scale2);
                pp[32 + 10] = float2int8(p0[22] * scale2);
                pp[32 + 11] = float2int8(p0[23] * scale2);
                pp[32 + 12] = float2int8(p0[28] * scale3);
                pp[32 + 13] = float2int8(p0[29] * scale3);
                pp[32 + 14] = float2int8(p0[30] * scale3);
                pp[32 + 15] = float2int8(p0[31] * scale3);
                pp[32 + 16] = float2int8(p0[36] * scale4);
                pp[32 + 17] = float2int8(p0[37] * scale4);
                pp[32 + 18] = float2int8(p0[38] * scale4);
                pp[32 + 19] = float2int8(p0[39] * scale4);
                pp[32 + 20] = float2int8(p0[44] * scale5);
                pp[32 + 21] = float2int8(p0[45] * scale5);
                pp[32 + 22] = float2int8(p0[46] * scale5);
                pp[32 + 23] = float2int8(p0[47] * scale5);
                pp[32 + 24] = float2int8(p0[52] * scale6);
                pp[32 + 25] = float2int8(p0[53] * scale6);
                pp[32 + 26] = float2int8(p0[54] * scale6);
                pp[32 + 27] = float2int8(p0[55] * scale6);
                pp[32 + 28] = float2int8(p0[60] * scale7);
                pp[32 + 29] = float2int8(p0[61] * scale7);
                pp[32 + 30] = float2int8(p0[62] * scale7);
                pp[32 + 31] = float2int8(p0[63] * scale7);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];

                w_shift0 += pp[32 + 0];
                w_shift0 += pp[32 + 1];
                w_shift0 += pp[32 + 2];
                w_shift0 += pp[32 + 3];
                w_shift1 += pp[32 + 4];
                w_shift1 += pp[32 + 5];
                w_shift1 += pp[32 + 6];
                w_shift1 += pp[32 + 7];
                w_shift2 += pp[32 + 8];
                w_shift2 += pp[32 + 9];
                w_shift2 += pp[32 + 10];
                w_shift2 += pp[32 + 11];
                w_shift3 += pp[32 + 12];
                w_shift3 += pp[32 + 13];
                w_shift3 += pp[32 + 14];
                w_shift3 += pp[32 + 15];
                w_shift4 += pp[32 + 16];
                w_shift4 += pp[32 + 17];
                w_shift4 += pp[32 + 18];
                w_shift4 += pp[32 + 19];
                w_shift5 += pp[32 + 20];
                w_shift5 += pp[32 + 21];
                w_shift5 += pp[32 + 22];
                w_shift5 += pp[32 + 23];
                w_shift6 += pp[32 + 24];
                w_shift6 += pp[32 + 25];
                w_shift6 += pp[32 + 26];
                w_shift6 += pp[32 + 27];
                w_shift7 += pp[32 + 28];
                w_shift7 += pp[32 + 29];
                w_shift7 += pp[32 + 30];
                w_shift7 += pp[32 + 31];
                pp += 64;
                p0 += A_hstep * 8;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#else // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[16] * scale2);
                pp[5] = float2int8(p0[17] * scale2);
                pp[6] = float2int8(p0[24] * scale3);
                pp[7] = float2int8(p0[25] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[32] * scale4);
                pp[9] = float2int8(p0[33] * scale4);
                pp[10] = float2int8(p0[40] * scale5);
                pp[11] = float2int8(p0[41] * scale5);
                pp[12] = float2int8(p0[48] * scale6);
                pp[13] = float2int8(p0[49] * scale6);
                pp[14] = float2int8(p0[56] * scale7);
                pp[15] = float2int8(p0[57] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[32] * scale4);
                pp1[1] = float2int8(p0[33] * scale4);
                pp1[2] = float2int8(p0[40] * scale5);
                pp1[3] = float2int8(p0[41] * scale5);
                pp1[4] = float2int8(p0[48] * scale6);
                pp1[5] = float2int8(p0[49] * scale6);
                pp1[6] = float2int8(p0[56] * scale7);
                pp1[7] = float2int8(p0[57] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[2] * scale0);
                pp[1] = float2int8(p0[3] * scale0);
                pp[2] = float2int8(p0[10] * scale1);
                pp[3] = float2int8(p0[11] * scale1);
                pp[4] = float2int8(p0[18] * scale2);
                pp[5] = float2int8(p0[19] * scale2);
                pp[6] = float2int8(p0[26] * scale3);
                pp[7] = float2int8(p0[27] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[34] * scale4);
                pp[9] = float2int8(p0[35] * scale4);
                pp[10] = float2int8(p0[42] * scale5);
                pp[11] = float2int8(p0[43] * scale5);
                pp[12] = float2int8(p0[50] * scale6);
                pp[13] = float2int8(p0[51] * scale6);
                pp[14] = float2int8(p0[58] * scale7);
                pp[15] = float2int8(p0[59] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[34] * scale4);
                pp1[1] = float2int8(p0[35] * scale4);
                pp1[2] = float2int8(p0[42] * scale5);
                pp1[3] = float2int8(p0[43] * scale5);
                pp1[4] = float2int8(p0[50] * scale6);
                pp1[5] = float2int8(p0[51] * scale6);
                pp1[6] = float2int8(p0[58] * scale7);
                pp1[7] = float2int8(p0[59] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[4] * scale0);
                pp[1] = float2int8(p0[5] * scale0);
                pp[2] = float2int8(p0[12] * scale1);
                pp[3] = float2int8(p0[13] * scale1);
                pp[4] = float2int8(p0[20] * scale2);
                pp[5] = float2int8(p0[21] * scale2);
                pp[6] = float2int8(p0[28] * scale3);
                pp[7] = float2int8(p0[29] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[36] * scale4);
                pp[9] = float2int8(p0[37] * scale4);
                pp[10] = float2int8(p0[44] * scale5);
                pp[11] = float2int8(p0[45] * scale5);
                pp[12] = float2int8(p0[52] * scale6);
                pp[13] = float2int8(p0[53] * scale6);
                pp[14] = float2int8(p0[60] * scale7);
                pp[15] = float2int8(p0[61] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[36] * scale4);
                pp1[1] = float2int8(p0[37] * scale4);
                pp1[2] = float2int8(p0[44] * scale5);
                pp1[3] = float2int8(p0[45] * scale5);
                pp1[4] = float2int8(p0[52] * scale6);
                pp1[5] = float2int8(p0[53] * scale6);
                pp1[6] = float2int8(p0[60] * scale7);
                pp1[7] = float2int8(p0[61] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[6] * scale0);
                pp[1] = float2int8(p0[7] * scale0);
                pp[2] = float2int8(p0[14] * scale1);
                pp[3] = float2int8(p0[15] * scale1);
                pp[4] = float2int8(p0[22] * scale2);
                pp[5] = float2int8(p0[23] * scale2);
                pp[6] = float2int8(p0[30] * scale3);
                pp[7] = float2int8(p0[31] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[38] * scale4);
                pp[9] = float2int8(p0[39] * scale4);
                pp[10] = float2int8(p0[46] * scale5);
                pp[11] = float2int8(p0[47] * scale5);
                pp[12] = float2int8(p0[54] * scale6);
                pp[13] = float2int8(p0[55] * scale6);
                pp[14] = float2int8(p0[62] * scale7);
                pp[15] = float2int8(p0[63] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[38] * scale4);
                pp1[1] = float2int8(p0[39] * scale4);
                pp1[2] = float2int8(p0[46] * scale5);
                pp1[3] = float2int8(p0[47] * scale5);
                pp1[4] = float2int8(p0[54] * scale6);
                pp1[5] = float2int8(p0[55] * scale6);
                pp1[6] = float2int8(p0[62] * scale7);
                pp1[7] = float2int8(p0[63] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[4] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);
                pp[8] = float2int8(p0[8] * scale2);
                pp[9] = float2int8(p0[9] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[11] * scale2);
                pp[12] = float2int8(p0[12] * scale3);
                pp[13] = float2int8(p0[13] * scale3);
                pp[14] = float2int8(p0[14] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                pp[16] = float2int8(p0[16] * scale4);
                pp[17] = float2int8(p0[17] * scale4);
                pp[18] = float2int8(p0[18] * scale4);
                pp[19] = float2int8(p0[19] * scale4);
                pp[20] = float2int8(p0[20] * scale5);
                pp[21] = float2int8(p0[21] * scale5);
                pp[22] = float2int8(p0[22] * scale5);
                pp[23] = float2int8(p0[23] * scale5);
                pp[24] = float2int8(p0[24] * scale6);
                pp[25] = float2int8(p0[25] * scale6);
                pp[26] = float2int8(p0[26] * scale6);
                pp[27] = float2int8(p0[27] * scale6);
                pp[28] = float2int8(p0[28] * scale7);
                pp[29] = float2int8(p0[29] * scale7);
                pp[30] = float2int8(p0[30] * scale7);
                pp[31] = float2int8(p0[31] * scale7);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];
                pp += 32;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#else // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[8] * scale2);
                pp[5] = float2int8(p0[9] * scale2);
                pp[6] = float2int8(p0[12] * scale3);
                pp[7] = float2int8(p0[13] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[16] * scale4);
                pp[9] = float2int8(p0[17] * scale4);
                pp[10] = float2int8(p0[20] * scale5);
                pp[11] = float2int8(p0[21] * scale5);
                pp[12] = float2int8(p0[24] * scale6);
                pp[13] = float2int8(p0[25] * scale6);
                pp[14] = float2int8(p0[28] * scale7);
                pp[15] = float2int8(p0[29] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[16] * scale4);
                pp1[1] = float2int8(p0[17] * scale4);
                pp1[2] = float2int8(p0[20] * scale5);
                pp1[3] = float2int8(p0[21] * scale5);
                pp1[4] = float2int8(p0[24] * scale6);
                pp1[5] = float2int8(p0[25] * scale6);
                pp1[6] = float2int8(p0[28] * scale7);
                pp1[7] = float2int8(p0[29] * scale7);
                pp += 8;
                pp1 += 8;
#endif

                pp[0] = float2int8(p0[2] * scale0);
                pp[1] = float2int8(p0[3] * scale0);
                pp[2] = float2int8(p0[6] * scale1);
                pp[3] = float2int8(p0[7] * scale1);
                pp[4] = float2int8(p0[10] * scale2);
                pp[5] = float2int8(p0[11] * scale2);
                pp[6] = float2int8(p0[14] * scale3);
                pp[7] = float2int8(p0[15] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[18] * scale4);
                pp[9] = float2int8(p0[19] * scale4);
                pp[10] = float2int8(p0[22] * scale5);
                pp[11] = float2int8(p0[23] * scale5);
                pp[12] = float2int8(p0[26] * scale6);
                pp[13] = float2int8(p0[27] * scale6);
                pp[14] = float2int8(p0[30] * scale7);
                pp[15] = float2int8(p0[31] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[18] * scale4);
                pp1[1] = float2int8(p0[19] * scale4);
                pp1[2] = float2int8(p0[22] * scale5);
                pp1[3] = float2int8(p0[23] * scale5);
                pp1[4] = float2int8(p0[26] * scale6);
                pp1[5] = float2int8(p0[27] * scale6);
                pp1[6] = float2int8(p0[30] * scale7);
                pp1[7] = float2int8(p0[31] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            int w_shift4 = 0;
            int w_shift5 = 0;
            int w_shift6 = 0;
            int w_shift7 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[A_hstep * 2] * scale0);
                pp[3] = float2int8(p0[A_hstep * 3] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep * 2 + 1] * scale1);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[A_hstep + 2] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 3 + 2] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[A_hstep + 3] * scale3);
                pp[14] = float2int8(p0[A_hstep * 2 + 3] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                pp[16] = float2int8(p0[4] * scale4);
                pp[17] = float2int8(p0[A_hstep + 4] * scale4);
                pp[18] = float2int8(p0[A_hstep * 2 + 4] * scale4);
                pp[19] = float2int8(p0[A_hstep * 3 + 4] * scale4);
                pp[20] = float2int8(p0[5] * scale5);
                pp[21] = float2int8(p0[A_hstep + 5] * scale5);
                pp[22] = float2int8(p0[A_hstep * 2 + 5] * scale5);
                pp[23] = float2int8(p0[A_hstep * 3 + 5] * scale5);
                pp[24] = float2int8(p0[6] * scale6);
                pp[25] = float2int8(p0[A_hstep + 6] * scale6);
                pp[26] = float2int8(p0[A_hstep * 2 + 6] * scale6);
                pp[27] = float2int8(p0[A_hstep * 3 + 6] * scale6);
                pp[28] = float2int8(p0[7] * scale7);
                pp[29] = float2int8(p0[A_hstep + 7] * scale7);
                pp[30] = float2int8(p0[A_hstep * 2 + 7] * scale7);
                pp[31] = float2int8(p0[A_hstep * 3 + 7] * scale7);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                w_shift4 += pp[16];
                w_shift4 += pp[17];
                w_shift4 += pp[18];
                w_shift4 += pp[19];
                w_shift5 += pp[20];
                w_shift5 += pp[21];
                w_shift5 += pp[22];
                w_shift5 += pp[23];
                w_shift6 += pp[24];
                w_shift6 += pp[25];
                w_shift6 += pp[26];
                w_shift6 += pp[27];
                w_shift7 += pp[28];
                w_shift7 += pp[29];
                w_shift7 += pp[30];
                w_shift7 += pp[31];
                pp += 32;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                ((int*)pp)[4] = w_shift4 * 127;
                ((int*)pp)[5] = w_shift5 * 127;
                ((int*)pp)[6] = w_shift6 * 127;
                ((int*)pp)[7] = w_shift7 * 127;
                pp += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[A_hstep + 2] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[A_hstep + 3] * scale3);
#if __AVX2__
                pp[8] = float2int8(p0[4] * scale4);
                pp[9] = float2int8(p0[A_hstep + 4] * scale4);
                pp[10] = float2int8(p0[5] * scale5);
                pp[11] = float2int8(p0[A_hstep + 5] * scale5);
                pp[12] = float2int8(p0[6] * scale6);
                pp[13] = float2int8(p0[A_hstep + 6] * scale6);
                pp[14] = float2int8(p0[7] * scale7);
                pp[15] = float2int8(p0[A_hstep + 7] * scale7);
                pp += 16;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[A_hstep + 4] * scale4);
                pp1[2] = float2int8(p0[5] * scale5);
                pp1[3] = float2int8(p0[A_hstep + 5] * scale5);
                pp1[4] = float2int8(p0[6] * scale6);
                pp1[5] = float2int8(p0[A_hstep + 6] * scale6);
                pp1[6] = float2int8(p0[7] * scale7);
                pp1[7] = float2int8(p0[A_hstep + 7] * scale7);
                pp += 8;
                pp1 += 8;
#endif
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);
#if __AVX2__
                pp[4] = float2int8(p0[4] * scale4);
                pp[5] = float2int8(p0[5] * scale5);
                pp[6] = float2int8(p0[6] * scale6);
                pp[7] = float2int8(p0[7] * scale7);
                pp += 8;
#else
                pp1[0] = float2int8(p0[4] * scale4);
                pp1[1] = float2int8(p0[5] * scale5);
                pp1[2] = float2int8(p0[6] * scale6);
                pp1[3] = float2int8(p0[7] * scale7);
                pp += 4;
                pp1 += 4;
#endif
                p0 += A_hstep;
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_kk * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2 + 0] * scale0);
                pp[3] = float2int8(p0[2 + 1] * scale0);
                pp[4] = float2int8(p0[16] * scale1);
                pp[5] = float2int8(p0[17] * scale1);
                pp[6] = float2int8(p0[2 + 16] * scale1);
                pp[7] = float2int8(p0[2 + 17] * scale1);
                pp[8] = float2int8(p0[32] * scale2);
                pp[9] = float2int8(p0[33] * scale2);
                pp[10] = float2int8(p0[2 + 32] * scale2);
                pp[11] = float2int8(p0[2 + 33] * scale2);
                pp[12] = float2int8(p0[48] * scale3);
                pp[13] = float2int8(p0[49] * scale3);
                pp[14] = float2int8(p0[2 + 48] * scale3);
                pp[15] = float2int8(p0[2 + 49] * scale3);

                pp[16 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[16 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[16 + 2] = float2int8(p0[6 + 0] * scale0);
                pp[16 + 3] = float2int8(p0[6 + 1] * scale0);
                pp[16 + 4] = float2int8(p0[4 + 16] * scale1);
                pp[16 + 5] = float2int8(p0[4 + 17] * scale1);
                pp[16 + 6] = float2int8(p0[6 + 16] * scale1);
                pp[16 + 7] = float2int8(p0[6 + 17] * scale1);
                pp[16 + 8] = float2int8(p0[4 + 32] * scale2);
                pp[16 + 9] = float2int8(p0[4 + 33] * scale2);
                pp[16 + 10] = float2int8(p0[6 + 32] * scale2);
                pp[16 + 11] = float2int8(p0[6 + 33] * scale2);
                pp[16 + 12] = float2int8(p0[4 + 48] * scale3);
                pp[16 + 13] = float2int8(p0[4 + 49] * scale3);
                pp[16 + 14] = float2int8(p0[6 + 48] * scale3);
                pp[16 + 15] = float2int8(p0[6 + 49] * scale3);

                pp[32 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[32 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[32 + 2] = float2int8(p0[10 + 0] * scale0);
                pp[32 + 3] = float2int8(p0[10 + 1] * scale0);
                pp[32 + 4] = float2int8(p0[8 + 16] * scale1);
                pp[32 + 5] = float2int8(p0[8 + 17] * scale1);
                pp[32 + 6] = float2int8(p0[10 + 16] * scale1);
                pp[32 + 7] = float2int8(p0[10 + 17] * scale1);
                pp[32 + 8] = float2int8(p0[8 + 32] * scale2);
                pp[32 + 9] = float2int8(p0[8 + 33] * scale2);
                pp[32 + 10] = float2int8(p0[10 + 32] * scale2);
                pp[32 + 11] = float2int8(p0[10 + 33] * scale2);
                pp[32 + 12] = float2int8(p0[8 + 48] * scale3);
                pp[32 + 13] = float2int8(p0[8 + 49] * scale3);
                pp[32 + 14] = float2int8(p0[10 + 48] * scale3);
                pp[32 + 15] = float2int8(p0[10 + 49] * scale3);

                pp[48 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[48 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[48 + 2] = float2int8(p0[14 + 0] * scale0);
                pp[48 + 3] = float2int8(p0[14 + 1] * scale0);
                pp[48 + 4] = float2int8(p0[12 + 16] * scale1);
                pp[48 + 5] = float2int8(p0[12 + 17] * scale1);
                pp[48 + 6] = float2int8(p0[14 + 16] * scale1);
                pp[48 + 7] = float2int8(p0[14 + 17] * scale1);
                pp[48 + 8] = float2int8(p0[12 + 32] * scale2);
                pp[48 + 9] = float2int8(p0[12 + 33] * scale2);
                pp[48 + 10] = float2int8(p0[14 + 32] * scale2);
                pp[48 + 11] = float2int8(p0[14 + 33] * scale2);
                pp[48 + 12] = float2int8(p0[12 + 48] * scale3);
                pp[48 + 13] = float2int8(p0[12 + 49] * scale3);
                pp[48 + 14] = float2int8(p0[14 + 48] * scale3);
                pp[48 + 15] = float2int8(p0[14 + 49] * scale3);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];

                w_shift0 += pp[16 + 0];
                w_shift0 += pp[16 + 1];
                w_shift0 += pp[16 + 2];
                w_shift0 += pp[16 + 3];
                w_shift1 += pp[16 + 4];
                w_shift1 += pp[16 + 5];
                w_shift1 += pp[16 + 6];
                w_shift1 += pp[16 + 7];
                w_shift2 += pp[16 + 8];
                w_shift2 += pp[16 + 9];
                w_shift2 += pp[16 + 10];
                w_shift2 += pp[16 + 11];
                w_shift3 += pp[16 + 12];
                w_shift3 += pp[16 + 13];
                w_shift3 += pp[16 + 14];
                w_shift3 += pp[16 + 15];

                w_shift0 += pp[32 + 0];
                w_shift0 += pp[32 + 1];
                w_shift0 += pp[32 + 2];
                w_shift0 += pp[32 + 3];
                w_shift1 += pp[32 + 4];
                w_shift1 += pp[32 + 5];
                w_shift1 += pp[32 + 6];
                w_shift1 += pp[32 + 7];
                w_shift2 += pp[32 + 8];
                w_shift2 += pp[32 + 9];
                w_shift2 += pp[32 + 10];
                w_shift2 += pp[32 + 11];
                w_shift3 += pp[32 + 12];
                w_shift3 += pp[32 + 13];
                w_shift3 += pp[32 + 14];
                w_shift3 += pp[32 + 15];

                w_shift0 += pp[48 + 0];
                w_shift0 += pp[48 + 1];
                w_shift0 += pp[48 + 2];
                w_shift0 += pp[48 + 3];
                w_shift1 += pp[48 + 4];
                w_shift1 += pp[48 + 5];
                w_shift1 += pp[48 + 6];
                w_shift1 += pp[48 + 7];
                w_shift2 += pp[48 + 8];
                w_shift2 += pp[48 + 9];
                w_shift2 += pp[48 + 10];
                w_shift2 += pp[48 + 11];
                w_shift3 += pp[48 + 12];
                w_shift3 += pp[48 + 13];
                w_shift3 += pp[48 + 14];
                w_shift3 += pp[48 + 15];

                pp += 64;
                p0 += A_hstep * 16;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[16] * scale1);
                pp[3] = float2int8(p0[17] * scale1);
                pp[4] = float2int8(p0[32] * scale2);
                pp[5] = float2int8(p0[33] * scale2);
                pp[6] = float2int8(p0[48] * scale3);
                pp[7] = float2int8(p0[49] * scale3);

                pp[8] = float2int8(p0[2 + 0] * scale0);
                pp[9] = float2int8(p0[2 + 1] * scale0);
                pp[10] = float2int8(p0[2 + 16] * scale1);
                pp[11] = float2int8(p0[2 + 17] * scale1);
                pp[12] = float2int8(p0[2 + 32] * scale2);
                pp[13] = float2int8(p0[2 + 33] * scale2);
                pp[14] = float2int8(p0[2 + 48] * scale3);
                pp[15] = float2int8(p0[2 + 49] * scale3);

                pp[16 + 0] = float2int8(p0[4 + 0] * scale0);
                pp[16 + 1] = float2int8(p0[4 + 1] * scale0);
                pp[16 + 2] = float2int8(p0[4 + 16] * scale1);
                pp[16 + 3] = float2int8(p0[4 + 17] * scale1);
                pp[16 + 4] = float2int8(p0[4 + 32] * scale2);
                pp[16 + 5] = float2int8(p0[4 + 33] * scale2);
                pp[16 + 6] = float2int8(p0[4 + 48] * scale3);
                pp[16 + 7] = float2int8(p0[4 + 49] * scale3);

                pp[16 + 8] = float2int8(p0[6 + 0] * scale0);
                pp[16 + 9] = float2int8(p0[6 + 1] * scale0);
                pp[16 + 10] = float2int8(p0[6 + 16] * scale1);
                pp[16 + 11] = float2int8(p0[6 + 17] * scale1);
                pp[16 + 12] = float2int8(p0[6 + 32] * scale2);
                pp[16 + 13] = float2int8(p0[6 + 33] * scale2);
                pp[16 + 14] = float2int8(p0[6 + 48] * scale3);
                pp[16 + 15] = float2int8(p0[6 + 49] * scale3);

                pp[32 + 0] = float2int8(p0[8 + 0] * scale0);
                pp[32 + 1] = float2int8(p0[8 + 1] * scale0);
                pp[32 + 2] = float2int8(p0[8 + 16] * scale1);
                pp[32 + 3] = float2int8(p0[8 + 17] * scale1);
                pp[32 + 4] = float2int8(p0[8 + 32] * scale2);
                pp[32 + 5] = float2int8(p0[8 + 33] * scale2);
                pp[32 + 6] = float2int8(p0[8 + 48] * scale3);
                pp[32 + 7] = float2int8(p0[8 + 49] * scale3);

                pp[32 + 8] = float2int8(p0[10 + 0] * scale0);
                pp[32 + 9] = float2int8(p0[10 + 1] * scale0);
                pp[32 + 10] = float2int8(p0[10 + 16] * scale1);
                pp[32 + 11] = float2int8(p0[10 + 17] * scale1);
                pp[32 + 12] = float2int8(p0[10 + 32] * scale2);
                pp[32 + 13] = float2int8(p0[10 + 33] * scale2);
                pp[32 + 14] = float2int8(p0[10 + 48] * scale3);
                pp[32 + 15] = float2int8(p0[10 + 49] * scale3);

                pp[48 + 0] = float2int8(p0[12 + 0] * scale0);
                pp[48 + 1] = float2int8(p0[12 + 1] * scale0);
                pp[48 + 2] = float2int8(p0[12 + 16] * scale1);
                pp[48 + 3] = float2int8(p0[12 + 17] * scale1);
                pp[48 + 4] = float2int8(p0[12 + 32] * scale2);
                pp[48 + 5] = float2int8(p0[12 + 33] * scale2);
                pp[48 + 6] = float2int8(p0[12 + 48] * scale3);
                pp[48 + 7] = float2int8(p0[12 + 49] * scale3);

                pp[48 + 8] = float2int8(p0[14 + 0] * scale0);
                pp[48 + 9] = float2int8(p0[14 + 1] * scale0);
                pp[48 + 10] = float2int8(p0[14 + 16] * scale1);
                pp[48 + 11] = float2int8(p0[14 + 17] * scale1);
                pp[48 + 12] = float2int8(p0[14 + 32] * scale2);
                pp[48 + 13] = float2int8(p0[14 + 33] * scale2);
                pp[48 + 14] = float2int8(p0[14 + 48] * scale3);
                pp[48 + 15] = float2int8(p0[14 + 49] * scale3);

                pp += 64;
                p0 += A_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[8] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[16] * scale2);
                pp[9] = float2int8(p0[17] * scale2);
                pp[10] = float2int8(p0[18] * scale2);
                pp[11] = float2int8(p0[19] * scale2);
                pp[12] = float2int8(p0[24] * scale3);
                pp[13] = float2int8(p0[25] * scale3);
                pp[14] = float2int8(p0[26] * scale3);
                pp[15] = float2int8(p0[27] * scale3);

                pp[16 + 0] = float2int8(p0[4] * scale0);
                pp[16 + 1] = float2int8(p0[5] * scale0);
                pp[16 + 2] = float2int8(p0[6] * scale0);
                pp[16 + 3] = float2int8(p0[7] * scale0);
                pp[16 + 4] = float2int8(p0[12] * scale1);
                pp[16 + 5] = float2int8(p0[13] * scale1);
                pp[16 + 6] = float2int8(p0[14] * scale1);
                pp[16 + 7] = float2int8(p0[15] * scale1);
                pp[16 + 8] = float2int8(p0[20] * scale2);
                pp[16 + 9] = float2int8(p0[21] * scale2);
                pp[16 + 10] = float2int8(p0[22] * scale2);
                pp[16 + 11] = float2int8(p0[23] * scale2);
                pp[16 + 12] = float2int8(p0[28] * scale3);
                pp[16 + 13] = float2int8(p0[29] * scale3);
                pp[16 + 14] = float2int8(p0[30] * scale3);
                pp[16 + 15] = float2int8(p0[31] * scale3);

                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];

                w_shift0 += pp[16 + 0];
                w_shift0 += pp[16 + 1];
                w_shift0 += pp[16 + 2];
                w_shift0 += pp[16 + 3];
                w_shift1 += pp[16 + 4];
                w_shift1 += pp[16 + 5];
                w_shift1 += pp[16 + 6];
                w_shift1 += pp[16 + 7];
                w_shift2 += pp[16 + 8];
                w_shift2 += pp[16 + 9];
                w_shift2 += pp[16 + 10];
                w_shift2 += pp[16 + 11];
                w_shift3 += pp[16 + 12];
                w_shift3 += pp[16 + 13];
                w_shift3 += pp[16 + 14];
                w_shift3 += pp[16 + 15];
                pp += 32;
                p0 += A_hstep * 8;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[16] * scale2);
                pp[5] = float2int8(p0[17] * scale2);
                pp[6] = float2int8(p0[24] * scale3);
                pp[7] = float2int8(p0[25] * scale3);

                pp[8] = float2int8(p0[2] * scale0);
                pp[9] = float2int8(p0[3] * scale0);
                pp[10] = float2int8(p0[10] * scale1);
                pp[11] = float2int8(p0[11] * scale1);
                pp[12] = float2int8(p0[18] * scale2);
                pp[13] = float2int8(p0[19] * scale2);
                pp[14] = float2int8(p0[26] * scale3);
                pp[15] = float2int8(p0[27] * scale3);

                pp[16 + 0] = float2int8(p0[4] * scale0);
                pp[16 + 1] = float2int8(p0[5] * scale0);
                pp[16 + 2] = float2int8(p0[12] * scale1);
                pp[16 + 3] = float2int8(p0[13] * scale1);
                pp[16 + 4] = float2int8(p0[20] * scale2);
                pp[16 + 5] = float2int8(p0[21] * scale2);
                pp[16 + 6] = float2int8(p0[28] * scale3);
                pp[16 + 7] = float2int8(p0[29] * scale3);

                pp[16 + 8] = float2int8(p0[6] * scale0);
                pp[16 + 9] = float2int8(p0[7] * scale0);
                pp[16 + 10] = float2int8(p0[14] * scale1);
                pp[16 + 11] = float2int8(p0[15] * scale1);
                pp[16 + 12] = float2int8(p0[22] * scale2);
                pp[16 + 13] = float2int8(p0[23] * scale2);
                pp[16 + 14] = float2int8(p0[30] * scale3);
                pp[16 + 15] = float2int8(p0[31] * scale3);

                pp += 32;
                p0 += A_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[4] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);
                pp[8] = float2int8(p0[8] * scale2);
                pp[9] = float2int8(p0[9] * scale2);
                pp[10] = float2int8(p0[10] * scale2);
                pp[11] = float2int8(p0[11] * scale2);
                pp[12] = float2int8(p0[12] * scale3);
                pp[13] = float2int8(p0[13] * scale3);
                pp[14] = float2int8(p0[14] * scale3);
                pp[15] = float2int8(p0[15] * scale3);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                pp += 16;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[8] * scale2);
                pp[5] = float2int8(p0[9] * scale2);
                pp[6] = float2int8(p0[12] * scale3);
                pp[7] = float2int8(p0[13] * scale3);
                pp[8] = float2int8(p0[2] * scale0);
                pp[9] = float2int8(p0[3] * scale0);
                pp[10] = float2int8(p0[6] * scale1);
                pp[11] = float2int8(p0[7] * scale1);
                pp[12] = float2int8(p0[10] * scale2);
                pp[13] = float2int8(p0[11] * scale2);
                pp[14] = float2int8(p0[14] * scale3);
                pp[15] = float2int8(p0[15] * scale3);

                pp += 16;
                p0 += A_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            int w_shift2 = 0;
            int w_shift3 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[A_hstep * 2] * scale0);
                pp[3] = float2int8(p0[A_hstep * 3] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep * 2 + 1] * scale1);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale1);
                pp[8] = float2int8(p0[2] * scale2);
                pp[9] = float2int8(p0[A_hstep + 2] * scale2);
                pp[10] = float2int8(p0[A_hstep * 2 + 2] * scale2);
                pp[11] = float2int8(p0[A_hstep * 3 + 2] * scale2);
                pp[12] = float2int8(p0[3] * scale3);
                pp[13] = float2int8(p0[A_hstep + 3] * scale3);
                pp[14] = float2int8(p0[A_hstep * 2 + 3] * scale3);
                pp[15] = float2int8(p0[A_hstep * 3 + 3] * scale3);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift2 += pp[8];
                w_shift2 += pp[9];
                w_shift2 += pp[10];
                w_shift2 += pp[11];
                w_shift3 += pp[12];
                w_shift3 += pp[13];
                w_shift3 += pp[14];
                w_shift3 += pp[15];
                pp += 16;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                ((int*)pp)[2] = w_shift2 * 127;
                ((int*)pp)[3] = w_shift3 * 127;
                pp += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp[4] = float2int8(p0[2] * scale2);
                pp[5] = float2int8(p0[A_hstep + 2] * scale2);
                pp[6] = float2int8(p0[3] * scale3);
                pp[7] = float2int8(p0[A_hstep + 3] * scale3);

                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp[2] = float2int8(p0[2] * scale2);
                pp[3] = float2int8(p0[3] * scale3);

                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[16] * scale1);
                pp[5] = float2int8(p0[17] * scale1);
                pp[6] = float2int8(p0[18] * scale1);
                pp[7] = float2int8(p0[19] * scale1);

                pp[8] = float2int8(p0[4] * scale0);
                pp[9] = float2int8(p0[5] * scale0);
                pp[10] = float2int8(p0[6] * scale0);
                pp[11] = float2int8(p0[7] * scale0);
                pp[12] = float2int8(p0[20] * scale1);
                pp[13] = float2int8(p0[21] * scale1);
                pp[14] = float2int8(p0[22] * scale1);
                pp[15] = float2int8(p0[23] * scale1);

                pp[16 + 0] = float2int8(p0[8] * scale0);
                pp[16 + 1] = float2int8(p0[9] * scale0);
                pp[16 + 2] = float2int8(p0[10] * scale0);
                pp[16 + 3] = float2int8(p0[11] * scale0);
                pp[16 + 4] = float2int8(p0[24] * scale1);
                pp[16 + 5] = float2int8(p0[25] * scale1);
                pp[16 + 6] = float2int8(p0[26] * scale1);
                pp[16 + 7] = float2int8(p0[27] * scale1);

                pp[16 + 8] = float2int8(p0[12] * scale0);
                pp[16 + 9] = float2int8(p0[13] * scale0);
                pp[16 + 10] = float2int8(p0[14] * scale0);
                pp[16 + 11] = float2int8(p0[15] * scale0);
                pp[16 + 12] = float2int8(p0[28] * scale1);
                pp[16 + 13] = float2int8(p0[29] * scale1);
                pp[16 + 14] = float2int8(p0[30] * scale1);
                pp[16 + 15] = float2int8(p0[31] * scale1);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift0 += pp[8];
                w_shift0 += pp[9];
                w_shift0 += pp[10];
                w_shift0 += pp[11];
                w_shift1 += pp[12];
                w_shift1 += pp[13];
                w_shift1 += pp[14];
                w_shift1 += pp[15];
                w_shift0 += pp[16 + 0];
                w_shift0 += pp[16 + 1];
                w_shift0 += pp[16 + 2];
                w_shift0 += pp[16 + 3];
                w_shift1 += pp[16 + 4];
                w_shift1 += pp[16 + 5];
                w_shift1 += pp[16 + 6];
                w_shift1 += pp[16 + 7];
                w_shift0 += pp[16 + 8];
                w_shift0 += pp[16 + 9];
                w_shift0 += pp[16 + 10];
                w_shift0 += pp[16 + 11];
                w_shift1 += pp[16 + 12];
                w_shift1 += pp[16 + 13];
                w_shift1 += pp[16 + 14];
                w_shift1 += pp[16 + 15];
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[16] * scale1);
                pp[3] = float2int8(p0[17] * scale1);

                pp[4] = float2int8(p0[2] * scale0);
                pp[5] = float2int8(p0[3] * scale0);
                pp[6] = float2int8(p0[18] * scale1);
                pp[7] = float2int8(p0[19] * scale1);

                pp[8] = float2int8(p0[4] * scale0);
                pp[9] = float2int8(p0[5] * scale0);
                pp[10] = float2int8(p0[20] * scale1);
                pp[11] = float2int8(p0[21] * scale1);

                pp[12] = float2int8(p0[6] * scale0);
                pp[13] = float2int8(p0[7] * scale0);
                pp[14] = float2int8(p0[22] * scale1);
                pp[15] = float2int8(p0[23] * scale1);

                pp[16 + 0] = float2int8(p0[8] * scale0);
                pp[16 + 1] = float2int8(p0[9] * scale0);
                pp[16 + 2] = float2int8(p0[24] * scale1);
                pp[16 + 3] = float2int8(p0[25] * scale1);

                pp[16 + 4] = float2int8(p0[10] * scale0);
                pp[16 + 5] = float2int8(p0[11] * scale0);
                pp[16 + 6] = float2int8(p0[26] * scale1);
                pp[16 + 7] = float2int8(p0[27] * scale1);

                pp[16 + 8] = float2int8(p0[12] * scale0);
                pp[16 + 9] = float2int8(p0[13] * scale0);
                pp[16 + 10] = float2int8(p0[28] * scale1);
                pp[16 + 11] = float2int8(p0[29] * scale1);

                pp[16 + 12] = float2int8(p0[14] * scale0);
                pp[16 + 13] = float2int8(p0[15] * scale0);
                pp[16 + 14] = float2int8(p0[30] * scale1);
                pp[16 + 15] = float2int8(p0[31] * scale1);
#endif // __AVX512VNNI__
                pp += 32;
                p0 += A_hstep * 16;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[8] * scale1);
                pp[5] = float2int8(p0[9] * scale1);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[4] * scale0);
                pp[9] = float2int8(p0[5] * scale0);
                pp[10] = float2int8(p0[6] * scale0);
                pp[11] = float2int8(p0[7] * scale0);
                pp[12] = float2int8(p0[12] * scale1);
                pp[13] = float2int8(p0[13] * scale1);
                pp[14] = float2int8(p0[14] * scale1);
                pp[15] = float2int8(p0[15] * scale1);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                w_shift0 += pp[8];
                w_shift0 += pp[9];
                w_shift0 += pp[10];
                w_shift0 += pp[11];
                w_shift1 += pp[12];
                w_shift1 += pp[13];
                w_shift1 += pp[14];
                w_shift1 += pp[15];
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[8] * scale1);
                pp[3] = float2int8(p0[9] * scale1);
                pp[4] = float2int8(p0[2] * scale0);
                pp[5] = float2int8(p0[3] * scale0);
                pp[6] = float2int8(p0[10] * scale1);
                pp[7] = float2int8(p0[11] * scale1);
                pp[8] = float2int8(p0[4] * scale0);
                pp[9] = float2int8(p0[5] * scale0);
                pp[10] = float2int8(p0[12] * scale1);
                pp[11] = float2int8(p0[13] * scale1);
                pp[12] = float2int8(p0[6] * scale0);
                pp[13] = float2int8(p0[7] * scale0);
                pp[14] = float2int8(p0[14] * scale1);
                pp[15] = float2int8(p0[15] * scale1);
#endif // __AVX512VNNI__
                pp += 16;
                p0 += A_hstep * 8;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
#endif // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[2] * scale0);
                pp[3] = float2int8(p0[3] * scale0);
                pp[4] = float2int8(p0[4] * scale1);
                pp[5] = float2int8(p0[5] * scale1);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[4] * scale1);
                pp[3] = float2int8(p0[5] * scale1);
                pp[4] = float2int8(p0[2] * scale0);
                pp[5] = float2int8(p0[3] * scale0);
                pp[6] = float2int8(p0[6] * scale1);
                pp[7] = float2int8(p0[7] * scale1);
#endif // __AVX512VNNI__
                pp += 8;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
            int w_shift0 = 0;
            int w_shift1 = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep + 0] * scale0);
                pp[2] = float2int8(p0[A_hstep * 2 + 0] * scale0);
                pp[3] = float2int8(p0[A_hstep * 3 + 0] * scale0);
                pp[4] = float2int8(p0[1] * scale1);
                pp[5] = float2int8(p0[A_hstep + 1] * scale1);
                pp[6] = float2int8(p0[A_hstep * 2 + 1] * scale1);
                pp[7] = float2int8(p0[A_hstep * 3 + 1] * scale1);
                w_shift0 += pp[0];
                w_shift0 += pp[1];
                w_shift0 += pp[2];
                w_shift0 += pp[3];
                w_shift1 += pp[4];
                w_shift1 += pp[5];
                w_shift1 += pp[6];
                w_shift1 += pp[7];
                pp += 8;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift0 * 127;
                ((int*)pp)[1] = w_shift1 * 127;
                pp += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep + 0] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale1);
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale = scales[i + ii];

        // if (max_kk == 32)
        // {
        //     NCNN_LOGE("===== %p  %d   %f", p0, p0[0], scale);
        // }

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift = 0;
#endif // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[8] * scale);
                pp[9] = float2int8(p0[9] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[12] * scale);
                pp[13] = float2int8(p0[13] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

#if __AVX512VNNI__
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
                w_shift += pp[4];
                w_shift += pp[5];
                w_shift += pp[6];
                w_shift += pp[7];
                w_shift += pp[8];
                w_shift += pp[9];
                w_shift += pp[10];
                w_shift += pp[11];
                w_shift += pp[12];
                w_shift += pp[13];
                w_shift += pp[14];
                w_shift += pp[15];
#endif // __AVX512VNNI__
                pp += 16;
                p0 += A_hstep * 16;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift = 0;
#endif // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
#if __AVX512VNNI__
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
                w_shift += pp[4];
                w_shift += pp[5];
                w_shift += pp[6];
                w_shift += pp[7];
#endif // __AVX512VNNI__
                pp += 8;
                p0 += A_hstep * 8;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift = 0;
#endif // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
#if __AVX512VNNI__
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
#endif // __AVX512VNNI__
                pp += 4;
                p0 += A_hstep * 4;
            }
#if __AVX512VNNI__
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            int w_shift = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[A_hstep] * scale);
                pp[2] = float2int8(p0[A_hstep * 2] * scale);
                pp[3] = float2int8(p0[A_hstep * 3] * scale);
                w_shift += pp[0];
                w_shift += pp[1];
                w_shift += pp[2];
                w_shift += pp[3];
                pp += 4;
                p0 += A_hstep * 4;
            }
            if (max_kk >= 4)
            {
                ((int*)pp)[0] = w_shift * 127;
                pp += 4;
            }
#endif // __AVX512VNNI__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void compute_B_fp32_int8_scale(const Mat& B, float& scale)
{
    // NCNN_LOGE("compute_B_fp32_int8_scale");

    float absmax = 0.f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _absmax_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _absmax_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _absmax = _mm_setzero_ps();
#endif // __SSE2__
    for (int i = 0; i < (B.dims == 3 ? B.c : B.h); i++)
    {
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;
        const float* ptr = (const float*)B + i * B_hstep * B.elempack;

        const int size = B.w * B.elempack;

        int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; j + 15 < size; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; j + 7 < size; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; j + 3 < size; j += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _absmax = _mm_max_ps(_absmax, abs_ps(_p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; j < size; j++)
        {
            absmax = std::max(absmax, (float)fabsf(ptr[0]));
            ptr++;
        }
    }
#if __SSE2__
#if __AVX__
#if __AVX512F__
    absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax_avx512));
#endif // __AVX512F__
    absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax_avx));
#endif // __AVX__
    absmax = std::max(absmax, _mm_reduce_max_ps(_absmax));
#endif

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        pack_B_tile_fp32_to_int8_avx512vnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        pack_B_tile_fp32_to_int8_avxvnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d %d", max_jj, max_kk, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[16] * scale) + 127;
                pp[2] = float2int8(p0[32] * scale) + 127;
                pp[3] = float2int8(p0[48] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[17] * scale) + 127;
                pp[6] = float2int8(p0[33] * scale) + 127;
                pp[7] = float2int8(p0[49] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[18] * scale) + 127;
                pp[10] = float2int8(p0[34] * scale) + 127;
                pp[11] = float2int8(p0[50] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[19] * scale) + 127;
                pp[14] = float2int8(p0[35] * scale) + 127;
                pp[15] = float2int8(p0[51] * scale) + 127;
                pp[16] = float2int8(p0[4] * scale) + 127;
                pp[17] = float2int8(p0[20] * scale) + 127;
                pp[18] = float2int8(p0[36] * scale) + 127;
                pp[19] = float2int8(p0[52] * scale) + 127;
                pp[20] = float2int8(p0[5] * scale) + 127;
                pp[21] = float2int8(p0[21] * scale) + 127;
                pp[22] = float2int8(p0[37] * scale) + 127;
                pp[23] = float2int8(p0[53] * scale) + 127;
                pp[24] = float2int8(p0[6] * scale) + 127;
                pp[25] = float2int8(p0[22] * scale) + 127;
                pp[26] = float2int8(p0[38] * scale) + 127;
                pp[27] = float2int8(p0[54] * scale) + 127;
                pp[28] = float2int8(p0[7] * scale) + 127;
                pp[29] = float2int8(p0[23] * scale) + 127;
                pp[30] = float2int8(p0[39] * scale) + 127;
                pp[31] = float2int8(p0[55] * scale) + 127;
                pp[32] = float2int8(p0[8] * scale) + 127;
                pp[33] = float2int8(p0[24] * scale) + 127;
                pp[34] = float2int8(p0[40] * scale) + 127;
                pp[35] = float2int8(p0[56] * scale) + 127;
                pp[36] = float2int8(p0[9] * scale) + 127;
                pp[37] = float2int8(p0[25] * scale) + 127;
                pp[38] = float2int8(p0[41] * scale) + 127;
                pp[39] = float2int8(p0[57] * scale) + 127;
                pp[40] = float2int8(p0[10] * scale) + 127;
                pp[41] = float2int8(p0[26] * scale) + 127;
                pp[42] = float2int8(p0[42] * scale) + 127;
                pp[43] = float2int8(p0[58] * scale) + 127;
                pp[44] = float2int8(p0[11] * scale) + 127;
                pp[45] = float2int8(p0[27] * scale) + 127;
                pp[46] = float2int8(p0[43] * scale) + 127;
                pp[47] = float2int8(p0[59] * scale) + 127;
                pp[48] = float2int8(p0[12] * scale) + 127;
                pp[49] = float2int8(p0[28] * scale) + 127;
                pp[50] = float2int8(p0[44] * scale) + 127;
                pp[51] = float2int8(p0[60] * scale) + 127;
                pp[52] = float2int8(p0[13] * scale) + 127;
                pp[53] = float2int8(p0[29] * scale) + 127;
                pp[54] = float2int8(p0[45] * scale) + 127;
                pp[55] = float2int8(p0[61] * scale) + 127;
                pp[56] = float2int8(p0[14] * scale) + 127;
                pp[57] = float2int8(p0[30] * scale) + 127;
                pp[58] = float2int8(p0[46] * scale) + 127;
                pp[59] = float2int8(p0[62] * scale) + 127;
                pp[60] = float2int8(p0[15] * scale) + 127;
                pp[61] = float2int8(p0[31] * scale) + 127;
                pp[62] = float2int8(p0[47] * scale) + 127;
                pp[63] = float2int8(p0[63] * scale) + 127;
                pp += 64;
                p0 += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[16] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[17] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[18] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[19] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[20] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[21] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[22] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[23] * scale);
                pp[16 + 0] = float2int8(p0[8] * scale);
                pp[16 + 1] = float2int8(p0[24] * scale);
                pp[16 + 2] = float2int8(p0[9] * scale);
                pp[16 + 3] = float2int8(p0[25] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[26] * scale);
                pp[16 + 6] = float2int8(p0[11] * scale);
                pp[16 + 7] = float2int8(p0[27] * scale);
                pp[16 + 8] = float2int8(p0[12] * scale);
                pp[16 + 9] = float2int8(p0[28] * scale);
                pp[16 + 10] = float2int8(p0[13] * scale);
                pp[16 + 11] = float2int8(p0[29] * scale);
                pp[16 + 12] = float2int8(p0[14] * scale);
                pp[16 + 13] = float2int8(p0[30] * scale);
                pp[16 + 14] = float2int8(p0[15] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);
                pp += 32;
                p0 += 32;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[8] * scale);
                pp[9] = float2int8(p0[9] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[12] * scale);
                pp[13] = float2int8(p0[13] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);
                pp += 16;
                p0 += 16;
            }
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[8] * scale) + 127;
                pp[2] = float2int8(p0[16] * scale) + 127;
                pp[3] = float2int8(p0[24] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[17] * scale) + 127;
                pp[7] = float2int8(p0[25] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[10] * scale) + 127;
                pp[10] = float2int8(p0[18] * scale) + 127;
                pp[11] = float2int8(p0[26] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[11] * scale) + 127;
                pp[14] = float2int8(p0[19] * scale) + 127;
                pp[15] = float2int8(p0[27] * scale) + 127;
                pp[16] = float2int8(p0[4] * scale) + 127;
                pp[17] = float2int8(p0[12] * scale) + 127;
                pp[18] = float2int8(p0[20] * scale) + 127;
                pp[19] = float2int8(p0[28] * scale) + 127;
                pp[20] = float2int8(p0[5] * scale) + 127;
                pp[21] = float2int8(p0[13] * scale) + 127;
                pp[22] = float2int8(p0[21] * scale) + 127;
                pp[23] = float2int8(p0[29] * scale) + 127;
                pp[24] = float2int8(p0[6] * scale) + 127;
                pp[25] = float2int8(p0[14] * scale) + 127;
                pp[26] = float2int8(p0[22] * scale) + 127;
                pp[27] = float2int8(p0[30] * scale) + 127;
                pp[28] = float2int8(p0[7] * scale) + 127;
                pp[29] = float2int8(p0[15] * scale) + 127;
                pp[30] = float2int8(p0[23] * scale) + 127;
                pp[31] = float2int8(p0[31] * scale) + 127;

                pp[32 + 0] = float2int8(p0[B_hstep * 8 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[B_hstep * 8 + 8] * scale) + 127;
                pp[32 + 2] = float2int8(p0[B_hstep * 8 + 16] * scale) + 127;
                pp[32 + 3] = float2int8(p0[B_hstep * 8 + 24] * scale) + 127;
                pp[32 + 4] = float2int8(p0[B_hstep * 8 + 1] * scale) + 127;
                pp[32 + 5] = float2int8(p0[B_hstep * 8 + 9] * scale) + 127;
                pp[32 + 6] = float2int8(p0[B_hstep * 8 + 17] * scale) + 127;
                pp[32 + 7] = float2int8(p0[B_hstep * 8 + 25] * scale) + 127;
                pp[32 + 8] = float2int8(p0[B_hstep * 8 + 2] * scale) + 127;
                pp[32 + 9] = float2int8(p0[B_hstep * 8 + 10] * scale) + 127;
                pp[32 + 10] = float2int8(p0[B_hstep * 8 + 18] * scale) + 127;
                pp[32 + 11] = float2int8(p0[B_hstep * 8 + 26] * scale) + 127;
                pp[32 + 12] = float2int8(p0[B_hstep * 8 + 3] * scale) + 127;
                pp[32 + 13] = float2int8(p0[B_hstep * 8 + 11] * scale) + 127;
                pp[32 + 14] = float2int8(p0[B_hstep * 8 + 19] * scale) + 127;
                pp[32 + 15] = float2int8(p0[B_hstep * 8 + 27] * scale) + 127;
                pp[32 + 16] = float2int8(p0[B_hstep * 8 + 4] * scale) + 127;
                pp[32 + 17] = float2int8(p0[B_hstep * 8 + 12] * scale) + 127;
                pp[32 + 18] = float2int8(p0[B_hstep * 8 + 20] * scale) + 127;
                pp[32 + 19] = float2int8(p0[B_hstep * 8 + 28] * scale) + 127;
                pp[32 + 20] = float2int8(p0[B_hstep * 8 + 5] * scale) + 127;
                pp[32 + 21] = float2int8(p0[B_hstep * 8 + 13] * scale) + 127;
                pp[32 + 22] = float2int8(p0[B_hstep * 8 + 21] * scale) + 127;
                pp[32 + 23] = float2int8(p0[B_hstep * 8 + 29] * scale) + 127;
                pp[32 + 24] = float2int8(p0[B_hstep * 8 + 6] * scale) + 127;
                pp[32 + 25] = float2int8(p0[B_hstep * 8 + 14] * scale) + 127;
                pp[32 + 26] = float2int8(p0[B_hstep * 8 + 22] * scale) + 127;
                pp[32 + 27] = float2int8(p0[B_hstep * 8 + 30] * scale) + 127;
                pp[32 + 28] = float2int8(p0[B_hstep * 8 + 7] * scale) + 127;
                pp[32 + 29] = float2int8(p0[B_hstep * 8 + 15] * scale) + 127;
                pp[32 + 30] = float2int8(p0[B_hstep * 8 + 23] * scale) + 127;
                pp[32 + 31] = float2int8(p0[B_hstep * 8 + 31] * scale) + 127;
                pp += 64;
                p0 += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[8] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[10] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[11] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[12] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[13] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[14] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp[16 + 0] = float2int8(p0[B_hstep * 8 + 0] * scale);
                pp[16 + 1] = float2int8(p0[B_hstep * 8 + 8] * scale);
                pp[16 + 2] = float2int8(p0[B_hstep * 8 + 1] * scale);
                pp[16 + 3] = float2int8(p0[B_hstep * 8 + 9] * scale);
                pp[16 + 4] = float2int8(p0[B_hstep * 8 + 2] * scale);
                pp[16 + 5] = float2int8(p0[B_hstep * 8 + 10] * scale);
                pp[16 + 6] = float2int8(p0[B_hstep * 8 + 3] * scale);
                pp[16 + 7] = float2int8(p0[B_hstep * 8 + 11] * scale);
                pp[16 + 8] = float2int8(p0[B_hstep * 8 + 4] * scale);
                pp[16 + 9] = float2int8(p0[B_hstep * 8 + 12] * scale);
                pp[16 + 10] = float2int8(p0[B_hstep * 8 + 5] * scale);
                pp[16 + 11] = float2int8(p0[B_hstep * 8 + 13] * scale);
                pp[16 + 12] = float2int8(p0[B_hstep * 8 + 6] * scale);
                pp[16 + 13] = float2int8(p0[B_hstep * 8 + 14] * scale);
                pp[16 + 14] = float2int8(p0[B_hstep * 8 + 7] * scale);
                pp[16 + 15] = float2int8(p0[B_hstep * 8 + 15] * scale);
                pp += 32;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[B_hstep * 8 + 0] * scale);
                pp[9] = float2int8(p0[B_hstep * 8 + 1] * scale);
                pp[10] = float2int8(p0[B_hstep * 8 + 2] * scale);
                pp[11] = float2int8(p0[B_hstep * 8 + 3] * scale);
                pp[12] = float2int8(p0[B_hstep * 8 + 4] * scale);
                pp[13] = float2int8(p0[B_hstep * 8 + 5] * scale);
                pp[14] = float2int8(p0[B_hstep * 8 + 6] * scale);
                pp[15] = float2int8(p0[B_hstep * 8 + 7] * scale);
                pp += 16;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[4] * scale) + 127;
                pp[2] = float2int8(p0[8] * scale) + 127;
                pp[3] = float2int8(p0[12] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[9] * scale) + 127;
                pp[7] = float2int8(p0[13] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[6] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[14] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[7] * scale) + 127;
                pp[14] = float2int8(p0[11] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;
                pp[16 + 0] = float2int8(p0[B_hstep * 4 + 0] * scale) + 127;
                pp[16 + 1] = float2int8(p0[B_hstep * 4 + 4] * scale) + 127;
                pp[16 + 2] = float2int8(p0[B_hstep * 4 + 8] * scale) + 127;
                pp[16 + 3] = float2int8(p0[B_hstep * 4 + 12] * scale) + 127;
                pp[16 + 4] = float2int8(p0[B_hstep * 4 + 1] * scale) + 127;
                pp[16 + 5] = float2int8(p0[B_hstep * 4 + 5] * scale) + 127;
                pp[16 + 6] = float2int8(p0[B_hstep * 4 + 9] * scale) + 127;
                pp[16 + 7] = float2int8(p0[B_hstep * 4 + 13] * scale) + 127;
                pp[16 + 8] = float2int8(p0[B_hstep * 4 + 2] * scale) + 127;
                pp[16 + 9] = float2int8(p0[B_hstep * 4 + 6] * scale) + 127;
                pp[16 + 10] = float2int8(p0[B_hstep * 4 + 10] * scale) + 127;
                pp[16 + 11] = float2int8(p0[B_hstep * 4 + 14] * scale) + 127;
                pp[16 + 12] = float2int8(p0[B_hstep * 4 + 3] * scale) + 127;
                pp[16 + 13] = float2int8(p0[B_hstep * 4 + 7] * scale) + 127;
                pp[16 + 14] = float2int8(p0[B_hstep * 4 + 11] * scale) + 127;
                pp[16 + 15] = float2int8(p0[B_hstep * 4 + 15] * scale) + 127;

                pp[32 + 0] = float2int8(p0[B_hstep * 8 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[B_hstep * 8 + 4] * scale) + 127;
                pp[32 + 2] = float2int8(p0[B_hstep * 8 + 8] * scale) + 127;
                pp[32 + 3] = float2int8(p0[B_hstep * 8 + 12] * scale) + 127;
                pp[32 + 4] = float2int8(p0[B_hstep * 8 + 1] * scale) + 127;
                pp[32 + 5] = float2int8(p0[B_hstep * 8 + 5] * scale) + 127;
                pp[32 + 6] = float2int8(p0[B_hstep * 8 + 9] * scale) + 127;
                pp[32 + 7] = float2int8(p0[B_hstep * 8 + 13] * scale) + 127;
                pp[32 + 8] = float2int8(p0[B_hstep * 8 + 2] * scale) + 127;
                pp[32 + 9] = float2int8(p0[B_hstep * 8 + 6] * scale) + 127;
                pp[32 + 10] = float2int8(p0[B_hstep * 8 + 10] * scale) + 127;
                pp[32 + 11] = float2int8(p0[B_hstep * 8 + 14] * scale) + 127;
                pp[32 + 12] = float2int8(p0[B_hstep * 8 + 3] * scale) + 127;
                pp[32 + 13] = float2int8(p0[B_hstep * 8 + 7] * scale) + 127;
                pp[32 + 14] = float2int8(p0[B_hstep * 8 + 11] * scale) + 127;
                pp[32 + 15] = float2int8(p0[B_hstep * 8 + 15] * scale) + 127;

                pp[48 + 0] = float2int8(p0[B_hstep * 12 + 0] * scale) + 127;
                pp[48 + 1] = float2int8(p0[B_hstep * 12 + 4] * scale) + 127;
                pp[48 + 2] = float2int8(p0[B_hstep * 12 + 8] * scale) + 127;
                pp[48 + 3] = float2int8(p0[B_hstep * 12 + 12] * scale) + 127;
                pp[48 + 4] = float2int8(p0[B_hstep * 12 + 1] * scale) + 127;
                pp[48 + 5] = float2int8(p0[B_hstep * 12 + 5] * scale) + 127;
                pp[48 + 6] = float2int8(p0[B_hstep * 12 + 9] * scale) + 127;
                pp[48 + 7] = float2int8(p0[B_hstep * 12 + 13] * scale) + 127;
                pp[48 + 8] = float2int8(p0[B_hstep * 12 + 2] * scale) + 127;
                pp[48 + 9] = float2int8(p0[B_hstep * 12 + 6] * scale) + 127;
                pp[48 + 10] = float2int8(p0[B_hstep * 12 + 10] * scale) + 127;
                pp[48 + 11] = float2int8(p0[B_hstep * 12 + 14] * scale) + 127;
                pp[48 + 12] = float2int8(p0[B_hstep * 12 + 3] * scale) + 127;
                pp[48 + 13] = float2int8(p0[B_hstep * 12 + 7] * scale) + 127;
                pp[48 + 14] = float2int8(p0[B_hstep * 12 + 11] * scale) + 127;
                pp[48 + 15] = float2int8(p0[B_hstep * 12 + 15] * scale) + 127;

                pp += 64;
                p0 += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[4] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[6] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[B_hstep * 4 + 0] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 4] * scale);
                pp[10] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[11] = float2int8(p0[B_hstep * 4 + 5] * scale);
                pp[12] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[13] = float2int8(p0[B_hstep * 4 + 6] * scale);
                pp[14] = float2int8(p0[B_hstep * 4 + 3] * scale);
                pp[15] = float2int8(p0[B_hstep * 4 + 7] * scale);

                pp[16 + 0] = float2int8(p0[B_hstep * 8 + 0] * scale);
                pp[16 + 1] = float2int8(p0[B_hstep * 8 + 4] * scale);
                pp[16 + 2] = float2int8(p0[B_hstep * 8 + 1] * scale);
                pp[16 + 3] = float2int8(p0[B_hstep * 8 + 5] * scale);
                pp[16 + 4] = float2int8(p0[B_hstep * 8 + 2] * scale);
                pp[16 + 5] = float2int8(p0[B_hstep * 8 + 6] * scale);
                pp[16 + 6] = float2int8(p0[B_hstep * 8 + 3] * scale);
                pp[16 + 7] = float2int8(p0[B_hstep * 8 + 7] * scale);

                pp[16 + 8] = float2int8(p0[B_hstep * 12 + 0] * scale);
                pp[16 + 9] = float2int8(p0[B_hstep * 12 + 4] * scale);
                pp[16 + 10] = float2int8(p0[B_hstep * 12 + 1] * scale);
                pp[16 + 11] = float2int8(p0[B_hstep * 12 + 5] * scale);
                pp[16 + 12] = float2int8(p0[B_hstep * 12 + 2] * scale);
                pp[16 + 13] = float2int8(p0[B_hstep * 12 + 6] * scale);
                pp[16 + 14] = float2int8(p0[B_hstep * 12 + 3] * scale);
                pp[16 + 15] = float2int8(p0[B_hstep * 12 + 7] * scale);

                pp += 32;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[7] = float2int8(p0[B_hstep * 4 + 3] * scale);
                pp[8] = float2int8(p0[B_hstep * 8] * scale);
                pp[9] = float2int8(p0[B_hstep * 8 + 1] * scale);
                pp[10] = float2int8(p0[B_hstep * 8 + 2] * scale);
                pp[11] = float2int8(p0[B_hstep * 8 + 3] * scale);
                pp[12] = float2int8(p0[B_hstep * 12] * scale);
                pp[13] = float2int8(p0[B_hstep * 12 + 1] * scale);
                pp[14] = float2int8(p0[B_hstep * 12 + 2] * scale);
                pp[15] = float2int8(p0[B_hstep * 12 + 3] * scale);
                pp += 16;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[B_hstep] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[8] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[12] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;
                pp[16] = float2int8(p0[B_hstep * 4] * scale) + 127;
                pp[17] = float2int8(p0[B_hstep * 4 + 1] * scale) + 127;
                pp[18] = float2int8(p0[B_hstep * 4 + 2] * scale) + 127;
                pp[19] = float2int8(p0[B_hstep * 4 + 3] * scale) + 127;
                pp[20] = float2int8(p0[B_hstep * 5] * scale) + 127;
                pp[21] = float2int8(p0[B_hstep * 5 + 1] * scale) + 127;
                pp[22] = float2int8(p0[B_hstep * 5 + 2] * scale) + 127;
                pp[23] = float2int8(p0[B_hstep * 5 + 3] * scale) + 127;
                pp[24] = float2int8(p0[B_hstep * 6] * scale) + 127;
                pp[25] = float2int8(p0[B_hstep * 6 + 1] * scale) + 127;
                pp[26] = float2int8(p0[B_hstep * 6 + 2] * scale) + 127;
                pp[27] = float2int8(p0[B_hstep * 6 + 3] * scale) + 127;
                pp[28] = float2int8(p0[B_hstep * 7] * scale) + 127;
                pp[29] = float2int8(p0[B_hstep * 7 + 1] * scale) + 127;
                pp[30] = float2int8(p0[B_hstep * 7 + 2] * scale) + 127;
                pp[31] = float2int8(p0[B_hstep * 7 + 3] * scale) + 127;

                pp[32 + 0] = float2int8(p0[B_hstep * 8] * scale) + 127;
                pp[32 + 1] = float2int8(p0[B_hstep * 8 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[B_hstep * 8 + 2] * scale) + 127;
                pp[32 + 3] = float2int8(p0[B_hstep * 8 + 3] * scale) + 127;
                pp[32 + 4] = float2int8(p0[B_hstep * 9] * scale) + 127;
                pp[32 + 5] = float2int8(p0[B_hstep * 9 + 1] * scale) + 127;
                pp[32 + 6] = float2int8(p0[B_hstep * 9 + 2] * scale) + 127;
                pp[32 + 7] = float2int8(p0[B_hstep * 9 + 3] * scale) + 127;
                pp[32 + 8] = float2int8(p0[B_hstep * 10] * scale) + 127;
                pp[32 + 9] = float2int8(p0[B_hstep * 10 + 1] * scale) + 127;
                pp[32 + 10] = float2int8(p0[B_hstep * 10 + 2] * scale) + 127;
                pp[32 + 11] = float2int8(p0[B_hstep * 10 + 3] * scale) + 127;
                pp[32 + 12] = float2int8(p0[B_hstep * 11] * scale) + 127;
                pp[32 + 13] = float2int8(p0[B_hstep * 11 + 1] * scale) + 127;
                pp[32 + 14] = float2int8(p0[B_hstep * 11 + 2] * scale) + 127;
                pp[32 + 15] = float2int8(p0[B_hstep * 11 + 3] * scale) + 127;
                pp[32 + 16] = float2int8(p0[B_hstep * 12] * scale) + 127;
                pp[32 + 17] = float2int8(p0[B_hstep * 12 + 1] * scale) + 127;
                pp[32 + 18] = float2int8(p0[B_hstep * 12 + 2] * scale) + 127;
                pp[32 + 19] = float2int8(p0[B_hstep * 12 + 3] * scale) + 127;
                pp[32 + 20] = float2int8(p0[B_hstep * 13] * scale) + 127;
                pp[32 + 21] = float2int8(p0[B_hstep * 13 + 1] * scale) + 127;
                pp[32 + 22] = float2int8(p0[B_hstep * 13 + 2] * scale) + 127;
                pp[32 + 23] = float2int8(p0[B_hstep * 13 + 3] * scale) + 127;
                pp[32 + 24] = float2int8(p0[B_hstep * 14] * scale) + 127;
                pp[32 + 25] = float2int8(p0[B_hstep * 14 + 1] * scale) + 127;
                pp[32 + 26] = float2int8(p0[B_hstep * 14 + 2] * scale) + 127;
                pp[32 + 27] = float2int8(p0[B_hstep * 14 + 3] * scale) + 127;
                pp[32 + 28] = float2int8(p0[B_hstep * 15] * scale) + 127;
                pp[32 + 29] = float2int8(p0[B_hstep * 15 + 1] * scale) + 127;
                pp[32 + 30] = float2int8(p0[B_hstep * 15 + 2] * scale) + 127;
                pp[32 + 31] = float2int8(p0[B_hstep * 15 + 3] * scale) + 127;

                pp += 64;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[B_hstep * 2] * scale);
                pp[5] = float2int8(p0[B_hstep * 2 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 3] * scale);
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);
                pp[8] = float2int8(p0[B_hstep * 4] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[10] = float2int8(p0[B_hstep * 5] * scale);
                pp[11] = float2int8(p0[B_hstep * 5 + 1] * scale);
                pp[12] = float2int8(p0[B_hstep * 6] * scale);
                pp[13] = float2int8(p0[B_hstep * 6 + 1] * scale);
                pp[14] = float2int8(p0[B_hstep * 7] * scale);
                pp[15] = float2int8(p0[B_hstep * 7 + 1] * scale);

                pp[16 + 0] = float2int8(p0[B_hstep * 8] * scale);
                pp[16 + 1] = float2int8(p0[B_hstep * 8 + 1] * scale);
                pp[16 + 2] = float2int8(p0[B_hstep * 9] * scale);
                pp[16 + 3] = float2int8(p0[B_hstep * 9 + 1] * scale);
                pp[16 + 4] = float2int8(p0[B_hstep * 10] * scale);
                pp[16 + 5] = float2int8(p0[B_hstep * 10 + 1] * scale);
                pp[16 + 6] = float2int8(p0[B_hstep * 11] * scale);
                pp[16 + 7] = float2int8(p0[B_hstep * 11 + 1] * scale);
                pp[16 + 8] = float2int8(p0[B_hstep * 12] * scale);
                pp[16 + 9] = float2int8(p0[B_hstep * 12 + 1] * scale);
                pp[16 + 10] = float2int8(p0[B_hstep * 13] * scale);
                pp[16 + 11] = float2int8(p0[B_hstep * 13 + 1] * scale);
                pp[16 + 12] = float2int8(p0[B_hstep * 14] * scale);
                pp[16 + 13] = float2int8(p0[B_hstep * 14 + 1] * scale);
                pp[16 + 14] = float2int8(p0[B_hstep * 15] * scale);
                pp[16 + 15] = float2int8(p0[B_hstep * 15 + 1] * scale);
                pp += 32;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[B_hstep * 2] * scale);
                pp[3] = float2int8(p0[B_hstep * 3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 5] * scale);
                pp[6] = float2int8(p0[B_hstep * 6] * scale);
                pp[7] = float2int8(p0[B_hstep * 7] * scale);
                pp[8] = float2int8(p0[B_hstep * 8] * scale);
                pp[9] = float2int8(p0[B_hstep * 9] * scale);
                pp[10] = float2int8(p0[B_hstep * 10] * scale);
                pp[11] = float2int8(p0[B_hstep * 11] * scale);
                pp[12] = float2int8(p0[B_hstep * 12] * scale);
                pp[13] = float2int8(p0[B_hstep * 13] * scale);
                pp[14] = float2int8(p0[B_hstep * 14] * scale);
                pp[15] = float2int8(p0[B_hstep * 15] * scale);
                pp += 16;
                p0++;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

#if __AVX__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[8] * scale) + 127;
                pp[2] = float2int8(p0[16] * scale) + 127;
                pp[3] = float2int8(p0[24] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[17] * scale) + 127;
                pp[7] = float2int8(p0[25] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[10] * scale) + 127;
                pp[10] = float2int8(p0[18] * scale) + 127;
                pp[11] = float2int8(p0[26] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[11] * scale) + 127;
                pp[14] = float2int8(p0[19] * scale) + 127;
                pp[15] = float2int8(p0[27] * scale) + 127;
                pp[16 + 0] = float2int8(p0[4] * scale) + 127;
                pp[16 + 1] = float2int8(p0[12] * scale) + 127;
                pp[16 + 2] = float2int8(p0[20] * scale) + 127;
                pp[16 + 3] = float2int8(p0[28] * scale) + 127;
                pp[16 + 4] = float2int8(p0[5] * scale) + 127;
                pp[16 + 5] = float2int8(p0[13] * scale) + 127;
                pp[16 + 6] = float2int8(p0[21] * scale) + 127;
                pp[16 + 7] = float2int8(p0[29] * scale) + 127;
                pp[16 + 8] = float2int8(p0[6] * scale) + 127;
                pp[16 + 9] = float2int8(p0[14] * scale) + 127;
                pp[16 + 10] = float2int8(p0[22] * scale) + 127;
                pp[16 + 11] = float2int8(p0[30] * scale) + 127;
                pp[16 + 12] = float2int8(p0[7] * scale) + 127;
                pp[16 + 13] = float2int8(p0[15] * scale) + 127;
                pp[16 + 14] = float2int8(p0[23] * scale) + 127;
                pp[16 + 15] = float2int8(p0[31] * scale) + 127;
                pp += 32;
                p0 += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[8] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[10] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[11] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[12] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[13] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[14] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += 16;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);

                pp += 8;
                p0 += 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[4] * scale) + 127;
                pp[2] = float2int8(p0[8] * scale) + 127;
                pp[3] = float2int8(p0[12] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[9] * scale) + 127;
                pp[7] = float2int8(p0[13] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[6] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[14] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[7] * scale) + 127;
                pp[14] = float2int8(p0[11] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;
                pp[16] = float2int8(p0[B_hstep * 4 + 0] * scale) + 127;
                pp[17] = float2int8(p0[B_hstep * 4 + 4] * scale) + 127;
                pp[18] = float2int8(p0[B_hstep * 4 + 8] * scale) + 127;
                pp[19] = float2int8(p0[B_hstep * 4 + 12] * scale) + 127;
                pp[20] = float2int8(p0[B_hstep * 4 + 1] * scale) + 127;
                pp[21] = float2int8(p0[B_hstep * 4 + 5] * scale) + 127;
                pp[22] = float2int8(p0[B_hstep * 4 + 9] * scale) + 127;
                pp[23] = float2int8(p0[B_hstep * 4 + 13] * scale) + 127;
                pp[24] = float2int8(p0[B_hstep * 4 + 2] * scale) + 127;
                pp[25] = float2int8(p0[B_hstep * 4 + 6] * scale) + 127;
                pp[26] = float2int8(p0[B_hstep * 4 + 10] * scale) + 127;
                pp[27] = float2int8(p0[B_hstep * 4 + 14] * scale) + 127;
                pp[28] = float2int8(p0[B_hstep * 4 + 3] * scale) + 127;
                pp[29] = float2int8(p0[B_hstep * 4 + 7] * scale) + 127;
                pp[30] = float2int8(p0[B_hstep * 4 + 11] * scale) + 127;
                pp[31] = float2int8(p0[B_hstep * 4 + 15] * scale) + 127;
                pp += 32;
                p0 += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[4] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[6] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[B_hstep * 4] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 4] * scale);
                pp[10] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[11] = float2int8(p0[B_hstep * 4 + 5] * scale);
                pp[12] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[13] = float2int8(p0[B_hstep * 4 + 6] * scale);
                pp[14] = float2int8(p0[B_hstep * 4 + 3] * scale);
                pp[15] = float2int8(p0[B_hstep * 4 + 7] * scale);

                pp += 16;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 4 + 2] * scale);
                pp[7] = float2int8(p0[B_hstep * 4 + 3] * scale);

                pp += 8;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[B_hstep] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[8] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[12] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;
                pp[16] = float2int8(p0[B_hstep * 4] * scale) + 127;
                pp[17] = float2int8(p0[B_hstep * 4 + 1] * scale) + 127;
                pp[18] = float2int8(p0[B_hstep * 4 + 2] * scale) + 127;
                pp[19] = float2int8(p0[B_hstep * 4 + 3] * scale) + 127;
                pp[20] = float2int8(p0[B_hstep * 5] * scale) + 127;
                pp[21] = float2int8(p0[B_hstep * 5 + 1] * scale) + 127;
                pp[22] = float2int8(p0[B_hstep * 5 + 2] * scale) + 127;
                pp[23] = float2int8(p0[B_hstep * 5 + 3] * scale) + 127;
                pp[24] = float2int8(p0[B_hstep * 6] * scale) + 127;
                pp[25] = float2int8(p0[B_hstep * 6 + 1] * scale) + 127;
                pp[26] = float2int8(p0[B_hstep * 6 + 2] * scale) + 127;
                pp[27] = float2int8(p0[B_hstep * 6 + 3] * scale) + 127;
                pp[28] = float2int8(p0[B_hstep * 7] * scale) + 127;
                pp[29] = float2int8(p0[B_hstep * 7 + 1] * scale) + 127;
                pp[30] = float2int8(p0[B_hstep * 7 + 2] * scale) + 127;
                pp[31] = float2int8(p0[B_hstep * 7 + 3] * scale) + 127;
                pp += 32;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[B_hstep * 2] * scale);
                pp[5] = float2int8(p0[B_hstep * 2 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 3] * scale);
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);
                pp[8] = float2int8(p0[B_hstep * 4] * scale);
                pp[9] = float2int8(p0[B_hstep * 4 + 1] * scale);
                pp[10] = float2int8(p0[B_hstep * 5] * scale);
                pp[11] = float2int8(p0[B_hstep * 5 + 1] * scale);
                pp[12] = float2int8(p0[B_hstep * 6] * scale);
                pp[13] = float2int8(p0[B_hstep * 6 + 1] * scale);
                pp[14] = float2int8(p0[B_hstep * 7] * scale);
                pp[15] = float2int8(p0[B_hstep * 7 + 1] * scale);

                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[B_hstep * 2] * scale);
                pp[3] = float2int8(p0[B_hstep * 3] * scale);
                pp[4] = float2int8(p0[B_hstep * 4] * scale);
                pp[5] = float2int8(p0[B_hstep * 5] * scale);
                pp[6] = float2int8(p0[B_hstep * 6] * scale);
                pp[7] = float2int8(p0[B_hstep * 7] * scale);

                pp += 8;
                p0++;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[4] * scale) + 127;
                pp[2] = float2int8(p0[8] * scale) + 127;
                pp[3] = float2int8(p0[12] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[9] * scale) + 127;
                pp[7] = float2int8(p0[13] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[6] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[14] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[7] * scale) + 127;
                pp[14] = float2int8(p0[11] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;

                pp += 16;
                p0 += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[4] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[6] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[7] * scale);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[B_hstep] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[8] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[12] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;

                pp += 16;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[B_hstep * 2] * scale);
                pp[5] = float2int8(p0[B_hstep * 2 + 1] * scale);
                pp[6] = float2int8(p0[B_hstep * 3] * scale);
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[B_hstep * 2] * scale);
                pp[3] = float2int8(p0[B_hstep * 3] * scale);

                pp += 4;
                p0++;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[B_hstep] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp += 8;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp += 4;
                p0 += 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp += 2;
                p0++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp += 4;
                p0 += 4;
            }
#endif // __AVX512VNNI__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        transpose_pack_B_tile_fp32_to_int8_avx512vnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        transpose_pack_B_tile_fp32_to_int8_avxvnni(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

    int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
    for (; jj + 15 < max_jj; jj += 16)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2 + 0] * scale) + 127;
                pp[3] = float2int8(p0[2 + 1] * scale) + 127;
                pp[4] = float2int8(p0[16] * scale) + 127;
                pp[5] = float2int8(p0[17] * scale) + 127;
                pp[6] = float2int8(p0[2 + 16] * scale) + 127;
                pp[7] = float2int8(p0[2 + 17] * scale) + 127;
                pp[8] = float2int8(p0[32] * scale) + 127;
                pp[9] = float2int8(p0[33] * scale) + 127;
                pp[10] = float2int8(p0[2 + 32] * scale) + 127;
                pp[11] = float2int8(p0[2 + 33] * scale) + 127;
                pp[12] = float2int8(p0[48] * scale) + 127;
                pp[13] = float2int8(p0[49] * scale) + 127;
                pp[14] = float2int8(p0[2 + 48] * scale) + 127;
                pp[15] = float2int8(p0[2 + 49] * scale) + 127;
                pp[16] = float2int8(p0[64] * scale) + 127;
                pp[17] = float2int8(p0[65] * scale) + 127;
                pp[18] = float2int8(p0[2 + 64] * scale) + 127;
                pp[19] = float2int8(p0[2 + 65] * scale) + 127;
                pp[20] = float2int8(p0[80] * scale) + 127;
                pp[21] = float2int8(p0[81] * scale) + 127;
                pp[22] = float2int8(p0[2 + 80] * scale) + 127;
                pp[23] = float2int8(p0[2 + 81] * scale) + 127;
                pp[24] = float2int8(p0[96] * scale) + 127;
                pp[25] = float2int8(p0[97] * scale) + 127;
                pp[26] = float2int8(p0[2 + 96] * scale) + 127;
                pp[27] = float2int8(p0[2 + 97] * scale) + 127;
                pp[28] = float2int8(p0[112] * scale) + 127;
                pp[29] = float2int8(p0[113] * scale) + 127;
                pp[30] = float2int8(p0[2 + 112] * scale) + 127;
                pp[31] = float2int8(p0[2 + 113] * scale) + 127;

                pp[32 + 0] = float2int8(p0[128 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[128 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[2 + 128 + 0] * scale) + 127;
                pp[32 + 3] = float2int8(p0[2 + 128 + 1] * scale) + 127;
                pp[32 + 4] = float2int8(p0[128 + 16] * scale) + 127;
                pp[32 + 5] = float2int8(p0[128 + 17] * scale) + 127;
                pp[32 + 6] = float2int8(p0[2 + 128 + 16] * scale) + 127;
                pp[32 + 7] = float2int8(p0[2 + 128 + 17] * scale) + 127;
                pp[32 + 8] = float2int8(p0[128 + 32] * scale) + 127;
                pp[32 + 9] = float2int8(p0[128 + 33] * scale) + 127;
                pp[32 + 10] = float2int8(p0[2 + 128 + 32] * scale) + 127;
                pp[32 + 11] = float2int8(p0[2 + 128 + 33] * scale) + 127;
                pp[32 + 12] = float2int8(p0[128 + 48] * scale) + 127;
                pp[32 + 13] = float2int8(p0[128 + 49] * scale) + 127;
                pp[32 + 14] = float2int8(p0[2 + 128 + 48] * scale) + 127;
                pp[32 + 15] = float2int8(p0[2 + 128 + 49] * scale) + 127;
                pp[32 + 16] = float2int8(p0[128 + 64] * scale) + 127;
                pp[32 + 17] = float2int8(p0[128 + 65] * scale) + 127;
                pp[32 + 18] = float2int8(p0[2 + 128 + 64] * scale) + 127;
                pp[32 + 19] = float2int8(p0[2 + 128 + 65] * scale) + 127;
                pp[32 + 20] = float2int8(p0[128 + 80] * scale) + 127;
                pp[32 + 21] = float2int8(p0[128 + 81] * scale) + 127;
                pp[32 + 22] = float2int8(p0[2 + 128 + 80] * scale) + 127;
                pp[32 + 23] = float2int8(p0[2 + 128 + 81] * scale) + 127;
                pp[32 + 24] = float2int8(p0[128 + 96] * scale) + 127;
                pp[32 + 25] = float2int8(p0[128 + 97] * scale) + 127;
                pp[32 + 26] = float2int8(p0[2 + 128 + 96] * scale) + 127;
                pp[32 + 27] = float2int8(p0[2 + 128 + 97] * scale) + 127;
                pp[32 + 28] = float2int8(p0[128 + 112] * scale) + 127;
                pp[32 + 29] = float2int8(p0[128 + 113] * scale) + 127;
                pp[32 + 30] = float2int8(p0[2 + 128 + 112] * scale) + 127;
                pp[32 + 31] = float2int8(p0[2 + 128 + 113] * scale) + 127;

                pp[64 + 0] = float2int8(p0[4 + 0] * scale) + 127;
                pp[64 + 1] = float2int8(p0[4 + 1] * scale) + 127;
                pp[64 + 2] = float2int8(p0[6 + 0] * scale) + 127;
                pp[64 + 3] = float2int8(p0[6 + 1] * scale) + 127;
                pp[64 + 4] = float2int8(p0[4 + 16] * scale) + 127;
                pp[64 + 5] = float2int8(p0[4 + 17] * scale) + 127;
                pp[64 + 6] = float2int8(p0[6 + 16] * scale) + 127;
                pp[64 + 7] = float2int8(p0[6 + 17] * scale) + 127;
                pp[64 + 8] = float2int8(p0[4 + 32] * scale) + 127;
                pp[64 + 9] = float2int8(p0[4 + 33] * scale) + 127;
                pp[64 + 10] = float2int8(p0[6 + 32] * scale) + 127;
                pp[64 + 11] = float2int8(p0[6 + 33] * scale) + 127;
                pp[64 + 12] = float2int8(p0[4 + 48] * scale) + 127;
                pp[64 + 13] = float2int8(p0[4 + 49] * scale) + 127;
                pp[64 + 14] = float2int8(p0[6 + 48] * scale) + 127;
                pp[64 + 15] = float2int8(p0[6 + 49] * scale) + 127;
                pp[64 + 16] = float2int8(p0[4 + 64] * scale) + 127;
                pp[64 + 17] = float2int8(p0[4 + 65] * scale) + 127;
                pp[64 + 18] = float2int8(p0[6 + 64] * scale) + 127;
                pp[64 + 19] = float2int8(p0[6 + 65] * scale) + 127;
                pp[64 + 20] = float2int8(p0[4 + 80] * scale) + 127;
                pp[64 + 21] = float2int8(p0[4 + 81] * scale) + 127;
                pp[64 + 22] = float2int8(p0[6 + 80] * scale) + 127;
                pp[64 + 23] = float2int8(p0[6 + 81] * scale) + 127;
                pp[64 + 24] = float2int8(p0[4 + 96] * scale) + 127;
                pp[64 + 25] = float2int8(p0[4 + 97] * scale) + 127;
                pp[64 + 26] = float2int8(p0[6 + 96] * scale) + 127;
                pp[64 + 27] = float2int8(p0[6 + 97] * scale) + 127;
                pp[64 + 28] = float2int8(p0[4 + 112] * scale) + 127;
                pp[64 + 29] = float2int8(p0[4 + 113] * scale) + 127;
                pp[64 + 30] = float2int8(p0[6 + 112] * scale) + 127;
                pp[64 + 31] = float2int8(p0[6 + 113] * scale) + 127;

                pp[96 + 0] = float2int8(p0[4 + 128 + 0] * scale) + 127;
                pp[96 + 1] = float2int8(p0[4 + 128 + 1] * scale) + 127;
                pp[96 + 2] = float2int8(p0[6 + 128 + 0] * scale) + 127;
                pp[96 + 3] = float2int8(p0[6 + 128 + 1] * scale) + 127;
                pp[96 + 4] = float2int8(p0[4 + 128 + 16] * scale) + 127;
                pp[96 + 5] = float2int8(p0[4 + 128 + 17] * scale) + 127;
                pp[96 + 6] = float2int8(p0[6 + 128 + 16] * scale) + 127;
                pp[96 + 7] = float2int8(p0[6 + 128 + 17] * scale) + 127;
                pp[96 + 8] = float2int8(p0[4 + 128 + 32] * scale) + 127;
                pp[96 + 9] = float2int8(p0[4 + 128 + 33] * scale) + 127;
                pp[96 + 10] = float2int8(p0[6 + 128 + 32] * scale) + 127;
                pp[96 + 11] = float2int8(p0[6 + 128 + 33] * scale) + 127;
                pp[96 + 12] = float2int8(p0[4 + 128 + 48] * scale) + 127;
                pp[96 + 13] = float2int8(p0[4 + 128 + 49] * scale) + 127;
                pp[96 + 14] = float2int8(p0[6 + 128 + 48] * scale) + 127;
                pp[96 + 15] = float2int8(p0[6 + 128 + 49] * scale) + 127;
                pp[96 + 16] = float2int8(p0[4 + 128 + 64] * scale) + 127;
                pp[96 + 17] = float2int8(p0[4 + 128 + 65] * scale) + 127;
                pp[96 + 18] = float2int8(p0[6 + 128 + 64] * scale) + 127;
                pp[96 + 19] = float2int8(p0[6 + 128 + 65] * scale) + 127;
                pp[96 + 20] = float2int8(p0[4 + 128 + 80] * scale) + 127;
                pp[96 + 21] = float2int8(p0[4 + 128 + 81] * scale) + 127;
                pp[96 + 22] = float2int8(p0[6 + 128 + 80] * scale) + 127;
                pp[96 + 23] = float2int8(p0[6 + 128 + 81] * scale) + 127;
                pp[96 + 24] = float2int8(p0[4 + 128 + 96] * scale) + 127;
                pp[96 + 25] = float2int8(p0[4 + 128 + 97] * scale) + 127;
                pp[96 + 26] = float2int8(p0[6 + 128 + 96] * scale) + 127;
                pp[96 + 27] = float2int8(p0[6 + 128 + 97] * scale) + 127;
                pp[96 + 28] = float2int8(p0[4 + 128 + 112] * scale) + 127;
                pp[96 + 29] = float2int8(p0[4 + 128 + 113] * scale) + 127;
                pp[96 + 30] = float2int8(p0[6 + 128 + 112] * scale) + 127;
                pp[96 + 31] = float2int8(p0[6 + 128 + 113] * scale) + 127;

                pp[128 + 0] = float2int8(p0[8 + 0] * scale) + 127;
                pp[128 + 1] = float2int8(p0[8 + 1] * scale) + 127;
                pp[128 + 2] = float2int8(p0[10 + 0] * scale) + 127;
                pp[128 + 3] = float2int8(p0[10 + 1] * scale) + 127;
                pp[128 + 4] = float2int8(p0[8 + 16] * scale) + 127;
                pp[128 + 5] = float2int8(p0[8 + 17] * scale) + 127;
                pp[128 + 6] = float2int8(p0[10 + 16] * scale) + 127;
                pp[128 + 7] = float2int8(p0[10 + 17] * scale) + 127;
                pp[128 + 8] = float2int8(p0[8 + 32] * scale) + 127;
                pp[128 + 9] = float2int8(p0[8 + 33] * scale) + 127;
                pp[128 + 10] = float2int8(p0[10 + 32] * scale) + 127;
                pp[128 + 11] = float2int8(p0[10 + 33] * scale) + 127;
                pp[128 + 12] = float2int8(p0[8 + 48] * scale) + 127;
                pp[128 + 13] = float2int8(p0[8 + 49] * scale) + 127;
                pp[128 + 14] = float2int8(p0[10 + 48] * scale) + 127;
                pp[128 + 15] = float2int8(p0[10 + 49] * scale) + 127;
                pp[128 + 16] = float2int8(p0[8 + 64] * scale) + 127;
                pp[128 + 17] = float2int8(p0[8 + 65] * scale) + 127;
                pp[128 + 18] = float2int8(p0[10 + 64] * scale) + 127;
                pp[128 + 19] = float2int8(p0[10 + 65] * scale) + 127;
                pp[128 + 20] = float2int8(p0[8 + 80] * scale) + 127;
                pp[128 + 21] = float2int8(p0[8 + 81] * scale) + 127;
                pp[128 + 22] = float2int8(p0[10 + 80] * scale) + 127;
                pp[128 + 23] = float2int8(p0[10 + 81] * scale) + 127;
                pp[128 + 24] = float2int8(p0[8 + 96] * scale) + 127;
                pp[128 + 25] = float2int8(p0[8 + 97] * scale) + 127;
                pp[128 + 26] = float2int8(p0[10 + 96] * scale) + 127;
                pp[128 + 27] = float2int8(p0[10 + 97] * scale) + 127;
                pp[128 + 28] = float2int8(p0[8 + 112] * scale) + 127;
                pp[128 + 29] = float2int8(p0[8 + 113] * scale) + 127;
                pp[128 + 30] = float2int8(p0[10 + 112] * scale) + 127;
                pp[128 + 31] = float2int8(p0[10 + 113] * scale) + 127;

                pp[160 + 0] = float2int8(p0[8 + 128 + 0] * scale) + 127;
                pp[160 + 1] = float2int8(p0[8 + 128 + 1] * scale) + 127;
                pp[160 + 2] = float2int8(p0[10 + 128 + 0] * scale) + 127;
                pp[160 + 3] = float2int8(p0[10 + 128 + 1] * scale) + 127;
                pp[160 + 4] = float2int8(p0[8 + 128 + 16] * scale) + 127;
                pp[160 + 5] = float2int8(p0[8 + 128 + 17] * scale) + 127;
                pp[160 + 6] = float2int8(p0[10 + 128 + 16] * scale) + 127;
                pp[160 + 7] = float2int8(p0[10 + 128 + 17] * scale) + 127;
                pp[160 + 8] = float2int8(p0[8 + 128 + 32] * scale) + 127;
                pp[160 + 9] = float2int8(p0[8 + 128 + 33] * scale) + 127;
                pp[160 + 10] = float2int8(p0[10 + 128 + 32] * scale) + 127;
                pp[160 + 11] = float2int8(p0[10 + 128 + 33] * scale) + 127;
                pp[160 + 12] = float2int8(p0[8 + 128 + 48] * scale) + 127;
                pp[160 + 13] = float2int8(p0[8 + 128 + 49] * scale) + 127;
                pp[160 + 14] = float2int8(p0[10 + 128 + 48] * scale) + 127;
                pp[160 + 15] = float2int8(p0[10 + 128 + 49] * scale) + 127;
                pp[160 + 16] = float2int8(p0[8 + 128 + 64] * scale) + 127;
                pp[160 + 17] = float2int8(p0[8 + 128 + 65] * scale) + 127;
                pp[160 + 18] = float2int8(p0[10 + 128 + 64] * scale) + 127;
                pp[160 + 19] = float2int8(p0[10 + 128 + 65] * scale) + 127;
                pp[160 + 20] = float2int8(p0[8 + 128 + 80] * scale) + 127;
                pp[160 + 21] = float2int8(p0[8 + 128 + 81] * scale) + 127;
                pp[160 + 22] = float2int8(p0[10 + 128 + 80] * scale) + 127;
                pp[160 + 23] = float2int8(p0[10 + 128 + 81] * scale) + 127;
                pp[160 + 24] = float2int8(p0[8 + 128 + 96] * scale) + 127;
                pp[160 + 25] = float2int8(p0[8 + 128 + 97] * scale) + 127;
                pp[160 + 26] = float2int8(p0[10 + 128 + 96] * scale) + 127;
                pp[160 + 27] = float2int8(p0[10 + 128 + 97] * scale) + 127;
                pp[160 + 28] = float2int8(p0[8 + 128 + 112] * scale) + 127;
                pp[160 + 29] = float2int8(p0[8 + 128 + 113] * scale) + 127;
                pp[160 + 30] = float2int8(p0[10 + 128 + 112] * scale) + 127;
                pp[160 + 31] = float2int8(p0[10 + 128 + 113] * scale) + 127;

                pp[192 + 0] = float2int8(p0[12 + 0] * scale) + 127;
                pp[192 + 1] = float2int8(p0[12 + 1] * scale) + 127;
                pp[192 + 2] = float2int8(p0[14 + 0] * scale) + 127;
                pp[192 + 3] = float2int8(p0[14 + 1] * scale) + 127;
                pp[192 + 4] = float2int8(p0[12 + 16] * scale) + 127;
                pp[192 + 5] = float2int8(p0[12 + 17] * scale) + 127;
                pp[192 + 6] = float2int8(p0[14 + 16] * scale) + 127;
                pp[192 + 7] = float2int8(p0[14 + 17] * scale) + 127;
                pp[192 + 8] = float2int8(p0[12 + 32] * scale) + 127;
                pp[192 + 9] = float2int8(p0[12 + 33] * scale) + 127;
                pp[192 + 10] = float2int8(p0[14 + 32] * scale) + 127;
                pp[192 + 11] = float2int8(p0[14 + 33] * scale) + 127;
                pp[192 + 12] = float2int8(p0[12 + 48] * scale) + 127;
                pp[192 + 13] = float2int8(p0[12 + 49] * scale) + 127;
                pp[192 + 14] = float2int8(p0[14 + 48] * scale) + 127;
                pp[192 + 15] = float2int8(p0[14 + 49] * scale) + 127;
                pp[192 + 16] = float2int8(p0[12 + 64] * scale) + 127;
                pp[192 + 17] = float2int8(p0[12 + 65] * scale) + 127;
                pp[192 + 18] = float2int8(p0[14 + 64] * scale) + 127;
                pp[192 + 19] = float2int8(p0[14 + 65] * scale) + 127;
                pp[192 + 20] = float2int8(p0[12 + 80] * scale) + 127;
                pp[192 + 21] = float2int8(p0[12 + 81] * scale) + 127;
                pp[192 + 22] = float2int8(p0[14 + 80] * scale) + 127;
                pp[192 + 23] = float2int8(p0[14 + 81] * scale) + 127;
                pp[192 + 24] = float2int8(p0[12 + 96] * scale) + 127;
                pp[192 + 25] = float2int8(p0[12 + 97] * scale) + 127;
                pp[192 + 26] = float2int8(p0[14 + 96] * scale) + 127;
                pp[192 + 27] = float2int8(p0[14 + 97] * scale) + 127;
                pp[192 + 28] = float2int8(p0[12 + 112] * scale) + 127;
                pp[192 + 29] = float2int8(p0[12 + 113] * scale) + 127;
                pp[192 + 30] = float2int8(p0[14 + 112] * scale) + 127;
                pp[192 + 31] = float2int8(p0[14 + 113] * scale) + 127;

                pp[224 + 0] = float2int8(p0[12 + 128 + 0] * scale) + 127;
                pp[224 + 1] = float2int8(p0[12 + 128 + 1] * scale) + 127;
                pp[224 + 2] = float2int8(p0[14 + 128 + 0] * scale) + 127;
                pp[224 + 3] = float2int8(p0[14 + 128 + 1] * scale) + 127;
                pp[224 + 4] = float2int8(p0[12 + 128 + 16] * scale) + 127;
                pp[224 + 5] = float2int8(p0[12 + 128 + 17] * scale) + 127;
                pp[224 + 6] = float2int8(p0[14 + 128 + 16] * scale) + 127;
                pp[224 + 7] = float2int8(p0[14 + 128 + 17] * scale) + 127;
                pp[224 + 8] = float2int8(p0[12 + 128 + 32] * scale) + 127;
                pp[224 + 9] = float2int8(p0[12 + 128 + 33] * scale) + 127;
                pp[224 + 10] = float2int8(p0[14 + 128 + 32] * scale) + 127;
                pp[224 + 11] = float2int8(p0[14 + 128 + 33] * scale) + 127;
                pp[224 + 12] = float2int8(p0[12 + 128 + 48] * scale) + 127;
                pp[224 + 13] = float2int8(p0[12 + 128 + 49] * scale) + 127;
                pp[224 + 14] = float2int8(p0[14 + 128 + 48] * scale) + 127;
                pp[224 + 15] = float2int8(p0[14 + 128 + 49] * scale) + 127;
                pp[224 + 16] = float2int8(p0[12 + 128 + 64] * scale) + 127;
                pp[224 + 17] = float2int8(p0[12 + 128 + 65] * scale) + 127;
                pp[224 + 18] = float2int8(p0[14 + 128 + 64] * scale) + 127;
                pp[224 + 19] = float2int8(p0[14 + 128 + 65] * scale) + 127;
                pp[224 + 20] = float2int8(p0[12 + 128 + 80] * scale) + 127;
                pp[224 + 21] = float2int8(p0[12 + 128 + 81] * scale) + 127;
                pp[224 + 22] = float2int8(p0[14 + 128 + 80] * scale) + 127;
                pp[224 + 23] = float2int8(p0[14 + 128 + 81] * scale) + 127;
                pp[224 + 24] = float2int8(p0[12 + 128 + 96] * scale) + 127;
                pp[224 + 25] = float2int8(p0[12 + 128 + 97] * scale) + 127;
                pp[224 + 26] = float2int8(p0[14 + 128 + 96] * scale) + 127;
                pp[224 + 27] = float2int8(p0[14 + 128 + 97] * scale) + 127;
                pp[224 + 28] = float2int8(p0[12 + 128 + 112] * scale) + 127;
                pp[224 + 29] = float2int8(p0[12 + 128 + 113] * scale) + 127;
                pp[224 + 30] = float2int8(p0[14 + 128 + 112] * scale) + 127;
                pp[224 + 31] = float2int8(p0[14 + 128 + 113] * scale) + 127;

                pp += 256;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[16] * scale);
                pp[3] = float2int8(p0[17] * scale);
                pp[4] = float2int8(p0[32] * scale);
                pp[5] = float2int8(p0[33] * scale);
                pp[6] = float2int8(p0[48] * scale);
                pp[7] = float2int8(p0[49] * scale);
                pp[8] = float2int8(p0[64] * scale);
                pp[9] = float2int8(p0[65] * scale);
                pp[10] = float2int8(p0[80] * scale);
                pp[11] = float2int8(p0[81] * scale);
                pp[12] = float2int8(p0[96] * scale);
                pp[13] = float2int8(p0[97] * scale);
                pp[14] = float2int8(p0[112] * scale);
                pp[15] = float2int8(p0[113] * scale);

                pp[16 + 0] = float2int8(p0[128 + 0] * scale);
                pp[16 + 1] = float2int8(p0[128 + 1] * scale);
                pp[16 + 2] = float2int8(p0[128 + 16] * scale);
                pp[16 + 3] = float2int8(p0[128 + 17] * scale);
                pp[16 + 4] = float2int8(p0[128 + 32] * scale);
                pp[16 + 5] = float2int8(p0[128 + 33] * scale);
                pp[16 + 6] = float2int8(p0[128 + 48] * scale);
                pp[16 + 7] = float2int8(p0[128 + 49] * scale);
                pp[16 + 8] = float2int8(p0[128 + 64] * scale);
                pp[16 + 9] = float2int8(p0[128 + 65] * scale);
                pp[16 + 10] = float2int8(p0[128 + 80] * scale);
                pp[16 + 11] = float2int8(p0[128 + 81] * scale);
                pp[16 + 12] = float2int8(p0[128 + 96] * scale);
                pp[16 + 13] = float2int8(p0[128 + 97] * scale);
                pp[16 + 14] = float2int8(p0[128 + 112] * scale);
                pp[16 + 15] = float2int8(p0[128 + 113] * scale);

                pp[32 + 0] = float2int8(p0[2 + 0] * scale);
                pp[32 + 1] = float2int8(p0[2 + 1] * scale);
                pp[32 + 2] = float2int8(p0[2 + 16] * scale);
                pp[32 + 3] = float2int8(p0[2 + 17] * scale);
                pp[32 + 4] = float2int8(p0[2 + 32] * scale);
                pp[32 + 5] = float2int8(p0[2 + 33] * scale);
                pp[32 + 6] = float2int8(p0[2 + 48] * scale);
                pp[32 + 7] = float2int8(p0[2 + 49] * scale);
                pp[32 + 8] = float2int8(p0[2 + 64] * scale);
                pp[32 + 9] = float2int8(p0[2 + 65] * scale);
                pp[32 + 10] = float2int8(p0[2 + 80] * scale);
                pp[32 + 11] = float2int8(p0[2 + 81] * scale);
                pp[32 + 12] = float2int8(p0[2 + 96] * scale);
                pp[32 + 13] = float2int8(p0[2 + 97] * scale);
                pp[32 + 14] = float2int8(p0[2 + 112] * scale);
                pp[32 + 15] = float2int8(p0[2 + 113] * scale);

                pp[48 + 0] = float2int8(p0[2 + 128 + 0] * scale);
                pp[48 + 1] = float2int8(p0[2 + 128 + 1] * scale);
                pp[48 + 2] = float2int8(p0[2 + 128 + 16] * scale);
                pp[48 + 3] = float2int8(p0[2 + 128 + 17] * scale);
                pp[48 + 4] = float2int8(p0[2 + 128 + 32] * scale);
                pp[48 + 5] = float2int8(p0[2 + 128 + 33] * scale);
                pp[48 + 6] = float2int8(p0[2 + 128 + 48] * scale);
                pp[48 + 7] = float2int8(p0[2 + 128 + 49] * scale);
                pp[48 + 8] = float2int8(p0[2 + 128 + 64] * scale);
                pp[48 + 9] = float2int8(p0[2 + 128 + 65] * scale);
                pp[48 + 10] = float2int8(p0[2 + 128 + 80] * scale);
                pp[48 + 11] = float2int8(p0[2 + 128 + 81] * scale);
                pp[48 + 12] = float2int8(p0[2 + 128 + 96] * scale);
                pp[48 + 13] = float2int8(p0[2 + 128 + 97] * scale);
                pp[48 + 14] = float2int8(p0[2 + 128 + 112] * scale);
                pp[48 + 15] = float2int8(p0[2 + 128 + 113] * scale);

                pp[64 + 0] = float2int8(p0[4 + 0] * scale);
                pp[64 + 1] = float2int8(p0[4 + 1] * scale);
                pp[64 + 2] = float2int8(p0[4 + 16] * scale);
                pp[64 + 3] = float2int8(p0[4 + 17] * scale);
                pp[64 + 4] = float2int8(p0[4 + 32] * scale);
                pp[64 + 5] = float2int8(p0[4 + 33] * scale);
                pp[64 + 6] = float2int8(p0[4 + 48] * scale);
                pp[64 + 7] = float2int8(p0[4 + 49] * scale);
                pp[64 + 8] = float2int8(p0[4 + 64] * scale);
                pp[64 + 9] = float2int8(p0[4 + 65] * scale);
                pp[64 + 10] = float2int8(p0[4 + 80] * scale);
                pp[64 + 11] = float2int8(p0[4 + 81] * scale);
                pp[64 + 12] = float2int8(p0[4 + 96] * scale);
                pp[64 + 13] = float2int8(p0[4 + 97] * scale);
                pp[64 + 14] = float2int8(p0[4 + 112] * scale);
                pp[64 + 15] = float2int8(p0[4 + 113] * scale);

                pp[80 + 0] = float2int8(p0[4 + 128 + 0] * scale);
                pp[80 + 1] = float2int8(p0[4 + 128 + 1] * scale);
                pp[80 + 2] = float2int8(p0[4 + 128 + 16] * scale);
                pp[80 + 3] = float2int8(p0[4 + 128 + 17] * scale);
                pp[80 + 4] = float2int8(p0[4 + 128 + 32] * scale);
                pp[80 + 5] = float2int8(p0[4 + 128 + 33] * scale);
                pp[80 + 6] = float2int8(p0[4 + 128 + 48] * scale);
                pp[80 + 7] = float2int8(p0[4 + 128 + 49] * scale);
                pp[80 + 8] = float2int8(p0[4 + 128 + 64] * scale);
                pp[80 + 9] = float2int8(p0[4 + 128 + 65] * scale);
                pp[80 + 10] = float2int8(p0[4 + 128 + 80] * scale);
                pp[80 + 11] = float2int8(p0[4 + 128 + 81] * scale);
                pp[80 + 12] = float2int8(p0[4 + 128 + 96] * scale);
                pp[80 + 13] = float2int8(p0[4 + 128 + 97] * scale);
                pp[80 + 14] = float2int8(p0[4 + 128 + 112] * scale);
                pp[80 + 15] = float2int8(p0[4 + 128 + 113] * scale);

                pp[96 + 0] = float2int8(p0[6 + 0] * scale);
                pp[96 + 1] = float2int8(p0[6 + 1] * scale);
                pp[96 + 2] = float2int8(p0[6 + 16] * scale);
                pp[96 + 3] = float2int8(p0[6 + 17] * scale);
                pp[96 + 4] = float2int8(p0[6 + 32] * scale);
                pp[96 + 5] = float2int8(p0[6 + 33] * scale);
                pp[96 + 6] = float2int8(p0[6 + 48] * scale);
                pp[96 + 7] = float2int8(p0[6 + 49] * scale);
                pp[96 + 8] = float2int8(p0[6 + 64] * scale);
                pp[96 + 9] = float2int8(p0[6 + 65] * scale);
                pp[96 + 10] = float2int8(p0[6 + 80] * scale);
                pp[96 + 11] = float2int8(p0[6 + 81] * scale);
                pp[96 + 12] = float2int8(p0[6 + 96] * scale);
                pp[96 + 13] = float2int8(p0[6 + 97] * scale);
                pp[96 + 14] = float2int8(p0[6 + 112] * scale);
                pp[96 + 15] = float2int8(p0[6 + 113] * scale);

                pp[112 + 0] = float2int8(p0[6 + 128 + 0] * scale);
                pp[112 + 1] = float2int8(p0[6 + 128 + 1] * scale);
                pp[112 + 2] = float2int8(p0[6 + 128 + 16] * scale);
                pp[112 + 3] = float2int8(p0[6 + 128 + 17] * scale);
                pp[112 + 4] = float2int8(p0[6 + 128 + 32] * scale);
                pp[112 + 5] = float2int8(p0[6 + 128 + 33] * scale);
                pp[112 + 6] = float2int8(p0[6 + 128 + 48] * scale);
                pp[112 + 7] = float2int8(p0[6 + 128 + 49] * scale);
                pp[112 + 8] = float2int8(p0[6 + 128 + 64] * scale);
                pp[112 + 9] = float2int8(p0[6 + 128 + 65] * scale);
                pp[112 + 10] = float2int8(p0[6 + 128 + 80] * scale);
                pp[112 + 11] = float2int8(p0[6 + 128 + 81] * scale);
                pp[112 + 12] = float2int8(p0[6 + 128 + 96] * scale);
                pp[112 + 13] = float2int8(p0[6 + 128 + 97] * scale);
                pp[112 + 14] = float2int8(p0[6 + 128 + 112] * scale);
                pp[112 + 15] = float2int8(p0[6 + 128 + 113] * scale);

                pp[128 + 0] = float2int8(p0[8 + 0] * scale);
                pp[128 + 1] = float2int8(p0[8 + 1] * scale);
                pp[128 + 2] = float2int8(p0[8 + 16] * scale);
                pp[128 + 3] = float2int8(p0[8 + 17] * scale);
                pp[128 + 4] = float2int8(p0[8 + 32] * scale);
                pp[128 + 5] = float2int8(p0[8 + 33] * scale);
                pp[128 + 6] = float2int8(p0[8 + 48] * scale);
                pp[128 + 7] = float2int8(p0[8 + 49] * scale);
                pp[128 + 8] = float2int8(p0[8 + 64] * scale);
                pp[128 + 9] = float2int8(p0[8 + 65] * scale);
                pp[128 + 10] = float2int8(p0[8 + 80] * scale);
                pp[128 + 11] = float2int8(p0[8 + 81] * scale);
                pp[128 + 12] = float2int8(p0[8 + 96] * scale);
                pp[128 + 13] = float2int8(p0[8 + 97] * scale);
                pp[128 + 14] = float2int8(p0[8 + 112] * scale);
                pp[128 + 15] = float2int8(p0[8 + 113] * scale);

                pp[16 + 128 + 0] = float2int8(p0[8 + 128 + 0] * scale);
                pp[16 + 128 + 1] = float2int8(p0[8 + 128 + 1] * scale);
                pp[16 + 128 + 2] = float2int8(p0[8 + 128 + 16] * scale);
                pp[16 + 128 + 3] = float2int8(p0[8 + 128 + 17] * scale);
                pp[16 + 128 + 4] = float2int8(p0[8 + 128 + 32] * scale);
                pp[16 + 128 + 5] = float2int8(p0[8 + 128 + 33] * scale);
                pp[16 + 128 + 6] = float2int8(p0[8 + 128 + 48] * scale);
                pp[16 + 128 + 7] = float2int8(p0[8 + 128 + 49] * scale);
                pp[16 + 128 + 8] = float2int8(p0[8 + 128 + 64] * scale);
                pp[16 + 128 + 9] = float2int8(p0[8 + 128 + 65] * scale);
                pp[16 + 128 + 10] = float2int8(p0[8 + 128 + 80] * scale);
                pp[16 + 128 + 11] = float2int8(p0[8 + 128 + 81] * scale);
                pp[16 + 128 + 12] = float2int8(p0[8 + 128 + 96] * scale);
                pp[16 + 128 + 13] = float2int8(p0[8 + 128 + 97] * scale);
                pp[16 + 128 + 14] = float2int8(p0[8 + 128 + 112] * scale);
                pp[16 + 128 + 15] = float2int8(p0[8 + 128 + 113] * scale);

                pp[32 + 128 + 0] = float2int8(p0[10 + 0] * scale);
                pp[32 + 128 + 1] = float2int8(p0[10 + 1] * scale);
                pp[32 + 128 + 2] = float2int8(p0[10 + 16] * scale);
                pp[32 + 128 + 3] = float2int8(p0[10 + 17] * scale);
                pp[32 + 128 + 4] = float2int8(p0[10 + 32] * scale);
                pp[32 + 128 + 5] = float2int8(p0[10 + 33] * scale);
                pp[32 + 128 + 6] = float2int8(p0[10 + 48] * scale);
                pp[32 + 128 + 7] = float2int8(p0[10 + 49] * scale);
                pp[32 + 128 + 8] = float2int8(p0[10 + 64] * scale);
                pp[32 + 128 + 9] = float2int8(p0[10 + 65] * scale);
                pp[32 + 128 + 10] = float2int8(p0[10 + 80] * scale);
                pp[32 + 128 + 11] = float2int8(p0[10 + 81] * scale);
                pp[32 + 128 + 12] = float2int8(p0[10 + 96] * scale);
                pp[32 + 128 + 13] = float2int8(p0[10 + 97] * scale);
                pp[32 + 128 + 14] = float2int8(p0[10 + 112] * scale);
                pp[32 + 128 + 15] = float2int8(p0[10 + 113] * scale);

                pp[48 + 128 + 0] = float2int8(p0[10 + 128 + 0] * scale);
                pp[48 + 128 + 1] = float2int8(p0[10 + 128 + 1] * scale);
                pp[48 + 128 + 2] = float2int8(p0[10 + 128 + 16] * scale);
                pp[48 + 128 + 3] = float2int8(p0[10 + 128 + 17] * scale);
                pp[48 + 128 + 4] = float2int8(p0[10 + 128 + 32] * scale);
                pp[48 + 128 + 5] = float2int8(p0[10 + 128 + 33] * scale);
                pp[48 + 128 + 6] = float2int8(p0[10 + 128 + 48] * scale);
                pp[48 + 128 + 7] = float2int8(p0[10 + 128 + 49] * scale);
                pp[48 + 128 + 8] = float2int8(p0[10 + 128 + 64] * scale);
                pp[48 + 128 + 9] = float2int8(p0[10 + 128 + 65] * scale);
                pp[48 + 128 + 10] = float2int8(p0[10 + 128 + 80] * scale);
                pp[48 + 128 + 11] = float2int8(p0[10 + 128 + 81] * scale);
                pp[48 + 128 + 12] = float2int8(p0[10 + 128 + 96] * scale);
                pp[48 + 128 + 13] = float2int8(p0[10 + 128 + 97] * scale);
                pp[48 + 128 + 14] = float2int8(p0[10 + 128 + 112] * scale);
                pp[48 + 128 + 15] = float2int8(p0[10 + 128 + 113] * scale);

                pp[64 + 128 + 0] = float2int8(p0[12 + 0] * scale);
                pp[64 + 128 + 1] = float2int8(p0[12 + 1] * scale);
                pp[64 + 128 + 2] = float2int8(p0[12 + 16] * scale);
                pp[64 + 128 + 3] = float2int8(p0[12 + 17] * scale);
                pp[64 + 128 + 4] = float2int8(p0[12 + 32] * scale);
                pp[64 + 128 + 5] = float2int8(p0[12 + 33] * scale);
                pp[64 + 128 + 6] = float2int8(p0[12 + 48] * scale);
                pp[64 + 128 + 7] = float2int8(p0[12 + 49] * scale);
                pp[64 + 128 + 8] = float2int8(p0[12 + 64] * scale);
                pp[64 + 128 + 9] = float2int8(p0[12 + 65] * scale);
                pp[64 + 128 + 10] = float2int8(p0[12 + 80] * scale);
                pp[64 + 128 + 11] = float2int8(p0[12 + 81] * scale);
                pp[64 + 128 + 12] = float2int8(p0[12 + 96] * scale);
                pp[64 + 128 + 13] = float2int8(p0[12 + 97] * scale);
                pp[64 + 128 + 14] = float2int8(p0[12 + 112] * scale);
                pp[64 + 128 + 15] = float2int8(p0[12 + 113] * scale);

                pp[80 + 128 + 0] = float2int8(p0[12 + 128 + 0] * scale);
                pp[80 + 128 + 1] = float2int8(p0[12 + 128 + 1] * scale);
                pp[80 + 128 + 2] = float2int8(p0[12 + 128 + 16] * scale);
                pp[80 + 128 + 3] = float2int8(p0[12 + 128 + 17] * scale);
                pp[80 + 128 + 4] = float2int8(p0[12 + 128 + 32] * scale);
                pp[80 + 128 + 5] = float2int8(p0[12 + 128 + 33] * scale);
                pp[80 + 128 + 6] = float2int8(p0[12 + 128 + 48] * scale);
                pp[80 + 128 + 7] = float2int8(p0[12 + 128 + 49] * scale);
                pp[80 + 128 + 8] = float2int8(p0[12 + 128 + 64] * scale);
                pp[80 + 128 + 9] = float2int8(p0[12 + 128 + 65] * scale);
                pp[80 + 128 + 10] = float2int8(p0[12 + 128 + 80] * scale);
                pp[80 + 128 + 11] = float2int8(p0[12 + 128 + 81] * scale);
                pp[80 + 128 + 12] = float2int8(p0[12 + 128 + 96] * scale);
                pp[80 + 128 + 13] = float2int8(p0[12 + 128 + 97] * scale);
                pp[80 + 128 + 14] = float2int8(p0[12 + 128 + 112] * scale);
                pp[80 + 128 + 15] = float2int8(p0[12 + 128 + 113] * scale);

                pp[96 + 128 + 0] = float2int8(p0[14 + 0] * scale);
                pp[96 + 128 + 1] = float2int8(p0[14 + 1] * scale);
                pp[96 + 128 + 2] = float2int8(p0[14 + 16] * scale);
                pp[96 + 128 + 3] = float2int8(p0[14 + 17] * scale);
                pp[96 + 128 + 4] = float2int8(p0[14 + 32] * scale);
                pp[96 + 128 + 5] = float2int8(p0[14 + 33] * scale);
                pp[96 + 128 + 6] = float2int8(p0[14 + 48] * scale);
                pp[96 + 128 + 7] = float2int8(p0[14 + 49] * scale);
                pp[96 + 128 + 8] = float2int8(p0[14 + 64] * scale);
                pp[96 + 128 + 9] = float2int8(p0[14 + 65] * scale);
                pp[96 + 128 + 10] = float2int8(p0[14 + 80] * scale);
                pp[96 + 128 + 11] = float2int8(p0[14 + 81] * scale);
                pp[96 + 128 + 12] = float2int8(p0[14 + 96] * scale);
                pp[96 + 128 + 13] = float2int8(p0[14 + 97] * scale);
                pp[96 + 128 + 14] = float2int8(p0[14 + 112] * scale);
                pp[96 + 128 + 15] = float2int8(p0[14 + 113] * scale);

                pp[112 + 128 + 0] = float2int8(p0[14 + 128 + 0] * scale);
                pp[112 + 128 + 1] = float2int8(p0[14 + 128 + 1] * scale);
                pp[112 + 128 + 2] = float2int8(p0[14 + 128 + 16] * scale);
                pp[112 + 128 + 3] = float2int8(p0[14 + 128 + 17] * scale);
                pp[112 + 128 + 4] = float2int8(p0[14 + 128 + 32] * scale);
                pp[112 + 128 + 5] = float2int8(p0[14 + 128 + 33] * scale);
                pp[112 + 128 + 6] = float2int8(p0[14 + 128 + 48] * scale);
                pp[112 + 128 + 7] = float2int8(p0[14 + 128 + 49] * scale);
                pp[112 + 128 + 8] = float2int8(p0[14 + 128 + 64] * scale);
                pp[112 + 128 + 9] = float2int8(p0[14 + 128 + 65] * scale);
                pp[112 + 128 + 10] = float2int8(p0[14 + 128 + 80] * scale);
                pp[112 + 128 + 11] = float2int8(p0[14 + 128 + 81] * scale);
                pp[112 + 128 + 12] = float2int8(p0[14 + 128 + 96] * scale);
                pp[112 + 128 + 13] = float2int8(p0[14 + 128 + 97] * scale);
                pp[112 + 128 + 14] = float2int8(p0[14 + 128 + 112] * scale);
                pp[112 + 128 + 15] = float2int8(p0[14 + 128 + 113] * scale);

                pp += 256;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[8] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[10] * scale) + 127;
                pp[7] = float2int8(p0[11] * scale) + 127;
                pp[8] = float2int8(p0[16] * scale) + 127;
                pp[9] = float2int8(p0[17] * scale) + 127;
                pp[10] = float2int8(p0[18] * scale) + 127;
                pp[11] = float2int8(p0[19] * scale) + 127;
                pp[12] = float2int8(p0[24] * scale) + 127;
                pp[13] = float2int8(p0[25] * scale) + 127;
                pp[14] = float2int8(p0[26] * scale) + 127;
                pp[15] = float2int8(p0[27] * scale) + 127;
                pp[16] = float2int8(p0[32] * scale) + 127;
                pp[17] = float2int8(p0[33] * scale) + 127;
                pp[18] = float2int8(p0[34] * scale) + 127;
                pp[19] = float2int8(p0[35] * scale) + 127;
                pp[20] = float2int8(p0[40] * scale) + 127;
                pp[21] = float2int8(p0[41] * scale) + 127;
                pp[22] = float2int8(p0[42] * scale) + 127;
                pp[23] = float2int8(p0[43] * scale) + 127;
                pp[24] = float2int8(p0[48] * scale) + 127;
                pp[25] = float2int8(p0[49] * scale) + 127;
                pp[26] = float2int8(p0[50] * scale) + 127;
                pp[27] = float2int8(p0[51] * scale) + 127;
                pp[28] = float2int8(p0[56] * scale) + 127;
                pp[29] = float2int8(p0[57] * scale) + 127;
                pp[30] = float2int8(p0[58] * scale) + 127;
                pp[31] = float2int8(p0[59] * scale) + 127;

                pp[32 + 0] = float2int8(p0[64 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[64 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[64 + 2] * scale) + 127;
                pp[32 + 3] = float2int8(p0[64 + 3] * scale) + 127;
                pp[32 + 4] = float2int8(p0[64 + 8] * scale) + 127;
                pp[32 + 5] = float2int8(p0[64 + 9] * scale) + 127;
                pp[32 + 6] = float2int8(p0[64 + 10] * scale) + 127;
                pp[32 + 7] = float2int8(p0[64 + 11] * scale) + 127;
                pp[32 + 8] = float2int8(p0[64 + 16] * scale) + 127;
                pp[32 + 9] = float2int8(p0[64 + 17] * scale) + 127;
                pp[32 + 10] = float2int8(p0[64 + 18] * scale) + 127;
                pp[32 + 11] = float2int8(p0[64 + 19] * scale) + 127;
                pp[32 + 12] = float2int8(p0[64 + 24] * scale) + 127;
                pp[32 + 13] = float2int8(p0[64 + 25] * scale) + 127;
                pp[32 + 14] = float2int8(p0[64 + 26] * scale) + 127;
                pp[32 + 15] = float2int8(p0[64 + 27] * scale) + 127;
                pp[32 + 16] = float2int8(p0[64 + 32] * scale) + 127;
                pp[32 + 17] = float2int8(p0[64 + 33] * scale) + 127;
                pp[32 + 18] = float2int8(p0[64 + 34] * scale) + 127;
                pp[32 + 19] = float2int8(p0[64 + 35] * scale) + 127;
                pp[32 + 20] = float2int8(p0[64 + 40] * scale) + 127;
                pp[32 + 21] = float2int8(p0[64 + 41] * scale) + 127;
                pp[32 + 22] = float2int8(p0[64 + 42] * scale) + 127;
                pp[32 + 23] = float2int8(p0[64 + 43] * scale) + 127;
                pp[32 + 24] = float2int8(p0[64 + 48] * scale) + 127;
                pp[32 + 25] = float2int8(p0[64 + 49] * scale) + 127;
                pp[32 + 26] = float2int8(p0[64 + 50] * scale) + 127;
                pp[32 + 27] = float2int8(p0[64 + 51] * scale) + 127;
                pp[32 + 28] = float2int8(p0[64 + 56] * scale) + 127;
                pp[32 + 29] = float2int8(p0[64 + 57] * scale) + 127;
                pp[32 + 30] = float2int8(p0[64 + 58] * scale) + 127;
                pp[32 + 31] = float2int8(p0[64 + 59] * scale) + 127;

                pp[64 + 0] = float2int8(p0[4] * scale) + 127;
                pp[64 + 1] = float2int8(p0[5] * scale) + 127;
                pp[64 + 2] = float2int8(p0[6] * scale) + 127;
                pp[64 + 3] = float2int8(p0[7] * scale) + 127;
                pp[64 + 4] = float2int8(p0[12] * scale) + 127;
                pp[64 + 5] = float2int8(p0[13] * scale) + 127;
                pp[64 + 6] = float2int8(p0[14] * scale) + 127;
                pp[64 + 7] = float2int8(p0[15] * scale) + 127;
                pp[64 + 8] = float2int8(p0[20] * scale) + 127;
                pp[64 + 9] = float2int8(p0[21] * scale) + 127;
                pp[64 + 10] = float2int8(p0[22] * scale) + 127;
                pp[64 + 11] = float2int8(p0[23] * scale) + 127;
                pp[64 + 12] = float2int8(p0[28] * scale) + 127;
                pp[64 + 13] = float2int8(p0[29] * scale) + 127;
                pp[64 + 14] = float2int8(p0[30] * scale) + 127;
                pp[64 + 15] = float2int8(p0[31] * scale) + 127;
                pp[64 + 16] = float2int8(p0[36] * scale) + 127;
                pp[64 + 17] = float2int8(p0[37] * scale) + 127;
                pp[64 + 18] = float2int8(p0[38] * scale) + 127;
                pp[64 + 19] = float2int8(p0[39] * scale) + 127;
                pp[64 + 20] = float2int8(p0[44] * scale) + 127;
                pp[64 + 21] = float2int8(p0[45] * scale) + 127;
                pp[64 + 22] = float2int8(p0[46] * scale) + 127;
                pp[64 + 23] = float2int8(p0[47] * scale) + 127;
                pp[64 + 24] = float2int8(p0[52] * scale) + 127;
                pp[64 + 25] = float2int8(p0[53] * scale) + 127;
                pp[64 + 26] = float2int8(p0[54] * scale) + 127;
                pp[64 + 27] = float2int8(p0[55] * scale) + 127;
                pp[64 + 28] = float2int8(p0[60] * scale) + 127;
                pp[64 + 29] = float2int8(p0[61] * scale) + 127;
                pp[64 + 30] = float2int8(p0[62] * scale) + 127;
                pp[64 + 31] = float2int8(p0[63] * scale) + 127;

                pp[96 + 0] = float2int8(p0[64 + 4] * scale) + 127;
                pp[96 + 1] = float2int8(p0[64 + 5] * scale) + 127;
                pp[96 + 2] = float2int8(p0[64 + 6] * scale) + 127;
                pp[96 + 3] = float2int8(p0[64 + 7] * scale) + 127;
                pp[96 + 4] = float2int8(p0[64 + 12] * scale) + 127;
                pp[96 + 5] = float2int8(p0[64 + 13] * scale) + 127;
                pp[96 + 6] = float2int8(p0[64 + 14] * scale) + 127;
                pp[96 + 7] = float2int8(p0[64 + 15] * scale) + 127;
                pp[96 + 8] = float2int8(p0[64 + 20] * scale) + 127;
                pp[96 + 9] = float2int8(p0[64 + 21] * scale) + 127;
                pp[96 + 10] = float2int8(p0[64 + 22] * scale) + 127;
                pp[96 + 11] = float2int8(p0[64 + 23] * scale) + 127;
                pp[96 + 12] = float2int8(p0[64 + 28] * scale) + 127;
                pp[96 + 13] = float2int8(p0[64 + 29] * scale) + 127;
                pp[96 + 14] = float2int8(p0[64 + 30] * scale) + 127;
                pp[96 + 15] = float2int8(p0[64 + 31] * scale) + 127;
                pp[96 + 16] = float2int8(p0[64 + 36] * scale) + 127;
                pp[96 + 17] = float2int8(p0[64 + 37] * scale) + 127;
                pp[96 + 18] = float2int8(p0[64 + 38] * scale) + 127;
                pp[96 + 19] = float2int8(p0[64 + 39] * scale) + 127;
                pp[96 + 20] = float2int8(p0[64 + 44] * scale) + 127;
                pp[96 + 21] = float2int8(p0[64 + 45] * scale) + 127;
                pp[96 + 22] = float2int8(p0[64 + 46] * scale) + 127;
                pp[96 + 23] = float2int8(p0[64 + 47] * scale) + 127;
                pp[96 + 24] = float2int8(p0[64 + 52] * scale) + 127;
                pp[96 + 25] = float2int8(p0[64 + 53] * scale) + 127;
                pp[96 + 26] = float2int8(p0[64 + 54] * scale) + 127;
                pp[96 + 27] = float2int8(p0[64 + 55] * scale) + 127;
                pp[96 + 28] = float2int8(p0[64 + 60] * scale) + 127;
                pp[96 + 29] = float2int8(p0[64 + 61] * scale) + 127;
                pp[96 + 30] = float2int8(p0[64 + 62] * scale) + 127;
                pp[96 + 31] = float2int8(p0[64 + 63] * scale) + 127;

                pp += 128;
                p0 += B_hstep * 8;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[16] * scale);
                pp[5] = float2int8(p0[17] * scale);
                pp[6] = float2int8(p0[24] * scale);
                pp[7] = float2int8(p0[25] * scale);
                pp[8] = float2int8(p0[32] * scale);
                pp[9] = float2int8(p0[33] * scale);
                pp[10] = float2int8(p0[40] * scale);
                pp[11] = float2int8(p0[41] * scale);
                pp[12] = float2int8(p0[48] * scale);
                pp[13] = float2int8(p0[49] * scale);
                pp[14] = float2int8(p0[56] * scale);
                pp[15] = float2int8(p0[57] * scale);

                pp[16 + 0] = float2int8(p0[64 + 0] * scale);
                pp[16 + 1] = float2int8(p0[64 + 1] * scale);
                pp[16 + 2] = float2int8(p0[64 + 8] * scale);
                pp[16 + 3] = float2int8(p0[64 + 9] * scale);
                pp[16 + 4] = float2int8(p0[64 + 16] * scale);
                pp[16 + 5] = float2int8(p0[64 + 17] * scale);
                pp[16 + 6] = float2int8(p0[64 + 24] * scale);
                pp[16 + 7] = float2int8(p0[64 + 25] * scale);
                pp[16 + 8] = float2int8(p0[64 + 32] * scale);
                pp[16 + 9] = float2int8(p0[64 + 33] * scale);
                pp[16 + 10] = float2int8(p0[64 + 40] * scale);
                pp[16 + 11] = float2int8(p0[64 + 41] * scale);
                pp[16 + 12] = float2int8(p0[64 + 48] * scale);
                pp[16 + 13] = float2int8(p0[64 + 49] * scale);
                pp[16 + 14] = float2int8(p0[64 + 56] * scale);
                pp[16 + 15] = float2int8(p0[64 + 57] * scale);

                pp[32 + 0] = float2int8(p0[2] * scale);
                pp[32 + 1] = float2int8(p0[3] * scale);
                pp[32 + 2] = float2int8(p0[10] * scale);
                pp[32 + 3] = float2int8(p0[11] * scale);
                pp[32 + 4] = float2int8(p0[18] * scale);
                pp[32 + 5] = float2int8(p0[19] * scale);
                pp[32 + 6] = float2int8(p0[26] * scale);
                pp[32 + 7] = float2int8(p0[27] * scale);
                pp[32 + 8] = float2int8(p0[34] * scale);
                pp[32 + 9] = float2int8(p0[35] * scale);
                pp[32 + 10] = float2int8(p0[42] * scale);
                pp[32 + 11] = float2int8(p0[43] * scale);
                pp[32 + 12] = float2int8(p0[50] * scale);
                pp[32 + 13] = float2int8(p0[51] * scale);
                pp[32 + 14] = float2int8(p0[58] * scale);
                pp[32 + 15] = float2int8(p0[59] * scale);

                pp[48 + 0] = float2int8(p0[64 + 2] * scale);
                pp[48 + 1] = float2int8(p0[64 + 3] * scale);
                pp[48 + 2] = float2int8(p0[64 + 10] * scale);
                pp[48 + 3] = float2int8(p0[64 + 11] * scale);
                pp[48 + 4] = float2int8(p0[64 + 18] * scale);
                pp[48 + 5] = float2int8(p0[64 + 19] * scale);
                pp[48 + 6] = float2int8(p0[64 + 26] * scale);
                pp[48 + 7] = float2int8(p0[64 + 27] * scale);
                pp[48 + 8] = float2int8(p0[64 + 34] * scale);
                pp[48 + 9] = float2int8(p0[64 + 35] * scale);
                pp[48 + 10] = float2int8(p0[64 + 42] * scale);
                pp[48 + 11] = float2int8(p0[64 + 43] * scale);
                pp[48 + 12] = float2int8(p0[64 + 50] * scale);
                pp[48 + 13] = float2int8(p0[64 + 51] * scale);
                pp[48 + 14] = float2int8(p0[64 + 58] * scale);
                pp[48 + 15] = float2int8(p0[64 + 59] * scale);

                pp[64 + 0] = float2int8(p0[4] * scale);
                pp[64 + 1] = float2int8(p0[5] * scale);
                pp[64 + 2] = float2int8(p0[12] * scale);
                pp[64 + 3] = float2int8(p0[13] * scale);
                pp[64 + 4] = float2int8(p0[20] * scale);
                pp[64 + 5] = float2int8(p0[21] * scale);
                pp[64 + 6] = float2int8(p0[28] * scale);
                pp[64 + 7] = float2int8(p0[29] * scale);
                pp[64 + 8] = float2int8(p0[36] * scale);
                pp[64 + 9] = float2int8(p0[37] * scale);
                pp[64 + 10] = float2int8(p0[44] * scale);
                pp[64 + 11] = float2int8(p0[45] * scale);
                pp[64 + 12] = float2int8(p0[52] * scale);
                pp[64 + 13] = float2int8(p0[53] * scale);
                pp[64 + 14] = float2int8(p0[60] * scale);
                pp[64 + 15] = float2int8(p0[61] * scale);

                pp[80 + 0] = float2int8(p0[64 + 4] * scale);
                pp[80 + 1] = float2int8(p0[64 + 5] * scale);
                pp[80 + 2] = float2int8(p0[64 + 12] * scale);
                pp[80 + 3] = float2int8(p0[64 + 13] * scale);
                pp[80 + 4] = float2int8(p0[64 + 20] * scale);
                pp[80 + 5] = float2int8(p0[64 + 21] * scale);
                pp[80 + 6] = float2int8(p0[64 + 28] * scale);
                pp[80 + 7] = float2int8(p0[64 + 29] * scale);
                pp[80 + 8] = float2int8(p0[64 + 36] * scale);
                pp[80 + 9] = float2int8(p0[64 + 37] * scale);
                pp[80 + 10] = float2int8(p0[64 + 44] * scale);
                pp[80 + 11] = float2int8(p0[64 + 45] * scale);
                pp[80 + 12] = float2int8(p0[64 + 52] * scale);
                pp[80 + 13] = float2int8(p0[64 + 53] * scale);
                pp[80 + 14] = float2int8(p0[64 + 60] * scale);
                pp[80 + 15] = float2int8(p0[64 + 61] * scale);

                pp[96 + 0] = float2int8(p0[6] * scale);
                pp[96 + 1] = float2int8(p0[7] * scale);
                pp[96 + 2] = float2int8(p0[14] * scale);
                pp[96 + 3] = float2int8(p0[15] * scale);
                pp[96 + 4] = float2int8(p0[22] * scale);
                pp[96 + 5] = float2int8(p0[23] * scale);
                pp[96 + 6] = float2int8(p0[30] * scale);
                pp[96 + 7] = float2int8(p0[31] * scale);
                pp[96 + 8] = float2int8(p0[38] * scale);
                pp[96 + 9] = float2int8(p0[39] * scale);
                pp[96 + 10] = float2int8(p0[46] * scale);
                pp[96 + 11] = float2int8(p0[47] * scale);
                pp[96 + 12] = float2int8(p0[54] * scale);
                pp[96 + 13] = float2int8(p0[55] * scale);
                pp[96 + 14] = float2int8(p0[62] * scale);
                pp[96 + 15] = float2int8(p0[63] * scale);

                pp[112 + 0] = float2int8(p0[64 + 6] * scale);
                pp[112 + 1] = float2int8(p0[64 + 7] * scale);
                pp[112 + 2] = float2int8(p0[64 + 14] * scale);
                pp[112 + 3] = float2int8(p0[64 + 15] * scale);
                pp[112 + 4] = float2int8(p0[64 + 22] * scale);
                pp[112 + 5] = float2int8(p0[64 + 23] * scale);
                pp[112 + 6] = float2int8(p0[64 + 30] * scale);
                pp[112 + 7] = float2int8(p0[64 + 31] * scale);
                pp[112 + 8] = float2int8(p0[64 + 38] * scale);
                pp[112 + 9] = float2int8(p0[64 + 39] * scale);
                pp[112 + 10] = float2int8(p0[64 + 46] * scale);
                pp[112 + 11] = float2int8(p0[64 + 47] * scale);
                pp[112 + 12] = float2int8(p0[64 + 54] * scale);
                pp[112 + 13] = float2int8(p0[64 + 55] * scale);
                pp[112 + 14] = float2int8(p0[64 + 62] * scale);
                pp[112 + 15] = float2int8(p0[64 + 63] * scale);

                pp += 128;
                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[4] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[6] * scale) + 127;
                pp[7] = float2int8(p0[7] * scale) + 127;
                pp[8] = float2int8(p0[8] * scale) + 127;
                pp[9] = float2int8(p0[9] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[11] * scale) + 127;
                pp[12] = float2int8(p0[12] * scale) + 127;
                pp[13] = float2int8(p0[13] * scale) + 127;
                pp[14] = float2int8(p0[14] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;
                pp[16] = float2int8(p0[16] * scale) + 127;
                pp[17] = float2int8(p0[17] * scale) + 127;
                pp[18] = float2int8(p0[18] * scale) + 127;
                pp[19] = float2int8(p0[19] * scale) + 127;
                pp[20] = float2int8(p0[20] * scale) + 127;
                pp[21] = float2int8(p0[21] * scale) + 127;
                pp[22] = float2int8(p0[22] * scale) + 127;
                pp[23] = float2int8(p0[23] * scale) + 127;
                pp[24] = float2int8(p0[24] * scale) + 127;
                pp[25] = float2int8(p0[25] * scale) + 127;
                pp[26] = float2int8(p0[26] * scale) + 127;
                pp[27] = float2int8(p0[27] * scale) + 127;
                pp[28] = float2int8(p0[28] * scale) + 127;
                pp[29] = float2int8(p0[29] * scale) + 127;
                pp[30] = float2int8(p0[30] * scale) + 127;
                pp[31] = float2int8(p0[31] * scale) + 127;

                pp[32 + 0] = float2int8(p0[32 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[32 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[32 + 2] * scale) + 127;
                pp[32 + 3] = float2int8(p0[32 + 3] * scale) + 127;
                pp[32 + 4] = float2int8(p0[32 + 4] * scale) + 127;
                pp[32 + 5] = float2int8(p0[32 + 5] * scale) + 127;
                pp[32 + 6] = float2int8(p0[32 + 6] * scale) + 127;
                pp[32 + 7] = float2int8(p0[32 + 7] * scale) + 127;
                pp[32 + 8] = float2int8(p0[32 + 8] * scale) + 127;
                pp[32 + 9] = float2int8(p0[32 + 9] * scale) + 127;
                pp[32 + 10] = float2int8(p0[32 + 10] * scale) + 127;
                pp[32 + 11] = float2int8(p0[32 + 11] * scale) + 127;
                pp[32 + 12] = float2int8(p0[32 + 12] * scale) + 127;
                pp[32 + 13] = float2int8(p0[32 + 13] * scale) + 127;
                pp[32 + 14] = float2int8(p0[32 + 14] * scale) + 127;
                pp[32 + 15] = float2int8(p0[32 + 15] * scale) + 127;
                pp[32 + 16] = float2int8(p0[32 + 16] * scale) + 127;
                pp[32 + 17] = float2int8(p0[32 + 17] * scale) + 127;
                pp[32 + 18] = float2int8(p0[32 + 18] * scale) + 127;
                pp[32 + 19] = float2int8(p0[32 + 19] * scale) + 127;
                pp[32 + 20] = float2int8(p0[32 + 20] * scale) + 127;
                pp[32 + 21] = float2int8(p0[32 + 21] * scale) + 127;
                pp[32 + 22] = float2int8(p0[32 + 22] * scale) + 127;
                pp[32 + 23] = float2int8(p0[32 + 23] * scale) + 127;
                pp[32 + 24] = float2int8(p0[32 + 24] * scale) + 127;
                pp[32 + 25] = float2int8(p0[32 + 25] * scale) + 127;
                pp[32 + 26] = float2int8(p0[32 + 26] * scale) + 127;
                pp[32 + 27] = float2int8(p0[32 + 27] * scale) + 127;
                pp[32 + 28] = float2int8(p0[32 + 28] * scale) + 127;
                pp[32 + 29] = float2int8(p0[32 + 29] * scale) + 127;
                pp[32 + 30] = float2int8(p0[32 + 30] * scale) + 127;
                pp[32 + 31] = float2int8(p0[32 + 31] * scale) + 127;

                pp += 64;
                p0 += B_hstep * 4;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[16] * scale);
                pp[9] = float2int8(p0[17] * scale);
                pp[10] = float2int8(p0[20] * scale);
                pp[11] = float2int8(p0[21] * scale);
                pp[12] = float2int8(p0[24] * scale);
                pp[13] = float2int8(p0[25] * scale);
                pp[14] = float2int8(p0[28] * scale);
                pp[15] = float2int8(p0[29] * scale);

                pp[16 + 0] = float2int8(p0[32 + 0] * scale);
                pp[16 + 1] = float2int8(p0[32 + 1] * scale);
                pp[16 + 2] = float2int8(p0[32 + 4] * scale);
                pp[16 + 3] = float2int8(p0[32 + 5] * scale);
                pp[16 + 4] = float2int8(p0[32 + 8] * scale);
                pp[16 + 5] = float2int8(p0[32 + 9] * scale);
                pp[16 + 6] = float2int8(p0[32 + 12] * scale);
                pp[16 + 7] = float2int8(p0[32 + 13] * scale);
                pp[16 + 8] = float2int8(p0[32 + 16] * scale);
                pp[16 + 9] = float2int8(p0[32 + 17] * scale);
                pp[16 + 10] = float2int8(p0[32 + 20] * scale);
                pp[16 + 11] = float2int8(p0[32 + 21] * scale);
                pp[16 + 12] = float2int8(p0[32 + 24] * scale);
                pp[16 + 13] = float2int8(p0[32 + 25] * scale);
                pp[16 + 14] = float2int8(p0[32 + 28] * scale);
                pp[16 + 15] = float2int8(p0[32 + 29] * scale);

                pp[32 + 0] = float2int8(p0[2] * scale);
                pp[32 + 1] = float2int8(p0[3] * scale);
                pp[32 + 2] = float2int8(p0[6] * scale);
                pp[32 + 3] = float2int8(p0[7] * scale);
                pp[32 + 4] = float2int8(p0[10] * scale);
                pp[32 + 5] = float2int8(p0[11] * scale);
                pp[32 + 6] = float2int8(p0[14] * scale);
                pp[32 + 7] = float2int8(p0[15] * scale);
                pp[32 + 8] = float2int8(p0[18] * scale);
                pp[32 + 9] = float2int8(p0[19] * scale);
                pp[32 + 10] = float2int8(p0[22] * scale);
                pp[32 + 11] = float2int8(p0[23] * scale);
                pp[32 + 12] = float2int8(p0[26] * scale);
                pp[32 + 13] = float2int8(p0[27] * scale);
                pp[32 + 14] = float2int8(p0[30] * scale);
                pp[32 + 15] = float2int8(p0[31] * scale);

                pp[48 + 0] = float2int8(p0[32 + 2] * scale);
                pp[48 + 1] = float2int8(p0[32 + 3] * scale);
                pp[48 + 2] = float2int8(p0[32 + 6] * scale);
                pp[48 + 3] = float2int8(p0[32 + 7] * scale);
                pp[48 + 4] = float2int8(p0[32 + 10] * scale);
                pp[48 + 5] = float2int8(p0[32 + 11] * scale);
                pp[48 + 6] = float2int8(p0[32 + 14] * scale);
                pp[48 + 7] = float2int8(p0[32 + 15] * scale);
                pp[48 + 8] = float2int8(p0[32 + 18] * scale);
                pp[48 + 9] = float2int8(p0[32 + 19] * scale);
                pp[48 + 10] = float2int8(p0[32 + 22] * scale);
                pp[48 + 11] = float2int8(p0[32 + 23] * scale);
                pp[48 + 12] = float2int8(p0[32 + 26] * scale);
                pp[48 + 13] = float2int8(p0[32 + 27] * scale);
                pp[48 + 14] = float2int8(p0[32 + 30] * scale);
                pp[48 + 15] = float2int8(p0[32 + 31] * scale);

                pp += 64;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[B_hstep] * scale) + 127;
                pp[2] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[3] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;
                pp[16] = float2int8(p0[4] * scale) + 127;
                pp[17] = float2int8(p0[B_hstep + 4] * scale) + 127;
                pp[18] = float2int8(p0[B_hstep * 2 + 4] * scale) + 127;
                pp[19] = float2int8(p0[B_hstep * 3 + 4] * scale) + 127;
                pp[20] = float2int8(p0[5] * scale) + 127;
                pp[21] = float2int8(p0[B_hstep + 5] * scale) + 127;
                pp[22] = float2int8(p0[B_hstep * 2 + 5] * scale) + 127;
                pp[23] = float2int8(p0[B_hstep * 3 + 5] * scale) + 127;
                pp[24] = float2int8(p0[6] * scale) + 127;
                pp[25] = float2int8(p0[B_hstep + 6] * scale) + 127;
                pp[26] = float2int8(p0[B_hstep * 2 + 6] * scale) + 127;
                pp[27] = float2int8(p0[B_hstep * 3 + 6] * scale) + 127;
                pp[28] = float2int8(p0[7] * scale) + 127;
                pp[29] = float2int8(p0[B_hstep + 7] * scale) + 127;
                pp[30] = float2int8(p0[B_hstep * 2 + 7] * scale) + 127;
                pp[31] = float2int8(p0[B_hstep * 3 + 7] * scale) + 127;

                pp[32 + 0] = float2int8(p0[8] * scale) + 127;
                pp[32 + 1] = float2int8(p0[B_hstep + 8] * scale) + 127;
                pp[32 + 2] = float2int8(p0[B_hstep * 2 + 8] * scale) + 127;
                pp[32 + 3] = float2int8(p0[B_hstep * 3 + 8] * scale) + 127;
                pp[32 + 4] = float2int8(p0[9] * scale) + 127;
                pp[32 + 5] = float2int8(p0[B_hstep + 9] * scale) + 127;
                pp[32 + 6] = float2int8(p0[B_hstep * 2 + 9] * scale) + 127;
                pp[32 + 7] = float2int8(p0[B_hstep * 3 + 9] * scale) + 127;
                pp[32 + 8] = float2int8(p0[10] * scale) + 127;
                pp[32 + 9] = float2int8(p0[B_hstep + 10] * scale) + 127;
                pp[32 + 10] = float2int8(p0[B_hstep * 2 + 10] * scale) + 127;
                pp[32 + 11] = float2int8(p0[B_hstep * 3 + 10] * scale) + 127;
                pp[32 + 12] = float2int8(p0[11] * scale) + 127;
                pp[32 + 13] = float2int8(p0[B_hstep + 11] * scale) + 127;
                pp[32 + 14] = float2int8(p0[B_hstep * 2 + 11] * scale) + 127;
                pp[32 + 15] = float2int8(p0[B_hstep * 3 + 11] * scale) + 127;
                pp[32 + 16] = float2int8(p0[12] * scale) + 127;
                pp[32 + 17] = float2int8(p0[B_hstep + 12] * scale) + 127;
                pp[32 + 18] = float2int8(p0[B_hstep * 2 + 12] * scale) + 127;
                pp[32 + 19] = float2int8(p0[B_hstep * 3 + 12] * scale) + 127;
                pp[32 + 20] = float2int8(p0[13] * scale) + 127;
                pp[32 + 21] = float2int8(p0[B_hstep + 13] * scale) + 127;
                pp[32 + 22] = float2int8(p0[B_hstep * 2 + 13] * scale) + 127;
                pp[32 + 23] = float2int8(p0[B_hstep * 3 + 13] * scale) + 127;
                pp[32 + 24] = float2int8(p0[14] * scale) + 127;
                pp[32 + 25] = float2int8(p0[B_hstep + 14] * scale) + 127;
                pp[32 + 26] = float2int8(p0[B_hstep * 2 + 14] * scale) + 127;
                pp[32 + 27] = float2int8(p0[B_hstep * 3 + 14] * scale) + 127;
                pp[32 + 28] = float2int8(p0[15] * scale) + 127;
                pp[32 + 29] = float2int8(p0[B_hstep + 15] * scale) + 127;
                pp[32 + 30] = float2int8(p0[B_hstep * 2 + 15] * scale) + 127;
                pp[32 + 31] = float2int8(p0[B_hstep * 3 + 15] * scale) + 127;
                pp += 64;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[B_hstep + 2] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[B_hstep + 3] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[B_hstep + 4] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[B_hstep + 5] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[B_hstep + 6] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[B_hstep + 7] * scale);

                pp[16 + 0] = float2int8(p0[8] * scale);
                pp[16 + 1] = float2int8(p0[B_hstep + 8] * scale);
                pp[16 + 2] = float2int8(p0[9] * scale);
                pp[16 + 3] = float2int8(p0[B_hstep + 9] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[B_hstep + 10] * scale);
                pp[16 + 6] = float2int8(p0[11] * scale);
                pp[16 + 7] = float2int8(p0[B_hstep + 11] * scale);
                pp[16 + 8] = float2int8(p0[12] * scale);
                pp[16 + 9] = float2int8(p0[B_hstep + 12] * scale);
                pp[16 + 10] = float2int8(p0[13] * scale);
                pp[16 + 11] = float2int8(p0[B_hstep + 13] * scale);
                pp[16 + 12] = float2int8(p0[14] * scale);
                pp[16 + 13] = float2int8(p0[B_hstep + 14] * scale);
                pp[16 + 14] = float2int8(p0[15] * scale);
                pp[16 + 15] = float2int8(p0[B_hstep + 15] * scale);
                pp += 32;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[8] * scale);
                pp[9] = float2int8(p0[9] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[12] * scale);
                pp[13] = float2int8(p0[13] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);
                pp += 16;
                p0 += B_hstep;
            }
        }
    }
#endif // __AVX512F__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2 + 0] * scale) + 127;
                pp[3] = float2int8(p0[2 + 1] * scale) + 127;
                pp[4] = float2int8(p0[16] * scale) + 127;
                pp[5] = float2int8(p0[17] * scale) + 127;
                pp[6] = float2int8(p0[2 + 16] * scale) + 127;
                pp[7] = float2int8(p0[2 + 17] * scale) + 127;
                pp[8] = float2int8(p0[32] * scale) + 127;
                pp[9] = float2int8(p0[33] * scale) + 127;
                pp[10] = float2int8(p0[2 + 32] * scale) + 127;
                pp[11] = float2int8(p0[2 + 33] * scale) + 127;
                pp[12] = float2int8(p0[48] * scale) + 127;
                pp[13] = float2int8(p0[49] * scale) + 127;
                pp[14] = float2int8(p0[2 + 48] * scale) + 127;
                pp[15] = float2int8(p0[2 + 49] * scale) + 127;
                pp[16] = float2int8(p0[64] * scale) + 127;
                pp[17] = float2int8(p0[65] * scale) + 127;
                pp[18] = float2int8(p0[2 + 64] * scale) + 127;
                pp[19] = float2int8(p0[2 + 65] * scale) + 127;
                pp[20] = float2int8(p0[80] * scale) + 127;
                pp[21] = float2int8(p0[81] * scale) + 127;
                pp[22] = float2int8(p0[2 + 80] * scale) + 127;
                pp[23] = float2int8(p0[2 + 81] * scale) + 127;
                pp[24] = float2int8(p0[96] * scale) + 127;
                pp[25] = float2int8(p0[97] * scale) + 127;
                pp[26] = float2int8(p0[2 + 96] * scale) + 127;
                pp[27] = float2int8(p0[2 + 97] * scale) + 127;
                pp[28] = float2int8(p0[112] * scale) + 127;
                pp[29] = float2int8(p0[113] * scale) + 127;
                pp[30] = float2int8(p0[2 + 112] * scale) + 127;
                pp[31] = float2int8(p0[2 + 113] * scale) + 127;

                pp[32 + 0] = float2int8(p0[4 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[4 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[6 + 0] * scale) + 127;
                pp[32 + 3] = float2int8(p0[6 + 1] * scale) + 127;
                pp[32 + 4] = float2int8(p0[4 + 16] * scale) + 127;
                pp[32 + 5] = float2int8(p0[4 + 17] * scale) + 127;
                pp[32 + 6] = float2int8(p0[6 + 16] * scale) + 127;
                pp[32 + 7] = float2int8(p0[6 + 17] * scale) + 127;
                pp[32 + 8] = float2int8(p0[4 + 32] * scale) + 127;
                pp[32 + 9] = float2int8(p0[4 + 33] * scale) + 127;
                pp[32 + 10] = float2int8(p0[6 + 32] * scale) + 127;
                pp[32 + 11] = float2int8(p0[6 + 33] * scale) + 127;
                pp[32 + 12] = float2int8(p0[4 + 48] * scale) + 127;
                pp[32 + 13] = float2int8(p0[4 + 49] * scale) + 127;
                pp[32 + 14] = float2int8(p0[6 + 48] * scale) + 127;
                pp[32 + 15] = float2int8(p0[6 + 49] * scale) + 127;
                pp[32 + 16] = float2int8(p0[4 + 64] * scale) + 127;
                pp[32 + 17] = float2int8(p0[4 + 65] * scale) + 127;
                pp[32 + 18] = float2int8(p0[6 + 64] * scale) + 127;
                pp[32 + 19] = float2int8(p0[6 + 65] * scale) + 127;
                pp[32 + 20] = float2int8(p0[4 + 80] * scale) + 127;
                pp[32 + 21] = float2int8(p0[4 + 81] * scale) + 127;
                pp[32 + 22] = float2int8(p0[6 + 80] * scale) + 127;
                pp[32 + 23] = float2int8(p0[6 + 81] * scale) + 127;
                pp[32 + 24] = float2int8(p0[4 + 96] * scale) + 127;
                pp[32 + 25] = float2int8(p0[4 + 97] * scale) + 127;
                pp[32 + 26] = float2int8(p0[6 + 96] * scale) + 127;
                pp[32 + 27] = float2int8(p0[6 + 97] * scale) + 127;
                pp[32 + 28] = float2int8(p0[4 + 112] * scale) + 127;
                pp[32 + 29] = float2int8(p0[4 + 113] * scale) + 127;
                pp[32 + 30] = float2int8(p0[6 + 112] * scale) + 127;
                pp[32 + 31] = float2int8(p0[6 + 113] * scale) + 127;

                pp[64 + 0] = float2int8(p0[8 + 0] * scale) + 127;
                pp[64 + 1] = float2int8(p0[8 + 1] * scale) + 127;
                pp[64 + 2] = float2int8(p0[10 + 0] * scale) + 127;
                pp[64 + 3] = float2int8(p0[10 + 1] * scale) + 127;
                pp[64 + 4] = float2int8(p0[8 + 16] * scale) + 127;
                pp[64 + 5] = float2int8(p0[8 + 17] * scale) + 127;
                pp[64 + 6] = float2int8(p0[10 + 16] * scale) + 127;
                pp[64 + 7] = float2int8(p0[10 + 17] * scale) + 127;
                pp[64 + 8] = float2int8(p0[8 + 32] * scale) + 127;
                pp[64 + 9] = float2int8(p0[8 + 33] * scale) + 127;
                pp[64 + 10] = float2int8(p0[10 + 32] * scale) + 127;
                pp[64 + 11] = float2int8(p0[10 + 33] * scale) + 127;
                pp[64 + 12] = float2int8(p0[8 + 48] * scale) + 127;
                pp[64 + 13] = float2int8(p0[8 + 49] * scale) + 127;
                pp[64 + 14] = float2int8(p0[10 + 48] * scale) + 127;
                pp[64 + 15] = float2int8(p0[10 + 49] * scale) + 127;
                pp[64 + 16] = float2int8(p0[8 + 64] * scale) + 127;
                pp[64 + 17] = float2int8(p0[8 + 65] * scale) + 127;
                pp[64 + 18] = float2int8(p0[10 + 64] * scale) + 127;
                pp[64 + 19] = float2int8(p0[10 + 65] * scale) + 127;
                pp[64 + 20] = float2int8(p0[8 + 80] * scale) + 127;
                pp[64 + 21] = float2int8(p0[8 + 81] * scale) + 127;
                pp[64 + 22] = float2int8(p0[10 + 80] * scale) + 127;
                pp[64 + 23] = float2int8(p0[10 + 81] * scale) + 127;
                pp[64 + 24] = float2int8(p0[8 + 96] * scale) + 127;
                pp[64 + 25] = float2int8(p0[8 + 97] * scale) + 127;
                pp[64 + 26] = float2int8(p0[10 + 96] * scale) + 127;
                pp[64 + 27] = float2int8(p0[10 + 97] * scale) + 127;
                pp[64 + 28] = float2int8(p0[8 + 112] * scale) + 127;
                pp[64 + 29] = float2int8(p0[8 + 113] * scale) + 127;
                pp[64 + 30] = float2int8(p0[10 + 112] * scale) + 127;
                pp[64 + 31] = float2int8(p0[10 + 113] * scale) + 127;

                pp[96 + 0] = float2int8(p0[12 + 0] * scale) + 127;
                pp[96 + 1] = float2int8(p0[12 + 1] * scale) + 127;
                pp[96 + 2] = float2int8(p0[14 + 0] * scale) + 127;
                pp[96 + 3] = float2int8(p0[14 + 1] * scale) + 127;
                pp[96 + 4] = float2int8(p0[12 + 16] * scale) + 127;
                pp[96 + 5] = float2int8(p0[12 + 17] * scale) + 127;
                pp[96 + 6] = float2int8(p0[14 + 16] * scale) + 127;
                pp[96 + 7] = float2int8(p0[14 + 17] * scale) + 127;
                pp[96 + 8] = float2int8(p0[12 + 32] * scale) + 127;
                pp[96 + 9] = float2int8(p0[12 + 33] * scale) + 127;
                pp[96 + 10] = float2int8(p0[14 + 32] * scale) + 127;
                pp[96 + 11] = float2int8(p0[14 + 33] * scale) + 127;
                pp[96 + 12] = float2int8(p0[12 + 48] * scale) + 127;
                pp[96 + 13] = float2int8(p0[12 + 49] * scale) + 127;
                pp[96 + 14] = float2int8(p0[14 + 48] * scale) + 127;
                pp[96 + 15] = float2int8(p0[14 + 49] * scale) + 127;
                pp[96 + 16] = float2int8(p0[12 + 64] * scale) + 127;
                pp[96 + 17] = float2int8(p0[12 + 65] * scale) + 127;
                pp[96 + 18] = float2int8(p0[14 + 64] * scale) + 127;
                pp[96 + 19] = float2int8(p0[14 + 65] * scale) + 127;
                pp[96 + 20] = float2int8(p0[12 + 80] * scale) + 127;
                pp[96 + 21] = float2int8(p0[12 + 81] * scale) + 127;
                pp[96 + 22] = float2int8(p0[14 + 80] * scale) + 127;
                pp[96 + 23] = float2int8(p0[14 + 81] * scale) + 127;
                pp[96 + 24] = float2int8(p0[12 + 96] * scale) + 127;
                pp[96 + 25] = float2int8(p0[12 + 97] * scale) + 127;
                pp[96 + 26] = float2int8(p0[14 + 96] * scale) + 127;
                pp[96 + 27] = float2int8(p0[14 + 97] * scale) + 127;
                pp[96 + 28] = float2int8(p0[12 + 112] * scale) + 127;
                pp[96 + 29] = float2int8(p0[12 + 113] * scale) + 127;
                pp[96 + 30] = float2int8(p0[14 + 112] * scale) + 127;
                pp[96 + 31] = float2int8(p0[14 + 113] * scale) + 127;

                pp += 128;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[16] * scale);
                pp[3] = float2int8(p0[17] * scale);
                pp[4] = float2int8(p0[32] * scale);
                pp[5] = float2int8(p0[33] * scale);
                pp[6] = float2int8(p0[48] * scale);
                pp[7] = float2int8(p0[49] * scale);
                pp[8] = float2int8(p0[64] * scale);
                pp[9] = float2int8(p0[65] * scale);
                pp[10] = float2int8(p0[80] * scale);
                pp[11] = float2int8(p0[81] * scale);
                pp[12] = float2int8(p0[96] * scale);
                pp[13] = float2int8(p0[97] * scale);
                pp[14] = float2int8(p0[112] * scale);
                pp[15] = float2int8(p0[113] * scale);

                pp[16 + 0] = float2int8(p0[2 + 0] * scale);
                pp[16 + 1] = float2int8(p0[2 + 1] * scale);
                pp[16 + 2] = float2int8(p0[2 + 16] * scale);
                pp[16 + 3] = float2int8(p0[2 + 17] * scale);
                pp[16 + 4] = float2int8(p0[2 + 32] * scale);
                pp[16 + 5] = float2int8(p0[2 + 33] * scale);
                pp[16 + 6] = float2int8(p0[2 + 48] * scale);
                pp[16 + 7] = float2int8(p0[2 + 49] * scale);
                pp[16 + 8] = float2int8(p0[2 + 64] * scale);
                pp[16 + 9] = float2int8(p0[2 + 65] * scale);
                pp[16 + 10] = float2int8(p0[2 + 80] * scale);
                pp[16 + 11] = float2int8(p0[2 + 81] * scale);
                pp[16 + 12] = float2int8(p0[2 + 96] * scale);
                pp[16 + 13] = float2int8(p0[2 + 97] * scale);
                pp[16 + 14] = float2int8(p0[2 + 112] * scale);
                pp[16 + 15] = float2int8(p0[2 + 113] * scale);

                pp[32 + 0] = float2int8(p0[4 + 0] * scale);
                pp[32 + 1] = float2int8(p0[4 + 1] * scale);
                pp[32 + 2] = float2int8(p0[4 + 16] * scale);
                pp[32 + 3] = float2int8(p0[4 + 17] * scale);
                pp[32 + 4] = float2int8(p0[4 + 32] * scale);
                pp[32 + 5] = float2int8(p0[4 + 33] * scale);
                pp[32 + 6] = float2int8(p0[4 + 48] * scale);
                pp[32 + 7] = float2int8(p0[4 + 49] * scale);
                pp[32 + 8] = float2int8(p0[4 + 64] * scale);
                pp[32 + 9] = float2int8(p0[4 + 65] * scale);
                pp[32 + 10] = float2int8(p0[4 + 80] * scale);
                pp[32 + 11] = float2int8(p0[4 + 81] * scale);
                pp[32 + 12] = float2int8(p0[4 + 96] * scale);
                pp[32 + 13] = float2int8(p0[4 + 97] * scale);
                pp[32 + 14] = float2int8(p0[4 + 112] * scale);
                pp[32 + 15] = float2int8(p0[4 + 113] * scale);

                pp[48 + 0] = float2int8(p0[6 + 0] * scale);
                pp[48 + 1] = float2int8(p0[6 + 1] * scale);
                pp[48 + 2] = float2int8(p0[6 + 16] * scale);
                pp[48 + 3] = float2int8(p0[6 + 17] * scale);
                pp[48 + 4] = float2int8(p0[6 + 32] * scale);
                pp[48 + 5] = float2int8(p0[6 + 33] * scale);
                pp[48 + 6] = float2int8(p0[6 + 48] * scale);
                pp[48 + 7] = float2int8(p0[6 + 49] * scale);
                pp[48 + 8] = float2int8(p0[6 + 64] * scale);
                pp[48 + 9] = float2int8(p0[6 + 65] * scale);
                pp[48 + 10] = float2int8(p0[6 + 80] * scale);
                pp[48 + 11] = float2int8(p0[6 + 81] * scale);
                pp[48 + 12] = float2int8(p0[6 + 96] * scale);
                pp[48 + 13] = float2int8(p0[6 + 97] * scale);
                pp[48 + 14] = float2int8(p0[6 + 112] * scale);
                pp[48 + 15] = float2int8(p0[6 + 113] * scale);

                pp[64 + 0] = float2int8(p0[8 + 0] * scale);
                pp[64 + 1] = float2int8(p0[8 + 1] * scale);
                pp[64 + 2] = float2int8(p0[8 + 16] * scale);
                pp[64 + 3] = float2int8(p0[8 + 17] * scale);
                pp[64 + 4] = float2int8(p0[8 + 32] * scale);
                pp[64 + 5] = float2int8(p0[8 + 33] * scale);
                pp[64 + 6] = float2int8(p0[8 + 48] * scale);
                pp[64 + 7] = float2int8(p0[8 + 49] * scale);
                pp[64 + 8] = float2int8(p0[8 + 64] * scale);
                pp[64 + 9] = float2int8(p0[8 + 65] * scale);
                pp[64 + 10] = float2int8(p0[8 + 80] * scale);
                pp[64 + 11] = float2int8(p0[8 + 81] * scale);
                pp[64 + 12] = float2int8(p0[8 + 96] * scale);
                pp[64 + 13] = float2int8(p0[8 + 97] * scale);
                pp[64 + 14] = float2int8(p0[8 + 112] * scale);
                pp[64 + 15] = float2int8(p0[8 + 113] * scale);

                pp[80 + 0] = float2int8(p0[10 + 0] * scale);
                pp[80 + 1] = float2int8(p0[10 + 1] * scale);
                pp[80 + 2] = float2int8(p0[10 + 16] * scale);
                pp[80 + 3] = float2int8(p0[10 + 17] * scale);
                pp[80 + 4] = float2int8(p0[10 + 32] * scale);
                pp[80 + 5] = float2int8(p0[10 + 33] * scale);
                pp[80 + 6] = float2int8(p0[10 + 48] * scale);
                pp[80 + 7] = float2int8(p0[10 + 49] * scale);
                pp[80 + 8] = float2int8(p0[10 + 64] * scale);
                pp[80 + 9] = float2int8(p0[10 + 65] * scale);
                pp[80 + 10] = float2int8(p0[10 + 80] * scale);
                pp[80 + 11] = float2int8(p0[10 + 81] * scale);
                pp[80 + 12] = float2int8(p0[10 + 96] * scale);
                pp[80 + 13] = float2int8(p0[10 + 97] * scale);
                pp[80 + 14] = float2int8(p0[10 + 112] * scale);
                pp[80 + 15] = float2int8(p0[10 + 113] * scale);

                pp[96 + 0] = float2int8(p0[12 + 0] * scale);
                pp[96 + 1] = float2int8(p0[12 + 1] * scale);
                pp[96 + 2] = float2int8(p0[12 + 16] * scale);
                pp[96 + 3] = float2int8(p0[12 + 17] * scale);
                pp[96 + 4] = float2int8(p0[12 + 32] * scale);
                pp[96 + 5] = float2int8(p0[12 + 33] * scale);
                pp[96 + 6] = float2int8(p0[12 + 48] * scale);
                pp[96 + 7] = float2int8(p0[12 + 49] * scale);
                pp[96 + 8] = float2int8(p0[12 + 64] * scale);
                pp[96 + 9] = float2int8(p0[12 + 65] * scale);
                pp[96 + 10] = float2int8(p0[12 + 80] * scale);
                pp[96 + 11] = float2int8(p0[12 + 81] * scale);
                pp[96 + 12] = float2int8(p0[12 + 96] * scale);
                pp[96 + 13] = float2int8(p0[12 + 97] * scale);
                pp[96 + 14] = float2int8(p0[12 + 112] * scale);
                pp[96 + 15] = float2int8(p0[12 + 113] * scale);

                pp[112 + 0] = float2int8(p0[14 + 0] * scale);
                pp[112 + 1] = float2int8(p0[14 + 1] * scale);
                pp[112 + 2] = float2int8(p0[14 + 16] * scale);
                pp[112 + 3] = float2int8(p0[14 + 17] * scale);
                pp[112 + 4] = float2int8(p0[14 + 32] * scale);
                pp[112 + 5] = float2int8(p0[14 + 33] * scale);
                pp[112 + 6] = float2int8(p0[14 + 48] * scale);
                pp[112 + 7] = float2int8(p0[14 + 49] * scale);
                pp[112 + 8] = float2int8(p0[14 + 64] * scale);
                pp[112 + 9] = float2int8(p0[14 + 65] * scale);
                pp[112 + 10] = float2int8(p0[14 + 80] * scale);
                pp[112 + 11] = float2int8(p0[14 + 81] * scale);
                pp[112 + 12] = float2int8(p0[14 + 96] * scale);
                pp[112 + 13] = float2int8(p0[14 + 97] * scale);
                pp[112 + 14] = float2int8(p0[14 + 112] * scale);
                pp[112 + 15] = float2int8(p0[14 + 113] * scale);

                pp += 128;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[8] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[10] * scale) + 127;
                pp[7] = float2int8(p0[11] * scale) + 127;
                pp[8] = float2int8(p0[16] * scale) + 127;
                pp[9] = float2int8(p0[17] * scale) + 127;
                pp[10] = float2int8(p0[18] * scale) + 127;
                pp[11] = float2int8(p0[19] * scale) + 127;
                pp[12] = float2int8(p0[24] * scale) + 127;
                pp[13] = float2int8(p0[25] * scale) + 127;
                pp[14] = float2int8(p0[26] * scale) + 127;
                pp[15] = float2int8(p0[27] * scale) + 127;
                pp[16] = float2int8(p0[32] * scale) + 127;
                pp[17] = float2int8(p0[33] * scale) + 127;
                pp[18] = float2int8(p0[34] * scale) + 127;
                pp[19] = float2int8(p0[35] * scale) + 127;
                pp[20] = float2int8(p0[40] * scale) + 127;
                pp[21] = float2int8(p0[41] * scale) + 127;
                pp[22] = float2int8(p0[42] * scale) + 127;
                pp[23] = float2int8(p0[43] * scale) + 127;
                pp[24] = float2int8(p0[48] * scale) + 127;
                pp[25] = float2int8(p0[49] * scale) + 127;
                pp[26] = float2int8(p0[50] * scale) + 127;
                pp[27] = float2int8(p0[51] * scale) + 127;
                pp[28] = float2int8(p0[56] * scale) + 127;
                pp[29] = float2int8(p0[57] * scale) + 127;
                pp[30] = float2int8(p0[58] * scale) + 127;
                pp[31] = float2int8(p0[59] * scale) + 127;

                pp[32 + 0] = float2int8(p0[4] * scale) + 127;
                pp[32 + 1] = float2int8(p0[5] * scale) + 127;
                pp[32 + 2] = float2int8(p0[6] * scale) + 127;
                pp[32 + 3] = float2int8(p0[7] * scale) + 127;
                pp[32 + 4] = float2int8(p0[12] * scale) + 127;
                pp[32 + 5] = float2int8(p0[13] * scale) + 127;
                pp[32 + 6] = float2int8(p0[14] * scale) + 127;
                pp[32 + 7] = float2int8(p0[15] * scale) + 127;
                pp[32 + 8] = float2int8(p0[20] * scale) + 127;
                pp[32 + 9] = float2int8(p0[21] * scale) + 127;
                pp[32 + 10] = float2int8(p0[22] * scale) + 127;
                pp[32 + 11] = float2int8(p0[23] * scale) + 127;
                pp[32 + 12] = float2int8(p0[28] * scale) + 127;
                pp[32 + 13] = float2int8(p0[29] * scale) + 127;
                pp[32 + 14] = float2int8(p0[30] * scale) + 127;
                pp[32 + 15] = float2int8(p0[31] * scale) + 127;
                pp[32 + 16] = float2int8(p0[36] * scale) + 127;
                pp[32 + 17] = float2int8(p0[37] * scale) + 127;
                pp[32 + 18] = float2int8(p0[38] * scale) + 127;
                pp[32 + 19] = float2int8(p0[39] * scale) + 127;
                pp[32 + 20] = float2int8(p0[44] * scale) + 127;
                pp[32 + 21] = float2int8(p0[45] * scale) + 127;
                pp[32 + 22] = float2int8(p0[46] * scale) + 127;
                pp[32 + 23] = float2int8(p0[47] * scale) + 127;
                pp[32 + 24] = float2int8(p0[52] * scale) + 127;
                pp[32 + 25] = float2int8(p0[53] * scale) + 127;
                pp[32 + 26] = float2int8(p0[54] * scale) + 127;
                pp[32 + 27] = float2int8(p0[55] * scale) + 127;
                pp[32 + 28] = float2int8(p0[60] * scale) + 127;
                pp[32 + 29] = float2int8(p0[61] * scale) + 127;
                pp[32 + 30] = float2int8(p0[62] * scale) + 127;
                pp[32 + 31] = float2int8(p0[63] * scale) + 127;

                pp += 64;
                p0 += B_hstep * 8;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[16] * scale);
                pp[5] = float2int8(p0[17] * scale);
                pp[6] = float2int8(p0[24] * scale);
                pp[7] = float2int8(p0[25] * scale);
                pp[8] = float2int8(p0[32] * scale);
                pp[9] = float2int8(p0[33] * scale);
                pp[10] = float2int8(p0[40] * scale);
                pp[11] = float2int8(p0[41] * scale);
                pp[12] = float2int8(p0[48] * scale);
                pp[13] = float2int8(p0[49] * scale);
                pp[14] = float2int8(p0[56] * scale);
                pp[15] = float2int8(p0[57] * scale);
                pp += 16;

                pp[0] = float2int8(p0[2] * scale);
                pp[1] = float2int8(p0[3] * scale);
                pp[2] = float2int8(p0[10] * scale);
                pp[3] = float2int8(p0[11] * scale);
                pp[4] = float2int8(p0[18] * scale);
                pp[5] = float2int8(p0[19] * scale);
                pp[6] = float2int8(p0[26] * scale);
                pp[7] = float2int8(p0[27] * scale);
                pp[8] = float2int8(p0[34] * scale);
                pp[9] = float2int8(p0[35] * scale);
                pp[10] = float2int8(p0[42] * scale);
                pp[11] = float2int8(p0[43] * scale);
                pp[12] = float2int8(p0[50] * scale);
                pp[13] = float2int8(p0[51] * scale);
                pp[14] = float2int8(p0[58] * scale);
                pp[15] = float2int8(p0[59] * scale);
                pp += 16;

                pp[0] = float2int8(p0[4] * scale);
                pp[1] = float2int8(p0[5] * scale);
                pp[2] = float2int8(p0[12] * scale);
                pp[3] = float2int8(p0[13] * scale);
                pp[4] = float2int8(p0[20] * scale);
                pp[5] = float2int8(p0[21] * scale);
                pp[6] = float2int8(p0[28] * scale);
                pp[7] = float2int8(p0[29] * scale);
                pp[8] = float2int8(p0[36] * scale);
                pp[9] = float2int8(p0[37] * scale);
                pp[10] = float2int8(p0[44] * scale);
                pp[11] = float2int8(p0[45] * scale);
                pp[12] = float2int8(p0[52] * scale);
                pp[13] = float2int8(p0[53] * scale);
                pp[14] = float2int8(p0[60] * scale);
                pp[15] = float2int8(p0[61] * scale);
                pp += 16;

                pp[0] = float2int8(p0[6] * scale);
                pp[1] = float2int8(p0[7] * scale);
                pp[2] = float2int8(p0[14] * scale);
                pp[3] = float2int8(p0[15] * scale);
                pp[4] = float2int8(p0[22] * scale);
                pp[5] = float2int8(p0[23] * scale);
                pp[6] = float2int8(p0[30] * scale);
                pp[7] = float2int8(p0[31] * scale);
                pp[8] = float2int8(p0[38] * scale);
                pp[9] = float2int8(p0[39] * scale);
                pp[10] = float2int8(p0[46] * scale);
                pp[11] = float2int8(p0[47] * scale);
                pp[12] = float2int8(p0[54] * scale);
                pp[13] = float2int8(p0[55] * scale);
                pp[14] = float2int8(p0[62] * scale);
                pp[15] = float2int8(p0[63] * scale);
                pp += 16;

                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[4] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[6] * scale) + 127;
                pp[7] = float2int8(p0[7] * scale) + 127;
                pp[8] = float2int8(p0[8] * scale) + 127;
                pp[9] = float2int8(p0[9] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[11] * scale) + 127;
                pp[12] = float2int8(p0[12] * scale) + 127;
                pp[13] = float2int8(p0[13] * scale) + 127;
                pp[14] = float2int8(p0[14] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;
                pp[16] = float2int8(p0[16] * scale) + 127;
                pp[17] = float2int8(p0[17] * scale) + 127;
                pp[18] = float2int8(p0[18] * scale) + 127;
                pp[19] = float2int8(p0[19] * scale) + 127;
                pp[20] = float2int8(p0[20] * scale) + 127;
                pp[21] = float2int8(p0[21] * scale) + 127;
                pp[22] = float2int8(p0[22] * scale) + 127;
                pp[23] = float2int8(p0[23] * scale) + 127;
                pp[24] = float2int8(p0[24] * scale) + 127;
                pp[25] = float2int8(p0[25] * scale) + 127;
                pp[26] = float2int8(p0[26] * scale) + 127;
                pp[27] = float2int8(p0[27] * scale) + 127;
                pp[28] = float2int8(p0[28] * scale) + 127;
                pp[29] = float2int8(p0[29] * scale) + 127;
                pp[30] = float2int8(p0[30] * scale) + 127;
                pp[31] = float2int8(p0[31] * scale) + 127;
                pp += 32;
                p0 += B_hstep * 4;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[16] * scale);
                pp[9] = float2int8(p0[17] * scale);
                pp[10] = float2int8(p0[20] * scale);
                pp[11] = float2int8(p0[21] * scale);
                pp[12] = float2int8(p0[24] * scale);
                pp[13] = float2int8(p0[25] * scale);
                pp[14] = float2int8(p0[28] * scale);
                pp[15] = float2int8(p0[29] * scale);

                pp[16 + 0] = float2int8(p0[2] * scale);
                pp[16 + 1] = float2int8(p0[3] * scale);
                pp[16 + 2] = float2int8(p0[6] * scale);
                pp[16 + 3] = float2int8(p0[7] * scale);
                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[11] * scale);
                pp[16 + 6] = float2int8(p0[14] * scale);
                pp[16 + 7] = float2int8(p0[15] * scale);
                pp[16 + 8] = float2int8(p0[18] * scale);
                pp[16 + 9] = float2int8(p0[19] * scale);
                pp[16 + 10] = float2int8(p0[22] * scale);
                pp[16 + 11] = float2int8(p0[23] * scale);
                pp[16 + 12] = float2int8(p0[26] * scale);
                pp[16 + 13] = float2int8(p0[27] * scale);
                pp[16 + 14] = float2int8(p0[30] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[B_hstep] * scale) + 127;
                pp[2] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[3] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;
                pp[16] = float2int8(p0[4] * scale) + 127;
                pp[17] = float2int8(p0[B_hstep + 4] * scale) + 127;
                pp[18] = float2int8(p0[B_hstep * 2 + 4] * scale) + 127;
                pp[19] = float2int8(p0[B_hstep * 3 + 4] * scale) + 127;
                pp[20] = float2int8(p0[5] * scale) + 127;
                pp[21] = float2int8(p0[B_hstep + 5] * scale) + 127;
                pp[22] = float2int8(p0[B_hstep * 2 + 5] * scale) + 127;
                pp[23] = float2int8(p0[B_hstep * 3 + 5] * scale) + 127;
                pp[24] = float2int8(p0[6] * scale) + 127;
                pp[25] = float2int8(p0[B_hstep + 6] * scale) + 127;
                pp[26] = float2int8(p0[B_hstep * 2 + 6] * scale) + 127;
                pp[27] = float2int8(p0[B_hstep * 3 + 6] * scale) + 127;
                pp[28] = float2int8(p0[7] * scale) + 127;
                pp[29] = float2int8(p0[B_hstep + 7] * scale) + 127;
                pp[30] = float2int8(p0[B_hstep * 2 + 7] * scale) + 127;
                pp[31] = float2int8(p0[B_hstep * 3 + 7] * scale) + 127;
                pp += 32;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[B_hstep + 2] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[B_hstep + 3] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[B_hstep + 4] * scale);
                pp[10] = float2int8(p0[5] * scale);
                pp[11] = float2int8(p0[B_hstep + 5] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[B_hstep + 6] * scale);
                pp[14] = float2int8(p0[7] * scale);
                pp[15] = float2int8(p0[B_hstep + 7] * scale);

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // defined(__x86_64__) || defined(_M_X64)
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2 + 0] * scale) + 127;
                pp[3] = float2int8(p0[2 + 1] * scale) + 127;
                pp[4] = float2int8(p0[16] * scale) + 127;
                pp[5] = float2int8(p0[17] * scale) + 127;
                pp[6] = float2int8(p0[2 + 16] * scale) + 127;
                pp[7] = float2int8(p0[2 + 17] * scale) + 127;
                pp[8] = float2int8(p0[32] * scale) + 127;
                pp[9] = float2int8(p0[33] * scale) + 127;
                pp[10] = float2int8(p0[2 + 32] * scale) + 127;
                pp[11] = float2int8(p0[2 + 33] * scale) + 127;
                pp[12] = float2int8(p0[48] * scale) + 127;
                pp[13] = float2int8(p0[49] * scale) + 127;
                pp[14] = float2int8(p0[2 + 48] * scale) + 127;
                pp[15] = float2int8(p0[2 + 49] * scale) + 127;

                pp[16 + 0] = float2int8(p0[4 + 0] * scale) + 127;
                pp[16 + 1] = float2int8(p0[4 + 1] * scale) + 127;
                pp[16 + 2] = float2int8(p0[6 + 0] * scale) + 127;
                pp[16 + 3] = float2int8(p0[6 + 1] * scale) + 127;
                pp[16 + 4] = float2int8(p0[4 + 16] * scale) + 127;
                pp[16 + 5] = float2int8(p0[4 + 17] * scale) + 127;
                pp[16 + 6] = float2int8(p0[6 + 16] * scale) + 127;
                pp[16 + 7] = float2int8(p0[6 + 17] * scale) + 127;
                pp[16 + 8] = float2int8(p0[4 + 32] * scale) + 127;
                pp[16 + 9] = float2int8(p0[4 + 33] * scale) + 127;
                pp[16 + 10] = float2int8(p0[6 + 32] * scale) + 127;
                pp[16 + 11] = float2int8(p0[6 + 33] * scale) + 127;
                pp[16 + 12] = float2int8(p0[4 + 48] * scale) + 127;
                pp[16 + 13] = float2int8(p0[4 + 49] * scale) + 127;
                pp[16 + 14] = float2int8(p0[6 + 48] * scale) + 127;
                pp[16 + 15] = float2int8(p0[6 + 49] * scale) + 127;

                pp[32 + 0] = float2int8(p0[8 + 0] * scale) + 127;
                pp[32 + 1] = float2int8(p0[8 + 1] * scale) + 127;
                pp[32 + 2] = float2int8(p0[10 + 0] * scale) + 127;
                pp[32 + 3] = float2int8(p0[10 + 1] * scale) + 127;
                pp[32 + 4] = float2int8(p0[8 + 16] * scale) + 127;
                pp[32 + 5] = float2int8(p0[8 + 17] * scale) + 127;
                pp[32 + 6] = float2int8(p0[10 + 16] * scale) + 127;
                pp[32 + 7] = float2int8(p0[10 + 17] * scale) + 127;
                pp[32 + 8] = float2int8(p0[8 + 32] * scale) + 127;
                pp[32 + 9] = float2int8(p0[8 + 33] * scale) + 127;
                pp[32 + 10] = float2int8(p0[10 + 32] * scale) + 127;
                pp[32 + 11] = float2int8(p0[10 + 33] * scale) + 127;
                pp[32 + 12] = float2int8(p0[8 + 48] * scale) + 127;
                pp[32 + 13] = float2int8(p0[8 + 49] * scale) + 127;
                pp[32 + 14] = float2int8(p0[10 + 48] * scale) + 127;
                pp[32 + 15] = float2int8(p0[10 + 49] * scale) + 127;

                pp[48 + 0] = float2int8(p0[12 + 0] * scale) + 127;
                pp[48 + 1] = float2int8(p0[12 + 1] * scale) + 127;
                pp[48 + 2] = float2int8(p0[14 + 0] * scale) + 127;
                pp[48 + 3] = float2int8(p0[14 + 1] * scale) + 127;
                pp[48 + 4] = float2int8(p0[12 + 16] * scale) + 127;
                pp[48 + 5] = float2int8(p0[12 + 17] * scale) + 127;
                pp[48 + 6] = float2int8(p0[14 + 16] * scale) + 127;
                pp[48 + 7] = float2int8(p0[14 + 17] * scale) + 127;
                pp[48 + 8] = float2int8(p0[12 + 32] * scale) + 127;
                pp[48 + 9] = float2int8(p0[12 + 33] * scale) + 127;
                pp[48 + 10] = float2int8(p0[14 + 32] * scale) + 127;
                pp[48 + 11] = float2int8(p0[14 + 33] * scale) + 127;
                pp[48 + 12] = float2int8(p0[12 + 48] * scale) + 127;
                pp[48 + 13] = float2int8(p0[12 + 49] * scale) + 127;
                pp[48 + 14] = float2int8(p0[14 + 48] * scale) + 127;
                pp[48 + 15] = float2int8(p0[14 + 49] * scale) + 127;

                pp += 64;
                p0 += B_hstep * 16;
            }
#else  // __AVX512VNNI__
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[16] * scale);
                pp[3] = float2int8(p0[17] * scale);
                pp[4] = float2int8(p0[32] * scale);
                pp[5] = float2int8(p0[33] * scale);
                pp[6] = float2int8(p0[48] * scale);
                pp[7] = float2int8(p0[49] * scale);

                pp[8] = float2int8(p0[2 + 0] * scale);
                pp[9] = float2int8(p0[2 + 1] * scale);
                pp[10] = float2int8(p0[2 + 16] * scale);
                pp[11] = float2int8(p0[2 + 17] * scale);
                pp[12] = float2int8(p0[2 + 32] * scale);
                pp[13] = float2int8(p0[2 + 33] * scale);
                pp[14] = float2int8(p0[2 + 48] * scale);
                pp[15] = float2int8(p0[2 + 49] * scale);

                pp[16 + 0] = float2int8(p0[4 + 0] * scale);
                pp[16 + 1] = float2int8(p0[4 + 1] * scale);
                pp[16 + 2] = float2int8(p0[4 + 16] * scale);
                pp[16 + 3] = float2int8(p0[4 + 17] * scale);
                pp[16 + 4] = float2int8(p0[4 + 32] * scale);
                pp[16 + 5] = float2int8(p0[4 + 33] * scale);
                pp[16 + 6] = float2int8(p0[4 + 48] * scale);
                pp[16 + 7] = float2int8(p0[4 + 49] * scale);

                pp[16 + 8] = float2int8(p0[6 + 0] * scale);
                pp[16 + 9] = float2int8(p0[6 + 1] * scale);
                pp[16 + 10] = float2int8(p0[6 + 16] * scale);
                pp[16 + 11] = float2int8(p0[6 + 17] * scale);
                pp[16 + 12] = float2int8(p0[6 + 32] * scale);
                pp[16 + 13] = float2int8(p0[6 + 33] * scale);
                pp[16 + 14] = float2int8(p0[6 + 48] * scale);
                pp[16 + 15] = float2int8(p0[6 + 49] * scale);

                pp[32 + 0] = float2int8(p0[8 + 0] * scale);
                pp[32 + 1] = float2int8(p0[8 + 1] * scale);
                pp[32 + 2] = float2int8(p0[8 + 16] * scale);
                pp[32 + 3] = float2int8(p0[8 + 17] * scale);
                pp[32 + 4] = float2int8(p0[8 + 32] * scale);
                pp[32 + 5] = float2int8(p0[8 + 33] * scale);
                pp[32 + 6] = float2int8(p0[8 + 48] * scale);
                pp[32 + 7] = float2int8(p0[8 + 49] * scale);

                pp[32 + 8] = float2int8(p0[10 + 0] * scale);
                pp[32 + 9] = float2int8(p0[10 + 1] * scale);
                pp[32 + 10] = float2int8(p0[10 + 16] * scale);
                pp[32 + 11] = float2int8(p0[10 + 17] * scale);
                pp[32 + 12] = float2int8(p0[10 + 32] * scale);
                pp[32 + 13] = float2int8(p0[10 + 33] * scale);
                pp[32 + 14] = float2int8(p0[10 + 48] * scale);
                pp[32 + 15] = float2int8(p0[10 + 49] * scale);

                pp[48 + 0] = float2int8(p0[12 + 0] * scale);
                pp[48 + 1] = float2int8(p0[12 + 1] * scale);
                pp[48 + 2] = float2int8(p0[12 + 16] * scale);
                pp[48 + 3] = float2int8(p0[12 + 17] * scale);
                pp[48 + 4] = float2int8(p0[12 + 32] * scale);
                pp[48 + 5] = float2int8(p0[12 + 33] * scale);
                pp[48 + 6] = float2int8(p0[12 + 48] * scale);
                pp[48 + 7] = float2int8(p0[12 + 49] * scale);

                pp[48 + 8] = float2int8(p0[14 + 0] * scale);
                pp[48 + 9] = float2int8(p0[14 + 1] * scale);
                pp[48 + 10] = float2int8(p0[14 + 16] * scale);
                pp[48 + 11] = float2int8(p0[14 + 17] * scale);
                pp[48 + 12] = float2int8(p0[14 + 32] * scale);
                pp[48 + 13] = float2int8(p0[14 + 33] * scale);
                pp[48 + 14] = float2int8(p0[14 + 48] * scale);
                pp[48 + 15] = float2int8(p0[14 + 49] * scale);

                pp += 64;
                p0 += B_hstep * 16;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[8] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[10] * scale) + 127;
                pp[7] = float2int8(p0[11] * scale) + 127;
                pp[8] = float2int8(p0[16] * scale) + 127;
                pp[9] = float2int8(p0[17] * scale) + 127;
                pp[10] = float2int8(p0[18] * scale) + 127;
                pp[11] = float2int8(p0[19] * scale) + 127;
                pp[12] = float2int8(p0[24] * scale) + 127;
                pp[13] = float2int8(p0[25] * scale) + 127;
                pp[14] = float2int8(p0[26] * scale) + 127;
                pp[15] = float2int8(p0[27] * scale) + 127;

                pp[16 + 0] = float2int8(p0[4] * scale) + 127;
                pp[16 + 1] = float2int8(p0[5] * scale) + 127;
                pp[16 + 2] = float2int8(p0[6] * scale) + 127;
                pp[16 + 3] = float2int8(p0[7] * scale) + 127;
                pp[16 + 4] = float2int8(p0[12] * scale) + 127;
                pp[16 + 5] = float2int8(p0[13] * scale) + 127;
                pp[16 + 6] = float2int8(p0[14] * scale) + 127;
                pp[16 + 7] = float2int8(p0[15] * scale) + 127;
                pp[16 + 8] = float2int8(p0[20] * scale) + 127;
                pp[16 + 9] = float2int8(p0[21] * scale) + 127;
                pp[16 + 10] = float2int8(p0[22] * scale) + 127;
                pp[16 + 11] = float2int8(p0[23] * scale) + 127;
                pp[16 + 12] = float2int8(p0[28] * scale) + 127;
                pp[16 + 13] = float2int8(p0[29] * scale) + 127;
                pp[16 + 14] = float2int8(p0[30] * scale) + 127;
                pp[16 + 15] = float2int8(p0[31] * scale) + 127;

                pp += 32;
                p0 += B_hstep * 8;
            }
#else  // __AVX512VNNI__
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[16] * scale);
                pp[5] = float2int8(p0[17] * scale);
                pp[6] = float2int8(p0[24] * scale);
                pp[7] = float2int8(p0[25] * scale);

                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[18] * scale);
                pp[13] = float2int8(p0[19] * scale);
                pp[14] = float2int8(p0[26] * scale);
                pp[15] = float2int8(p0[27] * scale);

                pp[16 + 0] = float2int8(p0[4] * scale);
                pp[16 + 1] = float2int8(p0[5] * scale);
                pp[16 + 2] = float2int8(p0[12] * scale);
                pp[16 + 3] = float2int8(p0[13] * scale);
                pp[16 + 4] = float2int8(p0[20] * scale);
                pp[16 + 5] = float2int8(p0[21] * scale);
                pp[16 + 6] = float2int8(p0[28] * scale);
                pp[16 + 7] = float2int8(p0[29] * scale);

                pp[16 + 8] = float2int8(p0[6] * scale);
                pp[16 + 9] = float2int8(p0[7] * scale);
                pp[16 + 10] = float2int8(p0[14] * scale);
                pp[16 + 11] = float2int8(p0[15] * scale);
                pp[16 + 12] = float2int8(p0[22] * scale);
                pp[16 + 13] = float2int8(p0[23] * scale);
                pp[16 + 14] = float2int8(p0[30] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);

                pp += 32;
                p0 += B_hstep * 8;
            }
#endif // __AVX512VNNI__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[4] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[6] * scale) + 127;
                pp[7] = float2int8(p0[7] * scale) + 127;
                pp[8] = float2int8(p0[8] * scale) + 127;
                pp[9] = float2int8(p0[9] * scale) + 127;
                pp[10] = float2int8(p0[10] * scale) + 127;
                pp[11] = float2int8(p0[11] * scale) + 127;
                pp[12] = float2int8(p0[12] * scale) + 127;
                pp[13] = float2int8(p0[13] * scale) + 127;
                pp[14] = float2int8(p0[14] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;

                pp += 16;
                p0 += B_hstep * 4;
            }
#else  // __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[8] * scale);
                pp[5] = float2int8(p0[9] * scale);
                pp[6] = float2int8(p0[12] * scale);
                pp[7] = float2int8(p0[13] * scale);
                pp[8] = float2int8(p0[2] * scale);
                pp[9] = float2int8(p0[3] * scale);
                pp[10] = float2int8(p0[6] * scale);
                pp[11] = float2int8(p0[7] * scale);
                pp[12] = float2int8(p0[10] * scale);
                pp[13] = float2int8(p0[11] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);

                pp += 16;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
        }
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[B_hstep] * scale) + 127;
                pp[2] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[3] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp[8] = float2int8(p0[2] * scale) + 127;
                pp[9] = float2int8(p0[B_hstep + 2] * scale) + 127;
                pp[10] = float2int8(p0[B_hstep * 2 + 2] * scale) + 127;
                pp[11] = float2int8(p0[B_hstep * 3 + 2] * scale) + 127;
                pp[12] = float2int8(p0[3] * scale) + 127;
                pp[13] = float2int8(p0[B_hstep + 3] * scale) + 127;
                pp[14] = float2int8(p0[B_hstep * 2 + 3] * scale) + 127;
                pp[15] = float2int8(p0[B_hstep * 3 + 3] * scale) + 127;

                pp += 16;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[B_hstep + 2] * scale);
                pp[6] = float2int8(p0[3] * scale);
                pp[7] = float2int8(p0[B_hstep + 3] * scale);

                pp += 8;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __SSE2__
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[16] * scale) + 127;
                pp[5] = float2int8(p0[17] * scale) + 127;
                pp[6] = float2int8(p0[18] * scale) + 127;
                pp[7] = float2int8(p0[19] * scale) + 127;

                pp[8] = float2int8(p0[4] * scale) + 127;
                pp[9] = float2int8(p0[5] * scale) + 127;
                pp[10] = float2int8(p0[6] * scale) + 127;
                pp[11] = float2int8(p0[7] * scale) + 127;
                pp[12] = float2int8(p0[20] * scale) + 127;
                pp[13] = float2int8(p0[21] * scale) + 127;
                pp[14] = float2int8(p0[22] * scale) + 127;
                pp[15] = float2int8(p0[23] * scale) + 127;

                pp[16 + 0] = float2int8(p0[8] * scale) + 127;
                pp[16 + 1] = float2int8(p0[9] * scale) + 127;
                pp[16 + 2] = float2int8(p0[10] * scale) + 127;
                pp[16 + 3] = float2int8(p0[11] * scale) + 127;
                pp[16 + 4] = float2int8(p0[24] * scale) + 127;
                pp[16 + 5] = float2int8(p0[25] * scale) + 127;
                pp[16 + 6] = float2int8(p0[26] * scale) + 127;
                pp[16 + 7] = float2int8(p0[27] * scale) + 127;

                pp[16 + 8] = float2int8(p0[12] * scale) + 127;
                pp[16 + 9] = float2int8(p0[13] * scale) + 127;
                pp[16 + 10] = float2int8(p0[14] * scale) + 127;
                pp[16 + 11] = float2int8(p0[15] * scale) + 127;
                pp[16 + 12] = float2int8(p0[28] * scale) + 127;
                pp[16 + 13] = float2int8(p0[29] * scale) + 127;
                pp[16 + 14] = float2int8(p0[30] * scale) + 127;
                pp[16 + 15] = float2int8(p0[31] * scale) + 127;
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[16] * scale);
                pp[3] = float2int8(p0[17] * scale);

                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[18] * scale);
                pp[7] = float2int8(p0[19] * scale);

                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[5] * scale);
                pp[10] = float2int8(p0[20] * scale);
                pp[11] = float2int8(p0[21] * scale);

                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[7] * scale);
                pp[14] = float2int8(p0[22] * scale);
                pp[15] = float2int8(p0[23] * scale);

                pp[16 + 0] = float2int8(p0[8] * scale);
                pp[16 + 1] = float2int8(p0[9] * scale);
                pp[16 + 2] = float2int8(p0[24] * scale);
                pp[16 + 3] = float2int8(p0[25] * scale);

                pp[16 + 4] = float2int8(p0[10] * scale);
                pp[16 + 5] = float2int8(p0[11] * scale);
                pp[16 + 6] = float2int8(p0[26] * scale);
                pp[16 + 7] = float2int8(p0[27] * scale);

                pp[16 + 8] = float2int8(p0[12] * scale);
                pp[16 + 9] = float2int8(p0[13] * scale);
                pp[16 + 10] = float2int8(p0[28] * scale);
                pp[16 + 11] = float2int8(p0[29] * scale);

                pp[16 + 12] = float2int8(p0[14] * scale);
                pp[16 + 13] = float2int8(p0[15] * scale);
                pp[16 + 14] = float2int8(p0[30] * scale);
                pp[16 + 15] = float2int8(p0[31] * scale);
#endif // __AVX512VNNI__
                pp += 32;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[8] * scale) + 127;
                pp[5] = float2int8(p0[9] * scale) + 127;
                pp[6] = float2int8(p0[10] * scale) + 127;
                pp[7] = float2int8(p0[11] * scale) + 127;
                pp[8] = float2int8(p0[4] * scale) + 127;
                pp[9] = float2int8(p0[5] * scale) + 127;
                pp[10] = float2int8(p0[6] * scale) + 127;
                pp[11] = float2int8(p0[7] * scale) + 127;
                pp[12] = float2int8(p0[12] * scale) + 127;
                pp[13] = float2int8(p0[13] * scale) + 127;
                pp[14] = float2int8(p0[14] * scale) + 127;
                pp[15] = float2int8(p0[15] * scale) + 127;
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[8] * scale);
                pp[3] = float2int8(p0[9] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[10] * scale);
                pp[7] = float2int8(p0[11] * scale);
                pp[8] = float2int8(p0[4] * scale);
                pp[9] = float2int8(p0[5] * scale);
                pp[10] = float2int8(p0[12] * scale);
                pp[11] = float2int8(p0[13] * scale);
                pp[12] = float2int8(p0[6] * scale);
                pp[13] = float2int8(p0[7] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);
#endif // __AVX512VNNI__
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[1] * scale) + 127;
                pp[2] = float2int8(p0[2] * scale) + 127;
                pp[3] = float2int8(p0[3] * scale) + 127;
                pp[4] = float2int8(p0[4] * scale) + 127;
                pp[5] = float2int8(p0[5] * scale) + 127;
                pp[6] = float2int8(p0[6] * scale) + 127;
                pp[7] = float2int8(p0[7] * scale) + 127;
#else  // __AVX512VNNI__
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[4] * scale);
                pp[3] = float2int8(p0[5] * scale);
                pp[4] = float2int8(p0[2] * scale);
                pp[5] = float2int8(p0[3] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
#endif // __AVX512VNNI__
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __SSE2__
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[B_hstep + 0] * scale) + 127;
                pp[2] = float2int8(p0[B_hstep * 2 + 0] * scale) + 127;
                pp[3] = float2int8(p0[B_hstep * 3 + 0] * scale) + 127;
                pp[4] = float2int8(p0[1] * scale) + 127;
                pp[5] = float2int8(p0[B_hstep + 1] * scale) + 127;
                pp[6] = float2int8(p0[B_hstep * 2 + 1] * scale) + 127;
                pp[7] = float2int8(p0[B_hstep * 3 + 1] * scale) + 127;
                pp += 8;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep + 0] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __SSE2__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
                pp[8] = float2int8(p0[8] * scale);
                pp[9] = float2int8(p0[9] * scale);
                pp[10] = float2int8(p0[10] * scale);
                pp[11] = float2int8(p0[11] * scale);
                pp[12] = float2int8(p0[12] * scale);
                pp[13] = float2int8(p0[13] * scale);
                pp[14] = float2int8(p0[14] * scale);
                pp[15] = float2int8(p0[15] * scale);
#if __AVX512VNNI__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
                pp[8] += 127;
                pp[9] += 127;
                pp[10] += 127;
                pp[11] += 127;
                pp[12] += 127;
                pp[13] += 127;
                pp[14] += 127;
                pp[15] += 127;
#endif // __AVX512VNNI__
                pp += 16;
                p0 += B_hstep * 16;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp[4] = float2int8(p0[4] * scale);
                pp[5] = float2int8(p0[5] * scale);
                pp[6] = float2int8(p0[6] * scale);
                pp[7] = float2int8(p0[7] * scale);
#if __AVX512VNNI__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
                pp[4] += 127;
                pp[5] += 127;
                pp[6] += 127;
                pp[7] += 127;
#endif // __AVX512VNNI__
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
#if __AVX512VNNI__
                pp[0] += 127;
                pp[1] += 127;
                pp[2] += 127;
                pp[3] += 127;
#endif // __AVX512VNNI__
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale) + 127;
                pp[1] = float2int8(p0[B_hstep] * scale) + 127;
                pp[2] = float2int8(p0[B_hstep * 2] * scale) + 127;
                pp[3] = float2int8(p0[B_hstep * 3] * scale) + 127;
                pp += 4;
                p0 += B_hstep * 4;
            }
#endif // __AVX512VNNI__
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        unpack_output_tile_int32_to_fp32_avx2(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta, output_transpose);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack, output_transpose);

    const int* pp = topT;

    int ii = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m512 _descale = _mm512_loadu_ps((const float*)descales + i + ii);

        __m512 _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm512_set1_ps(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm512_loadu_ps(pC);
                _c0 = _mm512_mul_ps(_c0, _mm512_set1_ps(beta));
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 112)));
            __m512 _f8 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128)));
            __m512 _f9 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 16)));
            __m512 _fa = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 32)));
            __m512 _fb = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 48)));
            __m512 _fc = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 64)));
            __m512 _fd = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 80)));
            __m512 _fe = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 96)));
            __m512 _ff = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 128 + 112)));
            pp += 256;

            // from

            // 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
            // 01 12 23 30 45 56 67 74 89 9a ab b8 cd de ef fc
            // 20 31 02 13 64 75 46 57 a8 b9 8a 9b ec fd ce df
            // 21 32 03 10 65 76 47 54 a9 ba 8b 98 ed fe cf dc
            // 08 19 2a 3b 4c 5d 6e 7f 80 91 a2 b3 c4 d5 e6 f7
            // 09 1a 2b 38 4d 5e 6f 7c 81 92 a3 b0 c5 d6 e7 f4
            // 28 39 0a 1b 6c 7d 4e 5f a0 b1 82 93 e4 f5 c6 d7
            // 29 3a 0b 18 6d 7e 4f 5c a1 b2 83 90 e5 f6 c7 d4
            // 40 51 62 73 04 15 26 37 c8 d9 ea fb 8c 9d ae bf
            // 41 52 63 70 05 16 27 34 c9 da eb f8 8d 9e af bc
            // 60 71 42 53 24 35 06 17 e8 f9 ca db ac bd 8e 9f
            // 61 72 43 50 25 36 07 14 e9 fa cb d8 ad be 8f 9c
            // 48 59 6a 7b 0c 1d 2e 3f c0 d1 e2 f3 84 95 a6 b7
            // 49 5a 6b 78 0d 1e 2f 3c c1 d2 e3 f0 85 96 a7 b4
            // 68 79 4a 5b 2c 3d 0e 1f e0 f1 c2 d3 a4 b5 86 97
            // 69 7a 4b 58 2d 3e 0f 1c e1 f2 c3 d0 a5 b6 87 94

            // _f0 = _mm512_setr_ps(0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff);
            // _f1 = _mm512_setr_ps(0x01,0x12,0x23,0x30,0x45,0x56,0x67,0x74,0x89,0x9a,0xab,0xb8,0xcd,0xde,0xef,0xfc);
            // _f2 = _mm512_setr_ps(0x20,0x31,0x02,0x13,0x64,0x75,0x46,0x57,0xa8,0xb9,0x8a,0x9b,0xec,0xfd,0xce,0xdf);
            // _f3 = _mm512_setr_ps(0x21,0x32,0x03,0x10,0x65,0x76,0x47,0x54,0xa9,0xba,0x8b,0x98,0xed,0xfe,0xcf,0xdc);
            // _f4 = _mm512_setr_ps(0x08,0x19,0x2a,0x3b,0x4c,0x5d,0x6e,0x7f,0x80,0x91,0xa2,0xb3,0xc4,0xd5,0xe6,0xf7);
            // _f5 = _mm512_setr_ps(0x09,0x1a,0x2b,0x38,0x4d,0x5e,0x6f,0x7c,0x81,0x92,0xa3,0xb0,0xc5,0xd6,0xe7,0xf4);
            // _f6 = _mm512_setr_ps(0x28,0x39,0x0a,0x1b,0x6c,0x7d,0x4e,0x5f,0xa0,0xb1,0x82,0x93,0xe4,0xf5,0xc6,0xd7);
            // _f7 = _mm512_setr_ps(0x29,0x3a,0x0b,0x18,0x6d,0x7e,0x4f,0x5c,0xa1,0xb2,0x83,0x90,0xe5,0xf6,0xc7,0xd4);
            // _f8 = _mm512_setr_ps(0x40,0x51,0x62,0x73,0x04,0x15,0x26,0x37,0xc8,0xd9,0xea,0xfb,0x8c,0x9d,0xae,0xbf);
            // _f9 = _mm512_setr_ps(0x41,0x52,0x63,0x70,0x05,0x16,0x27,0x34,0xc9,0xda,0xeb,0xf8,0x8d,0x9e,0xaf,0xbc);
            // _fa = _mm512_setr_ps(0x60,0x71,0x42,0x53,0x24,0x35,0x06,0x17,0xe8,0xf9,0xca,0xdb,0xac,0xbd,0x8e,0x9f);
            // _fb = _mm512_setr_ps(0x61,0x72,0x43,0x50,0x25,0x36,0x07,0x14,0xe9,0xfa,0xcb,0xd8,0xad,0xbe,0x8f,0x9c);
            // _fc = _mm512_setr_ps(0x48,0x59,0x6a,0x7b,0x0c,0x1d,0x2e,0x3f,0xc0,0xd1,0xe2,0xf3,0x84,0x95,0xa6,0xb7);
            // _fd = _mm512_setr_ps(0x49,0x5a,0x6b,0x78,0x0d,0x1e,0x2f,0x3c,0xc1,0xd2,0xe3,0xf0,0x85,0x96,0xa7,0xb4);
            // _fe = _mm512_setr_ps(0x68,0x79,0x4a,0x5b,0x2c,0x3d,0x0e,0x1f,0xe0,0xf1,0xc2,0xd3,0xa4,0xb5,0x86,0x97);
            // _ff = _mm512_setr_ps(0x69,0x7a,0x4b,0x58,0x2d,0x3e,0x0f,0x1c,0xe1,0xf2,0xc3,0xd0,0xa5,0xb6,0x87,0x94);

            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);
            // print(_f8);
            // print(_f9);
            // print(_fa);
            // print(_fb);
            // print(_fc);
            // print(_fd);
            // print(_fe);
            // print(_ff);

            // to

            // 00 10 20 30  40 50 60 70  80 90 a0 b0  c0 d0 e0 f0
            // 01 11 21 31  41 51 61 71  81 91 a1 b1  c1 d1 e1 f1
            // 02 12 22 32  42 52 62 72  82 92 a2 b2  c2 d2 e2 f2
            // 03 13 23 33  43 53 63 73  83 93 a3 b3  c3 d3 e3 f3
            // 04 14 24 34  44 54 64 74  84 94 a4 b4  c4 d4 e4 f4
            // 05 15 25 35  45 55 65 75  85 95 a5 b5  c5 d5 e5 f5
            // 06 16 26 36  46 56 66 76  86 96 a6 b6  c6 d6 e6 f6
            // 07 17 27 37  47 57 67 77  87 97 a7 b7  c7 d7 e7 f7
            // 08 18 28 38  48 58 68 78  88 98 a8 b8  c8 d8 e8 f8
            // 09 19 29 39  49 59 69 79  89 99 a9 b9  c9 d9 e9 f9
            // 0a 1a 2a 3a  4a 5a 6a 7a  8a 9a aa ba  ca da ea fa
            // 0b 1b 2b 3b  4b 5b 6b 7b  8b 9b ab bb  cb db eb fb
            // 0c 1c 2c 3c  4c 5c 6c 7c  8c 9c ac bc  cc dc ec fc
            // 0d 1d 2d 3d  4d 5d 6d 7d  8d 9d ad bd  cd dd ed fd
            // 0e 1e 2e 3e  4e 5e 6e 7e  8e 9e ae be  ce de ee fe
            // 0f 1f 2f 3f  4f 5f 6f 7f  8f 9f af bf  cf df ef ff

            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 11 22 33  44 55 66 77  88 99 aa bb  cc dd ee ff
                // 30 01 12 23  74 45 56 67  b8 89 9a ab  fc cd de ef
                // 20 31 02 13  64 75 46 57  a8 b9 8a 9b  ec fd ce df
                // 10 21 32 03  54 65 76 47  98 a9 ba 8b  dc ed fe cf

                // 08 19 2a 3b  4c 5d 6e 7f  80 91 a2 b3  c4 d5 e6 f7
                // 38 09 1a 2b  7c 4d 5e 6f  b0 81 92 a3  f4 c5 d6 e7
                // 28 39 0a 1b  6c 7d 4e 5f  a0 b1 82 93  e4 f5 c6 d7
                // 18 29 3a 0b  5c 6d 7e 4f  90 a1 b2 83  d4 e5 f6 c7

                // 40 51 62 73  04 15 26 37  c8 d9 ea fb  8c 9d ae bf
                // 70 41 52 63  34 05 16 27  f8 c9 da eb  bc 8d 9e af
                // 60 71 42 53  24 35 06 17  e8 f9 ca db  ac bd 8e 9f
                // 50 61 72 43  14 25 36 07  d8 e9 fa cb  9c ad be 8f

                // 48 59 6a 7b  0c 1d 2e 3f  c0 d1 e2 f3  84 95 a6 b7
                // 78 49 5a 6b  3c 0d 1e 2f  f0 c1 d2 e3  b4 85 96 a7
                // 68 79 4a 5b  2c 3d 0e 1f  e0 f1 c2 d3  a4 b5 86 97
                // 58 69 7a 4b  1c 2d 3e 0f  d0 e1 f2 c3  94 a5 b6 87

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);
                __m512 _tmp8 = _mm512_unpacklo_ps(_f8, _fb);
                __m512 _tmp9 = _mm512_unpackhi_ps(_f8, _fb);
                __m512 _tmpa = _mm512_unpacklo_ps(_fa, _f9);
                __m512 _tmpb = _mm512_unpackhi_ps(_fa, _f9);
                __m512 _tmpc = _mm512_unpacklo_ps(_fc, _ff);
                __m512 _tmpd = _mm512_unpackhi_ps(_fc, _ff);
                __m512 _tmpe = _mm512_unpacklo_ps(_fe, _fd);
                __m512 _tmpf = _mm512_unpackhi_ps(_fe, _fd);

                // 00 10 11 21  44 54 55 65  88 98 99 a9  cc dc dd ed
                // 22 32 33 03  66 76 77 47  aa ba bb 8b  ee fe ff cf
                // 20 30 31 01  64 74 75 45  a8 b8 b9 89  ec fc fd cd
                // 02 12 13 23  46 56 57 67  8a 9a 9b ab  ce de df ef

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _f9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp8), _mm512_castps_pd(_tmpa)));
                _fa = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpb), _mm512_castps_pd(_tmp9)));
                _fc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpc), _mm512_castps_pd(_tmpe)));
                _fe = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));
                _ff = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmpf), _mm512_castps_pd(_tmpd)));

                // 00 10 20 30  44 54 64 74  88 98 a8 b8  cc dc ec fc
                // 11 21 31 01  55 65 75 45  99 a9 b9 89  dd ed fd cd
                // 02 12 22 32  46 56 66 76  8a 9a aa ba  ce de ee fe
                // 13 23 33 03  57 67 77 47  9b ab bb 8b  df ef ff cf

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));
                _f9 = _mm512_permute_ps(_f9, _MM_SHUFFLE(2, 1, 0, 3));
                _fb = _mm512_permute_ps(_fb, _MM_SHUFFLE(2, 1, 0, 3));
                _fd = _mm512_permute_ps(_fd, _MM_SHUFFLE(2, 1, 0, 3));
                _ff = _mm512_permute_ps(_ff, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 10 20 30  44 54 64 74  88 98 a8 b8  cc dc ec fc
                // 01 11 21 31  45 55 65 75  89 99 a9 b9  cd dd ed fd
                // 02 12 22 32  46 56 66 76  8a 9a aa ba  ce de ee fe
                // 03 13 23 33  47 57 67 77  8b 9b ab bb  cf df ef ff

                // 08 18 28 38  4c 5c 6c 7c  80 90 a0 b0  c4 d4 e4 f4
                // 09 19 29 39  4d 5d 6d 7d  81 91 a1 b1  c5 d5 e5 f5
                // 0a 1a 2a 3a  4e 5e 6e 7e  82 92 a2 b2  c6 d6 e6 f6
                // 0b 1b 2b 3b  4f 5f 6f 7f  83 93 a3 b3  c7 d7 e7 f7

                // 40 50 60 70  04 14 24 34  c8 d8 e8 f8  8c 9c ac bc
                // 41 51 61 71  05 15 25 35  c9 d9 e9 f9  8d 9d ad bd
                // 42 52 62 72  06 16 26 36  ca da ea fa  8e 9e ae be
                // 43 53 63 73  07 17 27 37  cb db eb fb  8f 9f af bf

                // 48 58 68 78  0c 1c 2c 3c  c0 d0 e0 f0  84 94 a4 b4
                // 49 59 69 79  0d 1d 2d 3d  c1 d1 e1 f1  85 95 a5 b5
                // 4a 5a 6a 7a  0e 1e 2e 3e  c2 d2 e2 f2  86 96 a6 b6
                // 4b 5b 6b 7b  0f 1f 2f 3f  c3 d3 e3 f3  87 97 a7 b7

                // NCNN_LOGE("--------");
                // print(_f0);
                // print(_f1);
                // print(_f2);
                // print(_f3);
                // print(_f4);
                // print(_f5);
                // print(_f6);
                // print(_f7);
                // print(_f8);
                // print(_f9);
                // print(_fa);
                // print(_fb);
                // print(_fc);
                // print(_fd);
                // print(_fe);
                // print(_ff);

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f8, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f1, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp2 = _mm512_shuffle_f32x4(_f2, _fa, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f3, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp4 = _mm512_shuffle_f32x4(_f8, _f0, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp5 = _mm512_shuffle_f32x4(_f9, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp6 = _mm512_shuffle_f32x4(_fa, _f2, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp7 = _mm512_shuffle_f32x4(_fb, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                _tmp8 = _mm512_shuffle_f32x4(_f4, _fc, _MM_SHUFFLE(2, 0, 2, 0));
                _tmp9 = _mm512_shuffle_f32x4(_f5, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpa = _mm512_shuffle_f32x4(_f6, _fe, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpb = _mm512_shuffle_f32x4(_f7, _ff, _MM_SHUFFLE(2, 0, 2, 0));
                _tmpc = _mm512_shuffle_f32x4(_fc, _f4, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpd = _mm512_shuffle_f32x4(_fd, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpe = _mm512_shuffle_f32x4(_fe, _f6, _MM_SHUFFLE(3, 1, 3, 1));
                _tmpf = _mm512_shuffle_f32x4(_ff, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                // 00 10 20 30  88 98 a8 b8  40 50 60 70  c8 d8 e8 f8
                // 01 11 21 31  89 99 a9 b9  41 51 61 71  c9 d9 e9 f9
                // 02 12 22 32  8a 9a aa ba  42 52 62 72  ca da ea fa
                // 03 13 23 33  8b 9b ab bb  43 53 63 73  cb db eb fb
                // 04 14 24 34  8c 9c ac bc  44 54 64 74  cc dc ec fc
                // 05 15 25 35  8d 9d ad bd  45 55 65 75  cd dd ed fd
                // 06 16 26 36  8e 9e ae be  46 56 66 76  ce de ee fe
                // 07 17 27 37  8f 9f af bf  47 57 67 77  cf df ef ff

                // 08 18 28 38  80 90 a0 b0  48 58 68 78  c0 d0 e0 f0
                // 09 19 29 39  81 91 a1 b1  49 59 69 79  c1 d1 e1 f1
                // 0a 1a 2a 3a  82 92 a2 b2  4a 5a 6a 7a  c2 d2 e2 f2
                // 0b 1b 2b 3b  83 93 a3 b3  4b 5b 6b 7b  c3 d3 e3 f3
                // 0c 1c 2c 3c  84 94 a4 b4  4c 5c 6c 7c  c4 d4 e4 f4
                // 0d 1d 2d 3d  85 95 a5 b5  4d 5d 6d 7d  c5 d5 e5 f5
                // 0e 1e 2e 3e  86 96 a6 b6  4e 5e 6e 7e  c6 d6 e6 f6
                // 0f 1f 2f 3f  87 97 a7 b7  4f 5f 6f 7f  c7 d7 e7 f7

                // NCNN_LOGE("--------");
                // print(_tmp0);
                // print(_tmp1);
                // print(_tmp2);
                // print(_tmp3);
                // print(_tmp4);
                // print(_tmp5);
                // print(_tmp6);
                // print(_tmp7);
                // print(_tmp8);
                // print(_tmp9);
                // print(_tmpa);
                // print(_tmpb);
                // print(_tmpc);
                // print(_tmpd);
                // print(_tmpe);
                // print(_tmpf);

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 1, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 1, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 1, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 1, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 1, 2, 0));
                _f5 = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 1, 2, 0));
                _f6 = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 1, 2, 0));
                _f7 = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 1, 2, 0));
                _f8 = _mm512_shuffle_f32x4(_tmp8, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                _f9 = _mm512_shuffle_f32x4(_tmp9, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                _fa = _mm512_shuffle_f32x4(_tmpa, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                _fb = _mm512_shuffle_f32x4(_tmpb, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                _fc = _mm512_shuffle_f32x4(_tmpc, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                _fd = _mm512_shuffle_f32x4(_tmpd, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                _fe = _mm512_shuffle_f32x4(_tmpe, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                _ff = _mm512_shuffle_f32x4(_tmpf, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));
            }

            // NCNN_LOGE("--------");
            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);
            // print(_f8);
            // print(_f9);
            // print(_fa);
            // print(_fb);
            // print(_fc);
            // print(_fd);
            // print(_fe);
            // print(_ff);

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);
            _f4 = _mm512_mul_ps(_f4, _descale);
            _f5 = _mm512_mul_ps(_f5, _descale);
            _f6 = _mm512_mul_ps(_f6, _descale);
            _f7 = _mm512_mul_ps(_f7, _descale);
            _f8 = _mm512_mul_ps(_f8, _descale);
            _f9 = _mm512_mul_ps(_f9, _descale);
            _fa = _mm512_mul_ps(_fa, _descale);
            _fb = _mm512_mul_ps(_fb, _descale);
            _fc = _mm512_mul_ps(_fc, _descale);
            _fd = _mm512_mul_ps(_fd, _descale);
            _fe = _mm512_mul_ps(_fe, _descale);
            _ff = _mm512_mul_ps(_ff, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c0);
                    _fa = _mm512_add_ps(_fa, _c0);
                    _fb = _mm512_add_ps(_fb, _c0);
                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c0);
                    _fe = _mm512_add_ps(_fe, _c0);
                    _ff = _mm512_add_ps(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c0);
                    _fa = _mm512_add_ps(_fa, _c0);
                    _fb = _mm512_add_ps(_fb, _c0);
                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c0);
                    _fe = _mm512_add_ps(_fe, _c0);
                    _ff = _mm512_add_ps(_ff, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
                    __m512 _c8;
                    __m512 _c9;
                    __m512 _ca;
                    __m512 _cb;
                    __m512 _cc;
                    __m512 _cd;
                    __m512 _ce;
                    __m512 _cf;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + 64);
                        _c5 = _mm512_loadu_ps(pC + 80);
                        _c6 = _mm512_loadu_ps(pC + 96);
                        _c7 = _mm512_loadu_ps(pC + 112);
                        _c8 = _mm512_loadu_ps(pC + 128);
                        _c9 = _mm512_loadu_ps(pC + 128 + 16);
                        _ca = _mm512_loadu_ps(pC + 128 + 32);
                        _cb = _mm512_loadu_ps(pC + 128 + 48);
                        _cc = _mm512_loadu_ps(pC + 128 + 64);
                        _cd = _mm512_loadu_ps(pC + 128 + 80);
                        _ce = _mm512_loadu_ps(pC + 128 + 96);
                        _cf = _mm512_loadu_ps(pC + 128 + 112);
                        pC += 256;
                    }
                    if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + 64);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + 80);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + 96);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + 112);
                        __m512 _tmp8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmpa = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmpb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        __m512 _tmpc = _mm512_loadu_ps(pC + c_hstep * 8 + 64);
                        __m512 _tmpd = _mm512_loadu_ps(pC + c_hstep * 8 + 80);
                        __m512 _tmpe = _mm512_loadu_ps(pC + c_hstep * 8 + 96);
                        __m512 _tmpf = _mm512_loadu_ps(pC + c_hstep * 8 + 112);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp8, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp9, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                        _c8 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(1, 0, 1, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp4, _tmpc, _MM_SHUFFLE(3, 2, 3, 2));
                        _ca = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(1, 0, 1, 0));
                        _cb = _mm512_shuffle_f32x4(_tmp5, _tmpd, _MM_SHUFFLE(3, 2, 3, 2));
                        _cc = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
                        _cd = _mm512_shuffle_f32x4(_tmp6, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
                        _ce = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
                        _cf = _mm512_shuffle_f32x4(_tmp7, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 128;
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 8 + 48);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 12 + 16);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 12 + 32);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 12 + 48);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c4, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c8, _cc, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c5, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c9, _cd, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp8 = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp9 = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpa = _mm512_shuffle_f32x4(_c2, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpb = _mm512_shuffle_f32x4(_ca, _ce, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpc = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpd = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmpe = _mm512_shuffle_f32x4(_c3, _c7, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmpf = _mm512_shuffle_f32x4(_cb, _cf, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                        _c8 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                        _c9 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                        _ca = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                        _cb = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                        _cc = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                        _cd = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                        _ce = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                        _cf = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 64;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 7);
                        _c8 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c9 = _mm512_loadu_ps(pC + c_hstep * 9);
                        _ca = _mm512_loadu_ps(pC + c_hstep * 10);
                        _cb = _mm512_loadu_ps(pC + c_hstep * 11);
                        _cc = _mm512_loadu_ps(pC + c_hstep * 12);
                        _cd = _mm512_loadu_ps(pC + c_hstep * 13);
                        _ce = _mm512_loadu_ps(pC + c_hstep * 14);
                        _cf = _mm512_loadu_ps(pC + c_hstep * 15);
                        transpose16x16_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7, _c8, _c9, _ca, _cb, _cc, _cd, _ce, _cf);
                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                        _f4 = _mm512_add_ps(_f4, _c4);
                        _f5 = _mm512_add_ps(_f5, _c5);
                        _f6 = _mm512_add_ps(_f6, _c6);
                        _f7 = _mm512_add_ps(_f7, _c7);
                        _f8 = _mm512_add_ps(_f8, _c8);
                        _f9 = _mm512_add_ps(_f9, _c9);
                        _fa = _mm512_add_ps(_fa, _ca);
                        _fb = _mm512_add_ps(_fb, _cb);
                        _fc = _mm512_add_ps(_fc, _cc);
                        _fd = _mm512_add_ps(_fd, _cd);
                        _fe = _mm512_add_ps(_fe, _ce);
                        _ff = _mm512_add_ps(_ff, _cf);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                        _f8 = _mm512_fmadd_ps(_c8, _beta, _f8);
                        _f9 = _mm512_fmadd_ps(_c9, _beta, _f9);
                        _fa = _mm512_fmadd_ps(_ca, _beta, _fa);
                        _fb = _mm512_fmadd_ps(_cb, _beta, _fb);
                        _fc = _mm512_fmadd_ps(_cc, _beta, _fc);
                        _fd = _mm512_fmadd_ps(_cd, _beta, _fd);
                        _fe = _mm512_fmadd_ps(_ce, _beta, _fe);
                        _ff = _mm512_fmadd_ps(_cf, _beta, _ff);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);

                    _c0 = _mm512_set1_ps(pC[8] * beta);
                    _c1 = _mm512_set1_ps(pC[9] * beta);
                    _c2 = _mm512_set1_ps(pC[10] * beta);
                    _c3 = _mm512_set1_ps(pC[11] * beta);

                    _f8 = _mm512_add_ps(_f8, _c0);
                    _f9 = _mm512_add_ps(_f9, _c1);
                    _fa = _mm512_add_ps(_fa, _c2);
                    _fb = _mm512_add_ps(_fb, _c3);

                    _c0 = _mm512_set1_ps(pC[12] * beta);
                    _c1 = _mm512_set1_ps(pC[13] * beta);
                    _c2 = _mm512_set1_ps(pC[14] * beta);
                    _c3 = _mm512_set1_ps(pC[15] * beta);

                    _fc = _mm512_add_ps(_fc, _c0);
                    _fd = _mm512_add_ps(_fd, _c1);
                    _fe = _mm512_add_ps(_fe, _c2);
                    _ff = _mm512_add_ps(_ff, _c3);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
                _f8 = _mm512_mul_ps(_f8, _alpha);
                _f9 = _mm512_mul_ps(_f9, _alpha);
                _fa = _mm512_mul_ps(_fa, _alpha);
                _fb = _mm512_mul_ps(_fb, _alpha);
                _fc = _mm512_mul_ps(_fc, _alpha);
                _fd = _mm512_mul_ps(_fd, _alpha);
                _fe = _mm512_mul_ps(_fe, _alpha);
                _ff = _mm512_mul_ps(_ff, _alpha);
            }

            if (output_transpose)
            {
                // 00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                // 01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                // 02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                // 03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                // 04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
                // 05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
                // 06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
                // 07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7
                // 08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
                // 09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
                // 0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
                // 0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb
                // 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
                // 0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
                // 0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
                // 0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff

                if (out_elempack == 16)
                {
                    transpose16x16_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + 16 * 4, _f4);
                    _mm512_storeu_ps(p0 + 16 * 5, _f5);
                    _mm512_storeu_ps(p0 + 16 * 6, _f6);
                    _mm512_storeu_ps(p0 + 16 * 7, _f7);
                    _mm512_storeu_ps(p0 + 16 * 8, _f8);
                    _mm512_storeu_ps(p0 + 16 * 9, _f9);
                    _mm512_storeu_ps(p0 + 16 * 10, _fa);
                    _mm512_storeu_ps(p0 + 16 * 11, _fb);
                    _mm512_storeu_ps(p0 + 16 * 12, _fc);
                    _mm512_storeu_ps(p0 + 16 * 13, _fd);
                    _mm512_storeu_ps(p0 + 16 * 14, _fe);
                    _mm512_storeu_ps(p0 + 16 * 15, _ff);
                }
                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    transpose16x8_ps(_f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + 16 * 4, _f4);
                    _mm512_storeu_ps(p0 + 16 * 5, _f5);
                    _mm512_storeu_ps(p0 + 16 * 6, _f6);
                    _mm512_storeu_ps(p0 + 16 * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 2, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 3, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 4, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 5, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 6, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 7, _ff);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);
                    transpose16x4_ps(_f8, _f9, _fa, _fb);
                    transpose16x4_ps(_fc, _fd, _fe, _ff);

                    // 00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    // 01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    // 02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                    // 03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3

                    // 04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
                    // 05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
                    // 06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
                    // 07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7

                    // 08 18 28 38 48 58 68 78 88 98 a8 b8 c8 d8 e8 f8
                    // 09 19 29 39 49 59 69 79 89 99 a9 b9 c9 d9 e9 f9
                    // 0a 1a 2a 3a 4a 5a 6a 7a 8a 9a aa ba ca da ea fa
                    // 0b 1b 2b 3b 4b 5b 6b 7b 8b 9b ab bb cb db eb fb

                    // 0c 1c 2c 3c 4c 5c 6c 7c 8c 9c ac bc cc dc ec fc
                    // 0d 1d 2d 3d 4d 5d 6d 7d 8d 9d ad bd cd dd ed fd
                    // 0e 1e 2e 3e 4e 5e 6e 7e 8e 9e ae be ce de ee fe
                    // 0f 1f 2f 3f 4f 5f 6f 7f 8f 9f af bf cf df ef ff

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 32, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 48, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 32, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 48, _ff);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 9, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 10, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 11, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 13, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 14, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 15, _ff);
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    _mm512_store_ps(p0 + 32, _f2);
                    _mm512_store_ps(p0 + 48, _f3);
                    _mm512_store_ps(p0 + 64, _f4);
                    _mm512_store_ps(p0 + 80, _f5);
                    _mm512_store_ps(p0 + 96, _f6);
                    _mm512_store_ps(p0 + 112, _f7);
                    _mm512_store_ps(p0 + 128, _f8);
                    _mm512_store_ps(p0 + 128 + 16, _f9);
                    _mm512_store_ps(p0 + 128 + 32, _fa);
                    _mm512_store_ps(p0 + 128 + 48, _fb);
                    _mm512_store_ps(p0 + 128 + 64, _fc);
                    _mm512_store_ps(p0 + 128 + 80, _fd);
                    _mm512_store_ps(p0 + 128 + 96, _fe);
                    _mm512_store_ps(p0 + 128 + 112, _ff);
                    p0 += 256;
                }
                if (out_elempack == 8)
                {
                    // 00 10 20 30 40 50 60 70   80 90 a0 b0 c0 d0 e0 f0
                    // 01 11 21 31 41 51 61 71   81 91 a1 b1 c1 d1 e1 f1
                    // 02 12 22 32 42 52 62 72   82 92 a2 b2 c2 d2 e2 f2
                    // 03 13 23 33 43 53 63 73   83 93 a3 b3 c3 d3 e3 f3
                    // 04 14 24 34 44 54 64 74   84 94 a4 b4 c4 d4 e4 f4
                    // 05 15 25 35 45 55 65 75   85 95 a5 b5 c5 d5 e5 f5
                    // 06 16 26 36 46 56 66 76   86 96 a6 b6 c6 d6 e6 f6
                    // 07 17 27 37 47 57 67 77   87 97 a7 b7 c7 d7 e7 f7
                    // 08 18 28 38 48 58 68 78   88 98 a8 b8 c8 d8 e8 f8
                    // 09 19 29 39 49 59 69 79   89 99 a9 b9 c9 d9 e9 f9
                    // 0a 1a 2a 3a 4a 5a 6a 7a   8a 9a aa ba ca da ea fa
                    // 0b 1b 2b 3b 4b 5b 6b 7b   8b 9b ab bb cb db eb fb
                    // 0c 1c 2c 3c 4c 5c 6c 7c   8c 9c ac bc cc dc ec fc
                    // 0d 1d 2d 3d 4d 5d 6d 7d   8d 9d ad bd cd dd ed fd
                    // 0e 1e 2e 3e 4e 5e 6e 7e   8e 9e ae be ce de ee fe
                    // 0f 1f 2f 3f 4f 5f 6f 7f   8f 9f af bf cf df ef ff

                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + 16, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + 24, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + 32, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + 40, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + 48, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + 56, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + 64, _mm512_extractf32x8_ps(_f8, 0));
                    _mm256_storeu_ps(p0 + 64 + 8, _mm512_extractf32x8_ps(_f9, 0));
                    _mm256_storeu_ps(p0 + 64 + 16, _mm512_extractf32x8_ps(_fa, 0));
                    _mm256_storeu_ps(p0 + 64 + 24, _mm512_extractf32x8_ps(_fb, 0));
                    _mm256_storeu_ps(p0 + 64 + 32, _mm512_extractf32x8_ps(_fc, 0));
                    _mm256_storeu_ps(p0 + 64 + 40, _mm512_extractf32x8_ps(_fd, 0));
                    _mm256_storeu_ps(p0 + 64 + 48, _mm512_extractf32x8_ps(_fe, 0));
                    _mm256_storeu_ps(p0 + 64 + 56, _mm512_extractf32x8_ps(_ff, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 24, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 32, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 40, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 48, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 56, _mm512_extractf32x8_ps(_f7, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64, _mm512_extractf32x8_ps(_f8, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 8, _mm512_extractf32x8_ps(_f9, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 16, _mm512_extractf32x8_ps(_fa, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 24, _mm512_extractf32x8_ps(_fb, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 32, _mm512_extractf32x8_ps(_fc, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 40, _mm512_extractf32x8_ps(_fd, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 48, _mm512_extractf32x8_ps(_fe, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 64 + 56, _mm512_extractf32x8_ps(_ff, 1));
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    // 00 10 20 30   40 50 60 70   80 90 a0 b0   c0 d0 e0 f0
                    // 01 11 21 31   41 51 61 71   81 91 a1 b1   c1 d1 e1 f1
                    // 02 12 22 32   42 52 62 72   82 92 a2 b2   c2 d2 e2 f2
                    // 03 13 23 33   43 53 63 73   83 93 a3 b3   c3 d3 e3 f3
                    // 04 14 24 34   44 54 64 74   84 94 a4 b4   c4 d4 e4 f4
                    // 05 15 25 35   45 55 65 75   85 95 a5 b5   c5 d5 e5 f5
                    // 06 16 26 36   46 56 66 76   86 96 a6 b6   c6 d6 e6 f6
                    // 07 17 27 37   47 57 67 77   87 97 a7 b7   c7 d7 e7 f7
                    // 08 18 28 38   48 58 68 78   88 98 a8 b8   c8 d8 e8 f8
                    // 09 19 29 39   49 59 69 79   89 99 a9 b9   c9 d9 e9 f9
                    // 0a 1a 2a 3a   4a 5a 6a 7a   8a 9a aa ba   ca da ea fa
                    // 0b 1b 2b 3b   4b 5b 6b 7b   8b 9b ab bb   cb db eb fb
                    // 0c 1c 2c 3c   4c 5c 6c 7c   8c 9c ac bc   cc dc ec fc
                    // 0d 1d 2d 3d   4d 5d 6d 7d   8d 9d ad bd   cd dd ed fd
                    // 0e 1e 2e 3e   4e 5e 6e 7e   8e 9e ae be   ce de ee fe
                    // 0f 1f 2f 3f   4f 5f 6f 7f   8f 9f af bf   cf df ef ff

                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f8, _f9, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_fa, _fb, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_fc, _fd, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_fe, _ff, _MM_SHUFFLE(2, 0, 2, 0));

                    __m512 _tmp8 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp9 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpa = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpb = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpc = _mm512_shuffle_f32x4(_f8, _f9, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpd = _mm512_shuffle_f32x4(_fa, _fb, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpe = _mm512_shuffle_f32x4(_fc, _fd, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmpf = _mm512_shuffle_f32x4(_fe, _ff, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _f7 = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));

                    _f8 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f9 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _fa = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _fb = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));
                    _fc = _mm512_shuffle_f32x4(_tmp8, _tmp9, _MM_SHUFFLE(3, 1, 3, 1));
                    _fd = _mm512_shuffle_f32x4(_tmpa, _tmpb, _MM_SHUFFLE(3, 1, 3, 1));
                    _fe = _mm512_shuffle_f32x4(_tmpc, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _ff = _mm512_shuffle_f32x4(_tmpe, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 32, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 48, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 32, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 48, _ff);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    transpose16x16_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9, _fa, _fb, _fc, _fd, _fe, _ff);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f8);
                    _mm512_storeu_ps(p0 + out_hstep * 9, _f9);
                    _mm512_storeu_ps(p0 + out_hstep * 10, _fa);
                    _mm512_storeu_ps(p0 + out_hstep * 11, _fb);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _fc);
                    _mm512_storeu_ps(p0 + out_hstep * 13, _fd);
                    _mm512_storeu_ps(p0 + out_hstep * 14, _fe);
                    _mm512_storeu_ps(p0 + out_hstep * 15, _ff);
                    p0 += 16;
                }
            }
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 112)));
            pp += 128;

            // from
            //      00 11 22 33  44 55 66 77  80 91 a2 b3  c4 d5 e6 f7
            //      01 12 23 30  45 56 67 74  81 92 a3 b0  c5 d6 e7 f4
            //      20 31 02 13  64 75 46 57  a0 b1 82 93  e4 f5 c6 d7
            //      21 32 03 10  65 76 47 54  a1 b2 83 90  e5 f6 c7 d4
            //      04 15 26 37  40 51 62 73  84 95 a6 b7  c0 d1 e2 f3
            //      05 16 27 34  41 52 63 70  85 96 a7 b4  c1 d2 e3 f0
            //      24 35 06 17  60 71 42 53  a4 b5 86 97  e0 f1 c2 d3
            //      25 36 07 14  61 72 43 50  a5 b6 87 94  e1 f2 c3 d0
            //
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
            //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
            //      04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
            //      05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
            //      06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
            //      07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7

            // _f0 = _mm512_setr_ps(00,11,22,33,44,55,66,77,0x80,0x91,0xa2,0xb3,0xc4,0xd5,0xe6,0xf7);
            // _f1 = _mm512_setr_ps(01,12,23,30,45,56,67,74,0x81,0x92,0xa3,0xb0,0xc5,0xd6,0xe7,0xf4);
            // _f2 = _mm512_setr_ps(20,31,02,13,64,75,46,57,0xa0,0xb1,0x82,0x93,0xe4,0xf5,0xc6,0xd7);
            // _f3 = _mm512_setr_ps(21,32,03,10,65,76,47,54,0xa1,0xb2,0x83,0x90,0xe5,0xf6,0xc7,0xd4);
            // _f4 = _mm512_setr_ps(04,15,26,37,40,51,62,73,0x84,0x95,0xa6,0xb7,0xc0,0xd1,0xe2,0xf3);
            // _f5 = _mm512_setr_ps(05,16,27,34,41,52,63,70,0x85,0x96,0xa7,0xb4,0xc1,0xd2,0xe3,0xf0);
            // _f6 = _mm512_setr_ps(24,35,06,17,60,71,42,53,0xa4,0xb5,0x86,0x97,0xe0,0xf1,0xc2,0xd3);
            // _f7 = _mm512_setr_ps(25,36,07,14,61,72,43,50,0xa5,0xb6,0x87,0x94,0xe1,0xf2,0xc3,0xd0);

            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);

            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 11 22 33  44 55 66 77  80 91 a2 b3  c4 d5 e6 f7
                // 30 01 12 23  74 45 56 67  b0 81 92 a3  f4 c5 d6 e7
                // 20 31 02 13  64 75 46 57  a0 b1 82 93  e4 f5 c6 d7
                // 10 21 32 03  54 65 76 47  90 a1 b2 83  d4 e5 f6 c7

                // 04 15 26 37  40 51 62 73  84 95 a6 b7  c0 d1 e2 f3
                // 34 05 16 27  70 41 52 63  b4 85 96 a7  f0 c1 d2 e3
                // 24 35 06 17  60 71 42 53  a4 b5 86 97  e0 f1 c2 d3
                // 14 25 36 07  50 61 72 43  94 a5 b6 87  d0 e1 f2 c3

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);

                // 00 10 11 21  44 54 55 65  80 90 91 a1  c4 d4 d5 e5
                // 22 32 33 03  66 76 77 47  a2 b2 b3 83  e6 f6 f7 c7
                // 20 30 31 01  64 74 75 45  a0 b0 b1 81  e4 f4 f5 c5
                // 02 12 13 23  46 56 57 67  82 92 93 a3  c6 d6 d7 e7

                // 04 14 15 25  40 50 51 61
                // 26 36 37 07  62 72 73 43
                // 24 34 35 05  60 70 71 41
                // 06 16 17 27  42 52 53 63

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                // 00 10 20 30  44 54 64 74  80 90 a0 b0  c4 d4 e4 f4
                // 11 21 31 01  55 65 75 45  91 a1 b1 81  d5 e5 f5 c5
                // 02 12 22 32  46 56 66 76  82 92 a2 b2  c6 d6 e6 f6
                // 13 23 33 03  57 67 77 47  93 a3 b3 83  d7 e7 f7 c7

                // 04 14 24 34  40 50 60 70
                // 15 25 35 05  51 61 71 41
                // 06 16 26 36  42 52 62 72
                // 17 27 37 07  53 63 73 43

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 10 20 30  44 54 64 74  80 90 a0 b0  c4 d4 e4 f4
                // 01 11 21 31  45 55 65 75  81 91 a1 b1  c5 d5 e5 f5
                // 02 12 22 32  46 56 66 76  82 92 a2 b2  c6 d6 e6 f6
                // 03 13 23 33  47 57 67 77  83 93 a3 b3  c7 d7 e7 f7

                // 04 14 24 34  40 50 60 70  84 94 a4 b4  c0 d0 e0 f0
                // 05 15 25 35  41 51 61 71  85 95 a5 b5  c1 d1 e1 f1
                // 06 16 26 36  42 52 62 72  86 96 a6 b6  c2 d2 e2 f2
                // 07 17 27 37  43 53 63 73  87 97 a7 b7  c3 d3 e3 f3

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp2 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp4 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp5 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                // 00 10 20 30  44 54 64 74  40 50 60 70  04 14 24 34
                // 01 11 21 31  45 55 65 75  41 51 61 71  05 15 25 35
                // 02 12 22 32  46 56 66 76  42 52 62 72  06 16 26 36
                // 03 13 23 33  47 57 67 77  43 53 63 73  07 17 27 37

                // 80 90 a0 b0  c4 d4 e4 f4  c0 d0 e0 f0  84 94 a4 b4
                // 81 91 a1 b1  c5 d5 e5 f5  c1 d1 e1 f1  85 95 a5 b5
                // 82 92 a2 b2  c6 d6 e6 f6  c2 d2 e2 f2  86 96 a6 b6
                // 83 93 a3 b3  c7 d7 e7 f7  c3 d3 e3 f3  87 97 a7 b7

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));
            }

            // NCNN_LOGE("-------");
            //
            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);
            _f4 = _mm512_mul_ps(_f4, _descale);
            _f5 = _mm512_mul_ps(_f5, _descale);
            _f6 = _mm512_mul_ps(_f6, _descale);
            _f7 = _mm512_mul_ps(_f7, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c0);
                    _f6 = _mm512_add_ps(_f6, _c0);
                    _f7 = _mm512_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    __m512 _c4;
                    __m512 _c5;
                    __m512 _c6;
                    __m512 _c7;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        _c4 = _mm512_loadu_ps(pC + 64);
                        _c5 = _mm512_loadu_ps(pC + 80);
                        _c6 = _mm512_loadu_ps(pC + 96);
                        _c7 = _mm512_loadu_ps(pC + 112);
                        pC += 128;
                    }
                    if (c_elempack == 8)
                    {
                        __m512 _tmp0 = _mm512_loadu_ps(pC);
                        __m512 _tmp1 = _mm512_loadu_ps(pC + 16);
                        __m512 _tmp2 = _mm512_loadu_ps(pC + 32);
                        __m512 _tmp3 = _mm512_loadu_ps(pC + 48);
                        __m512 _tmp4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _tmp5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        __m512 _tmp6 = _mm512_loadu_ps(pC + c_hstep * 8 + 32);
                        __m512 _tmp7 = _mm512_loadu_ps(pC + c_hstep * 8 + 48);

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 2, 3, 2));
                        _c4 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        _c6 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

                        pC += 64;
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c4 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c5 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c6 = _mm512_loadu_ps(pC + c_hstep * 12);
                        _c7 = _mm512_loadu_ps(pC + c_hstep * 12 + 16);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c4, _c6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c1, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5, _c7, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c6 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        __m256 _cc4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        __m256 _cc6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        __m256 _cc7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        __m256 _cc8 = _mm256_loadu_ps(pC + c_hstep * 8);
                        __m256 _cc9 = _mm256_loadu_ps(pC + c_hstep * 9);
                        __m256 _cca = _mm256_loadu_ps(pC + c_hstep * 10);
                        __m256 _ccb = _mm256_loadu_ps(pC + c_hstep * 11);
                        __m256 _ccc = _mm256_loadu_ps(pC + c_hstep * 12);
                        __m256 _ccd = _mm256_loadu_ps(pC + c_hstep * 13);
                        __m256 _cce = _mm256_loadu_ps(pC + c_hstep * 14);
                        __m256 _ccf = _mm256_loadu_ps(pC + c_hstep * 15);
                        transpose8x8_ps(_cc0, _cc1, _cc2, _cc3, _cc4, _cc5, _cc6, _cc7);
                        transpose8x8_ps(_cc8, _cc9, _cca, _ccb, _ccc, _ccd, _cce, _ccf);
                        _c0 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc0), _cc8, 1);
                        _c1 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc1), _cc9, 1);
                        _c2 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc2), _cca, 1);
                        _c3 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc3), _ccb, 1);
                        _c4 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc4), _ccc, 1);
                        _c5 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc5), _ccd, 1);
                        _c6 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc6), _cce, 1);
                        _c7 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc7), _ccf, 1);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                        _f4 = _mm512_add_ps(_f4, _c4);
                        _f5 = _mm512_add_ps(_f5, _c5);
                        _f6 = _mm512_add_ps(_f6, _c6);
                        _f7 = _mm512_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);

                    _c0 = _mm512_set1_ps(pC[4] * beta);
                    _c1 = _mm512_set1_ps(pC[5] * beta);
                    _c2 = _mm512_set1_ps(pC[6] * beta);
                    _c3 = _mm512_set1_ps(pC[7] * beta);

                    _f4 = _mm512_add_ps(_f4, _c0);
                    _f5 = _mm512_add_ps(_f5, _c1);
                    _f6 = _mm512_add_ps(_f6, _c2);
                    _f7 = _mm512_add_ps(_f7, _c3);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
                //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                //      04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
                //      05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
                //      06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
                //      07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7

                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + 64, _f4);
                    _mm512_storeu_ps(p0 + 80, _f5);
                    _mm512_storeu_ps(p0 + 96, _f6);
                    _mm512_storeu_ps(p0 + 112, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + 64, _f4);
                    _mm512_storeu_ps(p0 + 80, _f5);
                    _mm512_storeu_ps(p0 + 96, _f6);
                    _mm512_storeu_ps(p0 + 112, _f7);
                    p0 += 128;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + 16, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + 24, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + 32, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + 40, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + 48, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + 56, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 24, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 32, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 40, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 48, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 56, _mm512_extractf32x8_ps(_f7, 1));
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                    //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                    //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                    //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                    //      04 14 24 34 44 54 64 74 84 94 a4 b4 c4 d4 e4 f4
                    //      05 15 25 35 45 55 65 75 85 95 a5 b5 c5 d5 e5 f5
                    //      06 16 26 36 46 56 66 76 86 96 a6 b6 c6 d6 e6 f6
                    //      07 17 27 37 47 57 67 77 87 97 a7 b7 c7 d7 e7 f7

                    //      00  40  80  c0
                    //      01  41  81  c1
                    //      02  42  82  c2
                    //      03  43  83  c3
                    //      04  44  84  c4
                    //      05  45  85  c5
                    //      06  46  86  c6
                    //      07  47  87  c7
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    //      00  80  01  81
                    //      02  82  03  83
                    //      04  84  05  85
                    //      06  86  07  87
                    //      40  c0  41  c1
                    //      42  c2  43  c3
                    //      44  c4  45  c5
                    //      46  c6  47  c7
                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    //      00  01  02  03
                    //      04  05  06  07
                    //      40  41  42  43
                    //      44  45  46  47
                    //      80  81  82  83
                    //      84  85  86  87
                    //      c0  c1  c2  c3
                    //      c4  c5  c6  c7

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + out_hstep, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x8_ps(_f7, 1));
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 48)));
            pp += 64;

            // from
            //      00 11 22 33 40 51 62 73 80 91 a2 b3 c0 d1 e2 f3
            //      01 12 23 30 41 52 63 70 81 92 a3 b0 c1 d2 e3 f0
            //      20 31 02 13 60 71 42 53 a0 b1 82 93 e0 f1 c2 d3
            //      21 32 03 10 61 72 43 50 a1 b2 83 90 e1 f2 c3 d0
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
            //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);
            _f2 = _mm512_mul_ps(_f2, _descale);
            _f3 = _mm512_mul_ps(_f3, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                    _f2 = _mm512_add_ps(_f2, _c0);
                    _f3 = _mm512_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    __m512 _c2;
                    __m512 _c3;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        _c2 = _mm512_loadu_ps(pC + 32);
                        _c3 = _mm512_loadu_ps(pC + 48);
                        pC += 64;
                    }
                    if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + 16);
                        __m512 _cc2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        __m512 _cc3 = _mm512_loadu_ps(pC + c_hstep * 8 + 16);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc2, _MM_SHUFFLE(3, 2, 3, 2));
                        _c2 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(1, 0, 1, 0));
                        _c3 = _mm512_shuffle_f32x4(_cc1, _cc3, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 32;
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c2 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c3 = _mm512_loadu_ps(pC + c_hstep * 12);
                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0, _c1, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2, _c3, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                        __m128 _cc8 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc9 = _mm_loadu_ps(pC + c_hstep * 9);
                        __m128 _cca = _mm_loadu_ps(pC + c_hstep * 10);
                        __m128 _ccb = _mm_loadu_ps(pC + c_hstep * 11);
                        __m128 _ccc = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _ccd = _mm_loadu_ps(pC + c_hstep * 13);
                        __m128 _cce = _mm_loadu_ps(pC + c_hstep * 14);
                        __m128 _ccf = _mm_loadu_ps(pC + c_hstep * 15);
                        _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                        _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);
                        _MM_TRANSPOSE4_PS(_cc8, _cc9, _cca, _ccb);
                        _MM_TRANSPOSE4_PS(_ccc, _ccd, _cce, _ccf);

                        __m256 _cc04 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc4, 1);
                        __m256 _cc15 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc5, 1);
                        __m256 _cc26 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc6, 1);
                        __m256 _cc37 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc3), _cc7, 1);
                        __m256 _cc8c = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc8), _ccc, 1);
                        __m256 _cc9d = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc9), _ccd, 1);
                        __m256 _ccae = _mm256_insertf128_ps(_mm256_castps128_ps256(_cca), _cce, 1);
                        __m256 _ccbf = _mm256_insertf128_ps(_mm256_castps128_ps256(_ccb), _ccf, 1);

                        _c0 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc04), _cc8c, 1);
                        _c1 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc15), _cc9d, 1);
                        _c2 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc26), _ccae, 1);
                        _c3 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc37), _ccbf, 1);

                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                        _f2 = _mm512_add_ps(_f2, _c2);
                        _f3 = _mm512_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    __m512 _c2 = _mm512_set1_ps(pC[2] * beta);
                    __m512 _c3 = _mm512_set1_ps(pC[3] * beta);

                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    _f2 = _mm512_add_ps(_f2, _c2);
                    _f3 = _mm512_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
                //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
                //      02 12 22 32 42 52 62 72 82 92 a2 b2 c2 d2 e2 f2
                //      03 13 23 33 43 53 63 73 83 93 a3 b3 c3 d3 e3 f3
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    p0 += 64;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + 16, _tmp1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _tmp2);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _tmp3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);

                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x4_ps(_f3, 3));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)(pp + 16)));
            pp += 32;

            // from
            //      00 11 20 31 40 51 60 71 80 91 a0 b1 c0 d1 e0 f1
            //      01 10 21 30 41 50 61 70 81 90 a1 b0 c1 d0 e1 f0
            // to
            //      00 10 20 30 40 50 60 70 80 90 a0 b0 c0 d0 e0 f0
            //      01 11 21 31 41 51 61 71 81 91 a1 b1 c1 d1 e1 f1
            {
                __m512 _tmp0 = _mm512_permute_ps(_f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m512 _tmp1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm512_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm512_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale);
            _f1 = _mm512_mul_ps(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1;
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        _c1 = _mm512_loadu_ps(pC + 16);
                        pC += 32;
                    }
                    if (c_elempack == 8)
                    {
                        __m512 _cc0 = _mm512_loadu_ps(pC);
                        __m512 _cc1 = _mm512_loadu_ps(pC + c_hstep * 8);
                        _c0 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(1, 0, 1, 0));
                        _c1 = _mm512_shuffle_f32x4(_cc0, _cc1, _MM_SHUFFLE(3, 2, 3, 2));
                        pC += 16;
                    }
                    if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 4 + 4);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 8 + 4);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 12);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 12 + 4);
                        __m256 _cc02 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc2, 1);
                        __m256 _cc46 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc4), _cc6, 1);
                        __m256 _cc13 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc3, 1);
                        __m256 _cc57 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc5), _cc7, 1);
                        _c0 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc02), _cc46, 1);
                        _c1 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc13), _cc57, 1);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        _c1 = _mm512_i32gather_ps(_vindex, pC + 1, sizeof(float));
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0);
                        _f1 = _mm512_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    __m512 _c1 = _mm512_set1_ps(pC[1] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
                    _f1 = _mm512_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm512_storeu_ps(p0, _f0);
                _mm512_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_store_ps(p0, _f0);
                    _mm512_store_ps(p0 + 16, _f1);
                    p0 += 32;
                }
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _tmp1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_f1, 3));
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm512_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i*)pp));
            pp += 16;

            _f0 = _mm512_mul_ps(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 16)
                    {
                        _c0 = _mm512_loadu_ps(pC);
                        pC += 16;
                    }
                    if (c_elempack == 8)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 8);
                        _c0 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc0), _cc1, 1);
                        pC += 8;
                    }
                    if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 8);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 12);
                        __m256 _cc01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc1, 1);
                        __m256 _cc23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc3, 1);
                        _c0 = _mm512_insertf32x8(_mm512_castps256_ps512(_cc01), _cc23, 1);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(c_hstep));
                        _c0 = _mm512_i32gather_ps(_vindex, pC, sizeof(float));
                        pC += 1;
                    }
                    _f0 = _mm512_fmadd_ps(_c0, _mm512_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm512_set1_ps(pC[0] * beta);
                    _f0 = _mm512_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));

            if (output_transpose)
            {
                _mm512_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_ps(p0, _f0);
                    p0 += 16;
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    p0++;
                }
            }
        }
    }
#endif // __AVX512F__
#if !__AVX2__
    const int* pp1 = pp + max_jj * 4;
#endif
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m256 _descale = _mm256_loadu_ps((const float*)descales + i + ii);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_broadcast_f32x8(_descale);
#endif

        __m256 _c0;
#if __AVX512F__
        __m512 _c0_avx512;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm256_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm256_loadu_ps(pC);
                _c0 = _mm256_mul_ps(_c0, _mm256_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x8(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 48)));
            __m512 _f4 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 64)));
            __m512 _f5 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 80)));
            __m512 _f6 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 96)));
            __m512 _f7 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 112)));
            pp += 128;

            // _f0 = _mm512_setr_ps(00,11,22,33,44,55,66,77,0x08,0x19,0x2a,0x3b,0x4c,0x5d,0x6e,0x7f);
            // _f1 = _mm512_setr_ps(01,12,23,30,45,56,67,74,0x09,0x1a,0x2b,0x38,0x4d,0x5e,0x6f,0x7c);
            // _f2 = _mm512_setr_ps(20,31,02,13,64,75,46,57,0x28,0x39,0x0a,0x1b,0x6c,0x7d,0x4e,0x5f);
            // _f3 = _mm512_setr_ps(21,32,03,10,65,76,47,54,0x29,0x3a,0x0b,0x18,0x6d,0x7e,0x4f,0x5c);
            // _f4 = _mm512_setr_ps(04,15,26,37,40,51,62,73,0x0c,0x1d,0x2e,0x3f,0x48,0x59,0x6a,0x7b);
            // _f5 = _mm512_setr_ps(05,16,27,34,41,52,63,70,0x0d,0x1e,0x2f,0x3c,0x49,0x5a,0x6b,0x78);
            // _f6 = _mm512_setr_ps(24,35,06,17,60,71,42,53,0x2c,0x3d,0x0e,0x1f,0x68,0x79,0x4a,0x5b);
            // _f7 = _mm512_setr_ps(25,36,07,14,61,72,43,50,0x2d,0x3e,0x0f,0x1c,0x69,0x7a,0x4b,0x58);
            //

            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);

            // from
            //      00 11 22 33  44 55 66 77  08 19 2a 3b  4c 5d 6e 7f
            //      01 12 23 30  45 56 67 74  09 1a 2b 38  4d 5e 6f 7c
            //      20 31 02 13  64 75 46 57  28 39 0a 1b  6c 7d 4e 5f
            //      21 32 03 10  65 76 47 54  29 3a 0b 18  6d 7e 4f 5c
            //      04 15 26 37  40 51 62 73  0c 1d 2e 3f  48 59 6a 7b
            //      05 16 27 34  41 52 63 70  0d 1e 2f 3c  49 5a 6b 78
            //      24 35 06 17  60 71 42 53  2c 3d 0e 1f  68 79 4a 5b
            //      25 36 07 14  61 72 43 50  2d 3e 0f 1c  69 7a 4b 58

            // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
            // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
            // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
            // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
            // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
            // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
            // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
            // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

            // to
            //      00 10 20 30  44 54 64 74  08 18 28 38  4c 5c 6c 7c
            //      01 11 21 31  45 55 65 75  09 19 29 39  4d 5d 6d 7d
            //      02 12 22 32  46 56 66 76  0a 1a 2a 3a  4e 5e 6e 7e
            //      03 13 23 33  47 57 67 77  0b 1b 2b 3b  4f 5f 6f 7f
            //      04 14 24 34  40 50 60 70  0c 1c 2c 3c  48 58 68 78
            //      05 15 25 35  41 51 61 71  0d 1d 2d 3d  49 59 69 79
            //      06 16 26 36  42 52 62 72  0e 1e 2e 3e  4a 5a 6a 7a
            //      07 17 27 37  43 53 63 73  0f 1f 2f 3f  4b 5b 6b 7b
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                //      00 11 22 33  44 55 66 77  08 19 2a 3b  4c 5d 6e 7f
                //      30 01 12 23  74 45 56 67  38 09 1a 2b  7c 4d 5e 6f
                //      20 31 02 13  64 75 46 57  28 39 0a 1b  6c 7d 4e 5f
                //      10 21 32 03  54 65 76 47  18 29 3a 0b  5c 6d 7e 4f

                //      04 15 26 37  40 51 62 73  0c 1d 2e 3f  48 59 6a 7b
                //      34 05 16 27  70 41 52 63  3c 0d 1e 2f  78 49 5a 6b
                //      24 35 06 17  60 71 42 53  2c 3d 0e 1f  68 79 4a 5b
                //      14 25 36 07  50 61 72 43  1c 2d 3e 0f  58 69 7a 4b

                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp2 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                __m512 _tmp4 = _mm512_unpacklo_ps(_f4, _f7);
                __m512 _tmp5 = _mm512_unpackhi_ps(_f4, _f7);
                __m512 _tmp6 = _mm512_unpacklo_ps(_f6, _f5);
                __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f5);

                // 00 10 11 21  44 54 55 65  08 18 19 29  4c 5c 5d 6d
                // 22 32 33 03  66 76 77 47  2a 3a 3b 0b  6e 7e 7f 4f
                // 20 30 31 01  64 74 75 45  28 38 39 09  6c 7c 7d 4d
                // 02 12 13 23  46 56 57 67  0a 1a 1b 2b  4e 5e 5f 6f

                // 04 14 15 25  40 50 51 61  0c 1c 1d 2d  48 58 59 69
                // 26 36 37 07  62 72 73 43  2e 3e 3f 0f  6a 7a 7b 4b
                // 24 34 35 05  60 70 71 41  2c 3c 3d 0d  68 78 79 49
                // 06 16 17 27  42 52 53 63  0e 1e 1f 2f  4a 5a 5b 6b

                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp2)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp1)));
                _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp6)));
                _f6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));
                _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp7), _mm512_castps_pd(_tmp5)));

                // 00 10 20 30  44 54 64 74  08 18 28 38  4c 5c 6c 7c
                // 11 21 31 01  55 65 75 45  19 29 39 09  5d 6d 7d 4d
                // 02 12 22 32  46 56 66 76  0a 1a 2a 3a  4e 5e 6e 7e
                // 13 23 33 03  57 67 77 47  1b 2b 3b 0b  5f 6f 7f 4f

                // 04 14 24 34  40 50 60 70  0c 1c 2c 3c  48 58 68 78
                // 15 25 35 05  51 61 71 41  1d 2d 3d 0d  59 69 79 49
                // 06 16 26 36  42 52 62 72  0e 1e 2e 3e  4a 5a 6a 7a
                // 17 27 37 07  53 63 73 43  1f 2f 3f 0f  5b 6b 7b 4b

                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm512_permute_ps(_f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm512_permute_ps(_f7, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 10 20 30  44 54 64 74  08 18 28 38  4c 5c 6c 7c
                // 01 11 21 31  45 55 65 75  09 19 29 39  4d 5d 6d 7d
                // 02 12 22 32  46 56 66 76  0a 1a 2a 3a  4e 5e 6e 7e
                // 03 13 23 33  47 57 67 77  0b 1b 2b 3b  4f 5f 6f 7f
                // 04 14 24 34  40 50 60 70  0c 1c 2c 3c  48 58 68 78
                // 05 15 25 35  41 51 61 71  0d 1d 2d 3d  49 59 69 79
                // 06 16 26 36  42 52 62 72  0e 1e 2e 3e  4a 5a 6a 7a
                // 07 17 27 37  43 53 63 73  0f 1f 2f 3f  4b 5b 6b 7b

                _tmp0 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp1 = _mm512_shuffle_f32x4(_f0, _f4, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp2 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp3 = _mm512_shuffle_f32x4(_f1, _f5, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp4 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp5 = _mm512_shuffle_f32x4(_f2, _f6, _MM_SHUFFLE(2, 3, 3, 2));
                _tmp6 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(0, 1, 1, 0));
                _tmp7 = _mm512_shuffle_f32x4(_f3, _f7, _MM_SHUFFLE(2, 3, 3, 2));

                // 00 10 20 30  44 54 64 74  40 50 60 70  04 14 24 34
                // 08 18 28 38  4c 5c 6c 7c  48 58 68 78  0c 1c 2c 3c
                // 01 11 21 31  45 55 65 75  41 51 61 71  05 15 25 35

                _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                _f2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                _f3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                _f4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(1, 3, 1, 3));
                _f5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(1, 3, 1, 3));
                _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(1, 3, 1, 3));
                _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(1, 3, 1, 3));

                // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f
            }

            // NCNN_LOGE("-----");

            // print(_f0);
            // print(_f1);
            // print(_f2);
            // print(_f3);
            // print(_f4);
            // print(_f5);
            // print(_f6);
            // print(_f7);

            _f0 = _mm512_mul_ps(_f0, _descale_avx512);
            _f1 = _mm512_mul_ps(_f1, _descale_avx512);
            _f2 = _mm512_mul_ps(_f2, _descale_avx512);
            _f3 = _mm512_mul_ps(_f3, _descale_avx512);
            _f4 = _mm512_mul_ps(_f4, _descale_avx512);
            _f5 = _mm512_mul_ps(_f5, _descale_avx512);
            _f6 = _mm512_mul_ps(_f6, _descale_avx512);
            _f7 = _mm512_mul_ps(_f7, _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                    _f4 = _mm512_add_ps(_f4, _c0_avx512);
                    _f5 = _mm512_add_ps(_f5, _c0_avx512);
                    _f6 = _mm512_add_ps(_f6, _c0_avx512);
                    _f7 = _mm512_add_ps(_f7, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                    // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                    // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                    // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                    // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                    // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                    // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                    // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    __m512 _c4_avx512;
                    __m512 _c5_avx512;
                    __m512 _c6_avx512;
                    __m512 _c7_avx512;
                    if (c_elempack == 8)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + 64);
                        _c5_avx512 = _mm512_loadu_ps(pC + 80);
                        _c6_avx512 = _mm512_loadu_ps(pC + 96);
                        _c7_avx512 = _mm512_loadu_ps(pC + 112);

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c4_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c5_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c6_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c3_avx512, _c7_avx512, _MM_SHUFFLE(3, 2, 3, 2));

                        _c0_avx512 = _tmp0;
                        _c1_avx512 = _tmp1;
                        _c2_avx512 = _tmp2;
                        _c3_avx512 = _tmp3;
                        _c4_avx512 = _tmp4;
                        _c5_avx512 = _tmp5;
                        _c6_avx512 = _tmp6;
                        _c7_avx512 = _tmp7;

                        pC += 128;
                    }
                    if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 16);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 32);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 4 + 48);

                        // 00   01   02   03
                        // 04   05   06   07

                        // 08   09   0a   0b
                        // 0c   0d   0e   0f

                        // 40   41   42   43
                        // 44   45   46   47

                        // 48   49   4a   4b
                        // 4c   4d   4e   4f

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c0_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c1_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp4 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp5 = _mm512_shuffle_f32x4(_c4_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        __m512 _tmp6 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        __m512 _tmp7 = _mm512_shuffle_f32x4(_c5_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        // 00   02   08   0a
                        // 01   03   09   0b
                        // 04   06   0c   0e
                        // 05   07   0d   0f

                        // 40   42   48   4a
                        // 41   43   49   4b
                        // 44   46   4c   4e
                        // 45   47   4d   4f

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp4, _MM_SHUFFLE(3, 1, 3, 1));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(2, 0, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp6, _MM_SHUFFLE(3, 1, 3, 1));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                        // 00 08 40 48
                        // 01 09 41 49
                        // 02 0a 42 4a

                        _c0_avx512 = _mm512_shuffle_f32x4(_c0_avx512, _c0_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_c1_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_c2_avx512, _c2_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_c3_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_c4_avx512, _c4_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_c5_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_c6_avx512, _c6_avx512, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_c7_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 64;
                    }
                    if (c_elempack == 1)
                    {
                        // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                        // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                        // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                        // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                        // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                        // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                        // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                        // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        _c4_avx512 = _mm512_loadu_ps(pC + c_hstep * 4);
                        _c5_avx512 = _mm512_loadu_ps(pC + c_hstep * 5);
                        _c6_avx512 = _mm512_loadu_ps(pC + c_hstep * 6);
                        _c7_avx512 = _mm512_loadu_ps(pC + c_hstep * 7);

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpacklo_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp3 = _mm512_unpacklo_ps(_c6_avx512, _c7_avx512);
                        __m512 _tmp4 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp5 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp6 = _mm512_unpackhi_ps(_c4_avx512, _c5_avx512);
                        __m512 _tmp7 = _mm512_unpackhi_ps(_c6_avx512, _c7_avx512);

                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c4_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c5_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                        _c6_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                        _c7_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                        _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp2 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp3 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(2, 0, 2, 0));
                        _tmp4 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp5 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp6 = _mm512_shuffle_f32x4(_c4_avx512, _c5_avx512, _MM_SHUFFLE(3, 1, 3, 1));
                        _tmp7 = _mm512_shuffle_f32x4(_c6_avx512, _c7_avx512, _MM_SHUFFLE(3, 1, 3, 1));

                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                        _c4_avx512 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                        _c5_avx512 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                        _c6_avx512 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                        _c7_avx512 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                        pC += 16;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                        _f4 = _mm512_add_ps(_f4, _c4_avx512);
                        _f5 = _mm512_add_ps(_f5, _c5_avx512);
                        _f6 = _mm512_add_ps(_f6, _c6_avx512);
                        _f7 = _mm512_add_ps(_f7, _c7_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                        _f4 = _mm512_fmadd_ps(_c4_avx512, _beta, _f4);
                        _f5 = _mm512_fmadd_ps(_c5_avx512, _beta, _f5);
                        _f6 = _mm512_fmadd_ps(_c6_avx512, _beta, _f6);
                        _f7 = _mm512_fmadd_ps(_c7_avx512, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    __m512 _cc0 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _cc1 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _cc2 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _cc3 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _c0_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c1_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c2_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c3_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(2, 2, 0, 0));
                    __m512 _c4_avx512 = _mm512_shuffle_f32x4(_cc0, _cc0, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c5_avx512 = _mm512_shuffle_f32x4(_cc1, _cc1, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c6_avx512 = _mm512_shuffle_f32x4(_cc2, _cc2, _MM_SHUFFLE(3, 3, 1, 1));
                    __m512 _c7_avx512 = _mm512_shuffle_f32x4(_cc3, _cc3, _MM_SHUFFLE(3, 3, 1, 1));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    _f4 = _mm512_add_ps(_f4, _c4_avx512);
                    _f5 = _mm512_add_ps(_f5, _c5_avx512);
                    _f6 = _mm512_add_ps(_f6, _c6_avx512);
                    _f7 = _mm512_add_ps(_f7, _c7_avx512);

                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
                _f4 = _mm512_mul_ps(_f4, _alpha);
                _f5 = _mm512_mul_ps(_f5, _alpha);
                _f6 = _mm512_mul_ps(_f6, _alpha);
                _f7 = _mm512_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
                // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                if (out_elempack == 16)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);

                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + 8, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + 16, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + 16 + 8, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + 16 * 2, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + 16 * 2 + 8, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + 16 * 3, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + 16 * 3 + 8, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + 16 * 4, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + 16 * 4 + 8, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + 16 * 5, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + 16 * 5 + 8, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + 16 * 6, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + 16 * 6 + 8, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + 16 * 7, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + 16 * 7 + 8, _mm512_extractf32x8_ps(_f7, 1));
                }
                if (out_elempack == 8)
                {
                    transpose16x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 16 * 2, _f2);
                    _mm512_storeu_ps(p0 + 16 * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 2, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16 * 3, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    transpose16x4_ps(_f4, _f5, _f6, _f7);

                    // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                    // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                    // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                    // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b

                    // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                    // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                    // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                    // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 8 + 16, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 12 + 16, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + out_hstep, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x8_ps(_f2, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x8_ps(_f3, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x8_ps(_f4, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x8_ps(_f5, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x8_ps(_f6, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x8_ps(_f7, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x8_ps(_f1, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x8_ps(_f2, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x8_ps(_f3, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x8_ps(_f4, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x8_ps(_f5, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x8_ps(_f6, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x8_ps(_f7, 1));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 8)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 2, 3, 2));

                    _mm512_storeu_ps(p0, _tmp0);
                    _mm512_storeu_ps(p0 + 16, _tmp1);
                    _mm512_storeu_ps(p0 + 32, _tmp2);
                    _mm512_storeu_ps(p0 + 48, _tmp3);
                    _mm512_storeu_ps(p0 + 64, _tmp4);
                    _mm512_storeu_ps(p0 + 80, _tmp5);
                    _mm512_storeu_ps(p0 + 96, _tmp6);
                    _mm512_storeu_ps(p0 + 112, _tmp7);
                    p0 += 128;
                }
                if (out_elempack == 4)
                {
                    // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                    // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                    // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                    // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                    // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                    // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                    // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                    // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512 _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512 _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    // 00  08  01  09
                    // 02  0a  03  0b
                    // 04  0c  05  0d
                    // 06  0e  06  0f
                    // 40  48  41  49

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _f4 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _f7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 16, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 32, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 4 + 48, _f7);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    // 00 10 20 30 40 50 60 70    08 18 28 38 48 58 68 78
                    // 01 11 21 31 41 51 61 71    09 19 29 39 49 59 69 79
                    // 02 12 22 32 42 52 62 72    0a 1a 2a 3a 4a 5a 6a 7a
                    // 03 13 23 33 43 53 63 73    0b 1b 2b 3b 4b 5b 6b 7b
                    // 04 14 24 34 44 54 64 74    0c 1c 2c 3c 4c 5c 6c 7c
                    // 05 15 25 35 45 55 65 75    0d 1d 2d 3d 4d 5d 6d 7d
                    // 06 16 26 36 46 56 66 76    0e 1e 2e 3e 4e 5e 6e 7e
                    // 07 17 27 37 47 57 67 77    0f 1f 2f 3f 4f 5f 6f 7f

                    __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f3);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_f4, _f5);
                    __m512 _tmp3 = _mm512_unpacklo_ps(_f6, _f7);
                    __m512 _tmp4 = _mm512_unpackhi_ps(_f0, _f1);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_f2, _f3);
                    __m512 _tmp6 = _mm512_unpackhi_ps(_f4, _f5);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_f6, _f7);

                    _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f1 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f2 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f5 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));
                    _f6 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp4), _mm512_castps_pd(_tmp5)));
                    _f7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp6), _mm512_castps_pd(_tmp7)));

                    _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_f4, _f5, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_f6, _f7, _MM_SHUFFLE(3, 1, 3, 1));

                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp0, _MM_SHUFFLE(3, 1, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp1, _tmp1, _MM_SHUFFLE(3, 1, 2, 0));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp2, _MM_SHUFFLE(3, 1, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp3, _tmp3, _MM_SHUFFLE(3, 1, 2, 0));
                    _f4 = _mm512_shuffle_f32x4(_tmp4, _tmp4, _MM_SHUFFLE(3, 1, 2, 0));
                    _f5 = _mm512_shuffle_f32x4(_tmp5, _tmp5, _MM_SHUFFLE(3, 1, 2, 0));
                    _f6 = _mm512_shuffle_f32x4(_tmp6, _tmp6, _MM_SHUFFLE(3, 1, 2, 0));
                    _f7 = _mm512_shuffle_f32x4(_tmp7, _tmp7, _MM_SHUFFLE(3, 1, 2, 0));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm512_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm512_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm512_storeu_ps(p0 + out_hstep * 7, _f7);

                    p0 += 16;
                }
            }
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
            __m256 _f4 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 32)));
            __m256 _f5 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 40)));
            __m256 _f6 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 48)));
            __m256 _f7 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 56)));
            pp += 64;

            // from
            //      00 11 22 33 44 55 66 77
            //      01 12 23 30 45 56 67 74
            //      60 71 42 53 24 35 06 17
            //      61 72 43 50 25 36 07 14
            //      02 13 20 31 46 57 64 75
            //      03 10 21 32 47 54 65 76
            //      62 73 40 51 26 37 04 15
            //      63 70 41 52 27 34 05 16

            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            //      04 14 24 34 44 54 64 74
            //      05 15 25 35 45 55 65 75
            //      06 16 26 36 46 56 66 76
            //      07 17 27 37 47 57 67 77
            {
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp2 = _f2;
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _mm256_shuffle_ps(_f4, _f4, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _tmp5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(0, 3, 2, 1));
                __m256 _tmp6 = _mm256_shuffle_ps(_f6, _f6, _MM_SHUFFLE(1, 0, 3, 2));
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(0, 3, 2, 1));

                // 00 11 22 33 44 55 66 77
                // 30 01 12 23 74 45 56 67
                // 60 71 42 53 24 35 06 17
                // 50 61 72 43 14 25 36 07
                // 20 31 02 13 64 75 46 57
                // 10 21 32 03 54 65 76 47
                // 40 51 62 73 04 15 26 37
                // 70 41 52 63 34 05 16 27

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp5, _tmp3, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp4, _tmp2, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp1, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp6, _tmp0, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_ps(_tmp3, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_ps(_tmp2, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_ps(_tmp7, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                // 00 11 22 33 40 51 62 73
                // 10 21 32 03 50 61 72 43
                // 20 31 02 13 60 71 42 53
                // 30 01 12 23 70 41 52 63
                // 04 15 26 37 44 55 66 77
                // 14 25 36 07 54 65 76 47
                // 24 35 06 17 64 75 46 57
                // 34 05 16 27 74 45 56 67

                _tmp0 = _mm256_unpacklo_ps(_f0, _f1);
                _tmp1 = _mm256_unpacklo_ps(_f2, _f3);
                _tmp2 = _mm256_unpackhi_ps(_f2, _f3);
                _tmp3 = _mm256_unpackhi_ps(_f0, _f1);
                _tmp4 = _mm256_unpacklo_ps(_f4, _f5);
                _tmp5 = _mm256_unpacklo_ps(_f6, _f7);
                _tmp6 = _mm256_unpackhi_ps(_f6, _f7);
                _tmp7 = _mm256_unpackhi_ps(_f4, _f5);

                // 00 10 11 21 40 50 51 61
                // 20 30 31 01 60 70 71 41
                // 02 12 13 23 42 52 53 63
                // 22 32 33 03 62 72 73 43

                // 04 14 15 25 44 54 55 65
                // 24 34 35 05 64 74 75 45
                // 06 16 17 27 46 56 57 67
                // 26 36 37 07 66 76 77 47

                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));
                _f7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));

                // 00 10 20 30 40 50 60 70
                // 11 21 31 01 51 61 71 41
                // 02 12 22 32 42 52 62 72
                // 13 23 33 03 53 63 73 43
                // 04 14 24 34 44 54 64 74
                // 15 25 35 05 55 65 75 45
                // 06 16 26 36 46 56 66 76
                // 17 27 37 07 57 67 77 47

                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
            }
#else  // __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
            __m256 _f4 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f5 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 8)));
            __m256 _f6 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 16)));
            __m256 _f7 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 24)));
            pp += 32;
            pp1 += 32;

            // from
            //      00 11 22 33 04 15 26 37
            //      20 31 02 13 24 35 06 17
            //      01 12 23 30 05 16 27 34
            //      21 32 03 10 25 36 07 14
            //      40 51 62 73 44 55 66 77
            //      60 71 42 53 64 75 46 57
            //      41 52 63 70 45 56 67 74
            //      61 72 43 50 65 76 47 54

            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            //      04 14 24 34 44 54 64 74
            //      05 15 25 35 45 55 65 75
            //      06 16 26 36 46 56 66 76
            //      07 17 27 37 47 57 67 77
            {
                __m256 _tmp0 = _f0;
                __m256 _tmp1 = _f1;
                __m256 _tmp2 = _mm256_shuffle_ps(_f2, _f2, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp4 = _f4;
                __m256 _tmp5 = _f5;
                __m256 _tmp6 = _mm256_shuffle_ps(_f6, _f6, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));

                // 00 11 22 33 04 15 26 37
                // 20 31 02 13 24 35 06 17
                // 30 01 12 23 34 05 16 27
                // 10 21 32 03 14 25 36 07
                // 40 51 62 73 44 55 66 77
                // 60 71 42 53 64 75 46 57
                // 70 41 52 63 74 45 56 67
                // 50 61 72 43 54 65 76 47

                _f0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                _f1 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                _f2 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                _f3 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                _f4 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                _f5 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                _f6 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                _f7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                // 00 11 22 33 40 51 62 73
                // 20 31 02 13 60 71 42 53
                // 30 01 12 23 70 41 52 63
                // 10 21 32 03 50 61 72 43
                // 04 15 26 37 44 55 66 77
                // 24 35 06 17 64 75 46 57
                // 34 05 16 27 74 45 56 67
                // 14 25 36 07 54 65 76 47

                _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                _tmp1 = _mm256_unpacklo_ps(_f1, _f2);
                _tmp2 = _mm256_unpackhi_ps(_f1, _f2);
                _tmp3 = _mm256_unpackhi_ps(_f0, _f3);
                _tmp4 = _mm256_unpacklo_ps(_f4, _f7);
                _tmp5 = _mm256_unpacklo_ps(_f5, _f6);
                _tmp6 = _mm256_unpackhi_ps(_f5, _f6);
                _tmp7 = _mm256_unpackhi_ps(_f4, _f7);

                // 00 10 11 21 40 50 51 61
                // 20 30 31 01 60 70 71 41
                // 02 12 13 23 42 52 53 63
                // 22 32 33 03 62 72 73 43
                // 04 14 15 25 44 54 55 65
                // 24 34 35 05 64 74 75 45
                // 06 16 17 27 46 56 57 67
                // 26 36 37 07 66 76 77 47

                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp1)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp2), _mm256_castps_pd(_tmp3)));
                _f4 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f5 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp4), _mm256_castps_pd(_tmp5)));
                _f6 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));
                _f7 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp6), _mm256_castps_pd(_tmp7)));

                // 00 10 20 30 40 50 60 70
                // 11 21 31 01 51 61 71 41
                // 02 12 22 32 42 52 62 72
                // 13 23 33 03 53 63 73 43
                // 04 14 24 34 44 54 64 74
                // 15 25 35 05

                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                _f5 = _mm256_shuffle_ps(_f5, _f5, _MM_SHUFFLE(2, 1, 0, 3));
                _f7 = _mm256_shuffle_ps(_f7, _f7, _MM_SHUFFLE(2, 1, 0, 3));
            }
#endif // __AVX2__

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);
            _f2 = _mm256_mul_ps(_f2, _descale);
            _f3 = _mm256_mul_ps(_f3, _descale);
            _f4 = _mm256_mul_ps(_f4, _descale);
            _f5 = _mm256_mul_ps(_f5, _descale);
            _f6 = _mm256_mul_ps(_f6, _descale);
            _f7 = _mm256_mul_ps(_f7, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c0);
                    _f6 = _mm256_add_ps(_f6, _c0);
                    _f7 = _mm256_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    __m256 _c4;
                    __m256 _c5;
                    __m256 _c6;
                    __m256 _c7;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        _c4 = _mm256_loadu_ps(pC + 32);
                        _c5 = _mm256_loadu_ps(pC + 40);
                        _c6 = _mm256_loadu_ps(pC + 48);
                        _c7 = _mm256_loadu_ps(pC + 56);
                        pC += 64;
                    }
                    if (c_elempack == 4)
                    {
                        __m256 _tmp0 = _mm256_loadu_ps(pC);
                        __m256 _tmp1 = _mm256_loadu_ps(pC + 8);
                        __m256 _tmp2 = _mm256_loadu_ps(pC + 16);
                        __m256 _tmp3 = _mm256_loadu_ps(pC + 24);
                        __m256 _tmp4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _tmp5 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                        __m256 _tmp6 = _mm256_loadu_ps(pC + c_hstep * 4 + 16);
                        __m256 _tmp7 = _mm256_loadu_ps(pC + c_hstep * 4 + 24);
                        _c0 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_tmp0, _tmp4, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_tmp1, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                        _c4 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 2, 0, 0));
                        _c5 = _mm256_permute2f128_ps(_tmp2, _tmp6, _MM_SHUFFLE(0, 3, 0, 1));
                        _c6 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                        _c7 = _mm256_permute2f128_ps(_tmp3, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + c_hstep);
                        _c2 = _mm256_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm256_loadu_ps(pC + c_hstep * 3);
                        _c4 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c5 = _mm256_loadu_ps(pC + c_hstep * 5);
                        _c6 = _mm256_loadu_ps(pC + c_hstep * 6);
                        _c7 = _mm256_loadu_ps(pC + c_hstep * 7);
                        transpose8x8_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                        _f4 = _mm256_add_ps(_f4, _c4);
                        _f5 = _mm256_add_ps(_f5, _c5);
                        _f6 = _mm256_add_ps(_f6, _c6);
                        _f7 = _mm256_add_ps(_f7, _c7);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                        _f4 = _mm256_comp_fmadd_ps(_c4, _beta, _f4);
                        _f5 = _mm256_comp_fmadd_ps(_c5, _beta, _f5);
                        _f6 = _mm256_comp_fmadd_ps(_c6, _beta, _f6);
                        _f7 = _mm256_comp_fmadd_ps(_c7, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);

                    _c0 = _mm256_set1_ps(pC[4] * beta);
                    _c1 = _mm256_set1_ps(pC[5] * beta);
                    _c2 = _mm256_set1_ps(pC[6] * beta);
                    _c3 = _mm256_set1_ps(pC[7] * beta);

                    _f4 = _mm256_add_ps(_f4, _c0);
                    _f5 = _mm256_add_ps(_f5, _c1);
                    _f6 = _mm256_add_ps(_f6, _c2);
                    _f7 = _mm256_add_ps(_f7, _c3);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
                _f4 = _mm256_mul_ps(_f4, _alpha);
                _f5 = _mm256_mul_ps(_f5, _alpha);
                _f6 = _mm256_mul_ps(_f6, _alpha);
                _f7 = _mm256_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 8)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + 32, _f4);
                    _mm256_storeu_ps(p0 + 40, _f5);
                    _mm256_storeu_ps(p0 + 48, _f6);
                    _mm256_storeu_ps(p0 + 56, _f7);
                }
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    transpose8x4_ps(_f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _f5);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 16, _f6);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 24, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + out_hstep, _f1);
                    _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm256_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm256_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm256_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    _mm256_storeu_ps(p0 + 32, _f4);
                    _mm256_storeu_ps(p0 + 40, _f5);
                    _mm256_storeu_ps(p0 + 48, _f6);
                    _mm256_storeu_ps(p0 + 56, _f7);
                    p0 += 64;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_f4, _f5, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_f6, _f7, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp4 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp5 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp6 = _mm256_permute2f128_ps(_f4, _f5, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp7 = _mm256_permute2f128_ps(_f6, _f7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + 8, _tmp1);
                    _mm256_storeu_ps(p0 + 16, _tmp2);
                    _mm256_storeu_ps(p0 + 24, _tmp3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp4);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _tmp5);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 16, _tmp6);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 24, _tmp7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_f0, _f1, _f2, _f3, _f4, _f5, _f6, _f7);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + out_hstep, _f1);
                    _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm256_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm256_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm256_storeu_ps(p0 + out_hstep * 7, _f7);
                    p0 += 8;
                }
            }
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f2 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 16)));
            __m256 _f3 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 24)));
            pp += 32;
#else
            __m256 _f01l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f23l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            __m256 _f01h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f23h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp1 + 8)));
            __m256 _f0 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f1 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
            __m256 _f2 = _mm256_permute2f128_ps(_f23l, _f23h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f3 = _mm256_permute2f128_ps(_f23l, _f23h, _MM_SHUFFLE(0, 3, 0, 1));
            pp += 16;
            pp1 += 16;
#endif

            // from
            //      00 11 22 33
            //      01 12 23 30
            //      20 31 02 13
            //      21 32 03 10

            // from
            //      00 11 22 33 40 51 62 73
            //      01 12 23 30 41 52 63 70
            //      20 31 02 13 60 71 42 53
            //      21 32 03 10 61 72 43 50
            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            //      02 12 22 32 42 52 62 72
            //      03 13 23 33 43 53 63 73
            {
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m256 _tmp0 = _mm256_unpacklo_ps(_f0, _f3);
                __m256 _tmp1 = _mm256_unpackhi_ps(_f0, _f3);
                __m256 _tmp2 = _mm256_unpacklo_ps(_f2, _f1);
                __m256 _tmp3 = _mm256_unpackhi_ps(_f2, _f1);
                _f0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp0), _mm256_castps_pd(_tmp2)));
                _f2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(_tmp3), _mm256_castps_pd(_tmp1)));
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm256_shuffle_ps(_f3, _f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);
            _f2 = _mm256_mul_ps(_f2, _descale);
            _f3 = _mm256_mul_ps(_f3, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                    _f2 = _mm256_add_ps(_f2, _c0);
                    _f3 = _mm256_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    __m256 _c2;
                    __m256 _c3;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        _c2 = _mm256_loadu_ps(pC + 16);
                        _c3 = _mm256_loadu_ps(pC + 24);
                        pC += 32;
                    }
                    if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + 8);
                        __m256 _cc2 = _mm256_loadu_ps(pC + c_hstep * 4);
                        __m256 _cc3 = _mm256_loadu_ps(pC + c_hstep * 4 + 8);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc2, _MM_SHUFFLE(0, 3, 0, 1));
                        _c2 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 2, 0, 0));
                        _c3 = _mm256_permute2f128_ps(_cc1, _cc3, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        // __m256i _vindex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        // _c0 = _mm256_i32gather_ps(pC, _vindex, c_hstep * sizeof(float));
                        // _c1 = _mm256_i32gather_ps(pC + 1, _vindex, c_hstep * sizeof(float));
                        // _c2 = _mm256_i32gather_ps(pC + 2, _vindex, c_hstep * sizeof(float));
                        // _c3 = _mm256_i32gather_ps(pC + 3, _vindex, c_hstep * sizeof(float));

                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep);
                        __m128 _cc2 = _mm_loadu_ps(pC + c_hstep * 2);
                        __m128 _cc3 = _mm_loadu_ps(pC + c_hstep * 3);
                        __m128 _cc4 = _mm_loadu_ps(pC + c_hstep * 4);
                        __m128 _cc5 = _mm_loadu_ps(pC + c_hstep * 5);
                        __m128 _cc6 = _mm_loadu_ps(pC + c_hstep * 6);
                        __m128 _cc7 = _mm_loadu_ps(pC + c_hstep * 7);
                        _MM_TRANSPOSE4_PS(_cc0, _cc1, _cc2, _cc3);
                        _MM_TRANSPOSE4_PS(_cc4, _cc5, _cc6, _cc7);

                        _c0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc4, 1);
                        _c1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc1), _cc5, 1);
                        _c2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc2), _cc6, 1);
                        _c3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc3), _cc7, 1);

                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                        _f2 = _mm256_add_ps(_f2, _c2);
                        _f3 = _mm256_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm256_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm256_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    __m256 _c2 = _mm256_set1_ps(pC[2] * beta);
                    __m256 _c3 = _mm256_set1_ps(pC[3] * beta);

                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    _f2 = _mm256_add_ps(_f2, _c2);
                    _f3 = _mm256_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
                _f2 = _mm256_mul_ps(_f2, _alpha);
                _f3 = _mm256_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + out_hstep, _f1);
                    _mm256_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm256_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    _mm256_storeu_ps(p0 + 16, _f2);
                    _mm256_storeu_ps(p0 + 24, _f3);
                    p0 += 32;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp2 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));
                    __m256 _tmp3 = _mm256_permute2f128_ps(_f2, _f3, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + 8, _tmp1);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp2);
                    _mm256_storeu_ps(p0 + out_hstep * 4 + 8, _tmp3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _mm256_extractf128_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep, _mm256_extractf128_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 2, _mm256_extractf128_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 3, _mm256_extractf128_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm256_extractf128_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 5, _mm256_extractf128_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 6, _mm256_extractf128_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 7, _mm256_extractf128_ps(_f3, 1));
                    p0 += 4;
                }
            }
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(pp + 8)));
            pp += 16;
#else
            __m256 _f01l = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            __m256 _f01h = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp1));
            __m256 _f0 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _f1 = _mm256_permute2f128_ps(_f01l, _f01h, _MM_SHUFFLE(0, 3, 0, 1));
            pp += 8;
            pp1 += 8;
#endif

            // from
            //      00 11 20 31 40 51 60 71
            //      01 10 21 30 41 50 61 70
            // to
            //      00 10 20 30 40 50 60 70
            //      01 11 21 31 41 51 61 71
            {
                __m256 _tmp0 = _mm256_shuffle_ps(_f0, _f0, _MM_SHUFFLE(3, 1, 2, 0));
                __m256 _tmp1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(0, 2, 3, 1));
                _f0 = _mm256_unpacklo_ps(_tmp0, _tmp1);
                _f1 = _mm256_unpackhi_ps(_tmp0, _tmp1);
                _f1 = _mm256_shuffle_ps(_f1, _f1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm256_mul_ps(_f0, _descale);
            _f1 = _mm256_mul_ps(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m256 _c1;
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        _c1 = _mm256_loadu_ps(pC + 8);
                        pC += 16;
                    }
                    if (c_elempack == 4)
                    {
                        __m256 _cc0 = _mm256_loadu_ps(pC);
                        __m256 _cc1 = _mm256_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 2, 0, 0));
                        _c1 = _mm256_permute2f128_ps(_cc0, _cc1, _MM_SHUFFLE(0, 3, 0, 1));
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
                        _c1 = _mm256_i32gather_ps(pC + 1, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
                        _c1 = _mm256_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1], pC[c_hstep * 4 + 1], pC[c_hstep * 5 + 1], pC[c_hstep * 6 + 1], pC[c_hstep * 7 + 1]);
#endif
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm256_add_ps(_f0, _c0);
                        _f1 = _mm256_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m256 _beta = _mm256_set1_ps(beta);
                        _f0 = _mm256_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm256_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    __m256 _c1 = _mm256_set1_ps(pC[1] * beta);
                    _f0 = _mm256_add_ps(_f0, _c0);
                    _f1 = _mm256_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m256 _alpha = _mm256_set1_ps(alpha);
                _f0 = _mm256_mul_ps(_f0, _alpha);
                _f1 = _mm256_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm256_storeu_ps(p0, _f0);
                _mm256_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    _mm256_storeu_ps(p0 + 8, _f1);
                    p0 += 16;
                }
                if (out_elempack == 4)
                {
                    __m256 _tmp0 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 2, 0, 0));
                    __m256 _tmp1 = _mm256_permute2f128_ps(_f0, _f1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_storeu_ps(p0, _tmp0);
                    _mm256_storeu_ps(p0 + out_hstep * 4, _tmp1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    _mm256_storeu_ps(sum0, _f0);
                    _mm256_storeu_ps(sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 4 + 1] = sum1[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 5 + 1] = sum1[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 6 + 1] = sum1[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0[out_hstep * 7 + 1] = sum1[7];

                    p0 += 2;
                }
            }
        }
        for (; jj < max_jj; jj++)
        {
#if __AVX2__
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)pp));
            pp += 8;
#else
            __m128i _f0l = _mm_loadu_si128((const __m128i*)pp);
            __m128i _f0h = _mm_loadu_si128((const __m128i*)pp1);
            __m256 _f0 = _mm256_cvtepi32_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_f0l), _f0h, 1));
            pp += 4;
            pp1 += 4;
#endif

            _f0 = _mm256_mul_ps(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm256_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 8)
                    {
                        _c0 = _mm256_loadu_ps(pC);
                        pC += 8;
                    }
                    if (c_elempack == 4)
                    {
                        __m128 _cc0 = _mm_loadu_ps(pC);
                        __m128 _cc1 = _mm_loadu_ps(pC + c_hstep * 4);
                        _c0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_cc0), _cc1, 1);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
#if __AVX2__
                        __m256i _vindex = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32(c_hstep));
                        _c0 = _mm256_i32gather_ps(pC, _vindex, sizeof(float));
#else
                        _c0 = _mm256_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3], pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]);
#endif
                        pC += 1;
                    }
                    _f0 = _mm256_comp_fmadd_ps(_c0, _mm256_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm256_set1_ps(pC[0] * beta);
                    _f0 = _mm256_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm256_mul_ps(_f0, _mm256_set1_ps(alpha));

            if (output_transpose)
            {
                _mm256_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _f0);
                    p0 += 8;
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _mm256_extractf128_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm256_extractf128_ps(_f0, 1));
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    _mm256_storeu_ps(sum0, _f0);
                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 4] = sum0[4];
                    p0[out_hstep * 5] = sum0[5];
                    p0[out_hstep * 6] = sum0[6];
                    p0[out_hstep * 7] = sum0[7];
                    p0++;
                }
            }
        }

#if !__AVX2__
        pp = pp1;
        pp1 = pp + max_jj * 4;
#endif
    }
#endif // __AVX__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;
        }

        __m128 _descale = _mm_loadu_ps((const float*)descales + i + ii);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_broadcast_f32x4(_descale);
#endif

        __m128 _c0;
#if __AVX512F__
        __m512 _c0_avx512;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = _mm_set1_ps(pC[0] * beta);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(pC[0] * beta);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = _mm_loadu_ps(pC);
                _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
#if __AVX512F__
                _c0_avx512 = _mm512_broadcast_f32x4(_c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp));
            __m512 _f1 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 16)));
            __m512 _f2 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 32)));
            __m512 _f3 = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)(pp + 48)));

            // from
            //      00 11 22 33 04 15 26 37 08 19 2a 3b 0c 1d 2e 3f
            //      01 12 23 30 05 16 27 34 09 1a 2b 38 0d 1e 2f 3c
            //      20 31 02 13 24 35 06 17 28 3a 0a 1b 2c 3d 0e 1f
            //      21 32 03 10 25 36 07 14 29 3a 0b 18 2d 3e 0f 1c
            // to
            //      00 10 20 30 04 14 24 34 08 18 28 38 0c 1c 2c 3c
            //      01 11 21 31 05 15 25 35 09 19 29 39 0d 1d 2d 3d
            //      02 12 22 32 06 16 26 36 0a 1a 2a 3a 0e 1e 2e 3e
            //      03 13 23 33 07 17 27 37 0b 1b 2b 3b 0f 1f 2f 3f
            {
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
                __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f3);
                __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f1);
                __m512 _tmp2 = _mm512_unpackhi_ps(_f0, _f3);
                __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f1);
                _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp3), _mm512_castps_pd(_tmp2)));
                _f1 = _mm512_permute_ps(_f1, _MM_SHUFFLE(2, 1, 0, 3));
                _f3 = _mm512_permute_ps(_f3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            _f0 = _mm512_mul_ps(_f0, _descale_avx512);
            _f1 = _mm512_mul_ps(_f1, _descale_avx512);
            _f2 = _mm512_mul_ps(_f2, _descale_avx512);
            _f3 = _mm512_mul_ps(_f3, _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    _f2 = _mm512_add_ps(_f2, _c0_avx512);
                    _f3 = _mm512_add_ps(_f3, _c0_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    __m512 _c1_avx512;
                    __m512 _c2_avx512;
                    __m512 _c3_avx512;
                    if (c_elempack == 4)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + 16);
                        _c2_avx512 = _mm512_loadu_ps(pC + 32);
                        _c3_avx512 = _mm512_loadu_ps(pC + 48);
                        pC += 64;

                        __m512 _tmp0 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp1 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(1, 0, 1, 0));
                        __m512 _tmp2 = _mm512_shuffle_f32x4(_c0_avx512, _c1_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        __m512 _tmp3 = _mm512_shuffle_f32x4(_c2_avx512, _c3_avx512, _MM_SHUFFLE(3, 2, 3, 2));
                        _c0_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                        _c1_avx512 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                        _c2_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                        _c3_avx512 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    }
                    if (c_elempack == 1)
                    {
                        _c0_avx512 = _mm512_loadu_ps(pC);
                        _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                        _c2_avx512 = _mm512_loadu_ps(pC + c_hstep * 2);
                        _c3_avx512 = _mm512_loadu_ps(pC + c_hstep * 3);
                        pC += 16;

                        __m512 _tmp0 = _mm512_unpacklo_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp1 = _mm512_unpacklo_ps(_c2_avx512, _c3_avx512);
                        __m512 _tmp2 = _mm512_unpackhi_ps(_c0_avx512, _c1_avx512);
                        __m512 _tmp3 = _mm512_unpackhi_ps(_c2_avx512, _c3_avx512);
                        _c0_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c1_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                        _c2_avx512 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                        _c3_avx512 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                        _f2 = _mm512_add_ps(_f2, _c2_avx512);
                        _f3 = _mm512_add_ps(_f3, _c3_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                        _f2 = _mm512_fmadd_ps(_c2_avx512, _beta, _f2);
                        _f3 = _mm512_fmadd_ps(_c3_avx512, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    __m512 _cc = _mm512_loadu_ps(pC);
                    _cc = _mm512_mul_ps(_cc, _mm512_set1_ps(beta));
                    _c0_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _c1_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _c2_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _c3_avx512 = _mm512_permute_ps(_cc, _MM_SHUFFLE(3, 3, 3, 3));

                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    _f2 = _mm512_add_ps(_f2, _c2_avx512);
                    _f3 = _mm512_add_ps(_f3, _c3_avx512);

                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
                _f2 = _mm512_mul_ps(_f2, _alpha);
                _f3 = _mm512_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + 8, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p0 + 12, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p0 + 16, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + 16 + 8, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p0 + 16 + 12, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p0 + 32, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + 32 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + 32 + 8, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p0 + 32 + 12, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p0 + 48, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + 48 + 4, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p0 + 48 + 8, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p0 + 48 + 12, _mm512_extractf32x4_ps(_f3, 3));
                }
                if (out_elempack == 8)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + 8, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + 12, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + 16, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + 16 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + 16 + 8, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + 16 + 12, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 12, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 16, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 16 + 4, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 16 + 8, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 16 + 12, _mm512_extractf32x4_ps(_f3, 3));
                }
                if (out_elempack == 4)
                {
                    transpose16x4_ps(_f0, _f1, _f2, _f3);
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 8, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 12, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + out_hstep, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 2, _mm512_extractf32x4_ps(_f2, 0));
                    _mm_storeu_ps(p0 + out_hstep * 3, _mm512_extractf32x4_ps(_f3, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 5, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 6, _mm512_extractf32x4_ps(_f2, 1));
                    _mm_storeu_ps(p0 + out_hstep * 7, _mm512_extractf32x4_ps(_f3, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 9, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + out_hstep * 10, _mm512_extractf32x4_ps(_f2, 2));
                    _mm_storeu_ps(p0 + out_hstep * 11, _mm512_extractf32x4_ps(_f3, 2));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + out_hstep * 13, _mm512_extractf32x4_ps(_f1, 3));
                    _mm_storeu_ps(p0 + out_hstep * 14, _mm512_extractf32x4_ps(_f2, 3));
                    _mm_storeu_ps(p0 + out_hstep * 15, _mm512_extractf32x4_ps(_f3, 3));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                if (out_elempack == 4)
                {
                    __m512 _tmp0 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp1 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp2 = _mm512_shuffle_f32x4(_f0, _f1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmp3 = _mm512_shuffle_f32x4(_f2, _f3, _MM_SHUFFLE(3, 2, 3, 2));
                    _f0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _f1 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _f2 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _f3 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                    _mm512_storeu_ps(p0 + 32, _f2);
                    _mm512_storeu_ps(p0 + 48, _f3);
                    p0 += 64;
                }
                if (out_elempack == 1)
                {
                    __m512 _tmp0 = _mm512_unpacklo_ps(_f0, _f1);
                    __m512 _tmp1 = _mm512_unpacklo_ps(_f2, _f3);
                    __m512 _tmp2 = _mm512_unpackhi_ps(_f0, _f1);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_f2, _f3);
                    _f0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp0), _mm512_castps_pd(_tmp1)));
                    _f2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));
                    _f3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(_tmp2), _mm512_castps_pd(_tmp3)));

                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + out_hstep, _f1);
                    _mm512_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm512_storeu_ps(p0 + out_hstep * 3, _f3);
                    p0 += 16;
                }
            }

            pp += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));
            __m128i _sum4 = _mm_loadu_si128((const __m128i*)(pp + 16));
            __m128i _sum5 = _mm_loadu_si128((const __m128i*)(pp + 20));
            __m128i _sum6 = _mm_loadu_si128((const __m128i*)(pp + 24));
            __m128i _sum7 = _mm_loadu_si128((const __m128i*)(pp + 28));

            // from
            //      00 11 22 33
            //      04 15 26 37
            //      20 31 02 13
            //      24 35 06 17
            //      01 12 23 30
            //      05 16 27 34
            //      21 32 03 10
            //      25 36 07 14
            // to
            //      00 10 20 30
            //      01 11 21 31
            //      02 12 22 32
            //      03 13 23 33
            //      04 14 24 34
            //      05 15 25 35
            //      06 16 26 36
            //      07 17 27 37
            {
                _sum4 = _mm_shuffle_epi32(_sum4, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum6 = _mm_shuffle_epi32(_sum6, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
                __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum6);
                __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum6);
                __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum7);
                __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum7);
                __m128i _tmp4 = _mm_unpacklo_epi32(_sum2, _sum4);
                __m128i _tmp5 = _mm_unpackhi_epi32(_sum2, _sum4);
                __m128i _tmp6 = _mm_unpacklo_epi32(_sum3, _sum5);
                __m128i _tmp7 = _mm_unpackhi_epi32(_sum3, _sum5);
                _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp4);
                _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp4);
                _sum2 = _mm_unpacklo_epi64(_tmp5, _tmp1);
                _sum3 = _mm_unpackhi_epi64(_tmp5, _tmp1);
                _sum4 = _mm_unpacklo_epi64(_tmp2, _tmp6);
                _sum5 = _mm_unpackhi_epi64(_tmp2, _tmp6);
                _sum6 = _mm_unpacklo_epi64(_tmp7, _tmp3);
                _sum7 = _mm_unpackhi_epi64(_tmp7, _tmp3);
                _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                _sum5 = _mm_shuffle_epi32(_sum5, _MM_SHUFFLE(2, 1, 0, 3));
                _sum7 = _mm_shuffle_epi32(_sum7, _MM_SHUFFLE(2, 1, 0, 3));
            }

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale);
            __m128 _f4 = _mm_mul_ps(_mm_cvtepi32_ps(_sum4), _descale);
            __m128 _f5 = _mm_mul_ps(_mm_cvtepi32_ps(_sum5), _descale);
            __m128 _f6 = _mm_mul_ps(_mm_cvtepi32_ps(_sum6), _descale);
            __m128 _f7 = _mm_mul_ps(_mm_cvtepi32_ps(_sum7), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c0);
                    _f6 = _mm_add_ps(_f6, _c0);
                    _f7 = _mm_add_ps(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC + 16);
                        _c1 = _mm_loadu_ps(pC + 20);
                        _c2 = _mm_loadu_ps(pC + 24);
                        _c3 = _mm_loadu_ps(pC + 28);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC + 4);
                        _c1 = _mm_loadu_ps(pC + c_hstep + 4);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2 + 4);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3 + 4);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f4 = _mm_add_ps(_f4, _c0);
                        _f5 = _mm_add_ps(_f5, _c1);
                        _f6 = _mm_add_ps(_f6, _c2);
                        _f7 = _mm_add_ps(_f7, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f4 = _mm_comp_fmadd_ps(_c0, _beta, _f4);
                        _f5 = _mm_comp_fmadd_ps(_c1, _beta, _f5);
                        _f6 = _mm_comp_fmadd_ps(_c2, _beta, _f6);
                        _f7 = _mm_comp_fmadd_ps(_c3, _beta, _f7);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);

                    _c0 = _mm_set1_ps(pC[4] * beta);
                    _c1 = _mm_set1_ps(pC[5] * beta);
                    _c2 = _mm_set1_ps(pC[6] * beta);
                    _c3 = _mm_set1_ps(pC[7] * beta);

                    _f4 = _mm_add_ps(_f4, _c0);
                    _f5 = _mm_add_ps(_f5, _c1);
                    _f6 = _mm_add_ps(_f6, _c2);
                    _f7 = _mm_add_ps(_f7, _c3);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
                _f4 = _mm_mul_ps(_f4, _alpha);
                _f5 = _mm_mul_ps(_f5, _alpha);
                _f6 = _mm_mul_ps(_f6, _alpha);
                _f7 = _mm_mul_ps(_f7, _alpha);
            }

            if (output_transpose)
            {
#if __AVX__
                if (out_elempack == 8)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f4);
                    _mm_storeu_ps(p0 + 8, _f1);
                    _mm_storeu_ps(p0 + 12, _f5);
                    _mm_storeu_ps(p0 + 16, _f2);
                    _mm_storeu_ps(p0 + 20, _f6);
                    _mm_storeu_ps(p0 + 24, _f3);
                    _mm_storeu_ps(p0 + 28, _f7);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 8, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 12, _f7);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep * 5, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 6, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 7, _f7);
                }
                p0 += out_hstep * 8;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                    _mm_storeu_ps(p0 + 16, _f4);
                    _mm_storeu_ps(p0 + 20, _f5);
                    _mm_storeu_ps(p0 + 24, _f6);
                    _mm_storeu_ps(p0 + 28, _f7);
                    p0 += 32;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _MM_TRANSPOSE4_PS(_f4, _f5, _f6, _f7);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f4);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep + 4, _f5);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 2 + 4, _f6);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    _mm_storeu_ps(p0 + out_hstep * 3 + 4, _f7);
                    p0 += 8;
                }
            }

            pp += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));

            // from
            //      00 11 22 33
            //      01 12 23 30
            //      20 31 02 13
            //      21 32 03 10
            // to
            //      00 10 20 30
            //      01 11 21 31
            //      02 12 22 32
            //      03 13 23 33
            {
                _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
                __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum3);
                __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum3);
                __m128i _tmp2 = _mm_unpacklo_epi32(_sum2, _sum1);
                __m128i _tmp3 = _mm_unpackhi_epi32(_sum2, _sum1);
                _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp2);
                _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp2);
                _sum2 = _mm_unpacklo_epi64(_tmp3, _tmp1);
                _sum3 = _mm_unpackhi_epi64(_tmp3, _tmp1);
                _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
                _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 1, 0, 3));
            }

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    __m128 _c2;
                    __m128 _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        _c2 = _mm_loadu_ps(pC + 8);
                        _c3 = _mm_loadu_ps(pC + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + c_hstep);
                        _c2 = _mm_loadu_ps(pC + c_hstep * 2);
                        _c3 = _mm_loadu_ps(pC + c_hstep * 3);
                        _MM_TRANSPOSE4_PS(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    __m128 _c2 = _mm_set1_ps(pC[2] * beta);
                    __m128 _c3 = _mm_set1_ps(pC[3] * beta);

                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c2);
                    _f3 = _mm_add_ps(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                }
                if (out_elempack == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                }
                p0 += out_hstep * 4;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                    p0 += 16;
                }
                if (out_elempack == 1)
                {
                    _MM_TRANSPOSE4_PS(_f0, _f1, _f2, _f3);
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + out_hstep, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 2, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 3, _f3);
                    p0 += 4;
                }
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));

            // from
            //      00 11 20 31
            //      01 10 21 30
            // to
            //      00 10 20 30
            //      01 11 21 31
            {
                __m128i _tmp0 = _mm_shuffle_epi32(_sum0, _MM_SHUFFLE(3, 1, 2, 0));
                __m128i _tmp1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 2, 3, 1));
                _sum0 = _mm_unpacklo_epi32(_tmp0, _tmp1);
                _sum1 = _mm_unpackhi_epi32(_tmp0, _tmp1);
                _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(2, 1, 0, 3));
            }

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    __m128 _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        _c1 = _mm_loadu_ps(pC + 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        _c1 = _mm_setr_ps(pC[1], pC[c_hstep + 1], pC[c_hstep * 2 + 1], pC[c_hstep * 3 + 1]);
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    __m128 _c1 = _mm_set1_ps(pC[1] * beta);
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + out_hstep, _f1);
                p0 += out_hstep * 2;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    p0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];

                    p0 += 2;
                }
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = _mm_loadu_ps(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = _mm_setr_ps(pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]);
                        pC += 1;
                    }
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_set1_ps(pC[0] * beta);
                    _f0 = _mm_add_ps(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));

            if (output_transpose)
            {
                _mm_storeu_ps(p0, _f0);
                p0 += out_hstep;
            }
            else
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    p0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    _mm_storeu_ps(sum0, _f0);
                    p0[0] = sum0[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0++;
                }
            }

            pp += 4;
        }
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            // out_elempack == 1
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale0 = descales[i + ii];
        const float descale1 = descales[i + ii + 1];
#if __SSE2__
        __m128 _descale0 = _mm_set1_ps(descale0);
        __m128 _descale1 = _mm_set1_ps(descale1);
#if __AVX512F__
        __m512 _descale0_avx512 = _mm512_set1_ps(descale0);
        __m512 _descale1_avx512 = _mm512_set1_ps(descale1);
#endif // __AVX512F__
#endif

        float c0;
        float c1;
#if __SSE2__
        __m128 _c0;
        __m128 _c1;
#if __AVX512F__
        __m512 _c0_avx512;
        __m512 _c1_avx512;
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
                _c1 = _mm_set1_ps(c1);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
                _c1_avx512 = _mm512_set1_ps(c1);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0 = _mm512_loadu_si512((const __m512i*)pp);
            __m512i _sum1 = _mm512_loadu_si512((const __m512i*)(pp + 16));

            // 00 11 02 13  04 15 06 17  08 19 0a 1b  0c 1d 0e 1f
            // 01 12 03 10  05 16 07 14  09 1a 0b 18  0d 1e 0f 1c

            __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
            __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);

            _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp1);
            _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp1);

            _sum1 = _mm512_shuffle_epi32(_sum1, _MM_PERM_CBAD);

            __m512 _f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_sum0), _descale0_avx512);
            __m512 _f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_sum1), _descale1_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c1_avx512);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c1_avx512 = _mm512_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm512_add_ps(_f0, _c0_avx512);
                        _f1 = _mm512_add_ps(_f1, _c1_avx512);
                    }
                    else
                    {
                        __m512 _beta = _mm512_set1_ps(beta);
                        _f0 = _mm512_fmadd_ps(_c0_avx512, _beta, _f0);
                        _f1 = _mm512_fmadd_ps(_c1_avx512, _beta, _f1);
                    }
                    pC += 16;
                }
                if (broadcast_type_C == 4)
                {
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _c0_avx512 = _mm512_mul_ps(_c0_avx512, _mm512_set1_ps(beta));
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                    _f1 = _mm512_add_ps(_f1, _c0_avx512);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                __m512 _alpha = _mm512_set1_ps(alpha);
                _f0 = _mm512_mul_ps(_f0, _alpha);
                _f1 = _mm512_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 16)
                {
                    _mm512_storeu_ps(p0, _f0);
                    _mm512_storeu_ps(p0 + 16, _f1);
                }
                if (out_elempack == 8)
                {
                    _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                    _mm256_storeu_ps(p0 + 8, _mm512_extractf32x8_ps(_f1, 0));
                    _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    _mm256_storeu_ps(p0 + out_hstep * 8 + 8, _mm512_extractf32x8_ps(_f1, 1));
                }
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                    _mm_storeu_ps(p0 + 4, _mm512_extractf32x4_ps(_f1, 0));
                    _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _mm512_extractf32x4_ps(_f1, 1));
                    _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                    _mm_storeu_ps(p0 + out_hstep * 8 + 4, _mm512_extractf32x4_ps(_f1, 2));
                    _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    _mm_storeu_ps(p0 + out_hstep * 12 + 4, _mm512_extractf32x4_ps(_f1, 3));
                }
                if (out_elempack == 1)
                {
                    __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                    _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    _mm512_i32scatter_ps(p0 + 1, _vindex, _f1, sizeof(float));
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm512_storeu_ps(p0, _f0);
                _mm512_storeu_ps(p0 + out_hstep, _f1);
                p0 += 16;
            }

            pp += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));
            __m128i _sum2 = _mm_loadu_si128((const __m128i*)(pp + 8));
            __m128i _sum3 = _mm_loadu_si128((const __m128i*)(pp + 12));

            // 00 11 02 13
            // 04 15 06 17
            // 10 01 12 03
            // 14 05 16 07
            _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
            _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

            // 00 11 02 13
            // 04 15 06 17
            // 01 10 03 12
            // 05 14 07 16

            __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum2);
            __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum2);
            __m128i _tmp2 = _mm_unpacklo_epi32(_sum1, _sum3);
            __m128i _tmp3 = _mm_unpackhi_epi32(_sum1, _sum3);

            // 00 01 11 10
            // 02 03 13 12
            // 04 05 15 14
            // 06 07 17 16

            _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _sum1 = _mm_unpacklo_epi64(_tmp2, _tmp3);
            _sum2 = _mm_unpackhi_epi64(_tmp0, _tmp1);
            _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);

            // 00 01 02 03
            // 04 05 06 07
            // 11 10 13 12
            // 15 14 17 16
            _sum2 = _mm_shuffle_epi32(_sum2, _MM_SHUFFLE(2, 3, 0, 1));
            _sum3 = _mm_shuffle_epi32(_sum3, _MM_SHUFFLE(2, 3, 0, 1));

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale0);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale0);
            __m128 _f2 = _mm_mul_ps(_mm_cvtepi32_ps(_sum2), _descale1);
            __m128 _f3 = _mm_mul_ps(_mm_cvtepi32_ps(_sum3), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    _f2 = _mm_add_ps(_f2, _c1);
                    _f3 = _mm_add_ps(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    __m128 _c2 = _mm_loadu_ps(pC + c_hstep);
                    __m128 _c3 = _mm_loadu_ps(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                        _f2 = _mm_add_ps(_f2, _c2);
                        _f3 = _mm_add_ps(_f3, _c3);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                        _f2 = _mm_comp_fmadd_ps(_c2, _beta, _f2);
                        _f3 = _mm_comp_fmadd_ps(_c3, _beta, _f3);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + 4);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _c1 = _mm_mul_ps(_c1, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                    _f2 = _mm_add_ps(_f2, _c0);
                    _f3 = _mm_add_ps(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
                _f2 = _mm_mul_ps(_f2, _alpha);
                _f3 = _mm_mul_ps(_f3, _alpha);
            }

            if (output_transpose)
            {
#if __AVX__
                if (out_elempack == 8)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                    _mm_storeu_ps(p0 + 8, _f2);
                    _mm_storeu_ps(p0 + 12, _f3);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f2);
                    _mm_storeu_ps(p0 + out_hstep * 4, _f1);
                    _mm_storeu_ps(p0 + out_hstep * 4 + 4, _f3);
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    float sum2[4];
                    float sum3[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);
                    _mm_storeu_ps(sum2, _f2);
                    _mm_storeu_ps(sum3, _f3);

                    p0[0] = sum0[0];
                    p0[1] = sum2[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum2[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum2[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum2[3];
                    p0[out_hstep * 4] = sum1[0];
                    p0[out_hstep * 4 + 1] = sum3[0];
                    p0[out_hstep * 5] = sum1[1];
                    p0[out_hstep * 5 + 1] = sum3[1];
                    p0[out_hstep * 6] = sum1[2];
                    p0[out_hstep * 6 + 1] = sum3[2];
                    p0[out_hstep * 7] = sum1[3];
                    p0[out_hstep * 7 + 1] = sum3[3];
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + 4, _f1);
                _mm_storeu_ps(p0 + out_hstep, _f2);
                _mm_storeu_ps(p0 + out_hstep + 4, _f3);
                p0 += 8;
            }

            pp += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0 = _mm_loadu_si128((const __m128i*)pp);
            __m128i _sum1 = _mm_loadu_si128((const __m128i*)(pp + 4));

            // 00 11 02 13
            // 01 12 03 10
            __m128i _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
            __m128i _tmp1 = _mm_unpackhi_epi32(_sum0, _sum1);

            // 00 01 11 12
            // 02 03 13 10
            _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
            _sum1 = _mm_unpackhi_epi64(_tmp1, _tmp0);

            // 00 01 02 03
            // 13 10 11 12
            _sum1 = _mm_shuffle_epi32(_sum1, _MM_SHUFFLE(0, 3, 2, 1));

            // 00 01 02 03
            // 10 11 12 13

            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _descale0);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _c1 = _mm_loadu_ps(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = _mm_add_ps(_f0, _c0);
                        _f1 = _mm_add_ps(_f1, _c1);
                    }
                    else
                    {
                        __m128 _beta = _mm_set1_ps(beta);
                        _f0 = _mm_comp_fmadd_ps(_c0, _beta, _f0);
                        _f1 = _mm_comp_fmadd_ps(_c1, _beta, _f1);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = _mm_loadu_ps(pC);
                    _c0 = _mm_mul_ps(_c0, _mm_set1_ps(beta));
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                if (out_elempack == 4)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    _mm_storeu_ps(sum0, _f0);
                    _mm_storeu_ps(sum1, _f1);

                    p0[0] = sum0[0];
                    p0[1] = sum1[0];
                    p0[out_hstep] = sum0[1];
                    p0[out_hstep + 1] = sum1[1];
                    p0[out_hstep * 2] = sum0[2];
                    p0[out_hstep * 2 + 1] = sum1[2];
                    p0[out_hstep * 3] = sum0[3];
                    p0[out_hstep * 3 + 1] = sum1[3];
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + out_hstep, _f1);
                p0 += 4;
            }

            pp += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0] * descale0;
            float f01 = pp[1] * descale0;
            float f10 = pp[2] * descale1;
            float f11 = pp[3] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[c_hstep] * beta;
                    f11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[0] * beta;
                    f11 += pC[1] * beta;
                    pC += 2;
                }
            }

            f00 *= alpha;
            f01 *= alpha;
            f10 *= alpha;
            f11 *= alpha;

            if (output_transpose)
            {
                p0[0] = f00;
                p0[1] = f10;
                p0[out_hstep] = f01;
                p0[out_hstep + 1] = f11;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f00;
                p0[1] = f01;
                p0[out_hstep] = f10;
                p0[out_hstep + 1] = f11;
                p0 += 2;
            }

            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale0;
            float f1 = pp[1] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += out_hstep;
            }
            else
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0++;
            }

            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;
        }
        else
        {
            // out_elempack == 1
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale = descales[i + ii];
#if __SSE2__
        __m128 _descale = _mm_set1_ps(descale);
#if __AVX512F__
        __m512 _descale_avx512 = _mm512_set1_ps(descale);
#endif // __AVX512F__
#endif

        float c0;
#if __SSE2__
        __m128 _c0;
#if __AVX512F__
        __m512 _c0_avx512;
#endif // __AVX512F__
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __SSE2__
                _c0 = _mm_set1_ps(c0);
#if __AVX512F__
                _c0_avx512 = _mm512_set1_ps(c0);
#endif // __AVX512F__
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const float*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512 _f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)pp)), _descale_avx512);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm512_add_ps(_f0, _c0_avx512);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0_avx512 = _mm512_loadu_ps(pC);
                    _f0 = _mm512_fmadd_ps(_c0_avx512, _mm512_set1_ps(beta), _f0);
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = _mm512_mul_ps(_f0, _mm512_set1_ps(alpha));
            }

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm512_storeu_ps(p0, _f0);
                }
                else
                {
                    if (out_elempack == 16)
                    {
                        _mm512_storeu_ps(p0, _f0);
                    }
                    if (out_elempack == 8)
                    {
                        _mm256_storeu_ps(p0, _mm512_extractf32x8_ps(_f0, 0));
                        _mm256_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x8_ps(_f0, 1));
                    }
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _mm512_extractf32x4_ps(_f0, 0));
                        _mm_storeu_ps(p0 + out_hstep * 4, _mm512_extractf32x4_ps(_f0, 1));
                        _mm_storeu_ps(p0 + out_hstep * 8, _mm512_extractf32x4_ps(_f0, 2));
                        _mm_storeu_ps(p0 + out_hstep * 12, _mm512_extractf32x4_ps(_f0, 3));
                    }
                    if (out_elempack == 1)
                    {
                        __m512i _vindex = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(out_hstep));
                        _mm512_i32scatter_ps(p0, _vindex, _f0, sizeof(float));
                    }
                }
                p0 += out_hstep * 16;
            }
            else
            {
                _mm512_storeu_ps(p0, _f0);
                p0 += 16;
            }

            pp += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);
            __m128 _f1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(pp + 4))), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                    _f1 = _mm_add_ps(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    __m128 _c1 = _mm_loadu_ps(pC + 4);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    _f1 = _mm_comp_fmadd_ps(_c1, _mm_set1_ps(beta), _f1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                __m128 _alpha = _mm_set1_ps(alpha);
                _f0 = _mm_mul_ps(_f0, _alpha);
                _f1 = _mm_mul_ps(_f1, _alpha);
            }

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                    _mm_storeu_ps(p0 + 4, _f1);
                }
                else
                {
#if __AVX__
                    if (out_elempack == 8)
                    {
                        _mm_storeu_ps(p0, _f0);
                        _mm_storeu_ps(p0 + 4, _f1);
                    }
#endif // __AVX__
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _f0);
                        _mm_storeu_ps(p0 + out_hstep * 4, _f1);
                    }
                    if (out_elempack == 1)
                    {
                        float sum0[4];
                        float sum1[4];
                        _mm_storeu_ps(sum0, _f0);
                        _mm_storeu_ps(sum1, _f1);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                        p0[out_hstep * 4] = sum1[0];
                        p0[out_hstep * 5] = sum1[1];
                        p0[out_hstep * 6] = sum1[2];
                        p0[out_hstep * 7] = sum1[3];
                    }
                }
                p0 += out_hstep * 8;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                _mm_storeu_ps(p0 + 4, _f1);
                p0 += 8;
            }

            pp += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128 _f0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = _mm_add_ps(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = _mm_loadu_ps(pC);
                    _f0 = _mm_comp_fmadd_ps(_c0, _mm_set1_ps(beta), _f0);
                    pC += 4;
                }
            }

            _f0 = _mm_mul_ps(_f0, _mm_set1_ps(alpha));

            if (output_transpose)
            {
                if (out_hstep == 1)
                {
                    _mm_storeu_ps(p0, _f0);
                }
                else
                {
                    if (out_elempack == 4)
                    {
                        _mm_storeu_ps(p0, _f0);
                    }
                    if (out_elempack == 1)
                    {
                        float sum0[4];
                        _mm_storeu_ps(sum0, _f0);

                        p0[0] = sum0[0];
                        p0[out_hstep] = sum0[1];
                        p0[out_hstep * 2] = sum0[2];
                        p0[out_hstep * 3] = sum0[3];
                    }
                }
                p0 += out_hstep * 4;
            }
            else
            {
                _mm_storeu_ps(p0, _f0);
                p0 += 4;
            }

            pp += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0] * descale;
            float f1 = pp[1] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += 2;
            }

            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = f0;

            if (output_transpose)
            {
                p0 += out_hstep;
            }
            else
            {
                p0++;
            }

            pp += 1;
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        gemm_transB_packed_tile_int8_avx512vnni(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        gemm_transB_packed_tile_int8_avxvnni(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        gemm_transB_packed_tile_int8_avx2(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        gemm_transB_packed_tile_int8_xop(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __SSE2__
#if __AVX2__
#if __AVX512F__
    for (; ii + 15 < max_ii; ii += 16)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;
            __m512i _sum8;
            __m512i _sum9;
            __m512i _suma;
            __m512i _sumb;
            __m512i _sumc;
            __m512i _sumd;
            __m512i _sume;
            __m512i _sumf;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
                _sum8 = _mm512_setzero_si512();
                _sum9 = _mm512_setzero_si512();
                _suma = _mm512_setzero_si512();
                _sumb = _mm512_setzero_si512();
                _sumc = _mm512_setzero_si512();
                _sumd = _mm512_setzero_si512();
                _sume = _mm512_setzero_si512();
                _sumf = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
                _sum8 = _mm512_load_si512((const __m512i*)(outptr + 128));
                _sum9 = _mm512_load_si512((const __m512i*)(outptr + 128 + 16));
                _suma = _mm512_load_si512((const __m512i*)(outptr + 128 + 32));
                _sumb = _mm512_load_si512((const __m512i*)(outptr + 128 + 48));
                _sumc = _mm512_load_si512((const __m512i*)(outptr + 128 + 64));
                _sumd = _mm512_load_si512((const __m512i*)(outptr + 128 + 80));
                _sume = _mm512_load_si512((const __m512i*)(outptr + 128 + 96));
                _sumf = _mm512_load_si512((const __m512i*)(outptr + 128 + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pA3 = _mm512_shuffle_epi32(_pA2, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA1);
                _sum8 = _mm512_dpbusd_epi32(_sum8, _pB0, _pA2);
                _sum9 = _mm512_dpbusd_epi32(_sum9, _pB1, _pA2);
                _suma = _mm512_dpbusd_epi32(_suma, _pB0, _pA3);
                _sumb = _mm512_dpbusd_epi32(_sumb, _pB1, _pA3);
                _sumc = _mm512_dpbusd_epi32(_sumc, _pB2, _pA2);
                _sumd = _mm512_dpbusd_epi32(_sumd, _pB3, _pA2);
                _sume = _mm512_dpbusd_epi32(_sume, _pB2, _pA3);
                _sumf = _mm512_dpbusd_epi32(_sumf, _pB3, _pA3);
                pA += 64;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                __m512i _w_shift2 = _mm512_shuffle_i32x4(_w_shift0, _w_shift0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _w_shift3 = _mm512_shuffle_epi32(_w_shift2, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                _sum4 = _mm512_sub_epi32(_sum4, _w_shift0);
                _sum5 = _mm512_sub_epi32(_sum5, _w_shift0);
                _sum6 = _mm512_sub_epi32(_sum6, _w_shift1);
                _sum7 = _mm512_sub_epi32(_sum7, _w_shift1);
                _sum8 = _mm512_sub_epi32(_sum8, _w_shift2);
                _sum9 = _mm512_sub_epi32(_sum9, _w_shift2);
                _suma = _mm512_sub_epi32(_suma, _w_shift3);
                _sumb = _mm512_sub_epi32(_sumb, _w_shift3);
                _sumc = _mm512_sub_epi32(_sumc, _w_shift2);
                _sumd = _mm512_sub_epi32(_sumd, _w_shift2);
                _sume = _mm512_sub_epi32(_sume, _w_shift3);
                _sumf = _mm512_sub_epi32(_sumf, _w_shift3);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                // 4567 0123 cdef 89ab
                // 6745 2301 efcd ab89
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pA2 = _mm512_shuffle_i32x4(_pA0, _pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m512i _pA3 = _mm512_shuffle_epi32(_pA2, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 89ab cdef 0123 4567
                // 9ab8 defc 1230 5674
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_shuffle_i32x4(_pB0, _pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA0, _pB2));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA0, _pB3));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));
                _sum8 = _mm512_add_epi32(_sum8, _mm512_madd_epi16(_pA2, _pB0));
                _sum9 = _mm512_add_epi32(_sum9, _mm512_madd_epi16(_pA2, _pB1));
                _suma = _mm512_add_epi32(_suma, _mm512_madd_epi16(_pA3, _pB0));
                _sumb = _mm512_add_epi32(_sumb, _mm512_madd_epi16(_pA3, _pB1));
                _sumc = _mm512_add_epi32(_sumc, _mm512_madd_epi16(_pA2, _pB2));
                _sumd = _mm512_add_epi32(_sumd, _mm512_madd_epi16(_pA2, _pB3));
                _sume = _mm512_add_epi32(_sume, _mm512_madd_epi16(_pA3, _pB2));
                _sumf = _mm512_add_epi32(_sumf, _mm512_madd_epi16(_pA3, _pB3));

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // 23016745 ab89efcd
                // 45670123 cdef89ab
                // 67452301 efcdab89
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pA2 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m256i _pA3 = _mm256_shuffle_epi32(_pA2, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                // 89abcdef 01234567
                // 9ab8defc 12305674
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_permute4x64_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3));
                __m512i _s8 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB0));
                __m512i _s9 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB1));
                __m512i _sa = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB0));
                __m512i _sb = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB1));
                __m512i _sc = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB2));
                __m512i _sd = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA2, _pB3));
                __m512i _se = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB2));
                __m512i _sf = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA3, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);
                _sum8 = _mm512_add_epi32(_sum8, _s8);
                _sum9 = _mm512_add_epi32(_sum9, _s9);
                _suma = _mm512_add_epi32(_suma, _sa);
                _sumb = _mm512_add_epi32(_sumb, _sb);
                _sumc = _mm512_add_epi32(_sumc, _sc);
                _sumd = _mm512_add_epi32(_sumd, _sd);
                _sume = _mm512_add_epi32(_sume, _se);
                _sumf = _mm512_add_epi32(_sumf, _sf);

                pA += 16;
                pB += 16;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);
            _mm512_store_si512((__m512i*)(outptr + 128), _sum8);
            _mm512_store_si512((__m512i*)(outptr + 128 + 16), _sum9);
            _mm512_store_si512((__m512i*)(outptr + 128 + 32), _suma);
            _mm512_store_si512((__m512i*)(outptr + 128 + 48), _sumb);
            _mm512_store_si512((__m512i*)(outptr + 128 + 64), _sumc);
            _mm512_store_si512((__m512i*)(outptr + 128 + 80), _sumd);
            _mm512_store_si512((__m512i*)(outptr + 128 + 96), _sume);
            _mm512_store_si512((__m512i*)(outptr + 128 + 112), _sumf);
            outptr += 256;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);
                __m512i _pB0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pB), _pB, 1);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA1);
                pA += 64;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                _sum4 = _mm512_sub_epi32(_sum4, _w_shift0);
                _sum5 = _mm512_sub_epi32(_sum5, _w_shift0);
                _sum6 = _mm512_sub_epi32(_sum6, _w_shift1);
                _sum7 = _mm512_sub_epi32(_sum7, _w_shift1);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 4567 0123 4567
                // 1230 5674 1230 5674
                // 4567 0123 4567 0123
                // 5674 1230 5674 1230
                __m512i _pB0 = _mm512_inserti32x8(_mm512_castsi256_si512(_pBB), _pBB, 1);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA0, _pB2));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA0, _pB3));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA1, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA1, _pB3));

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // 23016745 ab89efcd
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 01234567
                // 12305674 12305674
                // 45670123 45670123
                // 56741230 56741230
                __m256i _pB0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB, 1);
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB2));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB3));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);

                pA += 16;
                pB += 8;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);
            outptr += 128;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pB));
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 64;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef
                // 2301 6745 ab89 efcd
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 0123 0123 0123
                // 1230 1230 1230 1230
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef
                // 23016745 ab89efcd
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                // 01230123 01230123
                // 12301230 12301230
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                pA += 16;
                pB += 4;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            outptr += 64;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB0 = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pB)[0]));
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 64;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 89ab cdef

                // 0101 0101 0101 0101
                // 1010 1010 1010 1010
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CDAB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 89abcdef

                // 01010101 01010101
                // 10101010 10101010
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                pA += 16;
                pB += 2;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            outptr += 32;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_loadu_si512((const __m512i*)pA);
                __m512i _pB = _mm512_set1_epi32(((const int*)pB)[0]);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 64;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)pA);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                pA += 64;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pBBBB = _mm512_cvtepi8_epi16(_pB);

                // 0xxx0xxx0xxx0xxx -> 00000000...
                __m512i _pB0 = _mm512_shuffle_epi32(_pBBBB, _MM_PERM_AAAA);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));

                pA += 32;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_load_si128((const __m128i*)pA);
                __m256i _pB = _mm256_set1_epi16(pB[0]);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB));

                _sum0 = _mm512_add_epi32(_sum0, _s0);

                pA += 16;
                pB += 1;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            outptr += 16;
        }

        pAT += max_kk * 16;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 64;
        }
#endif // __AVX512VNNI__
    }
#endif // __AVX512F__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;
            __m512i _sum4;
            __m512i _sum5;
            __m512i _sum6;
            __m512i _sum7;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
                _sum4 = _mm512_setzero_si512();
                _sum5 = _mm512_setzero_si512();
                _sum6 = _mm512_setzero_si512();
                _sum7 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_load_si512((const __m512i*)outptr);
                _sum1 = _mm512_load_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_load_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_load_si512((const __m512i*)(outptr + 48));
                _sum4 = _mm512_load_si512((const __m512i*)(outptr + 64));
                _sum5 = _mm512_load_si512((const __m512i*)(outptr + 80));
                _sum6 = _mm512_load_si512((const __m512i*)(outptr + 96));
                _sum7 = _mm512_load_si512((const __m512i*)(outptr + 112));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA00);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA00);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA11);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA11);
                _sum4 = _mm512_dpbusd_epi32(_sum4, _pB2, _pA00);
                _sum5 = _mm512_dpbusd_epi32(_sum5, _pB3, _pA00);
                _sum6 = _mm512_dpbusd_epi32(_sum6, _pB2, _pA11);
                _sum7 = _mm512_dpbusd_epi32(_sum7, _pB3, _pA11);
                pA += 32;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m512i _w_shift00 = _mm512_inserti32x8(_mm512_castsi256_si512(_w_shift0), _w_shift0, 1);
                __m512i _w_shift11 = _mm512_shuffle_epi32(_w_shift00, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift00);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift00);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift11);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift11);
                _sum4 = _mm512_sub_epi32(_sum4, _w_shift00);
                _sum5 = _mm512_sub_epi32(_sum5, _w_shift00);
                _sum6 = _mm512_sub_epi32(_sum6, _w_shift11);
                _sum7 = _mm512_sub_epi32(_sum7, _w_shift11);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 4567 0123 4567
                // 2301 6745 2301 6745
                __m512i _pA00 = _mm512_inserti32x8(_mm512_castsi256_si512(_pA0), _pA0, 1);
                __m512i _pA11 = _mm512_shuffle_epi32(_pA00, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                // 4567 0123 cdef 89ab
                // 5674 1230 defc 9ab8
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                __m512i _pB2 = _mm512_permutex_epi64(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m512i _pB3 = _mm512_shuffle_epi32(_pB2, _MM_PERM_ADCB);

                // 4567 0123 4567 0123
                // __m512i _pA11 = _mm512_permutex_epi64(_pA00, _MM_SHUFFLE(1, 0, 3, 2));

                // 2301 6745 ab89 efcd
                // 3012 7456 b89a fcde
                // __m512i _pB2 = _mm512_shuffle_epi32(_pB0, _MM_PERM_BADC);
                // __m512i _pB3 = _mm512_shuffle_epi32(_pB0, _MM_PERM_CBAD);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA00, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA00, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA11, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA11, _pB1));
                _sum4 = _mm512_add_epi32(_sum4, _mm512_madd_epi16(_pA00, _pB2));
                _sum5 = _mm512_add_epi32(_sum5, _mm512_madd_epi16(_pA00, _pB3));
                _sum6 = _mm512_add_epi32(_sum6, _mm512_madd_epi16(_pA11, _pB2));
                _sum7 = _mm512_add_epi32(_sum7, _mm512_madd_epi16(_pA11, _pB3));

                pA += 16;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01234567 01234567
                // 23016745 23016745
                __m256i _pA00 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pA), _pA, 1);
                __m256i _pA11 = _mm256_shuffle_epi32(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                // 45670123 cdef89ab
                // 56741230 defc9ab8
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB2, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                // 45670123 45670123
                // __m256i _pA11 = _mm256_permute4x64_epi64(_pA00, _MM_SHUFFLE(2, 3, 0, 1));

                // 23016745 ab89efcd
                // 30127456 b89afcde
                // __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                // __m256i _pB3 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB1));
                __m512i _s4 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB2));
                __m512i _s5 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA00, _pB3));
                __m512i _s6 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB2));
                __m512i _s7 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA11, _pB3));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);
                _sum4 = _mm512_add_epi32(_sum4, _s4);
                _sum5 = _mm512_add_epi32(_sum5, _s5);
                _sum6 = _mm512_add_epi32(_sum6, _s6);
                _sum7 = _mm512_add_epi32(_sum7, _s7);

                pA += 8;
                pB += 16;
            }

            _mm512_store_si512((__m512i*)outptr, _sum0);
            _mm512_store_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_store_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_store_si512((__m512i*)(outptr + 48), _sum3);
            _mm512_store_si512((__m512i*)(outptr + 64), _sum4);
            _mm512_store_si512((__m512i*)(outptr + 80), _sum5);
            _mm512_store_si512((__m512i*)(outptr + 96), _sum6);
            _mm512_store_si512((__m512i*)(outptr + 112), _sum7);

            outptr += 128;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;
            __m256i _sum4;
            __m256i _sum5;
            __m256i _sum6;
            __m256i _sum7;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
                _sum4 = _mm256_setzero_si256();
                _sum5 = _mm256_setzero_si256();
                _sum6 = _mm256_setzero_si256();
                _sum7 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
                _sum4 = _mm256_load_si256((const __m256i*)(outptr + 32));
                _sum5 = _mm256_load_si256((const __m256i*)(outptr + 40));
                _sum6 = _mm256_load_si256((const __m256i*)(outptr + 48));
                _sum7 = _mm256_load_si256((const __m256i*)(outptr + 56));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_loadu_si256((const __m256i*)pB);
                __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(0, 1, 2, 3));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));
                _sum0 = _mm256_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm256_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm256_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm256_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm256_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm256_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm256_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm256_dpbusd_epi32(_sum7, _pB3, _pA1);
                pA += 32;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _w_shift1 = _mm256_permute4x64_epi64(_w_shift0, _MM_SHUFFLE(0, 1, 2, 3));
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                _sum4 = _mm256_sub_epi32(_sum4, _w_shift0);
                _sum5 = _mm256_sub_epi32(_sum5, _w_shift0);
                _sum6 = _mm256_sub_epi32(_sum6, _w_shift1);
                _sum7 = _mm256_sub_epi32(_sum7, _w_shift1);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // // 0123 4567
                // // 4567 0123
                // __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 6745 2301
                __m256i _pA1 = _mm256_permute4x64_epi64(_pA0, _MM_SHUFFLE(0, 1, 2, 3));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m256i _pB2 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB3 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 1, 0, 3));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));
                _sum4 = _mm256_add_epi32(_sum4, _mm256_madd_epi16(_pA0, _pB2));
                _sum5 = _mm256_add_epi32(_sum5, _mm256_madd_epi16(_pA0, _pB3));
                _sum6 = _mm256_add_epi32(_sum6, _mm256_madd_epi16(_pA1, _pB2));
                _sum7 = _mm256_add_epi32(_sum7, _mm256_madd_epi16(_pA1, _pB3));

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // // 0123 4567
                // // 4567 0123
                // __m128i _pA0 = _pA;
                // __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 4567
                // 6745 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 1, 2, 3));

                // 0123 4567
                // 1230 5674
                // 2301 6745
                // 3012 7456
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB2 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));
                __m128i _pB3 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(2, 1, 0, 3)), _MM_SHUFFLE(2, 1, 0, 3));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));
                __m256i _s4 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB2));
                __m256i _s5 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB3));
                __m256i _s6 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB2));
                __m256i _s7 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB3));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);
                _sum4 = _mm256_add_epi32(_sum4, _s4);
                _sum5 = _mm256_add_epi32(_sum5, _s5);
                _sum6 = _mm256_add_epi32(_sum6, _s6);
                _sum7 = _mm256_add_epi32(_sum7, _s7);

                pA += 8;
                pB += 8;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
            _mm256_store_si256((__m256i*)(outptr + 24), _sum3);
            _mm256_store_si256((__m256i*)(outptr + 32), _sum4);
            _mm256_store_si256((__m256i*)(outptr + 40), _sum5);
            _mm256_store_si256((__m256i*)(outptr + 48), _sum6);
            _mm256_store_si256((__m256i*)(outptr + 56), _sum7);

            outptr += 64;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;
            __m256i _sum2;
            __m256i _sum3;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
                _sum2 = _mm256_setzero_si256();
                _sum3 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
                _sum2 = _mm256_load_si256((const __m256i*)(outptr + 16));
                _sum3 = _mm256_load_si256((const __m256i*)(outptr + 24));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA0 = _mm256_loadu_si256((const __m256i*)pA);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                __m256i _pB0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_pB), _pB, 1);
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm256_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm256_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm256_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm256_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 32;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift0 = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _w_shift1 = _mm256_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm256_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm256_sub_epi32(_sum3, _w_shift1);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castpd_si128(_mm_load1_pd((const double*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567
                // 2301 6745
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123 0123
                // 1230 1230
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));
                _sum2 = _mm256_add_epi32(_sum2, _mm256_madd_epi16(_pA1, _pB0));
                _sum3 = _mm256_add_epi32(_sum3, _mm256_madd_epi16(_pA1, _pB1));

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567
                // 23016745
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01230123
                // 12301230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA0, _pB1));
                __m256i _s2 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB0));
                __m256i _s3 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);

                pA += 8;
                pB += 4;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);
            _mm256_store_si256((__m256i*)(outptr + 16), _sum2);
            _mm256_store_si256((__m256i*)(outptr + 24), _sum3);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m256i _sum0;
            __m256i _sum1;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
                _sum1 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
                _sum1 = _mm256_load_si256((const __m256i*)(outptr + 8));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB0 = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pB));
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));
                _sum0 = _mm256_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm256_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 32;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm256_sub_epi32(_sum1, _w_shift);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 0123 4567

                // 0101 0101
                // 1010 1010
                __m256i _pB1 = _mm256_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 1, 0, 1));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));
                _sum1 = _mm256_add_epi32(_sum1, _mm256_madd_epi16(_pA0, _pB1));

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);

                // 01234567

                // 01010101
                // 10101010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1)), _MM_SHUFFLE(0, 1, 0, 1));

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB0));
                __m256i _s1 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB1));

                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

                pA += 8;
                pB += 2;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);
            _mm256_store_si256((__m256i*)(outptr + 8), _sum1);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m256i _sum0;

            if (k == 0)
            {
                _sum0 = _mm256_setzero_si256();
            }
            else
            {
                _sum0 = _mm256_load_si256((const __m256i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m256i _pA = _mm256_loadu_si256((const __m256i*)pA);
                __m256i _pB = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pB));
                _sum0 = _mm256_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 32;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)pA);
                _sum0 = _mm256_sub_epi32(_sum0, _w_shift);
                pA += 32;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pBB = _mm256_cvtepi8_epi16(_pB);

                // 0xxx0xxx -> 00000000 11111111
                __m256i _pB0 = _mm256_shuffle_epi32(_pBB, _MM_SHUFFLE(0, 0, 0, 0));

                _sum0 = _mm256_add_epi32(_sum0, _mm256_madd_epi16(_pA0, _pB0));

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

                _pA = _mm_cvtepi8_epi16(_pA);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_pA, _pB));

                _sum0 = _mm256_add_epi32(_sum0, _s0);

                pA += 8;
                pB += 1;
            }

            _mm256_store_si256((__m256i*)outptr, _sum0);

            outptr += 8;
        }

        pAT += max_kk * 8;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 32;
        }
#endif // __AVX512VNNI__
    }
#endif // __AVX2__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            const signed char* pA = pAT;

            __m512i _sum0;
            __m512i _sum1;
            __m512i _sum2;
            __m512i _sum3;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
                _sum2 = _mm512_setzero_si512();
                _sum3 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
                _sum2 = _mm512_loadu_si512((const __m512i*)(outptr + 32));
                _sum3 = _mm512_loadu_si512((const __m512i*)(outptr + 48));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 16;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)pA));
                __m512i _w_shift1 = _mm512_shuffle_epi32(_w_shift0, _MM_PERM_BADC);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm512_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm512_sub_epi32(_sum3, _w_shift1);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0123 0123 0123 0123
                // 2301 2301 2301 2301
                __m512i _pA1 = _mm512_shuffle_epi32(_pA0, _MM_PERM_BADC);

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));
                _sum2 = _mm512_add_epi32(_sum2, _mm512_madd_epi16(_pA1, _pB0));
                _sum3 = _mm512_add_epi32(_sum3, _mm512_madd_epi16(_pA1, _pB1));

                pA += 8;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01230123 01230123
                // 23012301 23012301
                __m256i _pA1 = _mm256_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567 89abcdef
                // 12305674 9ab8defc
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));
                __m512i _s2 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB0));
                __m512i _s3 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA1, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                pA += 4;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);
            _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);
            _mm512_storeu_si512((__m512i*)(outptr + 32), _sum2);
            _mm512_storeu_si512((__m512i*)(outptr + 48), _sum3);

            outptr += 64;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;
            __m128i _sum4;
            __m128i _sum5;
            __m128i _sum6;
            __m128i _sum7;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
                _sum4 = _mm_setzero_si128();
                _sum5 = _mm_setzero_si128();
                _sum6 = _mm_setzero_si128();
                _sum7 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
                _sum4 = _mm_load_si128((const __m128i*)(outptr + 16));
                _sum5 = _mm_load_si128((const __m128i*)(outptr + 20));
                _sum6 = _mm_load_si128((const __m128i*)(outptr + 24));
                _sum7 = _mm_load_si128((const __m128i*)(outptr + 28));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 16));
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm_dpbusd_epi32(_sum3, _pB1, _pA1);
                _sum4 = _mm_dpbusd_epi32(_sum4, _pB2, _pA0);
                _sum5 = _mm_dpbusd_epi32(_sum5, _pB3, _pA0);
                _sum6 = _mm_dpbusd_epi32(_sum6, _pB2, _pA1);
                _sum7 = _mm_dpbusd_epi32(_sum7, _pB3, _pA1);
                pA += 16;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _w_shift1 = _mm_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm_sub_epi32(_sum3, _w_shift1);
                _sum4 = _mm_sub_epi32(_sum4, _w_shift0);
                _sum5 = _mm_sub_epi32(_sum5, _w_shift0);
                _sum6 = _mm_sub_epi32(_sum6, _w_shift1);
                _sum7 = _mm_sub_epi32(_sum7, _w_shift1);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif
                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pBl = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pBh = _mm_unpackhi_epi8(_pB, _extpB);

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 4567
                // 1230
                // 5674
                __m128i _pB0 = _pBl;
                __m128i _pB1 = _pBh;
                __m128i _pB2 = _mm_shuffle_epi32(_pBl, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pBh, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
                _sum4 = _mm_maddd_epi16(_pA0, _pB2, _sum4);
                _sum5 = _mm_maddd_epi16(_pA0, _pB3, _sum5);
                _sum6 = _mm_maddd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maddd_epi16(_pA1, _pB3, _sum7);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
                _sum4 = _mm_add_epi32(_sum4, _mm_madd_epi16(_pA0, _pB2));
                _sum5 = _mm_add_epi32(_sum5, _mm_madd_epi16(_pA0, _pB3));
                _sum6 = _mm_add_epi32(_sum6, _mm_madd_epi16(_pA1, _pB2));
                _sum7 = _mm_add_epi32(_sum7, _mm_madd_epi16(_pA1, _pB3));
#endif

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 44556677
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_unpackhi_epi16(_pB, _pB);
                __m128i _pB2 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                __m128i _pB3 = _mm_shuffle_epi32(_pB1, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
                _sum4 = _mm_maccd_epi16(_pA0, _pB2, _sum4);
                _sum5 = _mm_maccd_epi16(_pA0, _pB3, _sum5);
                _sum6 = _mm_maccd_epi16(_pA1, _pB2, _sum6);
                _sum7 = _mm_maccd_epi16(_pA1, _pB3, _sum7);
#else
                // 01230123
                // 23012301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567
                // 12305674
                __m128i _pB01 = _pB;
                __m128i _pB23 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _sl2 = _mm_mullo_epi16(_pA0, _pB23);
                __m128i _sh2 = _mm_mulhi_epi16(_pA0, _pB23);
                __m128i _sl3 = _mm_mullo_epi16(_pA1, _pB23);
                __m128i _sh3 = _mm_mulhi_epi16(_pA1, _pB23);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);
                __m128i _s4 = _mm_unpacklo_epi16(_sl2, _sh2);
                __m128i _s5 = _mm_unpackhi_epi16(_sl2, _sh2);
                __m128i _s6 = _mm_unpacklo_epi16(_sl3, _sh3);
                __m128i _s7 = _mm_unpackhi_epi16(_sl3, _sh3);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
                _sum4 = _mm_add_epi32(_sum4, _s4);
                _sum5 = _mm_add_epi32(_sum5, _s5);
                _sum6 = _mm_add_epi32(_sum6, _s6);
                _sum7 = _mm_add_epi32(_sum7, _s7);
#endif

                pA += 4;
                pB += 8;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);
            _mm_store_si128((__m128i*)(outptr + 16), _sum4);
            _mm_store_si128((__m128i*)(outptr + 20), _sum5);
            _mm_store_si128((__m128i*)(outptr + 24), _sum6);
            _mm_store_si128((__m128i*)(outptr + 28), _sum7);

            outptr += 32;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 16;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift0 = _mm_loadu_si128((const __m128i*)pA);
                __m128i _w_shift1 = _mm_shuffle_epi32(_w_shift0, _MM_SHUFFLE(1, 0, 3, 2));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm_sub_epi32(_sum3, _w_shift1);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0123
                // 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(1, 0, 3, 2));

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                // 22330011
                __m128i _pA0 = _mm_unpacklo_epi16(_pA, _pA);
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(1, 0, 3, 2));

                // 00112233
                // 1.2.3.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));

                _sum0 = _mm_maccd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maccd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maccd_epi16(_pA1, _pB1, _sum3);
#else
                // 0123 0123
                // 2301 2301
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123 1230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB01);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB01);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB01);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);
#endif

                pA += 4;
                pB += 4;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB0 = _mm_castpd_si128(_mm_load1_pd((const double*)pB));
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 16;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0123

                // 0101
                // 1010
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(2, 3, 0, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                // 00112233
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                // 00110011
                // 1.0.1.0.
                __m128i _pB0 = _mm_unpacklo_epi16(_pB, _pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(2, 3, 0, 1));

                _sum0 = _mm_maccd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maccd_epi16(_pA, _pB1, _sum1);
#else
                // 01230123
                // 01011010
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 1, 0, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
#endif

                pA += 4;
                pB += 2;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
            }

            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_loadu_si128((const __m128i*)pA);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 16;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)pA);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                pA += 16;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(((const short*)pB)[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB));
#endif

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_loadl_epi64((const __m128i*)pA);
                __m128i _pB = _mm_set1_epi16(pB[0]);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

#if __XOP__
                _pA = _mm_unpacklo_epi16(_pA, _pA);

                _sum0 = _mm_maccd_epi16(_pA, _pB, _sum0);
#else
                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
#endif

                pA += 4;
                pB += 1;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);

            outptr += 4;
        }

        pAT += max_kk * 4;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 16;
        }
#endif // __AVX512VNNI__
    }
#endif // __SSE2__
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0;
            __m512i _sum1;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
                _sum1 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
                _sum1 = _mm512_loadu_si512((const __m512i*)(outptr + 16));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                __m512i _pB0 = _mm512_loadu_si512((const __m512i*)pB);
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 8;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_castpd_si512(_mm512_set1_pd(((const double*)pA)[0]));
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm512_sub_epi32(_sum1, _w_shift);
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_castps_si256(_mm256_broadcast_ss((const float*)pA));
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                // 0101 0101 0101 0101

                // 0123 4567 89ab cdef
                // 1230 5674 9ab8 defc
                __m512i _pB1 = _mm512_shuffle_epi32(_pB0, _MM_PERM_ADCB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));
                _sum1 = _mm512_add_epi32(_sum1, _mm512_madd_epi16(_pA0, _pB1));

                pA += 4;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pA0 = _mm256_cvtepi8_epi16(_pA);
                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                // 01010101 01010101

                // 01234567 89abcdef
                // 12305674 9ab8defc
                __m256i _pB1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(_pB0, _MM_SHUFFLE(0, 3, 2, 1)), _MM_SHUFFLE(0, 3, 2, 1));

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB0));
                __m512i _s1 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA0, _pB1));

                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                pA += 2;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);
            _mm512_storeu_si512((__m512i*)(outptr + 16), _sum1);

            outptr += 32;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;
            __m128i _sum2;
            __m128i _sum3;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
                _sum2 = _mm_setzero_si128();
                _sum3 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
                _sum2 = _mm_load_si128((const __m128i*)(outptr + 8));
                _sum3 = _mm_load_si128((const __m128i*)(outptr + 12));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA0 = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pA1 = _mm_shuffle_epi32(_pA0, _MM_SHUFFLE(2, 3, 0, 1));
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 16));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA0);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA0);
                _sum2 = _mm_dpbusd_epi32(_sum2, _pB0, _pA1);
                _sum3 = _mm_dpbusd_epi32(_sum3, _pB1, _pA1);
                pA += 8;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift0 = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _w_shift1 = _mm_shuffle_epi32(_w_shift0, _MM_SHUFFLE(2, 3, 0, 1));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift0);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift0);
                _sum2 = _mm_sub_epi32(_sum2, _w_shift1);
                _sum3 = _mm_sub_epi32(_sum3, _w_shift1);
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

                // 0101
                // 1010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(2, 3, 0, 1));

                // 0123
                // 4567

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA0, _pB1, _sum1);
                _sum2 = _mm_maddd_epi16(_pA1, _pB0, _sum2);
                _sum3 = _mm_maddd_epi16(_pA1, _pB1, _sum3);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA0, _pB1));
                _sum2 = _mm_add_epi32(_sum2, _mm_madd_epi16(_pA1, _pB0));
                _sum3 = _mm_add_epi32(_sum3, _mm_madd_epi16(_pA1, _pB1));
#endif

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101
                // 10101010
                __m128i _pA0 = _pA;
                __m128i _pA1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(_pA, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));

                // 01234567

                __m128i _sl0 = _mm_mullo_epi16(_pA0, _pB);
                __m128i _sh0 = _mm_mulhi_epi16(_pA0, _pB);
                __m128i _sl1 = _mm_mullo_epi16(_pA1, _pB);
                __m128i _sh1 = _mm_mulhi_epi16(_pA1, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl0, _sh0);
                __m128i _s1 = _mm_unpackhi_epi16(_sl0, _sh0);
                __m128i _s2 = _mm_unpacklo_epi16(_sl1, _sh1);
                __m128i _s3 = _mm_unpackhi_epi16(_sl1, _sh1);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);
                _sum2 = _mm_add_epi32(_sum2, _s2);
                _sum3 = _mm_add_epi32(_sum3, _s3);

                pA += 2;
                pB += 8;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);
            _mm_store_si128((__m128i*)(outptr + 8), _sum2);
            _mm_store_si128((__m128i*)(outptr + 12), _sum3);

            outptr += 16;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_load_si128((const __m128i*)outptr);
                _sum1 = _mm_load_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_shuffle_epi32(_pB0, _MM_SHUFFLE(0, 3, 2, 1));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 8;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_castpd_si128(_mm_load1_pd((const double*)pA));
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0101

                // 0123
                // 1230
                __m128i _pB0 = _pB;
                __m128i _pB1 = _mm_shuffle_epi32(_pB, _MM_SHUFFLE(0, 3, 2, 1));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_castps_si128(_mm_load1_ps((const float*)pB));

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 01010101

                // 01231230
                __m128i _pB01 = _mm_shufflehi_epi16(_pB, _MM_SHUFFLE(0, 3, 2, 1));

                __m128i _sl = _mm_mullo_epi16(_pA, _pB01);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB01);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 2;
                pB += 4;
            }

            _mm_store_si128((__m128i*)outptr, _sum0);
            _mm_store_si128((__m128i*)(outptr + 4), _sum1);

            outptr += 8;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum00 += pA[0] * ((unsigned char*)pB)[0];
                sum00 += pA[1] * ((unsigned char*)pB)[1];
                sum00 += pA[2] * ((unsigned char*)pB)[2];
                sum00 += pA[3] * ((unsigned char*)pB)[3];
                sum01 += pA[0] * ((unsigned char*)pB)[4];
                sum01 += pA[1] * ((unsigned char*)pB)[5];
                sum01 += pA[2] * ((unsigned char*)pB)[6];
                sum01 += pA[3] * ((unsigned char*)pB)[7];
                sum10 += pA[4] * ((unsigned char*)pB)[0];
                sum10 += pA[5] * ((unsigned char*)pB)[1];
                sum10 += pA[6] * ((unsigned char*)pB)[2];
                sum10 += pA[7] * ((unsigned char*)pB)[3];
                sum11 += pA[4] * ((unsigned char*)pB)[4];
                sum11 += pA[5] * ((unsigned char*)pB)[5];
                sum11 += pA[6] * ((unsigned char*)pB)[6];
                sum11 += pA[7] * ((unsigned char*)pB)[7];
                pA += 8;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                int w_shift0 = ((int*)pA)[0];
                int w_shift1 = ((int*)pA)[1];
                sum00 = sum00 - w_shift0;
                sum01 = sum01 - w_shift0;
                sum10 = sum10 - w_shift1;
                sum11 = sum11 - w_shift1;
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum00 += pA[0] * pB[0];
                sum00 += pA[1] * pB[1];
                sum01 += pA[0] * pB[2];
                sum01 += pA[1] * pB[3];
                sum10 += pA[2] * pB[0];
                sum10 += pA[3] * pB[1];
                sum11 += pA[2] * pB[2];
                sum11 += pA[3] * pB[3];
                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[0] * pB[1];
                sum10 += pA[1] * pB[0];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * ((unsigned char*)pB)[0];
                sum0 += pA[1] * ((unsigned char*)pB)[1];
                sum0 += pA[2] * ((unsigned char*)pB)[2];
                sum0 += pA[3] * ((unsigned char*)pB)[3];
                sum1 += pA[4] * ((unsigned char*)pB)[0];
                sum1 += pA[5] * ((unsigned char*)pB)[1];
                sum1 += pA[6] * ((unsigned char*)pB)[2];
                sum1 += pA[7] * ((unsigned char*)pB)[3];
                pA += 8;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                int w_shift0 = ((int*)pA)[0];
                int w_shift1 = ((int*)pA)[1];
                sum0 = sum0 - w_shift0;
                sum1 = sum1 - w_shift1;
                pA += 8;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 8;
        }
#endif // __AVX512VNNI__
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __SSE2__
#if defined(__x86_64__) || defined(_M_X64)
#if __AVX512F__
        for (; jj + 15 < max_jj; jj += 16)
        {
            __m512i _sum0;

            if (k == 0)
            {
                _sum0 = _mm512_setzero_si512();
            }
            else
            {
                _sum0 = _mm512_loadu_si512((const __m512i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m512i _pA = _mm512_set1_epi32(((const int*)pA)[0]);
                __m512i _pB = _mm512_loadu_si512((const __m512i*)pB);
                _sum0 = _mm512_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 4;
                pB += 64;
            }
            if (max_kk >= 4)
            {
                __m512i _w_shift = _mm512_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm512_sub_epi32(_sum0, _w_shift);
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m256i _pA = _mm256_set1_epi16(((const short*)pA)[0]);
                __m256i _pB = _mm256_loadu_si256((const __m256i*)pB);

                __m512i _pA0 = _mm512_cvtepi8_epi16(_pA);
                __m512i _pB0 = _mm512_cvtepi8_epi16(_pB);

                _sum0 = _mm512_add_epi32(_sum0, _mm512_madd_epi16(_pA0, _pB0));

                pA += 2;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m256i _pA = _mm256_set1_epi16(pA[0]);
                __m128i _pB = _mm_load_si128((const __m128i*)pB);

                __m256i _pB0 = _mm256_cvtepi8_epi16(_pB);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_pA, _pB0));

                _sum0 = _mm512_add_epi32(_sum0, _s0);

                pA += 1;
                pB += 16;
            }

            _mm512_storeu_si512((__m512i*)outptr, _sum0);

            outptr += 16;
        }
#endif // __AVX512F__
        for (; jj + 7 < max_jj; jj += 8)
        {
            __m128i _sum0;
            __m128i _sum1;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
                _sum1 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_loadu_si128((const __m128i*)outptr);
                _sum1 = _mm_loadu_si128((const __m128i*)(outptr + 4));
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB0 = _mm_loadu_si128((const __m128i*)pB);
                __m128i _pB1 = _mm_loadu_si128((const __m128i*)(pB + 16));
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB0, _pA);
                _sum1 = _mm_dpbusd_epi32(_sum1, _pB1, _pA);
                pA += 4;
                pB += 32;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                _sum1 = _mm_sub_epi32(_sum1, _w_shift);
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_set1_epi16(((const short*)pA)[0]);
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
#endif

                __m128i _extpB = _mm_cmpgt_epi8(_mm_setzero_si128(), _pB);
                __m128i _pB0 = _mm_unpacklo_epi8(_pB, _extpB);
                __m128i _pB1 = _mm_unpackhi_epi8(_pB, _extpB);

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA, _pB0, _sum0);
                _sum1 = _mm_maddd_epi16(_pA, _pB1, _sum1);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA, _pB0));
                _sum1 = _mm_add_epi32(_sum1, _mm_madd_epi16(_pA, _pB1));
#endif

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                pA += 1;
                pB += 8;
            }

            _mm_storeu_si128((__m128i*)outptr, _sum0);
            _mm_storeu_si128((__m128i*)(outptr + 4), _sum1);

            outptr += 8;
        }
#endif // defined(__x86_64__) || defined(_M_X64)
        for (; jj + 3 < max_jj; jj += 4)
        {
            __m128i _sum0;

            if (k == 0)
            {
                _sum0 = _mm_setzero_si128();
            }
            else
            {
                _sum0 = _mm_loadu_si128((const __m128i*)outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadu_si128((const __m128i*)pB);
                _sum0 = _mm_dpbusd_epi32(_sum0, _pB, _pA);
                pA += 4;
                pB += 16;
            }
            if (max_kk >= 4)
            {
                __m128i _w_shift = _mm_set1_epi32(((const int*)pA)[0]);
                _sum0 = _mm_sub_epi32(_sum0, _w_shift);
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                __m128i _pA = _mm_castps_si128(_mm_load1_ps((const float*)pA));
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pA = _mm_cvtepi8_epi16(_pA);
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pA = _mm_unpacklo_epi8(_pA, _mm_cmpgt_epi8(_mm_setzero_si128(), _pA));
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                // 0xxx -> 0000
                __m128i _pA0 = _mm_shuffle_epi32(_pA, _MM_SHUFFLE(0, 0, 0, 0));

#if __XOP__
                _sum0 = _mm_maddd_epi16(_pA0, _pB, _sum0);
#else
                _sum0 = _mm_add_epi32(_sum0, _mm_madd_epi16(_pA0, _pB));
#endif

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                __m128i _pA = _mm_set1_epi16(pA[0]);
                __m128i _pB = _mm_loadl_epi64((const __m128i*)pB);

#if __SSE4_1__
                _pB = _mm_cvtepi8_epi16(_pB);
#else
                _pB = _mm_unpacklo_epi8(_pB, _mm_cmpgt_epi8(_mm_setzero_si128(), _pB));
#endif

                __m128i _sl = _mm_mullo_epi16(_pA, _pB);
                __m128i _sh = _mm_mulhi_epi16(_pA, _pB);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);

                pA += 1;
                pB += 4;
            }

            _mm_storeu_si128((__m128i*)outptr, _sum0);

            outptr += 4;
        }
#endif // __SSE2__
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * ((unsigned char*)pB)[0];
                sum0 += pA[1] * ((unsigned char*)pB)[1];
                sum0 += pA[2] * ((unsigned char*)pB)[2];
                sum0 += pA[3] * ((unsigned char*)pB)[3];
                sum1 += pA[0] * ((unsigned char*)pB)[4];
                sum1 += pA[1] * ((unsigned char*)pB)[5];
                sum1 += pA[2] * ((unsigned char*)pB)[6];
                sum1 += pA[3] * ((unsigned char*)pB)[7];
                pA += 4;
                pB += 8;
            }
            if (max_kk >= 4)
            {
                int w_shift = ((const int*)pA)[0];
                sum0 = sum0 - w_shift;
                sum1 = sum1 - w_shift;
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __AVX512VNNI__
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum += pA[0] * ((unsigned char*)pB)[0];
                sum += pA[1] * ((unsigned char*)pB)[1];
                sum += pA[2] * ((unsigned char*)pB)[2];
                sum += pA[3] * ((unsigned char*)pB)[3];
                pA += 4;
                pB += 4;
            }
            if (max_kk >= 4)
            {
                int w_shift = ((const int*)pA)[0];
                sum = sum - w_shift;
                pA += 4;
            }
#endif // __AVX512VNNI__
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
#if __AVX512VNNI__
        if (max_kk >= 4)
        {
            pAT += 4;
        }
#endif // __AVX512VNNI__
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(int)));

#if __AVX512F__
    TILE_M = std::max(16, tile_size / 16 * 16);
    TILE_K = std::max(16, tile_size / 16 * 16);
#elif __AVX__
    TILE_M = std::max(8, tile_size / 8 * 8);
    TILE_K = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_M = std::max(4, tile_size / 4 * 4);
    TILE_K = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = std::max(2, tile_size / 2 * 2);
    TILE_K = std::max(2, tile_size / 2 * 2);
#endif

#if __AVX512F__
    TILE_N = std::max(16, tile_size / 16 * 16);
#elif defined(__x86_64__) || defined(_M_X64)
    TILE_N = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
    TILE_N = std::max(4, tile_size / 4 * 4);
#else
    TILE_N = std::max(1, tile_size);
#endif

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __AVX512F__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 15) / 16 * 16);
#elif __AVX__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __SSE2__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

#if __AVX512F__
            TILE_M = std::max(16, tile_size / 16 * 16);
#elif __AVX__
            TILE_M = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
            TILE_M = std::max(4, tile_size / 4 * 4);
#else
            TILE_M = std::max(2, tile_size / 2 * 2);
#endif

#if __AVX512F__
            TILE_N = std::max(16, tile_size / 16 * 16);
#elif defined(__x86_64__) || defined(_M_X64)
            TILE_N = std::max(8, tile_size / 8 * 8);
#elif __SSE2__
            TILE_N = std::max(4, tile_size / 4 * 4);
#else
            TILE_N = std::max(1, tile_size);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __AVX512F__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __AVX512F__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 15) / 16 * 16);
#elif defined(__x86_64__) || defined(_M_X64)
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#elif __SSE2__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }

    if (nT > 1)
    {
#if __AVX512F__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 15) / 16 * 16);
#elif __AVX__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __SSE2__
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
#if __AVX512F__
        TILE_M = (constant_TILE_M + 15) / 16 * 16;
#elif __AVX__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __SSE2__
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __AVX512F__
        TILE_N = (constant_TILE_N + 15) / 16 * 16;
#elif defined(__x86_64__) || defined(_M_X64)
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#elif __SSE2__
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = constant_TILE_N;
#endif
    }

    if (constant_TILE_K > 0)
    {
#if __AVX512F__
        TILE_K = (constant_TILE_K + 15) / 16 * 16;
#elif __AVX__
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#elif __SSE2__
        TILE_K = (constant_TILE_K + 3) / 4 * 4;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}
