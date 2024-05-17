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

#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
void gemm_transB_packed_tile_fp16s_asimdfhm(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end);
#endif

static void pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __aarch64__
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
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p4)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p5)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p6)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p7)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7 + 4)));
            transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            vst1q_u16(pp, _r0);
            vst1q_u16(pp + 8, _r1);
            vst1q_u16(pp + 8 * 2, _r2);
            vst1q_u16(pp + 8 * 3, _r3);
            vst1q_u16(pp + 8 * 4, _r4);
            vst1q_u16(pp + 8 * 5, _r5);
            vst1q_u16(pp + 8 * 6, _r6);
            vst1q_u16(pp + 8 * 7, _r7);
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
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            vst4q_u16(pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x4_t _r0123;
            _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r01.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += A_hstep;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += A_hstep;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p0[1]);
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
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
        const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
        const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
        const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
        const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            uint16x4_t _r1 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            uint16x4_t _r2 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            uint16x4_t _r3 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            uint16x4_t _r4 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4));
            uint16x4_t _r5 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5));
            uint16x4_t _r6 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6));
            uint16x4_t _r7 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7));
            uint16x4_t _r8 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p8));
            uint16x4_t _r9 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p9));
            uint16x4_t _ra = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pa));
            uint16x4_t _rb = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pb));

            transpose4x4_u16(_r0, _r1, _r2, _r3);
            transpose4x4_u16(_r4, _r5, _r6, _r7);
            transpose4x4_u16(_r8, _r9, _ra, _rb);

            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r4);
            vst1_u16(pp + 4 * 2, _r8);
            vst1_u16(pp + 4 * 3, _r1);
            vst1_u16(pp + 4 * 4, _r5);
            vst1_u16(pp + 4 * 5, _r9);
            vst1_u16(pp + 4 * 6, _r2);
            vst1_u16(pp + 4 * 7, _r6);
            vst1_u16(pp + 4 * 8, _ra);
            vst1_u16(pp + 4 * 9, _r3);
            vst1_u16(pp + 4 * 10, _r7);
            vst1_u16(pp + 4 * 11, _rb);
            pp += 48;
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
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
            pp[8] = float32_to_float16(p8[0]);
            pp[9] = float32_to_float16(p9[0]);
            pp[10] = float32_to_float16(pa[0]);
            pp[11] = float32_to_float16(pb[0]);
            pp += 12;
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
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            uint16x8_t _r1 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            uint16x8_t _r2 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            uint16x8_t _r3 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            uint16x8_t _r4 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p4)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4 + 4)));
            uint16x8_t _r5 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p5)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5 + 4)));
            uint16x8_t _r6 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p6)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6 + 4)));
            uint16x8_t _r7 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p7)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7 + 4)));
            transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            vst1q_u16(pp, _r0);
            vst1q_u16(pp + 8, _r1);
            vst1q_u16(pp + 8 * 2, _r2);
            vst1q_u16(pp + 8 * 3, _r3);
            vst1q_u16(pp + 8 * 4, _r4);
            vst1q_u16(pp + 8 * 5, _r5);
            vst1q_u16(pp + 8 * 6, _r6);
            vst1q_u16(pp + 8 * 7, _r7);
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
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            uint16x4_t _r1 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            uint16x4_t _r2 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            uint16x4_t _r3 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            uint16x4_t _r4 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p4));
            uint16x4_t _r5 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p5));
            uint16x4_t _r6 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p6));
            uint16x4_t _r7 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p7));

            transpose4x4_u16(_r0, _r1, _r2, _r3);
            transpose4x4_u16(_r4, _r5, _r6, _r7);

            vst1_u16(pp, _r0);
            vst1_u16(pp + 4, _r4);
            vst1_u16(pp + 4 * 2, _r1);
            vst1_u16(pp + 4 * 3, _r5);
            vst1_u16(pp + 4 * 4, _r2);
            vst1_u16(pp + 4 * 5, _r6);
            vst1_u16(pp + 4 * 6, _r3);
            vst1_u16(pp + 4 * 7, _r7);
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
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x4_t _r0123;
            _r0123.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r0123.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            _r0123.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p2)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2 + 4)));
            _r0123.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p3)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3 + 4)));
            vst4q_u16(pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x4_t _r0123;
            _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p2));
            _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p3));
            vst4_u16(pp, _r0123);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8x2_t _r01;
            _r01.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            _r01.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p1)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1 + 4)));
            vst2q_u16(pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4x2_t _r01;
            _r01.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            _r01.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p1));
            vst2_u16(pp, _r01);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 7 < max_kk; kk += 8)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
            pp += 8;
            p0 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vst1_u16(pp, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)));
            vst1_u16(pp + 4, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1_u16(pp + 8, (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 8)));
            pp += 12;
            p0 += B_hstep;
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(p0)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0 + 4)));
            vst1q_u16(pp, _r0);
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
            uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(p0));
            vst1_u16(pp, _r0);
            pp += 4;
            p0 += B_hstep;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p0[1]);
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
            pp[0] = float32_to_float16(p0[0]);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void transpose_unpack_output_tile_fp32_to_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x8x4_t _r0;
                _r0.val[0] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4)));
                _r0.val[1] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 8)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 12)));
                _r0.val[2] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 16)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 20)));
                _r0.val[3] = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 24)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 28)));
                vst4q_u16(p0, _r0);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                uint16x8_t _r0 = vcombine_u16((uint16x4_t)vcvt_f16_f32(vld1q_f32(pp)), (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4)));
                vst1q_u16(p0, _r0);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4x4_t _r0123;
                _r0123.val[0] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
                _r0123.val[1] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 4));
                _r0123.val[2] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 8));
                _r0123.val[3] = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp + 12));
                vst4_u16(p0, _r0123);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
                vst1_u16(p0, _r0);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[2]);
                p0[2] = float32_to_float16(pp[4]);
                p0[3] = float32_to_float16(pp[6]);
                p0[4] = float32_to_float16(pp[1]);
                p0[5] = float32_to_float16(pp[3]);
                p0[6] = float32_to_float16(pp[5]);
                p0[7] = float32_to_float16(pp[7]);
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[1]);
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __ARM_NEON
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                uint16x4_t _r0 = (uint16x4_t)vcvt_f16_f32(vld1q_f32(pp));
                vst1_u16(p0, _r0);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_float16(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_fp16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        gemm_transB_packed_tile_fp16s_asimdfhm(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

#if __ARM_FEATURE_FP16_FML
    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
#else
    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
#endif
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

#if __ARM_FEATURE_FP16_FML
        const __fp16* pB = pBT;
#else
        const unsigned short* pB = pBT;
#endif

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
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;
            float32x4_t _sum80;
            float32x4_t _sum81;
            float32x4_t _sum90;
            float32x4_t _sum91;
            float32x4_t _suma0;
            float32x4_t _suma1;
            float32x4_t _sumb0;
            float32x4_t _sumb1;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);
                _sum80 = vdupq_n_f32(0.f);
                _sum81 = vdupq_n_f32(0.f);
                _sum90 = vdupq_n_f32(0.f);
                _sum91 = vdupq_n_f32(0.f);
                _suma0 = vdupq_n_f32(0.f);
                _suma1 = vdupq_n_f32(0.f);
                _sumb0 = vdupq_n_f32(0.f);
                _sumb1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum20 = _sum00;
                        _sum21 = _sum00;
                        _sum30 = _sum00;
                        _sum31 = _sum00;
                        _sum40 = _sum00;
                        _sum41 = _sum00;
                        _sum50 = _sum00;
                        _sum51 = _sum00;
                        _sum60 = _sum00;
                        _sum61 = _sum00;
                        _sum70 = _sum00;
                        _sum71 = _sum00;
                        _sum80 = _sum00;
                        _sum81 = _sum00;
                        _sum90 = _sum00;
                        _sum91 = _sum00;
                        _suma0 = _sum00;
                        _suma1 = _sum00;
                        _sumb0 = _sum00;
                        _sumb1 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                        _sum40 = _sum00;
                        _sum41 = _sum01;
                        _sum50 = _sum00;
                        _sum51 = _sum01;
                        _sum60 = _sum00;
                        _sum61 = _sum01;
                        _sum70 = _sum00;
                        _sum71 = _sum01;
                        _sum80 = _sum00;
                        _sum81 = _sum01;
                        _sum90 = _sum00;
                        _sum91 = _sum01;
                        _suma0 = _sum00;
                        _suma1 = _sum01;
                        _sumb0 = _sum00;
                        _sumb1 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        _sum80 = vld1q_f32(pC + 4 * 16);
                        _sum81 = vld1q_f32(pC + 4 * 17);
                        _sum90 = vld1q_f32(pC + 4 * 18);
                        _sum91 = vld1q_f32(pC + 4 * 19);
                        _suma0 = vld1q_f32(pC + 4 * 20);
                        _suma1 = vld1q_f32(pC + 4 * 21);
                        _sumb0 = vld1q_f32(pC + 4 * 22);
                        _sumb1 = vld1q_f32(pC + 4 * 23);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum80 = vdupq_n_f32(pC[8]);
                        _sum90 = vdupq_n_f32(pC[9]);
                        _suma0 = vdupq_n_f32(pC[10]);
                        _sumb0 = vdupq_n_f32(pC[11]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        _sum81 = _sum80;
                        _sum91 = _sum90;
                        _suma1 = _suma0;
                        _sumb1 = _sumb0;
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
                _sum80 = vld1q_f32(outptr + 4 * 16);
                _sum81 = vld1q_f32(outptr + 4 * 17);
                _sum90 = vld1q_f32(outptr + 4 * 18);
                _sum91 = vld1q_f32(outptr + 4 * 19);
                _suma0 = vld1q_f32(outptr + 4 * 20);
                _suma1 = vld1q_f32(outptr + 4 * 21);
                _sumb0 = vld1q_f32(outptr + 4 * 22);
                _sumb1 = vld1q_f32(outptr + 4 * 23);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pA = vld1q_f16(pA);

                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                _sum00 = vfmlalq_lane_low_f16(_sum00, _pA, _pB0, 0);
                _sum01 = vfmlalq_lane_high_f16(_sum01, _pA, _pB0, 0);
                _sum10 = vfmlalq_lane_low_f16(_sum10, _pA, _pB0, 1);
                _sum11 = vfmlalq_lane_high_f16(_sum11, _pA, _pB0, 1);
                _sum20 = vfmlalq_lane_low_f16(_sum20, _pA, _pB0, 2);
                _sum21 = vfmlalq_lane_high_f16(_sum21, _pA, _pB0, 2);
                _sum30 = vfmlalq_lane_low_f16(_sum30, _pA, _pB0, 3);
                _sum31 = vfmlalq_lane_high_f16(_sum31, _pA, _pB0, 3);
                _sum40 = vfmlalq_lane_low_f16(_sum40, _pA, _pB1, 0);
                _sum41 = vfmlalq_lane_high_f16(_sum41, _pA, _pB1, 0);
                _sum50 = vfmlalq_lane_low_f16(_sum50, _pA, _pB1, 1);
                _sum51 = vfmlalq_lane_high_f16(_sum51, _pA, _pB1, 1);
                _sum60 = vfmlalq_lane_low_f16(_sum60, _pA, _pB1, 2);
                _sum61 = vfmlalq_lane_high_f16(_sum61, _pA, _pB1, 2);
                _sum70 = vfmlalq_lane_low_f16(_sum70, _pA, _pB1, 3);
                _sum71 = vfmlalq_lane_high_f16(_sum71, _pA, _pB1, 3);
                _sum80 = vfmlalq_lane_low_f16(_sum80, _pA, _pB2, 0);
                _sum81 = vfmlalq_lane_high_f16(_sum81, _pA, _pB2, 0);
                _sum90 = vfmlalq_lane_low_f16(_sum90, _pA, _pB2, 1);
                _sum91 = vfmlalq_lane_high_f16(_sum91, _pA, _pB2, 1);
                _suma0 = vfmlalq_lane_low_f16(_suma0, _pA, _pB2, 2);
                _suma1 = vfmlalq_lane_high_f16(_suma1, _pA, _pB2, 2);
                _sumb0 = vfmlalq_lane_low_f16(_sumb0, _pA, _pB2, 3);
                _sumb1 = vfmlalq_lane_high_f16(_sumb1, _pA, _pB2, 3);
#else
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pA));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pA));

                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 4));
                float32x4_t _pB2 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 8));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);
#endif

                pA += 8;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
                _sum20 = vmulq_f32(_sum20, _alpha);
                _sum21 = vmulq_f32(_sum21, _alpha);
                _sum30 = vmulq_f32(_sum30, _alpha);
                _sum31 = vmulq_f32(_sum31, _alpha);
                _sum40 = vmulq_f32(_sum40, _alpha);
                _sum41 = vmulq_f32(_sum41, _alpha);
                _sum50 = vmulq_f32(_sum50, _alpha);
                _sum51 = vmulq_f32(_sum51, _alpha);
                _sum60 = vmulq_f32(_sum60, _alpha);
                _sum61 = vmulq_f32(_sum61, _alpha);
                _sum70 = vmulq_f32(_sum70, _alpha);
                _sum71 = vmulq_f32(_sum71, _alpha);
                _sum80 = vmulq_f32(_sum80, _alpha);
                _sum81 = vmulq_f32(_sum81, _alpha);
                _sum90 = vmulq_f32(_sum90, _alpha);
                _sum91 = vmulq_f32(_sum91, _alpha);
                _suma0 = vmulq_f32(_suma0, _alpha);
                _suma1 = vmulq_f32(_suma1, _alpha);
                _sumb0 = vmulq_f32(_sumb0, _alpha);
                _sumb1 = vmulq_f32(_sumb1, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum30));
                    vst1_u16(outptr0 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum40));
                    vst1_u16(outptr0 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum50));
                    vst1_u16(outptr0 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum60));
                    vst1_u16(outptr0 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum70));
                    vst1_u16(outptr0 + 4 * 8, (uint16x4_t)vcvt_f16_f32(_sum80));
                    vst1_u16(outptr0 + 4 * 9, (uint16x4_t)vcvt_f16_f32(_sum90));
                    vst1_u16(outptr0 + 4 * 10, (uint16x4_t)vcvt_f16_f32(_suma0));
                    vst1_u16(outptr0 + 4 * 11, (uint16x4_t)vcvt_f16_f32(_sumb0));

                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum71));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 8, (uint16x4_t)vcvt_f16_f32(_sum81));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 9, (uint16x4_t)vcvt_f16_f32(_sum91));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 10, (uint16x4_t)vcvt_f16_f32(_suma1));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 11, (uint16x4_t)vcvt_f16_f32(_sumb1));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x12_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, _sum80, _sum81, _sum90, _sum91, _suma0, _suma1, _sumb0, _sumb1);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + 8, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + out_hstep + 8, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum30));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, (uint16x4_t)vcvt_f16_f32(_sum31));
                    vst1_u16(outptr0 + out_hstep * 2 + 8, (uint16x4_t)vcvt_f16_f32(_sum40));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum41));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, (uint16x4_t)vcvt_f16_f32(_sum50));
                    vst1_u16(outptr0 + out_hstep * 3 + 8, (uint16x4_t)vcvt_f16_f32(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum60));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 8, (uint16x4_t)vcvt_f16_f32(_sum70));
                    vst1_u16(outptr0 + out_hstep * 5, (uint16x4_t)vcvt_f16_f32(_sum71));
                    vst1_u16(outptr0 + out_hstep * 5 + 4, (uint16x4_t)vcvt_f16_f32(_sum80));
                    vst1_u16(outptr0 + out_hstep * 5 + 8, (uint16x4_t)vcvt_f16_f32(_sum81));
                    vst1_u16(outptr0 + out_hstep * 6, (uint16x4_t)vcvt_f16_f32(_sum90));
                    vst1_u16(outptr0 + out_hstep * 6 + 4, (uint16x4_t)vcvt_f16_f32(_sum91));
                    vst1_u16(outptr0 + out_hstep * 6 + 8, (uint16x4_t)vcvt_f16_f32(_suma0));
                    vst1_u16(outptr0 + out_hstep * 7, (uint16x4_t)vcvt_f16_f32(_suma1));
                    vst1_u16(outptr0 + out_hstep * 7 + 4, (uint16x4_t)vcvt_f16_f32(_sumb0));
                    vst1_u16(outptr0 + out_hstep * 7 + 8, (uint16x4_t)vcvt_f16_f32(_sumb1));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
                vst1q_f32(outptr + 4 * 16, _sum80);
                vst1q_f32(outptr + 4 * 17, _sum81);
                vst1q_f32(outptr + 4 * 18, _sum90);
                vst1q_f32(outptr + 4 * 19, _sum91);
                vst1q_f32(outptr + 4 * 20, _suma0);
                vst1q_f32(outptr + 4 * 21, _suma1);
                vst1q_f32(outptr + 4 * 22, _sumb0);
                vst1q_f32(outptr + 4 * 23, _sumb1);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum20 = _sum00;
                        _sum21 = _sum00;
                        _sum30 = _sum00;
                        _sum31 = _sum00;
                        _sum40 = _sum00;
                        _sum41 = _sum00;
                        _sum50 = _sum00;
                        _sum51 = _sum00;
                        _sum60 = _sum00;
                        _sum61 = _sum00;
                        _sum70 = _sum00;
                        _sum71 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                        _sum40 = _sum00;
                        _sum41 = _sum01;
                        _sum50 = _sum00;
                        _sum51 = _sum01;
                        _sum60 = _sum00;
                        _sum61 = _sum01;
                        _sum70 = _sum00;
                        _sum71 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pA = vld1q_f16(pA);
                float16x8_t _pB = vld1q_f16(pB);

                _sum00 = vfmlalq_laneq_low_f16(_sum00, _pA, _pB, 0);
                _sum01 = vfmlalq_laneq_high_f16(_sum01, _pA, _pB, 0);
                _sum10 = vfmlalq_laneq_low_f16(_sum10, _pA, _pB, 1);
                _sum11 = vfmlalq_laneq_high_f16(_sum11, _pA, _pB, 1);
                _sum20 = vfmlalq_laneq_low_f16(_sum20, _pA, _pB, 2);
                _sum21 = vfmlalq_laneq_high_f16(_sum21, _pA, _pB, 2);
                _sum30 = vfmlalq_laneq_low_f16(_sum30, _pA, _pB, 3);
                _sum31 = vfmlalq_laneq_high_f16(_sum31, _pA, _pB, 3);
                _sum40 = vfmlalq_laneq_low_f16(_sum40, _pA, _pB, 4);
                _sum41 = vfmlalq_laneq_high_f16(_sum41, _pA, _pB, 4);
                _sum50 = vfmlalq_laneq_low_f16(_sum50, _pA, _pB, 5);
                _sum51 = vfmlalq_laneq_high_f16(_sum51, _pA, _pB, 5);
                _sum60 = vfmlalq_laneq_low_f16(_sum60, _pA, _pB, 6);
                _sum61 = vfmlalq_laneq_high_f16(_sum61, _pA, _pB, 6);
                _sum70 = vfmlalq_laneq_low_f16(_sum70, _pA, _pB, 7);
                _sum71 = vfmlalq_laneq_high_f16(_sum71, _pA, _pB, 7);
#else
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pA));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pA));

                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 4));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
#endif

                pA += 8;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
                _sum20 = vmulq_f32(_sum20, _alpha);
                _sum21 = vmulq_f32(_sum21, _alpha);
                _sum30 = vmulq_f32(_sum30, _alpha);
                _sum31 = vmulq_f32(_sum31, _alpha);
                _sum40 = vmulq_f32(_sum40, _alpha);
                _sum41 = vmulq_f32(_sum41, _alpha);
                _sum50 = vmulq_f32(_sum50, _alpha);
                _sum51 = vmulq_f32(_sum51, _alpha);
                _sum60 = vmulq_f32(_sum60, _alpha);
                _sum61 = vmulq_f32(_sum61, _alpha);
                _sum70 = vmulq_f32(_sum70, _alpha);
                _sum71 = vmulq_f32(_sum71, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum30));
                    vst1_u16(outptr0 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum40));
                    vst1_u16(outptr0 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum50));
                    vst1_u16(outptr0 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum60));
                    vst1_u16(outptr0 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum70));

                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum71));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum30));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, (uint16x4_t)vcvt_f16_f32(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum40));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum41));
                    vst1_u16(outptr0 + out_hstep * 5, (uint16x4_t)vcvt_f16_f32(_sum50));
                    vst1_u16(outptr0 + out_hstep * 5 + 4, (uint16x4_t)vcvt_f16_f32(_sum51));
                    vst1_u16(outptr0 + out_hstep * 6, (uint16x4_t)vcvt_f16_f32(_sum60));
                    vst1_u16(outptr0 + out_hstep * 6 + 4, (uint16x4_t)vcvt_f16_f32(_sum61));
                    vst1_u16(outptr0 + out_hstep * 7, (uint16x4_t)vcvt_f16_f32(_sum70));
                    vst1_u16(outptr0 + out_hstep * 7 + 4, (uint16x4_t)vcvt_f16_f32(_sum71));

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum20 = _sum00;
                        _sum21 = _sum00;
                        _sum30 = _sum00;
                        _sum31 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pA = vld1q_f16(pA);
                float16x4_t _pB = vld1_f16(pB);

                _sum00 = vfmlalq_lane_low_f16(_sum00, _pA, _pB, 0);
                _sum01 = vfmlalq_lane_high_f16(_sum01, _pA, _pB, 0);
                _sum10 = vfmlalq_lane_low_f16(_sum10, _pA, _pB, 1);
                _sum11 = vfmlalq_lane_high_f16(_sum11, _pA, _pB, 1);
                _sum20 = vfmlalq_lane_low_f16(_sum20, _pA, _pB, 2);
                _sum21 = vfmlalq_lane_high_f16(_sum21, _pA, _pB, 2);
                _sum30 = vfmlalq_lane_low_f16(_sum30, _pA, _pB, 3);
                _sum31 = vfmlalq_lane_high_f16(_sum31, _pA, _pB, 3);
#else
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pA));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pA));

                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
#endif

                pA += 8;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
                _sum20 = vmulq_f32(_sum20, _alpha);
                _sum21 = vmulq_f32(_sum21, _alpha);
                _sum30 = vmulq_f32(_sum30, _alpha);
                _sum31 = vmulq_f32(_sum31, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum30));

                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum31));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + out_hstep * 1, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum20));
                    vst1_u16(outptr0 + out_hstep * 5, (uint16x4_t)vcvt_f16_f32(_sum21));
                    vst1_u16(outptr0 + out_hstep * 6, (uint16x4_t)vcvt_f16_f32(_sum30));
                    vst1_u16(outptr0 + out_hstep * 7, (uint16x4_t)vcvt_f16_f32(_sum31));

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pA = vld1q_f16(pA);
                float16x4_t _pB0 = vdup_n_f16(pB[0]);
                float16x4_t _pB1 = vdup_n_f16(pB[1]);
                float16x8_t _pB01 = vcombine_f16(_pB0, _pB1);
                float16x8_t _pB10 = vcombine_f16(_pB1, _pB0);

                _sum00 = vfmlalq_low_f16(_sum00, _pA, _pB01);
                _sum01 = vfmlalq_high_f16(_sum01, _pA, _pB10);
                _sum10 = vfmlalq_low_f16(_sum10, _pA, _pB10);
                _sum11 = vfmlalq_high_f16(_sum11, _pA, _pB01);
#else
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pA));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pA));

                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pB[0]));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pB[1]));

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB0);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB0);
                _sum10 = vfmaq_f32(_sum10, _pA0, _pB1);
                _sum11 = vfmaq_f32(_sum11, _pA1, _pB1);
#endif

                pA += 8;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum10));

                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    vst1_u16(sum0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(sum0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(sum1, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(sum1 + 4, (uint16x4_t)vcvt_f16_f32(_sum11));

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
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pA = vld1q_f16(pA);
                float16x8_t _pB = vdupq_n_f16(pB[0]);

                _sum00 = vfmlalq_low_f16(_sum00, _pA, _pB);
                _sum01 = vfmlalq_high_f16(_sum01, _pA, _pB);
#else
                uint16x8_t _pA = vld1q_u16(pA);
                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pA));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pA));

                float32x4_t _pB = vcvt_f32_f16((float16x4_t)vld1_dup_u16(pB));

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);
#endif

                pA += 8;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + out_hstep * 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    vst1_u16(sum0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(sum0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));

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
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

#if __ARM_FEATURE_FP16_FML
        const __fp16* pB = pBT;
#else
        const unsigned short* pB = pBT;
#endif

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
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;
            float32x4_t _sum8;
            float32x4_t _sum9;
            float32x4_t _suma;
            float32x4_t _sumb;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);
                _sum8 = vdupq_n_f32(0.f);
                _sum9 = vdupq_n_f32(0.f);
                _suma = vdupq_n_f32(0.f);
                _sumb = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
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
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
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
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        _sum8 = vld1q_f32(pC + 32);
                        _sum9 = vld1q_f32(pC + 36);
                        _suma = vld1q_f32(pC + 40);
                        _sumb = vld1q_f32(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        _sum8 = vdupq_n_f32(pC[8]);
                        _sum9 = vdupq_n_f32(pC[9]);
                        _suma = vdupq_n_f32(pC[10]);
                        _sumb = vdupq_n_f32(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
                _sum8 = vld1q_f32(outptr + 4 * 8);
                _sum9 = vld1q_f32(outptr + 4 * 9);
                _suma = vld1q_f32(outptr + 4 * 10);
                _sumb = vld1q_f32(outptr + 4 * 11);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pAA = vcombine_f16(_pA, _pA);

                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);
                float16x4_t _pB2 = vld1_f16(pB + 8);

                _sum0 = vfmlalq_lane_low_f16(_sum0, _pAA, _pB0, 0);
                _sum1 = vfmlalq_lane_low_f16(_sum1, _pAA, _pB0, 1);
                _sum2 = vfmlalq_lane_low_f16(_sum2, _pAA, _pB0, 2);
                _sum3 = vfmlalq_lane_low_f16(_sum3, _pAA, _pB0, 3);
                _sum4 = vfmlalq_lane_low_f16(_sum4, _pAA, _pB1, 0);
                _sum5 = vfmlalq_lane_low_f16(_sum5, _pAA, _pB1, 1);
                _sum6 = vfmlalq_lane_low_f16(_sum6, _pAA, _pB1, 2);
                _sum7 = vfmlalq_lane_low_f16(_sum7, _pAA, _pB1, 3);
                _sum8 = vfmlalq_lane_low_f16(_sum8, _pAA, _pB2, 0);
                _sum9 = vfmlalq_lane_low_f16(_sum9, _pAA, _pB2, 1);
                _suma = vfmlalq_lane_low_f16(_suma, _pAA, _pB2, 2);
                _sumb = vfmlalq_lane_low_f16(_sumb, _pAA, _pB2, 3);

                pA += 4;
                pB += 12;
#else
#if __aarch64__
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 4));
                float32x4_t _pB2 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 8));

                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
#else // __aarch64__
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "pld        [%0, #64]       \n"
                    "pld        [%1, #192]      \n"
                    "vld1.u16   {d6}, [%0 :64]! \n"
                    "vld1.u16   {d2-d4}, [%1 :64]! \n"
                    "vcvt.f32.f16 q3, d6        \n"
                    "vcvt.f32.f16 q0, d2        \n"
                    "vcvt.f32.f16 q1, d3        \n"
                    "vcvt.f32.f16 q2, d4        \n"
                    "vmla.f32   %q2, q3, d0[0]  \n"
                    "vmla.f32   %q3, q3, d0[1]  \n"
                    "vmla.f32   %q4, q3, d1[0]  \n"
                    "vmla.f32   %q5, q3, d1[1]  \n"
                    "vmla.f32   %q6, q3, d2[0]  \n"
                    "vmla.f32   %q7, q3, d2[1]  \n"
                    "vmla.f32   %q8, q3, d3[0]  \n"
                    "vmla.f32   %q9, q3, d3[1]  \n"
                    "vmla.f32   %q10, q3, d4[0] \n"
                    "vmla.f32   %q11, q3, d4[1] \n"
                    "vmla.f32   %q12, q3, d5[0] \n"
                    "vmla.f32   %q13, q3, d5[1] \n"
                    : "=r"(pA),
                    "=r"(pB),
                    "=w"(_sum0),
                    "=w"(_sum1),
                    "=w"(_sum2),
                    "=w"(_sum3),
                    "=w"(_sum4),
                    "=w"(_sum5),
                    "=w"(_sum6),
                    "=w"(_sum7),
                    "=w"(_sum8),
                    "=w"(_sum9),
                    "=w"(_suma),
                    "=w"(_sumb)
                    : "0"(pA),
                    "1"(pB),
                    "2"(_sum0),
                    "3"(_sum1),
                    "4"(_sum2),
                    "5"(_sum3),
                    "6"(_sum4),
                    "7"(_sum5),
                    "8"(_sum6),
                    "9"(_sum7),
                    "10"(_sum8),
                    "11"(_sum9),
                    "12"(_suma),
                    "13"(_sumb)
                    : "memory", "q0", "q1", "q2", "q3");
#else
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 4));
                float32x4_t _pB2 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 8));

                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
                _sum8 = vmlaq_lane_f32(_sum8, _pA, vget_low_f32(_pB2), 0);
                _sum9 = vmlaq_lane_f32(_sum9, _pA, vget_low_f32(_pB2), 1);
                _suma = vmlaq_lane_f32(_suma, _pA, vget_high_f32(_pB2), 0);
                _sumb = vmlaq_lane_f32(_sumb, _pA, vget_high_f32(_pB2), 1);

                pA += 4;
                pB += 12;
#endif
#endif // __aarch64__
#endif
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
                _sum2 = vmulq_f32(_sum2, _alpha);
                _sum3 = vmulq_f32(_sum3, _alpha);
                _sum4 = vmulq_f32(_sum4, _alpha);
                _sum5 = vmulq_f32(_sum5, _alpha);
                _sum6 = vmulq_f32(_sum6, _alpha);
                _sum7 = vmulq_f32(_sum7, _alpha);
                _sum8 = vmulq_f32(_sum8, _alpha);
                _sum9 = vmulq_f32(_sum9, _alpha);
                _suma = vmulq_f32(_suma, _alpha);
                _sumb = vmulq_f32(_sumb, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum3));
                    vst1_u16(outptr0 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum4));
                    vst1_u16(outptr0 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum5));
                    vst1_u16(outptr0 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum6));
                    vst1_u16(outptr0 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum7));
                    vst1_u16(outptr0 + 4 * 8, (uint16x4_t)vcvt_f16_f32(_sum8));
                    vst1_u16(outptr0 + 4 * 9, (uint16x4_t)vcvt_f16_f32(_sum9));
                    vst1_u16(outptr0 + 4 * 10, (uint16x4_t)vcvt_f16_f32(_suma));
                    vst1_u16(outptr0 + 4 * 11, (uint16x4_t)vcvt_f16_f32(_sumb));
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + 8, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum3));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum4));
                    vst1_u16(outptr0 + out_hstep + 8, (uint16x4_t)vcvt_f16_f32(_sum5));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum6));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, (uint16x4_t)vcvt_f16_f32(_sum7));
                    vst1_u16(outptr0 + out_hstep * 2 + 8, (uint16x4_t)vcvt_f16_f32(_sum8));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum9));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, (uint16x4_t)vcvt_f16_f32(_suma));
                    vst1_u16(outptr0 + out_hstep * 3 + 8, (uint16x4_t)vcvt_f16_f32(_sumb));
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                vst1q_f32(outptr + 4 * 8, _sum8);
                vst1q_f32(outptr + 4 * 9, _sum9);
                vst1q_f32(outptr + 4 * 10, _suma);
                vst1q_f32(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
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
                        _sum0 = vld1q_f32(pC);
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
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pAA = vcombine_f16(_pA, _pA);

                float16x4_t _pB0 = vld1_f16(pB);
                float16x4_t _pB1 = vld1_f16(pB + 4);

                _sum0 = vfmlalq_lane_low_f16(_sum0, _pAA, _pB0, 0);
                _sum1 = vfmlalq_lane_low_f16(_sum1, _pAA, _pB0, 1);
                _sum2 = vfmlalq_lane_low_f16(_sum2, _pAA, _pB0, 2);
                _sum3 = vfmlalq_lane_low_f16(_sum3, _pAA, _pB0, 3);
                _sum4 = vfmlalq_lane_low_f16(_sum4, _pAA, _pB1, 0);
                _sum5 = vfmlalq_lane_low_f16(_sum5, _pAA, _pB1, 1);
                _sum6 = vfmlalq_lane_low_f16(_sum6, _pAA, _pB1, 2);
                _sum7 = vfmlalq_lane_low_f16(_sum7, _pAA, _pB1, 3);
#else
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 4));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
#endif
#endif

                pA += 4;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
                _sum2 = vmulq_f32(_sum2, _alpha);
                _sum3 = vmulq_f32(_sum3, _alpha);
                _sum4 = vmulq_f32(_sum4, _alpha);
                _sum5 = vmulq_f32(_sum5, _alpha);
                _sum6 = vmulq_f32(_sum6, _alpha);
                _sum7 = vmulq_f32(_sum7, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum3));
                    vst1_u16(outptr0 + 4 * 4, (uint16x4_t)vcvt_f16_f32(_sum4));
                    vst1_u16(outptr0 + 4 * 5, (uint16x4_t)vcvt_f16_f32(_sum5));
                    vst1_u16(outptr0 + 4 * 6, (uint16x4_t)vcvt_f16_f32(_sum6));
                    vst1_u16(outptr0 + 4 * 7, (uint16x4_t)vcvt_f16_f32(_sum7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum3));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum4));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, (uint16x4_t)vcvt_f16_f32(_sum5));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum6));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, (uint16x4_t)vcvt_f16_f32(_sum7));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pAA = vcombine_f16(_pA, _pA);

                float16x4_t _pB0 = vld1_f16(pB);

                _sum0 = vfmlalq_lane_low_f16(_sum0, _pAA, _pB0, 0);
                _sum1 = vfmlalq_lane_low_f16(_sum1, _pAA, _pB0, 1);
                _sum2 = vfmlalq_lane_low_f16(_sum2, _pAA, _pB0, 2);
                _sum3 = vfmlalq_lane_low_f16(_sum3, _pAA, _pB0, 3);
#else
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB = vcvt_f32_f16((float16x4_t)vld1_u16(pB));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB), 1);
#endif
#endif

                pA += 4;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
                _sum2 = vmulq_f32(_sum2, _alpha);
                _sum3 = vmulq_f32(_sum3, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + 4 * 2, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + 4 * 3, (uint16x4_t)vcvt_f16_f32(_sum3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + out_hstep * 2, (uint16x4_t)vcvt_f16_f32(_sum2));
                    vst1_u16(outptr0 + out_hstep * 3, (uint16x4_t)vcvt_f16_f32(_sum3));
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pAA = vcombine_f16(_pA, _pA);

                float16x4_t _pB0 = vdup_n_f16(pB[0]);
                float16x4_t _pB1 = vdup_n_f16(pB[1]);
                float16x8_t _pB01 = vcombine_f16(_pB0, _pB1);

                _sum0 = vfmlalq_low_f16(_sum0, _pAA, _pB01);
                _sum1 = vfmlalq_high_f16(_sum1, _pAA, _pB01);
#else
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pB[0]));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pB[1]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA, _pB1);
#endif
#endif

                pA += 4;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    unsigned short sum1[4];
                    vst1_u16(sum0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(sum1, (uint16x4_t)vcvt_f16_f32(_sum1));

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
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pA = vld1_f16(pA);
                float16x8_t _pAA = vcombine_f16(_pA, _pA);

                float16x8_t _pB = vdupq_n_f16(pB[0]);

                _sum0 = vfmlalq_low_f16(_sum0, _pAA, _pB);
#else
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vld1_u16(pA));
                float32x4_t _pB = vcvt_f32_f16((float16x4_t)vdup_n_u16(pB[0]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif
#endif

                pA += 4;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    vst1_u16(sum0, (uint16x4_t)vcvt_f16_f32(_sum0));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

#if __ARM_FEATURE_FP16_FML
        const __fp16* pB = pBT;
#else
        const unsigned short* pB = pBT;
#endif

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
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum02;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum12;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum02 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum12 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum12 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = _sum10;
                        _sum12 = _sum10;
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        float32x4x2_t _tmp45 = vld2q_f32(pC + 16);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum02 = _tmp45.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        _sum12 = _tmp45.val[1];
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum02 = vld1q_f32(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pB01 = vld1q_f16(pB);
                float16x4_t _pB2 = vld1_f16(pB + 8);
                float16x8_t _pB22 = vcombine_f16(_pB2, _pB2);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);
                float16x8_t _pA01 = vcombine_f16(_pA0, _pA1);
                float16x8_t _pA10 = vcombine_f16(_pA1, _pA0);

                _sum00 = vfmlalq_low_f16(_sum00, _pB01, _pA01);
                _sum01 = vfmlalq_high_f16(_sum01, _pB01, _pA10);
                _sum02 = vfmlalq_low_f16(_sum02, _pB22, _pA01);
                _sum10 = vfmlalq_low_f16(_sum10, _pB01, _pA10);
                _sum11 = vfmlalq_high_f16(_sum11, _pB01, _pA01);
                _sum12 = vfmlalq_low_f16(_sum12, _pB22, _pA10);
#else
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pB));
                float32x4_t _pB2 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 8));

                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[1]));
#if __aarch64__
                _sum00 = vfmaq_f32(_sum00, _pB0, _pA0);
                _sum01 = vfmaq_f32(_sum01, _pB1, _pA0);
                _sum02 = vfmaq_f32(_sum02, _pB2, _pA0);
                _sum10 = vfmaq_f32(_sum10, _pB0, _pA1);
                _sum11 = vfmaq_f32(_sum11, _pB1, _pA1);
                _sum12 = vfmaq_f32(_sum12, _pB2, _pA1);
#else
                _sum00 = vmlaq_f32(_sum00, _pB0, _pA0);
                _sum01 = vmlaq_f32(_sum01, _pB1, _pA0);
                _sum02 = vmlaq_f32(_sum02, _pB2, _pA0);
                _sum10 = vmlaq_f32(_sum10, _pB0, _pA1);
                _sum11 = vmlaq_f32(_sum11, _pB1, _pA1);
                _sum12 = vmlaq_f32(_sum12, _pB2, _pA1);
#endif
#endif

                pA += 2;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum02 = vmulq_f32(_sum02, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
                _sum12 = vmulq_f32(_sum12, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + 8, (uint16x4_t)vcvt_f16_f32(_sum02));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    vst1_u16(outptr0 + out_hstep + 8, (uint16x4_t)vcvt_f16_f32(_sum12));
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float32x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = _sum10;
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pB01 = vld1q_f16(pB);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);
                float16x8_t _pA01 = vcombine_f16(_pA0, _pA1);
                float16x8_t _pA10 = vcombine_f16(_pA1, _pA0);

                _sum00 = vfmlalq_low_f16(_sum00, _pB01, _pA01);
                _sum01 = vfmlalq_high_f16(_sum01, _pB01, _pA10);
                _sum10 = vfmlalq_low_f16(_sum10, _pB01, _pA10);
                _sum11 = vfmlalq_high_f16(_sum11, _pB01, _pA01);
#else
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pB));

                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[1]));
#if __aarch64__
                _sum00 = vfmaq_f32(_sum00, _pB0, _pA0);
                _sum01 = vfmaq_f32(_sum01, _pB1, _pA0);
                _sum10 = vfmaq_f32(_sum10, _pB0, _pA1);
                _sum11 = vfmaq_f32(_sum11, _pB1, _pA1);
#else
                _sum00 = vmlaq_f32(_sum00, _pB0, _pA0);
                _sum01 = vmlaq_f32(_sum01, _pB1, _pA0);
                _sum10 = vmlaq_f32(_sum10, _pB0, _pA1);
                _sum11 = vmlaq_f32(_sum11, _pB1, _pA1);
#endif
#endif

                pA += 2;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum00 = vmulq_f32(_sum00, _alpha);
                _sum01 = vmulq_f32(_sum01, _alpha);
                _sum10 = vmulq_f32(_sum10, _alpha);
                _sum11 = vmulq_f32(_sum11, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum00));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum01));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, (uint16x4_t)vcvt_f16_f32(_sum11));
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        _sum0 = _tmp01.val[0];
                        _sum1 = _tmp01.val[1];
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                float32x4_t _tmp0 = vld1q_f32(outptr);
                float32x4_t _tmp1 = vld1q_f32(outptr + 4);
                float32x4x2_t _tmp01 = vuzpq_f32(_tmp0, _tmp1);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pB = vld1_f16(pB);
                float16x8_t _pBB = vcombine_f16(_pB, _pB);

                float16x4_t _pA0 = vdup_n_f16(pA[0]);
                float16x4_t _pA1 = vdup_n_f16(pA[1]);
                float16x8_t _pA01 = vcombine_f16(_pA0, _pA1);

                _sum0 = vfmlalq_low_f16(_sum0, _pBB, _pA01);
                _sum1 = vfmlalq_high_f16(_sum1, _pBB, _pA01);
#else
                float32x4_t _pB = vcvt_f32_f16((float16x4_t)vld1_u16(pB));

                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
                float32x4_t _pA1 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[1]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pB, _pA0);
                _sum1 = vfmaq_f32(_sum1, _pB, _pA1);
#else
                _sum0 = vmlaq_f32(_sum0, _pB, _pA0);
                _sum1 = vmlaq_f32(_sum1, _pB, _pA1);
#endif
#endif

                pA += 2;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + out_hstep, (uint16x4_t)vcvt_f16_f32(_sum1));
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
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

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                __fp16 pA0 = pA[0];
                __fp16 pA1 = pA[1];
                __fp16 pB0 = pB[0];
                __fp16 pB1 = pB[1];
#else
                float pA0 = float16_to_float32(pA[0]);
                float pA1 = float16_to_float32(pA[1]);
                float pB0 = float16_to_float32(pB[0]);
                float pB1 = float16_to_float32(pB[1]);
#endif

                sum00 += pA0 * pB0;
                sum01 += pA1 * pB0;
                sum10 += pA0 * pB1;
                sum11 += pA1 * pB1;

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
                    outptr0[0] = float32_to_float16(sum00);
                    outptr0[1] = float32_to_float16(sum10);
                    outptr0[out_hstep] = float32_to_float16(sum01);
                    outptr0[out_hstep + 1] = float32_to_float16(sum11);
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

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                __fp16 pA0 = pA[0];
                __fp16 pA1 = pA[1];
                __fp16 pB0 = pB[0];
#else
                float pA0 = float16_to_float32(pA[0]);
                float pA1 = float16_to_float32(pA[1]);
                float pB0 = float16_to_float32(pB[0]);
#endif

                sum0 += pA0 * pB0;
                sum1 += pA1 * pB0;
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
                    outptr0[0] = float32_to_float16(sum0);
                    outptr0[out_hstep] = float32_to_float16(sum1);
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

#if __ARM_FEATURE_FP16_FML
        const __fp16* pB = pBT;
#else
        const unsigned short* pB = pBT;
#endif

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
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pB01 = vld1q_f16(pB);
                float16x4_t _pB2 = vld1_f16(pB + 8);
                float16x8_t _pB22 = vcombine_f16(_pB2, _pB2);

                float16x8_t _pA = vdupq_n_f16(pA[0]);

                _sum0 = vfmlalq_low_f16(_sum0, _pA, _pB01);
                _sum1 = vfmlalq_high_f16(_sum1, _pA, _pB01);
                _sum2 = vfmlalq_low_f16(_sum2, _pA, _pB22);
#else
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pB));
                float32x4_t _pB2 = vcvt_f32_f16((float16x4_t)vld1_u16(pB + 8));

                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vmlaq_f32(_sum2, _pA0, _pB2);
#endif
#endif

                pA += 1;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
                _sum2 = vmulq_f32(_sum2, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    vst1_u16(outptr0 + 8, (uint16x4_t)vcvt_f16_f32(_sum2));
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x8_t _pB01 = vld1q_f16(pB);
                float16x8_t _pA = vdupq_n_f16(pA[0]);

                _sum0 = vfmlalq_low_f16(_sum0, _pA, _pB01);
                _sum1 = vfmlalq_high_f16(_sum1, _pA, _pB01);
#else
                uint16x8_t _pB = vld1q_u16(pB);
                float32x4_t _pB0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_pB));
                float32x4_t _pB1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_pB));

                float32x4_t _pA0 = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
#endif
#endif

                pA += 1;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum0 = vmulq_f32(_sum0, _alpha);
                _sum1 = vmulq_f32(_sum1, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum0));
                    vst1_u16(outptr0 + 4, (uint16x4_t)vcvt_f16_f32(_sum1));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vld1q_f32(pC);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                float16x4_t _pB = vld1_f16(pB);
                float16x8_t _pBB = vcombine_f16(_pB, _pB);
                float16x8_t _pA = vdupq_n_f16(pA[0]);

                _sum = vfmlalq_low_f16(_sum, _pA, _pBB);
#else
                float32x4_t _pB = vcvt_f32_f16((float16x4_t)vld1_u16(pB));
                float32x4_t _pA = vcvt_f32_f16((float16x4_t)vdup_n_u16(pA[0]));
#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif
#endif

                pA += 1;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _sum = vmulq_f32(_sum, _alpha);
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, (uint16x4_t)vcvt_f16_f32(_sum));
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }

            outptr += 4;
        }
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

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                __fp16 pA0 = pA[0];
                __fp16 pB0 = pB[0];
                __fp16 pB1 = pB[1];
#else
                float pA0 = float16_to_float32(pA[0]);
                float pB0 = float16_to_float32(pB[0]);
                float pB1 = float16_to_float32(pB[1]);
#endif

                sum0 += pA0 * pB0;
                sum1 += pA0 * pB1;

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
                    outptr0[0] = float32_to_float16(sum0);
                    outptr0[1] = float32_to_float16(sum1);
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

#if __ARM_FEATURE_FP16_FML
            const __fp16* pA = pAT;
#else
            const unsigned short* pA = pAT;
#endif
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_FP16_FML
                __fp16 pA0 = pA[0];
                __fp16 pB0 = pB[0];
#else
                float pA0 = float16_to_float32(pA[0]);
                float pB0 = float16_to_float32(pB[0]);
#endif

                sum += pA0 * pB0;
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
                    outptr0[0] = float32_to_float16(sum);
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
