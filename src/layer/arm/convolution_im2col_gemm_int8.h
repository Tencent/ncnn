// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void convolution_im2col_gemm_transform_kernel_int8_i8mm(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt);
void convolution_im2col_gemm_int8_i8mm(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
void convolution_im2col_gemm_transform_kernel_int8_asimddp(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt);
void convolution_im2col_gemm_int8_asimddp(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt);
#endif
#endif

static void convolution_im2col_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    // A = (pa, maxk, inch/pa), outch
    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _r0 = vld1q_s8(p0);
            int8x16_t _r1 = vld1q_s8(p1);
            int8x16_t _r2 = vld1q_s8(p2);
            int8x16_t _r3 = vld1q_s8(p3);
            int8x16_t _r4 = vld1q_s8(p4);
            int8x16_t _r5 = vld1q_s8(p5);
            int8x16_t _r6 = vld1q_s8(p6);
            int8x16_t _r7 = vld1q_s8(p7);
            int8x16_t _t0 = vcombine_s8(vget_low_s8(_r0), vget_low_s8(_r1));
            int8x16_t _t1 = vcombine_s8(vget_low_s8(_r2), vget_low_s8(_r3));
            int8x16_t _t2 = vcombine_s8(vget_low_s8(_r4), vget_low_s8(_r5));
            int8x16_t _t3 = vcombine_s8(vget_low_s8(_r6), vget_low_s8(_r7));
            int8x16_t _t4 = vcombine_s8(vget_high_s8(_r0), vget_high_s8(_r1));
            int8x16_t _t5 = vcombine_s8(vget_high_s8(_r2), vget_high_s8(_r3));
            int8x16_t _t6 = vcombine_s8(vget_high_s8(_r4), vget_high_s8(_r5));
            int8x16_t _t7 = vcombine_s8(vget_high_s8(_r6), vget_high_s8(_r7));
            vst1q_s8(pp, _t0);
            vst1q_s8(pp + 16, _t1);
            vst1q_s8(pp + 32, _t2);
            vst1q_s8(pp + 48, _t3);
            vst1q_s8(pp + 64, _t4);
            vst1q_s8(pp + 80, _t5);
            vst1q_s8(pp + 96, _t6);
            vst1q_s8(pp + 112, _t7);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p1);
            int8x8_t _r2 = vld1_s8(p2);
            int8x8_t _r3 = vld1_s8(p3);
            int8x8_t _r4 = vld1_s8(p4);
            int8x8_t _r5 = vld1_s8(p5);
            int8x8_t _r6 = vld1_s8(p6);
            int8x8_t _r7 = vld1_s8(p7);
            vst1_s8(pp, _r0);
            vst1_s8(pp + 8, _r1);
            vst1_s8(pp + 16, _r2);
            vst1_s8(pp + 24, _r3);
            vst1_s8(pp + 32, _r4);
            vst1_s8(pp + 40, _r5);
            vst1_s8(pp + 48, _r6);
            vst1_s8(pp + 56, _r7);
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
#else  // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _r0 = vld1q_s8(p0);
            int8x16_t _r1 = vld1q_s8(p1);
            int8x16_t _r2 = vld1q_s8(p2);
            int8x16_t _r3 = vld1q_s8(p3);
            int8x16_t _r4 = vld1q_s8(p4);
            int8x16_t _r5 = vld1q_s8(p5);
            int8x16_t _r6 = vld1q_s8(p6);
            int8x16_t _r7 = vld1q_s8(p7);
            int32x4x2_t _r01 = vzipq_s32(vreinterpretq_s32_s8(_r0), vreinterpretq_s32_s8(_r1));
            int32x4x2_t _r23 = vzipq_s32(vreinterpretq_s32_s8(_r2), vreinterpretq_s32_s8(_r3));
            int32x4x2_t _r45 = vzipq_s32(vreinterpretq_s32_s8(_r4), vreinterpretq_s32_s8(_r5));
            int32x4x2_t _r67 = vzipq_s32(vreinterpretq_s32_s8(_r6), vreinterpretq_s32_s8(_r7));
            _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_r01.val[0]), vget_low_s32(_r23.val[0])));
            _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_r45.val[0]), vget_low_s32(_r67.val[0])));
            _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_r01.val[0]), vget_high_s32(_r23.val[0])));
            _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_r45.val[0]), vget_high_s32(_r67.val[0])));
            _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_r01.val[1]), vget_low_s32(_r23.val[1])));
            _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_r45.val[1]), vget_low_s32(_r67.val[1])));
            _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_r01.val[1]), vget_high_s32(_r23.val[1])));
            _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_r45.val[1]), vget_high_s32(_r67.val[1])));
            vst1q_s8(pp, _r0);
            vst1q_s8(pp + 16, _r1);
            vst1q_s8(pp + 32, _r2);
            vst1q_s8(pp + 48, _r3);
            vst1q_s8(pp + 64, _r4);
            vst1q_s8(pp + 80, _r5);
            vst1q_s8(pp + 96, _r6);
            vst1q_s8(pp + 112, _r7);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p1);
            int8x8_t _r2 = vld1_s8(p2);
            int8x8_t _r3 = vld1_s8(p3);
            int8x8_t _r4 = vld1_s8(p4);
            int8x8_t _r5 = vld1_s8(p5);
            int8x8_t _r6 = vld1_s8(p6);
            int8x8_t _r7 = vld1_s8(p7);
            int32x2x2_t _r01 = vzip_s32(vreinterpret_s32_s8(_r0), vreinterpret_s32_s8(_r1));
            int32x2x2_t _r23 = vzip_s32(vreinterpret_s32_s8(_r2), vreinterpret_s32_s8(_r3));
            int32x2x2_t _r45 = vzip_s32(vreinterpret_s32_s8(_r4), vreinterpret_s32_s8(_r5));
            int32x2x2_t _r67 = vzip_s32(vreinterpret_s32_s8(_r6), vreinterpret_s32_s8(_r7));
            int8x16_t _t0 = vreinterpretq_s8_s32(vcombine_s32(_r01.val[0], _r23.val[0]));
            int8x16_t _t1 = vreinterpretq_s8_s32(vcombine_s32(_r45.val[0], _r67.val[0]));
            int8x16_t _t2 = vreinterpretq_s8_s32(vcombine_s32(_r01.val[1], _r23.val[1]));
            int8x16_t _t3 = vreinterpretq_s8_s32(vcombine_s32(_r45.val[1], _r67.val[1]));
            vst1q_s8(pp, _t0);
            vst1q_s8(pp + 16, _t1);
            vst1q_s8(pp + 32, _t2);
            vst1q_s8(pp + 48, _t3);
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
#endif // __ARM_FEATURE_MATMUL_INT8
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
#else  // __ARM_FEATURE_DOTPROD
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _r0 = vld1q_s8(p0);
            int8x16_t _r1 = vld1q_s8(p1);
            int8x16_t _r2 = vld1q_s8(p2);
            int8x16_t _r3 = vld1q_s8(p3);
            int8x16_t _r4 = vld1q_s8(p4);
            int8x16_t _r5 = vld1q_s8(p5);
            int8x16_t _r6 = vld1q_s8(p6);
            int8x16_t _r7 = vld1q_s8(p7);
            int16x8x2_t _r01 = vzipq_s16(vreinterpretq_s16_s8(_r0), vreinterpretq_s16_s8(_r1));
            int16x8x2_t _r23 = vzipq_s16(vreinterpretq_s16_s8(_r2), vreinterpretq_s16_s8(_r3));
            int16x8x2_t _r45 = vzipq_s16(vreinterpretq_s16_s8(_r4), vreinterpretq_s16_s8(_r5));
            int16x8x2_t _r67 = vzipq_s16(vreinterpretq_s16_s8(_r6), vreinterpretq_s16_s8(_r7));
            int32x4x2_t _t0 = vzipq_s32(vreinterpretq_s32_s16(_r01.val[0]), vreinterpretq_s32_s16(_r23.val[0]));
            int32x4x2_t _t1 = vzipq_s32(vreinterpretq_s32_s16(_r01.val[1]), vreinterpretq_s32_s16(_r23.val[1]));
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_r45.val[0]), vreinterpretq_s32_s16(_r67.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_r45.val[1]), vreinterpretq_s32_s16(_r67.val[1]));
            _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t2.val[0])));
            _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t2.val[0])));
            _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t2.val[1])));
            _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t2.val[1])));
            _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t3.val[0])));
            _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t3.val[0])));
            _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t3.val[1])));
            _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t3.val[1])));
            vst1q_s8(pp, _r0);
            vst1q_s8(pp + 16, _r1);
            vst1q_s8(pp + 32, _r2);
            vst1q_s8(pp + 48, _r3);
            vst1q_s8(pp + 64, _r4);
            vst1q_s8(pp + 80, _r5);
            vst1q_s8(pp + 96, _r6);
            vst1q_s8(pp + 112, _r7);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p1);
            int8x8_t _r2 = vld1_s8(p2);
            int8x8_t _r3 = vld1_s8(p3);
            int8x8_t _r4 = vld1_s8(p4);
            int8x8_t _r5 = vld1_s8(p5);
            int8x8_t _r6 = vld1_s8(p6);
            int8x8_t _r7 = vld1_s8(p7);
            int16x8_t _r04 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r4));
            int16x8_t _r15 = vreinterpretq_s16_s8(vcombine_s8(_r1, _r5));
            int16x8_t _r26 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r6));
            int16x8_t _r37 = vreinterpretq_s16_s8(vcombine_s8(_r3, _r7));
            int16x8x2_t _t0 = vzipq_s16(_r04, _r15);
            int16x8x2_t _t1 = vzipq_s16(_r26, _r37);
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[0]), vreinterpretq_s32_s16(_t1.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[1]), vreinterpretq_s32_s16(_t1.val[1]));
            int8x16_t _t4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0])));
            int8x16_t _t5 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0])));
            int8x16_t _t6 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1])));
            int8x16_t _t7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1])));
            vst1q_s8(pp, _t4);
            vst1q_s8(pp + 16, _t5);
            vst1q_s8(pp + 32, _t6);
            vst1q_s8(pp + 48, _t7);
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
#endif // __ARM_FEATURE_DOTPROD
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int64x2x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s64_s8(vld1q_s8(p0));
            _r0123.val[1] = vreinterpretq_s64_s8(vld1q_s8(p1));
            _r0123.val[2] = vreinterpretq_s64_s8(vld1q_s8(p2));
            _r0123.val[3] = vreinterpretq_s64_s8(vld1q_s8(p3));
            vst4q_s64((int64_t*)pp, _r0123);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p1);
            int8x8_t _r2 = vld1_s8(p2);
            int8x8_t _r3 = vld1_s8(p3);
            vst1_s8(pp, _r0);
            vst1_s8(pp + 8, _r1);
            vst1_s8(pp + 16, _r2);
            vst1_s8(pp + 24, _r3);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
#else  // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int32x4x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s32_s8(vld1q_s8(p0));
            _r0123.val[1] = vreinterpretq_s32_s8(vld1q_s8(p1));
            _r0123.val[2] = vreinterpretq_s32_s8(vld1q_s8(p2));
            _r0123.val[3] = vreinterpretq_s32_s8(vld1q_s8(p3));
            vst4q_s32((int*)pp, _r0123);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int32x2x4_t _r0123;
            _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(p0));
            _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(p1));
            _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(p2));
            _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(p3));
            vst4_s32((int*)pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
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
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
#else  // __ARM_FEATURE_DOTPROD
        for (; kk + 15 < max_kk; kk += 16)
        {
            int16x8x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s16_s8(vld1q_s8(p0));
            _r0123.val[1] = vreinterpretq_s16_s8(vld1q_s8(p1));
            _r0123.val[2] = vreinterpretq_s16_s8(vld1q_s8(p2));
            _r0123.val[3] = vreinterpretq_s16_s8(vld1q_s8(p3));
            vst4q_s16((short*)pp, _r0123);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int16x4x4_t _r0123;
            _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
            _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(p1));
            _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(p2));
            _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(p3));
            vst4_s16((short*)pp, _r0123);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
#endif // __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int64x2x2_t _r01;
            _r01.val[0] = vreinterpretq_s64_s8(vld1q_s8(p0));
            _r01.val[1] = vreinterpretq_s64_s8(vld1q_s8(p1));
            vst2q_s64((int64_t*)pp, _r01);
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p1);
            vst1_s8(pp, _r0);
            vst1_s8(pp + 8, _r1);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
#else  // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 15 < max_kk; kk += 16)
        {
            int32x4x2_t _r01;
            _r01.val[0] = vreinterpretq_s32_s8(vld1q_s8(p0));
            _r01.val[1] = vreinterpretq_s32_s8(vld1q_s8(p1));
            vst2q_s32((int*)pp, _r01);
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int32x2x2_t _r01;
            _r01.val[0] = vreinterpret_s32_s8(vld1_s8(p0));
            _r01.val[1] = vreinterpret_s32_s8(vld1_s8(p1));
            vst2_s32((int*)pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
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
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
#else  // __ARM_FEATURE_DOTPROD
        for (; kk + 15 < max_kk; kk += 16)
        {
            int16x8x2_t _r01;
            _r01.val[0] = vreinterpretq_s16_s8(vld1q_s8(p0));
            _r01.val[1] = vreinterpretq_s16_s8(vld1q_s8(p1));
            vst2q_s16((short*)pp, _r01);
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int16x4x2_t _r01;
            _r01.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
            _r01.val[1] = vreinterpret_s16_s8(vld1_s8(p1));
            vst2_s16((short*)pp, _r01);
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
#endif // __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
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
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int kk = 0;
#if __ARM_NEON
        for (; kk + 15 < max_kk; kk += 16)
        {
            vst1q_s8(pp, vld1q_s8(p0));
            pp += 16;
            p0 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            vst1_s8(pp, vld1_s8(p0));
            pp += 8;
            p0 += 8;
        }
#endif // __ARM_NEON
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void convolution_gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const size_t out_hstep = top_blob.cstep;

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
#if !__ARM_FEATURE_MATMUL_INT8
                "cmp    %w9, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "sub    %0, %0, #192                \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "1:                                 \n"
#endif // !__ARM_FEATURE_MATMUL_INT8

#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v0.16b, v0.16b, v0.16b      \n"
                "eor    v1.16b, v1.16b, v1.16b      \n"
                "eor    v2.16b, v2.16b, v2.16b      \n"
                "eor    v3.16b, v3.16b, v3.16b      \n"
                "eor    v4.16b, v4.16b, v4.16b      \n"
                "eor    v5.16b, v5.16b, v5.16b      \n"
                "eor    v6.16b, v6.16b, v6.16b      \n"
                "eor    v7.16b, v7.16b, v7.16b      \n"
                "eor    v8.16b, v8.16b, v8.16b      \n"
                "eor    v9.16b, v9.16b, v9.16b      \n"
                "eor    v10.16b, v10.16b, v10.16b   \n"
                "eor    v11.16b, v11.16b, v11.16b   \n"
                "eor    v12.16b, v12.16b, v12.16b   \n"
                "eor    v13.16b, v13.16b, v13.16b   \n"
                "eor    v14.16b, v14.16b, v14.16b   \n"
                "eor    v15.16b, v15.16b, v15.16b   \n"

                "2:                                 \n"
                "ld1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%1], #64 \n"
                "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%2], #64 \n"
                "smmla  v0.4s, v16.16b, v20.16b     \n"
                "smmla  v1.4s, v17.16b, v20.16b     \n"
                "smmla  v2.4s, v16.16b, v21.16b     \n"
                "smmla  v3.4s, v17.16b, v21.16b     \n"
                "smmla  v4.4s, v18.16b, v20.16b     \n"
                "smmla  v5.4s, v19.16b, v20.16b     \n"
                "smmla  v6.4s, v18.16b, v21.16b     \n"
                "smmla  v7.4s, v19.16b, v21.16b     \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v8.4s, v16.16b, v22.16b     \n"
                "smmla  v9.4s, v17.16b, v22.16b     \n"
                "smmla  v10.4s, v16.16b, v23.16b    \n"
                "smmla  v11.4s, v17.16b, v23.16b    \n"
                "smmla  v12.4s, v18.16b, v22.16b    \n"
                "smmla  v13.4s, v19.16b, v22.16b    \n"
                "smmla  v14.4s, v18.16b, v23.16b    \n"
                "smmla  v15.4s, v19.16b, v23.16b    \n"
                "bne    2b                          \n"

                "uzp1   v16.4s, v0.4s, v1.4s        \n"
                "uzp2   v17.4s, v0.4s, v1.4s        \n"
                "uzp1   v18.4s, v2.4s, v3.4s        \n"
                "uzp2   v19.4s, v2.4s, v3.4s        \n"
                "uzp1   v20.4s, v4.4s, v5.4s        \n"
                "uzp2   v21.4s, v4.4s, v5.4s        \n"
                "uzp1   v22.4s, v6.4s, v7.4s        \n"
                "uzp2   v23.4s, v6.4s, v7.4s        \n"
                "uzp1   v24.4s, v8.4s, v9.4s        \n"
                "uzp2   v25.4s, v8.4s, v9.4s        \n"
                "uzp1   v26.4s, v10.4s, v11.4s      \n"
                "uzp2   v27.4s, v10.4s, v11.4s      \n"
                "uzp1   v28.4s, v12.4s, v13.4s      \n"
                "uzp2   v29.4s, v12.4s, v13.4s      \n"
                "uzp1   v30.4s, v14.4s, v15.4s      \n"
                "uzp2   v31.4s, v14.4s, v15.4s      \n"

                "cmp    %w9, #0                     \n"
                "beq    1f                          \n"

                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64   \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64 \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0]    \n"
                "sub    %0, %0, #192                \n"
                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
                "add    v24.4s, v24.4s, v8.4s       \n"
                "add    v25.4s, v25.4s, v9.4s       \n"
                "add    v26.4s, v26.4s, v10.4s      \n"
                "add    v27.4s, v27.4s, v11.4s      \n"
                "add    v28.4s, v28.4s, v12.4s      \n"
                "add    v29.4s, v29.4s, v13.4s      \n"
                "add    v30.4s, v30.4s, v14.4s      \n"
                "add    v31.4s, v31.4s, v15.4s      \n"
                "b      1f                          \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "2:                                 \n"
                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%1], #64 \n"
                "ld1    {v4.16b, v5.16b, v6.16b, v7.16b}, [%2], #64 \n"
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v4.4b[3]    \n"
                "sdot   v24.4s, v0.16b, v5.4b[0]    \n"
                "sdot   v25.4s, v0.16b, v5.4b[1]    \n"
                "sdot   v26.4s, v0.16b, v5.4b[2]    \n"
                "sdot   v27.4s, v0.16b, v5.4b[3]    \n"
                "sdot   v28.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v29.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v30.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v31.4s, v1.16b, v5.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v2.16b, v6.4b[0]    \n"
                "sdot   v17.4s, v2.16b, v6.4b[1]    \n"
                "sdot   v18.4s, v2.16b, v6.4b[2]    \n"
                "sdot   v19.4s, v2.16b, v6.4b[3]    \n"
                "sdot   v20.4s, v3.16b, v6.4b[0]    \n"
                "sdot   v21.4s, v3.16b, v6.4b[1]    \n"
                "sdot   v22.4s, v3.16b, v6.4b[2]    \n"
                "sdot   v23.4s, v3.16b, v6.4b[3]    \n"
                "sdot   v24.4s, v2.16b, v7.4b[0]    \n"
                "sdot   v25.4s, v2.16b, v7.4b[1]    \n"
                "sdot   v26.4s, v2.16b, v7.4b[2]    \n"
                "sdot   v27.4s, v2.16b, v7.4b[3]    \n"
                "sdot   v28.4s, v3.16b, v7.4b[0]    \n"
                "sdot   v29.4s, v3.16b, v7.4b[1]    \n"
                "sdot   v30.4s, v3.16b, v7.4b[2]    \n"
                "sdot   v31.4s, v3.16b, v7.4b[3]    \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
#if __ARM_FEATURE_MATMUL_INT8
                "cmp    %w9, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "sub    %0, %0, #192                \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
                "1:                                 \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "and    w4, %w8, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b, v3.16b}, [%2], #32 \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v2.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v2.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v2.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v2.4b[3]    \n"
                "sdot   v24.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v25.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v26.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v27.4s, v0.16b, v3.4b[3]    \n"
                "sdot   v28.4s, v1.16b, v3.4b[0]    \n"
                "sdot   v29.4s, v1.16b, v3.4b[1]    \n"
                "sdot   v30.4s, v1.16b, v3.4b[2]    \n"
                "sdot   v31.4s, v1.16b, v3.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v4.16b       \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v4.16b      \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull2 v13.8h, v0.16b, v6.16b      \n"
                "rev64  v3.4s, v1.4s                \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "smull2 v15.8h, v2.16b, v6.16b      \n"
                "rev64  v7.8h, v5.8h                \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v5.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v5.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v7.16b      \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v15.8h, v3.16b, v7.16b      \n"
                "ext    v0.16b, v0.16b, v0.16b, #8  \n"
                "ext    v2.16b, v2.16b, v2.16b, #8  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v20.4s, v10.8h              \n"
                "sadalp v21.4s, v11.8h              \n"
                "ext    v1.16b, v1.16b, v1.16b, #8  \n"
                "ext    v3.16b, v3.16b, v3.16b, #8  \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v4.16b       \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v4.16b      \n"
                "sadalp v24.4s, v12.8h              \n"
                "sadalp v25.4s, v13.8h              \n"
                "sadalp v28.4s, v14.8h              \n"
                "sadalp v29.4s, v15.8h              \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull2 v13.8h, v0.16b, v6.16b      \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "smull2 v15.8h, v2.16b, v6.16b      \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v5.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v5.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v7.16b      \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v15.8h, v3.16b, v7.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v18.4s, v8.8h               \n"
                "sadalp v19.4s, v9.8h               \n"
                "sadalp v22.4s, v10.8h              \n"
                "sadalp v23.4s, v11.8h              \n"
                "sadalp v26.4s, v12.8h              \n"
                "sadalp v27.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w8, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v1.16b}, [%2], #16         \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "dup    v4.8h, v1.h[4]              \n"
                "dup    v5.8h, v1.h[5]              \n"
                "dup    v6.8h, v1.h[6]              \n"
                "dup    v7.8h, v1.h[7]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v24.4s, v8.8h               \n"
                "sadalp v25.4s, v9.8h               \n"
                "sadalp v26.4s, v10.8h              \n"
                "sadalp v27.4s, v11.8h              \n"
                "sadalp v28.4s, v12.8h              \n"
                "sadalp v29.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v20.4s, v10.8h              \n"
                "sadalp v21.4s, v11.8h              \n"
                "sadalp v24.4s, v12.8h              \n"
                "sadalp v25.4s, v13.8h              \n"
                "sadalp v28.4s, v14.8h              \n"
                "sadalp v29.4s, v15.8h              \n"
                "ext    v0.16b, v0.16b, v0.16b, #8  \n"
                "ext    v1.16b, v1.16b, v1.16b, #8  \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v18.4s, v8.8h               \n"
                "sadalp v19.4s, v9.8h               \n"
                "sadalp v22.4s, v10.8h              \n"
                "sadalp v23.4s, v11.8h              \n"
                "sadalp v26.4s, v12.8h              \n"
                "sadalp v27.4s, v13.8h              \n"
                "sadalp v30.4s, v14.8h              \n"
                "sadalp v31.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w8, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v8.8b, v1.b[0]              \n"
                "dup    v9.8b, v1.b[1]              \n"
                "dup    v10.8b, v1.b[2]             \n"
                "dup    v11.8b, v1.b[3]             \n"
                "dup    v12.8b, v1.b[4]             \n"
                "dup    v13.8b, v1.b[5]             \n"
                "dup    v14.8b, v1.b[6]             \n"
                "dup    v15.8b, v1.b[7]             \n"
                "smull  v8.8h, v0.8b, v8.8b         \n"
                "smull  v9.8h, v0.8b, v9.8b         \n"
                "smull  v10.8h, v0.8b, v10.8b       \n"
                "smull  v11.8h, v0.8b, v11.8b       \n"
                "smull  v12.8h, v0.8b, v12.8b       \n"
                "smull  v13.8h, v0.8b, v13.8b       \n"
                "smull  v14.8h, v0.8b, v14.8b       \n"
                "smull  v15.8h, v0.8b, v15.8b       \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw  v18.4s, v18.4s, v10.4h      \n"
                "saddw  v19.4s, v19.4s, v11.4h      \n"
                "saddw2 v20.4s, v20.4s, v8.8h       \n"
                "saddw2 v21.4s, v21.4s, v9.8h       \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
                "saddw  v24.4s, v24.4s, v12.4h      \n"
                "saddw  v25.4s, v25.4s, v13.4h      \n"
                "saddw  v26.4s, v26.4s, v14.4h      \n"
                "saddw  v27.4s, v27.4s, v15.4h      \n"
                "saddw2 v28.4s, v28.4s, v12.8h      \n"
                "saddw2 v29.4s, v29.4s, v13.8h      \n"
                "saddw2 v30.4s, v30.4s, v14.8h      \n"
                "saddw2 v31.4s, v31.4s, v15.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v4.8b}, [%2], #8           \n"
                "ext    v1.8b, v0.8b, v0.8b, #4     \n"
                "rev32  v2.4h, v0.4h                \n"
                "rev64  v3.4h, v0.4h                \n"
                "rev32  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull  v11.8h, v3.8b, v4.8b        \n"
                "smull  v12.8h, v0.8b, v5.8b        \n"
                "smull  v13.8h, v1.8b, v5.8b        \n"
                "smull  v14.8h, v2.8b, v5.8b        \n"
                "smull  v15.8h, v3.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
                "saddw  v24.4s, v24.4s, v12.4h      \n"
                "saddw2 v25.4s, v25.4s, v12.8h      \n"
                "saddw  v26.4s, v26.4s, v13.4h      \n"
                "saddw2 v27.4s, v27.4s, v13.8h      \n"
                "saddw  v28.4s, v28.4s, v14.4h      \n"
                "saddw2 v29.4s, v29.4s, v14.8h      \n"
                "saddw  v30.4s, v30.4s, v15.4h      \n"
                "saddw2 v31.4s, v31.4s, v15.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "cmp    %w10, #0                    \n"
                "beq    10f                         \n"

#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                // if out_elempack == 4 / 8
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                // if out_elempack == 8
                "cmp    %w11, #8                    \n"
                "bne    7f                          \n"

                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v20.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [%3], #16         \n"
                "st1    {v21.4s}, [%3], #16         \n"
                "st1    {v18.4s}, [%3], #16         \n"
                "st1    {v22.4s}, [%3], #16         \n"
                "st1    {v19.4s}, [%3], #16         \n"
                "st1    {v23.4s}, [%3], #16         \n"
                "st1    {v24.4s}, [%3], #16         \n"
                "st1    {v28.4s}, [%3], #16         \n"
                "st1    {v25.4s}, [%3], #16         \n"
                "st1    {v29.4s}, [%3], #16         \n"
                "st1    {v26.4s}, [%3], #16         \n"
                "st1    {v30.4s}, [%3], #16         \n"
                "st1    {v27.4s}, [%3], #16         \n"
                "st1    {v31.4s}, [%3], #16         \n"
                "b      9f                          \n"

                // if out_elempack == 4
                "7:                                 \n"
                "add    x4, %3, %12, lsl #4         \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [x4], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                "zip1   v0.4s, v16.4s, v17.4s       \n"
                "zip2   v1.4s, v16.4s, v17.4s       \n"
                "zip1   v2.4s, v18.4s, v19.4s       \n"
                "zip2   v3.4s, v18.4s, v19.4s       \n"
                "zip1   v4.4s, v24.4s, v25.4s       \n"
                "zip2   v5.4s, v24.4s, v25.4s       \n"
                "zip1   v6.4s, v26.4s, v27.4s       \n"
                "zip2   v7.4s, v26.4s, v27.4s       \n"
                "zip1   v8.4s, v20.4s, v21.4s       \n"
                "zip2   v9.4s, v20.4s, v21.4s       \n"
                "zip1   v10.4s, v22.4s, v23.4s      \n"
                "zip2   v11.4s, v22.4s, v23.4s      \n"
                "zip1   v12.4s, v28.4s, v29.4s      \n"
                "zip2   v13.4s, v28.4s, v29.4s      \n"
                "zip1   v14.4s, v30.4s, v31.4s      \n"
                "zip2   v15.4s, v30.4s, v31.4s      \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v1.2d, v3.2d        \n"
                "zip1   v21.2d, v5.2d, v7.2d        \n"
                "zip2   v22.2d, v1.2d, v3.2d        \n"
                "zip2   v23.2d, v5.2d, v7.2d        \n"
                "zip1   v24.2d, v8.2d, v10.2d       \n"
                "zip1   v25.2d, v12.2d, v14.2d      \n"
                "zip2   v26.2d, v8.2d, v10.2d       \n"
                "zip2   v27.2d, v12.2d, v14.2d      \n"
                "zip1   v28.2d, v9.2d, v11.2d       \n"
                "zip1   v29.2d, v13.2d, v15.2d      \n"
                "zip2   v30.2d, v9.2d, v11.2d       \n"
                "zip2   v31.2d, v13.2d, v15.2d      \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s, v17.4s}, [%3], #32 \n"
                "st1    {v18.4s, v19.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s, v21.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s, v23.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v24.4s, v25.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v26.4s, v27.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v28.4s, v29.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v30.4s, v31.4s}, [x4]      \n"
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4
                // if out_elempack == 4 / 8
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                "rev64  v24.4s, v24.4s              \n"
                "rev64  v25.4s, v25.4s              \n"
                "rev64  v26.4s, v26.4s              \n"
                "rev64  v27.4s, v27.4s              \n"
                "rev64  v28.4s, v28.4s              \n"
                "rev64  v29.4s, v29.4s              \n"
                "rev64  v30.4s, v30.4s              \n"
                "rev64  v31.4s, v31.4s              \n"
                "ext    v24.16b, v24.16b, v24.16b, #8 \n"
                "ext    v25.16b, v25.16b, v25.16b, #8 \n"
                "ext    v26.16b, v26.16b, v26.16b, #8 \n"
                "ext    v27.16b, v27.16b, v27.16b, #8 \n"
                "ext    v28.16b, v28.16b, v28.16b, #8 \n"
                "ext    v29.16b, v29.16b, v29.16b, #8 \n"
                "ext    v30.16b, v30.16b, v30.16b, #8 \n"
                "ext    v31.16b, v31.16b, v31.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v28.4s       \n"
                "zip2   v1.4s, v16.4s, v28.4s       \n"
                "zip1   v2.4s, v20.4s, v24.4s       \n"
                "zip2   v3.4s, v20.4s, v24.4s       \n"
                "zip1   v4.4s, v18.4s, v30.4s       \n"
                "zip2   v5.4s, v18.4s, v30.4s       \n"
                "zip1   v6.4s, v22.4s, v26.4s       \n"
                "zip2   v7.4s, v22.4s, v26.4s       \n"
                "zip1   v8.4s, v19.4s, v31.4s       \n"
                "zip2   v9.4s, v19.4s, v31.4s       \n"
                "zip1   v10.4s, v23.4s, v27.4s      \n"
                "zip2   v11.4s, v23.4s, v27.4s      \n"
                "zip1   v12.4s, v17.4s, v29.4s      \n"
                "zip2   v13.4s, v17.4s, v29.4s      \n"
                "zip1   v14.4s, v21.4s, v25.4s      \n"
                "zip2   v15.4s, v21.4s, v25.4s      \n"

                // if out_elempack == 8
                "cmp    %w11, #8                    \n"
                "bne    7f                          \n"

                // to
                //      a0 b0 c0 d0
                //      e0 f0 g0 h0
                //      a1 b1 c1 d1
                //      e1 f1 g1 h1
                //      a2 b2 c2 d2
                //      e2 f2 g2 h2
                //      a3 b3 c3 d3
                //      e3 f3 g3 h3
                //      a4 b4 c4 d4
                //      e4 f4 g4 h4
                //      a5 b5 c5 d5
                //      e5 f5 g5 h5
                //      a6 b6 c6 d6
                //      e6 f6 g6 h6
                //      a7 b7 c7 d7
                //      e7 f7 g7 h7
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v3.2d, v1.2d        \n"
                "zip1   v21.2d, v7.2d, v5.2d        \n"
                "zip2   v22.2d, v3.2d, v1.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "zip1   v24.2d, v8.2d, v10.2d       \n"
                "zip1   v25.2d, v12.2d, v14.2d      \n"
                "zip2   v26.2d, v8.2d, v10.2d       \n"
                "zip2   v27.2d, v12.2d, v14.2d      \n"
                "zip1   v28.2d, v11.2d, v9.2d       \n"
                "zip1   v29.2d, v15.2d, v13.2d      \n"
                "zip2   v30.2d, v11.2d, v9.2d       \n"
                "zip2   v31.2d, v15.2d, v13.2d      \n"
                "rev64  v18.4s, v18.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"
                "rev64  v26.4s, v26.4s              \n"
                "rev64  v27.4s, v27.4s              \n"
                "rev64  v30.4s, v30.4s              \n"
                "rev64  v31.4s, v31.4s              \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%3], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 4
                "7:                                 \n"
                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v24.2d, v4.2d, v6.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip2   v25.2d, v4.2d, v6.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip1   v26.2d, v7.2d, v5.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "zip2   v27.2d, v7.2d, v5.2d        \n"
                "zip1   v20.2d, v8.2d, v10.2d       \n"
                "zip1   v28.2d, v12.2d, v14.2d      \n"
                "zip2   v21.2d, v8.2d, v10.2d       \n"
                "zip2   v29.2d, v12.2d, v14.2d      \n"
                "zip1   v22.2d, v11.2d, v9.2d       \n"
                "zip1   v30.2d, v15.2d, v13.2d      \n"
                "zip2   v23.2d, v11.2d, v9.2d       \n"
                "zip2   v31.2d, v15.2d, v13.2d      \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v21.4s, v21.4s              \n"
                "rev64  v23.4s, v23.4s              \n"
                "rev64  v25.4s, v25.4s              \n"
                "rev64  v27.4s, v27.4s              \n"
                "rev64  v29.4s, v29.4s              \n"
                "rev64  v31.4s, v31.4s              \n"

                "add    x4, %3, %12, lsl #4         \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [x4], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                //      e0 e1 e2 e3
                //      e4 e5 e6 e7
                //      f0 f1 f2 f3
                //      f4 f5 f6 f7
                //      g0 g1 g2 g3
                //      g4 g5 g6 g7
                //      h0 h1 h2 h3
                //      h4 h5 h6 h7
                "ext    v20.16b, v20.16b, v20.16b, #8 \n"
                "ext    v21.16b, v21.16b, v21.16b, #8 \n"
                "ext    v22.16b, v22.16b, v22.16b, #8 \n"
                "ext    v23.16b, v23.16b, v23.16b, #8 \n"
                "ext    v28.16b, v28.16b, v28.16b, #8 \n"
                "ext    v29.16b, v29.16b, v29.16b, #8 \n"
                "ext    v30.16b, v30.16b, v30.16b, #8 \n"
                "ext    v31.16b, v31.16b, v31.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v28.4s       \n"
                "zip2   v1.4s, v16.4s, v28.4s       \n"
                "zip1   v2.4s, v20.4s, v24.4s       \n"
                "zip2   v3.4s, v20.4s, v24.4s       \n"
                "zip1   v4.4s, v19.4s, v31.4s       \n"
                "zip2   v5.4s, v19.4s, v31.4s       \n"
                "zip1   v6.4s, v23.4s, v27.4s       \n"
                "zip2   v7.4s, v23.4s, v27.4s       \n"
                "zip1   v8.4s, v18.4s, v30.4s       \n"
                "zip2   v9.4s, v18.4s, v30.4s       \n"
                "zip1   v10.4s, v22.4s, v26.4s      \n"
                "zip2   v11.4s, v22.4s, v26.4s      \n"
                "zip1   v12.4s, v17.4s, v29.4s      \n"
                "zip2   v13.4s, v17.4s, v29.4s      \n"
                "zip1   v14.4s, v21.4s, v25.4s      \n"
                "zip2   v15.4s, v21.4s, v25.4s      \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v3.2d, v1.2d        \n"
                "zip1   v21.2d, v7.2d, v5.2d        \n"
                "zip2   v22.2d, v3.2d, v1.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "zip1   v24.2d, v8.2d, v10.2d       \n"
                "zip1   v25.2d, v12.2d, v14.2d      \n"
                "zip2   v26.2d, v8.2d, v10.2d       \n"
                "zip2   v27.2d, v12.2d, v14.2d      \n"
                "zip1   v28.2d, v11.2d, v9.2d       \n"
                "zip1   v29.2d, v15.2d, v13.2d      \n"
                "zip2   v30.2d, v11.2d, v9.2d       \n"
                "zip2   v31.2d, v15.2d, v13.2d      \n"
                "rev64  v18.4s, v18.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"
                "rev64  v26.4s, v26.4s              \n"
                "rev64  v27.4s, v27.4s              \n"
                "rev64  v30.4s, v30.4s              \n"
                "rev64  v31.4s, v31.4s              \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s, v17.4s}, [%3], #32 \n"
                "st1    {v18.4s, v19.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s, v21.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s, v23.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v24.4s, v25.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v26.4s, v27.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v28.4s, v29.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v30.4s, v31.4s}, [x4]      \n"
#endif // __ARM_FEATURE_DOTPROD

                "9:                                 \n"
                "add    %0, %0, #256                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;
            int32x4_t _sum8;
            int32x4_t _sum9;
            int32x4_t _suma;
            int32x4_t _sumb;
            int32x4_t _sumc;
            int32x4_t _sumd;
            int32x4_t _sume;
            int32x4_t _sumf;

#if __ARM_FEATURE_MATMUL_INT8
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
                _sum8 = vdupq_n_s32(0);
                _sum9 = vdupq_n_s32(0);
                _suma = vdupq_n_s32(0);
                _sumb = vdupq_n_s32(0);
                _sumc = vdupq_n_s32(0);
                _sumd = vdupq_n_s32(0);
                _sume = vdupq_n_s32(0);
                _sumf = vdupq_n_s32(0);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
                _sum8 = vdupq_n_s32(0);
                _sum9 = vdupq_n_s32(0);
                _suma = vdupq_n_s32(0);
                _sumb = vdupq_n_s32(0);
                _sumc = vdupq_n_s32(0);
                _sumd = vdupq_n_s32(0);
                _sume = vdupq_n_s32(0);
                _sumf = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
                _sum8 = vld1q_s32(outptr + 32);
                _sum9 = vld1q_s32(outptr + 36);
                _suma = vld1q_s32(outptr + 40);
                _sumb = vld1q_s32(outptr + 44);
                _sumc = vld1q_s32(outptr + 48);
                _sumd = vld1q_s32(outptr + 52);
                _sume = vld1q_s32(outptr + 56);
                _sumf = vld1q_s32(outptr + 60);
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    _sum0 = vmmlaq_s32(_sum0, _pA0, _pB0);
                    _sum1 = vmmlaq_s32(_sum1, _pA1, _pB0);
                    _sum2 = vmmlaq_s32(_sum2, _pA0, _pB1);
                    _sum3 = vmmlaq_s32(_sum3, _pA1, _pB1);
                    _sum4 = vmmlaq_s32(_sum4, _pA2, _pB0);
                    _sum5 = vmmlaq_s32(_sum5, _pA3, _pB0);
                    _sum6 = vmmlaq_s32(_sum6, _pA2, _pB1);
                    _sum7 = vmmlaq_s32(_sum7, _pA3, _pB1);
                    _sum8 = vmmlaq_s32(_sum8, _pA0, _pB2);
                    _sum9 = vmmlaq_s32(_sum9, _pA1, _pB2);
                    _suma = vmmlaq_s32(_suma, _pA0, _pB3);
                    _sumb = vmmlaq_s32(_sumb, _pA1, _pB3);
                    _sumc = vmmlaq_s32(_sumc, _pA2, _pB2);
                    _sumd = vmmlaq_s32(_sumd, _pA3, _pB2);
                    _sume = vmmlaq_s32(_sume, _pA2, _pB3);
                    _sumf = vmmlaq_s32(_sumf, _pA3, _pB3);

                    pA += 64;
                    pB += 64;
                }

                int32x4x2_t _ss0 = vuzpq_s32(_sum0, _sum1);
                int32x4x2_t _ss1 = vuzpq_s32(_sum2, _sum3);
                int32x4x2_t _ss2 = vuzpq_s32(_sum4, _sum5);
                int32x4x2_t _ss3 = vuzpq_s32(_sum6, _sum7);
                int32x4x2_t _ss4 = vuzpq_s32(_sum8, _sum9);
                int32x4x2_t _ss5 = vuzpq_s32(_suma, _sumb);
                int32x4x2_t _ss6 = vuzpq_s32(_sumc, _sumd);
                int32x4x2_t _ss7 = vuzpq_s32(_sume, _sumf);

                if (k == 0)
                {
                    _sum0 = _ss0.val[0];
                    _sum1 = _ss0.val[1];
                    _sum2 = _ss1.val[0];
                    _sum3 = _ss1.val[1];
                    _sum4 = _ss2.val[0];
                    _sum5 = _ss2.val[1];
                    _sum6 = _ss3.val[0];
                    _sum7 = _ss3.val[1];
                    _sum8 = _ss4.val[0];
                    _sum9 = _ss4.val[1];
                    _suma = _ss5.val[0];
                    _sumb = _ss5.val[1];
                    _sumc = _ss6.val[0];
                    _sumd = _ss6.val[1];
                    _sume = _ss7.val[0];
                    _sumf = _ss7.val[1];
                }
                else
                {
                    _sum0 = vld1q_s32(outptr);
                    _sum1 = vld1q_s32(outptr + 4);
                    _sum2 = vld1q_s32(outptr + 8);
                    _sum3 = vld1q_s32(outptr + 12);
                    _sum4 = vld1q_s32(outptr + 16);
                    _sum5 = vld1q_s32(outptr + 20);
                    _sum6 = vld1q_s32(outptr + 24);
                    _sum7 = vld1q_s32(outptr + 28);
                    _sum8 = vld1q_s32(outptr + 32);
                    _sum9 = vld1q_s32(outptr + 36);
                    _suma = vld1q_s32(outptr + 40);
                    _sumb = vld1q_s32(outptr + 44);
                    _sumc = vld1q_s32(outptr + 48);
                    _sumd = vld1q_s32(outptr + 52);
                    _sume = vld1q_s32(outptr + 56);
                    _sumf = vld1q_s32(outptr + 60);

                    _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                    _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                    _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                    _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                    _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                    _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                    _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                    _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
                    _sum8 = vaddq_s32(_sum8, _ss4.val[0]);
                    _sum9 = vaddq_s32(_sum9, _ss4.val[1]);
                    _suma = vaddq_s32(_suma, _ss5.val[0]);
                    _sumb = vaddq_s32(_sumb, _ss5.val[1]);
                    _sumc = vaddq_s32(_sumc, _ss6.val[0]);
                    _sumd = vaddq_s32(_sumd, _ss6.val[1]);
                    _sume = vaddq_s32(_sume, _ss7.val[0]);
                    _sumf = vaddq_s32(_sumf, _ss7.val[1]);
                }
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pA2 = vld1q_s8(pA + 32);
                int8x16_t _pA3 = vld1q_s8(pA + 48);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);
                int8x16_t _pB2 = vld1q_s8(pB + 32);
                int8x16_t _pB3 = vld1q_s8(pB + 48);

                // aaaa bbbb cccc dddd    eeee ffff gggg hhhh

                // 0000 1111 2222 3333    4444 5555 6666 7777
                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA0, _pB1, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA0, _pB1, 1);
                _suma = vdotq_laneq_s32(_suma, _pA0, _pB1, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA0, _pB1, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA1, _pB1, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA1, _pB1, 1);
                _sume = vdotq_laneq_s32(_sume, _pA1, _pB1, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA1, _pB1, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB2, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB2, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA2, _pB2, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA2, _pB2, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA3, _pB2, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA3, _pB2, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA3, _pB2, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA3, _pB2, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA2, _pB3, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA2, _pB3, 1);
                _suma = vdotq_laneq_s32(_suma, _pA2, _pB3, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA2, _pB3, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA3, _pB3, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA3, _pB3, 1);
                _sume = vdotq_laneq_s32(_sume, _pA3, _pB3, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA3, _pB3, 3);

                pA += 64;
                pB += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                // aaaa bbbb cccc dddd    eeee ffff gggg hhhh

                // 0000 1111 2222 3333    4444 5555 6666 7777
                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);
                _sum8 = vdotq_laneq_s32(_sum8, _pA0, _pB1, 0);
                _sum9 = vdotq_laneq_s32(_sum9, _pA0, _pB1, 1);
                _suma = vdotq_laneq_s32(_suma, _pA0, _pB1, 2);
                _sumb = vdotq_laneq_s32(_sumb, _pA0, _pB1, 3);
                _sumc = vdotq_laneq_s32(_sumc, _pA1, _pB1, 0);
                _sumd = vdotq_laneq_s32(_sumd, _pA1, _pB1, 1);
                _sume = vdotq_laneq_s32(_sume, _pA1, _pB1, 2);
                _sumf = vdotq_laneq_s32(_sumf, _pA1, _pB1, 3);

#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB2 = vld1q_s8(pB + 16);

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB3 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _s9 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sa = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sc = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB1));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sf = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB1));

                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_low_s8(_pB2));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB2));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA2), vget_low_s8(_pB2));
                _s3 = vmlal_s8(_s3, vget_low_s8(_pA2), vget_high_s8(_pB2));
                _s4 = vmlal_s8(_s4, vget_low_s8(_pA3), vget_low_s8(_pB2));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA3), vget_high_s8(_pB2));
                _s6 = vmlal_s8(_s6, vget_high_s8(_pA3), vget_low_s8(_pB2));
                _s7 = vmlal_s8(_s7, vget_low_s8(_pA3), vget_high_s8(_pB2));
                _s8 = vmlal_s8(_s8, vget_low_s8(_pA2), vget_low_s8(_pB3));
                _s9 = vmlal_s8(_s9, vget_high_s8(_pA2), vget_high_s8(_pB3));
                _sa = vmlal_s8(_sa, vget_high_s8(_pA2), vget_low_s8(_pB3));
                _sb = vmlal_s8(_sb, vget_low_s8(_pA2), vget_high_s8(_pB3));
                _sc = vmlal_s8(_sc, vget_low_s8(_pA3), vget_low_s8(_pB3));
                _sd = vmlal_s8(_sd, vget_high_s8(_pA3), vget_high_s8(_pB3));
                _se = vmlal_s8(_se, vget_high_s8(_pA3), vget_low_s8(_pB3));
                _sf = vmlal_s8(_sf, vget_low_s8(_pA3), vget_high_s8(_pB3));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // 00112233 44556677

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 0)));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 1)));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 2)));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 3)));
                int16x8_t _s4 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 0)));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 1)));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 2)));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB)), 3)));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 0)));
                int16x8_t _s9 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 1)));
                int16x8_t _sa = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 2)));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 3)));
                int16x8_t _sc = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 0)));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 1)));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 2)));
                int16x8_t _sf = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB)), 3)));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233 44556677

                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB0));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB0));
                int16x8_t _s8 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _s9 = vmull_s8(vget_high_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sa = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB1));
                int16x8_t _sb = vmull_s8(vget_low_s8(_pA0), vget_high_s8(_pB1));
                int16x8_t _sc = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sd = vmull_s8(vget_high_s8(_pA1), vget_high_s8(_pB1));
                int16x8_t _se = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB1));
                int16x8_t _sf = vmull_s8(vget_low_s8(_pA1), vget_high_s8(_pB1));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
                _sum8 = vpadalq_s16(_sum8, _s8);
                _sum9 = vpadalq_s16(_sum9, _s9);
                _suma = vpadalq_s16(_suma, _sa);
                _sumb = vpadalq_s16(_sumb, _sb);
                _sumc = vpadalq_s16(_sumc, _sc);
                _sumd = vpadalq_s16(_sumd, _sd);
                _sume = vpadalq_s16(_sume, _se);
                _sumf = vpadalq_s16(_sumf, _sf);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                // int8x8_t _pB0 = vld1_s8(pB);

                // abcd efgh
                // 0123 4567

                int16x8_t _s01 = vmull_s8(_pA, vdup_n_s8(pB[0]));
                int16x8_t _s23 = vmull_s8(_pA, vdup_n_s8(pB[1]));
                int16x8_t _s45 = vmull_s8(_pA, vdup_n_s8(pB[2]));
                int16x8_t _s67 = vmull_s8(_pA, vdup_n_s8(pB[3]));
                int16x8_t _s89 = vmull_s8(_pA, vdup_n_s8(pB[4]));
                int16x8_t _sab = vmull_s8(_pA, vdup_n_s8(pB[5]));
                int16x8_t _scd = vmull_s8(_pA, vdup_n_s8(pB[6]));
                int16x8_t _sef = vmull_s8(_pA, vdup_n_s8(pB[7]));

                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s23));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s45));
                _sum3 = vaddw_s16(_sum3, vget_low_s16(_s67));
                _sum4 = vaddw_s16(_sum4, vget_high_s16(_s01));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s23));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s45));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
                _sum8 = vaddw_s16(_sum8, vget_low_s16(_s89));
                _sum9 = vaddw_s16(_sum9, vget_low_s16(_sab));
                _suma = vaddw_s16(_suma, vget_low_s16(_scd));
                _sumb = vaddw_s16(_sumb, vget_low_s16(_sef));
                _sumc = vaddw_s16(_sumc, vget_high_s16(_s89));
                _sumd = vaddw_s16(_sumd, vget_high_s16(_sab));
                _sume = vaddw_s16(_sume, vget_high_s16(_scd));
                _sumf = vaddw_s16(_sumf, vget_high_s16(_sef));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // abcd efgh
                // efgh abcd
                // cdab ghef
                // ghef cdab

                // 0123 4567
                // 3210 7654

                // abcdefgh  ->  ghefcdab  ->  cdabghef

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);
                int8x8_t _pA2 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                int8x8_t _pA3 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pA0)));

                // 01234567  ->  32107654

                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA2, _pB0);
                int16x8_t _s67 = vmull_s8(_pA3, _pB0);
                int16x8_t _s89 = vmull_s8(_pA0, _pB1);
                int16x8_t _sab = vmull_s8(_pA1, _pB1);
                int16x8_t _scd = vmull_s8(_pA2, _pB1);
                int16x8_t _sef = vmull_s8(_pA3, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
                _sum8 = vaddw_s16(_sum8, vget_low_s16(_s89));
                _sum9 = vaddw_s16(_sum9, vget_high_s16(_s89));
                _suma = vaddw_s16(_suma, vget_low_s16(_sab));
                _sumb = vaddw_s16(_sumb, vget_high_s16(_sab));
                _sumc = vaddw_s16(_sumc, vget_low_s16(_scd));
                _sumd = vaddw_s16(_sumd, vget_high_s16(_scd));
                _sume = vaddw_s16(_sume, vget_low_s16(_sef));
                _sumf = vaddw_s16(_sumf, vget_high_s16(_sef));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                //      e4 f4 g4 h4
                //      e5 f5 g5 h5
                //      e6 f6 g6 h6
                //      e7 f7 g7 h7
                if (out_elempack == 8)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum4);
                    vst1q_s32(outptr0 + 8, _sum1);
                    vst1q_s32(outptr0 + 12, _sum5);
                    vst1q_s32(outptr0 + 16, _sum2);
                    vst1q_s32(outptr0 + 20, _sum6);
                    vst1q_s32(outptr0 + 24, _sum3);
                    vst1q_s32(outptr0 + 28, _sum7);
                    vst1q_s32(outptr0 + 32, _sum8);
                    vst1q_s32(outptr0 + 36, _sumc);
                    vst1q_s32(outptr0 + 40, _sum9);
                    vst1q_s32(outptr0 + 44, _sumd);
                    vst1q_s32(outptr0 + 48, _suma);
                    vst1q_s32(outptr0 + 52, _sume);
                    vst1q_s32(outptr0 + 56, _sumb);
                    vst1q_s32(outptr0 + 60, _sumf);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum8);
                    vst1q_s32(outptr0 + 20, _sum9);
                    vst1q_s32(outptr0 + 24, _suma);
                    vst1q_s32(outptr0 + 28, _sumb);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
                    vst1q_s32(outptr0 + out_hstep * 4 + 16, _sumc);
                    vst1q_s32(outptr0 + out_hstep * 4 + 20, _sumd);
                    vst1q_s32(outptr0 + out_hstep * 4 + 24, _sume);
                    vst1q_s32(outptr0 + out_hstep * 4 + 28, _sumf);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    //      e0 e1 e2 e3
                    //      e4 e5 e6 e7
                    //      f0 f1 f2 f3
                    //      f4 f5 f6 f7
                    //      g0 g1 g2 g3
                    //      g4 g5 g6 g7
                    //      h0 h1 h2 h3
                    //      h4 h5 h6 h7
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
                        int32x4x2_t _t2 = vzipq_s32(_sum8, _sum9);
                        int32x4x2_t _t3 = vzipq_s32(_suma, _sumb);
                        int32x4x2_t _t4 = vzipq_s32(_sum4, _sum5);
                        int32x4x2_t _t5 = vzipq_s32(_sum6, _sum7);
                        int32x4x2_t _t6 = vzipq_s32(_sumc, _sumd);
                        int32x4x2_t _t7 = vzipq_s32(_sume, _sumf);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
                        _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                        _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                        _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                        _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                        _sumc = vcombine_s32(vget_low_s32(_t4.val[1]), vget_low_s32(_t5.val[1]));
                        _sumd = vcombine_s32(vget_low_s32(_t6.val[1]), vget_low_s32(_t7.val[1]));
                        _sume = vcombine_s32(vget_high_s32(_t4.val[1]), vget_high_s32(_t5.val[1]));
                        _sumf = vcombine_s32(vget_high_s32(_t6.val[1]), vget_high_s32(_t7.val[1]));
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum8);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
                    vst1q_s32(outptr0 + out_hstep * 5, _suma);
                    vst1q_s32(outptr0 + out_hstep * 5 + 4, _sumb);
                    vst1q_s32(outptr0 + out_hstep * 6, _sumc);
                    vst1q_s32(outptr0 + out_hstep * 6 + 4, _sumd);
                    vst1q_s32(outptr0 + out_hstep * 7, _sume);
                    vst1q_s32(outptr0 + out_hstep * 7 + 4, _sumf);
                    outptr0 += 8;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      e4 f5 g6 h7
                //      e0 f1 g2 h3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      g4 h5 e6 f7
                //      g0 h1 e2 f3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      e7 f6 g5 h4
                //      e3 f2 g1 h0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      g7 h6 e5 f4
                //      g3 h2 e1 f0
                //      c7 d6 a5 b4
                if (out_elempack == 8)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    //      a2 b2 c2 d2
                    //      e2 f2 g2 h2
                    //      a3 b3 c3 d3
                    //      e3 f3 g3 h3
                    //      a4 b4 c4 d4
                    //      e4 f4 g4 h4
                    //      a5 b5 c5 d5
                    //      e5 f5 g5 h5
                    //      a6 b6 c6 d6
                    //      e6 f6 g6 h6
                    //      a7 b7 c7 d7
                    //      e7 f7 g7 h7
                    {
                        _sum8 = vrev64q_s32(_sum8);
                        _sum9 = vrev64q_s32(_sum9);
                        _suma = vrev64q_s32(_suma);
                        _sumb = vrev64q_s32(_sumb);
                        _sumc = vrev64q_s32(_sumc);
                        _sumd = vrev64q_s32(_sumd);
                        _sume = vrev64q_s32(_sume);
                        _sumf = vrev64q_s32(_sumf);
                        _sum8 = vextq_s32(_sum8, _sum8, 2);
                        _sum9 = vextq_s32(_sum9, _sum9, 2);
                        _suma = vextq_s32(_suma, _suma, 2);
                        _sumb = vextq_s32(_sumb, _sumb, 2);
                        _sumc = vextq_s32(_sumc, _sumc, 2);
                        _sumd = vextq_s32(_sumd, _sumd, 2);
                        _sume = vextq_s32(_sume, _sume, 2);
                        _sumf = vextq_s32(_sumf, _sumf, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                        int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                        int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                        int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                        int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                        int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                        int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                        int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                        _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                        _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                        _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                        _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                        _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                        _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                        _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _suma = vrev64q_s32(_suma);
                        _sumb = vrev64q_s32(_sumb);
                        _sume = vrev64q_s32(_sume);
                        _sumf = vrev64q_s32(_sumf);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    vst1q_s32(outptr0 + 32, _sum8);
                    vst1q_s32(outptr0 + 36, _sum9);
                    vst1q_s32(outptr0 + 40, _suma);
                    vst1q_s32(outptr0 + 44, _sumb);
                    vst1q_s32(outptr0 + 48, _sumc);
                    vst1q_s32(outptr0 + 52, _sumd);
                    vst1q_s32(outptr0 + 56, _sume);
                    vst1q_s32(outptr0 + 60, _sumf);
                    outptr0 += 64;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      a4 b4 c4 d4
                    //      a5 b5 c5 d5
                    //      a6 b6 c6 d6
                    //      a7 b7 c7 d7
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    //      e2 f2 g2 h2
                    //      e3 f3 g3 h3
                    //      e4 f4 g4 h4
                    //      e5 f5 g5 h5
                    //      e6 f6 g6 h6
                    //      e7 f7 g7 h7
                    {
                        _sum8 = vrev64q_s32(_sum8);
                        _sum9 = vrev64q_s32(_sum9);
                        _suma = vrev64q_s32(_suma);
                        _sumb = vrev64q_s32(_sumb);
                        _sumc = vrev64q_s32(_sumc);
                        _sumd = vrev64q_s32(_sumd);
                        _sume = vrev64q_s32(_sume);
                        _sumf = vrev64q_s32(_sumf);
                        _sum8 = vextq_s32(_sum8, _sum8, 2);
                        _sum9 = vextq_s32(_sum9, _sum9, 2);
                        _suma = vextq_s32(_suma, _suma, 2);
                        _sumb = vextq_s32(_sumb, _sumb, 2);
                        _sumc = vextq_s32(_sumc, _sumc, 2);
                        _sumd = vextq_s32(_sumd, _sumd, 2);
                        _sume = vextq_s32(_sume, _sume, 2);
                        _sumf = vextq_s32(_sumf, _sumf, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                        int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                        int32x4x2_t _t2 = vzipq_s32(_sum2, _sume);
                        int32x4x2_t _t3 = vzipq_s32(_sum6, _suma);
                        int32x4x2_t _t4 = vzipq_s32(_sum3, _sumf);
                        int32x4x2_t _t5 = vzipq_s32(_sum7, _sumb);
                        int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                        int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                        _sum8 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum9 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _suma = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sumb = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sumc = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                        _sumd = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                        _sume = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                        _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum9 = vrev64q_s32(_sum9);
                        _sumb = vrev64q_s32(_sumb);
                        _sumd = vrev64q_s32(_sumd);
                        _sumf = vrev64q_s32(_sumf);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum8);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
                    vst1q_s32(outptr0 + out_hstep * 4 + 8, _suma);
                    vst1q_s32(outptr0 + out_hstep * 4 + 12, _sumb);
                    vst1q_s32(outptr0 + out_hstep * 4 + 16, _sumc);
                    vst1q_s32(outptr0 + out_hstep * 4 + 20, _sumd);
                    vst1q_s32(outptr0 + out_hstep * 4 + 24, _sume);
                    vst1q_s32(outptr0 + out_hstep * 4 + 28, _sumf);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    //      e0 e1 e2 e3
                    //      e4 e5 e6 e7
                    //      f0 f1 f2 f3
                    //      f4 f5 f6 f7
                    //      g0 g1 g2 g3
                    //      g4 g5 g6 g7
                    //      h0 h1 h2 h3
                    //      h4 h5 h6 h7
                    {
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        _sumc = vextq_s32(_sumc, _sumc, 2);
                        _sumd = vextq_s32(_sumd, _sumd, 2);
                        _sume = vextq_s32(_sume, _sume, 2);
                        _sumf = vextq_s32(_sumf, _sumf, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sumc);
                        int32x4x2_t _t1 = vzipq_s32(_sum4, _sum8);
                        int32x4x2_t _t2 = vzipq_s32(_sum3, _sumf);
                        int32x4x2_t _t3 = vzipq_s32(_sum7, _sumb);
                        int32x4x2_t _t4 = vzipq_s32(_sum2, _sume);
                        int32x4x2_t _t5 = vzipq_s32(_sum6, _suma);
                        int32x4x2_t _t6 = vzipq_s32(_sum1, _sumd);
                        int32x4x2_t _t7 = vzipq_s32(_sum5, _sum9);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                        _sum9 = vcombine_s32(vget_low_s32(_t6.val[0]), vget_low_s32(_t7.val[0]));
                        _suma = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                        _sumb = vcombine_s32(vget_high_s32(_t6.val[0]), vget_high_s32(_t7.val[0]));
                        _sumc = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                        _sumd = vcombine_s32(vget_low_s32(_t7.val[1]), vget_low_s32(_t6.val[1]));
                        _sume = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
                        _sumf = vcombine_s32(vget_high_s32(_t7.val[1]), vget_high_s32(_t6.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _suma = vrev64q_s32(_suma);
                        _sumb = vrev64q_s32(_sumb);
                        _sume = vrev64q_s32(_sume);
                        _sumf = vrev64q_s32(_sumf);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum8);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum9);
                    vst1q_s32(outptr0 + out_hstep * 5, _suma);
                    vst1q_s32(outptr0 + out_hstep * 5 + 4, _sumb);
                    vst1q_s32(outptr0 + out_hstep * 6, _sumc);
                    vst1q_s32(outptr0 + out_hstep * 6 + 4, _sumd);
                    vst1q_s32(outptr0 + out_hstep * 7, _sume);
                    vst1q_s32(outptr0 + out_hstep * 7 + 4, _sumf);
                    outptr0 += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
                vst1q_s32(outptr + 32, _sum8);
                vst1q_s32(outptr + 36, _sum9);
                vst1q_s32(outptr + 40, _suma);
                vst1q_s32(outptr + 44, _sumb);
                vst1q_s32(outptr + 48, _sumc);
                vst1q_s32(outptr + 52, _sumd);
                vst1q_s32(outptr + 56, _sume);
                vst1q_s32(outptr + 60, _sumf);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cmp    %w9, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n"
                "sub    %0, %0, #64                 \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%1], #64 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v4.16b      \n"
                "smmla  v25.4s, v1.16b, v4.16b      \n"
                "smmla  v26.4s, v0.16b, v5.16b      \n"
                "smmla  v27.4s, v1.16b, v5.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v28.4s, v2.16b, v4.16b      \n"
                "smmla  v29.4s, v3.16b, v4.16b      \n"
                "smmla  v30.4s, v2.16b, v5.16b      \n"
                "smmla  v31.4s, v3.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v4.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v2.16b, v5.4b[0]    \n"
                "sdot   v17.4s, v2.16b, v5.4b[1]    \n"
                "sdot   v18.4s, v2.16b, v5.4b[2]    \n"
                "sdot   v19.4s, v2.16b, v5.4b[3]    \n"
                "sdot   v20.4s, v3.16b, v5.4b[0]    \n"
                "sdot   v21.4s, v3.16b, v5.4b[1]    \n"
                "sdot   v22.4s, v3.16b, v5.4b[2]    \n"
                "sdot   v23.4s, v3.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"
                "uzp1   v4.4s, v28.4s, v29.4s       \n"
                "uzp2   v5.4s, v28.4s, v29.4s       \n"
                "uzp1   v6.4s, v30.4s, v31.4s       \n"
                "uzp2   v7.4s, v30.4s, v31.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w8, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v2.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v2.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v2.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v2.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b}, [%2], #16         \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "ext    v5.16b, v4.16b, v4.16b, #8  \n"
                "smull2 v9.8h, v0.16b, v5.16b       \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull2 v11.8h, v2.16b, v5.16b      \n"
                "ext    v7.16b, v6.16b, v6.16b, #8  \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "rev64  v3.4s, v1.4s                \n"
                "smull2 v13.8h, v0.16b, v7.16b      \n"
                "smull2 v15.8h, v2.16b, v7.16b      \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal2 v11.8h, v3.16b, v4.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v6.16b      \n"
                "smlal2 v15.8h, v3.16b, v6.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w8, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "smull2 v12.8h, v0.16b, v4.16b      \n"
                "smull2 v13.8h, v0.16b, v5.16b      \n"
                "smull2 v14.8h, v0.16b, v6.16b      \n"
                "smull2 v15.8h, v0.16b, v7.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1r   {v2.2d}, [%2]               \n"
                "add    %2, %2, #8                  \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w8, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2]               \n"
                "add    %2, %2, #4                  \n"
                "dup    v8.8b, v1.b[0]              \n"
                "dup    v9.8b, v1.b[1]              \n"
                "dup    v10.8b, v1.b[2]             \n"
                "dup    v11.8b, v1.b[3]             \n"
                "smull  v8.8h, v0.8b, v8.8b         \n"
                "smull  v9.8h, v0.8b, v9.8b         \n"
                "smull  v10.8h, v0.8b, v10.8b       \n"
                "smull  v11.8h, v0.8b, v11.8b       \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw  v18.4s, v18.4s, v10.4h      \n"
                "saddw  v19.4s, v19.4s, v11.4h      \n"
                "saddw2 v20.4s, v20.4s, v8.8h       \n"
                "saddw2 v21.4s, v21.4s, v9.8h       \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1r   {v4.2s}, [%2]               \n"
                "add    %2, %2, #4                  \n"
                "rev32  v1.4h, v0.4h                \n"
                "rev64  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "smull  v10.8h, v0.8b, v5.8b        \n"
                "smull  v11.8h, v1.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "cmp    %w10, #0                    \n"
                "beq    10f                         \n"

#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                // if out_elempack == 4 / 8
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                // if out_elempack == 8
                "cmp    %w11, #8                    \n"
                "bne    7f                          \n"

                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v20.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [%3], #16         \n"
                "st1    {v21.4s}, [%3], #16         \n"
                "st1    {v18.4s}, [%3], #16         \n"
                "st1    {v22.4s}, [%3], #16         \n"
                "st1    {v19.4s}, [%3], #16         \n"
                "st1    {v23.4s}, [%3], #16         \n"
                "b      9f                          \n"

                // if out_elempack == 4
                "7:                                 \n"
                "add    x4, %3, %12, lsl #4         \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                "zip1   v0.4s, v16.4s, v17.4s       \n"
                "zip2   v1.4s, v16.4s, v17.4s       \n"
                "zip1   v2.4s, v18.4s, v19.4s       \n"
                "zip2   v3.4s, v18.4s, v19.4s       \n"
                "zip1   v4.4s, v20.4s, v21.4s       \n"
                "zip2   v5.4s, v20.4s, v21.4s       \n"
                "zip1   v6.4s, v22.4s, v23.4s       \n"
                "zip2   v7.4s, v22.4s, v23.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v1.2d, v3.2d        \n"
                "zip2   v19.2d, v1.2d, v3.2d        \n"
                "zip1   v20.2d, v4.2d, v6.2d        \n"
                "zip2   v21.2d, v4.2d, v6.2d        \n"
                "zip1   v22.2d, v5.2d, v7.2d        \n"
                "zip2   v23.2d, v5.2d, v7.2d        \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v18.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v19.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v21.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v23.4s}, [x4]              \n"
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0
                // if out_elempack == 4 / 8
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                "rev64  v20.4s, v20.4s              \n"
                "rev64  v21.4s, v21.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"
                "ext    v20.16b, v20.16b, v20.16b, #8 \n"
                "ext    v21.16b, v21.16b, v21.16b, #8 \n"
                "ext    v22.16b, v22.16b, v22.16b, #8 \n"
                "ext    v23.16b, v23.16b, v23.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v22.4s       \n"
                "zip2   v1.4s, v16.4s, v22.4s       \n"
                "zip1   v2.4s, v18.4s, v20.4s       \n"
                "zip2   v3.4s, v18.4s, v20.4s       \n"
                "zip1   v4.4s, v17.4s, v23.4s       \n"
                "zip2   v5.4s, v17.4s, v23.4s       \n"
                "zip1   v6.4s, v19.4s, v21.4s       \n"
                "zip2   v7.4s, v19.4s, v21.4s       \n"

                // if out_elempack == 8
                "cmp    %w11, #8                    \n"
                "bne    7f                          \n"

                // to
                //      a0 b0 c0 d0
                //      e0 f0 g0 h0
                //      a1 b1 c1 d1
                //      e1 f1 g1 h1
                //      a2 b2 c2 d2
                //      e2 f2 g2 h2
                //      a3 b3 c3 d3
                //      e3 f3 g3 h3
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v3.2d, v1.2d        \n"
                "zip1   v21.2d, v7.2d, v5.2d        \n"
                "zip2   v22.2d, v3.2d, v1.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "rev64  v18.4s, v18.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 4
                "7:                                 \n"

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v24.2d, v4.2d, v6.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip2   v25.2d, v4.2d, v6.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip1   v26.2d, v7.2d, v5.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "zip2   v27.2d, v7.2d, v5.2d        \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v25.4s, v25.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v27.4s, v27.4s              \n"

                "add    x4, %3, %12, lsl #4         \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [x4] \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                "ext    v18.16b, v18.16b, v18.16b, #8 \n"
                "ext    v19.16b, v19.16b, v19.16b, #8 \n"
                "ext    v22.16b, v22.16b, v22.16b, #8 \n"
                "ext    v23.16b, v23.16b, v23.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v22.4s       \n"
                "zip2   v1.4s, v16.4s, v22.4s       \n"
                "zip1   v2.4s, v18.4s, v20.4s       \n"
                "zip2   v3.4s, v18.4s, v20.4s       \n"
                "zip1   v4.4s, v17.4s, v23.4s       \n"
                "zip2   v5.4s, v17.4s, v23.4s       \n"
                "zip1   v6.4s, v19.4s, v21.4s       \n"
                "zip2   v7.4s, v19.4s, v21.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "zip1   v20.2d, v4.2d, v6.2d        \n"
                "zip2   v21.2d, v4.2d, v6.2d        \n"
                "zip1   v22.2d, v7.2d, v5.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v21.4s, v21.4s              \n"
                "rev64  v23.4s, v23.4s              \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v18.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v19.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v21.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v23.4s}, [x4]              \n"
#endif // __ARM_FEATURE_DOTPROD

                "9:                                 \n"
                "add    %0, %0, #128                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %9, #0              \n"
                "beq        0f                  \n"

                "vldm       %0!, {d16-d23}      \n"
                "vldm       %0, {d24-d31}       \n"
                "sub        %0, %0, #64         \n"
                "b          1f                  \n"

                "0:                             \n"
                "veor       q8, q8              \n"
                "veor       q9, q9              \n"
                "veor       q10, q10            \n"
                "veor       q11, q11            \n"
                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "veor       q14, q14            \n"
                "veor       q15, q15            \n"

                "1:                             \n"
                "lsr        r4, %8, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                ".align 4                       \n"
                "2:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.s8    {d0-d3}, [%1 :64]!  \n"
                "pld        [%2, #128]          \n"
                "vld1.s8    {d4-d5}, [%2]!      \n"
                "vmull.s8   q4, d0, d4          \n"
                "vrev64.32  q3, q0              \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d6, d4          \n"
                "vmull.s8   q7, d7, d4          \n"
                "vrev64.32  q0, q1              \n"
                "vmlal.s8   q4, d2, d5          \n"
                "vmlal.s8   q5, d3, d5          \n"
                "vmlal.s8   q6, d0, d5          \n"
                "vmlal.s8   q7, d1, d5          \n"
                "vrev64.16  q2, q2              \n"
                "vpadal.s16 q8, q4              \n"
                "vrev64.32  q1, q3              \n"
                "vpadal.s16 q9, q5              \n"
                "vmull.s8   q4, d6, d4          \n"
                "vpadal.s16 q10, q6             \n"
                "vmull.s8   q5, d7, d4          \n"
                "vpadal.s16 q11, q7             \n"
                "vmull.s8   q6, d2, d4          \n"
                "vmull.s8   q7, d3, d4          \n"
                "vrev64.32  q3, q0              \n"
                "vmlal.s8   q4, d0, d5          \n"
                "vmlal.s8   q5, d1, d5          \n"
                "vmlal.s8   q6, d6, d5          \n"
                "vmlal.s8   q7, d7, d5          \n"
                "subs       r4, r4, #1          \n"
                "vpadal.s16 q14, q4             \n"
                "vpadal.s16 q15, q5             \n"
                "vpadal.s16 q12, q6             \n"
                "vpadal.s16 q13, q7             \n"
                "bne        2b                  \n"

                "3:                             \n"
                "and        r4, %8, #2          \n" // r4 = remain = max_kk & 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                // kk += 2 part
                "vld1.s8    {d0-d1}, [%1 :64]!  \n"
                "vld1.s8    {d4}, [%2]!         \n"
                "vrev64.32  q1, q0              \n"
                "vrev64.16  d5, d4              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d2, d4          \n"
                "vmull.s8   q7, d3, d4          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"
                "vmull.s8   q4, d0, d5          \n"
                "vmull.s8   q5, d1, d5          \n"
                "vmull.s8   q6, d2, d5          \n"
                "vmull.s8   q7, d3, d5          \n"
                "vpadal.s16 q12, q4             \n"
                "vpadal.s16 q13, q5             \n"
                "vpadal.s16 q14, q6             \n"
                "vpadal.s16 q15, q7             \n"

                "4:                             \n"
                "and        r4, %8, #1          \n" // r4 = remain = max_kk & 1
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                // kk += 1 part
                "vld1.s8    {d0}, [%1 :64]!     \n"
                "vld1.s32   {d2[]}, [%2]!       \n"
                "vrev64.16  d1, d0              \n"
                "vrev64.8   d3, d2              \n"
                "vext.s8    d1, d1, #4          \n"
                "vmull.s8   q4, d0, d2          \n"
                "vmull.s8   q5, d1, d2          \n"
                "vmull.s8   q6, d0, d3          \n"
                "vmull.s8   q7, d1, d3          \n"
                "vaddw.s16  q8, d8              \n"
                "vaddw.s16  q9, d9              \n"
                "vaddw.s16  q10, d10            \n"
                "vaddw.s16  q11, d11            \n"
                "vaddw.s16  q12, d12            \n"
                "vaddw.s16  q13, d13            \n"
                "vaddw.s16  q14, d14            \n"
                "vaddw.s16  q15, d15            \n"

                "5:                             \n"
                "cmp        %10, #0             \n"
                "beq        10f                 \n"

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0
                // if out_elempack == 4 / 8
                "cmp        %11, #1             \n"
                "beq        8f                  \n"

                "vrev64.32  q12, q12            \n"
                "vrev64.32  q13, q13            \n"
                "vrev64.32  q14, q14            \n"
                "vrev64.32  q15, q15            \n"
                "vext.32    q12, q12, #2        \n"
                "vext.32    q13, q13, #2        \n"
                "vext.32    q14, q14, #2        \n"
                "vext.32    q15, q15, #2        \n"
                "vzip.32    q8, q14             \n"
                "vzip.32    q10, q12            \n"
                "vzip.32    q9, q15             \n"
                "vzip.32    q11, q13            \n"
                "vswp       d17, d20            \n"
                "vswp       d19, d22            \n"
                "vswp       d28, d25            \n"
                "vswp       d30, d27            \n"
                "vrev64.32  q10, q10            \n"
                "vrev64.32  q11, q11            \n"
                "vrev64.32  q14, q14            \n"
                "vrev64.32  q15, q15            \n"

                // if out_elempack == 8
                "cmp        %11, #8             \n"
                "bne        7f                  \n"

                // to
                //      a0 b0 c0 d0
                //      e0 f0 g0 h0
                //      a1 b1 c1 d1
                //      e1 f1 g1 h1
                //      a2 b2 c2 d2
                //      e2 f2 g2 h2
                //      a3 b3 c3 d3
                //      e3 f3 g3 h3
                "vstm       %3!, {d16-d23}      \n"
                "vstm       %3!, {d24-d31}      \n"
                "b          9f                  \n"

                // if out_elempack == 4
                "7:                             \n"
                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                "vswp       q9, q10             \n"
                "vswp       q13, q14            \n"
                "vswp       q10, q12            \n"
                "vswp       q11, q13            \n"

                "add        r4, %3, %12, lsl #4 \n"
                "vstm       %3!, {d16-d23}      \n"
                "vstm       r4, {d24-d31}       \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                //      e0 e1 e2 e3
                //      f0 f1 f2 f3
                //      g0 g1 g2 g3
                //      h0 h1 h2 h3
                "vext.32    q10, q10, #2        \n"
                "vext.32    q11, q11, #2        \n"
                "vext.32    q14, q14, #2        \n"
                "vext.32    q15, q15, #2        \n"
                "vzip.32    q8, q14             \n"
                "vzip.32    q10, q12            \n"
                "vzip.32    q9, q15             \n"
                "vzip.32    q11, q13            \n"
                "vswp       d17, d20            \n"
                "vswp       d19, d22            \n"
                "vswp       d28, d25            \n"
                "vswp       d30, d27            \n"
                "vrev64.32  q10, q10            \n"
                "vrev64.32  q11, q11            \n"
                "vrev64.32  q14, q14            \n"
                "vrev64.32  q15, q15            \n"

                "add        r4, %3, %12, lsl #2 \n"
                "vst1.s32   {d16-d17}, [%3]!    \n"
                "vst1.s32   {d20-d21}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d24-d25}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d28-d29}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d18-d19}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d22-d23}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d26-d27}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d30-d31}, [r4]     \n"

                "9:                             \n"
                "add        %0, %0, #128        \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d16-d23}      \n"
                "vstm       %0!, {d24-d31}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
                int32x4_t _s4 = vdupq_n_s32(0);
                int32x4_t _s5 = vdupq_n_s32(0);
                int32x4_t _s6 = vdupq_n_s32(0);
                int32x4_t _s7 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000 11111111 22222222 33333333

                    _s0 = vmmlaq_s32(_s0, _pA0, _pB0);
                    _s1 = vmmlaq_s32(_s1, _pA1, _pB0);
                    _s2 = vmmlaq_s32(_s2, _pA0, _pB1);
                    _s3 = vmmlaq_s32(_s3, _pA1, _pB1);
                    _s4 = vmmlaq_s32(_s4, _pA2, _pB0);
                    _s5 = vmmlaq_s32(_s5, _pA3, _pB0);
                    _s6 = vmmlaq_s32(_s6, _pA2, _pB1);
                    _s7 = vmmlaq_s32(_s7, _pA3, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB0, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB0, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB0, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB0, 3);

                    _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB1, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB1, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA2, _pB1, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA2, _pB1, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _pA3, _pB1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _pA3, _pB1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _pA3, _pB1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _pA3, _pB1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                int32x4x2_t _ss2 = vuzpq_s32(_s4, _s5);
                int32x4x2_t _ss3 = vuzpq_s32(_s6, _s7);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB = vld1q_s8(pB);

                // aaaa bbbb cccc dddd   eeee ffff gggg hhhh

                // 0000 1111 2222 3333

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));
                int8x16_t _pA3 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA2)));

                // 00112233 44556677

                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB02));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB13));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA3), vget_high_s8(_pB02));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA3), vget_high_s8(_pB02));
                _s4 = vmlal_s8(_s4, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _s6 = vmlal_s8(_s6, vget_low_s8(_pA3), vget_high_s8(_pB13));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA3), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh

                // 00112233
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));
                int16x8_t _s4 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s6 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA), vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd eeffgghh

                // ccddaabb gghheeff

                int8x16_t _pA1 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA0)));

                // 00112233

                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA1), _pB0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA1), _pB0);
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA0), _pB1);
                int16x8_t _s5 = vmull_s8(vget_high_s8(_pA0), _pB1);
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA1), _pB1);
                int16x8_t _s7 = vmull_s8(vget_high_s8(_pA1), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                // int8x8_t _pB0 = vreinterpret_s32_s8(vld1_dup_s32(pB));

                // abcdefgh

                // 0123

                int16x8_t _s01 = vmull_s8(_pA0, vdup_n_s8(pB[0]));
                int16x8_t _s23 = vmull_s8(_pA0, vdup_n_s8(pB[1]));
                int16x8_t _s45 = vmull_s8(_pA0, vdup_n_s8(pB[2]));
                int16x8_t _s67 = vmull_s8(_pA0, vdup_n_s8(pB[3]));
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s23));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s45));
                _sum3 = vaddw_s16(_sum3, vget_low_s16(_s67));
                _sum4 = vaddw_s16(_sum4, vget_high_s16(_s01));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s23));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s45));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                // int8x8_t _pB0 = vld1_s8(pB);
                // _pB0 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB0)).val[0]);

                // abcdefgh  ->  cdabghef
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));

                // 01230123  ->  32103210
                int8x8_t _pB1 = vrev64_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                //      e2 f2 g2 h2
                //      e3 f3 g3 h3
                if (out_elempack == 8)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum4);
                    vst1q_s32(outptr0 + 8, _sum1);
                    vst1q_s32(outptr0 + 12, _sum5);
                    vst1q_s32(outptr0 + 16, _sum2);
                    vst1q_s32(outptr0 + 20, _sum6);
                    vst1q_s32(outptr0 + 24, _sum3);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    //      e0 e1 e2 e3
                    //      f0 f1 f2 f3
                    //      g0 g1 g2 g3
                    //      h0 h1 h2 h3
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
                        int32x4x2_t _t2 = vzipq_s32(_sum4, _sum5);
                        int32x4x2_t _t3 = vzipq_s32(_sum6, _sum7);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 5, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 6, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 7, _sum7);
                    outptr0 += 4;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      e0 f1 g2 h3
                //      c0 d1 a2 b3
                //      g0 h1 e2 f3
                //      a3 b2 c1 d0
                //      e3 f2 g1 h0
                //      c3 d2 a1 b0
                //      g3 h2 e1 f0
                if (out_elempack == 8)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    //      a2 b2 c2 d2
                    //      e2 f2 g2 h2
                    //      a3 b3 c3 d3
                    //      e3 f3 g3 h3
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    //      e2 f2 g2 h2
                    //      e3 f3 g3 h3
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 4 + 8, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 4 + 12, _sum7);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    //      e0 e1 e2 e3
                    //      f0 f1 f2 f3
                    //      g0 g1 g2 g3
                    //      h0 h1 h2 h3
                    {
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 5, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 6, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 7, _sum7);
                    outptr0 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000 11111111

                    _s0 = vmmlaq_s32(_s0, _pA0, _pB);
                    _s1 = vmmlaq_s32(_s1, _pA1, _pB);
                    _s2 = vmmlaq_s32(_s2, _pA2, _pB);
                    _s3 = vmmlaq_s32(_s3, _pA3, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB, 0);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB, 1);

                    _sum0 = vdotq_laneq_s32(_sum0, _pA2, _pB, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA2, _pB, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _pA3, _pB, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _pA3, _pB, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss0 = vuzpq_s32(_s0, _s1);
                int32x4x2_t _ss1 = vuzpq_s32(_s2, _s3);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x8_t _pB = vld1_s8(pB);

                // aaaa bbbb cccc dddd eeee ffff gggg hhhh

                // 0000 1111

                _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA0, _pB, 1);
                _sum2 = vdotq_lane_s32(_sum2, _pA1, _pB, 0);
                _sum3 = vdotq_lane_s32(_sum3, _pA1, _pB, 1);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh   aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233

                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA0), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_low_s8(_pA2), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA2), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int16x4_t _pB = vreinterpret_s16_s32(vld1_dup_s32((const int*)pB));

                int16x4x2_t _pB01 = vuzp_s16(_pB, _pB);
                int8x8_t _pB0 = vreinterpret_s8_s16(_pB01.val[0]);
                int8x8_t _pB1 = vreinterpret_s8_s16(_pB01.val[1]);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), _pB1);
                int16x8_t _s2 = vmull_s8(vget_high_s8(_pA), _pB0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd eeffgghh

                // 00110011
                // 11001100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA), _pB1);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int8x8x2_t _pB01 = vuzp_s8(_pB, _pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB01.val[0]);
                int16x8_t _s1 = vmull_s8(_pA, _pB01.val[1]);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s1));
                _sum2 = vaddw_s16(_sum2, vget_high_s16(_s0));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcdefgh

                // 01010101
                // 10101010
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      e0 f0 g0 h0
                //      e1 f1 g1 h1
                if (out_elempack == 8)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum2);
                    vst1q_s32(outptr0 + 8, _sum1);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum3);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 b0 b1
                    //      c0 c1 d0 d1
                    //      e0 e1 f0 f1
                    //      g0 g1 h0 h1
                    {
                        int32x4x2_t _sum02 = vzipq_s32(_sum0, _sum1);
                        int32x4x2_t _sum13 = vzipq_s32(_sum2, _sum3);
                        _sum0 = _sum02.val[0];
                        _sum1 = _sum02.val[1];
                        _sum2 = _sum13.val[0];
                        _sum3 = _sum13.val[1];
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 2, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 4, vget_low_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 5, vget_high_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 6, vget_low_s32(_sum3));
                    vst1_s32(outptr0 + out_hstep * 7, vget_high_s32(_sum3));
                    outptr0 += 2;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c0 d1
                //      e0 f1 g0 h1
                //      a1 b0 c1 d0
                //      e1 f0 g1 h0
                if (out_elempack == 8)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      e0 f0 g0 h0
                    //      a1 b1 c1 d1
                    //      e1 f1 g1 h1
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      e0 f0 g0 h0
                    //      e1 f1 g1 h1
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 4 + 4, _sum3);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 c0 c1
                    //      b0 b1 d0 d1
                    //      e0 e1 g0 g1
                    //      f0 f1 h0 h1
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum2);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t1.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t1.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 4, vget_low_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 5, vget_low_s32(_sum3));
                    vst1_s32(outptr0 + out_hstep * 6, vget_high_s32(_sum2));
                    vst1_s32(outptr0 + out_hstep * 7, vget_high_s32(_sum3));
                    outptr0 += 2;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _s0 = vdupq_n_s32(0);
                int32x4_t _s1 = vdupq_n_s32(0);
                int32x4_t _s2 = vdupq_n_s32(0);
                int32x4_t _s3 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pA2 = vld1q_s8(pA + 32);
                    int8x16_t _pA3 = vld1q_s8(pA + 48);

                    int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb ..... hhhhhhhh
                    // 00000000
                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _s0 = vdotq_s32(_s0, _pA0, _pBB);
                    _s1 = vdotq_s32(_s1, _pA1, _pBB);
                    _s2 = vdotq_s32(_s2, _pA2, _pBB);
                    _s3 = vdotq_s32(_s3, _pA3, _pBB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _pA1, _pB, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pA2, _pB, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _pA3, _pB, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 64;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_s0, _s1));
                _sum1 = vaddq_s32(_sum1, vpaddq_s32(_s2, _s3));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);

                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aaaa bbbb cccc dddd eeee ffff gggg hhhh

                // 0000 0000

                _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA1, _pB, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA2 = vld1q_s8(pA + 16);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), _pB0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA0), _pB0);
                _s0 = vmlal_s8(_s0, vget_low_s8(_pA2), _pB1);
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA2), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_pA), _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 16;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 8)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep * 4, _sum1);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    outptr0[0] = vgetq_lane_s32(_sum0, 0);
                    outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
                    outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
                    outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
                    outptr0[out_hstep * 4] = vgetq_lane_s32(_sum1, 0);
                    outptr0[out_hstep * 5] = vgetq_lane_s32(_sum1, 1);
                    outptr0[out_hstep * 6] = vgetq_lane_s32(_sum1, 2);
                    outptr0[out_hstep * 7] = vgetq_lane_s32(_sum1, 3);
                    outptr0++;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cmp    %w9, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n"
                "sub    %0, %0, #64                 \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v2.16b, v3.16b, v4.16b, v5.16b}, [%2], #64 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v2.16b      \n"
                "smmla  v25.4s, v1.16b, v2.16b      \n"
                "smmla  v26.4s, v0.16b, v3.16b      \n"
                "smmla  v27.4s, v1.16b, v3.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v28.4s, v0.16b, v4.16b      \n"
                "smmla  v29.4s, v1.16b, v4.16b      \n"
                "smmla  v30.4s, v0.16b, v5.16b      \n"
                "smmla  v31.4s, v1.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v21.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v22.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v23.4s, v0.16b, v3.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v1.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v1.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v1.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v1.16b, v4.4b[3]    \n"
                "sdot   v20.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v21.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v22.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v23.4s, v1.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"
                "uzp1   v4.4s, v28.4s, v29.4s       \n"
                "uzp2   v5.4s, v28.4s, v29.4s       \n"
                "uzp1   v6.4s, v30.4s, v31.4s       \n"
                "uzp2   v7.4s, v30.4s, v31.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
                "add    v20.4s, v20.4s, v4.4s       \n"
                "add    v21.4s, v21.4s, v5.4s       \n"
                "add    v22.4s, v22.4s, v6.4s       \n"
                "add    v23.4s, v23.4s, v7.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w8, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b, v3.16b}, [%2], #32 \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v0.16b, v3.4b[0]    \n"
                "sdot   v21.4s, v0.16b, v3.4b[1]    \n"
                "sdot   v22.4s, v0.16b, v3.4b[2]    \n"
                "sdot   v23.4s, v0.16b, v3.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull2 v9.8h, v0.16b, v5.16b       \n"
                "rev64  v2.4s, v0.4s                \n"
                "smull  v10.8h, v2.8b, v4.8b        \n"
                "smull2 v11.8h, v2.16b, v5.16b      \n"
                "rev64  v6.8h, v4.8h                \n"
                "smull  v12.8h, v0.8b, v6.8b        \n"
                "smull  v14.8h, v2.8b, v6.8b        \n"
                "rev64  v7.8h, v5.8h                \n"
                "smull2 v13.8h, v0.16b, v7.16b      \n"
                "smull2 v15.8h, v2.16b, v7.16b      \n"
                "ext    v1.16b, v0.16b, v0.16b, #8  \n"
                "ext    v3.16b, v2.16b, v2.16b, #8  \n"
                "smlal  v8.8h, v1.8b, v5.8b         \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal  v10.8h, v3.8b, v5.8b        \n"
                "smlal2 v11.8h, v3.16b, v4.16b      \n"
                "smlal  v12.8h, v1.8b, v7.8b        \n"
                "smlal  v14.8h, v3.8b, v7.8b        \n"
                "smlal2 v13.8h, v1.16b, v6.16b      \n"
                "smlal2 v15.8h, v3.16b, v6.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w8, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.16b}, [%2], #16         \n"
                "dup    v4.8h, v1.h[0]              \n"
                "dup    v5.8h, v1.h[1]              \n"
                "dup    v6.8h, v1.h[2]              \n"
                "dup    v7.8h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "dup    v4.8h, v1.h[4]              \n"
                "dup    v5.8h, v1.h[5]              \n"
                "dup    v6.8h, v1.h[6]              \n"
                "dup    v7.8h, v1.h[7]              \n"
                "smull  v12.8h, v0.8b, v4.8b        \n"
                "smull  v13.8h, v0.8b, v5.8b        \n"
                "smull  v14.8h, v0.8b, v6.8b        \n"
                "smull  v15.8h, v0.8b, v7.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2d}, [%1]               \n"
                "add    %1, %1, #8                  \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "rev64  v1.4s, v0.4s                \n"
                "rev64  v3.8h, v2.8h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull2 v9.8h, v0.16b, v2.16b       \n"
                "smull  v10.8h, v1.8b, v2.8b        \n"
                "smull2 v11.8h, v1.16b, v2.16b      \n"
                "smull  v12.8h, v0.8b, v3.8b        \n"
                "smull2 v13.8h, v0.16b, v3.16b      \n"
                "smull  v14.8h, v1.8b, v3.8b        \n"
                "smull2 v15.8h, v1.16b, v3.16b      \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "sadalp v20.4s, v12.8h              \n"
                "sadalp v21.4s, v13.8h              \n"
                "sadalp v22.4s, v14.8h              \n"
                "sadalp v23.4s, v15.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w8, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "add    %1, %1, #4                  \n"
                "dup    v8.8h, v1.h[0]              \n"
                "dup    v9.8h, v1.h[1]              \n"
                "dup    v10.8h, v1.h[2]             \n"
                "dup    v11.8h, v1.h[3]             \n"
                "uzp1   v2.8b, v8.8b, v9.8b         \n"
                "uzp2   v3.8b, v8.8b, v9.8b         \n"
                "uzp1   v4.8b, v10.8b, v11.8b       \n"
                "uzp2   v5.8b, v10.8b, v11.8b       \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v0.8b, v3.8b         \n"
                "smull  v10.8h, v0.8b, v4.8b        \n"
                "smull  v11.8h, v0.8b, v5.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw  v17.4s, v17.4s, v9.4h       \n"
                "saddw2 v18.4s, v18.4s, v8.8h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw  v21.4s, v21.4s, v11.4h      \n"
                "saddw2 v22.4s, v22.4s, v10.8h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1    {v2.8b}, [%2], #8           \n"
                "add    %1, %1, #4                  \n"
                "ext    v1.8b, v0.8b, v0.8b, #2     \n"
                "rev32  v3.8b, v2.8b                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v1.8b, v2.8b         \n"
                "smull  v10.8h, v0.8b, v3.8b        \n"
                "smull  v11.8h, v1.8b, v3.8b        \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
                "saddw  v20.4s, v20.4s, v10.4h      \n"
                "saddw2 v21.4s, v21.4s, v10.8h      \n"
                "saddw  v22.4s, v22.4s, v11.4h      \n"
                "saddw2 v23.4s, v23.4s, v11.8h      \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "cmp    %w10, #0                    \n"
                "beq    10f                         \n"

#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                // if out_elempack == 4
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                "zip1   v0.4s, v16.4s, v17.4s       \n"
                "zip2   v1.4s, v16.4s, v17.4s       \n"
                "zip1   v2.4s, v18.4s, v19.4s       \n"
                "zip2   v3.4s, v18.4s, v19.4s       \n"
                "zip1   v4.4s, v20.4s, v21.4s       \n"
                "zip2   v5.4s, v20.4s, v21.4s       \n"
                "zip1   v6.4s, v22.4s, v23.4s       \n"
                "zip2   v7.4s, v22.4s, v23.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v1.2d, v3.2d        \n"
                "zip1   v21.2d, v5.2d, v7.2d        \n"
                "zip2   v22.2d, v1.2d, v3.2d        \n"
                "zip2   v23.2d, v5.2d, v7.2d        \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s, v17.4s}, [%3], #32 \n"
                "st1    {v18.4s, v19.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s, v21.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s, v23.4s}, [x4]      \n"
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4
                // if out_elempack == 4
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                "rev64  v20.4s, v20.4s              \n"
                "rev64  v21.4s, v21.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"
                "ext    v20.16b, v20.16b, v20.16b, #8 \n"
                "ext    v21.16b, v21.16b, v21.16b, #8 \n"
                "ext    v22.16b, v22.16b, v22.16b, #8 \n"
                "ext    v23.16b, v23.16b, v23.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v22.4s       \n"
                "zip2   v1.4s, v16.4s, v22.4s       \n"
                "zip1   v2.4s, v18.4s, v20.4s       \n"
                "zip2   v3.4s, v18.4s, v20.4s       \n"
                "zip1   v4.4s, v17.4s, v23.4s       \n"
                "zip2   v5.4s, v17.4s, v23.4s       \n"
                "zip1   v6.4s, v19.4s, v21.4s       \n"
                "zip2   v7.4s, v19.4s, v21.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "zip1   v20.2d, v4.2d, v6.2d        \n"
                "zip2   v21.2d, v4.2d, v6.2d        \n"
                "zip1   v22.2d, v7.2d, v5.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v21.4s, v21.4s              \n"
                "rev64  v23.4s, v23.4s              \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"

                // to
                //      a0 a1 a2 a3
                //      a4 a5 a6 a7
                //      b0 b1 b2 b3
                //      b4 b5 b6 b7
                //      c0 c1 c2 c3
                //      c4 c5 c6 c7
                //      d0 d1 d2 d3
                //      d4 d5 d6 d7
                "ext    v18.16b, v18.16b, v18.16b, #8 \n"
                "ext    v19.16b, v19.16b, v19.16b, #8 \n"
                "ext    v22.16b, v22.16b, v22.16b, #8 \n"
                "ext    v23.16b, v23.16b, v23.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v22.4s       \n"
                "zip2   v1.4s, v16.4s, v22.4s       \n"
                "zip1   v2.4s, v18.4s, v20.4s       \n"
                "zip2   v3.4s, v18.4s, v20.4s       \n"
                "zip1   v4.4s, v17.4s, v23.4s       \n"
                "zip2   v5.4s, v17.4s, v23.4s       \n"
                "zip1   v6.4s, v19.4s, v21.4s       \n"
                "zip2   v7.4s, v19.4s, v21.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip1   v17.2d, v4.2d, v6.2d        \n"
                "zip2   v18.2d, v0.2d, v2.2d        \n"
                "zip2   v19.2d, v4.2d, v6.2d        \n"
                "zip1   v20.2d, v3.2d, v1.2d        \n"
                "zip1   v21.2d, v7.2d, v5.2d        \n"
                "zip2   v22.2d, v3.2d, v1.2d        \n"
                "zip2   v23.2d, v7.2d, v5.2d        \n"
                "rev64  v18.4s, v18.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "rev64  v22.4s, v22.4s              \n"
                "rev64  v23.4s, v23.4s              \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s, v17.4s}, [%3], #32 \n"
                "st1    {v18.4s, v19.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v20.4s, v21.4s}, [x4]      \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v22.4s, v23.4s}, [x4]      \n"
#endif // __ARM_FEATURE_DOTPROD

                "9:                                 \n"
                "add    %0, %0, #128                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;
            int32x4_t _sum4;
            int32x4_t _sum5;
            int32x4_t _sum6;
            int32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
                _sum4 = vdupq_n_s32(0);
                _sum5 = vdupq_n_s32(0);
                _sum6 = vdupq_n_s32(0);
                _sum7 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
                _sum4 = vld1q_s32(outptr + 16);
                _sum5 = vld1q_s32(outptr + 20);
                _sum6 = vld1q_s32(outptr + 24);
                _sum7 = vld1q_s32(outptr + 28);
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum20 = vdupq_n_s32(0);
                int32x4_t _sum21 = vdupq_n_s32(0);
                int32x4_t _sum30 = vdupq_n_s32(0);
                int32x4_t _sum31 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111 22222222 33333333
                    // 44444444 55555555 66666666 77777777

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB0);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB0);
                    _sum10 = vmmlaq_s32(_sum10, _pA0, _pB1);
                    _sum11 = vmmlaq_s32(_sum11, _pA1, _pB1);
                    _sum20 = vmmlaq_s32(_sum20, _pA0, _pB2);
                    _sum21 = vmmlaq_s32(_sum21, _pA1, _pB2);
                    _sum30 = vmmlaq_s32(_sum30, _pA0, _pB3);
                    _sum31 = vmmlaq_s32(_sum31, _pA1, _pB3);

                    // a0 a1 b0 b1
                    // c0 c1 d0 d1
                    // a2 a3 b2 b3
                    // c2 c3 d2 d3
                    // a4 a5 b4 b5
                    // c4 c5 d4 d5
                    // a6 a7 b6 b7
                    // c6 c7 d6 d7

                    pA += 32;
                    pB += 64;
                }
                int32x4x2_t _ss0 = vuzpq_s32(_sum00, _sum01);
                int32x4x2_t _ss1 = vuzpq_s32(_sum10, _sum11);
                int32x4x2_t _ss2 = vuzpq_s32(_sum20, _sum21);
                int32x4x2_t _ss3 = vuzpq_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
                _sum4 = vaddq_s32(_sum4, _ss2.val[0]);
                _sum5 = vaddq_s32(_sum5, _ss2.val[1]);
                _sum6 = vaddq_s32(_sum6, _ss3.val[0]);
                _sum7 = vaddq_s32(_sum7, _ss3.val[1]);
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);
                int8x16_t _pB2 = vld1q_s8(pB + 32);
                int8x16_t _pB3 = vld1q_s8(pB + 48);

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA0, _pB1, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA0, _pB1, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA0, _pB1, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA0, _pB1, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB2, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB2, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB2, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB2, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA1, _pB3, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA1, _pB3, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA1, _pB3, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA1, _pB3, 3);

                pA += 32;
                pB += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _pA, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA, _pB0, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _pA, _pB1, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _pA, _pB1, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _pA, _pB1, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _pA, _pB1, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB2 = vld1q_s8(pB + 16);

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));
                int8x16_t _pB3 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(vget_low_s8(_pA02), vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(vget_low_s8(_pA13), vget_high_s8(_pB1));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_low_s8(_pB2));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA02), vget_high_s8(_pB2));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA13), vget_low_s8(_pB2));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB2));
                _s4 = vmlal_s8(_s4, vget_high_s8(_pA02), vget_low_s8(_pB3));
                _s5 = vmlal_s8(_s5, vget_high_s8(_pA02), vget_high_s8(_pB3));
                _s6 = vmlal_s8(_s6, vget_high_s8(_pA13), vget_low_s8(_pB3));
                _s7 = vmlal_s8(_s7, vget_high_s8(_pA13), vget_high_s8(_pB3));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x16_t _pB01 = vld1q_s8(pB);

                // aabbccdd

                // 00112233 44556677

                int16x8_t _s0 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 0)));
                int16x8_t _s1 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 1)));
                int16x8_t _s2 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 2)));
                int16x8_t _s3 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pB01)), 3)));
                int16x8_t _s4 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 0)));
                int16x8_t _s5 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 1)));
                int16x8_t _s6 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 2)));
                int16x8_t _s7 = vmull_s8(_pA0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pB01)), 3)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x16_t _pB0 = vld1q_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vreinterpret_s8_s32(vrev64_s32(vreinterpret_s32_s8(_pA0)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB1 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                int16x8_t _s4 = vmull_s8(_pA0, vget_low_s8(_pB1));
                int16x8_t _s5 = vmull_s8(_pA0, vget_high_s8(_pB1));
                int16x8_t _s6 = vmull_s8(_pA1, vget_low_s8(_pB1));
                int16x8_t _s7 = vmull_s8(_pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
                _sum4 = vpadalq_s16(_sum4, _s4);
                _sum5 = vpadalq_s16(_sum5, _s5);
                _sum6 = vpadalq_s16(_sum6, _s6);
                _sum7 = vpadalq_s16(_sum7, _s7);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pAA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_s8(pB);

                // abcdabcd
                // 01234567  ->  01010101 23232323 45454545 67676767
                int8x8_t _pB0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0));
                int8x8_t _pB2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1));
                int8x8_t _pB4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2));
                int8x8_t _pB6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3));

                int8x8x2_t _pB0123 = vuzp_s8(_pB0, _pB2);
                int8x8x2_t _pB4567 = vuzp_s8(_pB4, _pB6);

                int16x8_t _s02 = vmull_s8(_pAA, _pB0123.val[0]);
                int16x8_t _s13 = vmull_s8(_pAA, _pB0123.val[1]);
                int16x8_t _s46 = vmull_s8(_pAA, _pB4567.val[0]);
                int16x8_t _s57 = vmull_s8(_pAA, _pB4567.val[1]);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s02));
                _sum1 = vaddw_s16(_sum1, vget_low_s16(_s13));
                _sum2 = vaddw_s16(_sum2, vget_high_s16(_s02));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s13));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s46));
                _sum5 = vaddw_s16(_sum5, vget_low_s16(_s57));
                _sum6 = vaddw_s16(_sum6, vget_high_s16(_s46));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s57));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vld1_s8(pB);

                // abcd abcd
                // cdab cdab

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 2);

                // 0123 4567
                // 3210 7654

                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA0, _pB0);
                int16x8_t _s23 = vmull_s8(_pA1, _pB0);
                int16x8_t _s45 = vmull_s8(_pA0, _pB1);
                int16x8_t _s67 = vmull_s8(_pA1, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
                _sum4 = vaddw_s16(_sum4, vget_low_s16(_s45));
                _sum5 = vaddw_s16(_sum5, vget_high_s16(_s45));
                _sum6 = vaddw_s16(_sum6, vget_low_s16(_s67));
                _sum7 = vaddw_s16(_sum7, vget_high_s16(_s67));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                //      a4 b4 c4 d4
                //      a5 b5 c5 d5
                //      a6 b6 c6 d6
                //      a7 b7 c7 d7
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum3);
                        int32x4x2_t _t2 = vzipq_s32(_sum4, _sum5);
                        int32x4x2_t _t3 = vzipq_s32(_sum6, _sum7);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t1.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t1.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1]));
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      a4 b5 c6 d7
                //      c0 d1 a2 b3
                //      c4 d5 a6 b7
                //      a3 b2 c1 d0
                //      a7 b6 c5 d4
                //      c3 d2 a1 b0
                //      c7 d6 a5 b4
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    //      a4 b4 c4 d4
                    //      a5 b5 c5 d5
                    //      a6 b6 c6 d6
                    //      a7 b7 c7 d7
                    {
                        _sum4 = vrev64q_s32(_sum4);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                        _sum4 = vextq_s32(_sum4, _sum4, 2);
                        _sum5 = vextq_s32(_sum5, _sum5, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum5 = vrev64q_s32(_sum5);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    vst1q_s32(outptr0 + 16, _sum4);
                    vst1q_s32(outptr0 + 20, _sum5);
                    vst1q_s32(outptr0 + 24, _sum6);
                    vst1q_s32(outptr0 + 28, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      a4 a5 a6 a7
                    //      b0 b1 b2 b3
                    //      b4 b5 b6 b7
                    //      c0 c1 c2 c3
                    //      c4 c5 c6 c7
                    //      d0 d1 d2 d3
                    //      d4 d5 d6 d7
                    {
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        _sum6 = vextq_s32(_sum6, _sum6, 2);
                        _sum7 = vextq_s32(_sum7, _sum7, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum6);
                        int32x4x2_t _t1 = vzipq_s32(_sum2, _sum4);
                        int32x4x2_t _t2 = vzipq_s32(_sum1, _sum7);
                        int32x4x2_t _t3 = vzipq_s32(_sum3, _sum5);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                        _sum2 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum3 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                        _sum4 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum5 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                        _sum6 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum6 = vrev64q_s32(_sum6);
                        _sum7 = vrev64q_s32(_sum7);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_s32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_s32(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
                vst1q_s32(outptr + 16, _sum4);
                vst1q_s32(outptr + 20, _sum5);
                vst1q_s32(outptr + 24, _sum6);
                vst1q_s32(outptr + 28, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cmp    %w9, #0                     \n"
                "beq    0f                          \n"

                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0] \n"
                "b      1f                          \n"

                "0:                                 \n"
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"

                "1:                                 \n"
#if __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #3                 \n" // w4 = max_kk >> 3
                "cmp    w4, #0                      \n"
                "beq    101f                        \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "2:                                 \n"
                "ld1    {v0.16b, v1.16b}, [%1], #32 \n"
                "ld1    {v4.16b, v5.16b}, [%2], #32 \n"

#if __ARM_FEATURE_MATMUL_INT8
                "smmla  v24.4s, v0.16b, v4.16b      \n"
                "smmla  v25.4s, v1.16b, v4.16b      \n"
                "subs   w4, w4, #1                  \n"
                "smmla  v26.4s, v0.16b, v5.16b      \n"
                "smmla  v27.4s, v1.16b, v5.16b      \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "sdot   v16.4s, v0.16b, v4.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v4.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v4.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v4.4b[3]    \n"
                "subs   w4, w4, #1                  \n"
                "sdot   v16.4s, v1.16b, v5.4b[0]    \n"
                "sdot   v17.4s, v1.16b, v5.4b[1]    \n"
                "sdot   v18.4s, v1.16b, v5.4b[2]    \n"
                "sdot   v19.4s, v1.16b, v5.4b[3]    \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "bne    2b                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "uzp1   v0.4s, v24.4s, v25.4s       \n"
                "uzp2   v1.4s, v24.4s, v25.4s       \n"
                "uzp1   v2.4s, v26.4s, v27.4s       \n"
                "uzp2   v3.4s, v26.4s, v27.4s       \n"

                "add    v16.4s, v16.4s, v0.4s       \n"
                "add    v17.4s, v17.4s, v1.4s       \n"
                "add    v18.4s, v18.4s, v2.4s       \n"
                "add    v19.4s, v19.4s, v3.4s       \n"
#endif // __ARM_FEATURE_MATMUL_INT8

                "101:                               \n"
                "and    w4, %w8, #4                 \n" // w4 = remain = max_kk & 4
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                // kk += 4 part
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v2.16b}, [%2], #16         \n"
                "sdot   v16.4s, v0.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v0.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v0.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v0.16b, v2.4b[3]    \n"
#else  // __ARM_FEATURE_DOTPROD
                "lsr    w4, %w8, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.16b}, [%1], #16         \n"
                "ld1    {v4.16b}, [%2], #16         \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "rev64  v1.4s, v0.4s                \n"
                "smull  v9.8h, v1.8b, v4.8b         \n"
                "rev64  v5.8h, v4.8h                \n"
                "smull  v10.8h, v0.8b, v5.8b        \n"
                "smull  v11.8h, v1.8b, v5.8b        \n"
                "smlal2 v8.8h, v0.16b, v4.16b       \n"
                "smlal2 v9.8h, v1.16b, v4.16b       \n"
                "smlal2 v10.8h, v0.16b, v5.16b      \n"
                "smlal2 v11.8h, v1.16b, v5.16b      \n"
                "subs   w4, w4, #1                  \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
                "bne    2b                          \n"
#endif // __ARM_FEATURE_DOTPROD

                "3:                                 \n"
                "and    w4, %w8, #2                 \n" // w4 = remain = max_kk & 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                // kk += 2 part
#if __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v1.8b}, [%2], #8           \n"
                "dup    v4.4h, v1.h[0]              \n"
                "dup    v5.4h, v1.h[1]              \n"
                "dup    v6.4h, v1.h[2]              \n"
                "dup    v7.4h, v1.h[3]              \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "smull  v10.8h, v0.8b, v6.8b        \n"
                "smull  v11.8h, v0.8b, v7.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1], #8           \n"
                "ld1    {v2.8b}, [%2], #8           \n"
                "ext    v1.8b, v0.8b, v0.8b, #4     \n"
                "rev64  v3.4h, v2.4h                \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v1.8b, v2.8b         \n"
                "smull  v10.8h, v0.8b, v3.8b        \n"
                "smull  v11.8h, v1.8b, v3.8b        \n"
                "sadalp v16.4s, v8.8h               \n"
                "sadalp v17.4s, v9.8h               \n"
                "sadalp v18.4s, v10.8h              \n"
                "sadalp v19.4s, v11.8h              \n"
#endif // __ARM_FEATURE_DOTPROD

                "4:                                 \n"
                "and    w4, %w8, #1                 \n" // w4 = remain = max_kk & 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                // kk += 1 part
#if __ARM_FEATURE_DOTPROD
                "ld1r   {v0.2s}, [%1]               \n"
                "ld1r   {v1.2s}, [%2]               \n"
                "add    %1, %1, #4                  \n"
                "add    %2, %2, #4                  \n"
                "zip1   v1.8b, v1.8b, v1.8b         \n"
                "zip1   v2.4h, v1.4h, v1.4h         \n"
                "zip2   v3.4h, v1.4h, v1.4h         \n"
                "smull  v8.8h, v0.8b, v2.8b         \n"
                "smull  v9.8h, v0.8b, v3.8b         \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
#else  // __ARM_FEATURE_DOTPROD
                "ld1    {v0.8b}, [%1]               \n"
                "ld1r   {v4.2s}, [%2]               \n"
                "add    %1, %1, #4                  \n"
                "add    %2, %2, #4                  \n"
                "rev32  v1.4h, v0.4h                \n"
                "zip1   v0.2s, v0.2s, v1.2s         \n"
                "rev32  v5.8b, v4.8b                \n"
                "smull  v8.8h, v0.8b, v4.8b         \n"
                "smull  v9.8h, v0.8b, v5.8b         \n"
                "saddw  v16.4s, v16.4s, v8.4h       \n"
                "saddw2 v17.4s, v17.4s, v8.8h       \n"
                "saddw  v18.4s, v18.4s, v9.4h       \n"
                "saddw2 v19.4s, v19.4s, v9.8h       \n"
#endif // __ARM_FEATURE_DOTPROD

                "5:                                 \n"
                "cmp    %w10, #0                    \n"
                "beq    10f                         \n"

#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                // if out_elempack == 4
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                "zip1   v0.4s, v16.4s, v17.4s       \n"
                "zip2   v1.4s, v16.4s, v17.4s       \n"
                "zip1   v2.4s, v18.4s, v19.4s       \n"
                "zip2   v3.4s, v18.4s, v19.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v1.2d, v3.2d        \n"
                "zip2   v19.2d, v1.2d, v3.2d        \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v18.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v19.4s}, [x4]              \n"
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0
                // if out_elempack == 4
                "cmp    %w11, #1                    \n"
                "beq    8f                          \n"

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                "rev64  v18.4s, v18.4s              \n"
                "rev64  v19.4s, v19.4s              \n"
                "ext    v18.16b, v18.16b, v18.16b, #8 \n"
                "ext    v19.16b, v19.16b, v19.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v19.4s       \n"
                "zip2   v1.4s, v16.4s, v19.4s       \n"
                "zip1   v2.4s, v17.4s, v18.4s       \n"
                "zip2   v3.4s, v17.4s, v18.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v19.4s, v19.4s              \n"

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"

                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                "ext    v17.16b, v17.16b, v17.16b, #8 \n"
                "ext    v19.16b, v19.16b, v19.16b, #8 \n"
                "zip1   v0.4s, v16.4s, v19.4s       \n"
                "zip2   v1.4s, v16.4s, v19.4s       \n"
                "zip1   v2.4s, v17.4s, v18.4s       \n"
                "zip2   v3.4s, v17.4s, v18.4s       \n"
                "zip1   v16.2d, v0.2d, v2.2d        \n"
                "zip2   v17.2d, v0.2d, v2.2d        \n"
                "zip1   v18.2d, v3.2d, v1.2d        \n"
                "zip2   v19.2d, v3.2d, v1.2d        \n"
                "rev64  v17.4s, v17.4s              \n"
                "rev64  v19.4s, v19.4s              \n"

                "add    x4, %3, %12, lsl #2         \n"
                "st1    {v16.4s}, [%3], #16         \n"
                "st1    {v17.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v18.4s}, [x4]              \n"
                "add    x4, x4, %12, lsl #2         \n"
                "st1    {v19.4s}, [x4]              \n"
#endif // __ARM_FEATURE_DOTPROD

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %9, #0              \n"
                "beq        0f                  \n"

                "vldm       %0, {d16-d23}       \n"
                "b          1f                  \n"

                "0:                             \n"
                "veor       q8, q8              \n"
                "veor       q9, q9              \n"
                "veor       q10, q10            \n"
                "veor       q11, q11            \n"

                "1:                             \n"
                "lsr        r4, %8, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                ".align 4                       \n"
                "2:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.s8    {d0-d1}, [%1 :64]!  \n"
                "pld        [%2, #128]          \n"
                "vld1.s8    {d4-d5}, [%2]!      \n"
                "vrev64.32  q1, q0              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vrev64.16  q3, q2              \n"
                "vmull.s8   q5, d2, d4          \n"
                "vmull.s8   q6, d0, d6          \n"
                "vmull.s8   q7, d2, d6          \n"
                "vmlal.s8   q4, d1, d5          \n"
                "vmlal.s8   q5, d3, d5          \n"
                "vmlal.s8   q6, d1, d7          \n"
                "vmlal.s8   q7, d3, d7          \n"
                "subs       r4, r4, #1          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"
                "bne        2b                  \n"

                "3:                             \n"
                "and        r4, %8, #2          \n" // r4 = remain = max_kk & 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                // kk += 2 part
                "vld1.s8    {d0}, [%1 :64]!     \n"
                "vld1.s8    {d4}, [%2]!         \n"
                "vext.8     d1, d0, d0, #4      \n"
                "vrev64.16  d5, d4              \n"
                "vmull.s8   q4, d0, d4          \n"
                "vmull.s8   q5, d1, d4          \n"
                "vmull.s8   q6, d0, d5          \n"
                "vmull.s8   q7, d1, d5          \n"
                "vpadal.s16 q8, q4              \n"
                "vpadal.s16 q9, q5              \n"
                "vpadal.s16 q10, q6             \n"
                "vpadal.s16 q11, q7             \n"

                "4:                             \n"
                "and        r4, %8, #1          \n" // r4 = remain = max_kk & 1
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                // kk += 1 part
                "vld1.s32   {d0[0]}, [%1]!      \n"
                "vld1.s32   {d2[]}, [%2]!       \n"
                "vrev32.16  d1, d0              \n"
                "vrev32.s8  d3, d2              \n"
                "vzip.32    d0, d1              \n"
                "vmull.s8   q4, d0, d2          \n"
                "vmull.s8   q5, d0, d3          \n"
                "vaddw.s16  q8, d8              \n"
                "vaddw.s16  q9, d9              \n"
                "vaddw.s16  q10, d10            \n"
                "vaddw.s16  q11, d11            \n"

                "5:                             \n"
                "cmp        %10, #0             \n"
                "beq        10f                 \n"

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0
                // if out_elempack == 4
                "cmp        %11, #1             \n"
                "beq        8f                  \n"

                // to
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                "vrev64.32  q10, q10            \n"
                "vrev64.32  q11, q11            \n"
                "vext.32    q10, q10, #2        \n"
                "vext.32    q11, q11, #2        \n"
                "vzip.32    q8, q11             \n"
                "vzip.32    q9, q10             \n"
                "vswp       d17, d18            \n"
                "vswp       d21, d22            \n"
                "vrev64.32  q9, q9              \n"
                "vrev64.32  q11, q11            \n"

                "vstm       %3!, {d16-d23}      \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // to
                //      a0 a1 a2 a3
                //      b0 b1 b2 b3
                //      c0 c1 c2 c3
                //      d0 d1 d2 d3
                "vext.32    q9, q9, #2          \n"
                "vext.32    q11, q11, #2        \n"
                "vzip.32    q8, q11             \n"
                "vzip.32    q9, q10             \n"
                "vswp       d17, d18            \n"
                "vswp       d21, d22            \n"
                "vrev64.32  q9, q9              \n"
                "vrev64.32  q11, q11            \n"

                "add        r4, %3, %12, lsl #2 \n"
                "vst1.s32   {d16-d17}, [%3]!    \n"
                "vst1.s32   {d18-d19}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d20-d21}, [r4]     \n"
                "add        r4, r4, %12, lsl #2 \n"
                "vst1.s32   {d22-d23}, [r4]     \n"

                "9:                             \n"
                "add        %0, %0, #64         \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d16-d23}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(max_kk),       // %8
                "r"(k),            // %9
                "r"(k_end),        // %10
                "r"(out_elempack), // %11
                "r"(out_hstep)     // %12
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111 22222222 33333333

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB0);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB0);
                    _sum10 = vmmlaq_s32(_sum10, _pA0, _pB1);
                    _sum11 = vmmlaq_s32(_sum11, _pA1, _pB1);

                    // a0 a1 b0 b1
                    // c0 c1 d0 d1
                    // a2 a3 b2 b3
                    // c2 c3 d2 d3

                    pA += 32;
                    pB += 32;
                }
                int32x4x2_t _ss0 = vuzpq_s32(_sum00, _sum01);
                int32x4x2_t _ss1 = vuzpq_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, _ss0.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss0.val[1]);
                _sum2 = vaddq_s32(_sum2, _ss1.val[0]);
                _sum3 = vaddq_s32(_sum3, _ss1.val[1]);
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB0, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB0, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA0, _pB0, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA0, _pB0, 3);

                _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB1, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB1, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA1, _pB1, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA1, _pB1, 3);

                pA += 32;
                pB += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                _sum0 = vdotq_laneq_s32(_sum0, _pA, _pB, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _pA, _pB, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _pA, _pB, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _pA, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA02 = vld1q_s8(pA);
                int8x16_t _pB02 = vld1q_s8(pB);

                // aabbccdd eeffgghh
                // ccddaabb gghheeff

                int8x16_t _pA13 = vreinterpretq_s8_s32(vrev64q_s32(vreinterpretq_s32_s8(_pA02)));

                // 00112233 44556677
                // 33221100 77665544

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB02));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_pA02), vget_low_s8(_pB13));
                int16x8_t _s3 = vmull_s8(vget_low_s8(_pA13), vget_low_s8(_pB13));

                _s0 = vmlal_s8(_s0, vget_high_s8(_pA02), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA13), vget_high_s8(_pB02));
                _s2 = vmlal_s8(_s2, vget_high_s8(_pA02), vget_high_s8(_pB13));
                _s3 = vmlal_s8(_s3, vget_high_s8(_pA13), vget_high_s8(_pB13));

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                int16x8_t _s2 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 2)));
                int16x8_t _s3 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 3)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vld1_s8(pB);

                // aabbccdd
                // ccddaabb

                int8x8_t _pA1 = vext_s8(_pA0, _pA0, 4);

                // 00112233
                // 33221100

                int8x8_t _pB1 = vreinterpret_s8_s16(vrev64_s16(vreinterpret_s16_s8(_pB0)));

                int16x8_t _s0 = vmull_s8(_pA0, _pB0);
                int16x8_t _s1 = vmull_s8(_pA1, _pB0);
                int16x8_t _s2 = vmull_s8(_pA0, _pB1);
                int16x8_t _s3 = vmull_s8(_pA1, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                _pB = vzip_s8(_pB, _pB).val[0];
                int16x4x2_t _pB0123 = vzip_s16(vreinterpret_s16_s8(_pB), vreinterpret_s16_s8(_pB));

                int16x8_t _s01 = vmull_s8(_pA, vreinterpret_s8_s16(_pB0123.val[0]));
                int16x8_t _s23 = vmull_s8(_pA, vreinterpret_s8_s16(_pB0123.val[1]));
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#else  // __ARM_FEATURE_DOTPROD

                int8x8_t _pA0 = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // abcd.... -> cdab.... -> abcdcdab
                int8x8_t _pA1 = vreinterpret_s8_s16(vrev32_s16(vreinterpret_s16_s8(_pA0)));
                int8x8_t _pA01 = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pA0), vreinterpret_s32_s8(_pA1)).val[0]);

                // 01230123 -> 32103210
                int8x8_t _pB1 = vrev32_s8(_pB0);

                int16x8_t _s01 = vmull_s8(_pA01, _pB0);
                int16x8_t _s23 = vmull_s8(_pA01, _pB1);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s01));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s01));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s23));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s23));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                //      a2 b2 c2 d2
                //      a3 b3 c3 d3
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    {
                        int32x4x2_t _r01 = vzipq_s32(_sum0, _sum1);
                        int32x4x2_t _r23 = vzipq_s32(_sum2, _sum3);
                        _sum0 = vcombine_s32(vget_low_s32(_r01.val[0]), vget_low_s32(_r23.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_r01.val[0]), vget_high_s32(_r23.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_r01.val[1]), vget_low_s32(_r23.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_r01.val[1]), vget_high_s32(_r23.val[1]));
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c2 d3
                //      c0 d1 a2 b3
                //      a3 b2 c1 d0
                //      c3 d2 a1 b0
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    //      a2 b2 c2 d2
                    //      a3 b3 c3 d3
                    {
                        _sum2 = vrev64q_s32(_sum2);
                        _sum3 = vrev64q_s32(_sum3);
                        _sum2 = vextq_s32(_sum2, _sum2, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + 8, _sum2);
                    vst1q_s32(outptr0 + 12, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 a2 a3
                    //      b0 b1 b2 b3
                    //      c0 c1 c2 c3
                    //      d0 d1 d2 d3
                    {
                        _sum1 = vextq_s32(_sum1, _sum1, 2);
                        _sum3 = vextq_s32(_sum3, _sum3, 2);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum3);
                        int32x4x2_t _t1 = vzipq_s32(_sum1, _sum2);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t1.val[0]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t1.val[0]));
                        _sum2 = vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t0.val[1]));
                        _sum3 = vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                        _sum3 = vrev64q_s32(_sum3);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    vst1q_s32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_s32(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000 11111111

                    _sum00 = vmmlaq_s32(_sum00, _pA0, _pB);
                    _sum01 = vmmlaq_s32(_sum01, _pA1, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_laneq_s32(_sum0, _pA0, _pB, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA0, _pB, 1);
                    _sum0 = vdotq_laneq_s32(_sum0, _pA1, _pB, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _pA1, _pB, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 32;
                    pB += 16;
                }
#if __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _ss = vuzpq_s32(_sum00, _sum01);
                _sum0 = vaddq_s32(_sum0, _ss.val[0]);
                _sum1 = vaddq_s32(_sum1, _ss.val[1]);
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum0 = vdotq_lane_s32(_sum0, _pA, _pB, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pA, _pB, 1);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                // aabbccdd eeffgghh

                // 00112233 -> 00110011 22332233
                // 11001100 33223322

                int32x2x2_t _pBB = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));
                int8x16_t _pB02 = vreinterpretq_s8_s32(vcombine_s32(_pBB.val[0], _pBB.val[1]));

                int8x16_t _pB13 = vreinterpretq_s8_s16(vrev64q_s16(vreinterpretq_s16_s8(_pB02)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB02));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB13));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB02));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA), vget_high_s8(_pB13));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);
                // aabbccdd
                // 0011....
                int16x8_t _s0 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 0)));
                int16x8_t _s1 = vmull_s8(_pA, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pB), 1)));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                // aabbccdd

                // 00110011
                // 11001100
                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 2);

                int16x8_t _s0 = vmull_s8(_pA, _pB0);
                int16x8_t _s1 = vmull_s8(_pA, _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcdabcd

                // 01010101 -> 00001111
                _pB = vuzp_s8(_pB, vext_s8(_pB, _pB, 1)).val[0];

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                // abcd abcd

                // 0101 0101 -> 0101 1010

                int8x8_t _pB1 = vext_s8(_pB0, _pB0, 1);
                int8x8_t _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB0), vreinterpret_s32_s8(_pB1)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
#if __ARM_FEATURE_DOTPROD
                // from
                //      a0 b0 c0 d0
                //      a1 b1 c1 d1
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 b0 b1
                    //      c0 c1 d0 d1
                    {
                        int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);
                        _sum0 = _sum01.val[0];
                        _sum1 = _sum01.val[1];
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 2, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    outptr0 += 2;
                }
#else  // __ARM_FEATURE_DOTPROD

                // from
                //      a0 b1 c0 d1
                //      a1 b0 c1 d0
                if (out_elempack == 4)
                {
                    // to
                    //      a0 b0 c0 d0
                    //      a1 b1 c1 d1
                    {
                        _sum1 = vrev64q_s32(_sum1);
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                    }

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    // to
                    //      a0 a1 c0 c1
                    //      b0 b1 d0 d1
                    {
                        int32x4x2_t _t0 = vzipq_s32(_sum0, _sum1);
                        _sum0 = vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t0.val[1]));
                        _sum1 = vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t0.val[1]));
                        _sum1 = vrev64q_s32(_sum1);
                    }

                    vst1_s32(outptr0, vget_low_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep, vget_low_s32(_sum1));
                    vst1_s32(outptr0 + out_hstep * 2, vget_high_s32(_sum0));
                    vst1_s32(outptr0 + out_hstep * 3, vget_high_s32(_sum1));
                    outptr0 += 2;
                }
#endif // __ARM_FEATURE_DOTPROD
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA0 = vld1q_s8(pA);
                    int8x16_t _pA1 = vld1q_s8(pA + 16);
                    int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                    // aaaaaaaa bbbbbbbb cccccccc dddddddd

                    // 00000000

                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _sum01 = vdotq_s32(_sum01, _pA0, _pBB);
                    _sum23 = vdotq_s32(_sum23, _pA1, _pBB);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pA0, _pB, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pA1, _pB, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 32;
                    pB += 8;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum01, _sum23));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum0 = vdotq_lane_s32(_sum0, _pA, _pB, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                int8x8_t _pB1 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB + 2)));

                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), _pB0);
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), _pB1);
                _sum0 = vpadalq_s16(_sum0, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 8;
                pB += 2;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_dup_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_s32(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    outptr0[0] = vgetq_lane_s32(_sum0, 0);
                    outptr0[out_hstep] = vgetq_lane_s32(_sum0, 1);
                    outptr0[out_hstep * 2] = vgetq_lane_s32(_sum0, 2);
                    outptr0[out_hstep * 3] = vgetq_lane_s32(_sum0, 3);
                    outptr0++;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;
            int32x4_t _sum2;
            int32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
                _sum2 = vdupq_n_s32(0);
                _sum3 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
                _sum2 = vld1q_s32(outptr + 8);
                _sum3 = vld1q_s32(outptr + 12);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
                int32x4_t _sum45 = vdupq_n_s32(0);
                int32x4_t _sum67 = vdupq_n_s32(0);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
                int32x2_t _sum20 = vdup_n_s32(0);
                int32x2_t _sum21 = vdup_n_s32(0);
                int32x2_t _sum30 = vdup_n_s32(0);
                int32x2_t _sum31 = vdup_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

#if __ARM_FEATURE_MATMUL_INT8
                    _sum01 = vmmlaq_s32(_sum01, _pA, _pB0);
                    _sum23 = vmmlaq_s32(_sum23, _pA, _pB1);
                    _sum45 = vmmlaq_s32(_sum45, _pA, _pB2);
                    _sum67 = vmmlaq_s32(_sum67, _pA, _pB3);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum00 = vdot_laneq_s32(_sum00, vget_low_s8(_pA), _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_low_s8(_pA), _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_low_s8(_pA), _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_low_s8(_pA), _pB0, 3);
                    _sum20 = vdot_laneq_s32(_sum20, vget_low_s8(_pA), _pB1, 0);
                    _sum21 = vdot_laneq_s32(_sum21, vget_low_s8(_pA), _pB1, 1);
                    _sum30 = vdot_laneq_s32(_sum30, vget_low_s8(_pA), _pB1, 2);
                    _sum31 = vdot_laneq_s32(_sum31, vget_low_s8(_pA), _pB1, 3);
                    _sum00 = vdot_laneq_s32(_sum00, vget_high_s8(_pA), _pB2, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_high_s8(_pA), _pB2, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_high_s8(_pA), _pB2, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_high_s8(_pA), _pB2, 3);
                    _sum20 = vdot_laneq_s32(_sum20, vget_high_s8(_pA), _pB3, 0);
                    _sum21 = vdot_laneq_s32(_sum21, vget_high_s8(_pA), _pB3, 1);
                    _sum30 = vdot_laneq_s32(_sum30, vget_high_s8(_pA), _pB3, 2);
                    _sum31 = vdot_laneq_s32(_sum31, vget_high_s8(_pA), _pB3, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 16;
                    pB += 64;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vcombine_s32(vget_low_s32(_sum01), vget_low_s32(_sum23)));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(vget_low_s32(_sum45), vget_low_s32(_sum67)));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(vget_high_s32(_sum01), vget_high_s32(_sum23)));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(vget_high_s32(_sum45), vget_high_s32(_sum67)));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                int32x2x2_t _sum2x = vzip_s32(_sum20, _sum21);
                int32x2x2_t _sum3x = vzip_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum2x.val[0], _sum3x.val[0]));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(_sum2x.val[1], _sum3x.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_DOTPROD
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
                int32x2_t _sum20 = vdup_n_s32(0);
                int32x2_t _sum21 = vdup_n_s32(0);
                int32x2_t _sum30 = vdup_n_s32(0);
                int32x2_t _sum31 = vdup_n_s32(0);
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                    _sum00 = vdot_laneq_s32(_sum00, _pA, _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, _pA, _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, _pA, _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, _pA, _pB0, 3);
                    _sum20 = vdot_laneq_s32(_sum20, _pA, _pB1, 0);
                    _sum21 = vdot_laneq_s32(_sum21, _pA, _pB1, 1);
                    _sum30 = vdot_laneq_s32(_sum30, _pA, _pB1, 2);
                    _sum31 = vdot_laneq_s32(_sum31, _pA, _pB1, 3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB0));
                    int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB0));
                    _s0 = vmlal_s8(_s0, _pA2, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA2, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB1));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
#endif // __ARM_FEATURE_DOTPROD

                    pA += 8;
                    pB += 32;
                }
#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                int32x2x2_t _sum2x = vzip_s32(_sum20, _sum21);
                int32x2x2_t _sum3x = vzip_s32(_sum30, _sum31);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum2x.val[0], _sum3x.val[0]));
                _sum2 = vaddq_s32(_sum2, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
                _sum3 = vaddq_s32(_sum3, vcombine_s32(_sum2x.val[1], _sum3x.val[1]));
#endif // __ARM_FEATURE_DOTPROD
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vld1_dup_s32((const int*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB));
                int16x8_t _s2 = vmull_s8(_pA1, vget_low_s8(_pB));
                int16x8_t _s3 = vmull_s8(_pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                pA += 4;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vld1_s8(pB);

                int8x8x2_t _pA01 = vuzp_s8(_pA, _pA);

                int16x8_t _s0 = vmull_s8(_pA01.val[0], _pB);
                int16x8_t _s1 = vmull_s8(_pA01.val[1], _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    vst1q_s32(outptr0 + out_hstep, _sum2);
                    vst1q_s32(outptr0 + out_hstep + 4, _sum3);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
                vst1q_s32(outptr + 8, _sum2);
                vst1q_s32(outptr + 12, _sum3);
            }

            outptr += 16;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    _sum01 = vmmlaq_s32(_sum01, _pA, _pB0);
                    _sum23 = vmmlaq_s32(_sum23, _pA, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum00 = vdot_laneq_s32(_sum00, vget_low_s8(_pA), _pB0, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_low_s8(_pA), _pB0, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_low_s8(_pA), _pB0, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_low_s8(_pA), _pB0, 3);
                    _sum00 = vdot_laneq_s32(_sum00, vget_high_s8(_pA), _pB1, 0);
                    _sum01 = vdot_laneq_s32(_sum01, vget_high_s8(_pA), _pB1, 1);
                    _sum10 = vdot_laneq_s32(_sum10, vget_high_s8(_pA), _pB1, 2);
                    _sum11 = vdot_laneq_s32(_sum11, vget_high_s8(_pA), _pB1, 3);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 16;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vcombine_s32(vget_low_s32(_sum01), vget_low_s32(_sum23)));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(vget_high_s32(_sum01), vget_high_s32(_sum23)));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_DOTPROD
                int32x2_t _sum00 = vdup_n_s32(0);
                int32x2_t _sum01 = vdup_n_s32(0);
                int32x2_t _sum10 = vdup_n_s32(0);
                int32x2_t _sum11 = vdup_n_s32(0);
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                    _sum00 = vdot_laneq_s32(_sum00, _pA, _pB, 0);
                    _sum01 = vdot_laneq_s32(_sum01, _pA, _pB, 1);
                    _sum10 = vdot_laneq_s32(_sum10, _pA, _pB, 2);
                    _sum11 = vdot_laneq_s32(_sum11, _pA, _pB, 3);
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                    int16x8_t _s1 = vmull_s8(_pA1, vget_low_s8(_pB));
                    _s0 = vmlal_s8(_s0, _pA2, vget_high_s8(_pB));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                    pA += 8;
                    pB += 16;
                }
#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _sum0x = vzip_s32(_sum00, _sum01);
                int32x2x2_t _sum1x = vzip_s32(_sum10, _sum11);
                _sum0 = vaddq_s32(_sum0, vcombine_s32(_sum0x.val[0], _sum1x.val[0]));
                _sum1 = vaddq_s32(_sum1, vcombine_s32(_sum0x.val[1], _sum1x.val[1]));
#endif // __ARM_FEATURE_DOTPROD
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int16x4_t _pA = vreinterpret_s16_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x4x2_t _pA01 = vuzp_s16(_pA, _pA);
                int8x8_t _pA0 = vreinterpret_s8_s16(_pA01.val[0]);
                int8x8_t _pA1 = vreinterpret_s8_s16(_pA01.val[1]);

                int16x8_t _s0 = vmull_s8(_pA0, _pB);
                int16x8_t _s1 = vmull_s8(_pA1, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];
                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            int32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_s32(0);
            }
            else
            {
                _sum = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;

#if __ARM_FEATURE_DOTPROD
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_MATMUL_INT8
                _sum = vmmlaq_s32(_sum, _pA, _pB);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4x2_t _pAA = vzipq_s32(vreinterpretq_s32_s8(_pA), vreinterpretq_s32_s8(_pA));
                int8x16_t _pA01 = vreinterpretq_s8_s32(_pAA.val[0]);
                int8x16_t _pA23 = vreinterpretq_s8_s32(_pAA.val[1]);
                int8x16_t _pB01 = vcombine_s8(vget_low_s8(_pB), vget_low_s8(_pB));
                int8x16_t _pB23 = vcombine_s8(vget_high_s8(_pB), vget_high_s8(_pB));

                _sum = vdotq_s32(_sum, _pA01, _pB01);
                _sum = vdotq_s32(_sum, _pA23, _pB23);
#endif // __ARM_FEATURE_MATMUL_INT8

                pA += 16;
                pB += 16;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

#if __ARM_FEATURE_DOTPROD
                int32x2x2_t _pAA = vzip_s32(vreinterpret_s32_s8(_pA), vreinterpret_s32_s8(_pA));
                int8x16_t _pA01 = vreinterpretq_s8_s32(vcombine_s32(_pAA.val[0], _pAA.val[1]));

                int8x16_t _pB01 = vcombine_s8(_pB, _pB);

                _sum = vdotq_s32(_sum, _pA01, _pB01);
#else  // __ARM_FEATURE_DOTPROD
                int16x4x2_t _pA01 = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));
                int32x2x2_t _pB01 = vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB));

                int16x8_t _s0 = vmull_s8(vreinterpret_s8_s16(_pA01.val[0]), vreinterpret_s8_s32(_pB01.val[0]));
                _s0 = vmlal_s8(_s0, vreinterpret_s8_s16(_pA01.val[1]), vreinterpret_s8_s32(_pB01.val[1]));
                _sum = vpadalq_s16(_sum, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 8;
                pB += 8;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);
                _pB = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_pB), vreinterpret_s32_s8(_pB)).val[0]);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x8_t _pB = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pB)), 0));

                _pA = vzip_s8(_pA, _pA).val[0];

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vaddw_s16(_sum, vget_low_s16(_s0));

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_s32(outptr0, vget_low_s32(_sum));
                    vst1_s32(outptr0 + out_hstep, vget_high_s32(_sum));
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum);
            }

            outptr += 4;
#else // __ARM_NEON
            int sum00;
            int sum10;
            int sum01;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum10 = 0;
                sum01 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
            for (; kk + 1 < max_kk; kk += 2)
            {
                // fomit-frame-pointer implied in optimized flag spare one register
                // let us stay away from error: ‘asm’ operand has impossible constraints   --- nihui
#if __OPTIMIZE__
                asm volatile(
                    "ldr    r2, [%0], #4    \n" // int8x4_t _pA = *((int8x4_t*)pA); pA += 4;
                    "ldr    r4, [%1], #4    \n" // int8x4_t _pB = *((int8x4_t*)pB); pB += 4;
                    "ror    r3, r2, #8      \n" // int8x4_t _pA_r8 = __ror(_pA, 8);
                    "ror    r5, r4, #8      \n" // int8x4_t _pB_r8 = __ror(_pB, 8);
                    "sxtb16 r2, r2          \n" // int16x2_t _pA0 = __sxtb16(_pA);
                    "sxtb16 r4, r4          \n" // int16x2_t _pA1 = __sxtb16(_pA_r8);
                    "sxtb16 r3, r3          \n" // int16x2_t _pB0 = __sxtb16(_pB);
                    "sxtb16 r5, r5          \n" // int16x2_t _pB1 = __sxtb16(_pB_r8);
                    "smlad  %2, r2, r4, %2  \n" // sum00 = __smlad(_pA0, _pB0, sum00);
                    "smlad  %3, r3, r4, %3  \n" // sum10 = __smlad(_pA1, _pB0, sum10);
                    "smlad  %4, r2, r5, %4  \n" // sum01 = __smlad(_pA0, _pB1, sum01);
                    "smlad  %5, r3, r5, %5  \n" // sum11 = __smlad(_pA1, _pB1, sum11);
                    : "=r"(pA),
                    "=r"(pB),
                    "=r"(sum00),
                    "=r"(sum10),
                    "=r"(sum01),
                    "=r"(sum11)
                    : "0"(pA),
                    "1"(pB),
                    "2"(sum00),
                    "3"(sum10),
                    "4"(sum01),
                    "5"(sum11)
                    : "memory", "r2", "r3", "r4", "r5");
#else
                int _pA0 = *((int*)pA);
                int _pB0 = *((int*)pB);
                int _pA1;
                int _pB1;
                asm volatile("ror %0, %1, #8"
                             : "=r"(_pA1)
                             : "r"(_pA0)
                             :);
                asm volatile("ror %0, %1, #8"
                             : "=r"(_pB1)
                             : "r"(_pB0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pA0)
                             : "0"(_pA0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pA1)
                             : "0"(_pA1)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pB0)
                             : "0"(_pB0)
                             :);
                asm volatile("sxtb16 %0, %0"
                             : "=r"(_pB1)
                             : "0"(_pB1)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum00)
                             : "0"(sum00), "r"(_pA0), "r"(_pB0)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum10)
                             : "0"(sum10), "r"(_pA1), "r"(_pB0)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum01)
                             : "0"(sum01), "r"(_pA0), "r"(_pB1)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum11)
                             : "0"(sum11), "r"(_pA1), "r"(_pB1)
                             :);
                pA += 4;
                pB += 4;
#endif
            }
#endif // __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum10 += pA[1] * pB[0];
                sum01 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum01;
                    outptr0[out_hstep] = sum10;
                    outptr0[out_hstep + 1] = sum11;
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
#endif // __ARM_NEON
        }
        for (; jj < max_jj; jj += 1)
        {
#if __ARM_NEON
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __ARM_NEON
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
#endif // __ARM_NEON

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x8_t _pB = vld1_s8(pB);

                    int8x16_t _pBB = vcombine_s8(_pB, _pB);

                    _sum0 = vdotq_s32(_sum0, _pA, _pBB);

                    pA += 16;
                    pB += 8;
                }
                int32x2_t _ss = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                _sum = vdot_lane_s32(_sum, vget_low_s8(_pA), _pB, 0);
                _sum = vdot_lane_s32(_sum, vget_high_s8(_pA), _pB, 1);

                pA += 16;
                pB += 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));

                _sum = vdot_s32(_sum, _pA, _pB);

                pA += 8;
                pB += 4;
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                    _pB = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pB), vreinterpret_s16_s8(_pB)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 8;
                    pB += 4;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#endif // __ARM_FEATURE_DOTPROD
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[2] * pB[0];
                sum1 += pA[3] * pB[1];
                pA += 4;
                pB += 2;
            }
#endif // __ARM_NEON
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
        int* outptr0 = (int*)top_blob + (i + ii) * out_hstep + j;

        const signed char* pB = pBT;

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0;
            int32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
                _sum1 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
                _sum1 = vld1q_s32(outptr + 4);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _pAA = vcombine_s8(_pA, _pA);
                    _sum00 = vdotq_s32(_sum00, _pAA, _pB0);
                    _sum01 = vdotq_s32(_sum01, _pAA, _pB1);
                    _sum10 = vdotq_s32(_sum10, _pAA, _pB2);
                    _sum11 = vdotq_s32(_sum11, _pAA, _pB3);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _pB1, _pA, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pB2, _pA, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _pB3, _pA, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 8;
                    pB += 64;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum00, _sum01));
                _sum1 = vaddq_s32(_sum1, vpaddq_s32(_sum10, _sum11));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                _sum1 = vdotq_lane_s32(_sum1, _pB1, _pA, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 32;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vld1_dup_s16((const short*)pA));
                int8x16_t _pB = vld1q_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, vget_low_s8(_pB));
                int16x8_t _s1 = vmull_s8(_pA, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);

                pA += 2;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
                vst1q_s32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_s32(0);
            }
            else
            {
                _sum0 = vld1q_s32(outptr);
            }

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_FEATURE_DOTPROD
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x16_t _pAA = vcombine_s8(_pA, _pA);
                    _sum00 = vdotq_s32(_sum00, _pAA, _pB0);
                    _sum01 = vdotq_s32(_sum01, _pAA, _pB1);
#else  // __ARM_FEATURE_MATMUL_INT8
                    _sum0 = vdotq_lane_s32(_sum0, _pB0, _pA, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _pB1, _pA, 1);
#endif // __ARM_FEATURE_MATMUL_INT8

                    pA += 8;
                    pB += 32;
                }
#if __ARM_FEATURE_MATMUL_INT8
                _sum0 = vaddq_s32(_sum0, vpaddq_s32(_sum00, _sum01));
#endif // __ARM_FEATURE_MATMUL_INT8
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                _sum0 = vdotq_lane_s32(_sum0, _pB, _pA, 0);
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB));
                _sum0 = vpadalq_s16(_sum0, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 4;
                pB += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8_t _pA = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(pA)), 0));
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vpadalq_s16(_sum0, _s0);

                pA += 2;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                int8x8_t _pA = vld1_dup_s8(pA);
                int8x8_t _pB = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pB)), 0));

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_s32(outptr0, _sum0);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_s32(outptr, _sum0);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
#if __ARM_NEON
            int32x2_t _sum;

            if (k == 0)
            {
                _sum = vdup_n_s32(0);
            }
            else
            {
                _sum = vld1_s32(outptr);
            }
#else  // __ARM_NEON
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
#endif // __ARM_NEON

            const signed char* pA = pAT;
            int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

                    int8x16_t _pAA = vcombine_s8(_pA, _pA);

                    _sum0 = vdotq_s32(_sum0, _pAA, _pB);

                    pA += 8;
                    pB += 16;
                }
                int32x2_t _ss = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#else  // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

                _sum = vdot_lane_s32(_sum, vget_low_s8(_pB), _pA, 0);
                _sum = vdot_lane_s32(_sum, vget_high_s8(_pB), _pA, 1);

                pA += 8;
                pB += 16;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8_t _pA = vreinterpret_s8_s32(vld1_dup_s32((const int*)pA));
                int8x8_t _pB = vld1_s8(pB);

                _sum = vdot_s32(_sum, _pA, _pB);

                pA += 4;
                pB += 8;
            }
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int8x8_t _pA = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(vld1_s8(pA)), 0));
                    int8x8_t _pB = vld1_s8(pB);

                    _pA = vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA)).val[0]);

                    int16x8_t _s0 = vmull_s8(_pA, _pB);
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 4;
                    pB += 8;
                }
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                _sum = vadd_s32(_sum, _ss);
            }
#endif // __ARM_FEATURE_DOTPROD
            int sum0 = vget_lane_s32(_sum, 0);
            int sum1 = vget_lane_s32(_sum, 1);
            for (; kk + 1 < max_kk; kk += 2)
            {
                sum0 += pA[0] * pB[0];
                sum0 += pA[1] * pB[1];
                sum1 += pA[0] * pB[2];
                sum1 += pA[1] * pB[3];
                pA += 2;
                pB += 4;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);
            for (; kk + 15 < max_kk; kk += 16)
            {
                int8x16_t _pA = vld1q_s8(pA);
                int8x16_t _pB = vld1q_s8(pB);

#if __ARM_FEATURE_DOTPROD
                _sum = vdotq_s32(_sum, _pA, _pB);
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA), vget_low_s8(_pB));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA), vget_high_s8(_pB));
                _sum = vpadalq_s16(_sum, _s0);
#endif // __ARM_FEATURE_DOTPROD

                pA += 16;
                pB += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _pA = vld1_s8(pA);
                int8x8_t _pB = vld1_s8(pB);

                int16x8_t _s0 = vmull_s8(_pA, _pB);
                _sum = vpadalq_s16(_sum, _s0);

                pA += 8;
                pB += 8;
            }
#if __aarch64__
            sum += vaddvq_s32(_sum);
#else
            int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
            _ss = vpadd_s32(_ss, _ss);
            sum += vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_NEON
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

static void convolution_im2col_gemm_get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __ARM_NEON
        int tile_size = (l2_cache_size_int8 - 16) / 8;
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
#endif

#if __ARM_NEON
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __ARM_NEON
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __ARM_NEON
        int nn_M = (M + 31) / 32;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __ARM_NEON
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __ARM_NEON
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __ARM_NEON
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
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
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 4 + TILE_K);
        }

#if __ARM_NEON
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_input_tile_conv1x1s1d1_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
{
    const int elempack = bottom_blob.elempack;

    signed char* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], %4 \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                int8x16_t _r01 = vld1q_s8(p0);
                int8x16_t _r23 = vld1q_s8(p0 + 16);
                int8x16_t _r45 = vld1q_s8(p0 + 32);
                int8x16_t _r67 = vld1q_s8(p0 + 48);
                vst1q_s8(pp, _r01);
                vst1q_s8(pp + 16, _r23);
                vst1q_s8(pp + 32, _r45);
                vst1q_s8(pp + 48, _r67);
                pp += 64;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], %4 \n"
                    "uzp1   v4.4s, v0.4s, v1.4s         \n"
                    "uzp2   v6.4s, v0.4s, v1.4s         \n"
                    "uzp1   v5.4s, v2.4s, v3.4s         \n"
                    "uzp2   v7.4s, v2.4s, v3.4s         \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // NCNN_GNU_INLINE_ASM
                int32x4x2_t _r0246 = vld2q_s32((const int*)p0);
                int32x4x2_t _r1357 = vld2q_s32((const int*)(p0 + 32));
                vst1q_s32((int*)pp, _r0246.val[0]);
                vst1q_s32((int*)(pp + 16), _r1357.val[0]);
                vst1q_s32((int*)(pp + 32), _r0246.val[1]);
                vst1q_s32((int*)(pp + 48), _r1357.val[1]);
                pp += 64;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], %4 \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                int16x8x4_t _r0 = vld4q_s16((const short*)p0);
                vst1q_s16((short*)pp, _r0.val[0]);
                vst1q_s16((short*)(pp + 16), _r0.val[1]);
                vst1q_s16((short*)(pp + 32), _r0.val[2]);
                vst1q_s16((short*)(pp + 48), _r0.val[3]);
                pp += 64;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v0.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v1.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v0.d}[1], [%0], %4         \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v1.d}[1], [%0], %4         \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v2.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v3.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v2.d}[1], [%0], %4         \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v3.d}[1], [%0], %4         \n"
                    "zip1   v4.16b, v0.16b, v1.16b      \n"
                    "zip2   v5.16b, v0.16b, v1.16b      \n"
                    "zip1   v6.16b, v2.16b, v3.16b      \n"
                    "zip2   v7.16b, v2.16b, v3.16b      \n"
                    "st4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else  // NCNN_GNU_INLINE_ASM
                int8x8_t _r0 = vld1_s8(p0);
                int8x8_t _r1 = vld1_s8(p0 + cstep);
                int8x8_t _r2 = vld1_s8(p0 + cstep * 2);
                int8x8_t _r3 = vld1_s8(p0 + cstep * 3);
                int8x8_t _r4 = vld1_s8(p0 + cstep * 4);
                int8x8_t _r5 = vld1_s8(p0 + cstep * 5);
                int8x8_t _r6 = vld1_s8(p0 + cstep * 6);
                int8x8_t _r7 = vld1_s8(p0 + cstep * 7);
                // save as transpose8x8
                int8x8x2_t _r01 = vzip_s8(_r0, _r1);
                int8x8x2_t _r23 = vzip_s8(_r2, _r3);
                int8x8x2_t _r45 = vzip_s8(_r4, _r5);
                int8x8x2_t _r67 = vzip_s8(_r6, _r7);
                int16x8x4_t _r0246;
                _r0246.val[0] = vreinterpretq_s16_s8(vcombine_s8(_r01.val[0], _r01.val[1]));
                _r0246.val[1] = vreinterpretq_s16_s8(vcombine_s8(_r23.val[0], _r23.val[1]));
                _r0246.val[2] = vreinterpretq_s16_s8(vcombine_s8(_r45.val[0], _r45.val[1]));
                _r0246.val[3] = vreinterpretq_s16_s8(vcombine_s8(_r67.val[0], _r67.val[1]));
                vst4q_s16((short*)pp, _r0246);
                pp += 64;
                p0 += cstep * 8;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v0.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v1.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v2.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v3.8b}, [%0], %4           \n"
                    "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%1], #32 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                int8x8x4_t _r0123;
                _r0123.val[0] = vld1_s8(p0);
                _r0123.val[1] = vld1_s8(p0 + cstep);
                _r0123.val[2] = vld1_s8(p0 + cstep * 2);
                _r0123.val[3] = vld1_s8(p0 + cstep * 3);
                vst4_s8(pp, _r0123);
                pp += 32;
                p0 += cstep * 4;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v0.8b}, [%0], %4           \n"
                    "prfm   pldl1keep, [%0, #64]        \n"
                    "ld1    {v1.8b}, [%0], %4           \n"
                    "st2    {v0.8b, v1.8b}, [%1], #16   \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1");
#else  // NCNN_GNU_INLINE_ASM
                int8x8x2_t _r01;
                _r01.val[0] = vld1_s8(p0);
                _r01.val[1] = vld1_s8(p0 + cstep);
                vst2_s8(pp, _r01);
                pp += 16;
                p0 += cstep * 2;
#endif // NCNN_GNU_INLINE_ASM
            }
            for (; kk < max_kk; kk++)
            {
                vst1_s8(pp, vld1_s8(p0));
                pp += 8;
                p0 += cstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.16b, v1.16b}, [%0], %4  \n"
                    "st1    {v0.16b, v1.16b}, [%1], #32 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1");
#else  // NCNN_GNU_INLINE_ASM
                int8x16_t _r01 = vld1q_s8(p0);
                int8x16_t _r23 = vld1q_s8(p0 + 16);
                vst1q_s8(pp, _r01);
                vst1q_s8(pp + 16, _r23);
                pp += 32;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%0], %4 \n"
                    "st4    {v0.2s, v1.2s, v2.2s, v3.2s}, [%1], #32 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3");
#else  // NCNN_GNU_INLINE_ASM
                int32x2x4_t _r0123;
                _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(p0));
                _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(p0 + 8));
                _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(p0 + 16));
                _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(p0 + 24));
                vst4_s32((int*)pp, _r0123);
                pp += 32;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%0], %4 \n"
                    "st4    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1", "v2", "v3");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.s8    {d0-d3}, [%0], %4   \n"
                    "vst4.s16   {d0-d3}, [%1 :64]!  \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "q0", "q1");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                int16x4x4_t _r0123;
                _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
                _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(p0 + 8));
                _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(p0 + 16));
                _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(p0 + 24));
                vst4_s16((short*)pp, _r0123);
                pp += 32;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        }

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep + 0];
                pp[2] = p0[cstep * 2 + 0];
                pp[3] = p0[cstep * 3 + 0];
                pp[4] = p0[cstep * 4 + 0];
                pp[5] = p0[cstep * 5 + 0];
                pp[6] = p0[cstep * 6 + 0];
                pp[7] = p0[cstep * 7 + 0];
                pp[8] = p0[1];
                pp[9] = p0[cstep + 1];
                pp[10] = p0[cstep * 2 + 1];
                pp[11] = p0[cstep * 3 + 1];
                pp[12] = p0[cstep * 4 + 1];
                pp[13] = p0[cstep * 5 + 1];
                pp[14] = p0[cstep * 6 + 1];
                pp[15] = p0[cstep * 7 + 1];
                pp[16] = p0[2];
                pp[17] = p0[cstep + 2];
                pp[18] = p0[cstep * 2 + 2];
                pp[19] = p0[cstep * 3 + 2];
                pp[20] = p0[cstep * 4 + 2];
                pp[21] = p0[cstep * 5 + 2];
                pp[22] = p0[cstep * 6 + 2];
                pp[23] = p0[cstep * 7 + 2];
                pp[24] = p0[3];
                pp[25] = p0[cstep + 3];
                pp[26] = p0[cstep * 2 + 3];
                pp[27] = p0[cstep * 3 + 3];
                pp[28] = p0[cstep * 4 + 3];
                pp[29] = p0[cstep * 5 + 3];
                pp[30] = p0[cstep * 6 + 3];
                pp[31] = p0[cstep * 7 + 3];
                pp += 32;
                p0 += cstep * 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep + 0];
                pp[2] = p0[cstep * 2 + 0];
                pp[3] = p0[cstep * 3 + 0];
                pp[4] = p0[1];
                pp[5] = p0[cstep + 1];
                pp[6] = p0[cstep * 2 + 1];
                pp[7] = p0[cstep * 3 + 1];
                pp[8] = p0[2];
                pp[9] = p0[cstep + 2];
                pp[10] = p0[cstep * 2 + 2];
                pp[11] = p0[cstep * 3 + 2];
                pp[12] = p0[3];
                pp[13] = p0[cstep + 3];
                pp[14] = p0[cstep * 2 + 3];
                pp[15] = p0[cstep * 3 + 3];
                pp += 16;
                p0 += cstep * 4;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep + 0];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp[4] = p0[2];
                pp[5] = p0[cstep + 2];
                pp[6] = p0[3];
                pp[7] = p0[cstep + 3];
                pp += 8;
                p0 += cstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += cstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v0.16b}, [%0], %4          \n"
                    "st1    {v0.16b}, [%1], #16         \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0");
#else  // NCNN_GNU_INLINE_ASM
                vst1q_s8(pp, vld1q_s8(p0));
                pp += 16;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#elif __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v0.8b, v1.8b}, [%0], %4    \n"
                    "st2    {v0.2s, v1.2s}, [%1], #16   \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1");
#else  // NCNN_GNU_INLINE_ASM
                int32x2x2_t _r01;
                _r01.val[0] = vreinterpret_s32_s8(vld1_s8(p0));
                _r01.val[1] = vreinterpret_s32_s8(vld1_s8(p0 + 8));
                vst2_s32((int*)pp, _r01);
                pp += 16;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
            for (; kk < max_kk / 8; kk++)
            {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]       \n"
                    "ld1    {v0.8b, v1.8b}, [%0], %4    \n"
                    "st2    {v0.4h, v1.4h}, [%1], #16   \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "v0", "v1");
#else
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.s8    {d0-d1}, [%0], %4   \n"
                    "vst2.s16   {d0-d1}, [%1 :64]!  \n"
                    : "=r"(p0), // %0
                    "=r"(pp)  // %1
                    : "0"(p0),
                    "1"(pp),
                    "r"(cstep)
                    : "memory", "q0");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                int16x4x2_t _r01;
                _r01.val[0] = vreinterpret_s16_s8(vld1_s8(p0));
                _r01.val[1] = vreinterpret_s16_s8(vld1_s8(p0 + 8));
                vst2_s16((short*)pp, _r01);
                pp += 16;
                p0 += cstep;
#endif // NCNN_GNU_INLINE_ASM
            }
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[cstep * 2];
                pp[3] = p0[cstep * 3];
                pp[4] = p0[cstep * 4];
                pp[5] = p0[cstep * 5];
                pp[6] = p0[cstep * 6];
                pp[7] = p0[cstep * 7];
                pp[8] = p0[1];
                pp[9] = p0[cstep + 1];
                pp[10] = p0[cstep * 2 + 1];
                pp[11] = p0[cstep * 3 + 1];
                pp[12] = p0[cstep * 4 + 1];
                pp[13] = p0[cstep * 5 + 1];
                pp[14] = p0[cstep * 6 + 1];
                pp[15] = p0[cstep * 7 + 1];
                pp += 16;
                p0 += cstep * 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[cstep * 2];
                pp[3] = p0[cstep * 3];
                pp[4] = p0[1];
                pp[5] = p0[cstep + 1];
                pp[6] = p0[cstep * 2 + 1];
                pp[7] = p0[cstep * 3 + 1];
                pp += 8;
                p0 += cstep * 4;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[cstep];
                pp[2] = p0[1];
                pp[3] = p0[cstep + 1];
                pp += 4;
                p0 += cstep * 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += cstep;
            }
        }
    }
    for (; jj < max_jj; jj++)
    {
#if __ARM_NEON
        if (elempack == 8)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k / 8) + (j + jj) * 8;
            const size_t cstep = bottom_blob.cstep * 8;

            int kk = 0;
            for (; kk < max_kk / 8; kk++)
            {
                vst1_s8(pp, vld1_s8(p0));
                pp += 8;
                p0 += cstep;
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            const signed char* p0 = (const signed char*)bottom_blob.channel(k) + (j + jj);
            const size_t cstep = bottom_blob.cstep;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += cstep;
            }
        }
    }
}

template<int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h>
#if __ARM_FEATURE_MATMUL_INT8
void convolution_im2col_input_tile_int8_i8mm(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#elif __ARM_FEATURE_DOTPROD
void convolution_im2col_input_tile_int8_asimddp(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk)
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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

    signed char* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dy4 = (j + jj + 4) / outw * stride_h;
        int dy5 = (j + jj + 5) / outw * stride_h;
        int dy6 = (j + jj + 6) / outw * stride_h;
        int dy7 = (j + jj + 7) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;
        int dx4 = (j + jj + 4) % outw * stride_w;
        int dx5 = (j + jj + 5) % outw * stride_w;
        int dx6 = (j + jj + 6) % outw * stride_w;
        int dx7 = (j + jj + 7) % outw * stride_w;

        if (dy0 == dy7)
        {
            int kk = 0;
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        // save as transpose8x8
                        int8x8x2_t _r01 = vzip_s8(_r0, _r1);
                        int8x8x2_t _r23 = vzip_s8(_r2, _r3);
                        int8x8x2_t _r45 = vzip_s8(_r4, _r5);
                        int8x8x2_t _r67 = vzip_s8(_r6, _r7);
                        int16x8x4_t _r0246;
                        _r0246.val[0] = vreinterpretq_s16_s8(vcombine_s8(_r01.val[0], _r01.val[1]));
                        _r0246.val[1] = vreinterpretq_s16_s8(vcombine_s8(_r23.val[0], _r23.val[1]));
                        _r0246.val[2] = vreinterpretq_s16_s8(vcombine_s8(_r45.val[0], _r45.val[1]));
                        _r0246.val[3] = vreinterpretq_s16_s8(vcombine_s8(_r67.val[0], _r67.val[1]));
                        vst4q_s16((short*)pp, _r0246);
                        pp += 64;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r2 = vld1q_s8(sptr2);
                        int8x16_t _r3 = vld1q_s8(sptr3);
                        int8x16_t _r4 = vld1q_s8(sptr4);
                        int8x16_t _r5 = vld1q_s8(sptr5);
                        int8x16_t _r6 = vld1q_s8(sptr6);
                        int8x16_t _r7 = vld1q_s8(sptr7);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        int8x16_t _r23 = vtrnq_s8(_r2, _r3).val[0];
                        int8x16_t _r45 = vtrnq_s8(_r4, _r5).val[0];
                        int8x16_t _r67 = vtrnq_s8(_r6, _r7).val[0];
                        int16x8x4_t _r0123;
                        _r0123.val[0] = vreinterpretq_s16_s8(_r01);
                        _r0123.val[1] = vreinterpretq_s16_s8(_r23);
                        _r0123.val[2] = vreinterpretq_s16_s8(_r45);
                        _r0123.val[3] = vreinterpretq_s16_s8(_r67);
                        vst4q_s16((short*)pp, _r0123);
                        pp += 64;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr4[0];
                        pp[5] = sptr5[0];
                        pp[6] = sptr6[0];
                        pp[7] = sptr7[0];
                        pp[8] = sptr0[stride_w];
                        pp[9] = sptr1[stride_w];
                        pp[10] = sptr2[stride_w];
                        pp[11] = sptr3[stride_w];
                        pp[12] = sptr4[stride_w];
                        pp[13] = sptr5[stride_w];
                        pp[14] = sptr6[stride_w];
                        pp[15] = sptr7[stride_w];
                        pp[16] = sptr0[stride_w * 2];
                        pp[17] = sptr1[stride_w * 2];
                        pp[18] = sptr2[stride_w * 2];
                        pp[19] = sptr3[stride_w * 2];
                        pp[20] = sptr4[stride_w * 2];
                        pp[21] = sptr5[stride_w * 2];
                        pp[22] = sptr6[stride_w * 2];
                        pp[23] = sptr7[stride_w * 2];
                        pp[24] = sptr0[stride_w * 3];
                        pp[25] = sptr1[stride_w * 3];
                        pp[26] = sptr2[stride_w * 3];
                        pp[27] = sptr3[stride_w * 3];
                        pp[28] = sptr4[stride_w * 3];
                        pp[29] = sptr5[stride_w * 3];
                        pp[30] = sptr6[stride_w * 3];
                        pp[31] = sptr7[stride_w * 3];
                        pp[32] = sptr0[stride_w * 4];
                        pp[33] = sptr1[stride_w * 4];
                        pp[34] = sptr2[stride_w * 4];
                        pp[35] = sptr3[stride_w * 4];
                        pp[36] = sptr4[stride_w * 4];
                        pp[37] = sptr5[stride_w * 4];
                        pp[38] = sptr6[stride_w * 4];
                        pp[39] = sptr7[stride_w * 4];
                        pp[40] = sptr0[stride_w * 5];
                        pp[41] = sptr1[stride_w * 5];
                        pp[42] = sptr2[stride_w * 5];
                        pp[43] = sptr3[stride_w * 5];
                        pp[44] = sptr4[stride_w * 5];
                        pp[45] = sptr5[stride_w * 5];
                        pp[46] = sptr6[stride_w * 5];
                        pp[47] = sptr7[stride_w * 5];
                        pp[48] = sptr0[stride_w * 6];
                        pp[49] = sptr1[stride_w * 6];
                        pp[50] = sptr2[stride_w * 6];
                        pp[51] = sptr3[stride_w * 6];
                        pp[52] = sptr4[stride_w * 6];
                        pp[53] = sptr5[stride_w * 6];
                        pp[54] = sptr6[stride_w * 6];
                        pp[55] = sptr7[stride_w * 6];
                        pp[56] = sptr0[stride_w * 7];
                        pp[57] = sptr1[stride_w * 7];
                        pp[58] = sptr2[stride_w * 7];
                        pp[59] = sptr3[stride_w * 7];
                        pp[60] = sptr4[stride_w * 7];
                        pp[61] = sptr5[stride_w * 7];
                        pp[62] = sptr6[stride_w * 7];
                        pp[63] = sptr7[stride_w * 7];
                        pp += 64;
                    }
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    if (stride_w == 1)
                    {
                        int8x8x4_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        _r01.val[2] = vld1_s8(sptr2);
                        _r01.val[3] = vld1_s8(sptr3);
                        vst4_s8(pp, _r01);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r2 = vld1q_s8(sptr2);
                        int8x16_t _r3 = vld1q_s8(sptr3);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        int8x16_t _r23 = vtrnq_s8(_r2, _r3).val[0];
                        int16x8x2_t _r0123;
                        _r0123.val[0] = vreinterpretq_s16_s8(_r01);
                        _r0123.val[1] = vreinterpretq_s16_s8(_r23);
                        vst2q_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr0[stride_w];
                        pp[5] = sptr1[stride_w];
                        pp[6] = sptr2[stride_w];
                        pp[7] = sptr3[stride_w];
                        pp[8] = sptr0[stride_w * 2];
                        pp[9] = sptr1[stride_w * 2];
                        pp[10] = sptr2[stride_w * 2];
                        pp[11] = sptr3[stride_w * 2];
                        pp[12] = sptr0[stride_w * 3];
                        pp[13] = sptr1[stride_w * 3];
                        pp[14] = sptr2[stride_w * 3];
                        pp[15] = sptr3[stride_w * 3];
                        pp[16] = sptr0[stride_w * 4];
                        pp[17] = sptr1[stride_w * 4];
                        pp[18] = sptr2[stride_w * 4];
                        pp[19] = sptr3[stride_w * 4];
                        pp[20] = sptr0[stride_w * 5];
                        pp[21] = sptr1[stride_w * 5];
                        pp[22] = sptr2[stride_w * 5];
                        pp[23] = sptr3[stride_w * 5];
                        pp[24] = sptr0[stride_w * 6];
                        pp[25] = sptr1[stride_w * 6];
                        pp[26] = sptr2[stride_w * 6];
                        pp[27] = sptr3[stride_w * 6];
                        pp[28] = sptr0[stride_w * 7];
                        pp[29] = sptr1[stride_w * 7];
                        pp[30] = sptr2[stride_w * 7];
                        pp[31] = sptr3[stride_w * 7];
                        pp += 32;
                    }
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8x2_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        vst2_s8(pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        vst1q_s8(pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    int8x8_t _r2 = vld1_s8(sptr + stride_w * 16);
                    int8x8_t _r3 = vld1_s8(sptr + stride_w * 24);
                    int8x8_t _r4 = vld1_s8(sptr + stride_w * 32);
                    int8x8_t _r5 = vld1_s8(sptr + stride_w * 40);
                    int8x8_t _r6 = vld1_s8(sptr + stride_w * 48);
                    int8x8_t _r7 = vld1_s8(sptr + stride_w * 56);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    vst1_s8(pp + 32, _r4);
                    vst1_s8(pp + 40, _r5);
                    vst1_s8(pp + 48, _r6);
                    vst1_s8(pp + 56, _r7);
                    pp += 64;
#elif __ARM_FEATURE_DOTPROD
                    int32x2_t _r0 = vreinterpret_s32_s8(vld1_s8(sptr));
                    int32x2_t _r1 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    int32x2_t _r2 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 16));
                    int32x2_t _r3 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 24));
                    int32x2_t _r4 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 32));
                    int32x2_t _r5 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 40));
                    int32x2_t _r6 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 48));
                    int32x2_t _r7 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 56));
                    int32x2x2_t _r01 = vzip_s32(_r0, _r1);
                    int32x2x2_t _r23 = vzip_s32(_r2, _r3);
                    int32x2x2_t _r45 = vzip_s32(_r4, _r5);
                    int32x2x2_t _r67 = vzip_s32(_r6, _r7);
                    vst1_s32((int*)pp, _r01.val[0]);
                    vst1_s32((int*)(pp + 8), _r23.val[0]);
                    vst1_s32((int*)(pp + 16), _r45.val[0]);
                    vst1_s32((int*)(pp + 24), _r67.val[0]);
                    vst1_s32((int*)(pp + 32), _r01.val[1]);
                    vst1_s32((int*)(pp + 40), _r23.val[1]);
                    vst1_s32((int*)(pp + 48), _r45.val[1]);
                    vst1_s32((int*)(pp + 56), _r67.val[1]);
                    pp += 64;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 32));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 40));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 48));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 56));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int x24 = dx4 + dilation_w * v2;
                    int x25 = dx5 + dilation_w * v2;
                    int x26 = dx6 + dilation_w * v2;
                    int x27 = dx7 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;
                    int y24 = dy4 + dilation_h * u2;
                    int y25 = dy5 + dilation_h * u2;
                    int y26 = dy6 + dilation_h * u2;
                    int y27 = dy7 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int x34 = dx4 + dilation_w * v3;
                    int x35 = dx5 + dilation_w * v3;
                    int x36 = dx6 + dilation_w * v3;
                    int x37 = dx7 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;
                    int y34 = dy4 + dilation_h * u3;
                    int y35 = dy5 + dilation_h * u3;
                    int y36 = dy6 + dilation_h * u3;
                    int y37 = dy7 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int x42 = dx2 + dilation_w * v4;
                    int x43 = dx3 + dilation_w * v4;
                    int x44 = dx4 + dilation_w * v4;
                    int x45 = dx5 + dilation_w * v4;
                    int x46 = dx6 + dilation_w * v4;
                    int x47 = dx7 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int y42 = dy2 + dilation_h * u4;
                    int y43 = dy3 + dilation_h * u4;
                    int y44 = dy4 + dilation_h * u4;
                    int y45 = dy5 + dilation_h * u4;
                    int y46 = dy6 + dilation_h * u4;
                    int y47 = dy7 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int x52 = dx2 + dilation_w * v5;
                    int x53 = dx3 + dilation_w * v5;
                    int x54 = dx4 + dilation_w * v5;
                    int x55 = dx5 + dilation_w * v5;
                    int x56 = dx6 + dilation_w * v5;
                    int x57 = dx7 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;
                    int y52 = dy2 + dilation_h * u5;
                    int y53 = dy3 + dilation_h * u5;
                    int y54 = dy4 + dilation_h * u5;
                    int y55 = dy5 + dilation_h * u5;
                    int y56 = dy6 + dilation_h * u5;
                    int y57 = dy7 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int x62 = dx2 + dilation_w * v6;
                    int x63 = dx3 + dilation_w * v6;
                    int x64 = dx4 + dilation_w * v6;
                    int x65 = dx5 + dilation_w * v6;
                    int x66 = dx6 + dilation_w * v6;
                    int x67 = dx7 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int y62 = dy2 + dilation_h * u6;
                    int y63 = dy3 + dilation_h * u6;
                    int y64 = dy4 + dilation_h * u6;
                    int y65 = dy5 + dilation_h * u6;
                    int y66 = dy6 + dilation_h * u6;
                    int y67 = dy7 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int x72 = dx2 + dilation_w * v7;
                    int x73 = dx3 + dilation_w * v7;
                    int x74 = dx4 + dilation_w * v7;
                    int x75 = dx5 + dilation_w * v7;
                    int x76 = dx6 + dilation_w * v7;
                    int x77 = dx7 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;
                    int y72 = dy2 + dilation_h * u7;
                    int y73 = dy3 + dilation_h * u7;
                    int y74 = dy4 + dilation_h * u7;
                    int y75 = dy5 + dilation_h * u7;
                    int y76 = dy6 + dilation_h * u7;
                    int y77 = dy7 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;
                    const signed char* sptr24 = img2.row<const signed char>(y24) + x24;
                    const signed char* sptr25 = img2.row<const signed char>(y25) + x25;
                    const signed char* sptr26 = img2.row<const signed char>(y26) + x26;
                    const signed char* sptr27 = img2.row<const signed char>(y27) + x27;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;
                    const signed char* sptr34 = img3.row<const signed char>(y34) + x34;
                    const signed char* sptr35 = img3.row<const signed char>(y35) + x35;
                    const signed char* sptr36 = img3.row<const signed char>(y36) + x36;
                    const signed char* sptr37 = img3.row<const signed char>(y37) + x37;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr42 = img4.row<const signed char>(y42) + x42;
                    const signed char* sptr43 = img4.row<const signed char>(y43) + x43;
                    const signed char* sptr44 = img4.row<const signed char>(y44) + x44;
                    const signed char* sptr45 = img4.row<const signed char>(y45) + x45;
                    const signed char* sptr46 = img4.row<const signed char>(y46) + x46;
                    const signed char* sptr47 = img4.row<const signed char>(y47) + x47;

                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr52 = img5.row<const signed char>(y52) + x52;
                    const signed char* sptr53 = img5.row<const signed char>(y53) + x53;
                    const signed char* sptr54 = img5.row<const signed char>(y54) + x54;
                    const signed char* sptr55 = img5.row<const signed char>(y55) + x55;
                    const signed char* sptr56 = img5.row<const signed char>(y56) + x56;
                    const signed char* sptr57 = img5.row<const signed char>(y57) + x57;

                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr62 = img6.row<const signed char>(y62) + x62;
                    const signed char* sptr63 = img6.row<const signed char>(y63) + x63;
                    const signed char* sptr64 = img6.row<const signed char>(y64) + x64;
                    const signed char* sptr65 = img6.row<const signed char>(y65) + x65;
                    const signed char* sptr66 = img6.row<const signed char>(y66) + x66;
                    const signed char* sptr67 = img6.row<const signed char>(y67) + x67;

                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;
                    const signed char* sptr72 = img7.row<const signed char>(y72) + x72;
                    const signed char* sptr73 = img7.row<const signed char>(y73) + x73;
                    const signed char* sptr74 = img7.row<const signed char>(y74) + x74;
                    const signed char* sptr75 = img7.row<const signed char>(y75) + x75;
                    const signed char* sptr76 = img7.row<const signed char>(y76) + x76;
                    const signed char* sptr77 = img7.row<const signed char>(y77) + x77;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp[16] = sptr02[0];
                    pp[17] = sptr12[0];
                    pp[18] = sptr22[0];
                    pp[19] = sptr32[0];
                    pp[20] = sptr42[0];
                    pp[21] = sptr52[0];
                    pp[22] = sptr62[0];
                    pp[23] = sptr72[0];
                    pp[24] = sptr03[0];
                    pp[25] = sptr13[0];
                    pp[26] = sptr23[0];
                    pp[27] = sptr33[0];
                    pp[28] = sptr43[0];
                    pp[29] = sptr53[0];
                    pp[30] = sptr63[0];
                    pp[31] = sptr73[0];
                    pp[32] = sptr04[0];
                    pp[33] = sptr14[0];
                    pp[34] = sptr24[0];
                    pp[35] = sptr34[0];
                    pp[36] = sptr44[0];
                    pp[37] = sptr54[0];
                    pp[38] = sptr64[0];
                    pp[39] = sptr74[0];
                    pp[40] = sptr05[0];
                    pp[41] = sptr15[0];
                    pp[42] = sptr25[0];
                    pp[43] = sptr35[0];
                    pp[44] = sptr45[0];
                    pp[45] = sptr55[0];
                    pp[46] = sptr65[0];
                    pp[47] = sptr75[0];
                    pp[48] = sptr06[0];
                    pp[49] = sptr16[0];
                    pp[50] = sptr26[0];
                    pp[51] = sptr36[0];
                    pp[52] = sptr46[0];
                    pp[53] = sptr56[0];
                    pp[54] = sptr66[0];
                    pp[55] = sptr76[0];
                    pp[56] = sptr07[0];
                    pp[57] = sptr17[0];
                    pp[58] = sptr27[0];
                    pp[59] = sptr37[0];
                    pp[60] = sptr47[0];
                    pp[61] = sptr57[0];
                    pp[62] = sptr67[0];
                    pp[63] = sptr77[0];
                    pp += 64;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int x24 = dx4 + dilation_w * v2;
                    int x25 = dx5 + dilation_w * v2;
                    int x26 = dx6 + dilation_w * v2;
                    int x27 = dx7 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;
                    int y24 = dy4 + dilation_h * u2;
                    int y25 = dy5 + dilation_h * u2;
                    int y26 = dy6 + dilation_h * u2;
                    int y27 = dy7 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int x34 = dx4 + dilation_w * v3;
                    int x35 = dx5 + dilation_w * v3;
                    int x36 = dx6 + dilation_w * v3;
                    int x37 = dx7 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;
                    int y34 = dy4 + dilation_h * u3;
                    int y35 = dy5 + dilation_h * u3;
                    int y36 = dy6 + dilation_h * u3;
                    int y37 = dy7 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;
                    const signed char* sptr24 = img2.row<const signed char>(y24) + x24;
                    const signed char* sptr25 = img2.row<const signed char>(y25) + x25;
                    const signed char* sptr26 = img2.row<const signed char>(y26) + x26;
                    const signed char* sptr27 = img2.row<const signed char>(y27) + x27;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;
                    const signed char* sptr34 = img3.row<const signed char>(y34) + x34;
                    const signed char* sptr35 = img3.row<const signed char>(y35) + x35;
                    const signed char* sptr36 = img3.row<const signed char>(y36) + x36;
                    const signed char* sptr37 = img3.row<const signed char>(y37) + x37;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp[8] = sptr02[0];
                    pp[9] = sptr12[0];
                    pp[10] = sptr22[0];
                    pp[11] = sptr32[0];
                    pp[12] = sptr03[0];
                    pp[13] = sptr13[0];
                    pp[14] = sptr23[0];
                    pp[15] = sptr33[0];
                    pp[16] = sptr04[0];
                    pp[17] = sptr14[0];
                    pp[18] = sptr24[0];
                    pp[19] = sptr34[0];
                    pp[20] = sptr05[0];
                    pp[21] = sptr15[0];
                    pp[22] = sptr25[0];
                    pp[23] = sptr35[0];
                    pp[24] = sptr06[0];
                    pp[25] = sptr16[0];
                    pp[26] = sptr26[0];
                    pp[27] = sptr36[0];
                    pp[28] = sptr07[0];
                    pp[29] = sptr17[0];
                    pp[30] = sptr27[0];
                    pp[31] = sptr37[0];
                    pp += 32;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp[8] = sptr04[0];
                    pp[9] = sptr14[0];
                    pp[10] = sptr05[0];
                    pp[11] = sptr15[0];
                    pp[12] = sptr06[0];
                    pp[13] = sptr16[0];
                    pp[14] = sptr07[0];
                    pp[15] = sptr17[0];
                    pp += 16;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int x4 = dx4 + dilation_w * v;
                int x5 = dx5 + dilation_w * v;
                int x6 = dx6 + dilation_w * v;
                int x7 = dx7 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;
                int y4 = dy4 + dilation_h * u;
                int y5 = dy5 + dilation_h * u;
                int y6 = dy6 + dilation_h * u;
                int y7 = dy7 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    int8x8_t _r2 = vld1_s8(sptr2);
                    int8x8_t _r3 = vld1_s8(sptr3);
                    int8x8_t _r4 = vld1_s8(sptr4);
                    int8x8_t _r5 = vld1_s8(sptr5);
                    int8x8_t _r6 = vld1_s8(sptr6);
                    int8x8_t _r7 = vld1_s8(sptr7);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    vst1_s8(pp + 32, _r4);
                    vst1_s8(pp + 40, _r5);
                    vst1_s8(pp + 48, _r6);
                    vst1_s8(pp + 56, _r7);
                    pp += 64;
#elif __ARM_FEATURE_DOTPROD
                    int32x2_t _r0 = vreinterpret_s32_s8(vld1_s8(sptr0));
                    int32x2_t _r1 = vreinterpret_s32_s8(vld1_s8(sptr1));
                    int32x2_t _r2 = vreinterpret_s32_s8(vld1_s8(sptr2));
                    int32x2_t _r3 = vreinterpret_s32_s8(vld1_s8(sptr3));
                    int32x2_t _r4 = vreinterpret_s32_s8(vld1_s8(sptr4));
                    int32x2_t _r5 = vreinterpret_s32_s8(vld1_s8(sptr5));
                    int32x2_t _r6 = vreinterpret_s32_s8(vld1_s8(sptr6));
                    int32x2_t _r7 = vreinterpret_s32_s8(vld1_s8(sptr7));
                    int32x2x2_t _r01 = vzip_s32(_r0, _r1);
                    int32x2x2_t _r23 = vzip_s32(_r2, _r3);
                    int32x2x2_t _r45 = vzip_s32(_r4, _r5);
                    int32x2x2_t _r67 = vzip_s32(_r6, _r7);
                    vst1_s32((int*)pp, _r01.val[0]);
                    vst1_s32((int*)(pp + 8), _r23.val[0]);
                    vst1_s32((int*)(pp + 16), _r45.val[0]);
                    vst1_s32((int*)(pp + 24), _r67.val[0]);
                    vst1_s32((int*)(pp + 32), _r01.val[1]);
                    vst1_s32((int*)(pp + 40), _r23.val[1]);
                    vst1_s32((int*)(pp + 48), _r45.val[1]);
                    vst1_s32((int*)(pp + 56), _r67.val[1]);
                    pp += 64;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr0));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr1));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr2));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr3));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr4));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr5));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr6));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr7));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;

        if (dy0 == dy3)
        {
            int kk = 0;
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        int16x4x4_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(vzip_s8(_r0, _r1).val[0]);
                        _r0123.val[1] = vreinterpret_s16_s8(vzip_s8(_r2, _r3).val[0]);
                        _r0123.val[2] = vreinterpret_s16_s8(vzip_s8(_r4, _r5).val[0]);
                        _r0123.val[3] = vreinterpret_s16_s8(vzip_s8(_r6, _r7).val[0]);
                        vst4_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        int8x8_t _r23 = vtrn_s8(_r2, _r3).val[0];
                        int8x8_t _r45 = vtrn_s8(_r4, _r5).val[0];
                        int8x8_t _r67 = vtrn_s8(_r6, _r7).val[0];
                        int16x4x4_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(_r01);
                        _r0123.val[1] = vreinterpret_s16_s8(_r23);
                        _r0123.val[2] = vreinterpret_s16_s8(_r45);
                        _r0123.val[3] = vreinterpret_s16_s8(_r67);
                        vst4_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr4[0];
                        pp[5] = sptr5[0];
                        pp[6] = sptr6[0];
                        pp[7] = sptr7[0];
                        pp[8] = sptr0[stride_w];
                        pp[9] = sptr1[stride_w];
                        pp[10] = sptr2[stride_w];
                        pp[11] = sptr3[stride_w];
                        pp[12] = sptr4[stride_w];
                        pp[13] = sptr5[stride_w];
                        pp[14] = sptr6[stride_w];
                        pp[15] = sptr7[stride_w];
                        pp[16] = sptr0[stride_w * 2];
                        pp[17] = sptr1[stride_w * 2];
                        pp[18] = sptr2[stride_w * 2];
                        pp[19] = sptr3[stride_w * 2];
                        pp[20] = sptr4[stride_w * 2];
                        pp[21] = sptr5[stride_w * 2];
                        pp[22] = sptr6[stride_w * 2];
                        pp[23] = sptr7[stride_w * 2];
                        pp[24] = sptr0[stride_w * 3];
                        pp[25] = sptr1[stride_w * 3];
                        pp[26] = sptr2[stride_w * 3];
                        pp[27] = sptr3[stride_w * 3];
                        pp[28] = sptr4[stride_w * 3];
                        pp[29] = sptr5[stride_w * 3];
                        pp[30] = sptr6[stride_w * 3];
                        pp[31] = sptr7[stride_w * 3];
                        pp += 32;
                    }
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int16x4x2_t _r01;
                        _r01.val[0] = vreinterpret_s16_s8(vzip_s8(_r0, _r1).val[0]);
                        _r01.val[1] = vreinterpret_s16_s8(vzip_s8(_r2, _r3).val[0]);
                        vst2_s16((short*)pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        int8x8_t _r23 = vtrn_s8(_r2, _r3).val[0];
                        int16x4x2_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(_r01);
                        _r0123.val[1] = vreinterpret_s16_s8(_r23);
                        vst2_s16((short*)pp, _r0123);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr0[stride_w];
                        pp[5] = sptr1[stride_w];
                        pp[6] = sptr2[stride_w];
                        pp[7] = sptr3[stride_w];
                        pp[8] = sptr0[stride_w * 2];
                        pp[9] = sptr1[stride_w * 2];
                        pp[10] = sptr2[stride_w * 2];
                        pp[11] = sptr3[stride_w * 2];
                        pp[12] = sptr0[stride_w * 3];
                        pp[13] = sptr1[stride_w * 3];
                        pp[14] = sptr2[stride_w * 3];
                        pp[15] = sptr3[stride_w * 3];
                        pp += 16;
                    }
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vzip_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    int8x8_t _r2 = vld1_s8(sptr + stride_w * 16);
                    int8x8_t _r3 = vld1_s8(sptr + stride_w * 24);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    pp += 32;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s32((int*)pp, _r0123);
                    pp += 32;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int x42 = dx2 + dilation_w * v4;
                    int x43 = dx3 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int y42 = dy2 + dilation_h * u4;
                    int y43 = dy3 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int x52 = dx2 + dilation_w * v5;
                    int x53 = dx3 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;
                    int y52 = dy2 + dilation_h * u5;
                    int y53 = dy3 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int x62 = dx2 + dilation_w * v6;
                    int x63 = dx3 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int y62 = dy2 + dilation_h * u6;
                    int y63 = dy3 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int x72 = dx2 + dilation_w * v7;
                    int x73 = dx3 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;
                    int y72 = dy2 + dilation_h * u7;
                    int y73 = dy3 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr42 = img4.row<const signed char>(y42) + x42;
                    const signed char* sptr43 = img4.row<const signed char>(y43) + x43;

                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr52 = img5.row<const signed char>(y52) + x52;
                    const signed char* sptr53 = img5.row<const signed char>(y53) + x53;

                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr62 = img6.row<const signed char>(y62) + x62;
                    const signed char* sptr63 = img6.row<const signed char>(y63) + x63;

                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;
                    const signed char* sptr72 = img7.row<const signed char>(y72) + x72;
                    const signed char* sptr73 = img7.row<const signed char>(y73) + x73;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp[16] = sptr02[0];
                    pp[17] = sptr12[0];
                    pp[18] = sptr22[0];
                    pp[19] = sptr32[0];
                    pp[20] = sptr42[0];
                    pp[21] = sptr52[0];
                    pp[22] = sptr62[0];
                    pp[23] = sptr72[0];
                    pp[24] = sptr03[0];
                    pp[25] = sptr13[0];
                    pp[26] = sptr23[0];
                    pp[27] = sptr33[0];
                    pp[28] = sptr43[0];
                    pp[29] = sptr53[0];
                    pp[30] = sptr63[0];
                    pp[31] = sptr73[0];
                    pp += 32;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp[8] = sptr02[0];
                    pp[9] = sptr12[0];
                    pp[10] = sptr22[0];
                    pp[11] = sptr32[0];
                    pp[12] = sptr03[0];
                    pp[13] = sptr13[0];
                    pp[14] = sptr23[0];
                    pp[15] = sptr33[0];
                    pp += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp += 8;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    int8x8_t _r2 = vld1_s8(sptr2);
                    int8x8_t _r3 = vld1_s8(sptr3);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    pp += 32;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(sptr3));
                    vst4_s32((int*)pp, _r0123);
                    pp += 32;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr3));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;

        if (dy0 == dy1)
        {
            int kk = 0;
#if __ARM_NEON
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr0[stride_w];
                    pp[9] = sptr1[stride_w];
                    pp[10] = sptr2[stride_w];
                    pp[11] = sptr3[stride_w];
                    pp[12] = sptr4[stride_w];
                    pp[13] = sptr5[stride_w];
                    pp[14] = sptr6[stride_w];
                    pp[15] = sptr7[stride_w];
                    pp += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr0[stride_w];
                    pp[5] = sptr1[stride_w];
                    pp[6] = sptr2[stride_w];
                    pp[7] = sptr3[stride_w];
                    pp += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp += 4;
                }
            }
#endif // __ARM_NEON
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

#if __ARM_NEON
                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    pp += 16;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x2_t _r01;
                    _r01.val[0] = vreinterpret_s32_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s32((int*)pp, _r01);
                    pp += 16;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                }
#endif // __ARM_NEON
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
#if __ARM_NEON
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp += 4;
                }
            }
#endif // __ARM_NEON
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

#if __ARM_NEON
                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    pp += 16;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x2_t _r01;
                    _r01.val[0] = vreinterpret_s32_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s32_s8(vld1_s8(sptr1));
                    vst2_s32((int*)pp, _r01);
                    pp += 16;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                }
#endif // __ARM_NEON
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
        int dy = (j + jj) / outw * stride_h;
        int dx = (j + jj) % outw * stride_w;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = dx + dilation_w * v;
            int y = dy + dilation_h * u;

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

#if __ARM_NEON
            if (elempack == 8)
            {
                vst1_s8(pp, vld1_s8(sptr));
                pp += 8;
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

#if __ARM_FEATURE_MATMUL_INT8
template void convolution_im2col_input_tile_int8_i8mm<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_i8mm<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_i8mm<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_i8mm<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_i8mm<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_i8mm<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#elif __ARM_FEATURE_DOTPROD
template void convolution_im2col_input_tile_int8_asimddp<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_asimddp<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_asimddp<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_asimddp<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_asimddp<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8_asimddp<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
template void convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
template void convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD

static void convolution_im2col_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1)
    {
        convolution_im2col_input_tile_conv1x1s1d1_int8(bottom_blob, B, j, max_jj, k, max_kk);
        return;
    }

    if (kernel_w == 1 && kernel_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<1, 1, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<3, 3, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 1, 1>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<5, 5, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
#if __ARM_FEATURE_MATMUL_INT8
        convolution_im2col_input_tile_int8_i8mm<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#elif __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8_asimddp<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        convolution_im2col_input_tile_int8<7, 7, 1, 1, 2, 2>(bottom_blob, B, j, max_jj, k, max_kk);
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
        return;
    }

    const int w = bottom_blob.w;
    // const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    // j max_jj     outw*outh    split w and h

    // k max_kk     pa*maxk*(inch/pa)    split inch

    // k/max_kk shall be multiple of maxk

    const int maxk = kernel_w * kernel_h;

    signed char* pp = B;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dy4 = (j + jj + 4) / outw * stride_h;
        int dy5 = (j + jj + 5) / outw * stride_h;
        int dy6 = (j + jj + 6) / outw * stride_h;
        int dy7 = (j + jj + 7) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;
        int dx4 = (j + jj + 4) % outw * stride_w;
        int dx5 = (j + jj + 5) % outw * stride_w;
        int dx6 = (j + jj + 6) % outw * stride_w;
        int dx7 = (j + jj + 7) % outw * stride_w;

        if (dy0 == dy7)
        {
            int kk = 0;
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        // save as transpose8x8
                        int8x8x2_t _r01 = vzip_s8(_r0, _r1);
                        int8x8x2_t _r23 = vzip_s8(_r2, _r3);
                        int8x8x2_t _r45 = vzip_s8(_r4, _r5);
                        int8x8x2_t _r67 = vzip_s8(_r6, _r7);
                        int16x8x4_t _r0246;
                        _r0246.val[0] = vreinterpretq_s16_s8(vcombine_s8(_r01.val[0], _r01.val[1]));
                        _r0246.val[1] = vreinterpretq_s16_s8(vcombine_s8(_r23.val[0], _r23.val[1]));
                        _r0246.val[2] = vreinterpretq_s16_s8(vcombine_s8(_r45.val[0], _r45.val[1]));
                        _r0246.val[3] = vreinterpretq_s16_s8(vcombine_s8(_r67.val[0], _r67.val[1]));
                        vst4q_s16((short*)pp, _r0246);
                        pp += 64;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r2 = vld1q_s8(sptr2);
                        int8x16_t _r3 = vld1q_s8(sptr3);
                        int8x16_t _r4 = vld1q_s8(sptr4);
                        int8x16_t _r5 = vld1q_s8(sptr5);
                        int8x16_t _r6 = vld1q_s8(sptr6);
                        int8x16_t _r7 = vld1q_s8(sptr7);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        int8x16_t _r23 = vtrnq_s8(_r2, _r3).val[0];
                        int8x16_t _r45 = vtrnq_s8(_r4, _r5).val[0];
                        int8x16_t _r67 = vtrnq_s8(_r6, _r7).val[0];
                        int16x8x4_t _r0123;
                        _r0123.val[0] = vreinterpretq_s16_s8(_r01);
                        _r0123.val[1] = vreinterpretq_s16_s8(_r23);
                        _r0123.val[2] = vreinterpretq_s16_s8(_r45);
                        _r0123.val[3] = vreinterpretq_s16_s8(_r67);
                        vst4q_s16((short*)pp, _r0123);
                        pp += 64;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr4[0];
                        pp[5] = sptr5[0];
                        pp[6] = sptr6[0];
                        pp[7] = sptr7[0];
                        pp[8] = sptr0[stride_w];
                        pp[9] = sptr1[stride_w];
                        pp[10] = sptr2[stride_w];
                        pp[11] = sptr3[stride_w];
                        pp[12] = sptr4[stride_w];
                        pp[13] = sptr5[stride_w];
                        pp[14] = sptr6[stride_w];
                        pp[15] = sptr7[stride_w];
                        pp[16] = sptr0[stride_w * 2];
                        pp[17] = sptr1[stride_w * 2];
                        pp[18] = sptr2[stride_w * 2];
                        pp[19] = sptr3[stride_w * 2];
                        pp[20] = sptr4[stride_w * 2];
                        pp[21] = sptr5[stride_w * 2];
                        pp[22] = sptr6[stride_w * 2];
                        pp[23] = sptr7[stride_w * 2];
                        pp[24] = sptr0[stride_w * 3];
                        pp[25] = sptr1[stride_w * 3];
                        pp[26] = sptr2[stride_w * 3];
                        pp[27] = sptr3[stride_w * 3];
                        pp[28] = sptr4[stride_w * 3];
                        pp[29] = sptr5[stride_w * 3];
                        pp[30] = sptr6[stride_w * 3];
                        pp[31] = sptr7[stride_w * 3];
                        pp[32] = sptr0[stride_w * 4];
                        pp[33] = sptr1[stride_w * 4];
                        pp[34] = sptr2[stride_w * 4];
                        pp[35] = sptr3[stride_w * 4];
                        pp[36] = sptr4[stride_w * 4];
                        pp[37] = sptr5[stride_w * 4];
                        pp[38] = sptr6[stride_w * 4];
                        pp[39] = sptr7[stride_w * 4];
                        pp[40] = sptr0[stride_w * 5];
                        pp[41] = sptr1[stride_w * 5];
                        pp[42] = sptr2[stride_w * 5];
                        pp[43] = sptr3[stride_w * 5];
                        pp[44] = sptr4[stride_w * 5];
                        pp[45] = sptr5[stride_w * 5];
                        pp[46] = sptr6[stride_w * 5];
                        pp[47] = sptr7[stride_w * 5];
                        pp[48] = sptr0[stride_w * 6];
                        pp[49] = sptr1[stride_w * 6];
                        pp[50] = sptr2[stride_w * 6];
                        pp[51] = sptr3[stride_w * 6];
                        pp[52] = sptr4[stride_w * 6];
                        pp[53] = sptr5[stride_w * 6];
                        pp[54] = sptr6[stride_w * 6];
                        pp[55] = sptr7[stride_w * 6];
                        pp[56] = sptr0[stride_w * 7];
                        pp[57] = sptr1[stride_w * 7];
                        pp[58] = sptr2[stride_w * 7];
                        pp[59] = sptr3[stride_w * 7];
                        pp[60] = sptr4[stride_w * 7];
                        pp[61] = sptr5[stride_w * 7];
                        pp[62] = sptr6[stride_w * 7];
                        pp[63] = sptr7[stride_w * 7];
                        pp += 64;
                    }
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    if (stride_w == 1)
                    {
                        int8x8x4_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        _r01.val[2] = vld1_s8(sptr2);
                        _r01.val[3] = vld1_s8(sptr3);
                        vst4_s8(pp, _r01);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r2 = vld1q_s8(sptr2);
                        int8x16_t _r3 = vld1q_s8(sptr3);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        int8x16_t _r23 = vtrnq_s8(_r2, _r3).val[0];
                        int16x8x2_t _r0123;
                        _r0123.val[0] = vreinterpretq_s16_s8(_r01);
                        _r0123.val[1] = vreinterpretq_s16_s8(_r23);
                        vst2q_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr0[stride_w];
                        pp[5] = sptr1[stride_w];
                        pp[6] = sptr2[stride_w];
                        pp[7] = sptr3[stride_w];
                        pp[8] = sptr0[stride_w * 2];
                        pp[9] = sptr1[stride_w * 2];
                        pp[10] = sptr2[stride_w * 2];
                        pp[11] = sptr3[stride_w * 2];
                        pp[12] = sptr0[stride_w * 3];
                        pp[13] = sptr1[stride_w * 3];
                        pp[14] = sptr2[stride_w * 3];
                        pp[15] = sptr3[stride_w * 3];
                        pp[16] = sptr0[stride_w * 4];
                        pp[17] = sptr1[stride_w * 4];
                        pp[18] = sptr2[stride_w * 4];
                        pp[19] = sptr3[stride_w * 4];
                        pp[20] = sptr0[stride_w * 5];
                        pp[21] = sptr1[stride_w * 5];
                        pp[22] = sptr2[stride_w * 5];
                        pp[23] = sptr3[stride_w * 5];
                        pp[24] = sptr0[stride_w * 6];
                        pp[25] = sptr1[stride_w * 6];
                        pp[26] = sptr2[stride_w * 6];
                        pp[27] = sptr3[stride_w * 6];
                        pp[28] = sptr0[stride_w * 7];
                        pp[29] = sptr1[stride_w * 7];
                        pp[30] = sptr2[stride_w * 7];
                        pp[31] = sptr3[stride_w * 7];
                        pp += 32;
                    }
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8x2_t _r01;
                        _r01.val[0] = vld1_s8(sptr0);
                        _r01.val[1] = vld1_s8(sptr1);
                        vst2_s8(pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x16_t _r0 = vld1q_s8(sptr0);
                        int8x16_t _r1 = vld1q_s8(sptr1);
                        int8x16_t _r01 = vtrnq_s8(_r0, _r1).val[0];
                        vst1q_s8(pp, _r01);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp[8] = sptr0[stride_w * 4];
                        pp[9] = sptr1[stride_w * 4];
                        pp[10] = sptr0[stride_w * 5];
                        pp[11] = sptr1[stride_w * 5];
                        pp[12] = sptr0[stride_w * 6];
                        pp[13] = sptr1[stride_w * 6];
                        pp[14] = sptr0[stride_w * 7];
                        pp[15] = sptr1[stride_w * 7];
                        pp += 16;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    int8x8_t _r2 = vld1_s8(sptr + stride_w * 16);
                    int8x8_t _r3 = vld1_s8(sptr + stride_w * 24);
                    int8x8_t _r4 = vld1_s8(sptr + stride_w * 32);
                    int8x8_t _r5 = vld1_s8(sptr + stride_w * 40);
                    int8x8_t _r6 = vld1_s8(sptr + stride_w * 48);
                    int8x8_t _r7 = vld1_s8(sptr + stride_w * 56);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    vst1_s8(pp + 32, _r4);
                    vst1_s8(pp + 40, _r5);
                    vst1_s8(pp + 48, _r6);
                    vst1_s8(pp + 56, _r7);
                    pp += 64;
#elif __ARM_FEATURE_DOTPROD
                    int32x2_t _r0 = vreinterpret_s32_s8(vld1_s8(sptr));
                    int32x2_t _r1 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    int32x2_t _r2 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 16));
                    int32x2_t _r3 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 24));
                    int32x2_t _r4 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 32));
                    int32x2_t _r5 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 40));
                    int32x2_t _r6 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 48));
                    int32x2_t _r7 = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 56));
                    int32x2x2_t _r01 = vzip_s32(_r0, _r1);
                    int32x2x2_t _r23 = vzip_s32(_r2, _r3);
                    int32x2x2_t _r45 = vzip_s32(_r4, _r5);
                    int32x2x2_t _r67 = vzip_s32(_r6, _r7);
                    vst1_s32((int*)pp, _r01.val[0]);
                    vst1_s32((int*)(pp + 8), _r23.val[0]);
                    vst1_s32((int*)(pp + 16), _r45.val[0]);
                    vst1_s32((int*)(pp + 24), _r67.val[0]);
                    vst1_s32((int*)(pp + 32), _r01.val[1]);
                    vst1_s32((int*)(pp + 40), _r23.val[1]);
                    vst1_s32((int*)(pp + 48), _r45.val[1]);
                    vst1_s32((int*)(pp + 56), _r67.val[1]);
                    pp += 64;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 32));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 40));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 48));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 56));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int x24 = dx4 + dilation_w * v2;
                    int x25 = dx5 + dilation_w * v2;
                    int x26 = dx6 + dilation_w * v2;
                    int x27 = dx7 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;
                    int y24 = dy4 + dilation_h * u2;
                    int y25 = dy5 + dilation_h * u2;
                    int y26 = dy6 + dilation_h * u2;
                    int y27 = dy7 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int x34 = dx4 + dilation_w * v3;
                    int x35 = dx5 + dilation_w * v3;
                    int x36 = dx6 + dilation_w * v3;
                    int x37 = dx7 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;
                    int y34 = dy4 + dilation_h * u3;
                    int y35 = dy5 + dilation_h * u3;
                    int y36 = dy6 + dilation_h * u3;
                    int y37 = dy7 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int x42 = dx2 + dilation_w * v4;
                    int x43 = dx3 + dilation_w * v4;
                    int x44 = dx4 + dilation_w * v4;
                    int x45 = dx5 + dilation_w * v4;
                    int x46 = dx6 + dilation_w * v4;
                    int x47 = dx7 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int y42 = dy2 + dilation_h * u4;
                    int y43 = dy3 + dilation_h * u4;
                    int y44 = dy4 + dilation_h * u4;
                    int y45 = dy5 + dilation_h * u4;
                    int y46 = dy6 + dilation_h * u4;
                    int y47 = dy7 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int x52 = dx2 + dilation_w * v5;
                    int x53 = dx3 + dilation_w * v5;
                    int x54 = dx4 + dilation_w * v5;
                    int x55 = dx5 + dilation_w * v5;
                    int x56 = dx6 + dilation_w * v5;
                    int x57 = dx7 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;
                    int y52 = dy2 + dilation_h * u5;
                    int y53 = dy3 + dilation_h * u5;
                    int y54 = dy4 + dilation_h * u5;
                    int y55 = dy5 + dilation_h * u5;
                    int y56 = dy6 + dilation_h * u5;
                    int y57 = dy7 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int x62 = dx2 + dilation_w * v6;
                    int x63 = dx3 + dilation_w * v6;
                    int x64 = dx4 + dilation_w * v6;
                    int x65 = dx5 + dilation_w * v6;
                    int x66 = dx6 + dilation_w * v6;
                    int x67 = dx7 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int y62 = dy2 + dilation_h * u6;
                    int y63 = dy3 + dilation_h * u6;
                    int y64 = dy4 + dilation_h * u6;
                    int y65 = dy5 + dilation_h * u6;
                    int y66 = dy6 + dilation_h * u6;
                    int y67 = dy7 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int x72 = dx2 + dilation_w * v7;
                    int x73 = dx3 + dilation_w * v7;
                    int x74 = dx4 + dilation_w * v7;
                    int x75 = dx5 + dilation_w * v7;
                    int x76 = dx6 + dilation_w * v7;
                    int x77 = dx7 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;
                    int y72 = dy2 + dilation_h * u7;
                    int y73 = dy3 + dilation_h * u7;
                    int y74 = dy4 + dilation_h * u7;
                    int y75 = dy5 + dilation_h * u7;
                    int y76 = dy6 + dilation_h * u7;
                    int y77 = dy7 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;
                    const signed char* sptr24 = img2.row<const signed char>(y24) + x24;
                    const signed char* sptr25 = img2.row<const signed char>(y25) + x25;
                    const signed char* sptr26 = img2.row<const signed char>(y26) + x26;
                    const signed char* sptr27 = img2.row<const signed char>(y27) + x27;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;
                    const signed char* sptr34 = img3.row<const signed char>(y34) + x34;
                    const signed char* sptr35 = img3.row<const signed char>(y35) + x35;
                    const signed char* sptr36 = img3.row<const signed char>(y36) + x36;
                    const signed char* sptr37 = img3.row<const signed char>(y37) + x37;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr42 = img4.row<const signed char>(y42) + x42;
                    const signed char* sptr43 = img4.row<const signed char>(y43) + x43;
                    const signed char* sptr44 = img4.row<const signed char>(y44) + x44;
                    const signed char* sptr45 = img4.row<const signed char>(y45) + x45;
                    const signed char* sptr46 = img4.row<const signed char>(y46) + x46;
                    const signed char* sptr47 = img4.row<const signed char>(y47) + x47;

                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr52 = img5.row<const signed char>(y52) + x52;
                    const signed char* sptr53 = img5.row<const signed char>(y53) + x53;
                    const signed char* sptr54 = img5.row<const signed char>(y54) + x54;
                    const signed char* sptr55 = img5.row<const signed char>(y55) + x55;
                    const signed char* sptr56 = img5.row<const signed char>(y56) + x56;
                    const signed char* sptr57 = img5.row<const signed char>(y57) + x57;

                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr62 = img6.row<const signed char>(y62) + x62;
                    const signed char* sptr63 = img6.row<const signed char>(y63) + x63;
                    const signed char* sptr64 = img6.row<const signed char>(y64) + x64;
                    const signed char* sptr65 = img6.row<const signed char>(y65) + x65;
                    const signed char* sptr66 = img6.row<const signed char>(y66) + x66;
                    const signed char* sptr67 = img6.row<const signed char>(y67) + x67;

                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;
                    const signed char* sptr72 = img7.row<const signed char>(y72) + x72;
                    const signed char* sptr73 = img7.row<const signed char>(y73) + x73;
                    const signed char* sptr74 = img7.row<const signed char>(y74) + x74;
                    const signed char* sptr75 = img7.row<const signed char>(y75) + x75;
                    const signed char* sptr76 = img7.row<const signed char>(y76) + x76;
                    const signed char* sptr77 = img7.row<const signed char>(y77) + x77;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp[16] = sptr02[0];
                    pp[17] = sptr12[0];
                    pp[18] = sptr22[0];
                    pp[19] = sptr32[0];
                    pp[20] = sptr42[0];
                    pp[21] = sptr52[0];
                    pp[22] = sptr62[0];
                    pp[23] = sptr72[0];
                    pp[24] = sptr03[0];
                    pp[25] = sptr13[0];
                    pp[26] = sptr23[0];
                    pp[27] = sptr33[0];
                    pp[28] = sptr43[0];
                    pp[29] = sptr53[0];
                    pp[30] = sptr63[0];
                    pp[31] = sptr73[0];
                    pp[32] = sptr04[0];
                    pp[33] = sptr14[0];
                    pp[34] = sptr24[0];
                    pp[35] = sptr34[0];
                    pp[36] = sptr44[0];
                    pp[37] = sptr54[0];
                    pp[38] = sptr64[0];
                    pp[39] = sptr74[0];
                    pp[40] = sptr05[0];
                    pp[41] = sptr15[0];
                    pp[42] = sptr25[0];
                    pp[43] = sptr35[0];
                    pp[44] = sptr45[0];
                    pp[45] = sptr55[0];
                    pp[46] = sptr65[0];
                    pp[47] = sptr75[0];
                    pp[48] = sptr06[0];
                    pp[49] = sptr16[0];
                    pp[50] = sptr26[0];
                    pp[51] = sptr36[0];
                    pp[52] = sptr46[0];
                    pp[53] = sptr56[0];
                    pp[54] = sptr66[0];
                    pp[55] = sptr76[0];
                    pp[56] = sptr07[0];
                    pp[57] = sptr17[0];
                    pp[58] = sptr27[0];
                    pp[59] = sptr37[0];
                    pp[60] = sptr47[0];
                    pp[61] = sptr57[0];
                    pp[62] = sptr67[0];
                    pp[63] = sptr77[0];
                    pp += 64;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int x24 = dx4 + dilation_w * v2;
                    int x25 = dx5 + dilation_w * v2;
                    int x26 = dx6 + dilation_w * v2;
                    int x27 = dx7 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;
                    int y24 = dy4 + dilation_h * u2;
                    int y25 = dy5 + dilation_h * u2;
                    int y26 = dy6 + dilation_h * u2;
                    int y27 = dy7 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int x34 = dx4 + dilation_w * v3;
                    int x35 = dx5 + dilation_w * v3;
                    int x36 = dx6 + dilation_w * v3;
                    int x37 = dx7 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;
                    int y34 = dy4 + dilation_h * u3;
                    int y35 = dy5 + dilation_h * u3;
                    int y36 = dy6 + dilation_h * u3;
                    int y37 = dy7 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;
                    const signed char* sptr24 = img2.row<const signed char>(y24) + x24;
                    const signed char* sptr25 = img2.row<const signed char>(y25) + x25;
                    const signed char* sptr26 = img2.row<const signed char>(y26) + x26;
                    const signed char* sptr27 = img2.row<const signed char>(y27) + x27;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;
                    const signed char* sptr34 = img3.row<const signed char>(y34) + x34;
                    const signed char* sptr35 = img3.row<const signed char>(y35) + x35;
                    const signed char* sptr36 = img3.row<const signed char>(y36) + x36;
                    const signed char* sptr37 = img3.row<const signed char>(y37) + x37;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp[8] = sptr02[0];
                    pp[9] = sptr12[0];
                    pp[10] = sptr22[0];
                    pp[11] = sptr32[0];
                    pp[12] = sptr03[0];
                    pp[13] = sptr13[0];
                    pp[14] = sptr23[0];
                    pp[15] = sptr33[0];
                    pp[16] = sptr04[0];
                    pp[17] = sptr14[0];
                    pp[18] = sptr24[0];
                    pp[19] = sptr34[0];
                    pp[20] = sptr05[0];
                    pp[21] = sptr15[0];
                    pp[22] = sptr25[0];
                    pp[23] = sptr35[0];
                    pp[24] = sptr06[0];
                    pp[25] = sptr16[0];
                    pp[26] = sptr26[0];
                    pp[27] = sptr36[0];
                    pp[28] = sptr07[0];
                    pp[29] = sptr17[0];
                    pp[30] = sptr27[0];
                    pp[31] = sptr37[0];
                    pp += 32;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int x04 = dx4 + dilation_w * v0;
                    int x05 = dx5 + dilation_w * v0;
                    int x06 = dx6 + dilation_w * v0;
                    int x07 = dx7 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;
                    int y04 = dy4 + dilation_h * u0;
                    int y05 = dy5 + dilation_h * u0;
                    int y06 = dy6 + dilation_h * u0;
                    int y07 = dy7 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int x14 = dx4 + dilation_w * v1;
                    int x15 = dx5 + dilation_w * v1;
                    int x16 = dx6 + dilation_w * v1;
                    int x17 = dx7 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;
                    int y14 = dy4 + dilation_h * u1;
                    int y15 = dy5 + dilation_h * u1;
                    int y16 = dy6 + dilation_h * u1;
                    int y17 = dy7 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;
                    const signed char* sptr04 = img0.row<const signed char>(y04) + x04;
                    const signed char* sptr05 = img0.row<const signed char>(y05) + x05;
                    const signed char* sptr06 = img0.row<const signed char>(y06) + x06;
                    const signed char* sptr07 = img0.row<const signed char>(y07) + x07;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;
                    const signed char* sptr14 = img1.row<const signed char>(y14) + x14;
                    const signed char* sptr15 = img1.row<const signed char>(y15) + x15;
                    const signed char* sptr16 = img1.row<const signed char>(y16) + x16;
                    const signed char* sptr17 = img1.row<const signed char>(y17) + x17;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp[8] = sptr04[0];
                    pp[9] = sptr14[0];
                    pp[10] = sptr05[0];
                    pp[11] = sptr15[0];
                    pp[12] = sptr06[0];
                    pp[13] = sptr16[0];
                    pp[14] = sptr07[0];
                    pp[15] = sptr17[0];
                    pp += 16;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int x4 = dx4 + dilation_w * v;
                int x5 = dx5 + dilation_w * v;
                int x6 = dx6 + dilation_w * v;
                int x7 = dx7 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;
                int y4 = dy4 + dilation_h * u;
                int y5 = dy5 + dilation_h * u;
                int y6 = dy6 + dilation_h * u;
                int y7 = dy7 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;
                const signed char* sptr4 = img.row<const signed char>(y4) + x4 * elempack;
                const signed char* sptr5 = img.row<const signed char>(y5) + x5 * elempack;
                const signed char* sptr6 = img.row<const signed char>(y6) + x6 * elempack;
                const signed char* sptr7 = img.row<const signed char>(y7) + x7 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    int8x8_t _r2 = vld1_s8(sptr2);
                    int8x8_t _r3 = vld1_s8(sptr3);
                    int8x8_t _r4 = vld1_s8(sptr4);
                    int8x8_t _r5 = vld1_s8(sptr5);
                    int8x8_t _r6 = vld1_s8(sptr6);
                    int8x8_t _r7 = vld1_s8(sptr7);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    vst1_s8(pp + 32, _r4);
                    vst1_s8(pp + 40, _r5);
                    vst1_s8(pp + 48, _r6);
                    vst1_s8(pp + 56, _r7);
                    pp += 64;
#elif __ARM_FEATURE_DOTPROD
                    int32x2_t _r0 = vreinterpret_s32_s8(vld1_s8(sptr0));
                    int32x2_t _r1 = vreinterpret_s32_s8(vld1_s8(sptr1));
                    int32x2_t _r2 = vreinterpret_s32_s8(vld1_s8(sptr2));
                    int32x2_t _r3 = vreinterpret_s32_s8(vld1_s8(sptr3));
                    int32x2_t _r4 = vreinterpret_s32_s8(vld1_s8(sptr4));
                    int32x2_t _r5 = vreinterpret_s32_s8(vld1_s8(sptr5));
                    int32x2_t _r6 = vreinterpret_s32_s8(vld1_s8(sptr6));
                    int32x2_t _r7 = vreinterpret_s32_s8(vld1_s8(sptr7));
                    int32x2x2_t _r01 = vzip_s32(_r0, _r1);
                    int32x2x2_t _r23 = vzip_s32(_r2, _r3);
                    int32x2x2_t _r45 = vzip_s32(_r4, _r5);
                    int32x2x2_t _r67 = vzip_s32(_r6, _r7);
                    vst1_s32((int*)pp, _r01.val[0]);
                    vst1_s32((int*)(pp + 8), _r23.val[0]);
                    vst1_s32((int*)(pp + 16), _r45.val[0]);
                    vst1_s32((int*)(pp + 24), _r67.val[0]);
                    vst1_s32((int*)(pp + 32), _r01.val[1]);
                    vst1_s32((int*)(pp + 40), _r23.val[1]);
                    vst1_s32((int*)(pp + 48), _r45.val[1]);
                    vst1_s32((int*)(pp + 56), _r67.val[1]);
                    pp += 64;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4_t _r0 = vreinterpret_s16_s8(vld1_s8(sptr0));
                    int16x4_t _r1 = vreinterpret_s16_s8(vld1_s8(sptr1));
                    int16x4_t _r2 = vreinterpret_s16_s8(vld1_s8(sptr2));
                    int16x4_t _r3 = vreinterpret_s16_s8(vld1_s8(sptr3));
                    int16x4_t _r4 = vreinterpret_s16_s8(vld1_s8(sptr4));
                    int16x4_t _r5 = vreinterpret_s16_s8(vld1_s8(sptr5));
                    int16x4_t _r6 = vreinterpret_s16_s8(vld1_s8(sptr6));
                    int16x4_t _r7 = vreinterpret_s16_s8(vld1_s8(sptr7));
                    int16x4x2_t _r01 = vzip_s16(_r0, _r1);
                    int16x4x2_t _r23 = vzip_s16(_r2, _r3);
                    int16x4x2_t _r45 = vzip_s16(_r4, _r5);
                    int16x4x2_t _r67 = vzip_s16(_r6, _r7);
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s16(vcombine_s16(_r01.val[0], _r01.val[1]));
                    _r0123.val[1] = vreinterpretq_s32_s16(vcombine_s16(_r23.val[0], _r23.val[1]));
                    _r0123.val[2] = vreinterpretq_s32_s16(vcombine_s16(_r45.val[0], _r45.val[1]));
                    _r0123.val[3] = vreinterpretq_s32_s16(vcombine_s16(_r67.val[0], _r67.val[1]));
                    vst4q_s32((int*)pp, _r0123);
                    pp += 64;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dy2 = (j + jj + 2) / outw * stride_h;
        int dy3 = (j + jj + 3) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;
        int dx2 = (j + jj + 2) % outw * stride_w;
        int dx3 = (j + jj + 3) % outw * stride_w;

        if (dy0 == dy3)
        {
            int kk = 0;
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        int16x4x4_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(vzip_s8(_r0, _r1).val[0]);
                        _r0123.val[1] = vreinterpret_s16_s8(vzip_s8(_r2, _r3).val[0]);
                        _r0123.val[2] = vreinterpret_s16_s8(vzip_s8(_r4, _r5).val[0]);
                        _r0123.val[3] = vreinterpret_s16_s8(vzip_s8(_r6, _r7).val[0]);
                        vst4_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r4 = vld1_s8(sptr4);
                        int8x8_t _r5 = vld1_s8(sptr5);
                        int8x8_t _r6 = vld1_s8(sptr6);
                        int8x8_t _r7 = vld1_s8(sptr7);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        int8x8_t _r23 = vtrn_s8(_r2, _r3).val[0];
                        int8x8_t _r45 = vtrn_s8(_r4, _r5).val[0];
                        int8x8_t _r67 = vtrn_s8(_r6, _r7).val[0];
                        int16x4x4_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(_r01);
                        _r0123.val[1] = vreinterpret_s16_s8(_r23);
                        _r0123.val[2] = vreinterpret_s16_s8(_r45);
                        _r0123.val[3] = vreinterpret_s16_s8(_r67);
                        vst4_s16((short*)pp, _r0123);
                        pp += 32;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr4[0];
                        pp[5] = sptr5[0];
                        pp[6] = sptr6[0];
                        pp[7] = sptr7[0];
                        pp[8] = sptr0[stride_w];
                        pp[9] = sptr1[stride_w];
                        pp[10] = sptr2[stride_w];
                        pp[11] = sptr3[stride_w];
                        pp[12] = sptr4[stride_w];
                        pp[13] = sptr5[stride_w];
                        pp[14] = sptr6[stride_w];
                        pp[15] = sptr7[stride_w];
                        pp[16] = sptr0[stride_w * 2];
                        pp[17] = sptr1[stride_w * 2];
                        pp[18] = sptr2[stride_w * 2];
                        pp[19] = sptr3[stride_w * 2];
                        pp[20] = sptr4[stride_w * 2];
                        pp[21] = sptr5[stride_w * 2];
                        pp[22] = sptr6[stride_w * 2];
                        pp[23] = sptr7[stride_w * 2];
                        pp[24] = sptr0[stride_w * 3];
                        pp[25] = sptr1[stride_w * 3];
                        pp[26] = sptr2[stride_w * 3];
                        pp[27] = sptr3[stride_w * 3];
                        pp[28] = sptr4[stride_w * 3];
                        pp[29] = sptr5[stride_w * 3];
                        pp[30] = sptr6[stride_w * 3];
                        pp[31] = sptr7[stride_w * 3];
                        pp += 32;
                    }
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int16x4x2_t _r01;
                        _r01.val[0] = vreinterpret_s16_s8(vzip_s8(_r0, _r1).val[0]);
                        _r01.val[1] = vreinterpret_s16_s8(vzip_s8(_r2, _r3).val[0]);
                        vst2_s16((short*)pp, _r01);
                        pp += 16;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r2 = vld1_s8(sptr2);
                        int8x8_t _r3 = vld1_s8(sptr3);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        int8x8_t _r23 = vtrn_s8(_r2, _r3).val[0];
                        int16x4x2_t _r0123;
                        _r0123.val[0] = vreinterpret_s16_s8(_r01);
                        _r0123.val[1] = vreinterpret_s16_s8(_r23);
                        vst2_s16((short*)pp, _r0123);
                        pp += 16;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr2[0];
                        pp[3] = sptr3[0];
                        pp[4] = sptr0[stride_w];
                        pp[5] = sptr1[stride_w];
                        pp[6] = sptr2[stride_w];
                        pp[7] = sptr3[stride_w];
                        pp[8] = sptr0[stride_w * 2];
                        pp[9] = sptr1[stride_w * 2];
                        pp[10] = sptr2[stride_w * 2];
                        pp[11] = sptr3[stride_w * 2];
                        pp[12] = sptr0[stride_w * 3];
                        pp[13] = sptr1[stride_w * 3];
                        pp[14] = sptr2[stride_w * 3];
                        pp[15] = sptr3[stride_w * 3];
                        pp += 16;
                    }
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    if (stride_w == 1)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vzip_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else if (stride_w == 2)
                    {
                        int8x8_t _r0 = vld1_s8(sptr0);
                        int8x8_t _r1 = vld1_s8(sptr1);
                        int8x8_t _r01 = vtrn_s8(_r0, _r1).val[0];
                        vst1_s8(pp, _r01);
                        pp += 8;
                    }
                    else
                    {
                        pp[0] = sptr0[0];
                        pp[1] = sptr1[0];
                        pp[2] = sptr0[stride_w];
                        pp[3] = sptr1[stride_w];
                        pp[4] = sptr0[stride_w * 2];
                        pp[5] = sptr1[stride_w * 2];
                        pp[6] = sptr0[stride_w * 3];
                        pp[7] = sptr1[stride_w * 3];
                        pp += 8;
                    }
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    int8x8_t _r2 = vld1_s8(sptr + stride_w * 16);
                    int8x8_t _r3 = vld1_s8(sptr + stride_w * 24);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    pp += 32;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s32((int*)pp, _r0123);
                    pp += 32;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 16));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 24));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int x42 = dx2 + dilation_w * v4;
                    int x43 = dx3 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int y42 = dy2 + dilation_h * u4;
                    int y43 = dy3 + dilation_h * u4;

                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int x52 = dx2 + dilation_w * v5;
                    int x53 = dx3 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;
                    int y52 = dy2 + dilation_h * u5;
                    int y53 = dy3 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int x62 = dx2 + dilation_w * v6;
                    int x63 = dx3 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int y62 = dy2 + dilation_h * u6;
                    int y63 = dy3 + dilation_h * u6;

                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int x72 = dx2 + dilation_w * v7;
                    int x73 = dx3 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;
                    int y72 = dy2 + dilation_h * u7;
                    int y73 = dy3 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr42 = img4.row<const signed char>(y42) + x42;
                    const signed char* sptr43 = img4.row<const signed char>(y43) + x43;

                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr52 = img5.row<const signed char>(y52) + x52;
                    const signed char* sptr53 = img5.row<const signed char>(y53) + x53;

                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr62 = img6.row<const signed char>(y62) + x62;
                    const signed char* sptr63 = img6.row<const signed char>(y63) + x63;

                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;
                    const signed char* sptr72 = img7.row<const signed char>(y72) + x72;
                    const signed char* sptr73 = img7.row<const signed char>(y73) + x73;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp[16] = sptr02[0];
                    pp[17] = sptr12[0];
                    pp[18] = sptr22[0];
                    pp[19] = sptr32[0];
                    pp[20] = sptr42[0];
                    pp[21] = sptr52[0];
                    pp[22] = sptr62[0];
                    pp[23] = sptr72[0];
                    pp[24] = sptr03[0];
                    pp[25] = sptr13[0];
                    pp[26] = sptr23[0];
                    pp[27] = sptr33[0];
                    pp[28] = sptr43[0];
                    pp[29] = sptr53[0];
                    pp[30] = sptr63[0];
                    pp[31] = sptr73[0];
                    pp += 32;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int x22 = dx2 + dilation_w * v2;
                    int x23 = dx3 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int y22 = dy2 + dilation_h * u2;
                    int y23 = dy3 + dilation_h * u2;

                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int x32 = dx2 + dilation_w * v3;
                    int x33 = dx3 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;
                    int y32 = dy2 + dilation_h * u3;
                    int y33 = dy3 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr22 = img2.row<const signed char>(y22) + x22;
                    const signed char* sptr23 = img2.row<const signed char>(y23) + x23;

                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;
                    const signed char* sptr32 = img3.row<const signed char>(y32) + x32;
                    const signed char* sptr33 = img3.row<const signed char>(y33) + x33;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp[8] = sptr02[0];
                    pp[9] = sptr12[0];
                    pp[10] = sptr22[0];
                    pp[11] = sptr32[0];
                    pp[12] = sptr03[0];
                    pp[13] = sptr13[0];
                    pp[14] = sptr23[0];
                    pp[15] = sptr33[0];
                    pp += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int x02 = dx2 + dilation_w * v0;
                    int x03 = dx3 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int y02 = dy2 + dilation_h * u0;
                    int y03 = dy3 + dilation_h * u0;

                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int x12 = dx2 + dilation_w * v1;
                    int x13 = dx3 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int y12 = dy2 + dilation_h * u1;
                    int y13 = dy3 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr02 = img0.row<const signed char>(y02) + x02;
                    const signed char* sptr03 = img0.row<const signed char>(y03) + x03;

                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr12 = img1.row<const signed char>(y12) + x12;
                    const signed char* sptr13 = img1.row<const signed char>(y13) + x13;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp[4] = sptr02[0];
                    pp[5] = sptr12[0];
                    pp[6] = sptr03[0];
                    pp[7] = sptr13[0];
                    pp += 8;
                }
            }
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int x2 = dx2 + dilation_w * v;
                int x3 = dx3 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;
                int y2 = dy2 + dilation_h * u;
                int y3 = dy3 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;
                const signed char* sptr2 = img.row<const signed char>(y2) + x2 * elempack;
                const signed char* sptr3 = img.row<const signed char>(y3) + x3 * elempack;

                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    int8x8_t _r2 = vld1_s8(sptr2);
                    int8x8_t _r3 = vld1_s8(sptr3);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    vst1_s8(pp + 16, _r2);
                    vst1_s8(pp + 24, _r3);
                    pp += 32;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s32_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s32_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s32_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s32_s8(vld1_s8(sptr3));
                    vst4_s32((int*)pp, _r0123);
                    pp += 32;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r0123.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    _r0123.val[2] = vreinterpret_s16_s8(vld1_s8(sptr2));
                    _r0123.val[3] = vreinterpret_s16_s8(vld1_s8(sptr3));
                    vst4_s16((short*)pp, _r0123);
                    pp += 32;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        int dy0 = (j + jj) / outw * stride_h;
        int dy1 = (j + jj + 1) / outw * stride_h;
        int dx0 = (j + jj) % outw * stride_w;
        int dx1 = (j + jj + 1) % outw * stride_w;

        if (dy0 == dy1)
        {
            int kk = 0;
#if __ARM_NEON
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int x50 = dx0 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int x70 = dx0 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    const signed char* sptr4 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr5 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr6 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr7 = img7.row<const signed char>(y70) + x70;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr4[0];
                    pp[5] = sptr5[0];
                    pp[6] = sptr6[0];
                    pp[7] = sptr7[0];
                    pp[8] = sptr0[stride_w];
                    pp[9] = sptr1[stride_w];
                    pp[10] = sptr2[stride_w];
                    pp[11] = sptr3[stride_w];
                    pp[12] = sptr4[stride_w];
                    pp[13] = sptr5[stride_w];
                    pp[14] = sptr6[stride_w];
                    pp[15] = sptr7[stride_w];
                    pp += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int x20 = dx0 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr2 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr3 = img3.row<const signed char>(y30) + x30;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr2[0];
                    pp[3] = sptr3[0];
                    pp[4] = sptr0[stride_w];
                    pp[5] = sptr1[stride_w];
                    pp[6] = sptr2[stride_w];
                    pp[7] = sptr3[stride_w];
                    pp += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;

                    const signed char* sptr0 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr1 = img1.row<const signed char>(y10) + x10;

                    pp[0] = sptr0[0];
                    pp[1] = sptr1[0];
                    pp[2] = sptr0[stride_w];
                    pp[3] = sptr1[stride_w];
                    pp += 4;
                }
            }
#endif // __ARM_NEON
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;

                const signed char* sptr = img.row<const signed char>(y0) + x0 * elempack;

#if __ARM_NEON
                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr);
                    int8x8_t _r1 = vld1_s8(sptr + stride_w * 8);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    pp += 16;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x2_t _r01;
                    _r01.val[0] = vreinterpret_s32_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s32_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s32((int*)pp, _r01);
                    pp += 16;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr + stride_w * 8));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                }
#endif // __ARM_NEON
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
#if __ARM_NEON
            if (elempack == 1)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int p4 = (k + kk + 4) / maxk;
                    int p5 = (k + kk + 5) / maxk;
                    int p6 = (k + kk + 6) / maxk;
                    int p7 = (k + kk + 7) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int uv4 = (k + kk + 4) % maxk;
                    int uv5 = (k + kk + 5) % maxk;
                    int uv6 = (k + kk + 6) % maxk;
                    int uv7 = (k + kk + 7) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int u4 = uv4 / kernel_w;
                    int u5 = uv5 / kernel_w;
                    int u6 = uv6 / kernel_w;
                    int u7 = uv7 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;
                    int v4 = uv4 % kernel_w;
                    int v5 = uv5 % kernel_w;
                    int v6 = uv6 % kernel_w;
                    int v7 = uv7 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);
                    const Mat img4 = bottom_blob.channel(p4);
                    const Mat img5 = bottom_blob.channel(p5);
                    const Mat img6 = bottom_blob.channel(p6);
                    const Mat img7 = bottom_blob.channel(p7);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;

                    int x40 = dx0 + dilation_w * v4;
                    int x41 = dx1 + dilation_w * v4;
                    int y40 = dy0 + dilation_h * u4;
                    int y41 = dy1 + dilation_h * u4;
                    int x50 = dx0 + dilation_w * v5;
                    int x51 = dx1 + dilation_w * v5;
                    int y50 = dy0 + dilation_h * u5;
                    int y51 = dy1 + dilation_h * u5;

                    int x60 = dx0 + dilation_w * v6;
                    int x61 = dx1 + dilation_w * v6;
                    int y60 = dy0 + dilation_h * u6;
                    int y61 = dy1 + dilation_h * u6;
                    int x70 = dx0 + dilation_w * v7;
                    int x71 = dx1 + dilation_w * v7;
                    int y70 = dy0 + dilation_h * u7;
                    int y71 = dy1 + dilation_h * u7;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;

                    const signed char* sptr40 = img4.row<const signed char>(y40) + x40;
                    const signed char* sptr41 = img4.row<const signed char>(y41) + x41;
                    const signed char* sptr50 = img5.row<const signed char>(y50) + x50;
                    const signed char* sptr51 = img5.row<const signed char>(y51) + x51;
                    const signed char* sptr60 = img6.row<const signed char>(y60) + x60;
                    const signed char* sptr61 = img6.row<const signed char>(y61) + x61;
                    const signed char* sptr70 = img7.row<const signed char>(y70) + x70;
                    const signed char* sptr71 = img7.row<const signed char>(y71) + x71;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr40[0];
                    pp[5] = sptr50[0];
                    pp[6] = sptr60[0];
                    pp[7] = sptr70[0];
                    pp[8] = sptr01[0];
                    pp[9] = sptr11[0];
                    pp[10] = sptr21[0];
                    pp[11] = sptr31[0];
                    pp[12] = sptr41[0];
                    pp[13] = sptr51[0];
                    pp[14] = sptr61[0];
                    pp[15] = sptr71[0];
                    pp += 16;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
                for (; kk + 3 < max_kk; kk += 4)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int p2 = (k + kk + 2) / maxk;
                    int p3 = (k + kk + 3) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int uv2 = (k + kk + 2) % maxk;
                    int uv3 = (k + kk + 3) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int u2 = uv2 / kernel_w;
                    int u3 = uv3 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;
                    int v2 = uv2 % kernel_w;
                    int v3 = uv3 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);
                    const Mat img2 = bottom_blob.channel(p2);
                    const Mat img3 = bottom_blob.channel(p3);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;
                    int x20 = dx0 + dilation_w * v2;
                    int x21 = dx1 + dilation_w * v2;
                    int y20 = dy0 + dilation_h * u2;
                    int y21 = dy1 + dilation_h * u2;
                    int x30 = dx0 + dilation_w * v3;
                    int x31 = dx1 + dilation_w * v3;
                    int y30 = dy0 + dilation_h * u3;
                    int y31 = dy1 + dilation_h * u3;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;
                    const signed char* sptr20 = img2.row<const signed char>(y20) + x20;
                    const signed char* sptr21 = img2.row<const signed char>(y21) + x21;
                    const signed char* sptr30 = img3.row<const signed char>(y30) + x30;
                    const signed char* sptr31 = img3.row<const signed char>(y31) + x31;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr20[0];
                    pp[3] = sptr30[0];
                    pp[4] = sptr01[0];
                    pp[5] = sptr11[0];
                    pp[6] = sptr21[0];
                    pp[7] = sptr31[0];
                    pp += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    int p0 = (k + kk) / maxk;
                    int p1 = (k + kk + 1) / maxk;
                    int uv0 = (k + kk) % maxk;
                    int uv1 = (k + kk + 1) % maxk;
                    int u0 = uv0 / kernel_w;
                    int u1 = uv1 / kernel_w;
                    int v0 = uv0 % kernel_w;
                    int v1 = uv1 % kernel_w;

                    const Mat img0 = bottom_blob.channel(p0);
                    const Mat img1 = bottom_blob.channel(p1);

                    int x00 = dx0 + dilation_w * v0;
                    int x01 = dx1 + dilation_w * v0;
                    int y00 = dy0 + dilation_h * u0;
                    int y01 = dy1 + dilation_h * u0;
                    int x10 = dx0 + dilation_w * v1;
                    int x11 = dx1 + dilation_w * v1;
                    int y10 = dy0 + dilation_h * u1;
                    int y11 = dy1 + dilation_h * u1;

                    const signed char* sptr00 = img0.row<const signed char>(y00) + x00;
                    const signed char* sptr01 = img0.row<const signed char>(y01) + x01;
                    const signed char* sptr10 = img1.row<const signed char>(y10) + x10;
                    const signed char* sptr11 = img1.row<const signed char>(y11) + x11;

                    pp[0] = sptr00[0];
                    pp[1] = sptr10[0];
                    pp[2] = sptr01[0];
                    pp[3] = sptr11[0];
                    pp += 4;
                }
            }
#endif // __ARM_NEON
            for (; kk < max_kk / elempack; kk++)
            {
                int p = (k / elempack + kk) / maxk;
                int uv = (k / elempack + kk) % maxk;
                int u = uv / kernel_w;
                int v = uv % kernel_w;

                const Mat img = bottom_blob.channel(p);

                int x0 = dx0 + dilation_w * v;
                int x1 = dx1 + dilation_w * v;
                int y0 = dy0 + dilation_h * u;
                int y1 = dy1 + dilation_h * u;

                const signed char* sptr0 = img.row<const signed char>(y0) + x0 * elempack;
                const signed char* sptr1 = img.row<const signed char>(y1) + x1 * elempack;

#if __ARM_NEON
                if (elempack == 8)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = vld1_s8(sptr0);
                    int8x8_t _r1 = vld1_s8(sptr1);
                    vst1_s8(pp, _r0);
                    vst1_s8(pp + 8, _r1);
                    pp += 16;
#elif __ARM_FEATURE_DOTPROD
                    int32x2x2_t _r01;
                    _r01.val[0] = vreinterpret_s32_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s32_s8(vld1_s8(sptr1));
                    vst2_s32((int*)pp, _r01);
                    pp += 16;
#else  // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(vld1_s8(sptr0));
                    _r01.val[1] = vreinterpret_s16_s8(vld1_s8(sptr1));
                    vst2_s16((short*)pp, _r01);
                    pp += 16;
#endif // __ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD
                }
#endif // __ARM_NEON
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
        int dy = (j + jj) / outw * stride_h;
        int dx = (j + jj) % outw * stride_w;

        int kk = 0;
        for (; kk < max_kk / elempack; kk++)
        {
            int p = (k / elempack + kk) / maxk;
            int uv = (k / elempack + kk) % maxk;
            int u = uv / kernel_w;
            int v = uv % kernel_w;

            const Mat img = bottom_blob.channel(p);

            int x = dx + dilation_w * v;
            int y = dy + dilation_h * u;

            const signed char* sptr = img.row<const signed char>(y) + x * elempack;

#if __ARM_NEON
            if (elempack == 8)
            {
                vst1_s8(pp, vld1_s8(sptr));
                pp += 8;
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                pp[0] = sptr[0];
                pp += 1;
            }
        }
    }
}

static void convolution_im2col_gemm_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        convolution_im2col_gemm_transform_kernel_int8_i8mm(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        convolution_im2col_gemm_transform_kernel_int8_asimddp(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
        return;
    }
#endif
#endif

    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = inch % 8 == 0 ? 8 : 1;
    }
#endif // __ARM_NEON

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        A_data = kernel.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)1u, 1);

        for (int q = 0; q < outch; q += 1)
        {
            signed char* g00 = A_data.row<signed char>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const signed char* k00 = weight_data_r2.channel(q).row<const signed char>(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)1u, 1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_int8(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static void convolution_im2col_gemm_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        convolution_im2col_gemm_int8_i8mm(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        convolution_im2col_gemm_int8_asimddp(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
        return;
    }
#endif
#endif

    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 1u, opt.workspace_allocator);

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
        convolution_im2col_input_tile_int8(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

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

                convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
            }
        }
    }
}
