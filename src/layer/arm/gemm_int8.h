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

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void pack_A_tile_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void gemm_transB_packed_tile_int8_i8mm(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void pack_A_tile_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void transpose_pack_A_tile_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void pack_B_tile_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void transpose_pack_B_tile_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk);
void pack_A_tile_fp32_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_fp32_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_fp32_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_fp32_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void unpack_output_tile_int32_to_fp32_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta);
void transpose_unpack_output_tile_int32_to_fp32_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta);
void gemm_transB_packed_tile_int8_asimddp(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_A_tile_int8_i8mm(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_A_tile_int8_asimddp(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
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
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
            int8x16_t _p2 = vld1q_s8(p2);
            int8x16_t _p3 = vld1q_s8(p3);
            int8x16_t _p4 = vld1q_s8(p4);
            int8x16_t _p5 = vld1q_s8(p5);
            int8x16_t _p6 = vld1q_s8(p6);
            int8x16_t _p7 = vld1q_s8(p7);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int8x16_t _r0 = vcombine_s8(vget_low_s8(_p0), vget_low_s8(_p1));
            int8x16_t _r1 = vcombine_s8(vget_low_s8(_p2), vget_low_s8(_p3));
            int8x16_t _r2 = vcombine_s8(vget_low_s8(_p4), vget_low_s8(_p5));
            int8x16_t _r3 = vcombine_s8(vget_low_s8(_p6), vget_low_s8(_p7));
            int8x16_t _r4 = vcombine_s8(vget_high_s8(_p0), vget_high_s8(_p1));
            int8x16_t _r5 = vcombine_s8(vget_high_s8(_p2), vget_high_s8(_p3));
            int8x16_t _r6 = vcombine_s8(vget_high_s8(_p4), vget_high_s8(_p5));
            int8x16_t _r7 = vcombine_s8(vget_high_s8(_p6), vget_high_s8(_p7));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x2_t _p01 = vzipq_s32(vreinterpretq_s32_s8(_p0), vreinterpretq_s32_s8(_p1));
            int32x4x2_t _p23 = vzipq_s32(vreinterpretq_s32_s8(_p2), vreinterpretq_s32_s8(_p3));
            int32x4x2_t _p45 = vzipq_s32(vreinterpretq_s32_s8(_p4), vreinterpretq_s32_s8(_p5));
            int32x4x2_t _p67 = vzipq_s32(vreinterpretq_s32_s8(_p6), vreinterpretq_s32_s8(_p7));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p01.val[0]), vget_low_s32(_p23.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p45.val[0]), vget_low_s32(_p67.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p01.val[0]), vget_high_s32(_p23.val[0])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p45.val[0]), vget_high_s32(_p67.val[0])));
            int8x16_t _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p01.val[1]), vget_low_s32(_p23.val[1])));
            int8x16_t _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p45.val[1]), vget_low_s32(_p67.val[1])));
            int8x16_t _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p01.val[1]), vget_high_s32(_p23.val[1])));
            int8x16_t _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p45.val[1]), vget_high_s32(_p67.val[1])));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x2_t _p01 = vzipq_s16(vreinterpretq_s16_s8(_p0), vreinterpretq_s16_s8(_p1));
            int16x8x2_t _p23 = vzipq_s16(vreinterpretq_s16_s8(_p2), vreinterpretq_s16_s8(_p3));
            int16x8x2_t _p45 = vzipq_s16(vreinterpretq_s16_s8(_p4), vreinterpretq_s16_s8(_p5));
            int16x8x2_t _p67 = vzipq_s16(vreinterpretq_s16_s8(_p6), vreinterpretq_s16_s8(_p7));
            int32x4x2_t _t0 = vzipq_s32(vreinterpretq_s32_s16(_p01.val[0]), vreinterpretq_s32_s16(_p23.val[0]));
            int32x4x2_t _t1 = vzipq_s32(vreinterpretq_s32_s16(_p01.val[1]), vreinterpretq_s32_s16(_p23.val[1]));
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_p45.val[0]), vreinterpretq_s32_s16(_p67.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_p45.val[1]), vreinterpretq_s32_s16(_p67.val[1]));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t2.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t2.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t2.val[1])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t2.val[1])));
            int8x16_t _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t3.val[0])));
            int8x16_t _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t3.val[0])));
            int8x16_t _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t3.val[1])));
            int8x16_t _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t3.val[1])));
#endif // __ARM_FEATURE_DOTPROD
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
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
            int8x8_t _p2 = vld1_s8(p2);
            int8x8_t _p3 = vld1_s8(p3);
            int8x8_t _p4 = vld1_s8(p4);
            int8x8_t _p5 = vld1_s8(p5);
            int8x8_t _p6 = vld1_s8(p6);
            int8x8_t _p7 = vld1_s8(p7);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int8x16_t _r0 = vcombine_s8(_p0, _p1);
            int8x16_t _r1 = vcombine_s8(_p2, _p3);
            int8x16_t _r2 = vcombine_s8(_p4, _p5);
            int8x16_t _r3 = vcombine_s8(_p6, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x2_t _p01 = vzip_s32(vreinterpret_s32_s8(_p0), vreinterpret_s32_s8(_p1));
            int32x2x2_t _p23 = vzip_s32(vreinterpret_s32_s8(_p2), vreinterpret_s32_s8(_p3));
            int32x2x2_t _p45 = vzip_s32(vreinterpret_s32_s8(_p4), vreinterpret_s32_s8(_p5));
            int32x2x2_t _p67 = vzip_s32(vreinterpret_s32_s8(_p6), vreinterpret_s32_s8(_p7));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(_p01.val[0], _p23.val[0]));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(_p45.val[0], _p67.val[0]));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(_p01.val[1], _p23.val[1]));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(_p45.val[1], _p67.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8_t _p04 = vreinterpretq_s16_s8(vcombine_s8(_p0, _p4));
            int16x8_t _p15 = vreinterpretq_s16_s8(vcombine_s8(_p1, _p5));
            int16x8_t _p26 = vreinterpretq_s16_s8(vcombine_s8(_p2, _p6));
            int16x8_t _p37 = vreinterpretq_s16_s8(vcombine_s8(_p3, _p7));
            int16x8x2_t _t0 = vzipq_s16(_p04, _p15);
            int16x8x2_t _t1 = vzipq_s16(_p26, _p37);
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[0]), vreinterpretq_s32_s16(_t1.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[1]), vreinterpretq_s32_s16(_t1.val[1]));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1])));
#endif // __ARM_FEATURE_DOTPROD
            vst1q_s8(pp, _r0);
            vst1q_s8(pp + 16, _r1);
            vst1q_s8(pp + 32, _r2);
            vst1q_s8(pp + 48, _r3);
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
#if __ARM_FEATURE_DOTPROD
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
#else  // __ARM_FEATURE_DOTPROD
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
            pp[16] = p0[2];
            pp[17] = p0[3];
            pp[18] = p1[2];
            pp[19] = p1[3];
            pp[20] = p2[2];
            pp[21] = p2[3];
            pp[22] = p3[2];
            pp[23] = p3[3];
            pp[24] = p4[2];
            pp[25] = p4[3];
            pp[26] = p5[2];
            pp[27] = p5[3];
            pp[28] = p6[2];
            pp[29] = p6[3];
            pp[30] = p7[2];
            pp[31] = p7[3];
#endif // __ARM_FEATURE_DOTPROD
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
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;
        const signed char* p2 = A.row<const signed char>(i + ii + 2) + k;
        const signed char* p3 = A.row<const signed char>(i + ii + 3) + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
            int8x16_t _p2 = vld1q_s8(p2);
            int8x16_t _p3 = vld1q_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int64x2x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s64_s8(_p0);
            _r0123.val[1] = vreinterpretq_s64_s8(_p1);
            _r0123.val[2] = vreinterpretq_s64_s8(_p2);
            _r0123.val[3] = vreinterpretq_s64_s8(_p3);
            vst4q_s64((int64_t*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s32_s8(_p0);
            _r0123.val[1] = vreinterpretq_s32_s8(_p1);
            _r0123.val[2] = vreinterpretq_s32_s8(_p2);
            _r0123.val[3] = vreinterpretq_s32_s8(_p3);
            vst4q_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s16_s8(_p0);
            _r0123.val[1] = vreinterpretq_s16_s8(_p1);
            _r0123.val[2] = vreinterpretq_s16_s8(_p2);
            _r0123.val[3] = vreinterpretq_s16_s8(_p3);
            vst4q_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
            int8x8_t _p2 = vld1_s8(p2);
            int8x8_t _p3 = vld1_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            vst1q_s8(pp, vcombine_s8(_p0, _p1));
            vst1q_s8(pp + 16, vcombine_s8(_p2, _p3));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x4_t _r0123;
            _r0123.val[0] = vreinterpret_s32_s8(_p0);
            _r0123.val[1] = vreinterpret_s32_s8(_p1);
            _r0123.val[2] = vreinterpret_s32_s8(_p2);
            _r0123.val[3] = vreinterpret_s32_s8(_p3);
            vst4_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x4x4_t _r0123;
            _r0123.val[0] = vreinterpret_s16_s8(_p0);
            _r0123.val[1] = vreinterpret_s16_s8(_p1);
            _r0123.val[2] = vreinterpret_s16_s8(_p2);
            _r0123.val[3] = vreinterpret_s16_s8(_p3);
            vst4_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
#if __ARM_FEATURE_DOTPROD
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
#else  // __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p0[3];
            pp[10] = p1[2];
            pp[11] = p1[3];
            pp[12] = p2[2];
            pp[13] = p2[3];
            pp[14] = p3[2];
            pp[15] = p3[3];
#endif // __ARM_FEATURE_DOTPROD
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
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
        const signed char* p0 = A.row<const signed char>(i + ii) + k;
        const signed char* p1 = A.row<const signed char>(i + ii + 1) + k;

        int kk = 0;
#if __ARM_NEON
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int64x2x2_t _r01;
            _r01.val[0] = vreinterpretq_s64_s8(_p0);
            _r01.val[1] = vreinterpretq_s64_s8(_p1);
            vst2q_s64((int64_t*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x2_t _r01;
            _r01.val[0] = vreinterpretq_s32_s8(_p0);
            _r01.val[1] = vreinterpretq_s32_s8(_p1);
            vst2q_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x2_t _r01;
            _r01.val[0] = vreinterpretq_s16_s8(_p0);
            _r01.val[1] = vreinterpretq_s16_s8(_p1);
            vst2q_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            vst1q_s8(pp, vcombine_s8(_p0, _p1));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x2_t _r01;
            _r01.val[0] = vreinterpret_s32_s8(_p0);
            _r01.val[1] = vreinterpret_s32_s8(_p1);
            vst2_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x4x2_t _r01;
            _r01.val[0] = vreinterpret_s16_s8(_p0);
            _r01.val[1] = vreinterpret_s16_s8(_p1);
            vst2_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
#if __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
#else  // __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p0[2];
            pp[5] = p0[3];
            pp[6] = p1[2];
            pp[7] = p1[3];
#endif // __ARM_FEATURE_DOTPROD
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
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
        const signed char* p0 = A.row<const signed char>(i + ii) + k;

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

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_A_tile_int8_i8mm(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_A_tile_int8_asimddp(A, AT, i, max_ii, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_A_tile_int8");
    // assert A.elempack == 1
    // assert A.dims == 2

    const int A_hstep = A.w;

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p0 + A_hstep);
            int8x8_t _r2 = vld1_s8(p0 + A_hstep * 2);
            int8x8_t _r3 = vld1_s8(p0 + A_hstep * 3);
            int8x8_t _r4 = vld1_s8(p0 + A_hstep * 4);
            int8x8_t _r5 = vld1_s8(p0 + A_hstep * 5);
            int8x8_t _r6 = vld1_s8(p0 + A_hstep * 6);
            int8x8_t _r7 = vld1_s8(p0 + A_hstep * 7);
            // transpose8x8
            int8x8x2_t _r04 = vzip_s8(_r0, _r4);
            int8x8x2_t _r15 = vzip_s8(_r1, _r5);
            int8x8x2_t _r26 = vzip_s8(_r2, _r6);
            int8x8x2_t _r37 = vzip_s8(_r3, _r7);
            int8x8x4_t _r0123;
            _r0123.val[0] = _r04.val[0];
            _r0123.val[1] = _r15.val[0];
            _r0123.val[2] = _r26.val[0];
            _r0123.val[3] = _r37.val[0];
            int8x8x4_t _r4567;
            _r4567.val[0] = _r04.val[1];
            _r4567.val[1] = _r15.val[1];
            _r4567.val[2] = _r26.val[1];
            _r4567.val[3] = _r37.val[1];
            vst4_s8(pp, _r0123);
            vst4_s8(pp + 32, _r4567);
            pp += 64;
            p0 += A_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 3 < max_kk; kk += 4)
        {
            int8x8x4_t _r0123;
            _r0123.val[0] = vld1_s8(p0);
            _r0123.val[1] = vld1_s8(p0 + A_hstep);
            _r0123.val[2] = vld1_s8(p0 + A_hstep * 2);
            _r0123.val[3] = vld1_s8(p0 + A_hstep * 3);
            vst4_s8(pp, _r0123);
            pp += 32;
            p0 += A_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
        for (; kk + 1 < max_kk; kk += 2)
        {
            int8x8x2_t _r01;
            _r01.val[0] = vld1_s8(p0);
            _r01.val[1] = vld1_s8(p0 + A_hstep);
            vst2_s8(pp, _r01);
            pp += 16;
            p0 += A_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            vst1_s8(pp, vld1_s8(p0));
            pp += 8;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[A_hstep * 4];
            pp[5] = p0[A_hstep * 5];
            pp[6] = p0[A_hstep * 6];
            pp[7] = p0[A_hstep * 7];
            pp[8] = p0[1];
            pp[9] = p0[A_hstep + 1];
            pp[10] = p0[A_hstep * 2 + 1];
            pp[11] = p0[A_hstep * 3 + 1];
            pp[12] = p0[A_hstep * 4 + 1];
            pp[13] = p0[A_hstep * 5 + 1];
            pp[14] = p0[A_hstep * 6 + 1];
            pp[15] = p0[A_hstep * 7 + 1];
            pp[16] = p0[2];
            pp[17] = p0[A_hstep + 2];
            pp[18] = p0[A_hstep * 2 + 2];
            pp[19] = p0[A_hstep * 3 + 2];
            pp[20] = p0[A_hstep * 4 + 2];
            pp[21] = p0[A_hstep * 5 + 2];
            pp[22] = p0[A_hstep * 6 + 2];
            pp[23] = p0[A_hstep * 7 + 2];
            pp[24] = p0[3];
            pp[25] = p0[A_hstep + 3];
            pp[26] = p0[A_hstep * 2 + 3];
            pp[27] = p0[A_hstep * 3 + 3];
            pp[28] = p0[A_hstep * 4 + 3];
            pp[29] = p0[A_hstep * 5 + 3];
            pp[30] = p0[A_hstep * 6 + 3];
            pp[31] = p0[A_hstep * 7 + 3];
            pp += 32;
            p0 += A_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
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
            pp += 16;
            p0 += A_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = A.row<const signed char>(k) + (i + ii);

        int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp[4] = p0[A_hstep * 4];
            pp[5] = p0[A_hstep * 5];
            pp[6] = p0[A_hstep * 6];
            pp[7] = p0[A_hstep * 7];
            pp[8] = p0[1];
            pp[9] = p0[A_hstep + 1];
            pp[10] = p0[A_hstep * 2 + 1];
            pp[11] = p0[A_hstep * 3 + 1];
            pp[12] = p0[A_hstep * 4 + 1];
            pp[13] = p0[A_hstep * 5 + 1];
            pp[14] = p0[A_hstep * 6 + 1];
            pp[15] = p0[A_hstep * 7 + 1];
            pp += 16;
            p0 += A_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
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
            pp += 8;
            p0 += A_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[1];
            pp[3] = p0[A_hstep + 1];
            pp += 4;
            p0 += A_hstep * 2;
        }
#endif // __ARM_NEON
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
        // for (; kk + 1 < max_kk; kk += 2)
        // {
        //     pp[0] = p0[0];
        //     pp[1] = p0[A_hstep];
        //     pp += 2;
        //     p0 += A_hstep * 2;
        // }
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
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_B_tile_int8_i8mm(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_B_tile_int8_asimddp(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    signed char* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
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
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
            int8x16_t _p2 = vld1q_s8(p2);
            int8x16_t _p3 = vld1q_s8(p3);
            int8x16_t _p4 = vld1q_s8(p4);
            int8x16_t _p5 = vld1q_s8(p5);
            int8x16_t _p6 = vld1q_s8(p6);
            int8x16_t _p7 = vld1q_s8(p7);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int8x16_t _r0 = vcombine_s8(vget_low_s8(_p0), vget_low_s8(_p1));
            int8x16_t _r1 = vcombine_s8(vget_low_s8(_p2), vget_low_s8(_p3));
            int8x16_t _r2 = vcombine_s8(vget_low_s8(_p4), vget_low_s8(_p5));
            int8x16_t _r3 = vcombine_s8(vget_low_s8(_p6), vget_low_s8(_p7));
            int8x16_t _r4 = vcombine_s8(vget_high_s8(_p0), vget_high_s8(_p1));
            int8x16_t _r5 = vcombine_s8(vget_high_s8(_p2), vget_high_s8(_p3));
            int8x16_t _r6 = vcombine_s8(vget_high_s8(_p4), vget_high_s8(_p5));
            int8x16_t _r7 = vcombine_s8(vget_high_s8(_p6), vget_high_s8(_p7));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x2_t _p01 = vzipq_s32(vreinterpretq_s32_s8(_p0), vreinterpretq_s32_s8(_p1));
            int32x4x2_t _p23 = vzipq_s32(vreinterpretq_s32_s8(_p2), vreinterpretq_s32_s8(_p3));
            int32x4x2_t _p45 = vzipq_s32(vreinterpretq_s32_s8(_p4), vreinterpretq_s32_s8(_p5));
            int32x4x2_t _p67 = vzipq_s32(vreinterpretq_s32_s8(_p6), vreinterpretq_s32_s8(_p7));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p01.val[0]), vget_low_s32(_p23.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p45.val[0]), vget_low_s32(_p67.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p01.val[0]), vget_high_s32(_p23.val[0])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p45.val[0]), vget_high_s32(_p67.val[0])));
            int8x16_t _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p01.val[1]), vget_low_s32(_p23.val[1])));
            int8x16_t _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_p45.val[1]), vget_low_s32(_p67.val[1])));
            int8x16_t _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p01.val[1]), vget_high_s32(_p23.val[1])));
            int8x16_t _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_p45.val[1]), vget_high_s32(_p67.val[1])));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x2_t _p01 = vzipq_s16(vreinterpretq_s16_s8(_p0), vreinterpretq_s16_s8(_p1));
            int16x8x2_t _p23 = vzipq_s16(vreinterpretq_s16_s8(_p2), vreinterpretq_s16_s8(_p3));
            int16x8x2_t _p45 = vzipq_s16(vreinterpretq_s16_s8(_p4), vreinterpretq_s16_s8(_p5));
            int16x8x2_t _p67 = vzipq_s16(vreinterpretq_s16_s8(_p6), vreinterpretq_s16_s8(_p7));
            int32x4x2_t _t0 = vzipq_s32(vreinterpretq_s32_s16(_p01.val[0]), vreinterpretq_s32_s16(_p23.val[0]));
            int32x4x2_t _t1 = vzipq_s32(vreinterpretq_s32_s16(_p01.val[1]), vreinterpretq_s32_s16(_p23.val[1]));
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_p45.val[0]), vreinterpretq_s32_s16(_p67.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_p45.val[1]), vreinterpretq_s32_s16(_p67.val[1]));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[0]), vget_low_s32(_t2.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[0]), vget_high_s32(_t2.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t0.val[1]), vget_low_s32(_t2.val[1])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t0.val[1]), vget_high_s32(_t2.val[1])));
            int8x16_t _r4 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[0]), vget_low_s32(_t3.val[0])));
            int8x16_t _r5 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[0]), vget_high_s32(_t3.val[0])));
            int8x16_t _r6 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t1.val[1]), vget_low_s32(_t3.val[1])));
            int8x16_t _r7 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t1.val[1]), vget_high_s32(_t3.val[1])));
#endif // __ARM_FEATURE_DOTPROD
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
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
            int8x8_t _p2 = vld1_s8(p2);
            int8x8_t _p3 = vld1_s8(p3);
            int8x8_t _p4 = vld1_s8(p4);
            int8x8_t _p5 = vld1_s8(p5);
            int8x8_t _p6 = vld1_s8(p6);
            int8x8_t _p7 = vld1_s8(p7);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int8x16_t _r0 = vcombine_s8(_p0, _p1);
            int8x16_t _r1 = vcombine_s8(_p2, _p3);
            int8x16_t _r2 = vcombine_s8(_p4, _p5);
            int8x16_t _r3 = vcombine_s8(_p6, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x2_t _p01 = vzip_s32(vreinterpret_s32_s8(_p0), vreinterpret_s32_s8(_p1));
            int32x2x2_t _p23 = vzip_s32(vreinterpret_s32_s8(_p2), vreinterpret_s32_s8(_p3));
            int32x2x2_t _p45 = vzip_s32(vreinterpret_s32_s8(_p4), vreinterpret_s32_s8(_p5));
            int32x2x2_t _p67 = vzip_s32(vreinterpret_s32_s8(_p6), vreinterpret_s32_s8(_p7));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(_p01.val[0], _p23.val[0]));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(_p45.val[0], _p67.val[0]));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(_p01.val[1], _p23.val[1]));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(_p45.val[1], _p67.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8_t _p04 = vreinterpretq_s16_s8(vcombine_s8(_p0, _p4));
            int16x8_t _p15 = vreinterpretq_s16_s8(vcombine_s8(_p1, _p5));
            int16x8_t _p26 = vreinterpretq_s16_s8(vcombine_s8(_p2, _p6));
            int16x8_t _p37 = vreinterpretq_s16_s8(vcombine_s8(_p3, _p7));
            int16x8x2_t _t0 = vzipq_s16(_p04, _p15);
            int16x8x2_t _t1 = vzipq_s16(_p26, _p37);
            int32x4x2_t _t2 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[0]), vreinterpretq_s32_s16(_t1.val[0]));
            int32x4x2_t _t3 = vzipq_s32(vreinterpretq_s32_s16(_t0.val[1]), vreinterpretq_s32_s16(_t1.val[1]));
            int8x16_t _r0 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0])));
            int8x16_t _r1 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0])));
            int8x16_t _r2 = vreinterpretq_s8_s32(vcombine_s32(vget_low_s32(_t2.val[1]), vget_low_s32(_t3.val[1])));
            int8x16_t _r3 = vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(_t2.val[1]), vget_high_s32(_t3.val[1])));
#endif // __ARM_FEATURE_DOTPROD
            vst1q_s8(pp, _r0);
            vst1q_s8(pp + 16, _r1);
            vst1q_s8(pp + 32, _r2);
            vst1q_s8(pp + 48, _r3);
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
#if __ARM_FEATURE_DOTPROD
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
#else  // __ARM_FEATURE_DOTPROD
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
            pp[16] = p0[2];
            pp[17] = p0[3];
            pp[18] = p1[2];
            pp[19] = p1[3];
            pp[20] = p2[2];
            pp[21] = p2[3];
            pp[22] = p3[2];
            pp[23] = p3[3];
            pp[24] = p4[2];
            pp[25] = p4[3];
            pp[26] = p5[2];
            pp[27] = p5[3];
            pp[28] = p6[2];
            pp[29] = p6[3];
            pp[30] = p7[2];
            pp[31] = p7[3];
#endif // __ARM_FEATURE_DOTPROD
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
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;
        const signed char* p2 = B.row<const signed char>(j + jj + 2) + k;
        const signed char* p3 = B.row<const signed char>(j + jj + 3) + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
            int8x16_t _p2 = vld1q_s8(p2);
            int8x16_t _p3 = vld1q_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int64x2x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s64_s8(_p0);
            _r0123.val[1] = vreinterpretq_s64_s8(_p1);
            _r0123.val[2] = vreinterpretq_s64_s8(_p2);
            _r0123.val[3] = vreinterpretq_s64_s8(_p3);
            vst4q_s64((int64_t*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s32_s8(_p0);
            _r0123.val[1] = vreinterpretq_s32_s8(_p1);
            _r0123.val[2] = vreinterpretq_s32_s8(_p2);
            _r0123.val[3] = vreinterpretq_s32_s8(_p3);
            vst4q_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x4_t _r0123;
            _r0123.val[0] = vreinterpretq_s16_s8(_p0);
            _r0123.val[1] = vreinterpretq_s16_s8(_p1);
            _r0123.val[2] = vreinterpretq_s16_s8(_p2);
            _r0123.val[3] = vreinterpretq_s16_s8(_p3);
            vst4q_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
            int8x8_t _p2 = vld1_s8(p2);
            int8x8_t _p3 = vld1_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            vst1q_s8(pp, vcombine_s8(_p0, _p1));
            vst1q_s8(pp + 16, vcombine_s8(_p2, _p3));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x4_t _r0123;
            _r0123.val[0] = vreinterpret_s32_s8(_p0);
            _r0123.val[1] = vreinterpret_s32_s8(_p1);
            _r0123.val[2] = vreinterpret_s32_s8(_p2);
            _r0123.val[3] = vreinterpret_s32_s8(_p3);
            vst4_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x4x4_t _r0123;
            _r0123.val[0] = vreinterpret_s16_s8(_p0);
            _r0123.val[1] = vreinterpret_s16_s8(_p1);
            _r0123.val[2] = vreinterpret_s16_s8(_p2);
            _r0123.val[3] = vreinterpret_s16_s8(_p3);
            vst4_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
#if __ARM_FEATURE_DOTPROD
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
#else  // __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p2[0];
            pp[5] = p2[1];
            pp[6] = p3[0];
            pp[7] = p3[1];
            pp[8] = p0[2];
            pp[9] = p0[3];
            pp[10] = p1[2];
            pp[11] = p1[3];
            pp[12] = p2[2];
            pp[13] = p2[3];
            pp[14] = p3[2];
            pp[15] = p3[3];
#endif // __ARM_FEATURE_DOTPROD
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
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
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;
        const signed char* p1 = B.row<const signed char>(j + jj + 1) + k;

        int kk = 0;
#if __ARM_NEON
        for (; kk + 15 < max_kk; kk += 16)
        {
            int8x16_t _p0 = vld1q_s8(p0);
            int8x16_t _p1 = vld1q_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int64x2x2_t _r01;
            _r01.val[0] = vreinterpretq_s64_s8(_p0);
            _r01.val[1] = vreinterpretq_s64_s8(_p1);
            vst2q_s64((int64_t*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4x2_t _r01;
            _r01.val[0] = vreinterpretq_s32_s8(_p0);
            _r01.val[1] = vreinterpretq_s32_s8(_p1);
            vst2q_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x8x2_t _r01;
            _r01.val[0] = vreinterpretq_s16_s8(_p0);
            _r01.val[1] = vreinterpretq_s16_s8(_p1);
            vst2q_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
            pp += 32;
            p0 += 16;
            p1 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _p0 = vld1_s8(p0);
            int8x8_t _p1 = vld1_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            vst1q_s8(pp, vcombine_s8(_p0, _p1));
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x2x2_t _r01;
            _r01.val[0] = vreinterpret_s32_s8(_p0);
            _r01.val[1] = vreinterpret_s32_s8(_p1);
            vst2_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
            int16x4x2_t _r01;
            _r01.val[0] = vreinterpret_s16_s8(_p0);
            _r01.val[1] = vreinterpret_s16_s8(_p1);
            vst2_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
            pp += 16;
            p0 += 8;
            p1 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
#if __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
#else  // __ARM_FEATURE_DOTPROD
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p1[0];
            pp[3] = p1[1];
            pp[4] = p0[2];
            pp[5] = p0[3];
            pp[6] = p1[2];
            pp[7] = p1[3];
#endif // __ARM_FEATURE_DOTPROD
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
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
    for (; jj < max_jj; jj += 1)
    {
        const signed char* p0 = B.row<const signed char>(j + jj) + k;

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

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_B_tile_int8_i8mm(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_B_tile_int8_asimddp(B, BT, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("transpose_pack_B_tile_int8");
    // assert B.elempack == 1
    // assert B.dims == 2

    const int B_hstep = B.w;

    signed char* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            int8x8_t _r0 = vld1_s8(p0);
            int8x8_t _r1 = vld1_s8(p0 + B_hstep);
            int8x8_t _r2 = vld1_s8(p0 + B_hstep * 2);
            int8x8_t _r3 = vld1_s8(p0 + B_hstep * 3);
            int8x8_t _r4 = vld1_s8(p0 + B_hstep * 4);
            int8x8_t _r5 = vld1_s8(p0 + B_hstep * 5);
            int8x8_t _r6 = vld1_s8(p0 + B_hstep * 6);
            int8x8_t _r7 = vld1_s8(p0 + B_hstep * 7);
            // transpose8x8
            int8x8x2_t _r04 = vzip_s8(_r0, _r4);
            int8x8x2_t _r15 = vzip_s8(_r1, _r5);
            int8x8x2_t _r26 = vzip_s8(_r2, _r6);
            int8x8x2_t _r37 = vzip_s8(_r3, _r7);
            int8x8x4_t _r0123;
            _r0123.val[0] = _r04.val[0];
            _r0123.val[1] = _r15.val[0];
            _r0123.val[2] = _r26.val[0];
            _r0123.val[3] = _r37.val[0];
            int8x8x4_t _r4567;
            _r4567.val[0] = _r04.val[1];
            _r4567.val[1] = _r15.val[1];
            _r4567.val[2] = _r26.val[1];
            _r4567.val[3] = _r37.val[1];
            vst4_s8(pp, _r0123);
            vst4_s8(pp + 32, _r4567);
            pp += 64;
            p0 += B_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 3 < max_kk; kk += 4)
        {
            int8x8x4_t _r0123;
            _r0123.val[0] = vld1_s8(p0);
            _r0123.val[1] = vld1_s8(p0 + B_hstep);
            _r0123.val[2] = vld1_s8(p0 + B_hstep * 2);
            _r0123.val[3] = vld1_s8(p0 + B_hstep * 3);
            vst4_s8(pp, _r0123);
            pp += 32;
            p0 += B_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
        for (; kk + 1 < max_kk; kk += 2)
        {
            int8x8x2_t _r01;
            _r01.val[0] = vld1_s8(p0);
            _r01.val[1] = vld1_s8(p0 + B_hstep);
            vst2_s8(pp, _r01);
            pp += 16;
            p0 += B_hstep * 2;
        }
        for (; kk < max_kk; kk++)
        {
            vst1_s8(pp, vld1_s8(p0));
            pp += 8;
            p0 += B_hstep;
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp[4] = p0[B_hstep * 4];
            pp[5] = p0[B_hstep * 5];
            pp[6] = p0[B_hstep * 6];
            pp[7] = p0[B_hstep * 7];
            pp[8] = p0[1];
            pp[9] = p0[B_hstep + 1];
            pp[10] = p0[B_hstep * 2 + 1];
            pp[11] = p0[B_hstep * 3 + 1];
            pp[12] = p0[B_hstep * 4 + 1];
            pp[13] = p0[B_hstep * 5 + 1];
            pp[14] = p0[B_hstep * 6 + 1];
            pp[15] = p0[B_hstep * 7 + 1];
            pp[16] = p0[2];
            pp[17] = p0[B_hstep + 2];
            pp[18] = p0[B_hstep * 2 + 2];
            pp[19] = p0[B_hstep * 3 + 2];
            pp[20] = p0[B_hstep * 4 + 2];
            pp[21] = p0[B_hstep * 5 + 2];
            pp[22] = p0[B_hstep * 6 + 2];
            pp[23] = p0[B_hstep * 7 + 2];
            pp[24] = p0[3];
            pp[25] = p0[B_hstep + 3];
            pp[26] = p0[B_hstep * 2 + 3];
            pp[27] = p0[B_hstep * 3 + 3];
            pp[28] = p0[B_hstep * 4 + 3];
            pp[29] = p0[B_hstep * 5 + 3];
            pp[30] = p0[B_hstep * 6 + 3];
            pp[31] = p0[B_hstep * 7 + 3];
            pp += 32;
            p0 += B_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[B_hstep + 1];
            pp[6] = p0[B_hstep * 2 + 1];
            pp[7] = p0[B_hstep * 3 + 1];
            pp[8] = p0[2];
            pp[9] = p0[B_hstep + 2];
            pp[10] = p0[B_hstep * 2 + 2];
            pp[11] = p0[B_hstep * 3 + 2];
            pp[12] = p0[3];
            pp[13] = p0[B_hstep + 3];
            pp[14] = p0[B_hstep * 2 + 3];
            pp[15] = p0[B_hstep * 3 + 3];
            pp += 16;
            p0 += B_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = B.row<const signed char>(k) + (j + jj);

        int kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
        for (; kk + 7 < max_kk; kk += 8)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp[4] = p0[B_hstep * 4];
            pp[5] = p0[B_hstep * 5];
            pp[6] = p0[B_hstep * 6];
            pp[7] = p0[B_hstep * 7];
            pp[8] = p0[1];
            pp[9] = p0[B_hstep + 1];
            pp[10] = p0[B_hstep * 2 + 1];
            pp[11] = p0[B_hstep * 3 + 1];
            pp[12] = p0[B_hstep * 4 + 1];
            pp[13] = p0[B_hstep * 5 + 1];
            pp[14] = p0[B_hstep * 6 + 1];
            pp[15] = p0[B_hstep * 7 + 1];
            pp += 16;
            p0 += B_hstep * 8;
        }
#endif // __ARM_FEATURE_MATMUL_INT8
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp[4] = p0[1];
            pp[5] = p0[B_hstep + 1];
            pp[6] = p0[B_hstep * 2 + 1];
            pp[7] = p0[B_hstep * 3 + 1];
            pp += 8;
            p0 += B_hstep * 4;
        }
#endif // __ARM_FEATURE_DOTPROD
        for (; kk + 1 < max_kk; kk += 2)
        {
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[1];
            pp[3] = p0[B_hstep + 1];
            pp += 4;
            p0 += B_hstep * 2;
        }
#endif // __ARM_NEON
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
        // for (; kk + 1 < max_kk; kk += 2)
        // {
        //     pp[0] = p0[0];
        //     pp[1] = p0[B_hstep];
        //     pp += 2;
        //     p0 += B_hstep * 2;
        // }
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

#if __ARM_NEON
    if (elempack == 4)
    {
#if __aarch64__
        float32x4_t _v127 = vdupq_n_f32(127.f);
        float32x4_t _v127_B_scale = vdupq_n_f32(v127_B_scale);
#endif

        for (int ii = 0; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                p0 += 16;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax2);
            _absmax1 = vmaxq_f32(_absmax1, _absmax3);
            for (; kk + 1 < K; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = vld1q_f32(p0);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                p0 += 4;
            }

#if __aarch64__
            float32x4_t _scale = vdivq_f32(_v127, _absmax0);
            float32x4_t _out_descale = vdivq_f32(_absmax0, _v127_B_scale);

            vst1q_f32(ps, _scale);
            vst1q_f32(pods, _out_descale);
#else
            // float32x4_t _recp_absmax = vrecpeq_f32(_absmax0);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // float32x4_t _scale = vmulq_f32(_v127, _recp_absmax);
            // float32x4_t _out_descale = vmulq_f32(_absmax0, _recp_v127_B_scale);

            float tmp[4];
            vst1q_f32(tmp, _absmax0);

            ps[0] = 127.f / tmp[0];
            ps[1] = 127.f / tmp[1];
            ps[2] = 127.f / tmp[2];
            ps[3] = 127.f / tmp[3];

            pods[0] = tmp[0] / v127_B_scale;
            pods[1] = tmp[1] / v127_B_scale;
            pods[2] = tmp[2] / v127_B_scale;
            pods[3] = tmp[3] / v127_B_scale;

#endif
            ps += 4;
            pods += 4;
        }
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
        for (int ii = 0; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep;

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            for (; kk + 15 < K; kk += 16)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                p0 += 16;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax2);
            _absmax1 = vmaxq_f32(_absmax1, _absmax3);
            for (; kk + 7 < K; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p = vld1q_f32(p0);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                p0 += 4;
            }
            float32x2_t _aa = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            absmax = std::max(absmax, std::max(vget_lane_f32(_aa, 0), vget_lane_f32(_aa, 1)));
#endif // __ARM_NEON
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
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_A_tile_fp32_to_int8_i8mm(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_A_tile_fp32_to_int8_asimddp(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        float32x4_t _scale0 = vld1q_f32((const float*)scales + i + ii);
        float32x4_t _scale1 = vld1q_f32((const float*)scales + i + ii + 4);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + 16);
                float32x4x4_t _r = vld4q_f32(p0 + A_hstep * 4);
                float32x4x4_t _s = vld4q_f32(p0 + A_hstep * 4 + 16);

                float32x4_t _p0 = vmulq_laneq_f32(_p.val[0], _scale0, 0);
                float32x4_t _p1 = vmulq_laneq_f32(_p.val[1], _scale0, 1);
                float32x4_t _p2 = vmulq_laneq_f32(_p.val[2], _scale0, 2);
                float32x4_t _p3 = vmulq_laneq_f32(_p.val[3], _scale0, 3);
                float32x4_t _p4 = vmulq_laneq_f32(_q.val[0], _scale0, 0);
                float32x4_t _p5 = vmulq_laneq_f32(_q.val[1], _scale0, 1);
                float32x4_t _p6 = vmulq_laneq_f32(_q.val[2], _scale0, 2);
                float32x4_t _p7 = vmulq_laneq_f32(_q.val[3], _scale0, 3);
                float32x4_t _p8 = vmulq_laneq_f32(_r.val[0], _scale1, 0);
                float32x4_t _p9 = vmulq_laneq_f32(_r.val[1], _scale1, 1);
                float32x4_t _pa = vmulq_laneq_f32(_r.val[2], _scale1, 2);
                float32x4_t _pb = vmulq_laneq_f32(_r.val[3], _scale1, 3);
                float32x4_t _pc = vmulq_laneq_f32(_s.val[0], _scale1, 0);
                float32x4_t _pd = vmulq_laneq_f32(_s.val[1], _scale1, 1);
                float32x4_t _pe = vmulq_laneq_f32(_s.val[2], _scale1, 2);
                float32x4_t _pf = vmulq_laneq_f32(_s.val[3], _scale1, 3);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
                int8x8_t _r4 = float2int8(_p8, _pc);
                int8x8_t _r5 = float2int8(_p9, _pd);
                int8x8_t _r6 = float2int8(_pa, _pe);
                int8x8_t _r7 = float2int8(_pb, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p8, _p9);
                int8x8_t _r3 = float2int8(_pa, _pb);
                int8x8_t _r4 = float2int8(_p4, _p5);
                int8x8_t _r5 = float2int8(_p6, _p7);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);
                float32x4_t _p8 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + A_hstep * 4 + 8);
                float32x4_t _pb = vld1q_f32(p0 + A_hstep * 4 + 12);
                float32x4_t _pc = vld1q_f32(p0 + A_hstep * 4 + 16);
                float32x4_t _pd = vld1q_f32(p0 + A_hstep * 4 + 20);
                float32x4_t _pe = vld1q_f32(p0 + A_hstep * 4 + 24);
                float32x4_t _pf = vld1q_f32(p0 + A_hstep * 4 + 28);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale0);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale0);
                _p4 = vmulq_f32(_p4, _scale0);
                _p5 = vmulq_f32(_p5, _scale0);
                _p6 = vmulq_f32(_p6, _scale0);
                _p7 = vmulq_f32(_p7, _scale0);
                _p8 = vmulq_f32(_p8, _scale1);
                _p9 = vmulq_f32(_p9, _scale1);
                _pa = vmulq_f32(_pa, _scale1);
                _pb = vmulq_f32(_pb, _scale1);
                _pc = vmulq_f32(_pc, _scale1);
                _pd = vmulq_f32(_pd, _scale1);
                _pe = vmulq_f32(_pe, _scale1);
                _pf = vmulq_f32(_pf, _scale1);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p8), float2int8(_p2, _pa));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p9), float2int8(_p3, _pb));
                int8x16x2_t _r23;
                _r23.val[0] = vcombine_s8(float2int8(_p4, _pc), float2int8(_p6, _pe));
                _r23.val[1] = vcombine_s8(float2int8(_p5, _pd), float2int8(_p7, _pf));

                vst2q_s8(pp, _r01);
                vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += 32;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + A_hstep * 4);

                float32x4_t _p0 = vmulq_laneq_f32(_p.val[0], _scale0, 0);
                float32x4_t _p1 = vmulq_laneq_f32(_p.val[1], _scale0, 1);
                float32x4_t _p2 = vmulq_laneq_f32(_p.val[2], _scale0, 2);
                float32x4_t _p3 = vmulq_laneq_f32(_p.val[3], _scale0, 3);
                float32x4_t _p4 = vmulq_laneq_f32(_q.val[0], _scale1, 0);
                float32x4_t _p5 = vmulq_laneq_f32(_q.val[1], _scale1, 1);
                float32x4_t _p6 = vmulq_laneq_f32(_q.val[2], _scale1, 2);
                float32x4_t _p7 = vmulq_laneq_f32(_q.val[3], _scale1, 3);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 4 + 8);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 4 + 12);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale0);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale0);
                _p4 = vmulq_f32(_p4, _scale1);
                _p5 = vmulq_f32(_p5, _scale1);
                _p6 = vmulq_f32(_p6, _scale1);
                _p7 = vmulq_f32(_p7, _scale1);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p4), float2int8(_p2, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p5), float2int8(_p3, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p0n = vld1q_f32(p0 + 4);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p1n = vld1q_f32(p0 + A_hstep * 4 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p0n = vmulq_f32(_p0n, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);
                _p1n = vmulq_f32(_p1n, _scale1);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p0n, _p1n);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 3 + 4);
                float32x4_t _p8 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + A_hstep * 5);
                float32x4_t _pb = vld1q_f32(p0 + A_hstep * 5 + 4);
                float32x4_t _pc = vld1q_f32(p0 + A_hstep * 6);
                float32x4_t _pd = vld1q_f32(p0 + A_hstep * 6 + 4);
                float32x4_t _pe = vld1q_f32(p0 + A_hstep * 7);
                float32x4_t _pf = vld1q_f32(p0 + A_hstep * 7 + 4);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale0, 0);
                _p2 = vmulq_laneq_f32(_p2, _scale0, 1);
                _p3 = vmulq_laneq_f32(_p3, _scale0, 1);
                _p4 = vmulq_laneq_f32(_p4, _scale0, 2);
                _p5 = vmulq_laneq_f32(_p5, _scale0, 2);
                _p6 = vmulq_laneq_f32(_p6, _scale0, 3);
                _p7 = vmulq_laneq_f32(_p7, _scale0, 3);
                _p8 = vmulq_laneq_f32(_p8, _scale1, 0);
                _p9 = vmulq_laneq_f32(_p9, _scale1, 0);
                _pa = vmulq_laneq_f32(_pa, _scale1, 1);
                _pb = vmulq_laneq_f32(_pb, _scale1, 1);
                _pc = vmulq_laneq_f32(_pc, _scale1, 2);
                _pd = vmulq_laneq_f32(_pd, _scale1, 2);
                _pe = vmulq_laneq_f32(_pe, _scale1, 3);
                _pf = vmulq_laneq_f32(_pf, _scale1, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale0), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale0), 0);
                _p2 = vmulq_lane_f32(_p2, vget_low_f32(_scale0), 1);
                _p3 = vmulq_lane_f32(_p3, vget_low_f32(_scale0), 1);
                _p4 = vmulq_lane_f32(_p4, vget_high_f32(_scale0), 0);
                _p5 = vmulq_lane_f32(_p5, vget_high_f32(_scale0), 0);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale0), 1);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale0), 1);
                _p8 = vmulq_lane_f32(_p8, vget_low_f32(_scale1), 0);
                _p9 = vmulq_lane_f32(_p9, vget_low_f32(_scale1), 0);
                _pa = vmulq_lane_f32(_pa, vget_low_f32(_scale1), 1);
                _pb = vmulq_lane_f32(_pb, vget_low_f32(_scale1), 1);
                _pc = vmulq_lane_f32(_pc, vget_high_f32(_scale1), 0);
                _pd = vmulq_lane_f32(_pd, vget_high_f32(_scale1), 0);
                _pe = vmulq_lane_f32(_pe, vget_high_f32(_scale1), 1);
                _pf = vmulq_lane_f32(_pf, vget_high_f32(_scale1), 1);
#endif

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p4, _p6);
                int8x8_t _r2 = float2int8(_p8, _pa);
                int8x8_t _r3 = float2int8(_pc, _pe);
                int8x8_t _r4 = float2int8(_p1, _p3);
                int8x8_t _r5 = float2int8(_p5, _p7);
                int8x8_t _r6 = float2int8(_p9, _pb);
                int8x8_t _r7 = float2int8(_pd, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p8, _pa));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_pc, _pe));
                int16x4_t _t4 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4_t _t5 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                int16x4_t _t6 = vreinterpret_s16_s8(float2int8(_p9, _pb));
                int16x4_t _t7 = vreinterpret_s16_s8(float2int8(_pd, _pf));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int16x4x2_t _t45 = vuzp_s16(_t4, _t5);
                int16x4x2_t _t67 = vuzp_s16(_t6, _t7);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
                int8x8_t _r4 = vreinterpret_s8_s16(_t45.val[0]);
                int8x8_t _r5 = vreinterpret_s8_s16(_t67.val[0]);
                int8x8_t _r6 = vreinterpret_s8_s16(_t45.val[1]);
                int8x8_t _r7 = vreinterpret_s8_s16(_t67.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));

                pp += 64;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 5);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 6);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 7);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale0, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale0, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale0, 3);
                _p4 = vmulq_laneq_f32(_p4, _scale1, 0);
                _p5 = vmulq_laneq_f32(_p5, _scale1, 1);
                _p6 = vmulq_laneq_f32(_p6, _scale1, 2);
                _p7 = vmulq_laneq_f32(_p7, _scale1, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale0), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale0), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale0), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale0), 1);
                _p4 = vmulq_lane_f32(_p4, vget_low_f32(_scale1), 0);
                _p5 = vmulq_lane_f32(_p5, vget_low_f32(_scale1), 1);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale1), 0);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale1), 1);
#endif

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p4, _p5));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p6, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
                float32x2_t _p4 = vld1_f32(p0 + A_hstep * 4);
                float32x2_t _p5 = vld1_f32(p0 + A_hstep * 5);
                float32x2_t _p6 = vld1_f32(p0 + A_hstep * 6);
                float32x2_t _p7 = vld1_f32(p0 + A_hstep * 7);

                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);
                float32x4_t _p45 = vcombine_f32(_p4, _p5);
                float32x4_t _p67 = vcombine_f32(_p6, _p7);

                float32x4x2_t _scale01 = vzipq_f32(_scale0, _scale0);
                float32x4x2_t _scale23 = vzipq_f32(_scale1, _scale1);

                _p01 = vmulq_f32(_p01, _scale01.val[0]);
                _p23 = vmulq_f32(_p23, _scale01.val[1]);
                _p45 = vmulq_f32(_p45, _scale23.val[0]);
                _p67 = vmulq_f32(_p67, _scale23.val[1]);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0++;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k * elempack;

        float32x4_t _scale = vld1q_f32((const float*)scales + i + ii);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + 16);

                float32x4_t _p0 = vmulq_laneq_f32(_p.val[0], _scale, 0);
                float32x4_t _p1 = vmulq_laneq_f32(_p.val[1], _scale, 1);
                float32x4_t _p2 = vmulq_laneq_f32(_p.val[2], _scale, 2);
                float32x4_t _p3 = vmulq_laneq_f32(_p.val[3], _scale, 3);
                float32x4_t _p4 = vmulq_laneq_f32(_q.val[0], _scale, 0);
                float32x4_t _p5 = vmulq_laneq_f32(_q.val[1], _scale, 1);
                float32x4_t _p6 = vmulq_laneq_f32(_q.val[2], _scale, 2);
                float32x4_t _p7 = vmulq_laneq_f32(_q.val[3], _scale, 3);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += 32;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);

                float32x4_t _p0 = vmulq_laneq_f32(_p.val[0], _scale, 0);
                float32x4_t _p1 = vmulq_laneq_f32(_p.val[1], _scale, 1);
                float32x4_t _p2 = vmulq_laneq_f32(_p.val[2], _scale, 2);
                float32x4_t _p3 = vmulq_laneq_f32(_p.val[3], _scale, 3);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p2);
                _r01.val[1] = float2int8(_p1, _p3);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                _p0 = vmulq_f32(_p0, _scale);
                int8x8_t _r0 = float2int8(_p0, _p0);

                pp[0] = vget_lane_s8(_r0, 0);
                pp[1] = vget_lane_s8(_r0, 1);
                pp[2] = vget_lane_s8(_r0, 2);
                pp[3] = vget_lane_s8(_r0, 3);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 3 + 4);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale, 0);
                _p2 = vmulq_laneq_f32(_p2, _scale, 1);
                _p3 = vmulq_laneq_f32(_p3, _scale, 1);
                _p4 = vmulq_laneq_f32(_p4, _scale, 2);
                _p5 = vmulq_laneq_f32(_p5, _scale, 2);
                _p6 = vmulq_laneq_f32(_p6, _scale, 3);
                _p7 = vmulq_laneq_f32(_p7, _scale, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 0);
                _p2 = vmulq_lane_f32(_p2, vget_low_f32(_scale), 1);
                _p3 = vmulq_lane_f32(_p3, vget_low_f32(_scale), 1);
                _p4 = vmulq_lane_f32(_p4, vget_high_f32(_scale), 0);
                _p5 = vmulq_lane_f32(_p5, vget_high_f32(_scale), 0);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale), 1);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale), 1);
#endif

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p4, _p6);
                int8x8_t _r2 = float2int8(_p1, _p3);
                int8x8_t _r3 = float2int8(_p5, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale), 1);
#endif

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);

                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);

                float32x4x2_t _scale01 = vzipq_f32(_scale, _scale);

                _p01 = vmulq_f32(_p01, _scale01.val[0]);
                _p23 = vmulq_f32(_p23, _scale01.val[1]);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);

                _p0 = vmulq_f32(_p0, _scale);
                int8x8_t _r0 = float2int8(_p0, _p0);

                pp[0] = vget_lane_s8(_r0, 0);
                pp[1] = vget_lane_s8(_r0, 1);
                pp[2] = vget_lane_s8(_r0, 2);
                pp[3] = vget_lane_s8(_r0, 3);

                pp += 4;
                p0++;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        // if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            float32x4_t _scale0 = vdupq_n_f32(scale0);
            float32x4_t _scale1 = vdupq_n_f32(scale1);
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale0);
                _p2 = vmulq_f32(_p2, _scale1);
                _p3 = vmulq_f32(_p3, _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p1, _p3);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p2));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p2));
                float32x4_t _t2 = vcombine_f32(vget_low_f32(_p1), vget_low_f32(_p3));
                float32x4_t _t3 = vcombine_f32(vget_high_f32(_p1), vget_high_f32(_p3));
                int8x8_t _r0 = float2int8(_t0, _t1);
                int8x8_t _r1 = float2int8(_t2, _t3);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r0);
                vst1_s8(pp + 8, _r1);

                pp += 16;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                int8x8_t _r0 = float2int8(_t0, _t1);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[1] * scale0);
                pp[2] = float2int8(p0[A_hstep] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp += 4;
                p0 += 2;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            float32x4_t _scale = vdupq_n_f32(scale);
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 8;
            }
#endif // __ARM_NEON
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

#if __ARM_NEON
#if __aarch64__
    float32x4_t _v127 = vdupq_n_f32(127.f);
    float32x4_t _v127_B_scale = vdupq_n_f32(v127_B_scale);
#endif
#endif

    float* ps = (float*)scales + i;
    float* pods = (float*)out_descales + i;

#if __ARM_NEON
    if (elempack == 4)
    {
        int ii = 0;
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii) * 4;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            for (int kk = 0; kk < K; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                p0 += A_hstep * 4;
            }
            float32x2_t _aa0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            float32x2_t _aa1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
            float32x2_t _aa2 = vmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
            float32x2_t _aa3 = vmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
            float32x2_t _aa01 = vpmax_f32(_aa0, _aa1);
            float32x2_t _aa23 = vpmax_f32(_aa2, _aa3);
            float32x4_t _absmax = vcombine_f32(_aa01, _aa23);

#if __aarch64__
            float32x4_t _scale = vdivq_f32(_v127, _absmax);
            float32x4_t _out_descale = vdivq_f32(_absmax, _v127_B_scale);

            vst1q_f32(ps, _scale);
            vst1q_f32(pods, _out_descale);
#else
            float tmp[4];
            vst1q_f32(tmp, _absmax);

            ps[0] = 127.f / tmp[0];
            ps[1] = 127.f / tmp[1];
            ps[2] = 127.f / tmp[2];
            ps[3] = 127.f / tmp[3];

            pods[0] = tmp[0] / v127_B_scale;
            pods[1] = tmp[1] / v127_B_scale;
            pods[2] = tmp[2] / v127_B_scale;
            pods[3] = tmp[3] / v127_B_scale;

            // float32x4_t _recp_absmax = vrecpeq_f32(_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax, _recp_absmax), _recp_absmax);
            // float32x4_t _scale = vmulq_f32(_v127, _recp_absmax);
            // float32x4_t _out_descale = vmulq_f32(_absmax, _recp_v127_B_scale);
#endif

            ps += 4;
            pods += 4;
        }
        for (; ii < max_ii; ii++)
        {
            const float* p0 = (const float*)A + (i + ii) * 4;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 8);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 12);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                p0 += A_hstep * 16;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax2);
            _absmax1 = vmaxq_f32(_absmax1, _absmax3);
            for (; kk + 1 < K; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += A_hstep * 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = vld1q_f32(p0);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                p0 += A_hstep * 4;
            }
            float32x2_t _aa = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            float absmax = std::max(vget_lane_f32(_aa, 0), vget_lane_f32(_aa, 1));

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
        int ii = 0;
#if __ARM_NEON
        for (; ii + 3 < max_ii; ii += 4)
        {
            const float* p0 = (const float*)A + (i + ii);

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                p0 += A_hstep * 4;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax2);
            _absmax1 = vmaxq_f32(_absmax1, _absmax3);
            for (; kk + 1 < K; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += A_hstep * 2;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = vld1q_f32(p0);
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                p0 += A_hstep;
            }

#if __aarch64__
            float32x4_t _scale = vdivq_f32(_v127, _absmax0);
            float32x4_t _out_descale = vdivq_f32(_absmax0, _v127_B_scale);

            vst1q_f32(ps, _scale);
            vst1q_f32(pods, _out_descale);
#else
            float tmp[4];
            vst1q_f32(tmp, _absmax0);

            ps[0] = 127.f / tmp[0];
            ps[1] = 127.f / tmp[1];
            ps[2] = 127.f / tmp[2];
            ps[3] = 127.f / tmp[3];

            pods[0] = tmp[0] / v127_B_scale;
            pods[1] = tmp[1] / v127_B_scale;
            pods[2] = tmp[2] / v127_B_scale;
            pods[3] = tmp[3] / v127_B_scale;

            // float32x4_t _recp_absmax = vrecpeq_f32(_absmax0);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmulq_f32(vrecpsq_f32(_absmax0, _recp_absmax), _recp_absmax);
            // float32x4_t _scale = vmulq_f32(_v127, _recp_absmax);
            // float32x4_t _out_descale = vmulq_f32(_absmax0, _recp_v127_B_scale);
#endif

            ps += 4;
            pods += 4;
        }
        for (; ii + 1 < max_ii; ii += 2)
        {
            const float* p0 = (const float*)A + (i + ii);

            float32x2_t _absmax0 = vdup_n_f32(0.f);
            float32x2_t _absmax1 = vdup_n_f32(0.f);
            float32x2_t _absmax2 = vdup_n_f32(0.f);
            float32x2_t _absmax3 = vdup_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
                _absmax0 = vmax_f32(_absmax0, vabs_f32(_p0));
                _absmax1 = vmax_f32(_absmax1, vabs_f32(_p1));
                _absmax2 = vmax_f32(_absmax2, vabs_f32(_p2));
                _absmax3 = vmax_f32(_absmax3, vabs_f32(_p3));
                p0 += A_hstep * 4;
            }
            _absmax0 = vmax_f32(_absmax0, _absmax2);
            _absmax1 = vmax_f32(_absmax1, _absmax3);
            for (; kk + 1 < K; kk += 2)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                _absmax0 = vmax_f32(_absmax0, vabs_f32(_p0));
                _absmax1 = vmax_f32(_absmax1, vabs_f32(_p1));
                p0 += A_hstep * 2;
            }
            _absmax0 = vmax_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x2_t _p = vld1_f32(p0);
                _absmax0 = vmax_f32(_absmax0, vabs_f32(_p));
                p0 += A_hstep;
            }

#if __aarch64__
            float32x2_t _scale = vdiv_f32(vget_low_f32(_v127), _absmax0);
            float32x2_t _out_descale = vdiv_f32(_absmax0, vget_low_f32(_v127_B_scale));

            vst1_f32(ps, _scale);
            vst1_f32(pods, _out_descale);
#else
            float tmp[2];
            vst1_f32(tmp, _absmax0);

            ps[0] = 127.f / tmp[0];
            ps[1] = 127.f / tmp[1];

            pods[0] = tmp[0] / v127_B_scale;
            pods[1] = tmp[1] / v127_B_scale;

            // float32x2_t _recp_absmax = vrecpe_f32(_absmax0);
            // _recp_absmax = vmul_f32(vrecps_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmul_f32(vrecps_f32(_absmax0, _recp_absmax), _recp_absmax);
            // _recp_absmax = vmul_f32(vrecps_f32(_absmax0, _recp_absmax), _recp_absmax);
            // float32x2_t _scale = vmul_f32(vget_low_f32(_v127), _recp_absmax);
            // float32x2_t _out_descale = vmul_f32(_absmax0, vget_low_f32(_recp_v127_B_scale));
#endif

            ps += 2;
            pods += 2;
        }
#endif // __ARM_NEON
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
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_A_tile_fp32_to_int8_i8mm(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_A_tile_fp32_to_int8_asimddp(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_fp32_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        float32x4_t _scale0 = vld1q_f32((const float*)scales + i + ii);
        float32x4_t _scale1 = vld1q_f32((const float*)scales + i + ii + 4);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);
                float32x4_t _p8 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + A_hstep * 4 + 8);
                float32x4_t _pb = vld1q_f32(p0 + A_hstep * 4 + 12);
                float32x4_t _pc = vld1q_f32(p0 + A_hstep * 4 + 16);
                float32x4_t _pd = vld1q_f32(p0 + A_hstep * 4 + 20);
                float32x4_t _pe = vld1q_f32(p0 + A_hstep * 4 + 24);
                float32x4_t _pf = vld1q_f32(p0 + A_hstep * 4 + 28);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale0, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale0, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale0, 3);
                _p4 = vmulq_laneq_f32(_p4, _scale1, 0);
                _p5 = vmulq_laneq_f32(_p5, _scale1, 1);
                _p6 = vmulq_laneq_f32(_p6, _scale1, 2);
                _p7 = vmulq_laneq_f32(_p7, _scale1, 3);
                _p8 = vmulq_laneq_f32(_p8, _scale0, 0);
                _p9 = vmulq_laneq_f32(_p9, _scale0, 1);
                _pa = vmulq_laneq_f32(_pa, _scale0, 2);
                _pb = vmulq_laneq_f32(_pb, _scale0, 3);
                _pc = vmulq_laneq_f32(_pc, _scale1, 0);
                _pd = vmulq_laneq_f32(_pd, _scale1, 1);
                _pe = vmulq_laneq_f32(_pe, _scale1, 2);
                _pf = vmulq_laneq_f32(_pf, _scale1, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale0), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale0), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale0), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale0), 1);
                _p4 = vmulq_lane_f32(_p4, vget_low_f32(_scale1), 0);
                _p5 = vmulq_lane_f32(_p5, vget_low_f32(_scale1), 1);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale1), 0);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale1), 1);
                _p8 = vmulq_lane_f32(_p8, vget_low_f32(_scale0), 0);
                _p9 = vmulq_lane_f32(_p9, vget_low_f32(_scale0), 1);
                _pa = vmulq_lane_f32(_pa, vget_high_f32(_scale0), 0);
                _pb = vmulq_lane_f32(_pb, vget_high_f32(_scale0), 1);
                _pc = vmulq_lane_f32(_pc, vget_low_f32(_scale1), 0);
                _pd = vmulq_lane_f32(_pd, vget_low_f32(_scale1), 1);
                _pe = vmulq_lane_f32(_pe, vget_high_f32(_scale1), 0);
                _pf = vmulq_lane_f32(_pf, vget_high_f32(_scale1), 1);
#endif

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p8);
                int8x8_t _r1 = float2int8(_p1, _p9);
                int8x8_t _r2 = float2int8(_p2, _pa);
                int8x8_t _r3 = float2int8(_p3, _pb);
                int8x8_t _r4 = float2int8(_p4, _pc);
                int8x8_t _r5 = float2int8(_p5, _pd);
                int8x8_t _r6 = float2int8(_p6, _pe);
                int8x8_t _r7 = float2int8(_p7, _pf);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

                int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                int16x8_t _r45 = vreinterpretq_s16_s8(vcombine_s8(_r4, _r5));
                int16x8_t _r67 = vreinterpretq_s16_s8(vcombine_s8(_r6, _r7));
                int16x8x2_t _rr0 = vuzpq_s16(_r01, _r23);
                int16x8x2_t _rr1 = vuzpq_s16(_r45, _r67);

                vst1q_s8(pp, vreinterpretq_s8_s16(_rr0.val[0]));
                vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr0.val[1]));
                vst1q_s8(pp + 32, vreinterpretq_s8_s16(_rr1.val[0]));
                vst1q_s8(pp + 48, vreinterpretq_s8_s16(_rr1.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale0, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale0, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale0, 3);
                _p4 = vmulq_laneq_f32(_p4, _scale1, 0);
                _p5 = vmulq_laneq_f32(_p5, _scale1, 1);
                _p6 = vmulq_laneq_f32(_p6, _scale1, 2);
                _p7 = vmulq_laneq_f32(_p7, _scale1, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale0), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale0), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale0), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale0), 1);
                _p4 = vmulq_lane_f32(_p4, vget_low_f32(_scale1), 0);
                _p5 = vmulq_lane_f32(_p5, vget_low_f32(_scale1), 1);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale1), 0);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale1), 1);
#endif

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

#if __ARM_FEATURE_DOTPROD
                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                int16x8x2_t _rr = vuzpq_s16(_r01, _r23);

                vst1q_s8(pp, vreinterpretq_s8_s16(_rr.val[0]));
                vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 3 + 4);
                float32x4_t _p8 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + A_hstep * 5);
                float32x4_t _pb = vld1q_f32(p0 + A_hstep * 5 + 4);
                float32x4_t _pc = vld1q_f32(p0 + A_hstep * 6);
                float32x4_t _pd = vld1q_f32(p0 + A_hstep * 6 + 4);
                float32x4_t _pe = vld1q_f32(p0 + A_hstep * 7);
                float32x4_t _pf = vld1q_f32(p0 + A_hstep * 7 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale1);
                _p4 = vmulq_f32(_p4, _scale0);
                _p5 = vmulq_f32(_p5, _scale1);
                _p6 = vmulq_f32(_p6, _scale0);
                _p7 = vmulq_f32(_p7, _scale1);
                _p8 = vmulq_f32(_p8, _scale0);
                _p9 = vmulq_f32(_p9, _scale1);
                _pa = vmulq_f32(_pa, _scale0);
                _pb = vmulq_f32(_pb, _scale1);
                _pc = vmulq_f32(_pc, _scale0);
                _pd = vmulq_f32(_pd, _scale1);
                _pe = vmulq_f32(_pe, _scale0);
                _pf = vmulq_f32(_pf, _scale1);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                int8x16x4_t _r0123;
                _r0123.val[0] = vcombine_s8(_r04.val[0], _r04.val[1]);
                _r0123.val[1] = vcombine_s8(_r15.val[0], _r15.val[1]);
                _r0123.val[2] = vcombine_s8(_r26.val[0], _r26.val[1]);
                _r0123.val[3] = vcombine_s8(_r37.val[0], _r37.val[1]);

                vst4q_s8(pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x4_t _r0123;
                _r0123.val[0] = _r0;
                _r0123.val[1] = _r1;
                _r0123.val[2] = _r2;
                _r0123.val[3] = _r3;
                int8x8x4_t _r4567;
                _r4567.val[0] = _r4;
                _r4567.val[1] = _r5;
                _r4567.val[2] = _r6;
                _r4567.val[3] = _r7;

                vst4_s8(pp, _r0123);
                vst4_s8(pp + 32, _r4567);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(_r0, _r2);
                _r01.val[1] = vcombine_s8(_r1, _r3);
                int8x16x2_t _r23;
                _r23.val[0] = vcombine_s8(_r4, _r6);
                _r23.val[1] = vcombine_s8(_r5, _r7);

                vst2q_s8(pp, _r01);
                vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 3 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale1);
                _p4 = vmulq_f32(_p4, _scale0);
                _p5 = vmulq_f32(_p5, _scale1);
                _p6 = vmulq_f32(_p6, _scale0);
                _p7 = vmulq_f32(_p7, _scale1);

#if __ARM_FEATURE_DOTPROD
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p0, _p1);
                _r0123.val[1] = float2int8(_p2, _p3);
                _r0123.val[2] = float2int8(_p4, _p5);
                _r0123.val[3] = float2int8(_p6, _p7);

                vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale1);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        float32x4_t _scale = vld1q_f32((const float*)scales + i + ii);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 4 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 4 + 8);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 4 + 12);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale, 3);
                _p4 = vmulq_laneq_f32(_p4, _scale, 0);
                _p5 = vmulq_laneq_f32(_p5, _scale, 1);
                _p6 = vmulq_laneq_f32(_p6, _scale, 2);
                _p7 = vmulq_laneq_f32(_p7, _scale, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale), 1);
                _p4 = vmulq_lane_f32(_p4, vget_low_f32(_scale), 0);
                _p5 = vmulq_lane_f32(_p5, vget_low_f32(_scale), 1);
                _p6 = vmulq_lane_f32(_p6, vget_high_f32(_scale), 0);
                _p7 = vmulq_lane_f32(_p7, vget_high_f32(_scale), 1);
#endif

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p4, _p5));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p6, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

#if __aarch64__
                _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                _p3 = vmulq_laneq_f32(_p3, _scale, 3);
#else
                _p0 = vmulq_lane_f32(_p0, vget_low_f32(_scale), 0);
                _p1 = vmulq_lane_f32(_p1, vget_low_f32(_scale), 1);
                _p2 = vmulq_lane_f32(_p2, vget_high_f32(_scale), 0);
                _p3 = vmulq_lane_f32(_p3, vget_high_f32(_scale), 1);
#endif

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);
                float32x4_t _p4 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + A_hstep * 5);
                float32x4_t _p6 = vld1q_f32(p0 + A_hstep * 6);
                float32x4_t _p7 = vld1q_f32(p0 + A_hstep * 7);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);

                vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p0, _p4);
                _r0123.val[1] = float2int8(_p1, _p5);
                _r0123.val[2] = float2int8(_p2, _p6);
                _r0123.val[3] = float2int8(_p3, _p7);

                vst4_s8(pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
                transpose4x4_ps(_p0, _p1, _p2, _p3);

                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else  // __ARM_FEATURE_DOTPROD
                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p2);
                _r01.val[1] = float2int8(_p1, _p3);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                _p0 = vmulq_f32(_p0, _scale);
                int8x8_t _r0 = float2int8(_p0, _p0);

                pp[0] = vget_lane_s8(_r0, 0);
                pp[1] = vget_lane_s8(_r0, 1);
                pp[2] = vget_lane_s8(_r0, 2);
                pp[3] = vget_lane_s8(_r0, 3);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + k * A_hstep + (i + ii) * elempack;

        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

#if __ARM_NEON
        float32x4_t _scale0 = vdupq_n_f32(scale0);
        float32x4_t _scale1 = vdupq_n_f32(scale1);
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 4 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);
                _p2 = vmulq_f32(_p2, _scale0);
                _p3 = vmulq_f32(_p3, _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p1, _p3);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale0);
                _p1 = vmulq_f32(_p1, _scale1);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r01 = float2int8(_p0, _p1);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                int8x8_t _r01 = float2int8(_t0, _t1);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
#if __ARM_NEON
            float32x4_t _scale = vzipq_f32(_scale0, _scale1).val[0];
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);
                float32x2_t _p4 = vld1_f32(p0 + A_hstep * 4);
                float32x2_t _p5 = vld1_f32(p0 + A_hstep * 5);
                float32x2_t _p6 = vld1_f32(p0 + A_hstep * 6);
                float32x2_t _p7 = vld1_f32(p0 + A_hstep * 7);

#if __ARM_FEATURE_DOTPROD
                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);
                float32x4_t _p45 = vcombine_f32(_p4, _p5);
                float32x4_t _p67 = vcombine_f32(_p6, _p7);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r01 = vuzp_s8(_r0, _r1);

                vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r01 = vtrn_s8(_r0, _r1);
                int8x8x2_t _rr01 = vuzp_s8(_r01.val[0], _r01.val[1]);

                vst1q_s8(pp, vcombine_s8(_rr01.val[0], _rr01.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p02 = vcombine_f32(_p0, _p2);
                float32x4_t _p46 = vcombine_f32(_p4, _p6);
                float32x4_t _p13 = vcombine_f32(_p1, _p3);
                float32x4_t _p57 = vcombine_f32(_p5, _p7);

                _p02 = vmulq_f32(_p02, _scale);
                _p46 = vmulq_f32(_p46, _scale);
                _p13 = vmulq_f32(_p13, _scale);
                _p57 = vmulq_f32(_p57, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p02, _p46);
                _r01.val[1] = float2int8(_p13, _p57);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + A_hstep);
                float32x2_t _p2 = vld1_f32(p0 + A_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + A_hstep * 3);

#if __ARM_FEATURE_DOTPROD
                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                float32x4x2_t _pp = vuzpq_f32(_p01, _p23);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p02 = vcombine_f32(_p0, _p2);
                float32x4_t _p13 = vcombine_f32(_p1, _p3);

                _p02 = vmulq_f32(_p02, _scale);
                _p13 = vmulq_f32(_p13, _scale);

                float32x4x2_t _pp = vzipq_f32(_p02, _p13);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale0);
                pp[1] = float2int8(p0[A_hstep + 0] * scale0);
                pp[2] = float2int8(p0[1] * scale1);
                pp[3] = float2int8(p0[A_hstep + 1] * scale1);
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __ARM_NEON
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

#if __ARM_NEON
        float32x4_t _scale = vdupq_n_f32(scale);
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);
                float32x4_t _p2 = vld1q_f32(p0 + A_hstep * 8);
                float32x4_t _p3 = vld1q_f32(p0 + A_hstep * 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));

                pp += 16;
                p0 += A_hstep * 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + A_hstep * 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                float32x4_t _p2 = float32x4_t();
                float32x4_t _p3 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);
                _p2 = vsetq_lane_f32(p0[A_hstep * 8], _p2, 0);
                _p2 = vsetq_lane_f32(p0[A_hstep * 9], _p2, 1);
                _p2 = vsetq_lane_f32(p0[A_hstep * 10], _p2, 2);
                _p2 = vsetq_lane_f32(p0[A_hstep * 11], _p2, 3);
                _p3 = vsetq_lane_f32(p0[A_hstep * 12], _p3, 0);
                _p3 = vsetq_lane_f32(p0[A_hstep * 13], _p3, 1);
                _p3 = vsetq_lane_f32(p0[A_hstep * 14], _p3, 2);
                _p3 = vsetq_lane_f32(p0[A_hstep * 15], _p3, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));

                pp += 16;
                p0 += A_hstep * 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[A_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[A_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[A_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[A_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[A_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[A_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[A_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 8;
            }
#endif // __ARM_NEON
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
    float absmax = 0.f;
#if __ARM_NEON
    float32x4_t _absmax = vdupq_n_f32(0.f);
#endif
    for (int i = 0; i < (B.dims == 3 ? B.c : B.h); i++)
    {
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;
        const float* ptr = (const float*)B + i * B_hstep * B.elempack;

        const int size = B.w * B.elempack;

        int j = 0;
#if __ARM_NEON
        for (; j + 3 < size; j += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
            ptr += 4;
        }
#endif
        for (; j < size; j++)
        {
            absmax = std::max(absmax, (float)fabsf(ptr[0]));
            ptr++;
        }
    }
#if __ARM_NEON
    float32x2_t _aa = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
    absmax = std::max(absmax, std::max(vget_lane_f32(_aa, 0), vget_lane_f32(_aa, 1)));
#endif

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_B_tile_fp32_to_int8_i8mm(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_B_tile_fp32_to_int8_asimddp(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_fp32_to_int8 %d %d %d", max_jj, max_kk, elempack);

    signed char* pp = BT;

#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
#endif

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + 16);
                float32x4x4_t _r = vld4q_f32(p0 + B_hstep * 4);
                float32x4x4_t _s = vld4q_f32(p0 + B_hstep * 4 + 16);

                float32x4_t _p0 = vmulq_f32(_p.val[0], _scale);
                float32x4_t _p1 = vmulq_f32(_p.val[1], _scale);
                float32x4_t _p2 = vmulq_f32(_p.val[2], _scale);
                float32x4_t _p3 = vmulq_f32(_p.val[3], _scale);
                float32x4_t _p4 = vmulq_f32(_q.val[0], _scale);
                float32x4_t _p5 = vmulq_f32(_q.val[1], _scale);
                float32x4_t _p6 = vmulq_f32(_q.val[2], _scale);
                float32x4_t _p7 = vmulq_f32(_q.val[3], _scale);
                float32x4_t _p8 = vmulq_f32(_r.val[0], _scale);
                float32x4_t _p9 = vmulq_f32(_r.val[1], _scale);
                float32x4_t _pa = vmulq_f32(_r.val[2], _scale);
                float32x4_t _pb = vmulq_f32(_r.val[3], _scale);
                float32x4_t _pc = vmulq_f32(_s.val[0], _scale);
                float32x4_t _pd = vmulq_f32(_s.val[1], _scale);
                float32x4_t _pe = vmulq_f32(_s.val[2], _scale);
                float32x4_t _pf = vmulq_f32(_s.val[3], _scale);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
                int8x8_t _r4 = float2int8(_p8, _pc);
                int8x8_t _r5 = float2int8(_p9, _pd);
                int8x8_t _r6 = float2int8(_pa, _pe);
                int8x8_t _r7 = float2int8(_pb, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p8, _p9);
                int8x8_t _r3 = float2int8(_pa, _pb);
                int8x8_t _r4 = float2int8(_p4, _p5);
                int8x8_t _r5 = float2int8(_p6, _p7);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);
                float32x4_t _p8 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + B_hstep * 4 + 8);
                float32x4_t _pb = vld1q_f32(p0 + B_hstep * 4 + 12);
                float32x4_t _pc = vld1q_f32(p0 + B_hstep * 4 + 16);
                float32x4_t _pd = vld1q_f32(p0 + B_hstep * 4 + 20);
                float32x4_t _pe = vld1q_f32(p0 + B_hstep * 4 + 24);
                float32x4_t _pf = vld1q_f32(p0 + B_hstep * 4 + 28);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);
                _p8 = vmulq_f32(_p8, _scale);
                _p9 = vmulq_f32(_p9, _scale);
                _pa = vmulq_f32(_pa, _scale);
                _pb = vmulq_f32(_pb, _scale);
                _pc = vmulq_f32(_pc, _scale);
                _pd = vmulq_f32(_pd, _scale);
                _pe = vmulq_f32(_pe, _scale);
                _pf = vmulq_f32(_pf, _scale);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p8), float2int8(_p2, _pa));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p9), float2int8(_p3, _pb));
                int8x16x2_t _r23;
                _r23.val[0] = vcombine_s8(float2int8(_p4, _pc), float2int8(_p6, _pe));
                _r23.val[1] = vcombine_s8(float2int8(_p5, _pd), float2int8(_p7, _pf));

                vst2q_s8(pp, _r01);
                vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += 32;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + B_hstep * 4);

                float32x4_t _p0 = vmulq_f32(_p.val[0], _scale);
                float32x4_t _p1 = vmulq_f32(_p.val[1], _scale);
                float32x4_t _p2 = vmulq_f32(_p.val[2], _scale);
                float32x4_t _p3 = vmulq_f32(_p.val[3], _scale);
                float32x4_t _p4 = vmulq_f32(_q.val[0], _scale);
                float32x4_t _p5 = vmulq_f32(_q.val[1], _scale);
                float32x4_t _p6 = vmulq_f32(_q.val[2], _scale);
                float32x4_t _p7 = vmulq_f32(_q.val[3], _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 4 + 8);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 4 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p4), float2int8(_p2, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p5), float2int8(_p3, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 4 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p2);
                _r01.val[1] = float2int8(_p1, _p3);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep * 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 3 + 4);
                float32x4_t _p8 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + B_hstep * 5);
                float32x4_t _pb = vld1q_f32(p0 + B_hstep * 5 + 4);
                float32x4_t _pc = vld1q_f32(p0 + B_hstep * 6);
                float32x4_t _pd = vld1q_f32(p0 + B_hstep * 6 + 4);
                float32x4_t _pe = vld1q_f32(p0 + B_hstep * 7);
                float32x4_t _pf = vld1q_f32(p0 + B_hstep * 7 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);
                _p8 = vmulq_f32(_p8, _scale);
                _p9 = vmulq_f32(_p9, _scale);
                _pa = vmulq_f32(_pa, _scale);
                _pb = vmulq_f32(_pb, _scale);
                _pc = vmulq_f32(_pc, _scale);
                _pd = vmulq_f32(_pd, _scale);
                _pe = vmulq_f32(_pe, _scale);
                _pf = vmulq_f32(_pf, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p4, _p6);
                int8x8_t _r2 = float2int8(_p8, _pa);
                int8x8_t _r3 = float2int8(_pc, _pe);
                int8x8_t _r4 = float2int8(_p1, _p3);
                int8x8_t _r5 = float2int8(_p5, _p7);
                int8x8_t _r6 = float2int8(_p9, _pb);
                int8x8_t _r7 = float2int8(_pd, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p8, _pa));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_pc, _pe));
                int16x4_t _t4 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4_t _t5 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                int16x4_t _t6 = vreinterpret_s16_s8(float2int8(_p9, _pb));
                int16x4_t _t7 = vreinterpret_s16_s8(float2int8(_pd, _pf));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int16x4x2_t _t45 = vuzp_s16(_t4, _t5);
                int16x4x2_t _t67 = vuzp_s16(_t6, _t7);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
                int8x8_t _r4 = vreinterpret_s8_s16(_t45.val[0]);
                int8x8_t _r5 = vreinterpret_s8_s16(_t67.val[0]);
                int8x8_t _r6 = vreinterpret_s8_s16(_t45.val[1]);
                int8x8_t _r7 = vreinterpret_s8_s16(_t67.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));

                pp += 64;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 5);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 6);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 7);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p4, _p5));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p6, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + B_hstep);
                float32x2_t _p2 = vld1_f32(p0 + B_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + B_hstep * 3);
                float32x2_t _p4 = vld1_f32(p0 + B_hstep * 4);
                float32x2_t _p5 = vld1_f32(p0 + B_hstep * 5);
                float32x2_t _p6 = vld1_f32(p0 + B_hstep * 6);
                float32x2_t _p7 = vld1_f32(p0 + B_hstep * 7);

                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);
                float32x4_t _p45 = vcombine_f32(_p4, _p5);
                float32x4_t _p67 = vcombine_f32(_p6, _p7);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[B_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[B_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[B_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[B_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[B_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[B_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[B_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                pp += 8;
                p0++;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);
                float32x4x4_t _q = vld4q_f32(p0 + 16);

                float32x4_t _p0 = vmulq_f32(_p.val[0], _scale);
                float32x4_t _p1 = vmulq_f32(_p.val[1], _scale);
                float32x4_t _p2 = vmulq_f32(_p.val[2], _scale);
                float32x4_t _p3 = vmulq_f32(_p.val[3], _scale);
                float32x4_t _p4 = vmulq_f32(_q.val[0], _scale);
                float32x4_t _p5 = vmulq_f32(_q.val[1], _scale);
                float32x4_t _p6 = vmulq_f32(_q.val[2], _scale);
                float32x4_t _p7 = vmulq_f32(_q.val[3], _scale);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += 32;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
#if __ARM_FEATURE_DOTPROD
                float32x4x4_t _p = vld4q_f32(p0);

                float32x4_t _p0 = vmulq_f32(_p.val[0], _scale);
                float32x4_t _p1 = vmulq_f32(_p.val[1], _scale);
                float32x4_t _p2 = vmulq_f32(_p.val[2], _scale);
                float32x4_t _p3 = vmulq_f32(_p.val[3], _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p2);
                _r01.val[1] = float2int8(_p1, _p3);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += 16;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                _p0 = vmulq_f32(_p0, _scale);
                int8x8_t _r0 = float2int8(_p0, _p0);

                pp[0] = vget_lane_s8(_r0, 0);
                pp[1] = vget_lane_s8(_r0, 1);
                pp[2] = vget_lane_s8(_r0, 2);
                pp[3] = vget_lane_s8(_r0, 3);

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 3 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p4, _p6);
                int8x8_t _r2 = float2int8(_p1, _p3);
                int8x8_t _r3 = float2int8(_p5, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + B_hstep);
                float32x2_t _p2 = vld1_f32(p0 + B_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + B_hstep * 3);

                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[B_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[B_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[B_hstep * 3], _p0, 3);

                _p0 = vmulq_f32(_p0, _scale);
                int8x8_t _r0 = float2int8(_p0, _p0);

                pp[0] = vget_lane_s8(_r0, 0);
                pp[1] = vget_lane_s8(_r0, 1);
                pp[2] = vget_lane_s8(_r0, 2);
                pp[3] = vget_lane_s8(_r0, 3);

                pp += 4;
                p0++;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p1, _p3);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p2));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p2));
                float32x4_t _t2 = vcombine_f32(vget_low_f32(_p1), vget_low_f32(_p3));
                float32x4_t _t3 = vcombine_f32(vget_high_f32(_p1), vget_high_f32(_p3));
                int8x8_t _r0 = float2int8(_t0, _t1);
                int8x8_t _r1 = float2int8(_t2, _t3);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r0);
                vst1_s8(pp + 8, _r1);

                pp += 16;
                p0 += 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                int8x8_t _r0 = float2int8(_t0, _t1);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[B_hstep] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp += 4;
                p0 += 2;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 8;
            }
#endif // __ARM_NEON
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
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_B_tile_fp32_to_int8_i8mm(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_B_tile_fp32_to_int8_asimddp(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_fp32_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
#endif

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);
                float32x4_t _p8 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + B_hstep * 4 + 8);
                float32x4_t _pb = vld1q_f32(p0 + B_hstep * 4 + 12);
                float32x4_t _pc = vld1q_f32(p0 + B_hstep * 4 + 16);
                float32x4_t _pd = vld1q_f32(p0 + B_hstep * 4 + 20);
                float32x4_t _pe = vld1q_f32(p0 + B_hstep * 4 + 24);
                float32x4_t _pf = vld1q_f32(p0 + B_hstep * 4 + 28);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);
                _p8 = vmulq_f32(_p8, _scale);
                _p9 = vmulq_f32(_p9, _scale);
                _pa = vmulq_f32(_pa, _scale);
                _pb = vmulq_f32(_pb, _scale);
                _pc = vmulq_f32(_pc, _scale);
                _pd = vmulq_f32(_pd, _scale);
                _pe = vmulq_f32(_pe, _scale);
                _pf = vmulq_f32(_pf, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p8);
                int8x8_t _r1 = float2int8(_p1, _p9);
                int8x8_t _r2 = float2int8(_p2, _pa);
                int8x8_t _r3 = float2int8(_p3, _pb);
                int8x8_t _r4 = float2int8(_p4, _pc);
                int8x8_t _r5 = float2int8(_p5, _pd);
                int8x8_t _r6 = float2int8(_p6, _pe);
                int8x8_t _r7 = float2int8(_p7, _pf);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

                int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                int16x8_t _r45 = vreinterpretq_s16_s8(vcombine_s8(_r4, _r5));
                int16x8_t _r67 = vreinterpretq_s16_s8(vcombine_s8(_r6, _r7));
                int16x8x2_t _rr0 = vuzpq_s16(_r01, _r23);
                int16x8x2_t _rr1 = vuzpq_s16(_r45, _r67);

                vst1q_s8(pp, vreinterpretq_s8_s16(_rr0.val[0]));
                vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr0.val[1]));
                vst1q_s8(pp + 32, vreinterpretq_s8_s16(_rr1.val[0]));
                vst1q_s8(pp + 48, vreinterpretq_s8_s16(_rr1.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + 16);
                float32x4_t _p5 = vld1q_f32(p0 + 20);
                float32x4_t _p6 = vld1q_f32(p0 + 24);
                float32x4_t _p7 = vld1q_f32(p0 + 28);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

#if __ARM_FEATURE_DOTPROD
                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                int16x8x2_t _rr = vuzpq_s16(_r01, _r23);

                vst1q_s8(pp, vreinterpretq_s8_s16(_rr.val[0]));
                vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 3 + 4);
                float32x4_t _p8 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p9 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _pa = vld1q_f32(p0 + B_hstep * 5);
                float32x4_t _pb = vld1q_f32(p0 + B_hstep * 5 + 4);
                float32x4_t _pc = vld1q_f32(p0 + B_hstep * 6);
                float32x4_t _pd = vld1q_f32(p0 + B_hstep * 6 + 4);
                float32x4_t _pe = vld1q_f32(p0 + B_hstep * 7);
                float32x4_t _pf = vld1q_f32(p0 + B_hstep * 7 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);
                _p8 = vmulq_f32(_p8, _scale);
                _p9 = vmulq_f32(_p9, _scale);
                _pa = vmulq_f32(_pa, _scale);
                _pb = vmulq_f32(_pb, _scale);
                _pc = vmulq_f32(_pc, _scale);
                _pd = vmulq_f32(_pd, _scale);
                _pe = vmulq_f32(_pe, _scale);
                _pf = vmulq_f32(_pf, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
                int8x8_t _r4 = float2int8(_p8, _p9);
                int8x8_t _r5 = float2int8(_pa, _pb);
                int8x8_t _r6 = float2int8(_pc, _pd);
                int8x8_t _r7 = float2int8(_pe, _pf);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                int8x16x4_t _r0123;
                _r0123.val[0] = vcombine_s8(_r04.val[0], _r04.val[1]);
                _r0123.val[1] = vcombine_s8(_r15.val[0], _r15.val[1]);
                _r0123.val[2] = vcombine_s8(_r26.val[0], _r26.val[1]);
                _r0123.val[3] = vcombine_s8(_r37.val[0], _r37.val[1]);

                vst4q_s8(pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x4_t _r0123;
                _r0123.val[0] = _r0;
                _r0123.val[1] = _r1;
                _r0123.val[2] = _r2;
                _r0123.val[3] = _r3;
                int8x8x4_t _r4567;
                _r4567.val[0] = _r4;
                _r4567.val[1] = _r5;
                _r4567.val[2] = _r6;
                _r4567.val[3] = _r7;

                vst4_s8(pp, _r0123);
                vst4_s8(pp + 32, _r4567);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(_r0, _r2);
                _r01.val[1] = vcombine_s8(_r1, _r3);
                int8x16x2_t _r23;
                _r23.val[0] = vcombine_s8(_r4, _r6);
                _r23.val[1] = vcombine_s8(_r5, _r7);

                vst2q_s8(pp, _r01);
                vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                pp += 64;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 2 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 3 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p0, _p1);
                _r0123.val[1] = float2int8(_p2, _p3);
                _r0123.val[2] = float2int8(_p4, _p5);
                _r0123.val[3] = float2int8(_p6, _p7);

                vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p1);
                _r01.val[1] = float2int8(_p2, _p3);

                vst2_s8(pp, _r01);

                pp += 16;
                p0 += B_hstep * 2;
            }
            for (; kk < max_kk; kk++)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 4 + 4);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 4 + 8);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 4 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p4);
                int8x8_t _r1 = float2int8(_p1, _p5);
                int8x8_t _r2 = float2int8(_p2, _p6);
                int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p4, _p5));
                int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p6, _p7));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                pp += 32;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + 8);
                float32x4_t _p3 = vld1q_f32(p0 + 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 3);
                float32x4_t _p4 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p5 = vld1q_f32(p0 + B_hstep * 5);
                float32x4_t _p6 = vld1q_f32(p0 + B_hstep * 6);
                float32x4_t _p7 = vld1q_f32(p0 + B_hstep * 7);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);
                _p4 = vmulq_f32(_p4, _scale);
                _p5 = vmulq_f32(_p5, _scale);
                _p6 = vmulq_f32(_p6, _scale);
                _p7 = vmulq_f32(_p7, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);

                vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x4_t _r0123;
                _r0123.val[0] = float2int8(_p0, _p4);
                _r0123.val[1] = float2int8(_p1, _p5);
                _r0123.val[2] = float2int8(_p2, _p6);
                _r0123.val[3] = float2int8(_p3, _p7);

                vst4_s8(pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int8x16x2_t _r01;
                _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 32;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 2);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
                transpose4x4_ps(_p0, _p1, _p2, _p3);
                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else  // __ARM_FEATURE_DOTPROD
                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p0, _p2);
                _r01.val[1] = float2int8(_p1, _p3);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);
                int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                vst1_s8(pp, _r01);

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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + k * B_hstep + (j + jj) * elempack;

#if __ARM_NEON
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 4 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p2);
                int8x8_t _r1 = float2int8(_p1, _p3);
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1q_s8(pp, vcombine_s8(_r0, _r1));

                pp += 16;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

#if __ARM_FEATURE_DOTPROD
                int8x8_t _r01 = float2int8(_p0, _p1);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                int8x8_t _r01 = float2int8(_t0, _t1);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + B_hstep);
                float32x2_t _p2 = vld1_f32(p0 + B_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + B_hstep * 3);
                float32x2_t _p4 = vld1_f32(p0 + B_hstep * 4);
                float32x2_t _p5 = vld1_f32(p0 + B_hstep * 5);
                float32x2_t _p6 = vld1_f32(p0 + B_hstep * 6);
                float32x2_t _p7 = vld1_f32(p0 + B_hstep * 7);

#if __ARM_FEATURE_DOTPROD
                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);
                float32x4_t _p45 = vcombine_f32(_p4, _p5);
                float32x4_t _p67 = vcombine_f32(_p6, _p7);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);
                _p45 = vmulq_f32(_p45, _scale);
                _p67 = vmulq_f32(_p67, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);
                int8x8_t _r1 = float2int8(_p45, _p67);

#if __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r01 = vuzp_s8(_r0, _r1);

                vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
#else  // __ARM_FEATURE_MATMUL_INT8
                int8x8x2_t _r01 = vtrn_s8(_r0, _r1);
                int8x8x2_t _rr01 = vuzp_s8(_r01.val[0], _r01.val[1]);

                vst1q_s8(pp, vcombine_s8(_rr01.val[0], _rr01.val[1]));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p02 = vcombine_f32(_p0, _p2);
                float32x4_t _p46 = vcombine_f32(_p4, _p6);
                float32x4_t _p13 = vcombine_f32(_p1, _p3);
                float32x4_t _p57 = vcombine_f32(_p5, _p7);

                _p02 = vmulq_f32(_p02, _scale);
                _p46 = vmulq_f32(_p46, _scale);
                _p13 = vmulq_f32(_p13, _scale);
                _p57 = vmulq_f32(_p57, _scale);

                int8x8x2_t _r01;
                _r01.val[0] = float2int8(_p02, _p46);
                _r01.val[1] = float2int8(_p13, _p57);

                vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                pp += 16;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x2_t _p0 = vld1_f32(p0);
                float32x2_t _p1 = vld1_f32(p0 + B_hstep);
                float32x2_t _p2 = vld1_f32(p0 + B_hstep * 2);
                float32x2_t _p3 = vld1_f32(p0 + B_hstep * 3);

#if __ARM_FEATURE_DOTPROD
                float32x4_t _p01 = vcombine_f32(_p0, _p1);
                float32x4_t _p23 = vcombine_f32(_p2, _p3);

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                float32x4x2_t _pp = vuzpq_f32(_p01, _p23);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#else  // __ARM_FEATURE_DOTPROD
                float32x4_t _p02 = vcombine_f32(_p0, _p2);
                float32x4_t _p13 = vcombine_f32(_p1, _p3);

                _p02 = vmulq_f32(_p02, _scale);
                _p13 = vmulq_f32(_p13, _scale);

                float32x4x2_t _pp = vzipq_f32(_p02, _p13);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += B_hstep * 4;
            }
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[B_hstep + 0] * scale);
                pp[2] = float2int8(p0[1] * scale);
                pp[3] = float2int8(p0[B_hstep + 1] * scale);
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __ARM_NEON
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

#if __ARM_NEON
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep * 4);
                float32x4_t _p2 = vld1q_f32(p0 + B_hstep * 8);
                float32x4_t _p3 = vld1q_f32(p0 + B_hstep * 12);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));

                pp += 16;
                p0 += B_hstep * 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = vld1q_f32(p0);
                float32x4_t _p1 = vld1q_f32(p0 + B_hstep * 4);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp[1] = float2int8(p0[1] * scale);
                pp[2] = float2int8(p0[2] * scale);
                pp[3] = float2int8(p0[3] * scale);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                float32x4_t _p2 = float32x4_t();
                float32x4_t _p3 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[B_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[B_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[B_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[B_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[B_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[B_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[B_hstep * 7], _p1, 3);
                _p2 = vsetq_lane_f32(p0[B_hstep * 8], _p2, 0);
                _p2 = vsetq_lane_f32(p0[B_hstep * 9], _p2, 1);
                _p2 = vsetq_lane_f32(p0[B_hstep * 10], _p2, 2);
                _p2 = vsetq_lane_f32(p0[B_hstep * 11], _p2, 3);
                _p3 = vsetq_lane_f32(p0[B_hstep * 12], _p3, 0);
                _p3 = vsetq_lane_f32(p0[B_hstep * 13], _p3, 1);
                _p3 = vsetq_lane_f32(p0[B_hstep * 14], _p3, 2);
                _p3 = vsetq_lane_f32(p0[B_hstep * 15], _p3, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);
                _p2 = vmulq_f32(_p2, _scale);
                _p3 = vmulq_f32(_p3, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);
                int8x8_t _r23 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r01, _r23));

                pp += 16;
                p0 += B_hstep * 16;
            }
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _p0 = float32x4_t();
                float32x4_t _p1 = float32x4_t();
                _p0 = vsetq_lane_f32(p0[0], _p0, 0);
                _p0 = vsetq_lane_f32(p0[B_hstep], _p0, 1);
                _p0 = vsetq_lane_f32(p0[B_hstep * 2], _p0, 2);
                _p0 = vsetq_lane_f32(p0[B_hstep * 3], _p0, 3);
                _p1 = vsetq_lane_f32(p0[B_hstep * 4], _p1, 0);
                _p1 = vsetq_lane_f32(p0[B_hstep * 5], _p1, 1);
                _p1 = vsetq_lane_f32(p0[B_hstep * 6], _p1, 2);
                _p1 = vsetq_lane_f32(p0[B_hstep * 7], _p1, 3);

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += B_hstep * 8;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(p0[0] * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        unpack_output_tile_int32_to_fp32_asimddp(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    // NCNN_LOGE("unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack);

    const int* pp = topT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _descale0 = vld1q_f32((const float*)descales + i + ii);
        float32x4_t _descale1 = vld1q_f32((const float*)descales + i + ii + 4);

        float32x4_t _c0;
        float32x4_t _c1;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c1 = vld1q_f32(pC + 4);
                _c0 = vmulq_n_f32(_c0, beta);
                _c1 = vmulq_n_f32(_c1, beta);
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);
            int32x4_t _sum8 = vld1q_s32(pp + 32);
            int32x4_t _sum9 = vld1q_s32(pp + 36);
            int32x4_t _suma = vld1q_s32(pp + 40);
            int32x4_t _sumb = vld1q_s32(pp + 44);
            int32x4_t _sumc = vld1q_s32(pp + 48);
            int32x4_t _sumd = vld1q_s32(pp + 52);
            int32x4_t _sume = vld1q_s32(pp + 56);
            int32x4_t _sumf = vld1q_s32(pp + 60);

#if __ARM_FEATURE_DOTPROD
            // from/to
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
#else
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

            // to
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
                _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                _sum9 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                _suma = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                _sumb = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
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
#endif

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale0);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale0);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale0);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale0);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_suma), _descale0);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sumb), _descale0);
            float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale1);
            float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale1);
            float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_sum6), _descale1);
            float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sum7), _descale1);
            float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale1);
            float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale1);
            float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale1);
            float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                    _f8 = vaddq_f32(_f8, _c0);
                    _f9 = vaddq_f32(_f9, _c0);
                    _fa = vaddq_f32(_fa, _c0);
                    _fb = vaddq_f32(_fb, _c0);
                    _fc = vaddq_f32(_fc, _c0);
                    _fd = vaddq_f32(_fd, _c0);
                    _fe = vaddq_f32(_fe, _c0);
                    _ff = vaddq_f32(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                    _f8 = vaddq_f32(_f8, _c1);
                    _f9 = vaddq_f32(_f9, _c1);
                    _fa = vaddq_f32(_fa, _c1);
                    _fb = vaddq_f32(_fb, _c1);
                    _fc = vaddq_f32(_fc, _c1);
                    _fd = vaddq_f32(_fd, _c1);
                    _fe = vaddq_f32(_fe, _c1);
                    _ff = vaddq_f32(_ff, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 4 * 2);
                        float32x4_t _c3 = vld1q_f32(pC + 4 * 3);
                        float32x4_t _c4 = vld1q_f32(pC + 4 * 4);
                        float32x4_t _c5 = vld1q_f32(pC + 4 * 5);
                        float32x4_t _c6 = vld1q_f32(pC + 4 * 6);
                        float32x4_t _c7 = vld1q_f32(pC + 4 * 7);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 4 * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 4 * 3);
                        _c4 = vld1q_f32(pC + c_hstep * 4 + 4 * 4);
                        _c5 = vld1q_f32(pC + c_hstep * 4 + 4 * 5);
                        _c6 = vld1q_f32(pC + c_hstep * 4 + 4 * 6);
                        _c7 = vld1q_f32(pC + c_hstep * 4 + 4 * 7);
                        if (beta == 1.f)
                        {
                            _f8 = vaddq_f32(_f8, _c0);
                            _f9 = vaddq_f32(_f9, _c1);
                            _fa = vaddq_f32(_fa, _c2);
                            _fb = vaddq_f32(_fb, _c3);
                            _fc = vaddq_f32(_fc, _c4);
                            _fd = vaddq_f32(_fd, _c5);
                            _fe = vaddq_f32(_fe, _c6);
                            _ff = vaddq_f32(_ff, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f8 = vmlaq_f32(_f8, _c0, _beta);
                            _f9 = vmlaq_f32(_f9, _c1, _beta);
                            _fa = vmlaq_f32(_fa, _c2, _beta);
                            _fb = vmlaq_f32(_fb, _c3, _beta);
                            _fc = vmlaq_f32(_fc, _c4, _beta);
                            _fd = vmlaq_f32(_fd, _c5, _beta);
                            _fe = vmlaq_f32(_fe, _c6, _beta);
                            _ff = vmlaq_f32(_ff, _c7, _beta);
                        }
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 5);
                        _c3 = vld1q_f32(pC + c_hstep * 5 + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 6);
                        _c5 = vld1q_f32(pC + c_hstep * 6 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 7);
                        _c7 = vld1q_f32(pC + c_hstep * 7 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        if (beta == 1.f)
                        {
                            _f8 = vaddq_f32(_f8, _c0);
                            _f9 = vaddq_f32(_f9, _c1);
                            _fa = vaddq_f32(_fa, _c2);
                            _fb = vaddq_f32(_fb, _c3);
                            _fc = vaddq_f32(_fc, _c4);
                            _fd = vaddq_f32(_fd, _c5);
                            _fe = vaddq_f32(_fe, _c6);
                            _ff = vaddq_f32(_ff, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f8 = vmlaq_f32(_f8, _c0, _beta);
                            _f9 = vmlaq_f32(_f9, _c1, _beta);
                            _fa = vmlaq_f32(_fa, _c2, _beta);
                            _fb = vmlaq_f32(_fb, _c3, _beta);
                            _fc = vmlaq_f32(_fc, _c4, _beta);
                            _fd = vmlaq_f32(_fd, _c5, _beta);
                            _fe = vmlaq_f32(_fe, _c6, _beta);
                            _ff = vmlaq_f32(_ff, _c7, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _cc0 = vmulq_f32(_cc0, _beta);
                        _cc1 = vmulq_f32(_cc1, _beta);
                    }
                    _c0 = vdupq_laneq_f32(_cc0, 0);
                    _c1 = vdupq_laneq_f32(_cc0, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc0, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc0, 3);
                    float32x4_t _c4 = vdupq_laneq_f32(_cc1, 0);
                    float32x4_t _c5 = vdupq_laneq_f32(_cc1, 1);
                    float32x4_t _c6 = vdupq_laneq_f32(_cc1, 2);
                    float32x4_t _c7 = vdupq_laneq_f32(_cc1, 3);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c4);
                    _f5 = vaddq_f32(_f5, _c5);
                    _f6 = vaddq_f32(_f6, _c6);
                    _f7 = vaddq_f32(_f7, _c7);
                    _f8 = vaddq_f32(_f8, _c0);
                    _f9 = vaddq_f32(_f9, _c1);
                    _fa = vaddq_f32(_fa, _c2);
                    _fb = vaddq_f32(_fb, _c3);
                    _fc = vaddq_f32(_fc, _c4);
                    _fd = vaddq_f32(_fd, _c5);
                    _fe = vaddq_f32(_fe, _c6);
                    _ff = vaddq_f32(_ff, _c7);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
                _f8 = vmulq_f32(_f8, _alpha);
                _f9 = vmulq_f32(_f9, _alpha);
                _fa = vmulq_f32(_fa, _alpha);
                _fb = vmulq_f32(_fb, _alpha);
                _fc = vmulq_f32(_fc, _alpha);
                _fd = vmulq_f32(_fd, _alpha);
                _fe = vmulq_f32(_fe, _alpha);
                _ff = vmulq_f32(_ff, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + 16, _f4);
                vst1q_f32(p0 + 20, _f5);
                vst1q_f32(p0 + 24, _f6);
                vst1q_f32(p0 + 28, _f7);
                vst1q_f32(p0 + out_hstep * 4, _f8);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f9);
                vst1q_f32(p0 + out_hstep * 4 + 8, _fa);
                vst1q_f32(p0 + out_hstep * 4 + 12, _fb);
                vst1q_f32(p0 + out_hstep * 4 + 16, _fc);
                vst1q_f32(p0 + out_hstep * 4 + 20, _fd);
                vst1q_f32(p0 + out_hstep * 4 + 24, _fe);
                vst1q_f32(p0 + out_hstep * 4 + 28, _ff);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);
                transpose4x4_ps(_f8, _f9, _fa, _fb);
                transpose4x4_ps(_fc, _fd, _fe, _ff);
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f4);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f5);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _f6);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _f7);
                vst1q_f32(p0 + out_hstep * 4, _f8);
                vst1q_f32(p0 + out_hstep * 4 + 4, _fc);
                vst1q_f32(p0 + out_hstep * 5, _f9);
                vst1q_f32(p0 + out_hstep * 5 + 4, _fd);
                vst1q_f32(p0 + out_hstep * 6, _fa);
                vst1q_f32(p0 + out_hstep * 6 + 4, _fe);
                vst1q_f32(p0 + out_hstep * 7, _fb);
                vst1q_f32(p0 + out_hstep * 7 + 4, _ff);
                p0 += 8;
            }

            pp += 64;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
            //      e2 f2 g2 h2
            //      e3 f3 g3 h3
#else
            // from
            //      a0 b1 c2 d3
            //      e0 f1 g2 h3
            //      c0 d1 a2 b3
            //      g0 h1 e2 f3
            //      a3 b2 c1 d0
            //      e3 f2 g1 h0
            //      c3 d2 a1 b0
            //      g3 h2 e1 f0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale0);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale0);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale1);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale1);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale1);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c1);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c1);
                    _f7 = vaddq_f32(_f7, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 5);
                        _c2 = vld1q_f32(pC + c_hstep * 6);
                        _c3 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f4 = vaddq_f32(_f4, _c0);
                        _f5 = vaddq_f32(_f5, _c1);
                        _f6 = vaddq_f32(_f6, _c2);
                        _f7 = vaddq_f32(_f7, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f4 = vmlaq_f32(_f4, _c0, _beta);
                        _f5 = vmlaq_f32(_f5, _c1, _beta);
                        _f6 = vmlaq_f32(_f6, _c2, _beta);
                        _f7 = vmlaq_f32(_f7, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    _c = vmulq_n_f32(_c, beta);
#if __aarch64__
                    _c0 = vdupq_laneq_f32(_c, 0);
                    _c1 = vdupq_laneq_f32(_c, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_c, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_c, 3);
#else
                    _c0 = vdupq_lane_f32(vget_low_f32(_c), 0);
                    _c1 = vdupq_lane_f32(vget_low_f32(_c), 1);
                    float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_c), 0);
                    float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_c), 1);
#endif
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c2);
                    _f7 = vaddq_f32(_f7, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f5);
                vst1q_f32(p0 + out_hstep * 4 + 8, _f6);
                vst1q_f32(p0 + out_hstep * 4 + 12, _f7);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 5, _f5);
                vst1q_f32(p0 + out_hstep * 6, _f6);
                vst1q_f32(p0 + out_hstep * 7, _f7);
                p0 += 4;
            }

            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
#else
            // from
            //      a0 b1 c0 d1
            //      e0 f1 g0 h1
            //      a1 b0 c1 d0
            //      e1 f0 g1 h0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale1);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c1);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x4_t _c01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _c23 = vcombine_f32(_cc2, _cc3);
                        float32x4x2_t _ccc0 = vuzpq_f32(_c01, _c23);
                        _c0 = _ccc0.val[0];
                        _c1 = _ccc0.val[1];
                        float32x2_t _cc4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _cc5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _cc6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _cc7 = vld1_f32(pC + c_hstep * 7);
                        float32x4_t _c45 = vcombine_f32(_cc4, _cc5);
                        float32x4_t _c67 = vcombine_f32(_cc6, _cc7);
                        float32x4x2_t _ccc1 = vuzpq_f32(_c45, _c67);
                        _c2 = _ccc1.val[0];
                        _c3 = _ccc1.val[1];
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c = vmul_n_f32(_c, beta);
                    _c0 = vdupq_lane_f32(_c, 0);
                    _c1 = vdupq_lane_f32(_c, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + out_hstep * 4, _f2);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f3);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                float32x4x2_t _f23 = vzipq_f32(_f2, _f3);
                vst1_f32(p0, vget_low_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
                vst1_f32(p0 + out_hstep * 4, vget_low_f32(_f23.val[0]));
                vst1_f32(p0 + out_hstep * 5, vget_high_f32(_f23.val[0]));
                vst1_f32(p0 + out_hstep * 6, vget_low_f32(_f23.val[1]));
                vst1_f32(p0 + out_hstep * 7, vget_high_f32(_f23.val[1]));
                p0 += 2;
            }

            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 4);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 4], _c1, 0);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 5], _c1, 1);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 6], _c1, 2);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 7], _c1, 3);
                        pC += 1;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep * 4, _f1);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = vgetq_lane_f32(_f0, 0);
                p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                p0[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                p0[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                p0[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                p0[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                p0++;
            }

            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _descale = vld1q_f32((const float*)descales + i + ii);

        float32x4_t _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c0 = vmulq_n_f32(_c0, beta);
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      a4 b4 c4 d4
            //      a5 b5 c5 d5
            //      a6 b6 c6 d6
            //      a7 b7 c7 d7
#else
            // from
            //      a0 b1 c2 d3
            //      a4 b5 c6 d7
            //      c0 d1 a2 b3
            //      c4 d5 a6 b7
            //      a3 b2 c1 d0
            //      a7 b6 c5 d4
            //      c3 d2 a1 b0
            //      c7 d6 a5 b4

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    float32x4_t _c4;
                    float32x4_t _c5;
                    float32x4_t _c6;
                    float32x4_t _c7;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        _c4 = vld1q_f32(pC + 16);
                        _c5 = vld1q_f32(pC + 20);
                        _c6 = vld1q_f32(pC + 24);
                        _c7 = vld1q_f32(pC + 28);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + c_hstep);
                        _c3 = vld1q_f32(pC + c_hstep + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 2);
                        _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 3);
                        _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                        _f4 = vmlaq_f32(_f4, _c4, _beta);
                        _f5 = vmlaq_f32(_f5, _c5, _beta);
                        _f6 = vmlaq_f32(_f6, _c6, _beta);
                        _f7 = vmlaq_f32(_f7, _c7, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _cc0 = vmulq_f32(_cc0, _beta);
                        _cc1 = vmulq_f32(_cc1, _beta);
                    }
                    _c0 = vdupq_laneq_f32(_cc0, 0);
                    float32x4_t _c1 = vdupq_laneq_f32(_cc0, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc0, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc0, 3);
                    float32x4_t _c4 = vdupq_laneq_f32(_cc1, 0);
                    float32x4_t _c5 = vdupq_laneq_f32(_cc1, 1);
                    float32x4_t _c6 = vdupq_laneq_f32(_cc1, 2);
                    float32x4_t _c7 = vdupq_laneq_f32(_cc1, 3);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c4);
                    _f5 = vaddq_f32(_f5, _c5);
                    _f6 = vaddq_f32(_f6, _c6);
                    _f7 = vaddq_f32(_f7, _c7);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                vst1q_f32(p0 + 16, _f4);
                vst1q_f32(p0 + 20, _f5);
                vst1q_f32(p0 + 24, _f6);
                vst1q_f32(p0 + 28, _f7);
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f4);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f5);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _f6);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _f7);
                p0 += 8;
            }

            pp += 32;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
#else
            // from
            //      a0 b1 c2 d3
            //      c0 d1 a2 b3
            //      a3 b2 c1 d0
            //      c3 d2 a1 b0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 1);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _c = vld1q_f32(pC);
                    _c = vmulq_n_f32(_c, beta);
#if __aarch64__
                    _c0 = vdupq_laneq_f32(_c, 0);
                    float32x4_t _c1 = vdupq_laneq_f32(_c, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_c, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_c, 3);
#else
                    _c0 = vdupq_lane_f32(vget_low_f32(_c), 0);
                    float32x4_t _c1 = vdupq_lane_f32(vget_low_f32(_c), 1);
                    float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_c), 0);
                    float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_c), 1);
#endif
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                p0 += 4;
            }

            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
#else
            // from
            //      a0 b1 c0 d1
            //      a1 b0 c1 d0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x4_t _c01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _c23 = vcombine_f32(_cc2, _cc3);
                        float32x4x2_t _cc = vuzpq_f32(_c01, _c23);
                        _c0 = _cc.val[0];
                        _c1 = _cc.val[1];
                        pC += 2;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c = vmul_n_f32(_c, beta);
                    _c0 = vdupq_lane_f32(_c, 0);
                    float32x4_t _c1 = vdupq_lane_f32(_c, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                vst1_f32(p0, vget_low_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
                p0 += 2;
            }

            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        pC += 1;
                    }
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = vgetq_lane_f32(_f0, 0);
                p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                p0++;
            }

            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // out_elempack == 1
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float descale0 = descales[i + ii];
        const float descale1 = descales[i + ii + 1];
#if __ARM_NEON
        float32x2_t _descale = vld1_f32((const float*)descales + i + ii);
#endif

        float c0;
        float c1;
#if __ARM_NEON
        float32x4_t _c0;
        float32x4_t _c1;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
                _c1 = vdupq_n_f32(c1);
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
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

            float32x4_t _f0 = vmulq_lane_f32(vcvtq_f32_s32(_sum0), _descale, 0);
            float32x4_t _f1 = vmulq_lane_f32(vcvtq_f32_s32(_sum1), _descale, 0);
            float32x4_t _f2 = vmulq_lane_f32(vcvtq_f32_s32(_sum2), _descale, 1);
            float32x4_t _f3 = vmulq_lane_f32(vcvtq_f32_s32(_sum3), _descale, 1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c1);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _c0 = vmulq_f32(_c0, _beta);
                        _c1 = vmulq_f32(_c1, _beta);
                    }
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + 4, _f1);
            vst1q_f32(p0 + out_hstep, _f2);
            vst1q_f32(p0 + out_hstep + 4, _f3);

            pp += 16;
            p0 += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

            float32x4_t _f0 = vmulq_lane_f32(vcvtq_f32_s32(_sum0), _descale, 0);
            float32x4_t _f1 = vmulq_lane_f32(vcvtq_f32_s32(_sum1), _descale, 1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vld1q_f32(pC);
                    _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + out_hstep, _f1);

            pp += 8;
            p0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            int32x4_t _sum0 = vld1q_s32(pp);

            float32x2x2_t _descale01 = vzip_f32(_descale, _descale);
            float32x4_t _descale0011 = vcombine_f32(_descale01.val[0], _descale01.val[1]);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0011);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c0011 = vcombine_f32(vget_low_f32(_c0), vget_high_f32(_c1));
                    _f0 = vaddq_f32(_f0, _c0011);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c0 = vcombine_f32(_c, _c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1_f32(p0, vget_low_f32(_f0));
            vst1_f32(p0 + out_hstep, vget_high_f32(_f0));

            pp += 4;
            p0 += 2;
        }
#endif // __ARM_NEON
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

            p0[0] = f0;
            p0[out_hstep] = f1;

            pp += 2;
            p0++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // out_elempack == 1
        float* p0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float descale = descales[i + ii];
#if __ARM_NEON
        float32x4_t _descale = vdupq_n_f32(descale);
#endif

        float c0;
#if __ARM_NEON
        float32x4_t _c0;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
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
#if __ARM_NEON
        for (; jj + 15 < max_jj; jj += 16)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    float32x4_t _c3 = vld1q_f32(pC + 12);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + 4, _f1);
            vst1q_f32(p0 + 8, _f2);
            vst1q_f32(p0 + 12, _f3);

            pp += 16;
            p0 += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + 4, _f1);

            pp += 8;
            p0 += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 4;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1q_f32(p0, _f0);

            pp += 4;
            p0 += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _f0 = vmul_f32(vcvt_f32_s32(vld1_s32(pp)), vget_low_f32(_descale));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vadd_f32(_f0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // out_elempack == 1
                    float32x2_t _c = vld1_f32(pC);
                    _f0 = vmla_n_f32(_f0, _c, beta);
                    pC += 2;
                }
            }

            _f0 = vmul_n_f32(_f0, alpha);

            vst1_f32(p0, _f0);

            pp += 2;
            p0 += 2;
        }
#endif // __ARM_NEON
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

            pp += 1;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_unpack_output_tile_int32_to_fp32_asimddp(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const float* pC = C;

    // NCNN_LOGE("transpose_unpack_output_tile_int32_to_fp32  %d %d %d %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack);

    const int* pp = topT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        float32x4_t _descale0 = vld1q_f32((const float*)descales + i + ii);
        float32x4_t _descale1 = vld1q_f32((const float*)descales + i + ii + 4);

        float32x4_t _c0;
        float32x4_t _c1;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c1 = vld1q_f32(pC + 4);
                _c0 = vmulq_n_f32(_c0, beta);
                _c1 = vmulq_n_f32(_c1, beta);
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);
            int32x4_t _sum8 = vld1q_s32(pp + 32);
            int32x4_t _sum9 = vld1q_s32(pp + 36);
            int32x4_t _suma = vld1q_s32(pp + 40);
            int32x4_t _sumb = vld1q_s32(pp + 44);
            int32x4_t _sumc = vld1q_s32(pp + 48);
            int32x4_t _sumd = vld1q_s32(pp + 52);
            int32x4_t _sume = vld1q_s32(pp + 56);
            int32x4_t _sumf = vld1q_s32(pp + 60);

#if __ARM_FEATURE_DOTPROD
            // from/to
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
#else
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

            // to
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
                _sum4 = vcombine_s32(vget_low_s32(_t2.val[0]), vget_low_s32(_t3.val[0]));
                _sum5 = vcombine_s32(vget_high_s32(_t2.val[0]), vget_high_s32(_t3.val[0]));
                _sum6 = vcombine_s32(vget_low_s32(_t3.val[1]), vget_low_s32(_t2.val[1]));
                _sum7 = vcombine_s32(vget_high_s32(_t3.val[1]), vget_high_s32(_t2.val[1]));
                _sum8 = vcombine_s32(vget_low_s32(_t4.val[0]), vget_low_s32(_t5.val[0]));
                _sum9 = vcombine_s32(vget_high_s32(_t4.val[0]), vget_high_s32(_t5.val[0]));
                _suma = vcombine_s32(vget_low_s32(_t5.val[1]), vget_low_s32(_t4.val[1]));
                _sumb = vcombine_s32(vget_high_s32(_t5.val[1]), vget_high_s32(_t4.val[1]));
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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale0);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale0);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum8), _descale0);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum9), _descale0);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_suma), _descale0);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sumb), _descale0);
            float32x4_t _f8 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale1);
            float32x4_t _f9 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale1);
            float32x4_t _fa = vmulq_f32(vcvtq_f32_s32(_sum6), _descale1);
            float32x4_t _fb = vmulq_f32(vcvtq_f32_s32(_sum7), _descale1);
            float32x4_t _fc = vmulq_f32(vcvtq_f32_s32(_sumc), _descale1);
            float32x4_t _fd = vmulq_f32(vcvtq_f32_s32(_sumd), _descale1);
            float32x4_t _fe = vmulq_f32(vcvtq_f32_s32(_sume), _descale1);
            float32x4_t _ff = vmulq_f32(vcvtq_f32_s32(_sumf), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                    _f8 = vaddq_f32(_f8, _c0);
                    _f9 = vaddq_f32(_f9, _c0);
                    _fa = vaddq_f32(_fa, _c0);
                    _fb = vaddq_f32(_fb, _c0);
                    _fc = vaddq_f32(_fc, _c0);
                    _fd = vaddq_f32(_fd, _c0);
                    _fe = vaddq_f32(_fe, _c0);
                    _ff = vaddq_f32(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                    _f8 = vaddq_f32(_f8, _c1);
                    _f9 = vaddq_f32(_f9, _c1);
                    _fa = vaddq_f32(_fa, _c1);
                    _fb = vaddq_f32(_fb, _c1);
                    _fc = vaddq_f32(_fc, _c1);
                    _fd = vaddq_f32(_fd, _c1);
                    _fe = vaddq_f32(_fe, _c1);
                    _ff = vaddq_f32(_ff, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        float32x4_t _c4 = vld1q_f32(pC + 16);
                        float32x4_t _c5 = vld1q_f32(pC + 20);
                        float32x4_t _c6 = vld1q_f32(pC + 24);
                        float32x4_t _c7 = vld1q_f32(pC + 28);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        _c4 = vld1q_f32(pC + c_hstep * 4 + 16);
                        _c5 = vld1q_f32(pC + c_hstep * 4 + 20);
                        _c6 = vld1q_f32(pC + c_hstep * 4 + 24);
                        _c7 = vld1q_f32(pC + c_hstep * 4 + 28);
                        if (beta == 1.f)
                        {
                            _f8 = vaddq_f32(_f8, _c0);
                            _f9 = vaddq_f32(_f9, _c1);
                            _fa = vaddq_f32(_fa, _c2);
                            _fb = vaddq_f32(_fb, _c3);
                            _fc = vaddq_f32(_fc, _c4);
                            _fd = vaddq_f32(_fd, _c5);
                            _fe = vaddq_f32(_fe, _c6);
                            _ff = vaddq_f32(_ff, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f8 = vmlaq_f32(_f8, _c0, _beta);
                            _f9 = vmlaq_f32(_f9, _c1, _beta);
                            _fa = vmlaq_f32(_fa, _c2, _beta);
                            _fb = vmlaq_f32(_fb, _c3, _beta);
                            _fc = vmlaq_f32(_fc, _c4, _beta);
                            _fd = vmlaq_f32(_fd, _c5, _beta);
                            _fe = vmlaq_f32(_fe, _c6, _beta);
                            _ff = vmlaq_f32(_ff, _c7, _beta);
                        }
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                        float32x4_t _c4 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        float32x4_t _c6 = vld1q_f32(pC + c_hstep * 3);
                        float32x4_t _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                            _f4 = vaddq_f32(_f4, _c4);
                            _f5 = vaddq_f32(_f5, _c5);
                            _f6 = vaddq_f32(_f6, _c6);
                            _f7 = vaddq_f32(_f7, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                            _f4 = vmlaq_f32(_f4, _c4, _beta);
                            _f5 = vmlaq_f32(_f5, _c5, _beta);
                            _f6 = vmlaq_f32(_f6, _c6, _beta);
                            _f7 = vmlaq_f32(_f7, _c7, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 5);
                        _c3 = vld1q_f32(pC + c_hstep * 5 + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 6);
                        _c5 = vld1q_f32(pC + c_hstep * 6 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 7);
                        _c7 = vld1q_f32(pC + c_hstep * 7 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        if (beta == 1.f)
                        {
                            _f8 = vaddq_f32(_f8, _c0);
                            _f9 = vaddq_f32(_f9, _c1);
                            _fa = vaddq_f32(_fa, _c2);
                            _fb = vaddq_f32(_fb, _c3);
                            _fc = vaddq_f32(_fc, _c4);
                            _fd = vaddq_f32(_fd, _c5);
                            _fe = vaddq_f32(_fe, _c6);
                            _ff = vaddq_f32(_ff, _c7);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f8 = vmlaq_f32(_f8, _c0, _beta);
                            _f9 = vmlaq_f32(_f9, _c1, _beta);
                            _fa = vmlaq_f32(_fa, _c2, _beta);
                            _fb = vmlaq_f32(_fb, _c3, _beta);
                            _fc = vmlaq_f32(_fc, _c4, _beta);
                            _fd = vmlaq_f32(_fd, _c5, _beta);
                            _fe = vmlaq_f32(_fe, _c6, _beta);
                            _ff = vmlaq_f32(_ff, _c7, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _cc0 = vmulq_f32(_cc0, _beta);
                        _cc1 = vmulq_f32(_cc1, _beta);
                    }
                    _c0 = vdupq_laneq_f32(_cc0, 0);
                    _c1 = vdupq_laneq_f32(_cc0, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc0, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc0, 3);
                    float32x4_t _c4 = vdupq_laneq_f32(_cc1, 0);
                    float32x4_t _c5 = vdupq_laneq_f32(_cc1, 1);
                    float32x4_t _c6 = vdupq_laneq_f32(_cc1, 2);
                    float32x4_t _c7 = vdupq_laneq_f32(_cc1, 3);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c4);
                    _f5 = vaddq_f32(_f5, _c5);
                    _f6 = vaddq_f32(_f6, _c6);
                    _f7 = vaddq_f32(_f7, _c7);
                    _f8 = vaddq_f32(_f8, _c0);
                    _f9 = vaddq_f32(_f9, _c1);
                    _fa = vaddq_f32(_fa, _c2);
                    _fb = vaddq_f32(_fb, _c3);
                    _fc = vaddq_f32(_fc, _c4);
                    _fd = vaddq_f32(_fd, _c5);
                    _fe = vaddq_f32(_fe, _c6);
                    _ff = vaddq_f32(_ff, _c7);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
                _f8 = vmulq_f32(_f8, _alpha);
                _f9 = vmulq_f32(_f9, _alpha);
                _fa = vmulq_f32(_fa, _alpha);
                _fb = vmulq_f32(_fb, _alpha);
                _fc = vmulq_f32(_fc, _alpha);
                _fd = vmulq_f32(_fd, _alpha);
                _fe = vmulq_f32(_fe, _alpha);
                _ff = vmulq_f32(_ff, _alpha);
            }

            if (out_elempack == 4)
            {
                float32x4x4_t _ffa;
                float32x4x4_t _ffb;
                float32x4x4_t _ffc;
                float32x4x4_t _ffd;
                _ffa.val[0] = _f0;
                _ffa.val[1] = _f1;
                _ffa.val[2] = _f2;
                _ffa.val[3] = _f3;
                _ffb.val[0] = _f4;
                _ffb.val[1] = _f5;
                _ffb.val[2] = _f6;
                _ffb.val[3] = _f7;
                _ffc.val[0] = _f8;
                _ffc.val[1] = _f9;
                _ffc.val[2] = _fa;
                _ffc.val[3] = _fb;
                _ffd.val[0] = _fc;
                _ffd.val[1] = _fd;
                _ffd.val[2] = _fe;
                _ffd.val[3] = _ff;
                vst4q_f32(p0, _ffa);
                vst4q_f32(p0 + 16, _ffc);
                vst4q_f32(p0 + out_hstep * 4, _ffb);
                vst4q_f32(p0 + out_hstep * 4 + 16, _ffd);
            }
            if (out_elempack == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f8);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f9);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _fa);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _fb);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 4 + 4, _fc);
                vst1q_f32(p0 + out_hstep * 5, _f5);
                vst1q_f32(p0 + out_hstep * 5 + 4, _fd);
                vst1q_f32(p0 + out_hstep * 6, _f6);
                vst1q_f32(p0 + out_hstep * 6 + 4, _fe);
                vst1q_f32(p0 + out_hstep * 7, _f7);
                vst1q_f32(p0 + out_hstep * 7 + 4, _ff);
            }

            pp += 64;
            p0 += out_hstep * 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
            //      e2 f2 g2 h2
            //      e3 f3 g3 h3

#else
            // from
            //      a0 b1 c2 d3
            //      e0 f1 g2 h3
            //      c0 d1 a2 b3
            //      g0 h1 e2 f3
            //      a3 b2 c1 d0
            //      e3 f2 g1 h0
            //      c3 d2 a1 b0
            //      g3 h2 e1 f0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale0);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale0);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale1);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale1);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale1);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c1);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c1);
                    _f7 = vaddq_f32(_f7, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + 8);
                        float32x4_t _c3 = vld1q_f32(pC + 12);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 4 + 4);
                        _c2 = vld1q_f32(pC + c_hstep * 4 + 8);
                        _c3 = vld1q_f32(pC + c_hstep * 4 + 12);
                        if (beta == 1.f)
                        {
                            _f4 = vaddq_f32(_f4, _c0);
                            _f5 = vaddq_f32(_f5, _c1);
                            _f6 = vaddq_f32(_f6, _c2);
                            _f7 = vaddq_f32(_f7, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f4 = vmlaq_f32(_f4, _c0, _beta);
                            _f5 = vmlaq_f32(_f5, _c1, _beta);
                            _f6 = vmlaq_f32(_f6, _c2, _beta);
                            _f7 = vmlaq_f32(_f7, _c3, _beta);
                        }
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 2);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                        }
                        _c0 = vld1q_f32(pC + c_hstep * 4);
                        _c1 = vld1q_f32(pC + c_hstep * 5);
                        _c2 = vld1q_f32(pC + c_hstep * 6);
                        _c3 = vld1q_f32(pC + c_hstep * 7);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        if (beta == 1.f)
                        {
                            _f4 = vaddq_f32(_f4, _c0);
                            _f5 = vaddq_f32(_f5, _c1);
                            _f6 = vaddq_f32(_f6, _c2);
                            _f7 = vaddq_f32(_f7, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f4 = vmlaq_f32(_f4, _c0, _beta);
                            _f5 = vmlaq_f32(_f5, _c1, _beta);
                            _f6 = vmlaq_f32(_f6, _c2, _beta);
                            _f7 = vmlaq_f32(_f7, _c3, _beta);
                        }
                        pC += 4;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc = vld1q_f32(pC);
                    _cc = vmulq_n_f32(_cc, beta);
#if __aarch64__
                    _c0 = vdupq_laneq_f32(_cc, 0);
                    _c1 = vdupq_laneq_f32(_cc, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc, 3);
#else
                    _c0 = vdupq_lane_f32(vget_low_f32(_cc), 0);
                    _c1 = vdupq_lane_f32(vget_low_f32(_cc), 1);
                    float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_cc), 0);
                    float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_cc), 1);
#endif
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c1);
                    _f6 = vaddq_f32(_f6, _c2);
                    _f7 = vaddq_f32(_f7, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }

            if (out_elempack == 4)
            {
                float32x4x4_t _fa;
                float32x4x4_t _fb;
                _fa.val[0] = _f0;
                _fa.val[1] = _f1;
                _fa.val[2] = _f2;
                _fa.val[3] = _f3;
                _fb.val[0] = _f4;
                _fb.val[1] = _f5;
                _fb.val[2] = _f6;
                _fb.val[3] = _f7;
                vst4q_f32(p0, _fa);
                vst4q_f32(p0 + 16, _fb);
            }
            if (out_elempack == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f4);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep + 4, _f5);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 2 + 4, _f6);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 3 + 4, _f7);
            }

            pp += 32;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      e0 f0 g0 h0
            //      e1 f1 g1 h1
#else
            // from
            //      a0 b1 c0 d1
            //      e0 f1 g0 h1
            //      a1 b0 c1 d0
            //      e1 f0 g1 h0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale0);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale1);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c1);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x2_t _cc4 = vld1_f32(pC + c_hstep * 4);
                        float32x2_t _cc5 = vld1_f32(pC + c_hstep * 5);
                        float32x2_t _cc6 = vld1_f32(pC + c_hstep * 6);
                        float32x2_t _cc7 = vld1_f32(pC + c_hstep * 7);
                        float32x4_t _cc01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _cc23 = vcombine_f32(_cc2, _cc3);
                        float32x4_t _cc45 = vcombine_f32(_cc4, _cc5);
                        float32x4_t _cc67 = vcombine_f32(_cc6, _cc7);
                        float32x4x2_t _ccc0 = vuzpq_f32(_cc01, _cc23);
                        float32x4x2_t _ccc1 = vuzpq_f32(_cc45, _cc67);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _ccc0.val[0]);
                            _f1 = vaddq_f32(_f1, _ccc0.val[1]);
                            _f2 = vaddq_f32(_f2, _ccc1.val[0]);
                            _f3 = vaddq_f32(_f3, _ccc1.val[1]);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _ccc0.val[0], _beta);
                            _f1 = vmlaq_f32(_f1, _ccc0.val[1], _beta);
                            _f2 = vmlaq_f32(_f2, _ccc1.val[0], _beta);
                            _f3 = vmlaq_f32(_f3, _ccc1.val[1], _beta);
                        }
                        pC += 2;
                    }
                    else // if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        float32x4_t _c2 = vld1q_f32(pC + c_hstep * 4);
                        float32x4_t _c3 = vld1q_f32(pC + c_hstep * 4 + 4);
                        if (beta == 1.f)
                        {
                            _f0 = vaddq_f32(_f0, _c0);
                            _f1 = vaddq_f32(_f1, _c1);
                            _f2 = vaddq_f32(_f2, _c2);
                            _f3 = vaddq_f32(_f3, _c3);
                        }
                        else
                        {
                            float32x4_t _beta = vdupq_n_f32(beta);
                            _f0 = vmlaq_f32(_f0, _c0, _beta);
                            _f1 = vmlaq_f32(_f1, _c1, _beta);
                            _f2 = vmlaq_f32(_f2, _c2, _beta);
                            _f3 = vmlaq_f32(_f3, _c3, _beta);
                        }
                        pC += 8;
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _cc = vld1_f32(pC);
                    _cc = vmul_n_f32(_cc, beta);
                    _c0 = vdupq_lane_f32(_cc, 0);
                    _c1 = vdupq_lane_f32(_cc, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + 4, _f2);
            vst1q_f32(p0 + out_hstep, _f1);
            vst1q_f32(p0 + out_hstep + 4, _f3);

            pp += 16;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale0);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp + 4)), _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 4], _c1, 0);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 5], _c1, 1);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 6], _c1, 2);
                        _c1 = vsetq_lane_f32(pC[c_hstep * 7], _c1, 3);
                        pC += 1;
                    }
                    else // if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep * 4);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + 4, _f1);
            pp += 8;
            p0 += out_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        float32x4_t _descale = vld1q_f32((const float*)descales + i + ii);

        float32x4_t _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(pC[0] * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                _c0 = vld1q_f32(pC);
                _c0 = vmulq_n_f32(_c0, beta);
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
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);
            int32x4_t _sum4 = vld1q_s32(pp + 16);
            int32x4_t _sum5 = vld1q_s32(pp + 20);
            int32x4_t _sum6 = vld1q_s32(pp + 24);
            int32x4_t _sum7 = vld1q_s32(pp + 28);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
            //      a4 b4 c4 d4
            //      a5 b5 c5 d5
            //      a6 b6 c6 d6
            //      a7 b7 c7 d7
#else
            // from
            //      a0 b1 c2 d3
            //      a4 b5 c6 d7
            //      c0 d1 a2 b3
            //      c4 d5 a6 b7
            //      a3 b2 c1 d0
            //      a7 b6 c5 d4
            //      c3 d2 a1 b0
            //      c7 d6 a5 b4

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);
            float32x4_t _f4 = vmulq_f32(vcvtq_f32_s32(_sum4), _descale);
            float32x4_t _f5 = vmulq_f32(vcvtq_f32_s32(_sum5), _descale);
            float32x4_t _f6 = vmulq_f32(vcvtq_f32_s32(_sum6), _descale);
            float32x4_t _f7 = vmulq_f32(vcvtq_f32_s32(_sum7), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                    _f4 = vaddq_f32(_f4, _c0);
                    _f5 = vaddq_f32(_f5, _c0);
                    _f6 = vaddq_f32(_f6, _c0);
                    _f7 = vaddq_f32(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    float32x4_t _c4;
                    float32x4_t _c5;
                    float32x4_t _c6;
                    float32x4_t _c7;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        _c4 = vld1q_f32(pC + 16);
                        _c5 = vld1q_f32(pC + 20);
                        _c6 = vld1q_f32(pC + 24);
                        _c7 = vld1q_f32(pC + 28);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + c_hstep);
                        _c3 = vld1q_f32(pC + c_hstep + 4);
                        _c4 = vld1q_f32(pC + c_hstep * 2);
                        _c5 = vld1q_f32(pC + c_hstep * 2 + 4);
                        _c6 = vld1q_f32(pC + c_hstep * 3);
                        _c7 = vld1q_f32(pC + c_hstep * 3 + 4);
                        transpose8x4_ps(_c0, _c1, _c2, _c3, _c4, _c5, _c6, _c7);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                        _f4 = vaddq_f32(_f4, _c4);
                        _f5 = vaddq_f32(_f5, _c5);
                        _f6 = vaddq_f32(_f6, _c6);
                        _f7 = vaddq_f32(_f7, _c7);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                        _f4 = vmlaq_f32(_f4, _c4, _beta);
                        _f5 = vmlaq_f32(_f5, _c5, _beta);
                        _f6 = vmlaq_f32(_f6, _c6, _beta);
                        _f7 = vmlaq_f32(_f7, _c7, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc0 = vld1q_f32(pC);
                    float32x4_t _cc1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _cc0 = vmulq_f32(_cc0, _beta);
                        _cc1 = vmulq_f32(_cc1, _beta);
                    }
                    _c0 = vdupq_laneq_f32(_cc0, 0);
                    float32x4_t _c1 = vdupq_laneq_f32(_cc0, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc0, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc0, 3);
                    float32x4_t _c4 = vdupq_laneq_f32(_cc1, 0);
                    float32x4_t _c5 = vdupq_laneq_f32(_cc1, 1);
                    float32x4_t _c6 = vdupq_laneq_f32(_cc1, 2);
                    float32x4_t _c7 = vdupq_laneq_f32(_cc1, 3);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    _f4 = vaddq_f32(_f4, _c4);
                    _f5 = vaddq_f32(_f5, _c5);
                    _f6 = vaddq_f32(_f6, _c6);
                    _f7 = vaddq_f32(_f7, _c7);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
                _f4 = vmulq_f32(_f4, _alpha);
                _f5 = vmulq_f32(_f5, _alpha);
                _f6 = vmulq_f32(_f6, _alpha);
                _f7 = vmulq_f32(_f7, _alpha);
            }

            if (out_elempack == 4)
            {
                float32x4x4_t _fa;
                float32x4x4_t _fb;
                _fa.val[0] = _f0;
                _fa.val[1] = _f1;
                _fa.val[2] = _f2;
                _fa.val[3] = _f3;
                _fb.val[0] = _f4;
                _fb.val[1] = _f5;
                _fb.val[2] = _f6;
                _fb.val[3] = _f7;
                vst4q_f32(p0, _fa);
                vst4q_f32(p0 + out_hstep * 4, _fb);
            }
            if (out_elempack == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);
                vst1q_f32(p0 + out_hstep * 4, _f4);
                vst1q_f32(p0 + out_hstep * 5, _f5);
                vst1q_f32(p0 + out_hstep * 6, _f6);
                vst1q_f32(p0 + out_hstep * 7, _f7);
            }

            pp += 32;
            p0 += out_hstep * 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
            //      a2 b2 c2 d2
            //      a3 b3 c3 d3
#else
            // from
            //      a0 b1 c2 d3
            //      c0 d1 a2 b3
            //      a3 b2 c1 d0
            //      c3 d2 a1 b0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        _c2 = vld1q_f32(pC + 8);
                        _c3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + c_hstep);
                        _c2 = vld1q_f32(pC + c_hstep * 2);
                        _c3 = vld1q_f32(pC + c_hstep * 3);
                        transpose4x4_ps(_c0, _c1, _c2, _c3);
                        pC += 4;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x4_t _cc = vld1q_f32(pC);
                    _cc = vmulq_n_f32(_cc, beta);
#if __aarch64__
                    _c0 = vdupq_laneq_f32(_cc, 0);
                    float32x4_t _c1 = vdupq_laneq_f32(_cc, 1);
                    float32x4_t _c2 = vdupq_laneq_f32(_cc, 2);
                    float32x4_t _c3 = vdupq_laneq_f32(_cc, 3);
#else
                    _c0 = vdupq_lane_f32(vget_low_f32(_cc), 0);
                    float32x4_t _c1 = vdupq_lane_f32(vget_low_f32(_cc), 1);
                    float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_cc), 0);
                    float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_cc), 1);
#endif
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c2);
                    _f3 = vaddq_f32(_f3, _c3);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (out_elempack == 4)
            {
                float32x4x4_t _f;
                _f.val[0] = _f0;
                _f.val[1] = _f1;
                _f.val[2] = _f2;
                _f.val[3] = _f3;
                vst4q_f32(p0, _f);
            }
            if (out_elempack == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + out_hstep, _f1);
                vst1q_f32(p0 + out_hstep * 2, _f2);
                vst1q_f32(p0 + out_hstep * 3, _f3);
            }

            pp += 16;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

#if __ARM_FEATURE_DOTPROD
            // from/to
            //      a0 b0 c0 d0
            //      a1 b1 c1 d1
#else
            // from
            //      a0 b1 c0 d1
            //      a1 b0 c1 d0

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
#endif // __ARM_FEATURE_DOTPROD

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c1;
                    if (c_elempack == 1)
                    {
                        float32x2_t _cc0 = vld1_f32(pC);
                        float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                        float32x2_t _cc2 = vld1_f32(pC + c_hstep * 2);
                        float32x2_t _cc3 = vld1_f32(pC + c_hstep * 3);
                        float32x4_t _cc01 = vcombine_f32(_cc0, _cc1);
                        float32x4_t _cc23 = vcombine_f32(_cc2, _cc3);
                        float32x4x2_t _cc = vuzpq_f32(_cc01, _cc23);
                        _c0 = _cc.val[0];
                        _c1 = _cc.val[1];
                        pC += 2;
                    }
                    else // if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        _c1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    _c = vmul_n_f32(_c, beta);
                    _c0 = vdupq_lane_f32(_c, 0);
                    float32x4_t _c1 = vdupq_lane_f32(_c, 1);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            vst1q_f32(p0, _f0);
            vst1q_f32(p0 + out_hstep, _f1);

            pp += 8;
            p0 += out_hstep * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    if (c_elempack == 1)
                    {
                        _c0 = vsetq_lane_f32(pC[0], _c0, 0);
                        _c0 = vsetq_lane_f32(pC[c_hstep], _c0, 1);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 2], _c0, 2);
                        _c0 = vsetq_lane_f32(pC[c_hstep * 3], _c0, 3);
                        pC += 1;
                    }
                    else // if (c_elempack == 4)
                    {
                        _c0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(pC[0] * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1q_f32(p0, _f0);
            pp += 4;
            p0 += out_hstep;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        const float descale0 = descales[i + ii];
        const float descale1 = descales[i + ii + 1];
#if __ARM_NEON
        float32x2_t _descale01 = vld1_f32((const float*)descales + i + ii);
#endif

        float c0;
        float c1;
#if __ARM_NEON
        float32x4_t _c0;
        float32x4_t _c1;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
                _c1 = vdupq_n_f32(c1);
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
#if __ARM_NEON
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

            float32x4_t _f0 = vmulq_lane_f32(vcvtq_f32_s32(_sum0), _descale01, 0);
            float32x4_t _f1 = vmulq_lane_f32(vcvtq_f32_s32(_sum1), _descale01, 0);
            float32x4_t _f2 = vmulq_lane_f32(vcvtq_f32_s32(_sum2), _descale01, 1);
            float32x4_t _f3 = vmulq_lane_f32(vcvtq_f32_s32(_sum3), _descale01, 1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c1);
                    _f3 = vaddq_f32(_f3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + c_hstep);
                    float32x4_t _c3 = vld1q_f32(pC + c_hstep + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + 4);
                    if (beta != 1.f)
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _c0 = vmulq_f32(_c0, _beta);
                        _c1 = vmulq_f32(_c1, _beta);
                    }
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f2);
                vst1q_f32(p0 + out_hstep * 4, _f1);
                vst1q_f32(p0 + out_hstep * 4 + 4, _f3);
            }
            if (out_elempack == 1)
            {
                float32x4x2_t _f02 = vzipq_f32(_f0, _f2);
                float32x4x2_t _f13 = vzipq_f32(_f1, _f3);
                vst1_f32(p0, vget_low_f32(_f02.val[0]));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f02.val[0]));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f02.val[1]));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f02.val[1]));
                vst1_f32(p0 + out_hstep * 4, vget_low_f32(_f13.val[0]));
                vst1_f32(p0 + out_hstep * 5, vget_high_f32(_f13.val[0]));
                vst1_f32(p0 + out_hstep * 6, vget_low_f32(_f13.val[1]));
                vst1_f32(p0 + out_hstep * 7, vget_high_f32(_f13.val[1]));
            }

            pp += 16;
            p0 += out_hstep * 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            // a0 a1 a2 a3
            // b0 b1 b2 b3

            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

            float32x4_t _f0 = vmulq_lane_f32(vcvtq_f32_s32(_sum0), _descale01, 0);
            float32x4_t _f1 = vmulq_lane_f32(vcvtq_f32_s32(_sum1), _descale01, 1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _c1 = vld1q_f32(pC + c_hstep);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vld1q_f32(pC);
                    _c0 = vmulq_n_f32(_c0, beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (out_elempack == 4)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
            }
            if (out_elempack == 1)
            {
                float32x4x2_t _f01 = vzipq_f32(_f0, _f1);
                vst1_f32(p0, vget_low_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep, vget_high_f32(_f01.val[0]));
                vst1_f32(p0 + out_hstep * 2, vget_low_f32(_f01.val[1]));
                vst1_f32(p0 + out_hstep * 3, vget_high_f32(_f01.val[1]));
            }

            pp += 8;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            // a0 a1 b0 b1
            int32x2x2_t _sum0 = vld2_s32(pp);

            float32x4_t _descale = vcombine_f32(_descale01, _descale01);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vcombine_s32(_sum0.val[0], _sum0.val[1])), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _cc = vzipq_f32(_c0, _c1).val[0];
                    _f0 = vaddq_f32(_f0, _cc);
                }
                if (broadcast_type_C == 3)
                {
                    // c_elempack == 1
                    float32x2_t _cc0 = vld1_f32(pC);
                    float32x2_t _cc1 = vld1_f32(pC + c_hstep);
                    float32x2x2_t _c01 = vzip_f32(_cc0, _cc1);
                    _c0 = vcombine_f32(_c01.val[0], _c01.val[1]);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _cc = vld1_f32(pC);
                    float32x2x2_t _c01 = vzip_f32(_cc, _cc);
                    _c0 = vcombine_f32(_c01.val[0], _c01.val[1]);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1_f32(p0, vget_low_f32(_f0));
            vst1_f32(p0 + out_hstep, vget_high_f32(_f0));

            pp += 4;
            p0 += out_hstep * 2;
        }
#endif // __ARM_NEON
        for (; jj < max_jj; jj += 1)
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

            p0[0] = f0;
            p0[1] = f1;

            pp += 2;
            p0 += out_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        const float descale = descales[i + ii];
#if __ARM_NEON
        float32x4_t _descale = vdupq_n_f32(descale);
#endif

        float c0;
#if __ARM_NEON
        float32x4_t _c0;
#endif
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
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
#if __ARM_NEON
        for (; jj + 15 < max_jj; jj += 16)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);
            int32x4_t _sum2 = vld1q_s32(pp + 8);
            int32x4_t _sum3 = vld1q_s32(pp + 12);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);
            float32x4_t _f2 = vmulq_f32(vcvtq_f32_s32(_sum2), _descale);
            float32x4_t _f3 = vmulq_f32(vcvtq_f32_s32(_sum3), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                    _f2 = vaddq_f32(_f2, _c0);
                    _f3 = vaddq_f32(_f3, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    float32x4_t _c2 = vld1q_f32(pC + 8);
                    float32x4_t _c3 = vld1q_f32(pC + 12);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                        _f2 = vaddq_f32(_f2, _c2);
                        _f3 = vaddq_f32(_f3, _c3);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                        _f2 = vmlaq_f32(_f2, _c2, _beta);
                        _f3 = vmlaq_f32(_f3, _c3, _beta);
                    }
                    pC += 16;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
                _f2 = vmulq_f32(_f2, _alpha);
                _f3 = vmulq_f32(_f3, _alpha);
            }

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
                vst1q_f32(p0 + 8, _f2);
                vst1q_f32(p0 + 12, _f3);
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + out_hstep * 4, _f1);
                    vst1q_f32(p0 + out_hstep * 8, _f2);
                    vst1q_f32(p0 + out_hstep * 12, _f3);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vgetq_lane_f32(_f0, 0);
                    p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                    p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                    p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                    p0[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                    p0[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                    p0[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                    p0[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                    p0[out_hstep * 8] = vgetq_lane_f32(_f2, 0);
                    p0[out_hstep * 9] = vgetq_lane_f32(_f2, 1);
                    p0[out_hstep * 10] = vgetq_lane_f32(_f2, 2);
                    p0[out_hstep * 11] = vgetq_lane_f32(_f2, 3);
                    p0[out_hstep * 12] = vgetq_lane_f32(_f3, 0);
                    p0[out_hstep * 13] = vgetq_lane_f32(_f3, 1);
                    p0[out_hstep * 14] = vgetq_lane_f32(_f3, 2);
                    p0[out_hstep * 15] = vgetq_lane_f32(_f3, 3);
                }
            }

            pp += 16;
            p0 += out_hstep * 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            int32x4_t _sum0 = vld1q_s32(pp);
            int32x4_t _sum1 = vld1q_s32(pp + 4);

            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(_sum0), _descale);
            float32x4_t _f1 = vmulq_f32(vcvtq_f32_s32(_sum1), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                    _f1 = vaddq_f32(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    float32x4_t _c1 = vld1q_f32(pC + 4);
                    if (beta == 1.f)
                    {
                        _f0 = vaddq_f32(_f0, _c0);
                        _f1 = vaddq_f32(_f1, _c1);
                    }
                    else
                    {
                        float32x4_t _beta = vdupq_n_f32(beta);
                        _f0 = vmlaq_f32(_f0, _c0, _beta);
                        _f1 = vmlaq_f32(_f1, _c1, _beta);
                    }
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                float32x4_t _alpha = vdupq_n_f32(alpha);
                _f0 = vmulq_f32(_f0, _alpha);
                _f1 = vmulq_f32(_f1, _alpha);
            }

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _f0);
                vst1q_f32(p0 + 4, _f1);
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(p0, _f0);
                    vst1q_f32(p0 + out_hstep * 4, _f1);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vgetq_lane_f32(_f0, 0);
                    p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                    p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                    p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                    p0[out_hstep * 4] = vgetq_lane_f32(_f1, 0);
                    p0[out_hstep * 5] = vgetq_lane_f32(_f1, 1);
                    p0[out_hstep * 6] = vgetq_lane_f32(_f1, 2);
                    p0[out_hstep * 7] = vgetq_lane_f32(_f1, 3);
                }
            }

            pp += 8;
            p0 += out_hstep * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _f0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(pp)), _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vaddq_f32(_f0, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // c_elempack == 1
                    _c0 = vld1q_f32(pC);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 4;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            if (out_hstep == 1)
            {
                vst1q_f32(p0, _f0);
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(p0, _f0);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vgetq_lane_f32(_f0, 0);
                    p0[out_hstep] = vgetq_lane_f32(_f0, 1);
                    p0[out_hstep * 2] = vgetq_lane_f32(_f0, 2);
                    p0[out_hstep * 3] = vgetq_lane_f32(_f0, 3);
                }
            }

            pp += 4;
            p0 += out_hstep * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _f0 = vmul_f32(vcvt_f32_s32(vld1_s32(pp)), vget_low_f32(_descale));

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = vadd_f32(_f0, vget_low_f32(_c0));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    // c_elempack == 1
                    float32x2_t _c = vld1_f32(pC);
                    _f0 = vmla_n_f32(_f0, _c, beta);
                    pC += 2;
                }
            }

            _f0 = vmul_n_f32(_f0, alpha);

            if (out_hstep == 1)
            {
                vst1_f32(p0, _f0);
            }
            else
            {
                p0[0] = vget_lane_f32(_f0, 0);
                p0[out_hstep] = vget_lane_f32(_f0, 1);
            }

            pp += 2;
            p0 += out_hstep * 2;
        }
#endif // __ARM_NEON
        for (; jj < max_jj; jj += 1)
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
                    // c_elempack == 1
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = f0;

            pp += 1;
            p0 += out_hstep;
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        gemm_transB_packed_tile_int8_i8mm(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        gemm_transB_packed_tile_int8_asimddp(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
        return;
    }
#endif

    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
#if !__ARM_FEATURE_MATMUL_INT8
                "cmp    %w7, #0                     \n"
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
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
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

                "cmp    %w7, #0                     \n"
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
                "cmp    %w7, #0                     \n"
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

                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
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
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
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
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
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
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
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
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
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
                "cmp    %w7, #0                     \n"
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
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
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
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
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
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
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
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
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
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
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
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %7, #0              \n"
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
                "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
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
                "and        r4, %6, #2          \n" // r4 = remain = max_kk & 2
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
                "and        r4, %6, #1          \n" // r4 = remain = max_kk & 1
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
                "vstm       %0!, {d16-d23}      \n"
                "vstm       %0!, {d24-d31}      \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);
            vst1q_s32(outptr + 8, _sum2);
            vst1q_s32(outptr + 12, _sum3);
            vst1q_s32(outptr + 16, _sum4);
            vst1q_s32(outptr + 20, _sum5);
            vst1q_s32(outptr + 24, _sum6);
            vst1q_s32(outptr + 28, _sum7);

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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);
            vst1q_s32(outptr + 8, _sum2);
            vst1q_s32(outptr + 12, _sum3);

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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cmp    %w7, #0                     \n"
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
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
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
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
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
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
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
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
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
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
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
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);
            vst1q_s32(outptr + 8, _sum2);
            vst1q_s32(outptr + 12, _sum3);
            vst1q_s32(outptr + 16, _sum4);
            vst1q_s32(outptr + 20, _sum5);
            vst1q_s32(outptr + 24, _sum6);
            vst1q_s32(outptr + 28, _sum7);

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
                "cmp    %w7, #0                     \n"
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
                "lsr    w4, %w6, #3                 \n" // w4 = max_kk >> 3
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
                "and    w4, %w6, #4                 \n" // w4 = remain = max_kk & 4
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
                "lsr    w4, %w6, #2                 \n" // w4 = max_kk >> 2
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
                "and    w4, %w6, #2                 \n" // w4 = remain = max_kk & 2
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
                "and    w4, %w6, #1                 \n" // w4 = remain = max_kk & 1
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
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %7, #0              \n"
                "beq        0f                  \n"

                "vldm       %0, {d16-d23}       \n"
                "b          1f                  \n"

                "0:                             \n"
                "veor       q8, q8              \n"
                "veor       q9, q9              \n"
                "veor       q10, q10            \n"
                "veor       q11, q11            \n"

                "1:                             \n"
                "lsr        r4, %6, #2          \n" // r4 = max_kk >> 2
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
                "and        r4, %6, #2          \n" // r4 = remain = max_kk & 2
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
                "and        r4, %6, #1          \n" // r4 = remain = max_kk & 1
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
                "vstm       %0!, {d16-d23}      \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB)      // %2
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "r"(max_kk), // %6
                "r"(k)       // %7
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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);
            vst1q_s32(outptr + 8, _sum2);
            vst1q_s32(outptr + 12, _sum3);

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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);

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

            vst1q_s32(outptr, _sum0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);
            vst1q_s32(outptr + 8, _sum2);
            vst1q_s32(outptr + 12, _sum3);

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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);

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

                // A0 A1 A2 A3
                // B0 B1 B2 B3

                // A0 A1 A0 A1 A2 A3 A2 A3
                // B0 B1 B2 B3 B0 B1 B2 B3

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

                // A0 A1 A0 A1
                // B0 B1 B0 B1

                // A0 A0 A1 A1

                pA += 2;
                pB += 2;
            }

            vst1q_s32(outptr, _sum);

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

            outptr[0] = sum00;
            outptr[1] = sum10;
            outptr[2] = sum01;
            outptr[3] = sum11;

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

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
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
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    // TODO
                    // __builtin_prefetch(pA + 16);
                    // __builtin_prefetch(pB + 128);
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);
                    int8x16_t _pB4 = vld1q_s8(pB + 64);
                    int8x16_t _pB5 = vld1q_s8(pB + 80);
                    int8x16_t _pB6 = vld1q_s8(pB + 96);
                    int8x16_t _pB7 = vld1q_s8(pB + 112);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 3));
                    int8x8_t _pA4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 0));
                    int8x8_t _pA5 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 1));
                    int8x8_t _pA6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 2));
                    int8x8_t _pA7 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA2, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA2, vget_high_s8(_pB2));
                    int16x8_t _s4 = vmull_s8(_pA4, vget_low_s8(_pB4));
                    int16x8_t _s5 = vmull_s8(_pA4, vget_high_s8(_pB4));
                    int16x8_t _s6 = vmull_s8(_pA6, vget_low_s8(_pB6));
                    int16x8_t _s7 = vmull_s8(_pA6, vget_high_s8(_pB6));
                    _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB3));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB3));
                    _s4 = vmlal_s8(_s4, _pA5, vget_low_s8(_pB5));
                    _s5 = vmlal_s8(_s5, _pA5, vget_high_s8(_pB5));
                    _s6 = vmlal_s8(_s6, _pA7, vget_low_s8(_pB7));
                    _s7 = vmlal_s8(_s7, _pA7, vget_high_s8(_pB7));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);
                    _sum4 = vpadalq_s16(_sum4, _s4);
                    _sum5 = vpadalq_s16(_sum5, _s5);
                    _sum6 = vpadalq_s16(_sum6, _s6);
                    _sum7 = vpadalq_s16(_sum7, _s7);

                    pA += 16;
                    pB += 128;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA0, vget_high_s8(_pB0));
                    int16x8_t _s2 = vmull_s8(_pA2, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA2, vget_high_s8(_pB2));
                    _s0 = vmlal_s8(_s0, _pA1, vget_low_s8(_pB1));
                    _s1 = vmlal_s8(_s1, _pA1, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA3, vget_low_s8(_pB3));
                    _s3 = vmlal_s8(_s3, _pA3, vget_high_s8(_pB3));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);

                    pA += 8;
                    pB += 64;
                }
                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);
                _sum0 = vaddq_s32(_sum0, _sum4);
                _sum1 = vaddq_s32(_sum1, _sum5);
                _sum0 = vaddq_s32(_sum0, _sum6);
                _sum1 = vaddq_s32(_sum1, _sum7);
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

            vst1q_s32(outptr, _sum0);
            vst1q_s32(outptr + 4, _sum1);

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
#else  // __ARM_FEATURE_DOTPROD
            {
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    // TODO
                    // __builtin_prefetch(pA + 16);
                    // __builtin_prefetch(pB + 64);
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);
                    int8x16_t _pB2 = vld1q_s8(pB + 32);
                    int8x16_t _pB3 = vld1q_s8(pB + 48);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_low_s8(_pA)), 3));
                    int8x8_t _pA4 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 0));
                    int8x8_t _pA5 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 1));
                    int8x8_t _pA6 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 2));
                    int8x8_t _pA7 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vget_high_s8(_pA)), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    int16x8_t _s2 = vmull_s8(_pA4, vget_low_s8(_pB2));
                    int16x8_t _s3 = vmull_s8(_pA6, vget_low_s8(_pB3));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _s2 = vmlal_s8(_s2, _pA5, vget_high_s8(_pB2));
                    _s3 = vmlal_s8(_s3, _pA7, vget_high_s8(_pB3));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);
                    _sum2 = vpadalq_s16(_sum2, _s2);
                    _sum3 = vpadalq_s16(_sum3, _s3);

                    pA += 16;
                    pB += 64;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    int8x8_t _pA0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 0));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 1));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 2));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(_pA), 3));
                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);

                    pA += 8;
                    pB += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum0 = vaddq_s32(_sum0, _sum3);
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

            vst1q_s32(outptr, _sum0);

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
                int32x4_t _sum1 = vdupq_n_s32(0);
                for (; kk + 15 < max_kk; kk += 16)
                {
                    int8x16_t _pA = vld1q_s8(pA);
                    int8x16_t _pB0 = vld1q_s8(pB);
                    int8x16_t _pB1 = vld1q_s8(pB + 16);

                    int16x8x2_t _pAA = vzipq_s16(vreinterpretq_s16_s8(_pA), vreinterpretq_s16_s8(_pA));

                    int8x8_t _pA0 = vreinterpret_s8_s16(vget_low_s16(_pAA.val[0]));
                    int8x8_t _pA1 = vreinterpret_s8_s16(vget_high_s16(_pAA.val[0]));
                    int8x8_t _pA2 = vreinterpret_s8_s16(vget_low_s16(_pAA.val[1]));
                    int8x8_t _pA3 = vreinterpret_s8_s16(vget_high_s16(_pAA.val[1]));

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB0));
                    int16x8_t _s1 = vmull_s8(_pA2, vget_low_s8(_pB1));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB0));
                    _s1 = vmlal_s8(_s1, _pA3, vget_high_s8(_pB1));
                    _sum0 = vpadalq_s16(_sum0, _s0);
                    _sum1 = vpadalq_s16(_sum1, _s1);

                    pA += 16;
                    pB += 32;
                }
                _sum0 = vaddq_s32(_sum0, _sum1);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    int8x8_t _pA = vld1_s8(pA);
                    int8x16_t _pB = vld1q_s8(pB);

                    int16x4x2_t _pAA = vzip_s16(vreinterpret_s16_s8(_pA), vreinterpret_s16_s8(_pA));

                    int8x8_t _pA0 = vreinterpret_s8_s16(_pAA.val[0]);
                    int8x8_t _pA1 = vreinterpret_s8_s16(_pAA.val[1]);

                    int16x8_t _s0 = vmull_s8(_pA0, vget_low_s8(_pB));
                    _s0 = vmlal_s8(_s0, _pA1, vget_high_s8(_pB));
                    _sum0 = vpadalq_s16(_sum0, _s0);

                    pA += 8;
                    pB += 16;
                }
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
#if __ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            for (; kk + 31 < max_kk; kk += 32)
            {
                int8x16_t _pA0 = vld1q_s8(pA);
                int8x16_t _pA1 = vld1q_s8(pA + 16);
                int8x16_t _pB0 = vld1q_s8(pB);
                int8x16_t _pB1 = vld1q_s8(pB + 16);

#if __ARM_FEATURE_DOTPROD
                _sum = vdotq_s32(_sum, _pA0, _pB0);
                _sum1 = vdotq_s32(_sum1, _pA1, _pB1);
#else  // __ARM_FEATURE_DOTPROD
                int16x8_t _s0 = vmull_s8(vget_low_s8(_pA0), vget_low_s8(_pB0));
                int16x8_t _s1 = vmull_s8(vget_low_s8(_pA1), vget_low_s8(_pB1));
                _s0 = vmlal_s8(_s0, vget_high_s8(_pA0), vget_high_s8(_pB0));
                _s1 = vmlal_s8(_s1, vget_high_s8(_pA1), vget_high_s8(_pB1));
                _sum = vpadalq_s16(_sum, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
#endif // __ARM_FEATURE_DOTPROD

                pA += 32;
                pB += 32;
            }
            _sum = vaddq_s32(_sum, _sum1);
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

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / (2 * sizeof(signed char) + sizeof(int)));

    TILE_M = std::max(8, tile_size / 8 * 8);
#if __aarch64__
    TILE_N = std::max(8, tile_size / 8 * 8);
#else
    TILE_N = std::max(4, tile_size / 4 * 4);
#endif
    TILE_K = std::max(8, tile_size / 8 * 8);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(signed char) / TILE_K);

            TILE_M = std::max(8, tile_size / 8 * 8);
#if __aarch64__
            TILE_N = std::max(8, tile_size / 8 * 8);
#else
            TILE_N = std::max(4, tile_size / 4 * 4);
#endif
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#endif
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
    }

    if (constant_TILE_N > 0)
    {
#if __aarch64__
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#endif
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
    }
}
