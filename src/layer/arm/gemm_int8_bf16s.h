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
void pack_A_tile_bf16_to_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_bf16_to_int8_i8mm(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_bf16_to_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_bf16_to_int8_i8mm(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void pack_A_tile_bf16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void transpose_pack_A_tile_bf16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales);
void pack_B_tile_bf16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void transpose_pack_B_tile_bf16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale);
void unpack_output_tile_int32_to_bf16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta);
void transpose_unpack_output_tile_int32_to_bf16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta);
#endif

static void compute_A_tile_bf16_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    const int K = A.w;

    // NCNN_LOGE("compute_A_tile_bf16_int8_scales %d %d", max_ii, elempack);

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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p0));
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep;

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            for (; kk + 15 < K; kk += 16)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p0));
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                p0 += 4;
            }
            float32x2_t _aa = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            absmax = std::max(absmax, std::max(vget_lane_f32(_aa, 0), vget_lane_f32(_aa, 1)));
#endif // __ARM_NEON
            for (; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(bfloat16_to_float32(p0[0])));
                p0++;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void pack_A_tile_bf16_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_A_tile_bf16_to_int8_i8mm(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_A_tile_bf16_to_int8_asimddp(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("pack_A_tile_bf16_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

        float32x4_t _scale0 = vld1q_f32((const float*)scales + i + ii);
        float32x4_t _scale1 = vld1q_f32((const float*)scales + i + ii + 4);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                uint16x8x4_t _p = vld4q_u16(p0);
                uint16x8x4_t _q = vld4q_u16(p0 + A_hstep * 4);

                float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[0])), _scale0, 0);
                float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[1])), _scale0, 1);
                float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[2])), _scale0, 2);
                float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[3])), _scale0, 3);
                float32x4_t _p4 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[0])), _scale0, 0);
                float32x4_t _p5 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[1])), _scale0, 1);
                float32x4_t _p6 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[2])), _scale0, 2);
                float32x4_t _p7 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[3])), _scale0, 3);
                float32x4_t _p8 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q.val[0])), _scale1, 0);
                float32x4_t _p9 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q.val[1])), _scale1, 1);
                float32x4_t _pa = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q.val[2])), _scale1, 2);
                float32x4_t _pb = vmulq_laneq_f32(bfloat2float(vget_low_u16(_q.val[3])), _scale1, 3);
                float32x4_t _pc = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q.val[0])), _scale1, 0);
                float32x4_t _pd = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q.val[1])), _scale1, 1);
                float32x4_t _pe = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q.val[2])), _scale1, 2);
                float32x4_t _pf = vmulq_laneq_f32(bfloat2float(vget_high_u16(_q.val[3])), _scale1, 3);

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + A_hstep * 4 + 8);
                uint16x8_t _v = vld1q_u16(p0 + A_hstep * 4 + 16);
                uint16x8_t _w = vld1q_u16(p0 + A_hstep * 4 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x4x4_t _p = vld4_u16(p0);
                uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);

                float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(_p.val[0]), _scale0, 0);
                float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(_p.val[1]), _scale0, 1);
                float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(_p.val[2]), _scale0, 2);
                float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(_p.val[3]), _scale0, 3);
                float32x4_t _p4 = vmulq_laneq_f32(bfloat2float(_q.val[0]), _scale1, 0);
                float32x4_t _p5 = vmulq_laneq_f32(bfloat2float(_q.val[1]), _scale1, 1);
                float32x4_t _p6 = vmulq_laneq_f32(bfloat2float(_q.val[2]), _scale1, 2);
                float32x4_t _p7 = vmulq_laneq_f32(bfloat2float(_q.val[3]), _scale1, 3);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 4 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep * 4);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p0n = bfloat2float(vget_high_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p1n = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 3);
                uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + A_hstep * 5);
                uint16x8_t _v = vld1q_u16(p0 + A_hstep * 6);
                uint16x8_t _w = vld1q_u16(p0 + A_hstep * 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                float32x4_t _p4 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                float32x4_t _p5 = bfloat2float(vld1_u16(p0 + A_hstep * 5));
                float32x4_t _p6 = bfloat2float(vld1_u16(p0 + A_hstep * 6));
                float32x4_t _p7 = bfloat2float(vld1_u16(p0 + A_hstep * 7));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[A_hstep * 4], _q, 0);
                _q = vsetq_lane_u16(p0[A_hstep * 4 + 1], _q, 1);
                _q = vsetq_lane_u16(p0[A_hstep * 5], _q, 2);
                _q = vsetq_lane_u16(p0[A_hstep * 5 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[A_hstep * 6], _q, 4);
                _q = vsetq_lane_u16(p0[A_hstep * 6 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[A_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[A_hstep * 7 + 1], _q, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p45 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p67 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 7], _p, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

        float32x4_t _scale = vld1q_f32((const float*)scales + i + ii);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                uint16x8x4_t _p = vld4q_u16(p0);

                float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[0])), _scale, 0);
                float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[1])), _scale, 1);
                float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[2])), _scale, 2);
                float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(vget_low_u16(_p.val[3])), _scale, 3);
                float32x4_t _p4 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[0])), _scale, 0);
                float32x4_t _p5 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[1])), _scale, 1);
                float32x4_t _p6 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[2])), _scale, 2);
                float32x4_t _p7 = vmulq_laneq_f32(bfloat2float(vget_high_u16(_p.val[3])), _scale, 3);

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x4x4_t _p = vld4_u16(p0);

                float32x4_t _p0 = vmulq_laneq_f32(bfloat2float(_p.val[0]), _scale, 0);
                float32x4_t _p1 = vmulq_laneq_f32(bfloat2float(_p.val[1]), _scale, 1);
                float32x4_t _p2 = vmulq_laneq_f32(bfloat2float(_p.val[2]), _scale, 2);
                float32x4_t _p3 = vmulq_laneq_f32(bfloat2float(_p.val[3]), _scale, 3);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 3);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));

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
                uint16x4_t _p = uint16x4_t();
                _p = vset_lane_u16(p0[0], _p, 0);
                _p = vset_lane_u16(p0[A_hstep], _p, 1);
                _p = vset_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vset_lane_u16(p0[A_hstep * 3], _p, 3);
                float32x4_t _p0 = bfloat2float(_p);

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
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale0);
                pp[2] = float2int8(bfloat16_to_float32(p0[A_hstep]) * scale1);
                pp[3] = float2int8(bfloat16_to_float32(p0[A_hstep + 1]) * scale1);
                pp += 4;
                p0 += 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[A_hstep]) * scale1);
                pp += 2;
                p0++;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

        const float scale = scales[i + ii];

        // if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            float32x4_t _scale = vdupq_n_f32(scale);
            for (; kk + 15 < max_kk; kk += 16)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_compute_A_tile_bf16_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;
    const int K = A.dims == 3 ? A.c : A.h;

    // NCNN_LOGE("transpose_compute_A_tile_bf16_int8_scales %d %d", max_ii, elempack);

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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * 4;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            for (int kk = 0; kk < K; kk++)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * 4;

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 8));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 12));
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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += A_hstep * 8;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p0));
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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii);

            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            float32x4_t _absmax2 = vdupq_n_f32(0.f);
            float32x4_t _absmax3 = vdupq_n_f32(0.f);
            int kk = 0;
            for (; kk + 3 < K; kk += 4)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                p0 += A_hstep * 2;
            }
            _absmax0 = vmaxq_f32(_absmax0, _absmax1);
            for (; kk < K; kk++)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p0));
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
#endif // __ARM_NEON
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii);

            float absmax = 0.f;
            for (int kk = 0; kk < K; kk++)
            {
                absmax = std::max(absmax, (float)fabsf(bfloat16_to_float32(p0[0])));
                p0 += A_hstep;
            }

            ps[0] = 127.f / absmax;
            pods[0] = absmax / v127_B_scale;
            ps++;
            pods++;
        }
    }
}

static void transpose_pack_A_tile_bf16_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_A_tile_bf16_to_int8_i8mm(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_A_tile_bf16_to_int8_asimddp(A, AT, i, max_ii, k, max_kk, scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    // NCNN_LOGE("transpose_pack_A_tile_bf16_to_int8 %d %d", max_ii, elempack);

    signed char* pp = AT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

        float32x4_t _scale0 = vld1q_f32((const float*)scales + i + ii);
        float32x4_t _scale1 = vld1q_f32((const float*)scales + i + ii + 4);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + A_hstep * 4 + 8);
                uint16x8_t _v = vld1q_u16(p0 + A_hstep * 4 + 16);
                uint16x8_t _w = vld1q_u16(p0 + A_hstep * 4 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 3);
                uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + A_hstep * 5);
                uint16x8_t _v = vld1q_u16(p0 + A_hstep * 6);
                uint16x8_t _w = vld1q_u16(p0 + A_hstep * 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 3);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep);

                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

        float32x4_t _scale = vld1q_f32((const float*)scales + i + ii);

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + A_hstep * 4);
                uint16x8_t _s = vld1q_u16(p0 + A_hstep * 4 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));
                float32x4_t _p4 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                float32x4_t _p5 = bfloat2float(vld1_u16(p0 + A_hstep * 5));
                float32x4_t _p6 = bfloat2float(vld1_u16(p0 + A_hstep * 6));
                float32x4_t _p7 = bfloat2float(vld1_u16(p0 + A_hstep * 7));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 3));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
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
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + A_hstep * 4);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
            int kk = 0;
#if __ARM_NEON
            float32x4_t _scale = vzipq_f32(_scale0, _scale1).val[0];
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[A_hstep * 4], _q, 0);
                _q = vsetq_lane_u16(p0[A_hstep * 4 + 1], _q, 1);
                _q = vsetq_lane_u16(p0[A_hstep * 5], _q, 2);
                _q = vsetq_lane_u16(p0[A_hstep * 5 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[A_hstep * 6], _q, 4);
                _q = vsetq_lane_u16(p0[A_hstep * 6 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[A_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[A_hstep * 7 + 1], _q, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p45 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p67 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 4 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 6 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[A_hstep], _q, 0);
                _q = vsetq_lane_u16(p0[A_hstep + 1], _q, 1);
                _q = vsetq_lane_u16(p0[A_hstep * 3], _q, 2);
                _q = vsetq_lane_u16(p0[A_hstep * 3 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[A_hstep * 5], _q, 4);
                _q = vsetq_lane_u16(p0[A_hstep * 5 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[A_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[A_hstep * 7 + 1], _q, 7);
                float32x4_t _p02 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p46 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p13 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p57 = bfloat2float(vget_high_u16(_q));

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
#if __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                float32x4x2_t _pp = vuzpq_f32(_p01, _p23);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep * 2 + 1], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep + 1], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 3 + 1], _p, 7);
                float32x4_t _p02 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p13 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[A_hstep + 0]) * scale0);
                pp[2] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                pp[3] = float2int8(bfloat16_to_float32(p0[A_hstep + 1]) * scale1);
                pp += 4;
                p0 += A_hstep * 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale0);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale1);
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * elempack;

        const float scale = scales[i + ii];

#if __ARM_NEON
        float32x4_t _scale = vdupq_n_f32(scale);
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + A_hstep * 8));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + A_hstep * 12));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + A_hstep * 4));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += A_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 7], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[A_hstep * 8], _q, 0);
                _q = vsetq_lane_u16(p0[A_hstep * 9], _q, 1);
                _q = vsetq_lane_u16(p0[A_hstep * 10], _q, 2);
                _q = vsetq_lane_u16(p0[A_hstep * 11], _q, 3);
                _q = vsetq_lane_u16(p0[A_hstep * 12], _q, 4);
                _q = vsetq_lane_u16(p0[A_hstep * 13], _q, 5);
                _q = vsetq_lane_u16(p0[A_hstep * 14], _q, 6);
                _q = vsetq_lane_u16(p0[A_hstep * 15], _q, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[A_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[A_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[A_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[A_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[A_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[A_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[A_hstep * 7], _p, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void compute_B_bf16_int8_scale(const Mat& B, float& scale)
{
    float absmax = 0.f;
#if __ARM_NEON
    float32x4_t _absmax = vdupq_n_f32(0.f);
#endif
    for (int i = 0; i < (B.dims == 3 ? B.c : B.h); i++)
    {
        const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;
        const unsigned short* ptr = (const unsigned short*)B + i * B_hstep * B.elempack;

        const int size = B.w * B.elempack;

        int j = 0;
#if __ARM_NEON
        for (; j + 7 < size; j += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _absmax = vmaxq_f32(_absmax, vabsq_f32(_p0));
            _absmax = vmaxq_f32(_absmax, vabsq_f32(_p1));
            ptr += 8;
        }
        for (; j + 3 < size; j += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
            ptr += 4;
        }
#endif
        for (; j < size; j++)
        {
            absmax = std::max(absmax, (float)fabsf(bfloat16_to_float32(ptr[0])));
            ptr++;
        }
    }
#if __ARM_NEON
    float32x2_t _aa = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
    absmax = std::max(absmax, std::max(vget_lane_f32(_aa, 0), vget_lane_f32(_aa, 1)));
#endif

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_bf16_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        pack_B_tile_bf16_to_int8_i8mm(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        pack_B_tile_bf16_to_int8_asimddp(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("pack_B_tile_bf16_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
#endif

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                uint16x8x4_t _p = vld4q_u16(p0);
                uint16x8x4_t _q = vld4q_u16(p0 + B_hstep * 4);

                float32x4_t _p0 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[0])), _scale);
                float32x4_t _p1 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[1])), _scale);
                float32x4_t _p2 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[2])), _scale);
                float32x4_t _p3 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[3])), _scale);
                float32x4_t _p4 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[0])), _scale);
                float32x4_t _p5 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[1])), _scale);
                float32x4_t _p6 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[2])), _scale);
                float32x4_t _p7 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[3])), _scale);
                float32x4_t _p8 = vmulq_f32(bfloat2float(vget_low_u16(_q.val[0])), _scale);
                float32x4_t _p9 = vmulq_f32(bfloat2float(vget_low_u16(_q.val[1])), _scale);
                float32x4_t _pa = vmulq_f32(bfloat2float(vget_low_u16(_q.val[2])), _scale);
                float32x4_t _pb = vmulq_f32(bfloat2float(vget_low_u16(_q.val[3])), _scale);
                float32x4_t _pc = vmulq_f32(bfloat2float(vget_high_u16(_q.val[0])), _scale);
                float32x4_t _pd = vmulq_f32(bfloat2float(vget_high_u16(_q.val[1])), _scale);
                float32x4_t _pe = vmulq_f32(bfloat2float(vget_high_u16(_q.val[2])), _scale);
                float32x4_t _pf = vmulq_f32(bfloat2float(vget_high_u16(_q.val[3])), _scale);

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                uint16x8_t _t = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + B_hstep * 4 + 8);
                uint16x8_t _v = vld1q_u16(p0 + B_hstep * 4 + 16);
                uint16x8_t _w = vld1q_u16(p0 + B_hstep * 4 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x4x4_t _p = vld4_u16(p0);
                uint16x4x4_t _q = vld4_u16(p0 + B_hstep * 4);

                float32x4_t _p0 = vmulq_f32(bfloat2float(_p.val[0]), _scale);
                float32x4_t _p1 = vmulq_f32(bfloat2float(_p.val[1]), _scale);
                float32x4_t _p2 = vmulq_f32(bfloat2float(_p.val[2]), _scale);
                float32x4_t _p3 = vmulq_f32(bfloat2float(_p.val[3]), _scale);
                float32x4_t _p4 = vmulq_f32(bfloat2float(_q.val[0]), _scale);
                float32x4_t _p5 = vmulq_f32(bfloat2float(_q.val[1]), _scale);
                float32x4_t _p6 = vmulq_f32(bfloat2float(_q.val[2]), _scale);
                float32x4_t _p7 = vmulq_f32(bfloat2float(_q.val[3]), _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);
                int8x8_t _r2 = float2int8(_p4, _p5);
                int8x8_t _r3 = float2int8(_p6, _p7);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 4 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep * 4);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep * 4));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 3);
                uint16x8_t _t = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + B_hstep * 5);
                uint16x8_t _v = vld1q_u16(p0 + B_hstep * 6);
                uint16x8_t _w = vld1q_u16(p0 + B_hstep * 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 3));
                float32x4_t _p4 = bfloat2float(vld1_u16(p0 + B_hstep * 4));
                float32x4_t _p5 = bfloat2float(vld1_u16(p0 + B_hstep * 5));
                float32x4_t _p6 = bfloat2float(vld1_u16(p0 + B_hstep * 6));
                float32x4_t _p7 = bfloat2float(vld1_u16(p0 + B_hstep * 7));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 3 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[B_hstep * 4], _q, 0);
                _q = vsetq_lane_u16(p0[B_hstep * 4 + 1], _q, 1);
                _q = vsetq_lane_u16(p0[B_hstep * 5], _q, 2);
                _q = vsetq_lane_u16(p0[B_hstep * 5 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[B_hstep * 6], _q, 4);
                _q = vsetq_lane_u16(p0[B_hstep * 6 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[B_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[B_hstep * 7 + 1], _q, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p45 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p67 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 7], _p, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
#if __ARM_FEATURE_DOTPROD
                uint16x8x4_t _p = vld4q_u16(p0);

                float32x4_t _p0 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[0])), _scale);
                float32x4_t _p1 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[1])), _scale);
                float32x4_t _p2 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[2])), _scale);
                float32x4_t _p3 = vmulq_f32(bfloat2float(vget_low_u16(_p.val[3])), _scale);
                float32x4_t _p4 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[0])), _scale);
                float32x4_t _p5 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[1])), _scale);
                float32x4_t _p6 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[2])), _scale);
                float32x4_t _p7 = vmulq_f32(bfloat2float(vget_high_u16(_p.val[3])), _scale);

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x4x4_t _p = vld4_u16(p0);

                float32x4_t _p0 = vmulq_f32(bfloat2float(_p.val[0]), _scale);
                float32x4_t _p1 = vmulq_f32(bfloat2float(_p.val[1]), _scale);
                float32x4_t _p2 = vmulq_f32(bfloat2float(_p.val[2]), _scale);
                float32x4_t _p3 = vmulq_f32(bfloat2float(_p.val[3]), _scale);

                int8x8_t _r0 = float2int8(_p0, _p1);
                int8x8_t _r1 = float2int8(_p2, _p3);

                vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 3);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 3));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 3 + 1], _p, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                int8x8_t _r0 = float2int8(_p01, _p23);

                vst1_s8(pp, _r0);

                pp += 8;
                p0 += 2;
            }
            for (; kk < max_kk; kk++)
            {
                uint16x4_t _p = uint16x4_t();
                _p = vset_lane_u16(p0[0], _p, 0);
                _p = vset_lane_u16(p0[B_hstep], _p, 1);
                _p = vset_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vset_lane_u16(p0[B_hstep * 3], _p, 3);
                float32x4_t _p0 = bfloat2float(_p);

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
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[B_hstep]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[B_hstep + 1]) * scale);
                pp += 4;
                p0 += 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[B_hstep]) * scale);
                pp += 2;
                p0++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

        // if (elempack == 1)
        {
            int kk = 0;
#if __ARM_NEON
            for (; kk + 15 < max_kk; kk += 16)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_bf16_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_pack_B_tile_bf16_to_int8_i8mm(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_pack_B_tile_bf16_to_int8_asimddp(B, BT, j, max_jj, k, max_kk, scale);
        return;
    }
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    // NCNN_LOGE("transpose_pack_B_tile_bf16_to_int8 %d %d", max_jj, elempack);

    signed char* pp = BT;

#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
#endif

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                uint16x8_t _t = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + B_hstep * 4 + 8);
                uint16x8_t _v = vld1q_u16(p0 + B_hstep * 4 + 16);
                uint16x8_t _w = vld1q_u16(p0 + B_hstep * 4 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + 16);
                uint16x8_t _s = vld1q_u16(p0 + 24);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 3);
                uint16x8_t _t = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _u = vld1q_u16(p0 + B_hstep * 5);
                uint16x8_t _v = vld1q_u16(p0 + B_hstep * 6);
                uint16x8_t _w = vld1q_u16(p0 + B_hstep * 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));
                float32x4_t _p8 = bfloat2float(vget_low_u16(_t));
                float32x4_t _p9 = bfloat2float(vget_high_u16(_t));
                float32x4_t _pa = bfloat2float(vget_low_u16(_u));
                float32x4_t _pb = bfloat2float(vget_high_u16(_u));
                float32x4_t _pc = bfloat2float(vget_low_u16(_v));
                float32x4_t _pd = bfloat2float(vget_high_u16(_v));
                float32x4_t _pe = bfloat2float(vget_low_u16(_w));
                float32x4_t _pf = bfloat2float(vget_high_u16(_w));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 2);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 3);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                uint16x8_t _r = vld1q_u16(p0 + B_hstep * 4);
                uint16x8_t _s = vld1q_u16(p0 + B_hstep * 4 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
                float32x4_t _p4 = bfloat2float(vget_low_u16(_r));
                float32x4_t _p5 = bfloat2float(vget_high_u16(_r));
                float32x4_t _p6 = bfloat2float(vget_low_u16(_s));
                float32x4_t _p7 = bfloat2float(vget_high_u16(_s));

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
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + 8);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 3));
                float32x4_t _p4 = bfloat2float(vld1_u16(p0 + B_hstep * 4));
                float32x4_t _p5 = bfloat2float(vld1_u16(p0 + B_hstep * 5));
                float32x4_t _p6 = bfloat2float(vld1_u16(p0 + B_hstep * 6));
                float32x4_t _p7 = bfloat2float(vld1_u16(p0 + B_hstep * 7));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 2));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 3));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __ARM_NEON
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                uint16x8_t _p = vld1q_u16(p0);
                uint16x8_t _q = vld1q_u16(p0 + B_hstep * 4);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = vld1q_u16(p0);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
#if __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 3 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[B_hstep * 4], _q, 0);
                _q = vsetq_lane_u16(p0[B_hstep * 4 + 1], _q, 1);
                _q = vsetq_lane_u16(p0[B_hstep * 5], _q, 2);
                _q = vsetq_lane_u16(p0[B_hstep * 5 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[B_hstep * 6], _q, 4);
                _q = vsetq_lane_u16(p0[B_hstep * 6 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[B_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[B_hstep * 7 + 1], _q, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p45 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p67 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 4 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 6 + 1], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[B_hstep], _q, 0);
                _q = vsetq_lane_u16(p0[B_hstep + 1], _q, 1);
                _q = vsetq_lane_u16(p0[B_hstep * 3], _q, 2);
                _q = vsetq_lane_u16(p0[B_hstep * 3 + 1], _q, 3);
                _q = vsetq_lane_u16(p0[B_hstep * 5], _q, 4);
                _q = vsetq_lane_u16(p0[B_hstep * 5 + 1], _q, 5);
                _q = vsetq_lane_u16(p0[B_hstep * 7], _q, 6);
                _q = vsetq_lane_u16(p0[B_hstep * 7 + 1], _q, 7);
                float32x4_t _p02 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p46 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p13 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p57 = bfloat2float(vget_high_u16(_q));

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
#if __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 3 + 1], _p, 7);
                float32x4_t _p01 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p23 = bfloat2float(vget_high_u16(_p));

                _p01 = vmulq_f32(_p01, _scale);
                _p23 = vmulq_f32(_p23, _scale);

                float32x4x2_t _pp = vuzpq_f32(_p01, _p23);
                int8x8_t _r01 = float2int8(_pp.val[0], _pp.val[1]);
#else  // __ARM_FEATURE_DOTPROD
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[1], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep * 2 + 1], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep + 1], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 3 + 1], _p, 7);
                float32x4_t _p02 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p13 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[B_hstep + 0]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[B_hstep + 1]) * scale);
                pp += 4;
                p0 += B_hstep * 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * elempack;

#if __ARM_NEON
        if (elempack == 4)
        {
            int kk = 0;
            for (; kk + 15 < max_kk; kk += 16)
            {
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep * 4));
                float32x4_t _p2 = bfloat2float(vld1_u16(p0 + B_hstep * 8));
                float32x4_t _p3 = bfloat2float(vld1_u16(p0 + B_hstep * 12));

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
                float32x4_t _p0 = bfloat2float(vld1_u16(p0));
                float32x4_t _p1 = bfloat2float(vld1_u16(p0 + B_hstep * 4));

                _p0 = vmulq_f32(_p0, _scale);
                _p1 = vmulq_f32(_p1, _scale);

                int8x8_t _r01 = float2int8(_p0, _p1);

                vst1_s8(pp, _r01);

                pp += 8;
                p0 += B_hstep * 8;
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp[1] = float2int8(bfloat16_to_float32(p0[1]) * scale);
                pp[2] = float2int8(bfloat16_to_float32(p0[2]) * scale);
                pp[3] = float2int8(bfloat16_to_float32(p0[3]) * scale);
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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 7], _p, 7);
                uint16x8_t _q = uint16x8_t();
                _q = vsetq_lane_u16(p0[B_hstep * 8], _q, 0);
                _q = vsetq_lane_u16(p0[B_hstep * 9], _q, 1);
                _q = vsetq_lane_u16(p0[B_hstep * 10], _q, 2);
                _q = vsetq_lane_u16(p0[B_hstep * 11], _q, 3);
                _q = vsetq_lane_u16(p0[B_hstep * 12], _q, 4);
                _q = vsetq_lane_u16(p0[B_hstep * 13], _q, 5);
                _q = vsetq_lane_u16(p0[B_hstep * 14], _q, 6);
                _q = vsetq_lane_u16(p0[B_hstep * 15], _q, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
                float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
                float32x4_t _p3 = bfloat2float(vget_high_u16(_q));

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
                uint16x8_t _p = uint16x8_t();
                _p = vsetq_lane_u16(p0[0], _p, 0);
                _p = vsetq_lane_u16(p0[B_hstep], _p, 1);
                _p = vsetq_lane_u16(p0[B_hstep * 2], _p, 2);
                _p = vsetq_lane_u16(p0[B_hstep * 3], _p, 3);
                _p = vsetq_lane_u16(p0[B_hstep * 4], _p, 4);
                _p = vsetq_lane_u16(p0[B_hstep * 5], _p, 5);
                _p = vsetq_lane_u16(p0[B_hstep * 6], _p, 6);
                _p = vsetq_lane_u16(p0[B_hstep * 7], _p, 7);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p));

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
                pp[0] = float2int8(bfloat16_to_float32(p0[0]) * scale);
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void unpack_output_tile_int32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        unpack_output_tile_int32_to_bf16_asimddp(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const unsigned short* pC = C;

    // NCNN_LOGE("unpack_output_tile_int32_to_bf16  %d %d %d %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack);

    const int* pp = topT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _descale0 = vld1q_f32((const float*)descales + i + ii);
        float32x4_t _descale1 = vld1q_f32((const float*)descales + i + ii + 4);

        float32x4_t _c0;
        float32x4_t _c1;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                uint16x8_t _c = vld1q_u16(pC);
                _c0 = bfloat2float(vget_low_u16(_c));
                _c1 = bfloat2float(vget_high_u16(_c));
                _c0 = vmulq_n_f32(_c0, beta);
                _c1 = vmulq_n_f32(_c1, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        uint16x8_t _c45 = vld1q_u16(pC + 16);
                        uint16x8_t _c67 = vld1q_u16(pC + 24);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 4 + 8);
                        _c45 = vld1q_u16(pC + c_hstep * 4 + 16);
                        _c67 = vld1q_u16(pC + c_hstep * 4 + 24);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        _c4 = bfloat2float(vget_low_u16(_c45));
                        _c5 = bfloat2float(vget_high_u16(_c45));
                        _c6 = bfloat2float(vget_low_u16(_c67));
                        _c7 = bfloat2float(vget_high_u16(_c67));
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                        uint16x8_t _c45 = vld1q_u16(pC + c_hstep * 2);
                        uint16x8_t _c67 = vld1q_u16(pC + c_hstep * 3);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 5);
                        _c45 = vld1q_u16(pC + c_hstep * 6);
                        _c67 = vld1q_u16(pC + c_hstep * 7);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        _c4 = bfloat2float(vget_low_u16(_c45));
                        _c5 = bfloat2float(vget_high_u16(_c45));
                        _c6 = bfloat2float(vget_low_u16(_c67));
                        _c7 = bfloat2float(vget_high_u16(_c67));
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
                    uint16x8_t _cc = vld1q_u16(pC);
                    float32x4_t _cc0 = bfloat2float(vget_low_u16(_cc));
                    float32x4_t _cc1 = bfloat2float(vget_high_u16(_cc));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);
            uint16x4_t _bf4 = float2bfloat(_f4);
            uint16x4_t _bf5 = float2bfloat(_f5);
            uint16x4_t _bf6 = float2bfloat(_f6);
            uint16x4_t _bf7 = float2bfloat(_f7);
            uint16x4_t _bf8 = float2bfloat(_f8);
            uint16x4_t _bf9 = float2bfloat(_f9);
            uint16x4_t _bfa = float2bfloat(_fa);
            uint16x4_t _bfb = float2bfloat(_fb);
            uint16x4_t _bfc = float2bfloat(_fc);
            uint16x4_t _bfd = float2bfloat(_fd);
            uint16x4_t _bfe = float2bfloat(_fe);
            uint16x4_t _bff = float2bfloat(_ff);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                vst1q_u16(p0 + 16, vcombine_u16(_bf4, _bf5));
                vst1q_u16(p0 + 24, vcombine_u16(_bf6, _bf7));
                vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf8, _bf9));
                vst1q_u16(p0 + out_hstep * 4 + 8, vcombine_u16(_bfa, _bfb));
                vst1q_u16(p0 + out_hstep * 4 + 16, vcombine_u16(_bfc, _bfd));
                vst1q_u16(p0 + out_hstep * 4 + 24, vcombine_u16(_bfe, _bff));
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                transpose4x4_u16(_bf0, _bf1, _bf2, _bf3);
                transpose4x4_u16(_bf4, _bf5, _bf6, _bf7);
                transpose4x4_u16(_bf8, _bf9, _bfa, _bfb);
                transpose4x4_u16(_bfc, _bfd, _bfe, _bff);
                vst1q_u16(p0, vcombine_u16(_bf0, _bf4));
                vst1q_u16(p0 + out_hstep, vcombine_u16(_bf1, _bf5));
                vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf2, _bf6));
                vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf3, _bf7));
                vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf8, _bfc));
                vst1q_u16(p0 + out_hstep * 5, vcombine_u16(_bf9, _bfd));
                vst1q_u16(p0 + out_hstep * 6, vcombine_u16(_bfa, _bfe));
                vst1q_u16(p0 + out_hstep * 7, vcombine_u16(_bfb, _bff));
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
                    if (c_elempack == 4)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 4 + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
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
                        uint16x4_t _cc0 = vld1_u16(pC);
                        uint16x4_t _cc1 = vld1_u16(pC + c_hstep);
                        uint16x4_t _cc2 = vld1_u16(pC + c_hstep * 2);
                        uint16x4_t _cc3 = vld1_u16(pC + c_hstep * 3);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        float32x4_t _c2 = bfloat2float(_cc2);
                        float32x4_t _c3 = bfloat2float(_cc3);
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
                        _cc0 = vld1_u16(pC + c_hstep * 4);
                        _cc1 = vld1_u16(pC + c_hstep * 5);
                        _cc2 = vld1_u16(pC + c_hstep * 6);
                        _cc3 = vld1_u16(pC + c_hstep * 7);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        _c2 = bfloat2float(_cc2);
                        _c3 = bfloat2float(_cc3);
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
                    float32x4_t _c = bfloat2float(vld1_u16(pC));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);
            uint16x4_t _bf4 = float2bfloat(_f4);
            uint16x4_t _bf5 = float2bfloat(_f5);
            uint16x4_t _bf6 = float2bfloat(_f6);
            uint16x4_t _bf7 = float2bfloat(_f7);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf4, _bf5));
                vst1q_u16(p0 + out_hstep * 4 + 8, vcombine_u16(_bf6, _bf7));
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_u16(_bf0, _bf1, _bf2, _bf3);
                transpose4x4_u16(_bf4, _bf5, _bf6, _bf7);
                vst1_u16(p0, _bf0);
                vst1_u16(p0 + out_hstep, _bf1);
                vst1_u16(p0 + out_hstep * 2, _bf2);
                vst1_u16(p0 + out_hstep * 3, _bf3);
                vst1_u16(p0 + out_hstep * 4, _bf4);
                vst1_u16(p0 + out_hstep * 5, _bf5);
                vst1_u16(p0 + out_hstep * 6, _bf6);
                vst1_u16(p0 + out_hstep * 7, _bf7);
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
                    uint16x8_t _c01;
                    uint16x8_t _c23;
                    if (c_elempack == 4)
                    {
                        _c01 = vld1q_u16(pC);
                        _c23 = vld1q_u16(pC + c_hstep * 4);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        _c01 = uint16x8_t();
                        _c01 = vsetq_lane_u16(pC[0], _c01, 0);
                        _c01 = vsetq_lane_u16(pC[c_hstep], _c01, 1);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 2], _c01, 2);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 3], _c01, 3);
                        _c01 = vsetq_lane_u16(pC[1], _c01, 4);
                        _c01 = vsetq_lane_u16(pC[c_hstep + 1], _c01, 5);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 2 + 1], _c01, 6);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 3 + 1], _c01, 7);
                        _c23 = uint16x8_t();
                        _c23 = vsetq_lane_u16(pC[c_hstep * 4], _c23, 0);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 5], _c23, 1);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 6], _c23, 2);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 7], _c23, 3);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 4 + 1], _c23, 4);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 5 + 1], _c23, 5);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 6 + 1], _c23, 6);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 7 + 1], _c23, 7);
                        pC += 2;
                    }
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]) * beta);
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf2, _bf3));
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[1] = vget_lane_u16(_bf1, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep + 1] = vget_lane_u16(_bf1, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 2 + 1] = vget_lane_u16(_bf1, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0[out_hstep * 3 + 1] = vget_lane_u16(_bf1, 3);
                p0[out_hstep * 4] = vget_lane_u16(_bf2, 0);
                p0[out_hstep * 4 + 1] = vget_lane_u16(_bf3, 0);
                p0[out_hstep * 5] = vget_lane_u16(_bf2, 1);
                p0[out_hstep * 5 + 1] = vget_lane_u16(_bf3, 1);
                p0[out_hstep * 6] = vget_lane_u16(_bf2, 2);
                p0[out_hstep * 6 + 1] = vget_lane_u16(_bf3, 2);
                p0[out_hstep * 7] = vget_lane_u16(_bf2, 3);
                p0[out_hstep * 7 + 1] = vget_lane_u16(_bf3, 3);
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
                        _c0 = bfloat2float(vld1_u16(pC));
                        _c1 = bfloat2float(vld1_u16(pC + c_hstep * 4));
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        uint16x8_t _c01 = uint16x8_t();
                        _c01 = vsetq_lane_u16(pC[0], _c01, 0);
                        _c01 = vsetq_lane_u16(pC[c_hstep], _c01, 1);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 2], _c01, 2);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 3], _c01, 3);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 4], _c01, 4);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 5], _c01, 5);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 6], _c01, 6);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 7], _c01, 7);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);

            if (out_elempack == 4)
            {
                vst1_u16(p0, _bf0);
                vst1_u16(p0 + out_hstep * 4, _bf1);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
                p0++;
            }

            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        float32x4_t _descale = vld1q_f32((const float*)descales + i + ii);

        float32x4_t _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                _c0 = bfloat2float(vld1_u16(pC));
                _c0 = vmulq_n_f32(_c0, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01;
                    uint16x8_t _c23;
                    uint16x8_t _c45;
                    uint16x8_t _c67;
                    if (c_elempack == 4)
                    {
                        _c01 = vld1q_u16(pC);
                        _c23 = vld1q_u16(pC + 8);
                        _c45 = vld1q_u16(pC + 16);
                        _c67 = vld1q_u16(pC + 24);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c01 = vld1q_u16(pC);
                        _c23 = vld1q_u16(pC + c_hstep);
                        _c45 = vld1q_u16(pC + c_hstep * 2);
                        _c67 = vld1q_u16(pC + c_hstep * 3);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        pC += 8;
                    }
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                    float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                    float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                    float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                    float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                    uint16x8_t _c = vld1q_u16(pC);
                    float32x4_t _cc0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _cc1 = bfloat2float(vget_high_u16(_c));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);
            uint16x4_t _bf4 = float2bfloat(_f4);
            uint16x4_t _bf5 = float2bfloat(_f5);
            uint16x4_t _bf6 = float2bfloat(_f6);
            uint16x4_t _bf7 = float2bfloat(_f7);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                vst1q_u16(p0 + 16, vcombine_u16(_bf4, _bf5));
                vst1q_u16(p0 + 24, vcombine_u16(_bf6, _bf7));
                p0 += 32;
            }
            if (out_elempack == 1)
            {
                transpose4x4_u16(_bf0, _bf1, _bf2, _bf3);
                transpose4x4_u16(_bf4, _bf5, _bf6, _bf7);
                vst1q_u16(p0, vcombine_u16(_bf0, _bf4));
                vst1q_u16(p0 + out_hstep, vcombine_u16(_bf1, _bf5));
                vst1q_u16(p0 + out_hstep * 2, vcombine_u16(_bf2, _bf6));
                vst1q_u16(p0 + out_hstep * 3, vcombine_u16(_bf3, _bf7));
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        uint16x4_t _cc0 = vld1_u16(pC);
                        uint16x4_t _cc1 = vld1_u16(pC + c_hstep * 1);
                        uint16x4_t _cc2 = vld1_u16(pC + c_hstep * 2);
                        uint16x4_t _cc3 = vld1_u16(pC + c_hstep * 3);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        _c2 = bfloat2float(_cc2);
                        _c3 = bfloat2float(_cc3);
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
                    float32x4_t _c = bfloat2float(vld1_u16(pC));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
                p0 += 16;
            }
            if (out_elempack == 1)
            {
                transpose4x4_u16(_bf0, _bf1, _bf2, _bf3);
                vst1_u16(p0, _bf0);
                vst1_u16(p0 + out_hstep, _bf1);
                vst1_u16(p0 + out_hstep * 2, _bf2);
                vst1_u16(p0 + out_hstep * 3, _bf3);
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
                    uint16x8_t _c;
                    if (c_elempack == 4)
                    {
                        _c = vld1q_u16(pC);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        _c = uint16x8_t();
                        _c = vsetq_lane_u16(pC[0], _c, 0);
                        _c = vsetq_lane_u16(pC[c_hstep], _c, 1);
                        _c = vsetq_lane_u16(pC[c_hstep * 2], _c, 2);
                        _c = vsetq_lane_u16(pC[c_hstep * 3], _c, 3);
                        _c = vsetq_lane_u16(pC[1], _c, 4);
                        _c = vsetq_lane_u16(pC[c_hstep + 1], _c, 5);
                        _c = vsetq_lane_u16(pC[c_hstep * 2 + 1], _c, 6);
                        _c = vsetq_lane_u16(pC[c_hstep * 3 + 1], _c, 7);
                        pC += 2;
                    }
                    _c0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]) * beta);
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                p0 += 8;
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[1] = vget_lane_u16(_bf1, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep + 1] = vget_lane_u16(_bf1, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 2 + 1] = vget_lane_u16(_bf1, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0[out_hstep * 3 + 1] = vget_lane_u16(_bf1, 3);
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
                    uint16x4_t _c;
                    if (c_elempack == 4)
                    {
                        _c = vld1_u16(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c = uint16x4_t();
                        _c = vset_lane_u16(pC[0], _c, 0);
                        _c = vset_lane_u16(pC[c_hstep], _c, 1);
                        _c = vset_lane_u16(pC[c_hstep * 2], _c, 2);
                        _c = vset_lane_u16(pC[c_hstep * 3], _c, 3);
                        pC += 1;
                    }
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            uint16x4_t _bf0 = float2bfloat(_f0);

            if (out_elempack == 4)
            {
                vst1_u16(p0, _bf0);
                p0 += 4;
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0++;
            }

            pp += 4;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        // out_elempack == 1
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

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
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                c0 = bfloat16_to_float32(pC[0]) * beta;
                c1 = bfloat16_to_float32(pC[1]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
                _c1 = vdupq_n_f32(c1);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01 = vld1q_u16(pC);
                    uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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
                    uint16x8_t _c = vld1q_u16(pC);
                    _c0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
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

            vst1q_u16(p0, vcombine_u16(float2bfloat(_f0), float2bfloat(_f1)));
            vst1q_u16(p0 + out_hstep, vcombine_u16(float2bfloat(_f2), float2bfloat(_f3)));

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
                    _c0 = bfloat2float(vld1_u16(pC));
                    float32x4_t _c1 = bfloat2float(vld1_u16(pC + c_hstep));
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
                    _c0 = bfloat2float(vld1_u16(pC));
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

            vst1_u16(p0, float2bfloat(_f0));
            vst1_u16(p0 + out_hstep, float2bfloat(_f1));

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
                    uint16x4_t _c = uint16x4_t();
                    _c = vset_lane_u16(pC[0], _c, 0);
                    _c = vset_lane_u16(pC[1], _c, 1);
                    _c = vset_lane_u16(pC[c_hstep], _c, 2);
                    _c = vset_lane_u16(pC[c_hstep + 1], _c, 3);
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    uint16x4_t _c = uint16x4_t();
                    _c = vset_lane_u16(pC[0], _c, 0);
                    _c = vset_lane_u16(pC[1], _c, 1);
                    _c = vset_lane_u16(pC[0], _c, 2);
                    _c = vset_lane_u16(pC[1], _c, 3);
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            uint16x4_t _bf0 = float2bfloat(_f0);

            p0[0] = vget_lane_u16(_bf0, 0);
            p0[1] = vget_lane_u16(_bf0, 1);
            p0[out_hstep] = vget_lane_u16(_bf0, 2);
            p0[out_hstep + 1] = vget_lane_u16(_bf0, 3);

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
                    f0 += bfloat16_to_float32(pC[0]) * beta;
                    f1 += bfloat16_to_float32(pC[c_hstep]) * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += bfloat16_to_float32(pC[0]) * beta;
                    f1 += bfloat16_to_float32(pC[0]) * beta;
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
            }

            p0[0] = float32_to_bfloat16(f0);
            p0[out_hstep] = float32_to_bfloat16(f1);

            pp += 2;
            p0++;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // out_elempack == 1
        unsigned short* p0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

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
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01 = vld1q_u16(pC);
                    uint16x8_t _c23 = vld1q_u16(pC + 8);
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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

            vst1q_u16(p0, vcombine_u16(float2bfloat(_f0), float2bfloat(_f1)));
            vst1q_u16(p0 + 8, vcombine_u16(float2bfloat(_f2), float2bfloat(_f3)));

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
                    // c_elempack == 1
                    uint16x8_t _c01 = vld1q_u16(pC);
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
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

            vst1q_u16(p0, vcombine_u16(float2bfloat(_f0), float2bfloat(_f1)));

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
                    // c_elempack == 1
                    _c0 = bfloat2float(vld1_u16(pC));
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 4;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1_u16(p0, float2bfloat(_f0));

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
                    // c_elempack == 1
                    float32x2_t _cc = float32x2_t();
                    _cc = vset_lane_f32(bfloat16_to_float32(pC[0]), _cc, 0);
                    _cc = vset_lane_f32(bfloat16_to_float32(pC[1]), _cc, 1);
                    _f0 = vmla_n_f32(_f0, _cc, beta);
                    pC += 2;
                }
            }

            _f0 = vmul_n_f32(_f0, alpha);

            p0[0] = float32_to_bfloat16(vget_lane_f32(_f0, 0));
            p0[1] = float32_to_bfloat16(vget_lane_f32(_f0, 1));

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
                    // c_elempack == 1
                    f0 += bfloat16_to_float32(pC[0]) * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = float32_to_bfloat16(f0);

            pp += 1;
            p0++;
        }
    }
}

static void transpose_unpack_output_tile_int32_to_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_unpack_output_tile_int32_to_bf16_asimddp(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
        return;
    }
#endif

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const int c_hstep = C.dims == 3 ? (int)C.cstep : C.w;
    const int c_elempack = C.elempack;
    const unsigned short* pC = C;

    // NCNN_LOGE("transpose_unpack_output_tile_int32_to_bf16  %d %d %d %d  %d  %d  %d", i, max_ii, j, max_jj, out_elempack, broadcast_type_C, c_elempack);

    const int* pp = topT;

    int ii = 0;
#if __ARM_NEON
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        float32x4_t _descale0 = vld1q_f32((const float*)descales + i + ii);
        float32x4_t _descale1 = vld1q_f32((const float*)descales + i + ii + 4);

        float32x4_t _c0;
        float32x4_t _c1;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                uint16x8_t _c = vld1q_u16(pC);
                _c0 = bfloat2float(vget_low_u16(_c));
                _c1 = bfloat2float(vget_high_u16(_c));
                _c0 = vmulq_n_f32(_c0, beta);
                _c1 = vmulq_n_f32(_c1, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        uint16x8_t _c45 = vld1q_u16(pC + 16);
                        uint16x8_t _c67 = vld1q_u16(pC + 24);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 4 + 8);
                        _c45 = vld1q_u16(pC + c_hstep * 4 + 16);
                        _c67 = vld1q_u16(pC + c_hstep * 4 + 24);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        _c4 = bfloat2float(vget_low_u16(_c45));
                        _c5 = bfloat2float(vget_high_u16(_c45));
                        _c6 = bfloat2float(vget_low_u16(_c67));
                        _c7 = bfloat2float(vget_high_u16(_c67));
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                        uint16x8_t _c45 = vld1q_u16(pC + c_hstep * 2);
                        uint16x8_t _c67 = vld1q_u16(pC + c_hstep * 3);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                        float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                        float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                        float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                        float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 5);
                        _c45 = vld1q_u16(pC + c_hstep * 6);
                        _c67 = vld1q_u16(pC + c_hstep * 7);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        _c4 = bfloat2float(vget_low_u16(_c45));
                        _c5 = bfloat2float(vget_high_u16(_c45));
                        _c6 = bfloat2float(vget_low_u16(_c67));
                        _c7 = bfloat2float(vget_high_u16(_c67));
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
                    uint16x8_t _c = vld1q_u16(pC);
                    float32x4_t _cc0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _cc1 = bfloat2float(vget_high_u16(_c));
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

            uint16x8_t _bf0 = vcombine_u16(float2bfloat(_f0), float2bfloat(_f8));
            uint16x8_t _bf1 = vcombine_u16(float2bfloat(_f1), float2bfloat(_f9));
            uint16x8_t _bf2 = vcombine_u16(float2bfloat(_f2), float2bfloat(_fa));
            uint16x8_t _bf3 = vcombine_u16(float2bfloat(_f3), float2bfloat(_fb));
            uint16x8_t _bf4 = vcombine_u16(float2bfloat(_f4), float2bfloat(_fc));
            uint16x8_t _bf5 = vcombine_u16(float2bfloat(_f5), float2bfloat(_fd));
            uint16x8_t _bf6 = vcombine_u16(float2bfloat(_f6), float2bfloat(_fe));
            uint16x8_t _bf7 = vcombine_u16(float2bfloat(_f7), float2bfloat(_ff));

            if (out_elempack == 4)
            {
                uint16x8x4_t _bfa;
                uint16x8x4_t _bfb;
                _bfa.val[0] = _bf0;
                _bfa.val[1] = _bf1;
                _bfa.val[2] = _bf2;
                _bfa.val[3] = _bf3;
                _bfb.val[0] = _bf4;
                _bfb.val[1] = _bf5;
                _bfb.val[2] = _bf6;
                _bfb.val[3] = _bf7;
                vst4q_u16(p0, _bfa);
                vst4q_u16(p0 + out_hstep * 4, _bfb);
            }
            if (out_elempack == 1)
            {
                vst1q_u16(p0, _bf0);
                vst1q_u16(p0 + out_hstep, _bf1);
                vst1q_u16(p0 + out_hstep * 2, _bf2);
                vst1q_u16(p0 + out_hstep * 3, _bf3);
                vst1q_u16(p0 + out_hstep * 4, _bf4);
                vst1q_u16(p0 + out_hstep * 5, _bf5);
                vst1q_u16(p0 + out_hstep * 6, _bf6);
                vst1q_u16(p0 + out_hstep * 7, _bf7);
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                        float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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
                        _c01 = vld1q_u16(pC + c_hstep * 4);
                        _c23 = vld1q_u16(pC + c_hstep * 4 + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
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
                        uint16x4_t _cc0 = vld1_u16(pC);
                        uint16x4_t _cc1 = vld1_u16(pC + c_hstep);
                        uint16x4_t _cc2 = vld1_u16(pC + c_hstep * 2);
                        uint16x4_t _cc3 = vld1_u16(pC + c_hstep * 3);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        float32x4_t _c2 = bfloat2float(_cc2);
                        float32x4_t _c3 = bfloat2float(_cc3);
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
                        _cc0 = vld1_u16(pC + c_hstep * 4);
                        _cc1 = vld1_u16(pC + c_hstep * 5);
                        _cc2 = vld1_u16(pC + c_hstep * 6);
                        _cc3 = vld1_u16(pC + c_hstep * 7);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        _c2 = bfloat2float(_cc2);
                        _c3 = bfloat2float(_cc3);
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
                    float32x4_t _c = bfloat2float(vld1_u16(pC));
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

            uint16x8_t _bf0 = vcombine_u16(float2bfloat(_f0), float2bfloat(_f4));
            uint16x8_t _bf1 = vcombine_u16(float2bfloat(_f1), float2bfloat(_f5));
            uint16x8_t _bf2 = vcombine_u16(float2bfloat(_f2), float2bfloat(_f6));
            uint16x8_t _bf3 = vcombine_u16(float2bfloat(_f3), float2bfloat(_f7));

            if (out_elempack == 4)
            {
                uint16x8x4_t _bf;
                _bf.val[0] = _bf0;
                _bf.val[1] = _bf1;
                _bf.val[2] = _bf2;
                _bf.val[3] = _bf3;
                vst4q_u16(p0, _bf);
            }
            if (out_elempack == 1)
            {
                vst1q_u16(p0, _bf0);
                vst1q_u16(p0 + out_hstep, _bf1);
                vst1q_u16(p0 + out_hstep * 2, _bf2);
                vst1q_u16(p0 + out_hstep * 3, _bf3);
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
                    float32x4_t _c2;
                    float32x4_t _c3;
                    if (c_elempack == 4)
                    {
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + c_hstep * 4);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        uint16x8_t _c01 = uint16x8_t();
                        _c01 = vsetq_lane_u16(pC[0], _c01, 0);
                        _c01 = vsetq_lane_u16(pC[c_hstep], _c01, 1);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 2], _c01, 2);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 3], _c01, 3);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 4], _c01, 4);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 5], _c01, 5);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 6], _c01, 6);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 7], _c01, 7);

                        uint16x8_t _c23 = uint16x8_t();
                        _c23 = vsetq_lane_u16(pC[1], _c23, 0);
                        _c23 = vsetq_lane_u16(pC[c_hstep + 1], _c23, 1);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 2 + 1], _c23, 2);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 3 + 1], _c23, 3);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 4 + 1], _c23, 4);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 5 + 1], _c23, 5);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 6 + 1], _c23, 6);
                        _c23 = vsetq_lane_u16(pC[c_hstep * 7 + 1], _c23, 7);

                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_low_u16(_c23));
                        _c2 = bfloat2float(vget_high_u16(_c01));
                        _c3 = bfloat2float(vget_high_u16(_c23));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]) * beta);
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

            vst1q_u16(p0, vcombine_u16(float2bfloat(_f0), float2bfloat(_f2)));
            vst1q_u16(p0 + out_hstep, vcombine_u16(float2bfloat(_f1), float2bfloat(_f3)));

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
                    if (c_elempack == 4)
                    {
                        _c0 = bfloat2float(vld1_u16(pC));
                        _c1 = bfloat2float(vld1_u16(pC + c_hstep * 4));
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        uint16x8_t _c01 = uint16x8_t();
                        _c01 = vsetq_lane_u16(pC[0], _c01, 0);
                        _c01 = vsetq_lane_u16(pC[c_hstep], _c01, 1);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 2], _c01, 2);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 3], _c01, 3);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 4], _c01, 4);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 5], _c01, 5);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 6], _c01, 6);
                        _c01 = vsetq_lane_u16(pC[c_hstep * 7], _c01, 7);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
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

            vst1q_u16(p0, vcombine_u16(float2bfloat(_f0), float2bfloat(_f1)));
            pp += 8;
            p0 += out_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

        float32x4_t _descale = vld1q_f32((const float*)descales + i + ii);

        float32x4_t _c0;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                _c0 = bfloat2float(vld1_u16(pC));
                _c0 = vmulq_n_f32(_c0, beta);
            }
            if (broadcast_type_C == 3)
            {
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j * c_elempack;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01;
                    uint16x8_t _c23;
                    uint16x8_t _c45;
                    uint16x8_t _c67;
                    if (c_elempack == 4)
                    {
                        _c01 = vld1q_u16(pC);
                        _c23 = vld1q_u16(pC + 8);
                        _c45 = vld1q_u16(pC + 16);
                        _c67 = vld1q_u16(pC + 24);
                        pC += 32;
                    }
                    if (c_elempack == 1)
                    {
                        _c01 = vld1q_u16(pC);
                        _c23 = vld1q_u16(pC + c_hstep);
                        _c45 = vld1q_u16(pC + c_hstep * 2);
                        _c67 = vld1q_u16(pC + c_hstep * 3);
                        transpose8x4_u16(_c01, _c23, _c45, _c67);
                        pC += 8;
                    }
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
                    float32x4_t _c4 = bfloat2float(vget_low_u16(_c45));
                    float32x4_t _c5 = bfloat2float(vget_high_u16(_c45));
                    float32x4_t _c6 = bfloat2float(vget_low_u16(_c67));
                    float32x4_t _c7 = bfloat2float(vget_high_u16(_c67));
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
                    uint16x8_t _c = vld1q_u16(pC);
                    float32x4_t _cc0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _cc1 = bfloat2float(vget_high_u16(_c));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);
            uint16x4_t _bf4 = float2bfloat(_f4);
            uint16x4_t _bf5 = float2bfloat(_f5);
            uint16x4_t _bf6 = float2bfloat(_f6);
            uint16x4_t _bf7 = float2bfloat(_f7);

            if (out_elempack == 4)
            {
                uint16x4x4_t _bfa;
                uint16x4x4_t _bfb;
                _bfa.val[0] = _bf0;
                _bfa.val[1] = _bf1;
                _bfa.val[2] = _bf2;
                _bfa.val[3] = _bf3;
                _bfb.val[0] = _bf4;
                _bfb.val[1] = _bf5;
                _bfb.val[2] = _bf6;
                _bfb.val[3] = _bf7;
                vst4_u16(p0, _bfa);
                vst4_u16(p0 + out_hstep * 4, _bfb);
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, _bf0);
                vst1_u16(p0 + out_hstep, _bf1);
                vst1_u16(p0 + out_hstep * 2, _bf2);
                vst1_u16(p0 + out_hstep * 3, _bf3);
                vst1_u16(p0 + out_hstep * 4, _bf4);
                vst1_u16(p0 + out_hstep * 5, _bf5);
                vst1_u16(p0 + out_hstep * 6, _bf6);
                vst1_u16(p0 + out_hstep * 7, _bf7);
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
                        uint16x8_t _c01 = vld1q_u16(pC);
                        uint16x8_t _c23 = vld1q_u16(pC + 8);
                        _c0 = bfloat2float(vget_low_u16(_c01));
                        _c1 = bfloat2float(vget_high_u16(_c01));
                        _c2 = bfloat2float(vget_low_u16(_c23));
                        _c3 = bfloat2float(vget_high_u16(_c23));
                        pC += 16;
                    }
                    if (c_elempack == 1)
                    {
                        uint16x4_t _cc0 = vld1_u16(pC);
                        uint16x4_t _cc1 = vld1_u16(pC + c_hstep);
                        uint16x4_t _cc2 = vld1_u16(pC + c_hstep * 2);
                        uint16x4_t _cc3 = vld1_u16(pC + c_hstep * 3);
                        transpose4x4_u16(_cc0, _cc1, _cc2, _cc3);
                        _c0 = bfloat2float(_cc0);
                        _c1 = bfloat2float(_cc1);
                        _c2 = bfloat2float(_cc2);
                        _c3 = bfloat2float(_cc3);
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
                    float32x4_t _c = bfloat2float(vld1_u16(pC));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);

            if (out_elempack == 4)
            {
                uint16x4x4_t _bf;
                _bf.val[0] = _bf0;
                _bf.val[1] = _bf1;
                _bf.val[2] = _bf2;
                _bf.val[3] = _bf3;
                vst4_u16(p0, _bf);
            }
            if (out_elempack == 1)
            {
                vst1_u16(p0, _bf0);
                vst1_u16(p0 + out_hstep, _bf1);
                vst1_u16(p0 + out_hstep * 2, _bf2);
                vst1_u16(p0 + out_hstep * 3, _bf3);
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
                    uint16x8_t _c;
                    if (c_elempack == 4)
                    {
                        _c = vld1q_u16(pC);
                        pC += 8;
                    }
                    if (c_elempack == 1)
                    {
                        _c = uint16x8_t();
                        _c = vsetq_lane_u16(pC[0], _c, 0);
                        _c = vsetq_lane_u16(pC[c_hstep], _c, 1);
                        _c = vsetq_lane_u16(pC[c_hstep * 2], _c, 2);
                        _c = vsetq_lane_u16(pC[c_hstep * 3], _c, 3);
                        _c = vsetq_lane_u16(pC[1], _c, 4);
                        _c = vsetq_lane_u16(pC[c_hstep + 1], _c, 5);
                        _c = vsetq_lane_u16(pC[c_hstep * 2 + 1], _c, 6);
                        _c = vsetq_lane_u16(pC[c_hstep * 3 + 1], _c, 7);
                        pC += 2;
                    }
                    _c0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
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
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    float32x4_t _c1 = vdupq_n_f32(bfloat16_to_float32(pC[1]) * beta);
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

            vst1_u16(p0, float2bfloat(_f0));
            vst1_u16(p0 + out_hstep, float2bfloat(_f1));

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
                    uint16x4_t _c;
                    if (c_elempack == 4)
                    {
                        _c = vld1_u16(pC);
                        pC += 4;
                    }
                    if (c_elempack == 1)
                    {
                        _c = uint16x4_t();
                        _c = vset_lane_u16(pC[0], _c, 0);
                        _c = vset_lane_u16(pC[c_hstep], _c, 1);
                        _c = vset_lane_u16(pC[c_hstep * 2], _c, 2);
                        _c = vset_lane_u16(pC[c_hstep * 3], _c, 3);
                        pC += 1;
                    }
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _c0 = vdupq_n_f32(bfloat16_to_float32(pC[0]) * beta);
                    _f0 = vaddq_f32(_f0, _c0);
                    pC += 1;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            vst1_u16(p0, float2bfloat(_f0));
            pp += 4;
            p0 += out_hstep;
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

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
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                c0 = bfloat16_to_float32(pC[0]) * beta;
                c1 = bfloat16_to_float32(pC[1]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
                _c1 = vdupq_n_f32(c1);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01 = vld1q_u16(pC);
                    uint16x8_t _c23 = vld1q_u16(pC + c_hstep);
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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
                    uint16x8_t _c = vld1q_u16(pC);
                    _c0 = bfloat2float(vget_low_u16(_c));
                    _c1 = bfloat2float(vget_high_u16(_c));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf2));
                vst1q_u16(p0 + out_hstep * 4, vcombine_u16(_bf1, _bf3));
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[1] = vget_lane_u16(_bf2, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep + 1] = vget_lane_u16(_bf2, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 2 + 1] = vget_lane_u16(_bf2, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0[out_hstep * 3 + 1] = vget_lane_u16(_bf2, 3);
                p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                p0[out_hstep * 4 + 1] = vget_lane_u16(_bf3, 0);
                p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                p0[out_hstep * 5 + 1] = vget_lane_u16(_bf3, 1);
                p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                p0[out_hstep * 6 + 1] = vget_lane_u16(_bf3, 2);
                p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
                p0[out_hstep * 7 + 1] = vget_lane_u16(_bf3, 3);
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
                    _c0 = bfloat2float(vld1_u16(pC));
                    _c1 = bfloat2float(vld1_u16(pC + c_hstep));
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
                    _c0 = bfloat2float(vld1_u16(pC));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);

            if (out_elempack == 4)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
            }
            if (out_elempack == 1)
            {
                p0[0] = vget_lane_u16(_bf0, 0);
                p0[1] = vget_lane_u16(_bf1, 0);
                p0[out_hstep] = vget_lane_u16(_bf0, 1);
                p0[out_hstep + 1] = vget_lane_u16(_bf1, 1);
                p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                p0[out_hstep * 2 + 1] = vget_lane_u16(_bf1, 2);
                p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                p0[out_hstep * 3 + 1] = vget_lane_u16(_bf1, 3);
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
                    uint16x4_t _c = uint16x4_t();
                    _c = vset_lane_u16(pC[0], _c, 0);
                    _c = vset_lane_u16(pC[c_hstep], _c, 1);
                    _c = vset_lane_u16(pC[1], _c, 2);
                    _c = vset_lane_u16(pC[c_hstep + 1], _c, 3);
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    uint16x4_t _c = uint16x4_t();
                    _c = vset_lane_u16(pC[0], _c, 0);
                    _c = vset_lane_u16(pC[0], _c, 1);
                    _c = vset_lane_u16(pC[1], _c, 2);
                    _c = vset_lane_u16(pC[1], _c, 3);
                    _c0 = bfloat2float(_c);
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 2;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            uint16x4_t _bf0 = float2bfloat(_f0);

            p0[0] = vget_lane_u16(_bf0, 0);
            p0[1] = vget_lane_u16(_bf0, 1);
            p0[out_hstep] = vget_lane_u16(_bf0, 2);
            p0[out_hstep + 1] = vget_lane_u16(_bf0, 3);

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
                    f0 += bfloat16_to_float32(pC[0]) * beta;
                    f1 += bfloat16_to_float32(pC[c_hstep]) * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    c0 = bfloat16_to_float32(pC[0]) * beta;
                    f0 += c0;
                    f1 += c0;
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                f0 *= alpha;
                f1 *= alpha;
            }

            p0[0] = float32_to_bfloat16(f0);
            p0[1] = float32_to_bfloat16(f1);
            pp += 2;
            p0 += out_hstep;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * out_elempack;

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
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const unsigned short*)C + i + ii;
                c0 = bfloat16_to_float32(pC[0]) * beta;
#if __ARM_NEON
                _c0 = vdupq_n_f32(c0);
#endif
            }
            if (broadcast_type_C == 3)
            {
                // c_elempack == 1
                pC = (const unsigned short*)C + (i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const unsigned short*)C + j;
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
                    uint16x8_t _c01 = vld1q_u16(pC);
                    uint16x8_t _c23 = vld1q_u16(pC + 8);
                    _c0 = bfloat2float(vget_low_u16(_c01));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c01));
                    float32x4_t _c2 = bfloat2float(vget_low_u16(_c23));
                    float32x4_t _c3 = bfloat2float(vget_high_u16(_c23));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);
            uint16x4_t _bf2 = float2bfloat(_f2);
            uint16x4_t _bf3 = float2bfloat(_f3);

            if (out_hstep == 1)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
                vst1q_u16(p0 + 8, vcombine_u16(_bf2, _bf3));
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1_u16(p0, _bf0);
                    vst1_u16(p0 + out_hstep * 4, _bf1);
                    vst1_u16(p0 + out_hstep * 8, _bf2);
                    vst1_u16(p0 + out_hstep * 12, _bf3);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vget_lane_u16(_bf0, 0);
                    p0[out_hstep] = vget_lane_u16(_bf0, 1);
                    p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                    p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                    p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                    p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                    p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                    p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
                    p0[out_hstep * 8] = vget_lane_u16(_bf2, 0);
                    p0[out_hstep * 9] = vget_lane_u16(_bf2, 1);
                    p0[out_hstep * 10] = vget_lane_u16(_bf2, 2);
                    p0[out_hstep * 11] = vget_lane_u16(_bf2, 3);
                    p0[out_hstep * 12] = vget_lane_u16(_bf3, 0);
                    p0[out_hstep * 13] = vget_lane_u16(_bf3, 1);
                    p0[out_hstep * 14] = vget_lane_u16(_bf3, 2);
                    p0[out_hstep * 15] = vget_lane_u16(_bf3, 3);
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
                    uint16x8_t _c = vld1q_u16(pC);
                    _c0 = bfloat2float(vget_low_u16(_c));
                    float32x4_t _c1 = bfloat2float(vget_high_u16(_c));
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

            uint16x4_t _bf0 = float2bfloat(_f0);
            uint16x4_t _bf1 = float2bfloat(_f1);

            if (out_hstep == 1)
            {
                vst1q_u16(p0, vcombine_u16(_bf0, _bf1));
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1_u16(p0, _bf0);
                    vst1_u16(p0 + out_hstep * 4, _bf1);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vget_lane_u16(_bf0, 0);
                    p0[out_hstep] = vget_lane_u16(_bf0, 1);
                    p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                    p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
                    p0[out_hstep * 4] = vget_lane_u16(_bf1, 0);
                    p0[out_hstep * 5] = vget_lane_u16(_bf1, 1);
                    p0[out_hstep * 6] = vget_lane_u16(_bf1, 2);
                    p0[out_hstep * 7] = vget_lane_u16(_bf1, 3);
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
                    _c0 = bfloat2float(vld1_u16(pC));
                    _f0 = vmlaq_n_f32(_f0, _c0, beta);
                    pC += 4;
                }
            }

            _f0 = vmulq_n_f32(_f0, alpha);

            uint16x4_t _bf0 = float2bfloat(_f0);

            if (out_hstep == 1)
            {
                vst1_u16(p0, _bf0);
            }
            else
            {
                if (out_elempack == 4)
                {
                    vst1_u16(p0, _bf0);
                }
                if (out_elempack == 1)
                {
                    p0[0] = vget_lane_u16(_bf0, 0);
                    p0[out_hstep] = vget_lane_u16(_bf0, 1);
                    p0[out_hstep * 2] = vget_lane_u16(_bf0, 2);
                    p0[out_hstep * 3] = vget_lane_u16(_bf0, 3);
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
                    float32x2_t _c = float32x2_t();
                    _c = vset_lane_f32(bfloat16_to_float32(pC[0]), _c, 0);
                    _c = vset_lane_f32(bfloat16_to_float32(pC[1]), _c, 1);
                    _f0 = vmla_n_f32(_f0, _c, beta);
                    pC += 2;
                }
            }

            _f0 = vmul_n_f32(_f0, alpha);

            p0[0] = float32_to_bfloat16(vget_lane_f32(_f0, 0));
            p0[out_hstep] = float32_to_bfloat16(vget_lane_f32(_f0, 1));

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
                    f0 += bfloat16_to_float32(pC[0]) * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = float32_to_bfloat16(f0);

            pp += 1;
            p0 += out_hstep;
        }
    }
}
