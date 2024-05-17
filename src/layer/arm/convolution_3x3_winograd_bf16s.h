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

static inline void conv3x3s1_winograd23_transform_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __ARM_NEON
#if __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        _r00 = bfloat2float(vld1_u16(r0));
                        _r01 = bfloat2float(vld1_u16(r1));
                        if (tj * 2 + 1 < w)
                        {
                            _r10 = bfloat2float(vld1_u16(r0 + 4));
                            _r11 = bfloat2float(vld1_u16(r1 + 4));
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r20 = bfloat2float(vld1_u16(r0 + 8));
                            _r21 = bfloat2float(vld1_u16(r1 + 8));
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r30 = bfloat2float(vld1_u16(r0 + 12));
                            _r31 = bfloat2float(vld1_u16(r1 + 12));
                        }
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;
                        const unsigned short* r4 = r0 + N * 4;
                        const unsigned short* r5 = r0 + N * 5;
                        const unsigned short* r6 = r0 + N * 6;
                        const unsigned short* r7 = r0 + N * 7;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);
                        uint16x4_t _t4 = vld1_u16(r4);
                        uint16x4_t _t5 = vld1_u16(r5);
                        uint16x4_t _t6 = vld1_u16(r6);
                        uint16x4_t _t7 = vld1_u16(r7);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);
                        transpose4x4_u16(_t4, _t5, _t6, _t7);

                        _r00 = bfloat2float(_t0);
                        _r01 = bfloat2float(_t4);
                        if (tj * 2 + 1 < w)
                        {
                            _r10 = bfloat2float(_t1);
                            _r11 = bfloat2float(_t5);
                        }
                        if (tj * 2 + 2 < w)
                        {
                            _r20 = bfloat2float(_t2);
                            _r21 = bfloat2float(_t6);
                        }
                        if (tj * 2 + 3 < w)
                        {
                            _r30 = bfloat2float(_t3);
                            _r31 = bfloat2float(_t7);
                        }
                    }
                }

                float32x4_t _tmp00 = vsubq_f32(_r00, _r20);
                float32x4_t _tmp01 = vsubq_f32(_r01, _r21);
                float32x4_t _tmp10 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp11 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp20 = vsubq_f32(_r20, _r10);
                float32x4_t _tmp21 = vsubq_f32(_r21, _r11);
                float32x4_t _tmp30 = vsubq_f32(_r30, _r10);
                float32x4_t _tmp31 = vsubq_f32(_r31, _r11);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);

                float32x4_t _tmp00 = vsubq_f32(_r00, _r20);
                float32x4_t _tmp01 = vsubq_f32(_r01, _r21);
                float32x4_t _tmp10 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp11 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp20 = vsubq_f32(_r20, _r10);
                float32x4_t _tmp21 = vsubq_f32(_r21, _r11);
                float32x4_t _tmp30 = vsubq_f32(_r30, _r10);
                float32x4_t _tmp31 = vsubq_f32(_r31, _r11);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);

                p0 += max_jj * 4 * 8;
                p1 += max_jj * 4 * 8;
                p2 += max_jj * 4 * 8;
                p3 += max_jj * 4 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 2) + (tj * 2) * elempack;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);

                if (ti * 2 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        if (tj * 2 + 1 < w) _r1 = bfloat2float(vld1_u16(r0 + 4));
                        if (tj * 2 + 2 < w) _r2 = bfloat2float(vld1_u16(r0 + 8));
                        if (tj * 2 + 3 < w) _r3 = bfloat2float(vld1_u16(r0 + 12));
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);

                        _r0 = bfloat2float(_t0);
                        if (tj * 2 + 1 < w) _r1 = bfloat2float(_t1);
                        if (tj * 2 + 2 < w) _r2 = bfloat2float(_t2);
                        if (tj * 2 + 3 < w) _r3 = bfloat2float(_t3);
                    }
                }

                float32x4_t _tmp0 = vsubq_f32(_r0, _r2);
                float32x4_t _tmp1 = vaddq_f32(_r1, _r2);
                float32x4_t _tmp2 = vsubq_f32(_r2, _r1);
                float32x4_t _tmp3 = vsubq_f32(_r3, _r1);

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);

                float32x4_t _tmp0 = vsubq_f32(_r0, _r2);
                float32x4_t _tmp1 = vaddq_f32(_r1, _r2);
                float32x4_t _tmp2 = vsubq_f32(_r2, _r1);
                float32x4_t _tmp3 = vsubq_f32(_r3, _r1);

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);

                p0 += max_jj * 4 * 4;
                p1 += max_jj * 4 * 4;
                p2 += max_jj * 4 * 4;
                p3 += max_jj * 4 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
#endif

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;

#if __ARM_NEON
                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4x2_t _t01 = vzip_u16(_t0, _t1);
                        float32x4_t _t0_fp32 = bfloat2float(_t01.val[0]);
                        float32x4_t _t1_fp32 = bfloat2float(_t01.val[1]);

                        _r0 = vget_low_f32(_t0_fp32);
                        if (tj * 2 + 1 < w) _r1 = vget_high_f32(_t0_fp32);
                        if (tj * 2 + 2 < w) _r2 = vget_low_f32(_t1_fp32);
                        if (tj * 2 + 3 < w) _r3 = vget_high_f32(_t1_fp32);
#else
                        r00 = bfloat16_to_float32(r0[0]);
                        r01 = bfloat16_to_float32(r1[0]);
                        if (tj * 2 + 1 < w)
                        {
                            r10 = bfloat16_to_float32(r0[1]);
                            r11 = bfloat16_to_float32(r1[1]);
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = bfloat16_to_float32(r0[2]);
                            r21 = bfloat16_to_float32(r1[2]);
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = bfloat16_to_float32(r0[3]);
                            r31 = bfloat16_to_float32(r1[3]);
                        }
#endif
                    }
                }

#if __ARM_NEON
                float32x2_t _tmp0 = vsub_f32(_r0, _r2);
                float32x2_t _tmp1 = vadd_f32(_r1, _r2);
                float32x2_t _tmp2 = vsub_f32(_r2, _r1);
                float32x2_t _tmp3 = vsub_f32(_r3, _r1);

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
#else
                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);

                float32x2_t _tmp0 = vsub_f32(_r0, _r2);
                float32x2_t _tmp1 = vadd_f32(_r1, _r2);
                float32x2_t _tmp2 = vsub_f32(_r2, _r1);
                float32x2_t _tmp3 = vsub_f32(_r3, _r1);

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;
#endif

                p0 += max_jj * 4 * 2;
                p1 += max_jj * 4 * 2;
                p2 += max_jj * 4 * 2;
                p3 += max_jj * 4 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        float tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0123 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = bfloat16_to_float32(r0123[0]);
                        if (tj * 2 + 1 < w) r1 = bfloat16_to_float32(r0123[1]);
                        if (tj * 2 + 2 < w) r2 = bfloat16_to_float32(r0123[2]);
                        if (tj * 2 + 3 < w) r3 = bfloat16_to_float32(r0123[3]);
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 16 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                p0[0] = r0 - r2;
                p1[0] = r1 + r2;
                p2[0] = r2 - r1;
                p3[0] = r3 - r1;

                p0 += max_jj * 4;
                p1 += max_jj * 4;
                p2 += max_jj * 4;
                p3 += max_jj * 4;
            }
        }
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_bf16s(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[2][4][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _r10), _r20);
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _r11), _r21);
                float32x4_t _tmp10 = vaddq_f32(vsubq_f32(_r10, _r20), _r30);
                float32x4_t _tmp11 = vaddq_f32(vsubq_f32(_r11, _r21), _r31);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);

                r0 += max_jj * 4 * 8;
                r1 += max_jj * 4 * 8;
                r2 += max_jj * 4 * 8;
                r3 += max_jj * 4 * 8;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);

                float32x4_t _tmp00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r00, _r10), _r20));
                float32x4_t _tmp01 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r01, _r11), _r21));
                float32x4_t _tmp10 = vaddq_f32(_bias0, vaddq_f32(vsubq_f32(_r10, _r20), _r30));
                float32x4_t _tmp11 = vaddq_f32(_bias1, vaddq_f32(vsubq_f32(_r11, _r21), _r31));

                if (out_elempack == 4)
                {
                    unsigned short* outptr1 = outptr0 + N;

                    vst1_u16(outptr0, float2bfloat(_tmp00));
                    vst1_u16(outptr1, float2bfloat(_tmp01));
                    if (tj * 2 + 1 < outw)
                    {
                        vst1_u16(outptr0 + 4, float2bfloat(_tmp10));
                        vst1_u16(outptr1 + 4, float2bfloat(_tmp11));
                    }
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[8];
                    unsigned short tmp1[8];
                    vst1_u16(tmp0, float2bfloat(_tmp00));
                    vst1_u16(tmp0 + 4, float2bfloat(_tmp01));
                    vst1_u16(tmp1, float2bfloat(_tmp10));
                    vst1_u16(tmp1 + 4, float2bfloat(_tmp11));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;
                    unsigned short* outptr4 = outptr0 + N * 4;
                    unsigned short* outptr5 = outptr0 + N * 5;
                    unsigned short* outptr6 = outptr0 + N * 6;
                    unsigned short* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[2][4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;

            for (int m = 0; m < 4; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _r1), _r2);
                float32x4_t _tmp1 = vaddq_f32(vsubq_f32(_r1, _r2), _r3);

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);

                r0 += max_jj * 4 * 4;
                r1 += max_jj * 4 * 4;
                r2 += max_jj * 4 * 4;
                r3 += max_jj * 4 * 4;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 2) + (tj * 2) * out_elempack;

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);

                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _r1), _r2));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vaddq_f32(vsubq_f32(_r1, _r2), _r3));

                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_tmp0));
                    if (tj * 2 + 1 < outw) vst1_u16(outptr0 + 4, float2bfloat(_tmp1));
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[4];
                    unsigned short tmp1[4];
                    vst1_u16(tmp0, float2bfloat(_tmp0));
                    vst1_u16(tmp1, float2bfloat(_tmp1));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];

                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _r1), _r2);
                float32x2_t _tmp1 = vadd_f32(vsub_f32(_r1, _r2), _r3);

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
#else
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];
#endif

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);

                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _r1), _r2));
                float32x2_t _tmp1 = vadd_f32(_bias0, vadd_f32(vsub_f32(_r1, _r2), _r3));
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];

                float tmp00 = bias0 + r00 + r10 + r20;
                float tmp01 = bias1 + r01 + r11 + r21;
                float tmp10 = bias0 + r10 - r20 + r30;
                float tmp11 = bias1 + r11 - r21 + r31;
#endif

                // if (out_elempack == 1)
                {
                    unsigned short* outptr1 = outptr0 + N;

#if __ARM_NEON
                    uint16x4_t _tmp01 = float2bfloat(vcombine_f32(_tmp0, _tmp1));

                    outptr0[0] = vget_lane_u16(_tmp01, 0);
                    outptr1[0] = vget_lane_u16(_tmp01, 1);
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_u16(_tmp01, 2);
                        outptr1[1] = vget_lane_u16(_tmp01, 3);
                    }
#else
                    outptr0[0] = float32_to_bfloat16(tmp00);
                    outptr1[0] = float32_to_bfloat16(tmp01);
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = float32_to_bfloat16(tmp10);
                        outptr1[1] = float32_to_bfloat16(tmp11);
                    }
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 16 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];

                float tmp0 = bias0 + r0 + r1 + r2;
                float tmp1 = bias0 + r1 - r2 + r3;

                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(tmp0);
                    if (tj * 2 + 1 < outw) outptr0[1] = float32_to_bfloat16(tmp1);
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd23_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 2n+2, winograd F(2,3)
    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    // NCNN_LOGE("conv3x3s1_winograd23_bf16s %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd23_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    const float sq2 = 1.41421356237;
    const float sq2_d2 = 1.41421356237 / 2;

    // const float itm[6][6] = {
    //     {1.0f,  0.0f,  -2.5f,  0.0f,  1.0f, 0.0f},
    //     {0.0f, -sq2,   -2.0f,  sq2/2, 1.0f, 0.0f},
    //     {0.0f,  sq2,   -2.0f, -sq2/2, 1.0f, 0.0f},
    //     {0.0f, -sq2/2, -0.5f,  sq2,   1.0f, 0.0f},
    //     {0.0f,  sq2/2, -0.5f, -sq2,   1.0f, 0.0f},
    //     {0.0f,  1.0f,   0.0f,  -2.5f, 0.0f, 1.0f}
    // };

    // 0 =  r00 + r04 - 2.5f * r02
    // 1 = -(sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 2 =  (sq2 * r01 - sq2_d2 * r03) + (r04 - 2 * r02)
    // 3 =  (sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 4 = -(sq2 * r03 - sq2_d2 * r01) + (r04 - 0.5f * r02)
    // 5 =  r01 + r05 - 2.5f * r03

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __ARM_NEON
#if __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][6][8];

        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _vm2_5 = vdupq_n_f32(-2.5f);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);
                float32x4_t _r40 = vdupq_n_f32(0.f);
                float32x4_t _r41 = vdupq_n_f32(0.f);
                float32x4_t _r50 = vdupq_n_f32(0.f);
                float32x4_t _r51 = vdupq_n_f32(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        _r00 = bfloat2float(vld1_u16(r0));
                        _r01 = bfloat2float(vld1_u16(r1));
                        if (tj * 4 + 1 < w)
                        {
                            _r10 = bfloat2float(vld1_u16(r0 + 4));
                            _r11 = bfloat2float(vld1_u16(r1 + 4));
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r20 = bfloat2float(vld1_u16(r0 + 8));
                            _r21 = bfloat2float(vld1_u16(r1 + 8));
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r30 = bfloat2float(vld1_u16(r0 + 12));
                            _r31 = bfloat2float(vld1_u16(r1 + 12));
                        }
                        if (tj * 4 + 4 < w)
                        {
                            _r40 = bfloat2float(vld1_u16(r0 + 16));
                            _r41 = bfloat2float(vld1_u16(r1 + 16));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            _r50 = bfloat2float(vld1_u16(r0 + 20));
                            _r51 = bfloat2float(vld1_u16(r1 + 20));
                        }
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;
                        const unsigned short* r4 = r0 + N * 4;
                        const unsigned short* r5 = r0 + N * 5;
                        const unsigned short* r6 = r0 + N * 6;
                        const unsigned short* r7 = r0 + N * 7;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);
                        uint16x4_t _t4 = vld1_u16(r4);
                        uint16x4_t _t5 = vld1_u16(r5);
                        uint16x4_t _t6 = vld1_u16(r6);
                        uint16x4_t _t7 = vld1_u16(r7);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);
                        transpose4x4_u16(_t4, _t5, _t6, _t7);

                        _r00 = bfloat2float(_t0);
                        _r01 = bfloat2float(_t4);
                        if (tj * 4 + 1 < w)
                        {
                            _r10 = bfloat2float(_t1);
                            _r11 = bfloat2float(_t5);
                        }
                        if (tj * 4 + 2 < w)
                        {
                            _r20 = bfloat2float(_t2);
                            _r21 = bfloat2float(_t6);
                        }
                        if (tj * 4 + 3 < w)
                        {
                            _r30 = bfloat2float(_t3);
                            _r31 = bfloat2float(_t7);
                        }
                        if (tj * 4 + 4 < w)
                        {
                            unsigned short tmp[8] = {r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4]};
                            _r40 = bfloat2float(vld1_u16(tmp));
                            _r41 = bfloat2float(vld1_u16(tmp + 4));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            unsigned short tmp[8] = {r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5]};
                            _r50 = bfloat2float(vld1_u16(tmp));
                            _r51 = bfloat2float(vld1_u16(tmp + 4));
                        }
                    }
                }

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs, 0), _r30, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs, 0), _r31, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 2);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 2);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r30, _coeffs, 0), _r10, _coeffs, 1);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r31, _coeffs, 0), _r11, _coeffs, 1);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 3);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 3);

                float32x4_t _tmp00 = vfmaq_f32(vaddq_f32(_r00, _r40), _r20, _vm2_5);
                float32x4_t _tmp01 = vfmaq_f32(vaddq_f32(_r01, _r41), _r21, _vm2_5);
                float32x4_t _tmp10 = vsubq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp11 = vsubq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp20 = vaddq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp21 = vaddq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp50 = vfmaq_f32(vaddq_f32(_r10, _r50), _r30, _vm2_5);
                float32x4_t _tmp51 = vfmaq_f32(vaddq_f32(_r11, _r51), _r31, _vm2_5);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs, 0), _r30, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs, 0), _r31, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 2);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 2);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vmulq_laneq_f32(_r30, _coeffs, 0), _r10, _coeffs, 1);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vmulq_laneq_f32(_r31, _coeffs, 0), _r11, _coeffs, 1);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(_r40, _r20, _coeffs, 3);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(_r41, _r21, _coeffs, 3);

                float32x4_t _tmp00 = vfmaq_f32(vaddq_f32(_r00, _r40), _r20, _vm2_5);
                float32x4_t _tmp01 = vfmaq_f32(vaddq_f32(_r01, _r41), _r21, _vm2_5);
                float32x4_t _tmp10 = vsubq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp11 = vsubq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp20 = vaddq_f32(_tmp12b0, _tmp12a0);
                float32x4_t _tmp21 = vaddq_f32(_tmp12b1, _tmp12a1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34b0, _tmp34a0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34b1, _tmp34a1);
                float32x4_t _tmp50 = vfmaq_f32(vaddq_f32(_r10, _r50), _r30, _vm2_5);
                float32x4_t _tmp51 = vfmaq_f32(vaddq_f32(_r11, _r51), _r31, _vm2_5);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);
                vst1q_f32(p4, _tmp40);
                vst1q_f32(p4 + 4, _tmp41);
                vst1q_f32(p5, _tmp50);
                vst1q_f32(p5 + 4, _tmp51);

                p0 += max_jj * 6 * 8;
                p1 += max_jj * 6 * 8;
                p2 += max_jj * 6 * 8;
                p3 += max_jj * 6 * 8;
                p4 += max_jj * 6 * 8;
                p5 += max_jj * 6 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][6][4];

        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _vm2_5 = vdupq_n_f32(-2.5f);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 4) + (tj * 4) * elempack;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);
                float32x4_t _r4 = vdupq_n_f32(0.f);
                float32x4_t _r5 = vdupq_n_f32(0.f);

                if (ti * 4 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        if (tj * 4 + 1 < w) _r1 = bfloat2float(vld1_u16(r0 + 4));
                        if (tj * 4 + 2 < w) _r2 = bfloat2float(vld1_u16(r0 + 8));
                        if (tj * 4 + 3 < w) _r3 = bfloat2float(vld1_u16(r0 + 12));
                        if (tj * 4 + 4 < w) _r4 = bfloat2float(vld1_u16(r0 + 16));
                        if (tj * 4 + 5 < w) _r5 = bfloat2float(vld1_u16(r0 + 20));
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);

                        _r0 = bfloat2float(_t0);
                        if (tj * 4 + 1 < w) _r1 = bfloat2float(_t1);
                        if (tj * 4 + 2 < w) _r2 = bfloat2float(_t2);
                        if (tj * 4 + 3 < w) _r3 = bfloat2float(_t3);
                        if (tj * 4 + 4 < w)
                        {
                            unsigned short tmp[4] = {r0[4], r1[4], r2[4], r3[4]};
                            _r4 = bfloat2float(vld1_u16(tmp));
                        }
                        if (tj * 4 + 5 < w)
                        {
                            unsigned short tmp[4] = {r0[5], r1[5], r2[5], r3[5]};
                            _r5 = bfloat2float(vld1_u16(tmp));
                        }
                    }
                }

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vmulq_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x4_t _tmp34b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmulq_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x4_t _tmp0 = vmlaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x4_t _tmp1 = vsubq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp2 = vaddq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp3 = vaddq_f32(_tmp34b, _tmp34a);
                float32x4_t _tmp4 = vsubq_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x4_t _tmp5 = vfmaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x4_t _tmp5 = vmlaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vmulq_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x4_t _tmp34b = vfmaq_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmulq_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34b = vmlaq_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x4_t _tmp0 = vmlaq_f32(vaddq_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x4_t _tmp1 = vsubq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp2 = vaddq_f32(_tmp12b, _tmp12a);
                float32x4_t _tmp3 = vaddq_f32(_tmp34b, _tmp34a);
                float32x4_t _tmp4 = vsubq_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x4_t _tmp5 = vfmaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x4_t _tmp5 = vmlaq_f32(vaddq_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);
                vst1q_f32(p4, _tmp4);
                vst1q_f32(p5, _tmp5);

                p0 += max_jj * 6 * 4;
                p1 += max_jj * 6 * 4;
                p2 += max_jj * 6 * 4;
                p3 += max_jj * 6 * 4;
                p4 += max_jj * 6 * 4;
                p5 += max_jj * 6 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[6][6][2];

#if __ARM_NEON
        const float coeffs[4] = {sq2, -sq2_d2, -2.f, -0.5f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x2_t _vm2_5 = vdup_n_f32(-2.5f);
#endif

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
                float32x2_t _r4 = vdup_n_f32(0.f);
                float32x2_t _r5 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
#endif

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;

#if __ARM_NEON
                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4x2_t _t01 = vzip_u16(_t0, _t1);
                        float32x4_t _t0_fp32 = bfloat2float(_t01.val[0]);
                        float32x4_t _t1_fp32 = bfloat2float(_t01.val[1]);

                        _r0 = vget_low_f32(_t0_fp32);
                        if (tj * 4 + 1 < w) _r1 = vget_high_f32(_t0_fp32);
                        if (tj * 4 + 2 < w) _r2 = vget_low_f32(_t1_fp32);
                        if (tj * 4 + 3 < w) _r3 = vget_high_f32(_t1_fp32);
                        if (tj * 4 + 4 < w)
                        {
                            float tmp[2] = {bfloat16_to_float32(r0[4]), bfloat16_to_float32(r1[4])};
                            _r4 = vld1_f32(tmp);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            float tmp[2] = {bfloat16_to_float32(r0[5]), bfloat16_to_float32(r1[5])};
                            _r5 = vld1_f32(tmp);
                        }
#else
                        r00 = bfloat16_to_float32(r0[0]);
                        r01 = bfloat16_to_float32(r1[0]);
                        if (tj * 4 + 1 < w)
                        {
                            r10 = bfloat16_to_float32(r0[1]);
                            r11 = bfloat16_to_float32(r1[1]);
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = bfloat16_to_float32(r0[2]);
                            r21 = bfloat16_to_float32(r1[2]);
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = bfloat16_to_float32(r0[3]);
                            r31 = bfloat16_to_float32(r1[3]);
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = bfloat16_to_float32(r0[4]);
                            r41 = bfloat16_to_float32(r1[4]);
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = bfloat16_to_float32(r0[5]);
                            r51 = bfloat16_to_float32(r1[5]);
                        }
#endif
                    }
                }

#if __ARM_NEON
#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x2_t _tmp34a = vfma_laneq_f32(vmul_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x2_t _tmp34b = vfma_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34a = vmla_lane_f32(vmul_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x2_t _tmp0 = vmla_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x2_t _tmp1 = vsub_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp2 = vadd_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp3 = vadd_f32(_tmp34b, _tmp34a);
                float32x2_t _tmp4 = vsub_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x2_t _tmp5 = vfma_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x2_t _tmp5 = vmla_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
#else
                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                tmp[0][m][0] = r00 + r40 - 2.5f * r20;
                tmp[0][m][1] = r01 + r41 - 2.5f * r21;
                tmp[1][m][0] = tmp12b0 - tmp12a0;
                tmp[1][m][1] = tmp12b1 - tmp12a1;
                tmp[2][m][0] = tmp12b0 + tmp12a0;
                tmp[2][m][1] = tmp12b1 + tmp12a1;
                tmp[3][m][0] = tmp34b0 + tmp34a0;
                tmp[3][m][1] = tmp34b1 + tmp34a1;
                tmp[4][m][0] = tmp34b0 - tmp34a0;
                tmp[4][m][1] = tmp34b1 - tmp34a1;
                tmp[5][m][0] = r10 + r50 - 2.5f * r30;
                tmp[5][m][1] = r11 + r51 - 2.5f * r31;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);

#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs, 0), _r3, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(_r4, _r2, _coeffs, 2);
                float32x2_t _tmp34a = vfma_laneq_f32(vmul_laneq_f32(_r3, _coeffs, 0), _r1, _coeffs, 1);
                float32x2_t _tmp34b = vfma_laneq_f32(_r4, _r2, _coeffs, 3);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs), 0), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34a = vmla_lane_f32(vmul_lane_f32(_r3, vget_low_f32(_coeffs), 0), _r1, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34b = vmla_lane_f32(_r4, _r2, vget_high_f32(_coeffs), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#else
                float32x2_t _tmp0 = vmla_f32(vadd_f32(_r0, _r4), _r2, _vm2_5);
#endif
                float32x2_t _tmp1 = vsub_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp2 = vadd_f32(_tmp12b, _tmp12a);
                float32x2_t _tmp3 = vadd_f32(_tmp34b, _tmp34a);
                float32x2_t _tmp4 = vsub_f32(_tmp34b, _tmp34a);
#if __aarch64__
                float32x2_t _tmp5 = vfma_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#else
                float32x2_t _tmp5 = vmla_f32(vadd_f32(_r1, _r5), _r3, _vm2_5);
#endif

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
                vst1_f32(p4, _tmp4);
                vst1_f32(p5, _tmp5);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp12a0 = sq2 * r10 - sq2_d2 * r30;
                float tmp12a1 = sq2 * r11 - sq2_d2 * r31;
                float tmp12b0 = r40 - 2 * r20;
                float tmp12b1 = r41 - 2 * r21;
                float tmp34a0 = sq2 * r30 - sq2_d2 * r10;
                float tmp34a1 = sq2 * r31 - sq2_d2 * r11;
                float tmp34b0 = r40 - 0.5f * r20;
                float tmp34b1 = r41 - 0.5f * r21;

                p0[0] = r00 + r40 - 2.5f * r20;
                p0[1] = r01 + r41 - 2.5f * r21;
                p1[0] = tmp12b0 - tmp12a0;
                p1[1] = tmp12b1 - tmp12a1;
                p2[0] = tmp12b0 + tmp12a0;
                p2[1] = tmp12b1 + tmp12a1;
                p3[0] = tmp34b0 + tmp34a0;
                p3[1] = tmp34b1 + tmp34a1;
                p4[0] = tmp34b0 - tmp34a0;
                p4[1] = tmp34b1 - tmp34a1;
                p5[0] = r10 + r50 - 2.5f * r30;
                p5[1] = r11 + r51 - 2.5f * r31;
#endif

                p0 += max_jj * 6 * 2;
                p1 += max_jj * 6 * 2;
                p2 += max_jj * 6 * 2;
                p3 += max_jj * 6 * 2;
                p4 += max_jj * 6 * 2;
                p5 += max_jj * 6 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        float tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0123 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = bfloat16_to_float32(r0123[0]);
                        if (tj * 4 + 1 < w) r1 = bfloat16_to_float32(r0123[1]);
                        if (tj * 4 + 2 < w) r2 = bfloat16_to_float32(r0123[2]);
                        if (tj * 4 + 3 < w) r3 = bfloat16_to_float32(r0123[3]);
                        if (tj * 4 + 4 < w) r4 = bfloat16_to_float32(r0123[4]);
                        if (tj * 4 + 5 < w) r5 = bfloat16_to_float32(r0123[5]);
                    }
                }

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                tmp[0][m] = r0 + r4 - 2.5f * r2;
                tmp[1][m] = tmp12b - tmp12a;
                tmp[2][m] = tmp12b + tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r1 + r5 - 2.5f * r3;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 36 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp12a = sq2 * r1 - sq2_d2 * r3;
                float tmp12b = r4 - 2 * r2;
                float tmp34a = sq2 * r3 - sq2_d2 * r1;
                float tmp34b = r4 - 0.5f * r2;

                p0[0] = r0 + r4 - 2.5f * r2;
                p1[0] = tmp12b - tmp12a;
                p2[0] = tmp12b + tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r1 + r5 - 2.5f * r3;

                p0 += max_jj * 6;
                p1 += max_jj * 6;
                p2 += max_jj * 6;
                p3 += max_jj * 6;
                p4 += max_jj * 6;
                p5 += max_jj * 6;
            }
        }
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_bf16s(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    const float sq2 = 1.41421356237;
    const float sq2_m2 = 1.41421356237 * 2;
    const float sq2_d2 = 1.41421356237 / 2;
    const float sq2_d4 = 1.41421356237 / 4;

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,   1.0f,  1.0f,  1.0f,   0.0f},
    //     {0.0f, sq2/2, -sq2/2, sq2,   -sq2,   0.0f},
    //     {0.0f, 0.5f,   0.5f,  2.0f,  2.0f,   0.0f},
    //     {0.0f, sq2/4, -sq2/4, sq2*2, -sq2*2, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) * sq2_d2 + (r03 - r04) * sq2
    // 2 =       (r01 + r02) * 0.5f + (r03 + r04) * 2
    // 3 = r05 + (r01 - r02) * sq2_d4 + (r03 - r04) * sq2_m2

#if __ARM_NEON
    const float coeffs[6] = {sq2, sq2_d2, sq2_d4, sq2_m2, 0.5f, 2.f};
    float32x4_t _coeffs = vld1q_f32(coeffs);
    float32x2_t _coeffs2 = vld1_f32(coeffs + 4);
#endif // __ARM_NEON

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);

                float32x4_t _tmp02a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp02a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp02b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp02b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp13a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp13a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp13b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp13b1 = vsubq_f32(_r31, _r41);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp02a0), _tmp02b0);
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp02a1), _tmp02b1);
                float32x4_t _tmp10 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a0, _coeffs, 1), _tmp13b0, _coeffs, 0);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a1, _coeffs, 1), _tmp13b1, _coeffs, 0);
                float32x4_t _tmp20 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a0, _coeffs2, 0), _tmp02b0, _coeffs2, 1);
                float32x4_t _tmp21 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a1, _coeffs2, 0), _tmp02b1, _coeffs2, 1);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r50, _tmp13a0, _coeffs, 2), _tmp13b0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r51, _tmp13a1, _coeffs, 2), _tmp13b1, _coeffs, 3);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);

                r0 += max_jj * 6 * 8;
                r1 += max_jj * 6 * 8;
                r2 += max_jj * 6 * 8;
                r3 += max_jj * 6 * 8;
                r4 += max_jj * 6 * 8;
                r5 += max_jj * 6 * 8;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);

                float32x4_t _tmp02a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp02a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp02b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp02b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp13a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp13a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp13b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp13b1 = vsubq_f32(_r31, _r41);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp02a0), vaddq_f32(_tmp02b0, _bias0));
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp02a1), vaddq_f32(_tmp02b1, _bias1));
                float32x4_t _tmp10 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias0, _tmp13a0, _coeffs, 1), _tmp13b0, _coeffs, 0);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias1, _tmp13a1, _coeffs, 1), _tmp13b1, _coeffs, 0);
                float32x4_t _tmp20 = vfmaq_lane_f32(vfmaq_lane_f32(_bias0, _tmp02a0, _coeffs2, 0), _tmp02b0, _coeffs2, 1);
                float32x4_t _tmp21 = vfmaq_lane_f32(vfmaq_lane_f32(_bias1, _tmp02a1, _coeffs2, 0), _tmp02b1, _coeffs2, 1);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r50, _bias0), _tmp13a0, _coeffs, 2), _tmp13b0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r51, _bias1), _tmp13a1, _coeffs, 2), _tmp13b1, _coeffs, 3);

                if (out_elempack == 4)
                {
                    unsigned short* outptr1 = outptr0 + N;

                    vst1_u16(outptr0, float2bfloat(_tmp00));
                    vst1_u16(outptr1, float2bfloat(_tmp01));
                    if (tj * 4 + 1 < outw)
                    {
                        vst1_u16(outptr0 + 4, float2bfloat(_tmp10));
                        vst1_u16(outptr1 + 4, float2bfloat(_tmp11));
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        vst1_u16(outptr0 + 8, float2bfloat(_tmp20));
                        vst1_u16(outptr1 + 8, float2bfloat(_tmp21));
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        vst1_u16(outptr0 + 12, float2bfloat(_tmp30));
                        vst1_u16(outptr1 + 12, float2bfloat(_tmp31));
                    }
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[8];
                    unsigned short tmp1[8];
                    unsigned short tmp2[8];
                    unsigned short tmp3[8];
                    vst1_u16(tmp0, float2bfloat(_tmp00));
                    vst1_u16(tmp0 + 4, float2bfloat(_tmp01));
                    vst1_u16(tmp1, float2bfloat(_tmp10));
                    vst1_u16(tmp1 + 4, float2bfloat(_tmp11));
                    vst1_u16(tmp2, float2bfloat(_tmp20));
                    vst1_u16(tmp2 + 4, float2bfloat(_tmp21));
                    vst1_u16(tmp3, float2bfloat(_tmp30));
                    vst1_u16(tmp3 + 4, float2bfloat(_tmp31));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;
                    unsigned short* outptr4 = outptr0 + N * 4;
                    unsigned short* outptr5 = outptr0 + N * 5;
                    unsigned short* outptr6 = outptr0 + N * 6;
                    unsigned short* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[4][6][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;

            for (int m = 0; m < 6; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);
                float32x4_t _r4 = vld1q_f32(r4);
                float32x4_t _r5 = vld1q_f32(r5);

                float32x4_t _tmp02a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp02b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp13a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp13b = vsubq_f32(_r3, _r4);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp02a), _tmp02b);
#if __aarch64__
                float32x4_t _tmp1 = vfmaq_laneq_f32(vmulq_laneq_f32(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x4_t _tmp2 = vfmaq_lane_f32(vmulq_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x4_t _tmp1 = vmlaq_lane_f32(vmulq_lane_f32(_tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmulq_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(_r5, _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);

                r0 += max_jj * 6 * 4;
                r1 += max_jj * 6 * 4;
                r2 += max_jj * 6 * 4;
                r3 += max_jj * 6 * 4;
                r4 += max_jj * 6 * 4;
                r5 += max_jj * 6 * 4;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 4) + (tj * 4) * out_elempack;

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);

                float32x4_t _tmp02a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp02b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp13a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp13b = vsubq_f32(_r3, _r4);

                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp02a), vaddq_f32(_tmp02b, _bias0));
#if __aarch64__
                float32x4_t _tmp1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x4_t _tmp2 = vfmaq_lane_f32(vfmaq_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(vaddq_f32(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x4_t _tmp1 = vmlaq_lane_f32(vmlaq_lane_f32(_bias0, _tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmlaq_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(vaddq_f32(_r5, _bias0), _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_tmp0));
                    if (tj * 4 + 1 < outw) vst1_u16(outptr0 + 4, float2bfloat(_tmp1));
                    if (tj * 4 + 2 < outw) vst1_u16(outptr0 + 8, float2bfloat(_tmp2));
                    if (tj * 4 + 3 < outw) vst1_u16(outptr0 + 12, float2bfloat(_tmp3));
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[4];
                    unsigned short tmp1[4];
                    unsigned short tmp2[4];
                    unsigned short tmp3[4];
                    vst1_u16(tmp0, float2bfloat(_tmp0));
                    vst1_u16(tmp1, float2bfloat(_tmp1));
                    vst1_u16(tmp2, float2bfloat(_tmp2));
                    vst1_u16(tmp3, float2bfloat(_tmp3));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);
                float32x2_t _r4 = vld1_f32(r4);
                float32x2_t _r5 = vld1_f32(r5);

                float32x2_t _tmp02a = vadd_f32(_r1, _r2);
                float32x2_t _tmp02b = vadd_f32(_r3, _r4);
                float32x2_t _tmp13a = vsub_f32(_r1, _r2);
                float32x2_t _tmp13b = vsub_f32(_r3, _r4);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp02a), _tmp02b);
#if __aarch64__
                float32x2_t _tmp1 = vfma_laneq_f32(vmul_laneq_f32(_tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x2_t _tmp2 = vfma_lane_f32(vmul_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(_r5, _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x2_t _tmp1 = vmla_lane_f32(vmul_lane_f32(_tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x2_t _tmp2 = vmla_lane_f32(vmul_lane_f32(_tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(_r5, _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
#else
                float tmp02a0 = r1[0] + r2[0];
                float tmp02a1 = r1[1] + r2[1];
                float tmp02b0 = r3[0] + r4[0];
                float tmp02b1 = r3[1] + r4[1];
                float tmp13a0 = r1[0] - r2[0];
                float tmp13a1 = r1[1] - r2[1];
                float tmp13b0 = r3[0] - r4[0];
                float tmp13b1 = r3[1] - r4[1];

                tmp[0][m][0] = r0[0] + tmp02a0 + tmp02b0;
                tmp[0][m][1] = r0[1] + tmp02a1 + tmp02b1;
                tmp[1][m][0] = tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                tmp[1][m][1] = tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                tmp[2][m][0] = tmp02a0 * 0.5f + tmp02b0 * 2;
                tmp[2][m][1] = tmp02a1 * 0.5f + tmp02b1 * 2;
                tmp[3][m][0] = r5[0] + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                tmp[3][m][1] = r5[1] + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;
#endif

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);

                float32x2_t _tmp02a = vadd_f32(_r1, _r2);
                float32x2_t _tmp02b = vadd_f32(_r3, _r4);
                float32x2_t _tmp13a = vsub_f32(_r1, _r2);
                float32x2_t _tmp13b = vsub_f32(_r3, _r4);

                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp02a), vadd_f32(_tmp02b, _bias0));
#if __aarch64__
                float32x2_t _tmp1 = vfma_laneq_f32(vfma_laneq_f32(_bias0, _tmp13a, _coeffs, 1), _tmp13b, _coeffs, 0);
                float32x2_t _tmp2 = vfma_lane_f32(vfma_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(vadd_f32(_r5, _bias0), _tmp13a, _coeffs, 2), _tmp13b, _coeffs, 3);
#else
                float32x2_t _tmp1 = vmla_lane_f32(vmla_lane_f32(_bias0, _tmp13a, vget_low_f32(_coeffs), 1), _tmp13b, vget_low_f32(_coeffs), 0);
                float32x2_t _tmp2 = vmla_lane_f32(vmla_lane_f32(_bias0, _tmp02a, _coeffs2, 0), _tmp02b, _coeffs2, 1);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(vadd_f32(_r5, _bias0), _tmp13a, vget_high_f32(_coeffs), 0), _tmp13b, vget_high_f32(_coeffs), 1);
#endif
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];

                float tmp02a0 = r10 + r20;
                float tmp02a1 = r11 + r21;
                float tmp02b0 = r30 + r40;
                float tmp02b1 = r31 + r41;
                float tmp13a0 = r10 - r20;
                float tmp13a1 = r11 - r21;
                float tmp13b0 = r30 - r40;
                float tmp13b1 = r31 - r41;

                float tmp00 = bias0 + r00 + tmp02a0 + tmp02b0;
                float tmp01 = bias1 + r01 + tmp02a1 + tmp02b1;
                float tmp10 = bias0 + tmp13a0 * sq2_d2 + tmp13b0 * sq2;
                float tmp11 = bias1 + tmp13a1 * sq2_d2 + tmp13b1 * sq2;
                float tmp20 = bias0 + tmp02a0 * 0.5f + tmp02b0 * 2;
                float tmp21 = bias1 + tmp02a1 * 0.5f + tmp02b1 * 2;
                float tmp30 = bias0 + r50 + tmp13a0 * sq2_d4 + tmp13b0 * sq2_m2;
                float tmp31 = bias1 + r51 + tmp13a1 * sq2_d4 + tmp13b1 * sq2_m2;
#endif

                // if (out_elempack == 1)
                {
                    unsigned short* outptr1 = outptr0 + N;

#if __ARM_NEON
                    uint16x4_t _tmp01 = float2bfloat(vcombine_f32(_tmp0, _tmp1));
                    uint16x4_t _tmp23 = float2bfloat(vcombine_f32(_tmp2, _tmp3));

                    outptr0[0] = vget_lane_u16(_tmp01, 0);
                    outptr1[0] = vget_lane_u16(_tmp01, 1);
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_u16(_tmp01, 2);
                        outptr1[1] = vget_lane_u16(_tmp01, 3);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = vget_lane_u16(_tmp23, 0);
                        outptr1[2] = vget_lane_u16(_tmp23, 1);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = vget_lane_u16(_tmp23, 2);
                        outptr1[3] = vget_lane_u16(_tmp23, 3);
                    }
#else
                    outptr0[0] = float32_to_bfloat16(tmp00);
                    outptr1[0] = float32_to_bfloat16(tmp01);
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = float32_to_bfloat16(tmp10);
                        outptr1[1] = float32_to_bfloat16(tmp11);
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = float32_to_bfloat16(tmp20);
                        outptr1[2] = float32_to_bfloat16(tmp21);
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = float32_to_bfloat16(tmp30);
                        outptr1[3] = float32_to_bfloat16(tmp31);
                    }
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 36 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                float tmp02a = r1[0] + r2[0];
                float tmp02b = r3[0] + r4[0];
                float tmp13a = r1[0] - r2[0];
                float tmp13b = r3[0] - r4[0];

                tmp[0][m] = r0[0] + tmp02a + tmp02b;
                tmp[1][m] = tmp13a * sq2_d2 + tmp13b * sq2;
                tmp[2][m] = tmp02a * 0.5f + tmp02b * 2;
                tmp[3][m] = r5[0] + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];

                float tmp02a = r1 + r2;
                float tmp02b = r3 + r4;
                float tmp13a = r1 - r2;
                float tmp13b = r3 - r4;

                float tmp0 = bias0 + r0 + tmp02a + tmp02b;
                float tmp1 = bias0 + tmp13a * sq2_d2 + tmp13b * sq2;
                float tmp2 = bias0 + tmp02a * 0.5f + tmp02b * 2;
                float tmp3 = bias0 + r5 + tmp13a * sq2_d4 + tmp13b * sq2_m2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(tmp0);
                    if (tj * 4 + 1 < outw) outptr0[1] = float32_to_bfloat16(tmp1);
                    if (tj * 4 + 2 < outw) outptr0[2] = float32_to_bfloat16(tmp2);
                    if (tj * 4 + 3 < outw) outptr0[3] = float32_to_bfloat16(tmp3);
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd43_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 4n+2, winograd F(4,3)
    int w_tiles = (outw + 3) / 4;
    int h_tiles = (outh + 3) / 4;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    // NCNN_LOGE("conv3x3s1_winograd43_bf16s %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd43_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd63_transform_input_tile_bf16s(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[8][8] = {
    //     {1.0f, 0.0f,-5.25f, 0.00f, 5.25f, 0.00f,-1.0f, 0.0f},
    //     {0.0f, 1.0f, 1.00f,-4.25f,-4.25f, 1.00f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 1.00f, 4.25f,-4.25f,-1.00f, 1.0f, 0.0f},
    //     {0.0f, 0.5f, 0.25f,-2.50f,-1.25f, 2.00f, 1.0f, 0.0f},
    //     {0.0f,-0.5f, 0.25f, 2.50f,-1.25f,-2.00f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, 4.00f,-2.50f,-5.00f, 0.50f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, 4.00f, 2.50f,-5.00f,-0.50f, 1.0f, 0.0f},
    //     {0.0f,-1.0f, 0.00f, 5.25f, 0.00f,-5.25f, 0.0f, 1.0f}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 3) / 6;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
#if __ARM_NEON
#if __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 8;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 8;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[8][8][8];

        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vdupq_n_f32(0.f);
                float32x4_t _r01 = vdupq_n_f32(0.f);
                float32x4_t _r10 = vdupq_n_f32(0.f);
                float32x4_t _r11 = vdupq_n_f32(0.f);
                float32x4_t _r20 = vdupq_n_f32(0.f);
                float32x4_t _r21 = vdupq_n_f32(0.f);
                float32x4_t _r30 = vdupq_n_f32(0.f);
                float32x4_t _r31 = vdupq_n_f32(0.f);
                float32x4_t _r40 = vdupq_n_f32(0.f);
                float32x4_t _r41 = vdupq_n_f32(0.f);
                float32x4_t _r50 = vdupq_n_f32(0.f);
                float32x4_t _r51 = vdupq_n_f32(0.f);
                float32x4_t _r60 = vdupq_n_f32(0.f);
                float32x4_t _r61 = vdupq_n_f32(0.f);
                float32x4_t _r70 = vdupq_n_f32(0.f);
                float32x4_t _r71 = vdupq_n_f32(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        const unsigned short* r1 = r0 + N;

                        _r00 = bfloat2float(vld1_u16(r0));
                        _r01 = bfloat2float(vld1_u16(r1));
                        if (tj * 6 + 1 < w)
                        {
                            _r10 = bfloat2float(vld1_u16(r0 + 4));
                            _r11 = bfloat2float(vld1_u16(r1 + 4));
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r20 = bfloat2float(vld1_u16(r0 + 8));
                            _r21 = bfloat2float(vld1_u16(r1 + 8));
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r30 = bfloat2float(vld1_u16(r0 + 12));
                            _r31 = bfloat2float(vld1_u16(r1 + 12));
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _r40 = bfloat2float(vld1_u16(r0 + 16));
                            _r41 = bfloat2float(vld1_u16(r1 + 16));
                        }
                        if (tj * 6 + 5 < w)
                        {
                            _r50 = bfloat2float(vld1_u16(r0 + 20));
                            _r51 = bfloat2float(vld1_u16(r1 + 20));
                        }
                        if (tj * 6 + 6 < w)
                        {
                            _r60 = bfloat2float(vld1_u16(r0 + 24));
                            _r61 = bfloat2float(vld1_u16(r1 + 24));
                        }
                        if (tj * 6 + 7 < w)
                        {
                            _r70 = bfloat2float(vld1_u16(r0 + 28));
                            _r71 = bfloat2float(vld1_u16(r1 + 28));
                        }
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;
                        const unsigned short* r4 = r0 + N * 4;
                        const unsigned short* r5 = r0 + N * 5;
                        const unsigned short* r6 = r0 + N * 6;
                        const unsigned short* r7 = r0 + N * 7;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);
                        uint16x4_t _t4 = vld1_u16(r4);
                        uint16x4_t _t5 = vld1_u16(r5);
                        uint16x4_t _t6 = vld1_u16(r6);
                        uint16x4_t _t7 = vld1_u16(r7);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);
                        transpose4x4_u16(_t4, _t5, _t6, _t7);

                        _r00 = bfloat2float(_t0);
                        _r01 = bfloat2float(_t4);
                        if (tj * 6 + 1 < w)
                        {
                            _r10 = bfloat2float(_t1);
                            _r11 = bfloat2float(_t5);
                        }
                        if (tj * 6 + 2 < w)
                        {
                            _r20 = bfloat2float(_t2);
                            _r21 = bfloat2float(_t6);
                        }
                        if (tj * 6 + 3 < w)
                        {
                            _r30 = bfloat2float(_t3);
                            _r31 = bfloat2float(_t7);
                        }
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1_u16(r0 + 4);
                            _t1 = vld1_u16(r1 + 4);
                            _t2 = vld1_u16(r2 + 4);
                            _t3 = vld1_u16(r3 + 4);
                            _t4 = vld1_u16(r4 + 4);
                            _t5 = vld1_u16(r5 + 4);
                            _t6 = vld1_u16(r6 + 4);
                            _t7 = vld1_u16(r7 + 4);

                            transpose4x4_u16(_t0, _t1, _t2, _t3);
                            transpose4x4_u16(_t4, _t5, _t6, _t7);

                            _r40 = bfloat2float(_t0);
                            _r41 = bfloat2float(_t4);
                            if (tj * 6 + 5 < w)
                            {
                                _r50 = bfloat2float(_t1);
                                _r51 = bfloat2float(_t5);
                            }
                            if (tj * 6 + 6 < w)
                            {
                                _r60 = bfloat2float(_t2);
                                _r61 = bfloat2float(_t6);
                            }
                            if (tj * 6 + 7 < w)
                            {
                                _r70 = bfloat2float(_t3);
                                _r71 = bfloat2float(_t7);
                            }
                        }
                    }
                }

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vaddq_f32(_r20, _r60), _r40, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vaddq_f32(_r21, _r61), _r41, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(vaddq_f32(_r10, _r50), _r30, _coeffs, 1);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(vaddq_f32(_r11, _r51), _r31, _coeffs, 1);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r60, _r20, _coeffs, 3), _r40, _coeffs, 2);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r61, _r21, _coeffs, 3), _r41, _coeffs, 2);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 1), _r30, _coeffs2, 0), _r50, _coeffs2, 2);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 1), _r31, _coeffs2, 0), _r51, _coeffs2, 2);
                float32x4_t _tmp56a0 = vfmaq_laneq_f32(_r60, vfmaq_laneq_f32(_r20, _r40, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56a1 = vfmaq_laneq_f32(_r61, vfmaq_laneq_f32(_r21, _r41, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 2), _r30, _coeffs2, 0), _r50, _coeffs2, 1);
                float32x4_t _tmp56b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 2), _r31, _coeffs2, 0), _r51, _coeffs2, 1);

                float32x4_t _tmp00 = vfmaq_laneq_f32(vsubq_f32(_r00, _r60), vsubq_f32(_r40, _r20), _coeffs, 0);
                float32x4_t _tmp01 = vfmaq_laneq_f32(vsubq_f32(_r01, _r61), vsubq_f32(_r41, _r21), _coeffs, 0);
                float32x4_t _tmp10 = vaddq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp11 = vaddq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp20 = vsubq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp21 = vsubq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp50 = vaddq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp51 = vaddq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp60 = vsubq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp61 = vsubq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp70 = vfmaq_laneq_f32(vsubq_f32(_r70, _r10), vsubq_f32(_r30, _r50), _coeffs, 0);
                float32x4_t _tmp71 = vfmaq_laneq_f32(vsubq_f32(_r71, _r11), vsubq_f32(_r31, _r51), _coeffs, 0);

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);
                vst1q_f32(tmp[6][m], _tmp60);
                vst1q_f32(tmp[6][m] + 4, _tmp61);
                vst1q_f32(tmp[7][m], _tmp70);
                vst1q_f32(tmp[7][m] + 4, _tmp71);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 8;
            float* p1 = p0 + max_jj * 8;
            float* p2 = p0 + max_jj * 8 * 2;
            float* p3 = p0 + max_jj * 8 * 3;
            float* p4 = p0 + max_jj * 8 * 4;
            float* p5 = p0 + max_jj * 8 * 5;
            float* p6 = p0 + max_jj * 8 * 6;
            float* p7 = p0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);
                float32x4_t _r60 = vld1q_f32(tmp[m][6]);
                float32x4_t _r61 = vld1q_f32(tmp[m][6] + 4);
                float32x4_t _r70 = vld1q_f32(tmp[m][7]);
                float32x4_t _r71 = vld1q_f32(tmp[m][7] + 4);

                float32x4_t _tmp12a0 = vfmaq_laneq_f32(vaddq_f32(_r20, _r60), _r40, _coeffs, 1);
                float32x4_t _tmp12a1 = vfmaq_laneq_f32(vaddq_f32(_r21, _r61), _r41, _coeffs, 1);
                float32x4_t _tmp12b0 = vfmaq_laneq_f32(vaddq_f32(_r10, _r50), _r30, _coeffs, 1);
                float32x4_t _tmp12b1 = vfmaq_laneq_f32(vaddq_f32(_r11, _r51), _r31, _coeffs, 1);
                float32x4_t _tmp34a0 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r60, _r20, _coeffs, 3), _r40, _coeffs, 2);
                float32x4_t _tmp34a1 = vfmaq_laneq_f32(vfmaq_laneq_f32(_r61, _r21, _coeffs, 3), _r41, _coeffs, 2);
                float32x4_t _tmp34b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 1), _r30, _coeffs2, 0), _r50, _coeffs2, 2);
                float32x4_t _tmp34b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 1), _r31, _coeffs2, 0), _r51, _coeffs2, 2);
                float32x4_t _tmp56a0 = vfmaq_laneq_f32(_r60, vfmaq_laneq_f32(_r20, _r40, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56a1 = vfmaq_laneq_f32(_r61, vfmaq_laneq_f32(_r21, _r41, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b0 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r10, _coeffs2, 2), _r30, _coeffs2, 0), _r50, _coeffs2, 1);
                float32x4_t _tmp56b1 = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r11, _coeffs2, 2), _r31, _coeffs2, 0), _r51, _coeffs2, 1);

                float32x4_t _tmp00 = vfmaq_laneq_f32(vsubq_f32(_r00, _r60), vsubq_f32(_r40, _r20), _coeffs, 0);
                float32x4_t _tmp01 = vfmaq_laneq_f32(vsubq_f32(_r01, _r61), vsubq_f32(_r41, _r21), _coeffs, 0);
                float32x4_t _tmp10 = vaddq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp11 = vaddq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp20 = vsubq_f32(_tmp12a0, _tmp12b0);
                float32x4_t _tmp21 = vsubq_f32(_tmp12a1, _tmp12b1);
                float32x4_t _tmp30 = vaddq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp31 = vaddq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp40 = vsubq_f32(_tmp34a0, _tmp34b0);
                float32x4_t _tmp41 = vsubq_f32(_tmp34a1, _tmp34b1);
                float32x4_t _tmp50 = vaddq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp51 = vaddq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp60 = vsubq_f32(_tmp56a0, _tmp56b0);
                float32x4_t _tmp61 = vsubq_f32(_tmp56a1, _tmp56b1);
                float32x4_t _tmp70 = vfmaq_laneq_f32(vsubq_f32(_r70, _r10), vsubq_f32(_r30, _r50), _coeffs, 0);
                float32x4_t _tmp71 = vfmaq_laneq_f32(vsubq_f32(_r71, _r11), vsubq_f32(_r31, _r51), _coeffs, 0);

                vst1q_f32(p0, _tmp00);
                vst1q_f32(p0 + 4, _tmp01);
                vst1q_f32(p1, _tmp10);
                vst1q_f32(p1 + 4, _tmp11);
                vst1q_f32(p2, _tmp20);
                vst1q_f32(p2 + 4, _tmp21);
                vst1q_f32(p3, _tmp30);
                vst1q_f32(p3 + 4, _tmp31);
                vst1q_f32(p4, _tmp40);
                vst1q_f32(p4 + 4, _tmp41);
                vst1q_f32(p5, _tmp50);
                vst1q_f32(p5 + 4, _tmp51);
                vst1q_f32(p6, _tmp60);
                vst1q_f32(p6 + 4, _tmp61);
                vst1q_f32(p7, _tmp70);
                vst1q_f32(p7 + 4, _tmp71);

                p0 += max_jj * 8 * 8;
                p1 += max_jj * 8 * 8;
                p2 += max_jj * 8 * 8;
                p3 += max_jj * 8 * 8;
                p4 += max_jj * 8 * 8;
                p5 += max_jj * 8 * 8;
                p6 += max_jj * 8 * 8;
                p7 += max_jj * 8 * 8;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 8;
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
#else // __aarch64__
    nn_max_kk = (max_kk - remain_max_kk_start) / 4;
    #pragma omp parallel for num_threads(nT)
#endif // __aarch64__
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 4;

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[8][8][4];

        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel((k + kk) / elempack).row<const unsigned short>(ti * 6) + (tj * 6) * elempack;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vdupq_n_f32(0.f);
                float32x4_t _r1 = vdupq_n_f32(0.f);
                float32x4_t _r2 = vdupq_n_f32(0.f);
                float32x4_t _r3 = vdupq_n_f32(0.f);
                float32x4_t _r4 = vdupq_n_f32(0.f);
                float32x4_t _r5 = vdupq_n_f32(0.f);
                float32x4_t _r6 = vdupq_n_f32(0.f);
                float32x4_t _r7 = vdupq_n_f32(0.f);

                if (ti * 6 + m < h)
                {
                    if (elempack == 4)
                    {
                        _r0 = bfloat2float(vld1_u16(r0));
                        if (tj * 6 + 1 < w) _r1 = bfloat2float(vld1_u16(r0 + 4));
                        if (tj * 6 + 2 < w) _r2 = bfloat2float(vld1_u16(r0 + 8));
                        if (tj * 6 + 3 < w) _r3 = bfloat2float(vld1_u16(r0 + 12));
                        if (tj * 6 + 4 < w) _r4 = bfloat2float(vld1_u16(r0 + 16));
                        if (tj * 6 + 5 < w) _r5 = bfloat2float(vld1_u16(r0 + 20));
                        if (tj * 6 + 6 < w) _r6 = bfloat2float(vld1_u16(r0 + 24));
                        if (tj * 6 + 7 < w) _r7 = bfloat2float(vld1_u16(r0 + 28));
                    }
                    if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;
                        const unsigned short* r2 = r0 + N * 2;
                        const unsigned short* r3 = r0 + N * 3;

                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4_t _t2 = vld1_u16(r2);
                        uint16x4_t _t3 = vld1_u16(r3);

                        transpose4x4_u16(_t0, _t1, _t2, _t3);

                        _r0 = bfloat2float(_t0);
                        if (tj * 6 + 1 < w) _r1 = bfloat2float(_t1);
                        if (tj * 6 + 2 < w) _r2 = bfloat2float(_t2);
                        if (tj * 6 + 3 < w) _r3 = bfloat2float(_t3);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1_u16(r0 + 4);
                            _t1 = vld1_u16(r1 + 4);
                            _t2 = vld1_u16(r2 + 4);
                            _t3 = vld1_u16(r3 + 4);

                            transpose4x4_u16(_t0, _t1, _t2, _t3);

                            _r4 = bfloat2float(_t0);
                            if (tj * 6 + 5 < w) _r5 = bfloat2float(_t1);
                            if (tj * 6 + 6 < w) _r6 = bfloat2float(_t2);
                            if (tj * 6 + 7 < w) _r7 = bfloat2float(_t3);
                        }
                    }
                }

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vaddq_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(vaddq_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vfmaq_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x4_t _tmp34b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x4_t _tmp56a = vfmaq_laneq_f32(_r6, vfmaq_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vaddq_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(vaddq_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmlaq_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x4_t _tmp56a = vmlaq_lane_f32(_r6, vmlaq_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x4_t _tmp56b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_laneq_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), _coeffs, 0);
#else
                float32x4_t _tmp0 = vmlaq_lane_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x4_t _tmp7 = vfmaq_laneq_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), _coeffs, 0);
#else
                float32x4_t _tmp7 = vmlaq_lane_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);
                vst1q_f32(tmp[6][m], _tmp6);
                vst1q_f32(tmp[7][m], _tmp7);

                r0 += w * elempack;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 4;
            float* p1 = p0 + max_jj * 4;
            float* p2 = p0 + max_jj * 4 * 2;
            float* p3 = p0 + max_jj * 4 * 3;
            float* p4 = p0 + max_jj * 4 * 4;
            float* p5 = p0 + max_jj * 4 * 5;
            float* p6 = p0 + max_jj * 4 * 6;
            float* p7 = p0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);
                float32x4_t _r6 = vld1q_f32(tmp[m][6]);
                float32x4_t _r7 = vld1q_f32(tmp[m][7]);

#if __aarch64__
                float32x4_t _tmp12a = vfmaq_laneq_f32(vaddq_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x4_t _tmp12b = vfmaq_laneq_f32(vaddq_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x4_t _tmp34a = vfmaq_laneq_f32(vfmaq_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x4_t _tmp34b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x4_t _tmp56a = vfmaq_laneq_f32(_r6, vfmaq_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x4_t _tmp56b = vfmaq_laneq_f32(vfmaq_laneq_f32(vmulq_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x4_t _tmp12a = vmlaq_lane_f32(vaddq_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp12b = vmlaq_lane_f32(vaddq_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp34a = vmlaq_lane_f32(vmlaq_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp34b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x4_t _tmp56a = vmlaq_lane_f32(_r6, vmlaq_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x4_t _tmp56b = vmlaq_lane_f32(vmlaq_lane_f32(vmulq_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x4_t _tmp0 = vfmaq_laneq_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), _coeffs, 0);
#else
                float32x4_t _tmp0 = vmlaq_lane_f32(vsubq_f32(_r0, _r6), vsubq_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);
                float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);
                float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x4_t _tmp7 = vfmaq_laneq_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), _coeffs, 0);
#else
                float32x4_t _tmp7 = vmlaq_lane_f32(vsubq_f32(_r7, _r1), vsubq_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1q_f32(p0, _tmp0);
                vst1q_f32(p1, _tmp1);
                vst1q_f32(p2, _tmp2);
                vst1q_f32(p3, _tmp3);
                vst1q_f32(p4, _tmp4);
                vst1q_f32(p5, _tmp5);
                vst1q_f32(p6, _tmp6);
                vst1q_f32(p7, _tmp7);

                p0 += max_jj * 8 * 4;
                p1 += max_jj * 8 * 4;
                p2 += max_jj * 8 * 4;
                p3 += max_jj * 8 * 4;
                p4 += max_jj * 8 * 4;
                p5 += max_jj * 8 * 4;
                p6 += max_jj * 8 * 4;
                p7 += max_jj * 8 * 4;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 4;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
#else // __ARM_NEON
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
#endif // __ARM_NEON
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[8][8][2];

#if __ARM_NEON
        const float coeffs[8] = {5.25f, -4.25f, -1.25f, 0.25f, -2.5f, 0.5f, 2.f, 4.f};
        float32x4_t _coeffs = vld1q_f32(coeffs);
        float32x4_t _coeffs2 = vld1q_f32(coeffs + 4);
#endif

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vdup_n_f32(0.f);
                float32x2_t _r1 = vdup_n_f32(0.f);
                float32x2_t _r2 = vdup_n_f32(0.f);
                float32x2_t _r3 = vdup_n_f32(0.f);
                float32x2_t _r4 = vdup_n_f32(0.f);
                float32x2_t _r5 = vdup_n_f32(0.f);
                float32x2_t _r6 = vdup_n_f32(0.f);
                float32x2_t _r7 = vdup_n_f32(0.f);
#else
                float r00 = 0.f;
                float r01 = 0.f;
                float r10 = 0.f;
                float r11 = 0.f;
                float r20 = 0.f;
                float r21 = 0.f;
                float r30 = 0.f;
                float r31 = 0.f;
                float r40 = 0.f;
                float r41 = 0.f;
                float r50 = 0.f;
                float r51 = 0.f;
                float r60 = 0.f;
                float r61 = 0.f;
                float r70 = 0.f;
                float r71 = 0.f;
#endif

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const unsigned short* r1 = r0 + N;

#if __ARM_NEON
                        uint16x4_t _t0 = vld1_u16(r0);
                        uint16x4_t _t1 = vld1_u16(r1);
                        uint16x4x2_t _t01 = vzip_u16(_t0, _t1);
                        float32x4_t _t0_fp32 = bfloat2float(_t01.val[0]);
                        float32x4_t _t1_fp32 = bfloat2float(_t01.val[1]);

                        _r0 = vget_low_f32(_t0_fp32);
                        if (tj * 6 + 1 < w) _r1 = vget_high_f32(_t0_fp32);
                        if (tj * 6 + 2 < w) _r2 = vget_low_f32(_t1_fp32);
                        if (tj * 6 + 3 < w) _r3 = vget_high_f32(_t1_fp32);
                        if (tj * 6 + 4 < w)
                        {
                            _t0 = vld1_u16(r0 + 4);
                            _t1 = vld1_u16(r1 + 4);
                            _t01 = vzip_u16(_t0, _t1);
                            _t0_fp32 = bfloat2float(_t01.val[0]);
                            _t1_fp32 = bfloat2float(_t01.val[1]);

                            _r4 = vget_low_f32(_t0_fp32);
                            if (tj * 6 + 5 < w) _r5 = vget_high_f32(_t0_fp32);
                            if (tj * 6 + 6 < w) _r6 = vget_low_f32(_t1_fp32);
                            if (tj * 6 + 7 < w) _r7 = vget_high_f32(_t1_fp32);
                        }
#else
                        r00 = bfloat16_to_float32(r0[0]);
                        r01 = bfloat16_to_float32(r1[0]);
                        if (tj * 6 + 1 < w)
                        {
                            r10 = bfloat16_to_float32(r0[1]);
                            r11 = bfloat16_to_float32(r1[1]);
                        }
                        if (tj * 6 + 2 < w)
                        {
                            r20 = bfloat16_to_float32(r0[2]);
                            r21 = bfloat16_to_float32(r1[2]);
                        }
                        if (tj * 6 + 3 < w)
                        {
                            r30 = bfloat16_to_float32(r0[3]);
                            r31 = bfloat16_to_float32(r1[3]);
                        }
                        if (tj * 6 + 4 < w)
                        {
                            r40 = bfloat16_to_float32(r0[4]);
                            r41 = bfloat16_to_float32(r1[4]);
                        }
                        if (tj * 6 + 5 < w)
                        {
                            r50 = bfloat16_to_float32(r0[5]);
                            r51 = bfloat16_to_float32(r1[5]);
                        }
                        if (tj * 6 + 6 < w)
                        {
                            r60 = bfloat16_to_float32(r0[6]);
                            r61 = bfloat16_to_float32(r1[6]);
                        }
                        if (tj * 6 + 7 < w)
                        {
                            r70 = bfloat16_to_float32(r0[7]);
                            r71 = bfloat16_to_float32(r1[7]);
                        }
#endif
                    }
                }

#if __ARM_NEON
#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vadd_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(vadd_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x2_t _tmp34a = vfma_laneq_f32(vfma_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x2_t _tmp34b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x2_t _tmp56a = vfma_laneq_f32(_r6, vfma_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x2_t _tmp56b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vadd_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(vadd_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34a = vmla_lane_f32(vmla_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x2_t _tmp56a = vmla_lane_f32(_r6, vmla_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x2_t _tmp56b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_laneq_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), _coeffs, 0);
#else
                float32x2_t _tmp0 = vmla_lane_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x2_t _tmp1 = vadd_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp2 = vsub_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp3 = vadd_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp4 = vsub_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp5 = vadd_f32(_tmp56a, _tmp56b);
                float32x2_t _tmp6 = vsub_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x2_t _tmp7 = vfma_laneq_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), _coeffs, 0);
#else
                float32x2_t _tmp7 = vmla_lane_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
                vst1_f32(tmp[6][m], _tmp6);
                vst1_f32(tmp[7][m], _tmp7);
#else
                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                tmp[0][m][0] = r00 - r60 + (r40 - r20) * 5.25f;
                tmp[0][m][1] = r01 - r61 + (r41 - r21) * 5.25f;
                tmp[1][m][0] = tmp12a0 + tmp12b0;
                tmp[1][m][1] = tmp12a1 + tmp12b1;
                tmp[2][m][0] = tmp12a0 - tmp12b0;
                tmp[2][m][1] = tmp12a1 - tmp12b1;
                tmp[3][m][0] = tmp34a0 + tmp34b0;
                tmp[3][m][1] = tmp34a1 + tmp34b1;
                tmp[4][m][0] = tmp34a0 - tmp34b0;
                tmp[4][m][1] = tmp34a1 - tmp34b1;
                tmp[5][m][0] = tmp56a0 + tmp56b0;
                tmp[5][m][1] = tmp56a1 + tmp56b1;
                tmp[6][m][0] = tmp56a0 - tmp56b0;
                tmp[6][m][1] = tmp56a1 - tmp56b1;
                tmp[7][m][0] = r70 - r10 + (r30 - r50) * 5.25f;
                tmp[7][m][1] = r71 - r11 + (r31 - r51) * 5.25f;
#endif

                r0 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj * 2;
            float* p1 = p0 + max_jj * 2;
            float* p2 = p0 + max_jj * 2 * 2;
            float* p3 = p0 + max_jj * 2 * 3;
            float* p4 = p0 + max_jj * 2 * 4;
            float* p5 = p0 + max_jj * 2 * 5;
            float* p6 = p0 + max_jj * 2 * 6;
            float* p7 = p0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);
                float32x2_t _r6 = vld1_f32(tmp[m][6]);
                float32x2_t _r7 = vld1_f32(tmp[m][7]);

#if __aarch64__
                float32x2_t _tmp12a = vfma_laneq_f32(vadd_f32(_r2, _r6), _r4, _coeffs, 1);
                float32x2_t _tmp12b = vfma_laneq_f32(vadd_f32(_r1, _r5), _r3, _coeffs, 1);
                float32x2_t _tmp34a = vfma_laneq_f32(vfma_laneq_f32(_r6, _r2, _coeffs, 3), _r4, _coeffs, 2);
                float32x2_t _tmp34b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 1), _r3, _coeffs2, 0), _r5, _coeffs2, 2);
                float32x2_t _tmp56a = vfma_laneq_f32(_r6, vfma_laneq_f32(_r2, _r4, _coeffs, 2), _coeffs2, 3);
                float32x2_t _tmp56b = vfma_laneq_f32(vfma_laneq_f32(vmul_laneq_f32(_r1, _coeffs2, 2), _r3, _coeffs2, 0), _r5, _coeffs2, 1);
#else
                float32x2_t _tmp12a = vmla_lane_f32(vadd_f32(_r2, _r6), _r4, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp12b = vmla_lane_f32(vadd_f32(_r1, _r5), _r3, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp34a = vmla_lane_f32(vmla_lane_f32(_r6, _r2, vget_high_f32(_coeffs), 1), _r4, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp34b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_low_f32(_coeffs2), 1), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_high_f32(_coeffs2), 0);
                float32x2_t _tmp56a = vmla_lane_f32(_r6, vmla_lane_f32(_r2, _r4, vget_high_f32(_coeffs), 0), vget_high_f32(_coeffs2), 1);
                float32x2_t _tmp56b = vmla_lane_f32(vmla_lane_f32(vmul_lane_f32(_r1, vget_high_f32(_coeffs2), 0), _r3, vget_low_f32(_coeffs2), 0), _r5, vget_low_f32(_coeffs2), 1);
#endif

#if __aarch64__
                float32x2_t _tmp0 = vfma_laneq_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), _coeffs, 0);
#else
                float32x2_t _tmp0 = vmla_lane_f32(vsub_f32(_r0, _r6), vsub_f32(_r4, _r2), vget_low_f32(_coeffs), 0);
#endif
                float32x2_t _tmp1 = vadd_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp2 = vsub_f32(_tmp12a, _tmp12b);
                float32x2_t _tmp3 = vadd_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp4 = vsub_f32(_tmp34a, _tmp34b);
                float32x2_t _tmp5 = vadd_f32(_tmp56a, _tmp56b);
                float32x2_t _tmp6 = vsub_f32(_tmp56a, _tmp56b);
#if __aarch64__
                float32x2_t _tmp7 = vfma_laneq_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), _coeffs, 0);
#else
                float32x2_t _tmp7 = vmla_lane_f32(vsub_f32(_r7, _r1), vsub_f32(_r3, _r5), vget_low_f32(_coeffs), 0);
#endif

                vst1_f32(p0, _tmp0);
                vst1_f32(p1, _tmp1);
                vst1_f32(p2, _tmp2);
                vst1_f32(p3, _tmp3);
                vst1_f32(p4, _tmp4);
                vst1_f32(p5, _tmp5);
                vst1_f32(p6, _tmp6);
                vst1_f32(p7, _tmp7);
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp12a0 = r20 + r60 - r40 * 4.25f;
                float tmp12a1 = r21 + r61 - r41 * 4.25f;
                float tmp12b0 = r10 + r50 - r30 * 4.25f;
                float tmp12b1 = r11 + r51 - r31 * 4.25f;
                float tmp34a0 = r60 + r20 * 0.25f - r40 * 1.25f;
                float tmp34a1 = r61 + r21 * 0.25f - r41 * 1.25f;
                float tmp34b0 = r10 * 0.5f - r30 * 2.5f + r50 * 2.f;
                float tmp34b1 = r11 * 0.5f - r31 * 2.5f + r51 * 2.f;
                float tmp56a0 = r20 * 4.f - r40 * 5.f + r60;
                float tmp56a1 = r21 * 4.f - r41 * 5.f + r61;
                float tmp56b0 = r10 * 2.f - r30 * 2.5f + r50 * 0.5f;
                float tmp56b1 = r11 * 2.f - r31 * 2.5f + r51 * 0.5f;

                p0[0] = r00 - r60 + (r40 - r20) * 5.25f;
                p0[1] = r01 - r61 + (r41 - r21) * 5.25f;
                p1[0] = tmp12a0 + tmp12b0;
                p1[1] = tmp12a1 + tmp12b1;
                p2[0] = tmp12a0 - tmp12b0;
                p2[1] = tmp12a1 - tmp12b1;
                p3[0] = tmp34a0 + tmp34b0;
                p3[1] = tmp34a1 + tmp34b1;
                p4[0] = tmp34a0 - tmp34b0;
                p4[1] = tmp34a1 - tmp34b1;
                p5[0] = tmp56a0 + tmp56b0;
                p5[1] = tmp56a1 + tmp56b1;
                p6[0] = tmp56a0 - tmp56b0;
                p6[1] = tmp56a1 - tmp56b1;
                p7[0] = r70 - r10 + (r30 - r50) * 5.25f;
                p7[1] = r71 - r11 + (r31 - r51) * 5.25f;
#endif

                p0 += max_jj * 8 * 2;
                p1 += max_jj * 8 * 2;
                p2 += max_jj * 8 * 2;
                p3 += max_jj * 8 * 2;
                p4 += max_jj * 8 * 2;
                p5 += max_jj * 8 * 2;
                p6 += max_jj * 8 * 2;
                p7 += max_jj * 8 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        float tmp[8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const unsigned short* r0123 = bottom_blob.channel(k + kk).row<const unsigned short>(ti * 6) + (tj * 6);

            for (int m = 0; m < 8; m++)
            {
                float r0 = 0.f;
                float r1 = 0.f;
                float r2 = 0.f;
                float r3 = 0.f;
                float r4 = 0.f;
                float r5 = 0.f;
                float r6 = 0.f;
                float r7 = 0.f;

                if (ti * 6 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = bfloat16_to_float32(r0123[0]);
                        if (tj * 6 + 1 < w) r1 = bfloat16_to_float32(r0123[1]);
                        if (tj * 6 + 2 < w) r2 = bfloat16_to_float32(r0123[2]);
                        if (tj * 6 + 3 < w) r3 = bfloat16_to_float32(r0123[3]);
                        if (tj * 6 + 4 < w) r4 = bfloat16_to_float32(r0123[4]);
                        if (tj * 6 + 5 < w) r5 = bfloat16_to_float32(r0123[5]);
                        if (tj * 6 + 6 < w) r6 = bfloat16_to_float32(r0123[6]);
                        if (tj * 6 + 7 < w) r7 = bfloat16_to_float32(r0123[7]);
                    }
                }

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                tmp[0][m] = r0 - r6 + (r4 - r2) * 5.25f;
                tmp[1][m] = tmp12a + tmp12b;
                tmp[2][m] = tmp12a - tmp12b;
                tmp[3][m] = tmp34a + tmp34b;
                tmp[4][m] = tmp34a - tmp34b;
                tmp[5][m] = tmp56a + tmp56b;
                tmp[6][m] = tmp56a - tmp56b;
                tmp[7][m] = r7 - r1 + (r3 - r5) * 5.25f;

                r0123 += w;
            }

            float* p0 = (float*)B + kk * max_jj * 64 + jj;
            float* p1 = p0 + max_jj;
            float* p2 = p0 + max_jj * 2;
            float* p3 = p0 + max_jj * 3;
            float* p4 = p0 + max_jj * 4;
            float* p5 = p0 + max_jj * 5;
            float* p6 = p0 + max_jj * 6;
            float* p7 = p0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp12a = r2 + r6 - r4 * 4.25f;
                float tmp12b = r1 + r5 - r3 * 4.25f;
                float tmp34a = r6 + r2 * 0.25f - r4 * 1.25f;
                float tmp34b = r1 * 0.5f - r3 * 2.5f + r5 * 2.f;
                float tmp56a = r2 * 4.f - r4 * 5.f + r6;
                float tmp56b = r1 * 2.f - r3 * 2.5f + r5 * 0.5f;

                p0[0] = r0 - r6 + (r4 - r2) * 5.25f;
                p1[0] = tmp12a + tmp12b;
                p2[0] = tmp12a - tmp12b;
                p3[0] = tmp34a + tmp34b;
                p4[0] = tmp34a - tmp34b;
                p5[0] = tmp56a + tmp56b;
                p6[0] = tmp56a - tmp56b;
                p7[0] = r7 - r1 + (r3 - r5) * 5.25f;

                p0 += max_jj * 8;
                p1 += max_jj * 8;
                p2 += max_jj * 8;
                p3 += max_jj * 8;
                p4 += max_jj * 8;
                p5 += max_jj * 8;
                p6 += max_jj * 8;
                p7 += max_jj * 8;
            }
        }
    }
}

static inline void conv3x3s1_winograd63_transform_output_tile_bf16s(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj)
{
    // const float otm[6][8] = {
    //     {1.0f, 1.0f,  1.0f,  1.0f,  1.0f, 32.0f, 32.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 16.0f,-16.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f,  4.0f,  4.0f,  8.0f,  8.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f,  8.0f, -8.0f,  4.0f, -4.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 16.0f, 16.0f,  2.0f,  2.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 32.0f,-32.0f,  1.0f, -1.0f, 1.0f}
    // };

#if __ARM_NEON
    const float coeffs[4] = {32.f, 16.f, 8.f, 4.f};
    float32x4_t _coeffs = vld1q_f32(coeffs);
    float32x2_t _v2 = vdup_n_f32(2.f);
#endif

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 5) / 6;

    const float* biasptr = bias;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = biasptr ? vld1q_f32(biasptr + i + ii + 4) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][8][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 8;
            const float* r1 = r0 + max_jj * 8;
            const float* r2 = r0 + max_jj * 8 * 2;
            const float* r3 = r0 + max_jj * 8 * 3;
            const float* r4 = r0 + max_jj * 8 * 4;
            const float* r5 = r0 + max_jj * 8 * 5;
            const float* r6 = r0 + max_jj * 8 * 6;
            const float* r7 = r0 + max_jj * 8 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r00 = vld1q_f32(r0);
                float32x4_t _r01 = vld1q_f32(r0 + 4);
                float32x4_t _r10 = vld1q_f32(r1);
                float32x4_t _r11 = vld1q_f32(r1 + 4);
                float32x4_t _r20 = vld1q_f32(r2);
                float32x4_t _r21 = vld1q_f32(r2 + 4);
                float32x4_t _r30 = vld1q_f32(r3);
                float32x4_t _r31 = vld1q_f32(r3 + 4);
                float32x4_t _r40 = vld1q_f32(r4);
                float32x4_t _r41 = vld1q_f32(r4 + 4);
                float32x4_t _r50 = vld1q_f32(r5);
                float32x4_t _r51 = vld1q_f32(r5 + 4);
                float32x4_t _r60 = vld1q_f32(r6);
                float32x4_t _r61 = vld1q_f32(r6 + 4);
                float32x4_t _r70 = vld1q_f32(r7);
                float32x4_t _r71 = vld1q_f32(r7 + 4);

                float32x4_t _tmp024a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp024a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp135a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp135a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp024b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp024b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp135b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp135b1 = vsubq_f32(_r31, _r41);
                float32x4_t _tmp024c0 = vaddq_f32(_r50, _r60);
                float32x4_t _tmp024c1 = vaddq_f32(_r51, _r61);
                float32x4_t _tmp135c0 = vsubq_f32(_r50, _r60);
                float32x4_t _tmp135c1 = vsubq_f32(_r51, _r61);

                float32x4_t _tmp00 = vaddq_f32(vaddq_f32(_r00, _tmp024a0), vfmaq_laneq_f32(_tmp024b0, _tmp024c0, _coeffs, 0));
                float32x4_t _tmp01 = vaddq_f32(vaddq_f32(_r01, _tmp024a1), vfmaq_laneq_f32(_tmp024b1, _tmp024c1, _coeffs, 0));
                float32x4_t _tmp10 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a0, _tmp135b0, _v2, 0), _tmp135c0, _coeffs, 1);
                float32x4_t _tmp11 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a1, _tmp135b1, _v2, 0), _tmp135c1, _coeffs, 1);
                float32x4_t _tmp20 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 3), _tmp024c0, _coeffs, 2);
                float32x4_t _tmp21 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 3), _tmp024c1, _coeffs, 2);
                float32x4_t _tmp30 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a0, _tmp135b0, _coeffs, 2), _tmp135c0, _coeffs, 3);
                float32x4_t _tmp31 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a1, _tmp135b1, _coeffs, 2), _tmp135c1, _coeffs, 3);
                float32x4_t _tmp40 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 1), _tmp024c0, _v2, 0);
                float32x4_t _tmp41 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 1), _tmp024c1, _v2, 0);
                float32x4_t _tmp50 = vaddq_f32(vaddq_f32(_r70, _tmp135a0), vfmaq_laneq_f32(_tmp135c0, _tmp135b0, _coeffs, 0));
                float32x4_t _tmp51 = vaddq_f32(vaddq_f32(_r71, _tmp135a1), vfmaq_laneq_f32(_tmp135c1, _tmp135b1, _coeffs, 0));

                vst1q_f32(tmp[0][m], _tmp00);
                vst1q_f32(tmp[0][m] + 4, _tmp01);
                vst1q_f32(tmp[1][m], _tmp10);
                vst1q_f32(tmp[1][m] + 4, _tmp11);
                vst1q_f32(tmp[2][m], _tmp20);
                vst1q_f32(tmp[2][m] + 4, _tmp21);
                vst1q_f32(tmp[3][m], _tmp30);
                vst1q_f32(tmp[3][m] + 4, _tmp31);
                vst1q_f32(tmp[4][m], _tmp40);
                vst1q_f32(tmp[4][m] + 4, _tmp41);
                vst1q_f32(tmp[5][m], _tmp50);
                vst1q_f32(tmp[5][m] + 4, _tmp51);

                r0 += max_jj * 8 * 8;
                r1 += max_jj * 8 * 8;
                r2 += max_jj * 8 * 8;
                r3 += max_jj * 8 * 8;
                r4 += max_jj * 8 * 8;
                r5 += max_jj * 8 * 8;
                r6 += max_jj * 8 * 8;
                r7 += max_jj * 8 * 8;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float32x4_t _r00 = vld1q_f32(tmp[m][0]);
                float32x4_t _r01 = vld1q_f32(tmp[m][0] + 4);
                float32x4_t _r10 = vld1q_f32(tmp[m][1]);
                float32x4_t _r11 = vld1q_f32(tmp[m][1] + 4);
                float32x4_t _r20 = vld1q_f32(tmp[m][2]);
                float32x4_t _r21 = vld1q_f32(tmp[m][2] + 4);
                float32x4_t _r30 = vld1q_f32(tmp[m][3]);
                float32x4_t _r31 = vld1q_f32(tmp[m][3] + 4);
                float32x4_t _r40 = vld1q_f32(tmp[m][4]);
                float32x4_t _r41 = vld1q_f32(tmp[m][4] + 4);
                float32x4_t _r50 = vld1q_f32(tmp[m][5]);
                float32x4_t _r51 = vld1q_f32(tmp[m][5] + 4);
                float32x4_t _r60 = vld1q_f32(tmp[m][6]);
                float32x4_t _r61 = vld1q_f32(tmp[m][6] + 4);
                float32x4_t _r70 = vld1q_f32(tmp[m][7]);
                float32x4_t _r71 = vld1q_f32(tmp[m][7] + 4);

                float32x4_t _tmp024a0 = vaddq_f32(_r10, _r20);
                float32x4_t _tmp024a1 = vaddq_f32(_r11, _r21);
                float32x4_t _tmp135a0 = vsubq_f32(_r10, _r20);
                float32x4_t _tmp135a1 = vsubq_f32(_r11, _r21);
                float32x4_t _tmp024b0 = vaddq_f32(_r30, _r40);
                float32x4_t _tmp024b1 = vaddq_f32(_r31, _r41);
                float32x4_t _tmp135b0 = vsubq_f32(_r30, _r40);
                float32x4_t _tmp135b1 = vsubq_f32(_r31, _r41);
                float32x4_t _tmp024c0 = vaddq_f32(_r50, _r60);
                float32x4_t _tmp024c1 = vaddq_f32(_r51, _r61);
                float32x4_t _tmp135c0 = vsubq_f32(_r50, _r60);
                float32x4_t _tmp135c1 = vsubq_f32(_r51, _r61);

                float32x4_t _tmp00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r00, _tmp024a0), vfmaq_laneq_f32(_tmp024b0, _tmp024c0, _coeffs, 0)));
                float32x4_t _tmp01 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r01, _tmp024a1), vfmaq_laneq_f32(_tmp024b1, _tmp024c1, _coeffs, 0)));
                float32x4_t _tmp10 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a0, _tmp135b0, _v2, 0), _tmp135c0, _coeffs, 1));
                float32x4_t _tmp11 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a1, _tmp135b1, _v2, 0), _tmp135c1, _coeffs, 1));
                float32x4_t _tmp20 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 3), _tmp024c0, _coeffs, 2));
                float32x4_t _tmp21 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 3), _tmp024c1, _coeffs, 2));
                float32x4_t _tmp30 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a0, _tmp135b0, _coeffs, 2), _tmp135c0, _coeffs, 3));
                float32x4_t _tmp31 = vaddq_f32(_bias1, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a1, _tmp135b1, _coeffs, 2), _tmp135c1, _coeffs, 3));
                float32x4_t _tmp40 = vaddq_f32(_bias0, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a0, _tmp024b0, _coeffs, 1), _tmp024c0, _v2, 0));
                float32x4_t _tmp41 = vaddq_f32(_bias1, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a1, _tmp024b1, _coeffs, 1), _tmp024c1, _v2, 0));
                float32x4_t _tmp50 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r70, _tmp135a0), vfmaq_laneq_f32(_tmp135c0, _tmp135b0, _coeffs, 0)));
                float32x4_t _tmp51 = vaddq_f32(_bias1, vaddq_f32(vaddq_f32(_r71, _tmp135a1), vfmaq_laneq_f32(_tmp135c1, _tmp135b1, _coeffs, 0)));

                if (out_elempack == 4)
                {
                    unsigned short* outptr1 = outptr0 + N;

                    vst1_u16(outptr0, float2bfloat(_tmp00));
                    vst1_u16(outptr1, float2bfloat(_tmp01));
                    if (tj * 6 + 1 < outw)
                    {
                        vst1_u16(outptr0 + 4, float2bfloat(_tmp10));
                        vst1_u16(outptr1 + 4, float2bfloat(_tmp11));
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        vst1_u16(outptr0 + 8, float2bfloat(_tmp20));
                        vst1_u16(outptr1 + 8, float2bfloat(_tmp21));
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        vst1_u16(outptr0 + 12, float2bfloat(_tmp30));
                        vst1_u16(outptr1 + 12, float2bfloat(_tmp31));
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        vst1_u16(outptr0 + 16, float2bfloat(_tmp40));
                        vst1_u16(outptr1 + 16, float2bfloat(_tmp41));
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        vst1_u16(outptr0 + 20, float2bfloat(_tmp50));
                        vst1_u16(outptr1 + 20, float2bfloat(_tmp51));
                    }
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[8];
                    unsigned short tmp1[8];
                    unsigned short tmp2[8];
                    unsigned short tmp3[8];
                    unsigned short tmp4[8];
                    unsigned short tmp5[8];
                    vst1_u16(tmp0, float2bfloat(_tmp00));
                    vst1_u16(tmp0 + 4, float2bfloat(_tmp01));
                    vst1_u16(tmp1, float2bfloat(_tmp10));
                    vst1_u16(tmp1 + 4, float2bfloat(_tmp11));
                    vst1_u16(tmp2, float2bfloat(_tmp20));
                    vst1_u16(tmp2 + 4, float2bfloat(_tmp21));
                    vst1_u16(tmp3, float2bfloat(_tmp30));
                    vst1_u16(tmp3 + 4, float2bfloat(_tmp31));
                    vst1_u16(tmp4, float2bfloat(_tmp40));
                    vst1_u16(tmp4 + 4, float2bfloat(_tmp41));
                    vst1_u16(tmp5, float2bfloat(_tmp50));
                    vst1_u16(tmp5 + 4, float2bfloat(_tmp51));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;
                    unsigned short* outptr4 = outptr0 + N * 4;
                    unsigned short* outptr5 = outptr0 + N * 5;
                    unsigned short* outptr6 = outptr0 + N * 6;
                    unsigned short* outptr7 = outptr0 + N * 7;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    outptr4[0] = tmp0[4];
                    outptr5[0] = tmp0[5];
                    outptr6[0] = tmp0[6];
                    outptr7[0] = tmp0[7];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                        outptr4[1] = tmp1[4];
                        outptr5[1] = tmp1[5];
                        outptr6[1] = tmp1[6];
                        outptr7[1] = tmp1[7];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                        outptr4[2] = tmp2[4];
                        outptr5[2] = tmp2[5];
                        outptr6[2] = tmp2[6];
                        outptr7[2] = tmp2[7];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                        outptr4[3] = tmp3[4];
                        outptr5[3] = tmp3[5];
                        outptr6[3] = tmp3[6];
                        outptr7[3] = tmp3[7];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                        outptr4[4] = tmp4[4];
                        outptr5[4] = tmp4[5];
                        outptr6[4] = tmp4[6];
                        outptr7[4] = tmp4[7];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                        outptr4[5] = tmp5[4];
                        outptr5[5] = tmp5[5];
                        outptr6[5] = tmp5[6];
                        outptr7[5] = tmp5[7];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float32x4_t _bias0 = biasptr ? vld1q_f32(biasptr + i + ii) : vdupq_n_f32(0.f);

#ifdef _MSC_VER
        __declspec(align(16))
#else
        __attribute__((aligned(16)))
#endif
        float tmp[6][8][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 4;
            const float* r1 = r0 + max_jj * 4;
            const float* r2 = r0 + max_jj * 4 * 2;
            const float* r3 = r0 + max_jj * 4 * 3;
            const float* r4 = r0 + max_jj * 4 * 4;
            const float* r5 = r0 + max_jj * 4 * 5;
            const float* r6 = r0 + max_jj * 4 * 6;
            const float* r7 = r0 + max_jj * 4 * 7;

            for (int m = 0; m < 8; m++)
            {
                float32x4_t _r0 = vld1q_f32(r0);
                float32x4_t _r1 = vld1q_f32(r1);
                float32x4_t _r2 = vld1q_f32(r2);
                float32x4_t _r3 = vld1q_f32(r3);
                float32x4_t _r4 = vld1q_f32(r4);
                float32x4_t _r5 = vld1q_f32(r5);
                float32x4_t _r6 = vld1q_f32(r6);
                float32x4_t _r7 = vld1q_f32(r7);

                float32x4_t _tmp024a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp135a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp024b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp135b = vsubq_f32(_r3, _r4);
                float32x4_t _tmp024c = vaddq_f32(_r5, _r6);
                float32x4_t _tmp135c = vsubq_f32(_r5, _r6);

#if __aarch64__
                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp024a), vfmaq_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0));
                float32x4_t _tmp1 = vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, _coeffs, 1);
                float32x4_t _tmp2 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float32x4_t _tmp3 = vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float32x4_t _tmp4 = vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2, 0);
                float32x4_t _tmp5 = vaddq_f32(vaddq_f32(_r7, _tmp135a), vfmaq_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0));
#else
                float32x4_t _tmp0 = vaddq_f32(vaddq_f32(_r0, _tmp024a), vmlaq_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0));
                float32x4_t _tmp1 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, vget_low_f32(_coeffs), 1);
                float32x4_t _tmp2 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0);
                float32x4_t _tmp3 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1);
                float32x4_t _tmp4 = vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2, 0);
                float32x4_t _tmp5 = vaddq_f32(vaddq_f32(_r7, _tmp135a), vmlaq_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0));
#endif

                vst1q_f32(tmp[0][m], _tmp0);
                vst1q_f32(tmp[1][m], _tmp1);
                vst1q_f32(tmp[2][m], _tmp2);
                vst1q_f32(tmp[3][m], _tmp3);
                vst1q_f32(tmp[4][m], _tmp4);
                vst1q_f32(tmp[5][m], _tmp5);

                r0 += max_jj * 8 * 4;
                r1 += max_jj * 8 * 4;
                r2 += max_jj * 8 * 4;
                r3 += max_jj * 8 * 4;
                r4 += max_jj * 8 * 4;
                r5 += max_jj * 8 * 4;
                r6 += max_jj * 8 * 4;
                r7 += max_jj * 8 * 4;
            }

            unsigned short* outptr0 = top_blob.channel((i + ii) / out_elempack).row<unsigned short>(ti * 6) + (tj * 6) * out_elempack;

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float32x4_t _r0 = vld1q_f32(tmp[m][0]);
                float32x4_t _r1 = vld1q_f32(tmp[m][1]);
                float32x4_t _r2 = vld1q_f32(tmp[m][2]);
                float32x4_t _r3 = vld1q_f32(tmp[m][3]);
                float32x4_t _r4 = vld1q_f32(tmp[m][4]);
                float32x4_t _r5 = vld1q_f32(tmp[m][5]);
                float32x4_t _r6 = vld1q_f32(tmp[m][6]);
                float32x4_t _r7 = vld1q_f32(tmp[m][7]);

                float32x4_t _tmp024a = vaddq_f32(_r1, _r2);
                float32x4_t _tmp135a = vsubq_f32(_r1, _r2);
                float32x4_t _tmp024b = vaddq_f32(_r3, _r4);
                float32x4_t _tmp135b = vsubq_f32(_r3, _r4);
                float32x4_t _tmp024c = vaddq_f32(_r5, _r6);
                float32x4_t _tmp135c = vsubq_f32(_r5, _r6);

#if __aarch64__
                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _tmp024a), vfmaq_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0)));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, _coeffs, 1));
                float32x4_t _tmp2 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float32x4_t _tmp3 = vaddq_f32(_bias0, vfmaq_laneq_f32(vfmaq_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float32x4_t _tmp4 = vaddq_f32(_bias0, vfmaq_lane_f32(vfmaq_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2, 0));
                float32x4_t _tmp5 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r7, _tmp135a), vfmaq_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0)));
#else
                float32x4_t _tmp0 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r0, _tmp024a), vmlaq_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0)));
                float32x4_t _tmp1 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, _v2, 0), _tmp135c, vget_low_f32(_coeffs), 1));
                float32x4_t _tmp2 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0));
                float32x4_t _tmp3 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1));
                float32x4_t _tmp4 = vaddq_f32(_bias0, vmlaq_lane_f32(vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2, 0));
                float32x4_t _tmp5 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_r7, _tmp135a), vmlaq_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0)));
#endif

                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_tmp0));
                    if (tj * 6 + 1 < outw) vst1_u16(outptr0 + 4, float2bfloat(_tmp1));
                    if (tj * 6 + 2 < outw) vst1_u16(outptr0 + 8, float2bfloat(_tmp2));
                    if (tj * 6 + 3 < outw) vst1_u16(outptr0 + 12, float2bfloat(_tmp3));
                    if (tj * 6 + 4 < outw) vst1_u16(outptr0 + 16, float2bfloat(_tmp4));
                    if (tj * 6 + 5 < outw) vst1_u16(outptr0 + 20, float2bfloat(_tmp5));
                }
                if (out_elempack == 1)
                {
                    unsigned short tmp0[4];
                    unsigned short tmp1[4];
                    unsigned short tmp2[4];
                    unsigned short tmp3[4];
                    unsigned short tmp4[4];
                    unsigned short tmp5[4];
                    vst1_u16(tmp0, float2bfloat(_tmp0));
                    vst1_u16(tmp1, float2bfloat(_tmp1));
                    vst1_u16(tmp2, float2bfloat(_tmp2));
                    vst1_u16(tmp3, float2bfloat(_tmp3));
                    vst1_u16(tmp4, float2bfloat(_tmp4));
                    vst1_u16(tmp5, float2bfloat(_tmp5));

                    unsigned short* outptr1 = outptr0 + N;
                    unsigned short* outptr2 = outptr0 + N * 2;
                    unsigned short* outptr3 = outptr0 + N * 3;

                    outptr0[0] = tmp0[0];
                    outptr1[0] = tmp0[1];
                    outptr2[0] = tmp0[2];
                    outptr3[0] = tmp0[3];
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = tmp1[0];
                        outptr1[1] = tmp1[1];
                        outptr2[1] = tmp1[2];
                        outptr3[1] = tmp1[3];
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = tmp2[0];
                        outptr1[2] = tmp2[1];
                        outptr2[2] = tmp2[2];
                        outptr3[2] = tmp2[3];
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = tmp3[0];
                        outptr1[3] = tmp3[1];
                        outptr2[3] = tmp3[2];
                        outptr3[3] = tmp3[3];
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = tmp4[0];
                        outptr1[4] = tmp4[1];
                        outptr2[4] = tmp4[2];
                        outptr3[4] = tmp4[3];
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = tmp5[0];
                        outptr1[5] = tmp5[1];
                        outptr2[5] = tmp5[2];
                        outptr3[5] = tmp5[3];
                    }
                }

                outptr0 += outw * out_elempack;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        float32x2_t _bias0 = biasptr ? vld1_f32(biasptr + i + ii) : vdup_n_f32(0.f);
#else
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;
        float bias1 = biasptr ? biasptr[i + ii + 1] : 0.f;
#endif

#ifdef _MSC_VER
        __declspec(align(8))
#else
        __attribute__((aligned(8)))
#endif
        float tmp[6][8][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj * 2;
            const float* r1 = r0 + max_jj * 2;
            const float* r2 = r0 + max_jj * 2 * 2;
            const float* r3 = r0 + max_jj * 2 * 3;
            const float* r4 = r0 + max_jj * 2 * 4;
            const float* r5 = r0 + max_jj * 2 * 5;
            const float* r6 = r0 + max_jj * 2 * 6;
            const float* r7 = r0 + max_jj * 2 * 7;

            for (int m = 0; m < 8; m++)
            {
#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(r0);
                float32x2_t _r1 = vld1_f32(r1);
                float32x2_t _r2 = vld1_f32(r2);
                float32x2_t _r3 = vld1_f32(r3);
                float32x2_t _r4 = vld1_f32(r4);
                float32x2_t _r5 = vld1_f32(r5);
                float32x2_t _r6 = vld1_f32(r6);
                float32x2_t _r7 = vld1_f32(r7);

                float32x2_t _tmp024a = vadd_f32(_r1, _r2);
                float32x2_t _tmp135a = vsub_f32(_r1, _r2);
                float32x2_t _tmp024b = vadd_f32(_r3, _r4);
                float32x2_t _tmp135b = vsub_f32(_r3, _r4);
                float32x2_t _tmp024c = vadd_f32(_r5, _r6);
                float32x2_t _tmp135c = vsub_f32(_r5, _r6);

#if __aarch64__
                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp024a), vfma_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0));
                float32x2_t _tmp1 = vfma_laneq_f32(vfma_f32(_tmp135a, _tmp135b, _v2), _tmp135c, _coeffs, 1);
                float32x2_t _tmp2 = vfma_laneq_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2);
                float32x2_t _tmp3 = vfma_laneq_f32(vfma_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3);
                float32x2_t _tmp4 = vfma_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2);
                float32x2_t _tmp5 = vadd_f32(vadd_f32(_r7, _tmp135a), vfma_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0));
#else
                float32x2_t _tmp0 = vadd_f32(vadd_f32(_r0, _tmp024a), vmla_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0));
                float32x2_t _tmp1 = vmla_lane_f32(vmla_f32(_tmp135a, _tmp135b, _v2), _tmp135c, vget_low_f32(_coeffs), 1);
                float32x2_t _tmp2 = vmla_lane_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0);
                float32x2_t _tmp3 = vmla_lane_f32(vmla_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1);
                float32x2_t _tmp4 = vmla_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2);
                float32x2_t _tmp5 = vadd_f32(vadd_f32(_r7, _tmp135a), vmla_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0));
#endif

                vst1_f32(tmp[0][m], _tmp0);
                vst1_f32(tmp[1][m], _tmp1);
                vst1_f32(tmp[2][m], _tmp2);
                vst1_f32(tmp[3][m], _tmp3);
                vst1_f32(tmp[4][m], _tmp4);
                vst1_f32(tmp[5][m], _tmp5);
#else
                float tmp024a0 = r1[0] + r2[0];
                float tmp024a1 = r1[1] + r2[1];
                float tmp135a0 = r1[0] - r2[0];
                float tmp135a1 = r1[1] - r2[1];
                float tmp024b0 = r3[0] + r4[0];
                float tmp024b1 = r3[1] + r4[1];
                float tmp135b0 = r3[0] - r4[0];
                float tmp135b1 = r3[1] - r4[1];
                float tmp024c0 = r5[0] + r6[0];
                float tmp024c1 = r5[1] + r6[1];
                float tmp135c0 = r5[0] - r6[0];
                float tmp135c1 = r5[1] - r6[1];

                tmp[0][m][0] = r0[0] + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                tmp[0][m][1] = r0[1] + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                tmp[1][m][0] = tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                tmp[1][m][1] = tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                tmp[2][m][0] = tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                tmp[2][m][1] = tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                tmp[3][m][0] = tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                tmp[3][m][1] = tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                tmp[4][m][0] = tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                tmp[4][m][1] = tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                tmp[5][m][0] = r7[0] + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                tmp[5][m][1] = r7[1] + tmp135a1 + tmp135b1 * 32 + tmp135c1;
#endif

                r0 += max_jj * 8 * 2;
                r1 += max_jj * 8 * 2;
                r2 += max_jj * 8 * 2;
                r3 += max_jj * 8 * 2;
                r4 += max_jj * 8 * 2;
                r5 += max_jj * 8 * 2;
                r6 += max_jj * 8 * 2;
                r7 += max_jj * 8 * 2;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

#if __ARM_NEON
                float32x2_t _r0 = vld1_f32(tmp[m][0]);
                float32x2_t _r1 = vld1_f32(tmp[m][1]);
                float32x2_t _r2 = vld1_f32(tmp[m][2]);
                float32x2_t _r3 = vld1_f32(tmp[m][3]);
                float32x2_t _r4 = vld1_f32(tmp[m][4]);
                float32x2_t _r5 = vld1_f32(tmp[m][5]);
                float32x2_t _r6 = vld1_f32(tmp[m][6]);
                float32x2_t _r7 = vld1_f32(tmp[m][7]);

                float32x2_t _tmp024a = vadd_f32(_r1, _r2);
                float32x2_t _tmp135a = vsub_f32(_r1, _r2);
                float32x2_t _tmp024b = vadd_f32(_r3, _r4);
                float32x2_t _tmp135b = vsub_f32(_r3, _r4);
                float32x2_t _tmp024c = vadd_f32(_r5, _r6);
                float32x2_t _tmp135c = vsub_f32(_r5, _r6);

#if __aarch64__
                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _tmp024a), vfma_laneq_f32(_tmp024b, _tmp024c, _coeffs, 0)));
                float32x2_t _tmp1 = vadd_f32(_bias0, vfma_laneq_f32(vfma_f32(_tmp135a, _tmp135b, _v2), _tmp135c, _coeffs, 1));
                float32x2_t _tmp2 = vadd_f32(_bias0, vfma_laneq_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 3), _tmp024c, _coeffs, 2));
                float32x2_t _tmp3 = vadd_f32(_bias0, vfma_laneq_f32(vfma_laneq_f32(_tmp135a, _tmp135b, _coeffs, 2), _tmp135c, _coeffs, 3));
                float32x2_t _tmp4 = vadd_f32(_bias0, vfma_f32(vfma_laneq_f32(_tmp024a, _tmp024b, _coeffs, 1), _tmp024c, _v2));
                float32x2_t _tmp5 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r7, _tmp135a), vfma_laneq_f32(_tmp135c, _tmp135b, _coeffs, 0)));
#else
                float32x2_t _tmp0 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r0, _tmp024a), vmla_lane_f32(_tmp024b, _tmp024c, vget_low_f32(_coeffs), 0)));
                float32x2_t _tmp1 = vadd_f32(_bias0, vmla_lane_f32(vmla_f32(_tmp135a, _tmp135b, _v2), _tmp135c, vget_low_f32(_coeffs), 1));
                float32x2_t _tmp2 = vadd_f32(_bias0, vmla_lane_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeffs), 1), _tmp024c, vget_high_f32(_coeffs), 0));
                float32x2_t _tmp3 = vadd_f32(_bias0, vmla_lane_f32(vmla_lane_f32(_tmp135a, _tmp135b, vget_high_f32(_coeffs), 0), _tmp135c, vget_high_f32(_coeffs), 1));
                float32x2_t _tmp4 = vadd_f32(_bias0, vmla_f32(vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeffs), 1), _tmp024c, _v2));
                float32x2_t _tmp5 = vadd_f32(_bias0, vadd_f32(vadd_f32(_r7, _tmp135a), vmla_lane_f32(_tmp135c, _tmp135b, vget_low_f32(_coeffs), 0)));
#endif
#else
                float r00 = tmp[m][0][0];
                float r01 = tmp[m][0][1];
                float r10 = tmp[m][1][0];
                float r11 = tmp[m][1][1];
                float r20 = tmp[m][2][0];
                float r21 = tmp[m][2][1];
                float r30 = tmp[m][3][0];
                float r31 = tmp[m][3][1];
                float r40 = tmp[m][4][0];
                float r41 = tmp[m][4][1];
                float r50 = tmp[m][5][0];
                float r51 = tmp[m][5][1];
                float r60 = tmp[m][6][0];
                float r61 = tmp[m][6][1];
                float r70 = tmp[m][7][0];
                float r71 = tmp[m][7][1];

                float tmp024a0 = r10 + r20;
                float tmp024a1 = r11 + r21;
                float tmp135a0 = r10 - r20;
                float tmp135a1 = r11 - r21;
                float tmp024b0 = r30 + r40;
                float tmp024b1 = r31 + r41;
                float tmp135b0 = r30 - r40;
                float tmp135b1 = r31 - r41;
                float tmp024c0 = r50 + r60;
                float tmp024c1 = r51 + r61;
                float tmp135c0 = r50 - r60;
                float tmp135c1 = r51 - r61;

                float tmp00 = bias0 + r00 + tmp024a0 + tmp024b0 + tmp024c0 * 32;
                float tmp01 = bias1 + r01 + tmp024a1 + tmp024b1 + tmp024c1 * 32;
                float tmp10 = bias0 + tmp135a0 + tmp135b0 + tmp135b0 + tmp135c0 * 16;
                float tmp11 = bias1 + tmp135a1 + tmp135b1 + tmp135b1 + tmp135c1 * 16;
                float tmp20 = bias0 + tmp024a0 + tmp024b0 * 4 + tmp024c0 * 8;
                float tmp21 = bias1 + tmp024a1 + tmp024b1 * 4 + tmp024c1 * 8;
                float tmp30 = bias0 + tmp135a0 + tmp135b0 * 8 + tmp135c0 * 4;
                float tmp31 = bias1 + tmp135a1 + tmp135b1 * 8 + tmp135c1 * 4;
                float tmp40 = bias0 + tmp024a0 + tmp024b0 * 16 + tmp024c0 + tmp024c0;
                float tmp41 = bias1 + tmp024a1 + tmp024b1 * 16 + tmp024c1 + tmp024c1;
                float tmp50 = bias0 + r70 + tmp135a0 + tmp135b0 * 32 + tmp135c0;
                float tmp51 = bias1 + r71 + tmp135a1 + tmp135b1 * 32 + tmp135c1;
#endif

                // if (out_elempack == 1)
                {
                    unsigned short* outptr1 = outptr0 + N;

#if __ARM_NEON
                    uint16x4_t _tmp01 = float2bfloat(vcombine_f32(_tmp0, _tmp1));
                    uint16x4_t _tmp23 = float2bfloat(vcombine_f32(_tmp2, _tmp3));
                    uint16x4_t _tmp45 = float2bfloat(vcombine_f32(_tmp4, _tmp5));

                    outptr0[0] = vget_lane_u16(_tmp01, 0);
                    outptr1[0] = vget_lane_u16(_tmp01, 1);
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = vget_lane_u16(_tmp01, 2);
                        outptr1[1] = vget_lane_u16(_tmp01, 3);
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = vget_lane_u16(_tmp23, 0);
                        outptr1[2] = vget_lane_u16(_tmp23, 1);
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = vget_lane_u16(_tmp23, 2);
                        outptr1[3] = vget_lane_u16(_tmp23, 3);
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = vget_lane_u16(_tmp45, 0);
                        outptr1[4] = vget_lane_u16(_tmp45, 1);
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = vget_lane_u16(_tmp45, 2);
                        outptr1[5] = vget_lane_u16(_tmp45, 3);
                    }
#else
                    outptr0[0] = float32_to_bfloat16(tmp00);
                    outptr1[0] = float32_to_bfloat16(tmp01);
                    if (tj * 6 + 1 < outw)
                    {
                        outptr0[1] = float32_to_bfloat16(tmp10);
                        outptr1[1] = float32_to_bfloat16(tmp11);
                    }
                    if (tj * 6 + 2 < outw)
                    {
                        outptr0[2] = float32_to_bfloat16(tmp20);
                        outptr1[2] = float32_to_bfloat16(tmp21);
                    }
                    if (tj * 6 + 3 < outw)
                    {
                        outptr0[3] = float32_to_bfloat16(tmp30);
                        outptr1[3] = float32_to_bfloat16(tmp31);
                    }
                    if (tj * 6 + 4 < outw)
                    {
                        outptr0[4] = float32_to_bfloat16(tmp40);
                        outptr1[4] = float32_to_bfloat16(tmp41);
                    }
                    if (tj * 6 + 5 < outw)
                    {
                        outptr0[5] = float32_to_bfloat16(tmp50);
                        outptr1[5] = float32_to_bfloat16(tmp51);
                    }
#endif
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        float bias0 = biasptr ? biasptr[i + ii] : 0.f;

        float tmp[6][8];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const float* r0 = (const float*)top_tile + ii * max_jj * 64 + jj;
            const float* r1 = r0 + max_jj;
            const float* r2 = r0 + max_jj * 2;
            const float* r3 = r0 + max_jj * 3;
            const float* r4 = r0 + max_jj * 4;
            const float* r5 = r0 + max_jj * 5;
            const float* r6 = r0 + max_jj * 6;
            const float* r7 = r0 + max_jj * 7;

            for (int m = 0; m < 8; m++)
            {
                float tmp024a = r1[0] + r2[0];
                float tmp135a = r1[0] - r2[0];
                float tmp024b = r3[0] + r4[0];
                float tmp135b = r3[0] - r4[0];
                float tmp024c = r5[0] + r6[0];
                float tmp135c = r5[0] - r6[0];

                tmp[0][m] = r0[0] + tmp024a + tmp024b + tmp024c * 32;
                tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                tmp[5][m] = r7[0] + tmp135a + tmp135b * 32 + tmp135c;

                r0 += max_jj * 8;
                r1 += max_jj * 8;
                r2 += max_jj * 8;
                r3 += max_jj * 8;
                r4 += max_jj * 8;
                r5 += max_jj * 8;
                r6 += max_jj * 8;
                r7 += max_jj * 8;
            }

            unsigned short* outptr0 = top_blob.channel(i + ii).row<unsigned short>(ti * 6) + (tj * 6);

            for (int m = 0; m < 6; m++)
            {
                if (ti * 6 + m >= outh)
                    continue;

                float r0 = tmp[m][0];
                float r1 = tmp[m][1];
                float r2 = tmp[m][2];
                float r3 = tmp[m][3];
                float r4 = tmp[m][4];
                float r5 = tmp[m][5];
                float r6 = tmp[m][6];
                float r7 = tmp[m][7];

                float tmp024a = r1 + r2;
                float tmp135a = r1 - r2;
                float tmp024b = r3 + r4;
                float tmp135b = r3 - r4;
                float tmp024c = r5 + r6;
                float tmp135c = r5 - r6;

                float tmp0 = bias0 + r0 + tmp024a + tmp024b + tmp024c * 32;
                float tmp1 = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                float tmp2 = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                float tmp3 = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                float tmp4 = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;
                float tmp5 = bias0 + r7 + tmp135a + tmp135b * 32 + tmp135c;

                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(tmp0);
                    if (tj * 6 + 1 < outw) outptr0[1] = float32_to_bfloat16(tmp1);
                    if (tj * 6 + 2 < outw) outptr0[2] = float32_to_bfloat16(tmp2);
                    if (tj * 6 + 3 < outw) outptr0[3] = float32_to_bfloat16(tmp3);
                    if (tj * 6 + 4 < outw) outptr0[4] = float32_to_bfloat16(tmp4);
                    if (tj * 6 + 5 < outw) outptr0[5] = float32_to_bfloat16(tmp5);
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd63_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 6n+2, winograd F(6,3)
    int w_tiles = (outw + 5) / 6;
    int h_tiles = (outh + 5) / 6;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 64;

    // NCNN_LOGE("conv3x3s1_winograd63_bf16s %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd63_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd63_transform_input_tile_bf16s(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            conv3x3s1_winograd_transpose_pack_B_tile(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                conv3x3s1_winograd_gemm_transB_packed_tile(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, opt.use_a53_a55_optimized_kernel);
            }

            // transform output
            conv3x3s1_winograd63_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj);
        }
    }
}
