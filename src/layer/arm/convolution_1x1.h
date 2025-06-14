// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);
        Mat out4 = top_blob.channel(p + 4);
        Mat out5 = top_blob.channel(p + 5);
        Mat out6 = top_blob.channel(p + 6);
        Mat out7 = top_blob.channel(p + 7);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;
        const float bias4 = bias ? bias[p + 4] : 0.f;
        const float bias5 = bias ? bias[p + 5] : 0.f;
        const float bias6 = bias ? bias[p + 6] : 0.f;
        const float bias7 = bias ? bias[p + 7] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);
        out4.fill(bias4);
        out5.fill(bias5);
        out6.fill(bias6);
        out7.fill(bias7);

        int q = 0;

        for (; q + 7 < inch; q += 8)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;
            float* outptr6 = out6;
            float* outptr7 = out7;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);
            const float* img4 = bottom_blob.channel(q + 4);
            const float* img5 = bottom_blob.channel(q + 5);
            const float* img6 = bottom_blob.channel(q + 6);
            const float* img7 = bottom_blob.channel(q + 7);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;
            const float* kernel4 = kernel + (p + 4) * inch + q;
            const float* kernel5 = kernel + (p + 5) * inch + q;
            const float* kernel6 = kernel + (p + 6) * inch + q;
            const float* kernel7 = kernel + (p + 7) * inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;
            const float* r4 = img4;
            const float* r5 = img5;
            const float* r6 = img6;
            const float* r7 = img7;

            int size = outw * outh;

            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);
            float32x4_t _k4 = vld1q_f32(kernel4);
            float32x4_t _k5 = vld1q_f32(kernel5);
            float32x4_t _k6 = vld1q_f32(kernel6);
            float32x4_t _k7 = vld1q_f32(kernel7);

            float32x4_t _k0n = vld1q_f32(kernel0 + 4);
            float32x4_t _k1n = vld1q_f32(kernel1 + 4);
            float32x4_t _k2n = vld1q_f32(kernel2 + 4);
            float32x4_t _k3n = vld1q_f32(kernel3 + 4);
            float32x4_t _k4n = vld1q_f32(kernel4 + 4);
            float32x4_t _k5n = vld1q_f32(kernel5 + 4);
            float32x4_t _k6n = vld1q_f32(kernel6 + 4);
            float32x4_t _k7n = vld1q_f32(kernel7 + 4);

#ifdef __clang__
            // gcc reject over 30 oprands :(
            if (nn > 0)
            {
                asm volatile(
                    "prfm   pldl1keep, [%9, #128]       \n"
                    "ld1    {v17.4s}, [%9], #16         \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v18.4s}, [%1]              \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v19.4s}, [%2]              \n"

                    "0:                                 \n"

                    "fmla   v18.4s, v17.4s, %34.s[0]    \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v20.4s}, [%3]              \n"

                    "fmla   v19.4s, v17.4s, %35.s[0]    \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v21.4s}, [%4]              \n"

                    "fmla   v20.4s, v17.4s, %36.s[0]    \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v22.4s}, [%5]              \n"

                    "fmla   v21.4s, v17.4s, %37.s[0]    \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v23.4s}, [%6]              \n"

                    "fmla   v22.4s, v17.4s, %38.s[0]    \n"

                    "prfm   pldl1keep, [%10, #128]      \n"
                    "ld1    {v16.4s}, [%10], #16        \n"

                    "fmla   v23.4s, v17.4s, %39.s[0]    \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v24.4s}, [%7]              \n"

                    "fmla   v18.4s, v16.4s, %34.s[1]    \n"
                    "fmla   v19.4s, v16.4s, %35.s[1]    \n"

                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v25.4s}, [%8]              \n"

                    "fmla   v24.4s, v17.4s, %40.s[0]    \n"
                    "fmla   v25.4s, v17.4s, %41.s[0]    \n"

                    "fmla   v20.4s, v16.4s, %36.s[1]    \n"
                    "fmla   v21.4s, v16.4s, %37.s[1]    \n"

                    "prfm   pldl1keep, [%11, #128]      \n"
                    "ld1    {v17.4s}, [%11], #16        \n"

                    "fmla   v22.4s, v16.4s, %38.s[1]    \n"
                    "fmla   v23.4s, v16.4s, %39.s[1]    \n"

                    "fmla   v18.4s, v17.4s, %34.s[2]    \n"
                    "fmla   v19.4s, v17.4s, %35.s[2]    \n"

                    "fmla   v24.4s, v16.4s, %40.s[1]    \n"
                    "fmla   v25.4s, v16.4s, %41.s[1]    \n"

                    "fmla   v20.4s, v17.4s, %36.s[2]    \n"
                    "fmla   v21.4s, v17.4s, %37.s[2]    \n"

                    "prfm   pldl1keep, [%12, #128]      \n"
                    "ld1    {v16.4s}, [%12], #16        \n"

                    "fmla   v22.4s, v17.4s, %38.s[2]    \n"
                    "fmla   v23.4s, v17.4s, %39.s[2]    \n"

                    "fmla   v18.4s, v16.4s, %34.s[3]    \n"
                    "fmla   v19.4s, v16.4s, %35.s[3]    \n"

                    "fmla   v24.4s, v17.4s, %40.s[2]    \n"
                    "fmla   v25.4s, v17.4s, %41.s[2]    \n"

                    "fmla   v20.4s, v16.4s, %36.s[3]    \n"
                    "fmla   v21.4s, v16.4s, %37.s[3]    \n"

                    "prfm   pldl1keep, [%13, #128]      \n"
                    "ld1    {v17.4s}, [%13], #16        \n"

                    "fmla   v22.4s, v16.4s, %38.s[3]    \n"
                    "fmla   v23.4s, v16.4s, %39.s[3]    \n"

                    "fmla   v18.4s, v17.4s, %42.s[0]    \n"
                    "fmla   v19.4s, v17.4s, %43.s[0]    \n"

                    "fmla   v24.4s, v16.4s, %40.s[3]    \n"
                    "fmla   v25.4s, v16.4s, %41.s[3]    \n"

                    "fmla   v20.4s, v17.4s, %44.s[0]    \n"
                    "fmla   v21.4s, v17.4s, %45.s[0]    \n"

                    "prfm   pldl1keep, [%14, #128]      \n"
                    "ld1    {v16.4s}, [%14], #16        \n"

                    "fmla   v22.4s, v17.4s, %46.s[0]    \n"
                    "fmla   v23.4s, v17.4s, %47.s[0]    \n"

                    "fmla   v18.4s, v16.4s, %42.s[1]    \n"
                    "fmla   v19.4s, v16.4s, %43.s[1]    \n"

                    "fmla   v24.4s, v17.4s, %48.s[0]    \n"
                    "fmla   v25.4s, v17.4s, %49.s[0]    \n"

                    "fmla   v20.4s, v16.4s, %44.s[1]    \n"
                    "fmla   v21.4s, v16.4s, %45.s[1]    \n"

                    "prfm   pldl1keep, [%15, #128]      \n"
                    "ld1    {v17.4s}, [%15], #16        \n"

                    "fmla   v22.4s, v16.4s, %46.s[1]    \n"
                    "fmla   v23.4s, v16.4s, %47.s[1]    \n"

                    "fmla   v18.4s, v17.4s, %42.s[2]    \n"
                    "fmla   v19.4s, v17.4s, %43.s[2]    \n"

                    "fmla   v24.4s, v16.4s, %48.s[1]    \n"
                    "fmla   v25.4s, v16.4s, %49.s[1]    \n"

                    "fmla   v20.4s, v17.4s, %44.s[2]    \n"
                    "fmla   v21.4s, v17.4s, %45.s[2]    \n"

                    "prfm   pldl1keep, [%16, #128]      \n"
                    "ld1    {v16.4s}, [%16], #16        \n"

                    "fmla   v22.4s, v17.4s, %46.s[2]    \n"
                    "fmla   v23.4s, v17.4s, %47.s[2]    \n"

                    "fmla   v18.4s, v16.4s, %42.s[3]    \n"
                    "fmla   v19.4s, v16.4s, %43.s[3]    \n"

                    "fmla   v24.4s, v17.4s, %48.s[2]    \n"
                    "fmla   v25.4s, v17.4s, %49.s[2]    \n"

                    "fmla   v20.4s, v16.4s, %44.s[3]    \n"
                    "fmla   v21.4s, v16.4s, %45.s[3]    \n"

                    "st1    {v18.4s}, [%1], #16         \n"

                    "fmla   v22.4s, v16.4s, %46.s[3]    \n"

                    "st1    {v19.4s}, [%2], #16         \n"

                    "fmla   v23.4s, v16.4s, %47.s[3]    \n"

                    "st1    {v20.4s}, [%3], #16         \n"

                    "prfm   pldl1keep, [%9, #128]       \n"
                    "ld1    {v17.4s}, [%9], #16         \n"

                    "fmla   v24.4s, v16.4s, %48.s[3]    \n"

                    "st1    {v21.4s}, [%4], #16         \n"

                    "fmla   v25.4s, v16.4s, %49.s[3]    \n"

                    "st1    {v22.4s}, [%5], #16         \n"

                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v18.4s}, [%1]              \n"

                    "st1    {v23.4s}, [%6], #16         \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v19.4s}, [%2]              \n"

                    "st1    {v24.4s}, [%7], #16         \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v25.4s}, [%8], #16         \n"

                    "bne    0b                          \n"
                    "sub    %9, %9, #16                 \n"
                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(outptr6), // %7
                    "=r"(outptr7), // %8
                    "=r"(r0),      // %9
                    "=r"(r1),      // %10
                    "=r"(r2),      // %11
                    "=r"(r3),      // %12
                    "=r"(r4),      // %13
                    "=r"(r5),      // %14
                    "=r"(r6),      // %15
                    "=r"(r7)       // %16
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(outptr6),
                    "8"(outptr7),
                    "9"(r0),
                    "10"(r1),
                    "11"(r2),
                    "12"(r3),
                    "13"(r4),
                    "14"(r5),
                    "15"(r6),
                    "16"(r7),
                    "w"(_k0),                                                                            // %34
                    "w"(_k1),                                                                            // %35
                    "w"(_k2),                                                                            // %36
                    "w"(_k3),                                                                            // %37
                    "w"(_k4),                                                                            // %38
                    "w"(_k5),                                                                            // %39
                    "w"(_k6),                                                                            // %40
                    "w"(_k7),                                                                            // %41
                    "w"(_k0n),                                                                           // %42
                    "w"(_k1n),                                                                           // %43
                    "w"(_k2n),                                                                           // %44
                    "w"(_k3n),                                                                           // %45
                    "w"(_k4n),                                                                           // %46
                    "w"(_k5n),                                                                           // %47
                    "w"(_k6n),                                                                           // %48
                    "w"(_k7n)                                                                            // %49
                    : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25" //, "v26", "v27", "v28", "v29", "v30", "v31"
                );
            }
#else
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);

                float32x4_t _out0p = vld1q_f32(outptr0);
                float32x4_t _out1p = vld1q_f32(outptr1);
                float32x4_t _out2p = vld1q_f32(outptr2);
                float32x4_t _out3p = vld1q_f32(outptr3);
                float32x4_t _out4p = vld1q_f32(outptr4);
                float32x4_t _out5p = vld1q_f32(outptr5);
                float32x4_t _out6p = vld1q_f32(outptr6);
                float32x4_t _out7p = vld1q_f32(outptr7);

                _out0p = vfmaq_laneq_f32(_out0p, _p, _k0, 0);
                _out1p = vfmaq_laneq_f32(_out1p, _p, _k1, 0);
                _out2p = vfmaq_laneq_f32(_out2p, _p, _k2, 0);
                _out3p = vfmaq_laneq_f32(_out3p, _p, _k3, 0);
                _out4p = vfmaq_laneq_f32(_out4p, _p, _k4, 0);
                _out5p = vfmaq_laneq_f32(_out5p, _p, _k5, 0);
                _out6p = vfmaq_laneq_f32(_out6p, _p, _k6, 0);
                _out7p = vfmaq_laneq_f32(_out7p, _p, _k7, 0);

                float32x4_t _p1 = vld1q_f32(r1);

                _out0p = vfmaq_laneq_f32(_out0p, _p1, _k0, 1);
                _out1p = vfmaq_laneq_f32(_out1p, _p1, _k1, 1);
                _out2p = vfmaq_laneq_f32(_out2p, _p1, _k2, 1);
                _out3p = vfmaq_laneq_f32(_out3p, _p1, _k3, 1);
                _out4p = vfmaq_laneq_f32(_out4p, _p1, _k4, 1);
                _out5p = vfmaq_laneq_f32(_out5p, _p1, _k5, 1);
                _out6p = vfmaq_laneq_f32(_out6p, _p1, _k6, 1);
                _out7p = vfmaq_laneq_f32(_out7p, _p1, _k7, 1);

                float32x4_t _p2 = vld1q_f32(r2);

                _out0p = vfmaq_laneq_f32(_out0p, _p2, _k0, 2);
                _out1p = vfmaq_laneq_f32(_out1p, _p2, _k1, 2);
                _out2p = vfmaq_laneq_f32(_out2p, _p2, _k2, 2);
                _out3p = vfmaq_laneq_f32(_out3p, _p2, _k3, 2);
                _out4p = vfmaq_laneq_f32(_out4p, _p2, _k4, 2);
                _out5p = vfmaq_laneq_f32(_out5p, _p2, _k5, 2);
                _out6p = vfmaq_laneq_f32(_out6p, _p2, _k6, 2);
                _out7p = vfmaq_laneq_f32(_out7p, _p2, _k7, 2);

                float32x4_t _p3 = vld1q_f32(r3);

                _out0p = vfmaq_laneq_f32(_out0p, _p3, _k0, 3);
                _out1p = vfmaq_laneq_f32(_out1p, _p3, _k1, 3);
                _out2p = vfmaq_laneq_f32(_out2p, _p3, _k2, 3);
                _out3p = vfmaq_laneq_f32(_out3p, _p3, _k3, 3);
                _out4p = vfmaq_laneq_f32(_out4p, _p3, _k4, 3);
                _out5p = vfmaq_laneq_f32(_out5p, _p3, _k5, 3);
                _out6p = vfmaq_laneq_f32(_out6p, _p3, _k6, 3);
                _out7p = vfmaq_laneq_f32(_out7p, _p3, _k7, 3);

                float32x4_t _p4 = vld1q_f32(r4);

                _out0p = vfmaq_laneq_f32(_out0p, _p4, _k0n, 0);
                _out1p = vfmaq_laneq_f32(_out1p, _p4, _k1n, 0);
                _out2p = vfmaq_laneq_f32(_out2p, _p4, _k2n, 0);
                _out3p = vfmaq_laneq_f32(_out3p, _p4, _k3n, 0);
                _out4p = vfmaq_laneq_f32(_out4p, _p4, _k4n, 0);
                _out5p = vfmaq_laneq_f32(_out5p, _p4, _k5n, 0);
                _out6p = vfmaq_laneq_f32(_out6p, _p4, _k6n, 0);
                _out7p = vfmaq_laneq_f32(_out7p, _p4, _k7n, 0);

                float32x4_t _p5 = vld1q_f32(r5);

                _out0p = vfmaq_laneq_f32(_out0p, _p5, _k0n, 1);
                _out1p = vfmaq_laneq_f32(_out1p, _p5, _k1n, 1);
                _out2p = vfmaq_laneq_f32(_out2p, _p5, _k2n, 1);
                _out3p = vfmaq_laneq_f32(_out3p, _p5, _k3n, 1);
                _out4p = vfmaq_laneq_f32(_out4p, _p5, _k4n, 1);
                _out5p = vfmaq_laneq_f32(_out5p, _p5, _k5n, 1);
                _out6p = vfmaq_laneq_f32(_out6p, _p5, _k6n, 1);
                _out7p = vfmaq_laneq_f32(_out7p, _p5, _k7n, 1);

                float32x4_t _p6 = vld1q_f32(r6);

                _out0p = vfmaq_laneq_f32(_out0p, _p6, _k0n, 2);
                _out1p = vfmaq_laneq_f32(_out1p, _p6, _k1n, 2);
                _out2p = vfmaq_laneq_f32(_out2p, _p6, _k2n, 2);
                _out3p = vfmaq_laneq_f32(_out3p, _p6, _k3n, 2);
                _out4p = vfmaq_laneq_f32(_out4p, _p6, _k4n, 2);
                _out5p = vfmaq_laneq_f32(_out5p, _p6, _k5n, 2);
                _out6p = vfmaq_laneq_f32(_out6p, _p6, _k6n, 2);
                _out7p = vfmaq_laneq_f32(_out7p, _p6, _k7n, 2);

                float32x4_t _p7 = vld1q_f32(r7);

                _out0p = vfmaq_laneq_f32(_out0p, _p7, _k0n, 3);
                _out1p = vfmaq_laneq_f32(_out1p, _p7, _k1n, 3);
                _out2p = vfmaq_laneq_f32(_out2p, _p7, _k2n, 3);
                _out3p = vfmaq_laneq_f32(_out3p, _p7, _k3n, 3);
                _out4p = vfmaq_laneq_f32(_out4p, _p7, _k4n, 3);
                _out5p = vfmaq_laneq_f32(_out5p, _p7, _k5n, 3);
                _out6p = vfmaq_laneq_f32(_out6p, _p7, _k6n, 3);
                _out7p = vfmaq_laneq_f32(_out7p, _p7, _k7n, 3);

                vst1q_f32(outptr0, _out0p);
                vst1q_f32(outptr1, _out1p);
                vst1q_f32(outptr2, _out2p);
                vst1q_f32(outptr3, _out3p);
                vst1q_f32(outptr4, _out4p);
                vst1q_f32(outptr5, _out5p);
                vst1q_f32(outptr6, _out6p);
                vst1q_f32(outptr7, _out7p);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                r6 += 4;
                r7 += 4;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
                outptr4 += 4;
                outptr5 += 4;
                outptr6 += 4;
                outptr7 += 4;
            }
#endif
            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];
                float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7];
                float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7];
                float sum6 = *r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7];
                float sum7 = *r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;
            float* outptr6 = out6;
            float* outptr7 = out7;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;
            const float* kernel4 = kernel + (p + 4) * inch + q;
            const float* kernel5 = kernel + (p + 5) * inch + q;
            const float* kernel6 = kernel + (p + 6) * inch + q;
            const float* kernel7 = kernel + (p + 7) * inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];
            const float k4 = kernel4[0];
            const float k5 = kernel5[0];
            const float k6 = kernel6[0];
            const float k7 = kernel7[0];

            const float* r0 = img0;

            int size = outw * outh;

            int nn = size >> 2;
            int remain = size & 3;

            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
            float32x4_t _k4 = vdupq_n_f32(k4);
            float32x4_t _k5 = vdupq_n_f32(k5);
            float32x4_t _k6 = vdupq_n_f32(k6);
            float32x4_t _k7 = vdupq_n_f32(k7);

            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);

                float32x4_t _out0p = vld1q_f32(outptr0);
                float32x4_t _out1p = vld1q_f32(outptr1);
                float32x4_t _out2p = vld1q_f32(outptr2);
                float32x4_t _out3p = vld1q_f32(outptr3);
                float32x4_t _out4p = vld1q_f32(outptr4);
                float32x4_t _out5p = vld1q_f32(outptr5);
                float32x4_t _out6p = vld1q_f32(outptr6);
                float32x4_t _out7p = vld1q_f32(outptr7);

                _out0p = vfmaq_f32(_out0p, _p, _k0);
                _out1p = vfmaq_f32(_out1p, _p, _k1);
                _out2p = vfmaq_f32(_out2p, _p, _k2);
                _out3p = vfmaq_f32(_out3p, _p, _k3);
                _out4p = vfmaq_f32(_out4p, _p, _k4);
                _out5p = vfmaq_f32(_out5p, _p, _k5);
                _out6p = vfmaq_f32(_out6p, _p, _k6);
                _out7p = vfmaq_f32(_out7p, _p, _k7);

                vst1q_f32(outptr0, _out0p);
                vst1q_f32(outptr1, _out1p);
                vst1q_f32(outptr2, _out2p);
                vst1q_f32(outptr3, _out3p);
                vst1q_f32(outptr4, _out4p);
                vst1q_f32(outptr5, _out5p);
                vst1q_f32(outptr6, _out6p);
                vst1q_f32(outptr7, _out7p);

                r0 += 4;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
                outptr4 += 4;
                outptr5 += 4;
                outptr6 += 4;
                outptr7 += 4;
            }
            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;
                float sum4 = *r0 * k4;
                float sum5 = *r0 * k5;
                float sum6 = *r0 * k6;
                float sum7 = *r0 * k7;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;
            }
        }
    }

#else

    nn_outch = outch / 6;
    remain_outch_start = nn_outch * 6;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 6;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);
        Mat out4 = top_blob.channel(p + 4);
        Mat out5 = top_blob.channel(p + 5);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;
        const float bias4 = bias ? bias[p + 4] : 0.f;
        const float bias5 = bias ? bias[p + 5] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);
        out4.fill(bias4);
        out5.fill(bias5);

        int q = 0;

        for (; q + 3 < inch; q += 4)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;
            const float* kernel4 = kernel + (p + 4) * inch + q;
            const float* kernel5 = kernel + (p + 5) * inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size & 3;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);
            float32x4_t _k4 = vld1q_f32(kernel4);
            float32x4_t _k5 = vld1q_f32(kernel5);

            for (; nn > 0; nn--)
            {
                asm volatile(
                    "pld        [%6, #128]              \n"
                    "vld1.f32   {d24-d25}, [%6 :128]!   \n" // q12 = r0

                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d12-d13}, [%0 :128]    \n" // q6 = outptr0

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d14-d15}, [%1 :128]    \n" // q7 = outptr1

                    "vmla.f32   q6, q12, %e20[0]        \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d16-d17}, [%2 :128]    \n" // q8 = outptr2

                    "vmla.f32   q7, q12, %e21[0]        \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d18-d19}, [%3 :128]    \n" // q9 = outptr3

                    "vmla.f32   q8, q12, %e22[0]        \n"

                    "pld        [%7, #128]              \n"
                    "vld1.f32   {d26-d27}, [%7 :128]!   \n" // q13 = r1

                    "vmla.f32   q9, q12, %e23[0]        \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d20-d21}, [%4 :128]    \n" // q10 = outptr4

                    "vmla.f32   q6, q13, %e20[1]        \n"
                    "vmla.f32   q7, q13, %e21[1]        \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d22-d23}, [%5 :128]    \n" // q11 = outptr5

                    "vmla.f32   q10, q12, %e24[0]       \n"
                    "vmla.f32   q11, q12, %e25[0]       \n"

                    "vmla.f32   q8, q13, %e22[1]        \n"
                    "vmla.f32   q9, q13, %e23[1]        \n"

                    "pld        [%8, #128]              \n"
                    "vld1.f32   {d28-d29}, [%8 :128]!   \n" // q14 = r2

                    "vmla.f32   q10, q13, %e24[1]       \n"
                    "vmla.f32   q11, q13, %e25[1]       \n"

                    "vmla.f32   q6, q14, %f20[0]        \n"
                    "vmla.f32   q7, q14, %f21[0]        \n"
                    "vmla.f32   q8, q14, %f22[0]        \n"
                    "vmla.f32   q9, q14, %f23[0]        \n"

                    "pld        [%9, #128]             \n"
                    "vld1.f32   {d30-d31}, [%9 :128]!  \n" // q15 = r3

                    "vmla.f32   q10, q14, %f24[0]       \n"
                    "vmla.f32   q11, q14, %f25[0]       \n"

                    "vmla.f32   q6, q15, %f20[1]        \n"
                    "vmla.f32   q7, q15, %f21[1]        \n"
                    "vmla.f32   q8, q15, %f22[1]        \n"
                    "vmla.f32   q9, q15, %f23[1]        \n"

                    "vmla.f32   q10, q15, %f24[1]       \n"
                    "vmla.f32   q11, q15, %f25[1]       \n"

                    "vst1.f32   {d12-d13}, [%0 :128]!   \n"
                    "vst1.f32   {d14-d15}, [%1 :128]!   \n"
                    "vst1.f32   {d16-d17}, [%2 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%3 :128]!   \n"
                    "vst1.f32   {d20-d21}, [%4 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%5 :128]!   \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(outptr2), // %2
                    "=r"(outptr3), // %3
                    "=r"(outptr4), // %4
                    "=r"(outptr5), // %5
                    "=r"(r0),      // %6
                    "=r"(r1),      // %7
                    "=r"(r2),      // %8
                    "=r"(r3)       // %9
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(outptr2),
                    "3"(outptr3),
                    "4"(outptr4),
                    "5"(outptr5),
                    "6"(r0),
                    "7"(r1),
                    "8"(r2),
                    "9"(r3),
                    "w"(_k0), // %20
                    "w"(_k1), // %21
                    "w"(_k2), // %22
                    "w"(_k3), // %23
                    "w"(_k4), // %24
                    "w"(_k5)  // %25
                    : "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3];
                float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;
            float* outptr4 = out4;
            float* outptr5 = out5;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;
            const float* kernel4 = kernel + (p + 4) * inch + q;
            const float* kernel5 = kernel + (p + 5) * inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];
            const float k4 = kernel4[0];
            const float k5 = kernel5[0];

            const float* r0 = img0;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size & 3;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
            float32x4_t _k4 = vdupq_n_f32(k4);
            float32x4_t _k5 = vdupq_n_f32(k5);

            if (nn > 0)
            {
                asm volatile(
                    "pld        [%7, #128]              \n"
                    "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                    "0:                                 \n"

                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d14-d15}, [%2 :128]    \n" // q7 = outptr1

                    "vmla.f32   q6, q12, %q16           \n"

                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d16-d17}, [%3 :128]    \n" // q8 = outptr2

                    "vmla.f32   q7, q12, %q17           \n"

                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d18-d19}, [%4 :128]    \n" // q9 = outptr3

                    "vmla.f32   q8, q12, %q18           \n"

                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d20-d21}, [%5 :128]    \n" // q10 = outptr4

                    "vmla.f32   q9, q12, %q19           \n"

                    "pld        [%6, #128]              \n"
                    "vld1.f32   {d22-d23}, [%6 :128]    \n" // q11 = outptr5

                    "vmla.f32   q10, q12, %q20          \n"
                    "vmla.f32   q11, q12, %q21          \n"

                    "pld        [%7, #128]              \n"
                    "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                    "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                    "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                    "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                    "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                    "subs       %0, #1                  \n"

                    "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                    "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                    "bne        0b                      \n"

                    "sub        %7, #16                 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(r0)       // %7
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(r0),
                    "w"(_k0), // %16
                    "w"(_k1), // %17
                    "w"(_k2), // %18
                    "w"(_k3), // %19
                    "w"(_k4), // %20
                    "w"(_k5)  // %21
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;
                float sum4 = *r0 * k4;
                float sum5 = *r0 * k5;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
            }
        }
    }
#endif // __ARM_NEON && __aarch64__

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);

        int q = 0;

        for (; q + 3 < inch; q += 4)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);
            float32x4_t _k2 = vld1q_f32(kernel2);
            float32x4_t _k3 = vld1q_f32(kernel3);

#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v8.4s, v9.4s}, [%1]        \n"

                    "0:                                 \n"

                    "fmla   v8.4s, v6.4s, %18.s[0]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v10.4s, v11.4s}, [%2]      \n"

                    "fmla   v9.4s, v7.4s, %18.s[0]      \n"

                    "fmla   v10.4s, v6.4s, %19.s[0]     \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v12.4s, v13.4s}, [%3]      \n"

                    "fmla   v11.4s, v7.4s, %19.s[0]     \n"

                    "fmla   v12.4s, v6.4s, %20.s[0]     \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v14.4s, v15.4s}, [%4]      \n"

                    "fmla   v13.4s, v7.4s, %20.s[0]     \n"

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%6], #32   \n"

                    "fmla   v14.4s, v6.4s, %21.s[0]     \n"
                    "fmla   v15.4s, v7.4s, %21.s[0]     \n"

                    "fmla   v8.4s, v4.4s, %18.s[1]      \n"
                    "fmla   v9.4s, v5.4s, %18.s[1]      \n"

                    "fmla   v10.4s, v4.4s, %19.s[1]     \n"
                    "fmla   v11.4s, v5.4s, %19.s[1]     \n"

                    "fmla   v12.4s, v4.4s, %20.s[1]     \n"
                    "fmla   v13.4s, v5.4s, %20.s[1]     \n"

                    "prfm   pldl1keep, [%7, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%7], #32   \n"

                    "fmla   v14.4s, v4.4s, %21.s[1]     \n"
                    "fmla   v15.4s, v5.4s, %21.s[1]     \n"

                    "fmla   v8.4s, v6.4s, %18.s[2]      \n"
                    "fmla   v9.4s, v7.4s, %18.s[2]      \n"

                    "fmla   v10.4s, v6.4s, %19.s[2]     \n"
                    "fmla   v11.4s, v7.4s, %19.s[2]     \n"

                    "fmla   v12.4s, v6.4s, %20.s[2]     \n"
                    "fmla   v13.4s, v7.4s, %20.s[2]     \n"

                    "prfm   pldl1keep, [%8, #256]       \n"
                    "ld1    {v4.4s, v5.4s}, [%8], #32   \n"

                    "fmla   v14.4s, v6.4s, %21.s[2]     \n"
                    "fmla   v15.4s, v7.4s, %21.s[2]     \n"

                    "fmla   v8.4s, v4.4s, %18.s[3]      \n"
                    "fmla   v9.4s, v5.4s, %18.s[3]      \n"

                    "fmla   v10.4s, v4.4s, %19.s[3]     \n"
                    "fmla   v11.4s, v5.4s, %19.s[3]     \n"

                    "st1    {v8.4s, v9.4s}, [%1], #32   \n"

                    "fmla   v12.4s, v4.4s, %20.s[3]     \n"
                    "fmla   v13.4s, v5.4s, %20.s[3]     \n"

                    "st1    {v10.4s, v11.4s}, [%2], #32 \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                    "fmla   v14.4s, v4.4s, %21.s[3]     \n"
                    "fmla   v15.4s, v5.4s, %21.s[3]     \n"

                    "st1    {v12.4s, v13.4s}, [%3], #32 \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v8.4s, v9.4s}, [%1]        \n"

                    "subs   %w0, %w0, #1                \n"

                    "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                    "bne    0b                          \n"
                    "sub    %5, %5, #32                 \n"
                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(r0),      // %5
                    "=r"(r1),      // %6
                    "=r"(r2),      // %7
                    "=r"(r3)       // %8
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(r0),
                    "6"(r1),
                    "7"(r2),
                    "8"(r3),
                    "w"(_k0), // %18
                    "w"(_k1), // %19
                    "w"(_k2), // %20
                    "w"(_k3)  // %21
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"
                    "0:                                 \n"

                    "vmla.f32   q8, q6, %e18[0]         \n"

                    "pld        [%2, #256]              \n"
                    "vld1.f32   {d20-d23}, [%2 :128]    \n"
                    "vmla.f32   q9, q7, %e18[0]         \n"

                    "vmla.f32   q10, q6, %e19[0]        \n"

                    "pld        [%3, #256]              \n"
                    "vld1.f32   {d24-d27}, [%3 :128]    \n"
                    "vmla.f32   q11, q7, %e19[0]        \n"

                    "vmla.f32   q12, q6, %e20[0]        \n"

                    "pld        [%4, #256]              \n"
                    "vld1.f32   {d28-d31}, [%4 :128]    \n"
                    "vmla.f32   q13, q7, %e20[0]        \n"

                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d8-d11}, [%6 :128]!    \n"

                    "vmla.f32   q14, q6, %e21[0]        \n"
                    "vmla.f32   q15, q7, %e21[0]        \n"

                    "vmla.f32   q8, q4, %e18[1]         \n"
                    "vmla.f32   q9, q5, %e18[1]         \n"

                    "vmla.f32   q10, q4, %e19[1]        \n"
                    "vmla.f32   q11, q5, %e19[1]        \n"

                    "vmla.f32   q12, q4, %e20[1]        \n"
                    "vmla.f32   q13, q5, %e20[1]        \n"

                    "pld        [%7, #256]              \n"
                    "vld1.f32   {d12-d15}, [%7 :128]!   \n"

                    "vmla.f32   q14, q4, %e21[1]        \n"
                    "vmla.f32   q15, q5, %e21[1]        \n"

                    "vmla.f32   q8, q6, %f18[0]         \n"
                    "vmla.f32   q9, q7, %f18[0]         \n"

                    "vmla.f32   q10, q6, %f19[0]        \n"
                    "vmla.f32   q11, q7, %f19[0]        \n"

                    "vmla.f32   q12, q6, %f20[0]        \n"
                    "vmla.f32   q13, q7, %f20[0]        \n"

                    "pld        [%8, #256]              \n"
                    "vld1.f32   {d8-d11}, [%8 :128]!    \n"

                    "vmla.f32   q14, q6, %f21[0]        \n"
                    "vmla.f32   q15, q7, %f21[0]        \n"

                    "vmla.f32   q8, q4, %f18[1]         \n"
                    "vmla.f32   q9, q5, %f18[1]         \n"

                    "vmla.f32   q10, q4, %f19[1]        \n"
                    "vmla.f32   q11, q5, %f19[1]        \n"

                    "vmla.f32   q12, q4, %f20[1]        \n"
                    "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                    "vmla.f32   q13, q5, %f20[1]        \n"

                    "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                    "vmla.f32   q14, q4, %f21[1]        \n"
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"

                    "vmla.f32   q15, q5, %f21[1]        \n"

                    "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"

                    "subs       %0, #1                  \n"
                    "vst1.f32   {d28-d31}, [%4 :128]!   \n"

                    "bne        0b                      \n"
                    "sub        %5, #32                 \n"
                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(r0),      // %5
                    "=r"(r1),      // %6
                    "=r"(r2),      // %7
                    "=r"(r3)       // %8
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(r0),
                    "6"(r1),
                    "7"(r2),
                    "8"(r3),
                    "w"(_k0), // %18
                    "w"(_k1), // %19
                    "w"(_k2), // %20
                    "w"(_k3)  // %21
                    : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float* r0 = img0;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"
                    "fmla       v8.4s, v6.4s, %12.4s           \n"
                    "fmla       v9.4s, v7.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"
                    "fmla       v10.4s, v6.4s, %13.4s          \n"
                    "fmla       v11.4s, v7.4s, %13.4s          \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"
                    "fmla       v12.4s, v6.4s, %14.4s          \n"
                    "fmla       v13.4s, v7.4s, %14.4s          \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"
                    "fmla       v14.4s, v6.4s, %15.4s          \n"
                    "fmla       v15.4s, v7.4s, %15.4s          \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                    "bne        0b                             \n"
                    "sub        %5, %5, #32                    \n"
                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(r0)       // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(r0),
                    "w"(_k0), // %12
                    "w"(_k1), // %13
                    "w"(_k2), // %14
                    "w"(_k3)  // %15
                    : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "0:                                 \n"
                    "pld        [%1, #256]              \n"
                    "vld1.f32   {d16-d19}, [%1 :128]    \n"
                    "vmla.f32   q8, q6, %q12            \n"
                    "vmla.f32   q9, q7, %q12            \n"

                    "pld        [%2, #256]              \n"
                    "vld1.f32   {d20-d23}, [%2 :128]    \n"
                    "vmla.f32   q10, q6, %q13           \n"
                    "vmla.f32   q11, q7, %q13           \n"

                    "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                    "pld        [%3, #256]              \n"
                    "vld1.f32   {d24-d27}, [%3 :128]    \n"
                    "vmla.f32   q12, q6, %q14           \n"
                    "vmla.f32   q13, q7, %q14           \n"

                    "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                    "pld        [%4, #256]              \n"
                    "vld1.f32   {d28-d31}, [%4 :128]    \n"
                    "vmla.f32   q14, q6, %q15           \n"
                    "vmla.f32   q15, q7, %q15           \n"

                    "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                    "subs       %0, #1                  \n"
                    "vst1.f32   {d28-d31}, [%4 :128]!   \n"
                    "bne        0b                      \n"
                    "sub        %5, #32                 \n"
                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(r0)       // %5
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(r0),
                    "w"(_k0), // %12
                    "w"(_k1), // %13
                    "w"(_k2), // %14
                    "w"(_k3)  // %15
                    : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                // TODO neon optimize
                float sum0 = *r0 * k0;
                float sum1 = *r0 * k1;
                float sum2 = *r0 * k2;
                float sum3 = *r0 * k3;

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q + 3 < inch; q += 4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %12.4s           \n"
                    "fmla       v1.4s, v3.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%3], #32      \n"
                    "fmla       v0.4s, v2.4s, %13.4s           \n"
                    "fmla       v1.4s, v3.4s, %13.4s           \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%4], #32      \n"
                    "fmla       v0.4s, v2.4s, %14.4s           \n"
                    "fmla       v1.4s, v3.4s, %14.4s           \n"

                    "prfm       pldl1keep, [%5, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%5], #32      \n"
                    "fmla       v0.4s, v2.4s, %15.4s           \n"
                    "fmla       v1.4s, v3.4s, %15.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                    "=r"(outptr), // %1
                    "=r"(r0),     // %2
                    "=r"(r1),     // %3
                    "=r"(r2),     // %4
                    "=r"(r3)      // %5
                    : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k0), // %12
                    "w"(_k1), // %13
                    "w"(_k2), // %14
                    "w"(_k3)  // %15
                    : "cc", "memory", "v0", "v1", "v2", "v3");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]  \n"
                    "vmla.f32   q0, q2, %q12        \n"
                    "vmla.f32   q1, q3, %q12        \n"
                    "pld        [%3, #256]          \n"
                    "vld1.f32   {d4-d7}, [%3 :128]! \n"
                    "vmla.f32   q0, q2, %q13        \n"
                    "vmla.f32   q1, q3, %q13        \n"
                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d4-d7}, [%4 :128]! \n"
                    "vmla.f32   q0, q2, %q14        \n"
                    "vmla.f32   q1, q3, %q14        \n"
                    "pld        [%5, #256]          \n"
                    "vld1.f32   {d4-d7}, [%5 :128]! \n"
                    "vmla.f32   q0, q2, %q15        \n"
                    "vmla.f32   q1, q3, %q15        \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                    "=r"(outptr), // %1
                    "=r"(r0),     // %2
                    "=r"(r1),     // %3
                    "=r"(r2),     // %4
                    "=r"(r3)      // %5
                    : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "w"(_k0), // %12
                    "w"(_k1), // %13
                    "w"(_k2), // %14
                    "w"(_k3)  // %15
                    : "cc", "memory", "q0", "q1", "q2", "q3");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
            if (nn > 0)
            {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %6.4s            \n"
                    "fmla       v1.4s, v3.4s, %6.4s            \n"
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                    "=r"(outptr), // %1
                    "=r"(r0)      // %2
                    : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "w"(_k0) // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3");
            }
#else
            if (nn > 0)
            {
                asm volatile(
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1 :128]  \n"
                    "vmla.f32   q0, q2, %q6         \n"
                    "vmla.f32   q1, q3, %q6         \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2 :128]! \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    "bne        0b                  \n"
                    "sub        %2, #32             \n"
                    : "=r"(nn),     // %0
                    "=r"(outptr), // %1
                    "=r"(r0)      // %2
                    : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "w"(_k0) // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3");
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain > 0; remain--)
            {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
        }
    }
}

static void conv1x1s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;

        out0.fill(bias0);
        out1.fill(bias1);
        out2.fill(bias2);
        out3.fill(bias3);

        int q = 0;

        for (; q + 3 < inch; q += 4)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
                int size = outw;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vld1q_f32(kernel0);
                float32x4_t _k1 = vld1q_f32(kernel1);
                float32x4_t _k2 = vld1q_f32(kernel2);
                float32x4_t _k3 = vld1q_f32(kernel3);
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                        \n"

                        "prfm       pldl1keep, [%5, #512]          \n"
                        "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                        "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                        "and        v5.16b, v6.16b, v6.16b         \n" // v4 v5

                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v8.4s, v9.4s}, [%1]           \n"

                        "fmla       v8.4s, v4.4s, %18.s[0]         \n"
                        "fmla       v9.4s, v5.4s, %18.s[0]         \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v10.4s, v11.4s}, [%2]         \n"

                        "fmla       v10.4s, v4.4s, %19.s[0]        \n"
                        "fmla       v11.4s, v5.4s, %19.s[0]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld1        {v12.4s, v13.4s}, [%3]         \n"

                        "fmla       v12.4s, v4.4s, %20.s[0]        \n"
                        "fmla       v13.4s, v5.4s, %20.s[0]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld1        {v14.4s, v15.4s}, [%4]         \n"

                        "prfm       pldl1keep, [%6, #512]          \n"
                        "ld2        {v6.4s, v7.4s}, [%6], #32      \n"

                        "fmla       v14.4s, v4.4s, %21.s[0]        \n"
                        "fmla       v15.4s, v5.4s, %21.s[0]        \n"

                        "ld2        {v4.4s, v5.4s}, [%6], #32      \n"
                        "and        v7.16b, v4.16b, v4.16b         \n" // v6 v7

                        "fmla       v8.4s, v6.4s, %18.s[1]         \n"
                        "fmla       v9.4s, v7.4s, %18.s[1]         \n"

                        "fmla       v10.4s, v6.4s, %19.s[1]        \n"
                        "fmla       v11.4s, v7.4s, %19.s[1]        \n"

                        "fmla       v12.4s, v6.4s, %20.s[1]        \n"
                        "fmla       v13.4s, v7.4s, %20.s[1]        \n"

                        "prfm       pldl1keep, [%7, #512]          \n"
                        "ld2        {v4.4s, v5.4s}, [%7], #32      \n"

                        "fmla       v14.4s, v6.4s, %21.s[1]        \n"
                        "fmla       v15.4s, v7.4s, %21.s[1]        \n"

                        "ld2        {v6.4s, v7.4s}, [%7], #32      \n"
                        "and        v5.16b, v6.16b, v6.16b         \n" // v4 v5

                        "fmla       v8.4s, v4.4s, %18.s[2]         \n"
                        "fmla       v9.4s, v5.4s, %18.s[2]         \n"

                        "fmla       v10.4s, v4.4s, %19.s[2]        \n"
                        "fmla       v11.4s, v5.4s, %19.s[2]        \n"

                        "fmla       v12.4s, v4.4s, %20.s[2]        \n"
                        "fmla       v13.4s, v5.4s, %20.s[2]        \n"

                        "prfm       pldl1keep, [%8, #512]          \n"
                        "ld2        {v6.4s, v7.4s}, [%8], #32      \n"

                        "fmla       v14.4s, v4.4s, %21.s[2]        \n"
                        "fmla       v15.4s, v5.4s, %21.s[2]        \n"

                        "ld2        {v4.4s, v5.4s}, [%8], #32      \n"
                        "and        v7.16b, v4.16b, v4.16b         \n" // v6 v7

                        "fmla       v8.4s, v6.4s, %18.s[3]         \n"
                        "fmla       v9.4s, v7.4s, %18.s[3]         \n"

                        "fmla       v10.4s, v6.4s, %19.s[3]        \n"
                        "fmla       v11.4s, v7.4s, %19.s[3]        \n"

                        "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                        "fmla       v12.4s, v6.4s, %20.s[3]        \n"
                        "fmla       v13.4s, v7.4s, %20.s[3]        \n"

                        "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                        "fmla       v14.4s, v6.4s, %21.s[3]        \n"
                        "fmla       v15.4s, v7.4s, %21.s[3]        \n"

                        "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v14.4s, v15.4s}, [%4], #32    \n"

                        "bne        0b                             \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0),      // %5
                        "=r"(r1),      // %6
                        "=r"(r2),      // %7
                        "=r"(r3)       // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k0), // %18
                        "w"(_k1), // %19
                        "w"(_k2), // %20
                        "w"(_k3)  // %21
                        : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%5, #512]          \n"
                        "vld2.f32   {d8-d11}, [%5]!     \n"
                        "vld2.f32   {d12-d15}, [%5]!    \n"
                        "vand       q5, q6, q6          \n" // q4 q5

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d16-d19}, [%1]     \n"

                        "vmla.f32   q8, q4, %e18[0]     \n"
                        "vmla.f32   q9, q5, %e18[0]     \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d20-d23}, [%2]     \n"

                        "vmla.f32   q10, q4, %e19[0]    \n"
                        "vmla.f32   q11, q5, %e19[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d24-d27}, [%3]     \n"

                        "vmla.f32   q12, q4, %e20[0]    \n"
                        "vmla.f32   q13, q5, %e20[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d28-d31}, [%4]     \n"

                        "pld        [%6, #512]          \n"
                        "vld2.f32   {d12-d15}, [%6]!    \n"

                        "vmla.f32   q14, q4, %e21[0]    \n"
                        "vmla.f32   q15, q5, %e21[0]    \n"

                        "vld2.f32   {d8-d11}, [%6]!     \n"
                        "vand       q7, q4, q4          \n" // q6 q7

                        "vmla.f32   q8, q6, %e18[1]     \n"
                        "vmla.f32   q9, q7, %e18[1]     \n"

                        "vmla.f32   q10, q6, %e19[1]    \n"
                        "vmla.f32   q11, q7, %e19[1]    \n"

                        "vmla.f32   q12, q6, %e20[1]    \n"
                        "vmla.f32   q13, q7, %e20[1]    \n"

                        "pld        [%7, #512]          \n"
                        "vld2.f32   {d8-d11}, [%7]!     \n"

                        "vmla.f32   q14, q6, %e21[1]    \n"
                        "vmla.f32   q15, q7, %e21[1]    \n"

                        "vld2.f32   {d12-d15}, [%7]!    \n"
                        "vand       q5, q6, q6          \n" // q4 q5

                        "vmla.f32   q8, q4, %f18[0]     \n"
                        "vmla.f32   q9, q5, %f18[0]     \n"

                        "vmla.f32   q10, q4, %f19[0]    \n"
                        "vmla.f32   q11, q5, %f19[0]    \n"

                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vmla.f32   q13, q5, %f20[0]    \n"

                        "pld        [%8, #512]          \n"
                        "vld2.f32   {d12-d15}, [%8]!    \n"

                        "vmla.f32   q14, q4, %f21[0]    \n"
                        "vmla.f32   q15, q5, %f21[0]    \n"

                        "vld2.f32   {d8-d11}, [%8]!     \n"
                        "vand       q7, q4, q4          \n" // q6 q7

                        "vmla.f32   q8, q6, %f18[1]     \n"
                        "vmla.f32   q9, q7, %f18[1]     \n"

                        "vmla.f32   q10, q6, %f19[1]    \n"
                        "vmla.f32   q11, q7, %f19[1]    \n"

                        "vst1.f32   {d16-d19}, [%1]!    \n"

                        "vmla.f32   q12, q6, %f20[1]    \n"
                        "vmla.f32   q13, q7, %f20[1]    \n"

                        "vst1.f32   {d20-d23}, [%2]!    \n"

                        "vmla.f32   q14, q6, %f21[1]    \n"
                        "vmla.f32   q15, q7, %f21[1]    \n"

                        "vst1.f32   {d24-d27}, [%3]!    \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d28-d31}, [%4]!    \n"

                        "bne        0b                  \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0),      // %5
                        "=r"(r1),      // %6
                        "=r"(r2),      // %7
                        "=r"(r3)       // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k0), // %18
                        "w"(_k1), // %19
                        "w"(_k2), // %20
                        "w"(_k3)  // %21
                        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr0 = out0;
            float* outptr1 = out1;
            float* outptr2 = out2;
            float* outptr3 = out3;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float* kernel1 = kernel + (p + 1) * inch + q;
            const float* kernel2 = kernel + (p + 2) * inch + q;
            const float* kernel3 = kernel + (p + 3) * inch + q;

            const float k0 = kernel0[0];
            const float k1 = kernel1[0];
            const float k2 = kernel2[0];
            const float k3 = kernel3[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
                int size = outw;

#if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                        \n"

                        "prfm       pldl1keep, [%5, #512]          \n"
                        "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                        "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                        "and        v5.16b, v6.16b, v6.16b         \n"

                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v8.4s, v9.4s}, [%1]           \n"

                        "fmla       v8.4s, v4.4s, %12.4s           \n"
                        "fmla       v9.4s, v5.4s, %12.4s           \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v10.4s, v11.4s}, [%2]         \n"

                        "fmla       v10.4s, v4.4s, %13.4s          \n"
                        "fmla       v11.4s, v5.4s, %13.4s          \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld1        {v12.4s, v13.4s}, [%3]         \n"

                        "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                        "fmla       v12.4s, v4.4s, %14.4s          \n"
                        "fmla       v13.4s, v5.4s, %14.4s          \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld1        {v14.4s, v15.4s}, [%4]         \n"

                        "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                        "fmla       v14.4s, v4.4s, %15.4s          \n"
                        "fmla       v15.4s, v5.4s, %15.4s          \n"

                        "st1        {v12.4s, v13.4s}, [%3], #32    \n"
                        "subs       %w0, %w0, #1                   \n"

                        "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                        "bne        0b                             \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0)       // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"

                        "pld        [%5, #512]          \n"
                        "vld2.f32   {d8-d11}, [%5]!     \n"
                        "vld2.f32   {d12-d15}, [%5]!    \n"
                        "vand       q5, q6, q6          \n" // q4 q5

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d16-d19}, [%1]     \n"

                        "vmla.f32   q8, q4, %q12        \n"
                        "vmla.f32   q9, q5, %q12        \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d20-d23}, [%2]     \n"

                        "vmla.f32   q10, q4, %q13       \n"
                        "vmla.f32   q11, q5, %q13       \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d24-d27}, [%3]     \n"

                        "vst1.f32   {d16-d19}, [%1]!    \n"

                        "vmla.f32   q12, q4, %q14       \n"
                        "vmla.f32   q13, q5, %q14       \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d28-d31}, [%4]     \n"

                        "vst1.f32   {d20-d23}, [%2]!    \n"

                        "vmla.f32   q14, q4, %q15       \n"
                        "vmla.f32   q15, q5, %q15       \n"

                        "vst1.f32   {d24-d27}, [%3]!    \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d28-d31}, [%4]!    \n"
                        "bne        0b                  \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0)       // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * k0;
                    float sum1 = *r0 * k1;
                    float sum2 = *r0 * k2;
                    float sum3 = *r0 * k3;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0 += 2;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }

                r0 += tailstep;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q + 3 < inch; q += 4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q + 1);
            const float* img2 = bottom_blob.channel(q + 2);
            const float* img3 = bottom_blob.channel(q + 3);

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n"
                        "0:                                        \n"

                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v0.4s, v1.4s}, [%1]           \n"
                        "fmla       v0.4s, v2.4s, %12.4s           \n"
                        "fmla       v1.4s, v8.4s, %12.4s           \n"

                        "prfm       pldl1keep, [%3, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%3], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%3], #32      \n"
                        "fmla       v0.4s, v2.4s, %13.4s           \n"
                        "fmla       v1.4s, v8.4s, %13.4s           \n"

                        "prfm       pldl1keep, [%4, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%4], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%4], #32      \n"
                        "fmla       v0.4s, v2.4s, %14.4s           \n"
                        "fmla       v1.4s, v8.4s, %14.4s           \n"

                        "prfm       pldl1keep, [%5, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%5], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%5], #32      \n"
                        "fmla       v0.4s, v2.4s, %15.4s           \n"
                        "fmla       v1.4s, v8.4s, %15.4s           \n"

                        "prfm       pldl1keep, [%2, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #64                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #512]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1]       \n"
                        "vmla.f32   q0, q2, %q12        \n"
                        "vmla.f32   q1, q8, %q12        \n"
                        "pld        [%3, #512]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"
                        "vld2.f32   {d16-d19}, [%3]!    \n"
                        "vmla.f32   q0, q2, %q13        \n"
                        "vmla.f32   q1, q8, %q13        \n"
                        "pld        [%4, #512]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"
                        "vld2.f32   {d16-d19}, [%4]!    \n"
                        "vmla.f32   q0, q2, %q14        \n"
                        "vmla.f32   q1, q8, %q14        \n"
                        "pld        [%5, #512]          \n"
                        "vld2.f32   {d4-d7}, [%5]!      \n"
                        "vld2.f32   {d16-d19}, [%5]!    \n"
                        "vmla.f32   q0, q2, %q15        \n"
                        "vmla.f32   q1, q8, %q15        \n"
                        "pld        [%2, #512]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #64             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
        }

        for (; q < inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p * inch + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                        "0:                                        \n"

                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v0.4s, v1.4s}, [%1]           \n"
                        "fmla       v0.4s, v2.4s, %6.4s            \n"
                        "fmla       v1.4s, v8.4s, %6.4s            \n"

                        "prfm       pldl1keep, [%2, #512]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #64                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0)      // %2
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "w"(_k0) // %6
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #512]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1]       \n"
                        "vmla.f32   q0, q2, %q6         \n"
                        "vmla.f32   q1, q8, %q6         \n"
                        "pld        [%2, #512]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #64             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0)      // %2
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "w"(_k0) // %6
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }
        }
    }
}
