// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sgemm_transform_kernel_pack4_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4b-4a-inch/4a-outch/4b
    kernel_tm_pack4.create(2 * 1, inch / 4, (outch / 4) / 2 + (outch / 4) % 2, (size_t)2u * 16, 16);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;
        const float* k4 = (const float*)kernel + (q + 4) * inch;
        const float* k5 = (const float*)kernel + (q + 5) * inch;
        const float* k6 = (const float*)kernel + (q + 6) * inch;
        const float* k7 = (const float*)kernel + (q + 7) * inch;

        __fp16* g0 = kernel_tm_pack4.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = (__fp16)k0[0];
            g0[1] = (__fp16)k1[0];
            g0[2] = (__fp16)k2[0];
            g0[3] = (__fp16)k3[0];
            g0[4] = (__fp16)k4[0];
            g0[5] = (__fp16)k5[0];
            g0[6] = (__fp16)k6[0];
            g0[7] = (__fp16)k7[0];

            g0[8] = (__fp16)k0[1];
            g0[9] = (__fp16)k1[1];
            g0[10] = (__fp16)k2[1];
            g0[11] = (__fp16)k3[1];
            g0[12] = (__fp16)k4[1];
            g0[13] = (__fp16)k5[1];
            g0[14] = (__fp16)k6[1];
            g0[15] = (__fp16)k7[1];

            g0[16] = (__fp16)k0[2];
            g0[17] = (__fp16)k1[2];
            g0[18] = (__fp16)k2[2];
            g0[19] = (__fp16)k3[2];
            g0[20] = (__fp16)k4[2];
            g0[21] = (__fp16)k5[2];
            g0[22] = (__fp16)k6[2];
            g0[23] = (__fp16)k7[2];

            g0[24] = (__fp16)k0[3];
            g0[25] = (__fp16)k1[3];
            g0[26] = (__fp16)k2[3];
            g0[27] = (__fp16)k3[3];
            g0[28] = (__fp16)k4[3];
            g0[29] = (__fp16)k5[3];
            g0[30] = (__fp16)k6[3];
            g0[31] = (__fp16)k7[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            k4 += 4;
            k5 += 4;
            k6 += 4;
            k7 += 4;
            g0 += 32;
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;

        __fp16* g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = (__fp16)k0[0];
            g0[1] = (__fp16)k1[0];
            g0[2] = (__fp16)k2[0];
            g0[3] = (__fp16)k3[0];

            g0[4] = (__fp16)k0[1];
            g0[5] = (__fp16)k1[1];
            g0[6] = (__fp16)k2[1];
            g0[7] = (__fp16)k3[1];

            g0[8] = (__fp16)k0[2];
            g0[9] = (__fp16)k1[2];
            g0[10] = (__fp16)k2[2];
            g0[11] = (__fp16)k3[2];

            g0[12] = (__fp16)k0[3];
            g0[13] = (__fp16)k1[3];
            g0[14] = (__fp16)k2[3];
            g0[15] = (__fp16)k3[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            g0 += 16;
        }
    }
}

static void conv1x1s1_sgemm_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const __fp16* bias = _bias;

    // interleave
    Mat tmp;
    if (size >= 8)
        tmp.create(8, inch, size / 8 + (size % 8) / 4 + size % 4, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4, inch, size / 4 + size % 4, elemsize, elempack, opt.workspace_allocator);
    else // if (size >= 1)
        tmp.create(1, inch, size, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size;
        int remain_size_start = 0;

        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");

                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x4
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld4    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0] \n"
                    "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");

                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]    \n"
                    "ld1    {v0.4h}, [%0]           \n"
                    "st1    {v0.4h}, [%1], #8       \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0");

                img0 += bottom_blob.cstep * 4;
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        __fp16* outptr0 = top_blob.channel(p);
        __fp16* outptr1 = top_blob.channel(p + 1);

        const __fp16 zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 4 : zeros;
        float16x8_t _bias0 = vld1q_f16(biasptr);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(pp);

            int nn = inch; // inch always > 0

            asm volatile(
                "mov    v24.16b, %10.16b            \n"
                "mov    v25.16b, %10.16b            \n"
                "mov    v26.16b, %10.16b            \n"
                "mov    v27.16b, %10.16b            \n"
                "mov    v28.16b, %10.16b            \n"
                "mov    v29.16b, %10.16b            \n"
                "mov    v30.16b, %10.16b            \n"
                "mov    v31.16b, %10.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r01 r23 r45 r67

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%4], #64 \n" // k0123

                "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                "fmla   v27.8h, v4.8h, v0.h[3]      \n"
                "fmla   v28.8h, v4.8h, v0.h[4]      \n"
                "fmla   v29.8h, v4.8h, v0.h[5]      \n"
                "fmla   v30.8h, v4.8h, v0.h[6]      \n"
                "fmla   v31.8h, v4.8h, v0.h[7]      \n"

                "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                "fmla   v27.8h, v5.8h, v1.h[3]      \n"
                "fmla   v28.8h, v5.8h, v1.h[4]      \n"
                "fmla   v29.8h, v5.8h, v1.h[5]      \n"
                "fmla   v30.8h, v5.8h, v1.h[6]      \n"
                "fmla   v31.8h, v5.8h, v1.h[7]      \n"

                "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                "fmla   v27.8h, v6.8h, v2.h[3]      \n"
                "fmla   v28.8h, v6.8h, v2.h[4]      \n"
                "fmla   v29.8h, v6.8h, v2.h[5]      \n"
                "fmla   v30.8h, v6.8h, v2.h[6]      \n"
                "fmla   v31.8h, v6.8h, v2.h[7]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                "fmla   v27.8h, v7.8h, v3.h[3]      \n"
                "fmla   v28.8h, v7.8h, v3.h[4]      \n"
                "fmla   v29.8h, v7.8h, v3.h[5]      \n"
                "fmla   v30.8h, v7.8h, v3.h[6]      \n"
                "fmla   v31.8h, v7.8h, v3.h[7]      \n"

                "bne    0b                          \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"
                "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%1], #32 \n"

                "ext    v24.16b, v24.16b, v24.16b, #8 \n"
                "ext    v25.16b, v25.16b, v25.16b, #8 \n"
                "ext    v26.16b, v26.16b, v26.16b, #8 \n"
                "ext    v27.16b, v27.16b, v27.16b, #8 \n"
                "ext    v28.16b, v28.16b, v28.16b, #8 \n"
                "ext    v29.16b, v29.16b, v29.16b, #8 \n"
                "ext    v30.16b, v30.16b, v30.16b, #8 \n"
                "ext    v31.16b, v31.16b, v31.16b, #8 \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"
                "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr)     // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr),
                "w"(_bias0) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(pp);

            int nn = inch; // inch always > 0

            asm volatile(
                "mov    v24.16b, %10.16b            \n"
                "mov    v25.16b, %10.16b            \n"
                "mov    v26.16b, %10.16b            \n"
                "mov    v27.16b, %10.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r01 r23 r45 r67

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%4], #64 \n" // k0123

                "fmla   v24.8h, v4.8h, v0.h[0]      \n"
                "fmla   v25.8h, v4.8h, v0.h[1]      \n"
                "fmla   v26.8h, v4.8h, v0.h[2]      \n"
                "fmla   v27.8h, v4.8h, v0.h[3]      \n"

                "fmla   v24.8h, v5.8h, v1.h[0]      \n"
                "fmla   v25.8h, v5.8h, v1.h[1]      \n"
                "fmla   v26.8h, v5.8h, v1.h[2]      \n"
                "fmla   v27.8h, v5.8h, v1.h[3]      \n"

                "fmla   v24.8h, v6.8h, v2.h[0]      \n"
                "fmla   v25.8h, v6.8h, v2.h[1]      \n"
                "fmla   v26.8h, v6.8h, v2.h[2]      \n"
                "fmla   v27.8h, v6.8h, v2.h[3]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.8h, v7.8h, v3.h[0]      \n"
                "fmla   v25.8h, v7.8h, v3.h[1]      \n"
                "fmla   v26.8h, v7.8h, v3.h[2]      \n"
                "fmla   v27.8h, v7.8h, v3.h[3]      \n"

                "bne    0b                          \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"

                "ext    v24.16b, v24.16b, v24.16b, #8 \n"
                "ext    v25.16b, v25.16b, v25.16b, #8 \n"
                "ext    v26.16b, v26.16b, v26.16b, #8 \n"
                "ext    v27.16b, v27.16b, v27.16b, #8 \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr)     // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr),
                "w"(_bias0) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(pp);

            float16x8_t _sum0 = _bias0;

            for (int q = 0; q < inch; q++)
            {
                float16x4_t _r0 = vld1_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);
                float16x8_t _k1 = vld1q_f16(kptr + 8);
                float16x8_t _k2 = vld1q_f16(kptr + 16);
                float16x8_t _k3 = vld1q_f16(kptr + 24);

                _sum0 = vfmaq_lane_f16(_sum0, _k0, _r0, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _k1, _r0, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _k2, _r0, 2);
                _sum0 = vfmaq_lane_f16(_sum0, _k3, _r0, 3);

                kptr += 32;
                tmpptr += 4;
            }

            vst1_f16(outptr0, vget_low_f16(_sum0));
            vst1_f16(outptr1, vget_high_f16(_sum0));

            outptr0 += 4;
            outptr1 += 4;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        const __fp16 zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 4 : zeros;
        float16x4_t _bias0 = vld1_f16(biasptr);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch; // inch always > 0

            asm volatile(
                "mov    v24.16b, %8.16b             \n"
                "mov    v25.16b, %8.16b             \n"
                "mov    v26.16b, %8.16b             \n"
                "mov    v27.16b, %8.16b             \n"
                "mov    v28.16b, %8.16b             \n"
                "mov    v29.16b, %8.16b             \n"
                "mov    v30.16b, %8.16b             \n"
                "mov    v31.16b, %8.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r01 r23 r45 r67

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // k0123

                "fmla   v24.4h, v4.4h, v0.h[0]      \n"
                "fmla   v25.4h, v4.4h, v0.h[1]      \n"
                "fmla   v26.4h, v4.4h, v0.h[2]      \n"
                "fmla   v27.4h, v4.4h, v0.h[3]      \n"
                "fmla   v28.4h, v4.4h, v0.h[4]      \n"
                "fmla   v29.4h, v4.4h, v0.h[5]      \n"
                "fmla   v30.4h, v4.4h, v0.h[6]      \n"
                "fmla   v31.4h, v4.4h, v0.h[7]      \n"

                "fmla   v24.4h, v5.4h, v1.h[0]      \n"
                "fmla   v25.4h, v5.4h, v1.h[1]      \n"
                "fmla   v26.4h, v5.4h, v1.h[2]      \n"
                "fmla   v27.4h, v5.4h, v1.h[3]      \n"
                "fmla   v28.4h, v5.4h, v1.h[4]      \n"
                "fmla   v29.4h, v5.4h, v1.h[5]      \n"
                "fmla   v30.4h, v5.4h, v1.h[6]      \n"
                "fmla   v31.4h, v5.4h, v1.h[7]      \n"

                "fmla   v24.4h, v6.4h, v2.h[0]      \n"
                "fmla   v25.4h, v6.4h, v2.h[1]      \n"
                "fmla   v26.4h, v6.4h, v2.h[2]      \n"
                "fmla   v27.4h, v6.4h, v2.h[3]      \n"
                "fmla   v28.4h, v6.4h, v2.h[4]      \n"
                "fmla   v29.4h, v6.4h, v2.h[5]      \n"
                "fmla   v30.4h, v6.4h, v2.h[6]      \n"
                "fmla   v31.4h, v6.4h, v2.h[7]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.4h, v7.4h, v3.h[0]      \n"
                "fmla   v25.4h, v7.4h, v3.h[1]      \n"
                "fmla   v26.4h, v7.4h, v3.h[2]      \n"
                "fmla   v27.4h, v7.4h, v3.h[3]      \n"
                "fmla   v28.4h, v7.4h, v3.h[4]      \n"
                "fmla   v29.4h, v7.4h, v3.h[5]      \n"
                "fmla   v30.4h, v7.4h, v3.h[6]      \n"
                "fmla   v31.4h, v7.4h, v3.h[7]      \n"

                "bne    0b                          \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"
                "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "w"(_bias0) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch; // inch always > 0

            asm volatile(
                "mov    v24.16b, %8.16b             \n"
                "mov    v25.16b, %8.16b             \n"
                "mov    v26.16b, %8.16b             \n"
                "mov    v27.16b, %8.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r01 r23 r45 r67

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // k0123

                "fmla   v24.4h, v4.4h, v0.h[0]      \n"
                "fmla   v25.4h, v4.4h, v0.h[1]      \n"
                "fmla   v26.4h, v4.4h, v0.h[2]      \n"
                "fmla   v27.4h, v4.4h, v0.h[3]      \n"

                "fmla   v24.4h, v5.4h, v1.h[0]      \n"
                "fmla   v25.4h, v5.4h, v1.h[1]      \n"
                "fmla   v26.4h, v5.4h, v1.h[2]      \n"
                "fmla   v27.4h, v5.4h, v1.h[3]      \n"

                "fmla   v24.4h, v6.4h, v2.h[0]      \n"
                "fmla   v25.4h, v6.4h, v2.h[1]      \n"
                "fmla   v26.4h, v6.4h, v2.h[2]      \n"
                "fmla   v27.4h, v6.4h, v2.h[3]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.4h, v7.4h, v3.h[0]      \n"
                "fmla   v25.4h, v7.4h, v3.h[1]      \n"
                "fmla   v26.4h, v7.4h, v3.h[2]      \n"
                "fmla   v27.4h, v7.4h, v3.h[3]      \n"

                "bne    0b                          \n"

                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "w"(_bias0) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26", "v27");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            float16x4_t _sum0 = _bias0;

            for (int q = 0; q < inch; q++)
            {
                float16x4_t _r0 = vld1_f16(tmpptr);

                float16x4_t _k0 = vld1_f16(kptr);
                float16x4_t _k1 = vld1_f16(kptr + 4);
                float16x4_t _k2 = vld1_f16(kptr + 8);
                float16x4_t _k3 = vld1_f16(kptr + 12);

                _sum0 = vfma_lane_f16(_sum0, _k0, _r0, 0);
                _sum0 = vfma_lane_f16(_sum0, _k1, _r0, 1);
                _sum0 = vfma_lane_f16(_sum0, _k2, _r0, 2);
                _sum0 = vfma_lane_f16(_sum0, _k3, _r0, 3);

                kptr += 16;
                tmpptr += 4;
            }

            vst1_f16(outptr0, _sum0);

            outptr0 += 4;
        }
    }

    //     // NOTE sgemm
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         const short bias0 = bias ? bias[p] : 0.f;
    //
    //         __fp16* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             short sum = bias0;
    //
    //             const __fp16* kptr = _kernel.channel(p);
    //
    //             for (int q=0; q<inch; q++)
    //             {
    //                 const __fp16* img0 = bottom_blob.channel(q);
    //
    //                 sum += img0[i] * kptr[0];
    //                 kptr ++;
    //             }
    //
    //             outptr0[i] = sum;
    //         }
    //     }
}

static void conv1x1s2_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const __fp16* r0 = bottom_blob.channel(p);
        __fp16* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                float16x4_t _v0 = vld1_f16(r0);
                float16x4_t _v1 = vld1_f16(r0 + 8);
                float16x4_t _v2 = vld1_f16(r0 + 16);
                float16x4_t _v3 = vld1_f16(r0 + 24);
                float16x8_t _v01 = vcombine_f16(_v0, _v1);
                float16x8_t _v23 = vcombine_f16(_v2, _v3);
                vst1q_f16(outptr, _v01);
                vst1q_f16(outptr + 8, _v23);

                r0 += 32;
                outptr += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                float16x4_t _v0 = vld1_f16(r0);
                float16x4_t _v1 = vld1_f16(r0 + 8);
                float16x8_t _v = vcombine_f16(_v0, _v1);
                vst1q_f16(outptr, _v);

                r0 += 16;
                outptr += 8;
            }
            for (; j < outw; j++)
            {
                float16x4_t _v = vld1_f16(r0);
                vst1_f16(outptr, _v);

                r0 += 8;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4_fp16sa_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
