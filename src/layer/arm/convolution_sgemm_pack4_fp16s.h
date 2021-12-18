// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_pack4_fp16sa_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 8u, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 8u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 4, opt.workspace_allocator);
    {
        int nn_size = size / 8;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
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
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
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
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
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
                    img0 += size * 4;
                }
            }
        }
    }

    int remain_outch_start = 0;

    int nn_outch = outch >> 1;
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
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

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
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

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
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum0 = _bias0;

            for (int q = 0; q < nn; q++)
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
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

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
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

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
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float16x4_t _sum0 = _bias0;

            for (int q = 0; q < nn; q++)
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
}

static void convolution_im2col_sgemm_transform_kernel_pack4_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(32 * maxk, inch / 4, outch / 8 + (outch % 8) / 4, (size_t)2u);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);
        const Mat k4 = kernel.channel(q + 4);
        const Mat k5 = kernel.channel(q + 5);
        const Mat k6 = kernel.channel(q + 6);
        const Mat k7 = kernel.channel(q + 7);

        __fp16* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);

            const float* k40 = k4.row(p);
            const float* k41 = k4.row(p + 1);
            const float* k42 = k4.row(p + 2);
            const float* k43 = k4.row(p + 3);

            const float* k50 = k5.row(p);
            const float* k51 = k5.row(p + 1);
            const float* k52 = k5.row(p + 2);
            const float* k53 = k5.row(p + 3);

            const float* k60 = k6.row(p);
            const float* k61 = k6.row(p + 1);
            const float* k62 = k6.row(p + 2);
            const float* k63 = k6.row(p + 3);

            const float* k70 = k7.row(p);
            const float* k71 = k7.row(p + 1);
            const float* k72 = k7.row(p + 2);
            const float* k73 = k7.row(p + 3);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];
                g00[4] = (__fp16)k40[k];
                g00[5] = (__fp16)k50[k];
                g00[6] = (__fp16)k60[k];
                g00[7] = (__fp16)k70[k];

                g00[8] = (__fp16)k01[k];
                g00[9] = (__fp16)k11[k];
                g00[10] = (__fp16)k21[k];
                g00[11] = (__fp16)k31[k];
                g00[12] = (__fp16)k41[k];
                g00[13] = (__fp16)k51[k];
                g00[14] = (__fp16)k61[k];
                g00[15] = (__fp16)k71[k];

                g00[16] = (__fp16)k02[k];
                g00[17] = (__fp16)k12[k];
                g00[18] = (__fp16)k22[k];
                g00[19] = (__fp16)k32[k];
                g00[20] = (__fp16)k42[k];
                g00[21] = (__fp16)k52[k];
                g00[22] = (__fp16)k62[k];
                g00[23] = (__fp16)k72[k];

                g00[24] = (__fp16)k03[k];
                g00[25] = (__fp16)k13[k];
                g00[26] = (__fp16)k23[k];
                g00[27] = (__fp16)k33[k];
                g00[28] = (__fp16)k43[k];
                g00[29] = (__fp16)k53[k];
                g00[30] = (__fp16)k63[k];
                g00[31] = (__fp16)k73[k];

                g00 += 32;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        __fp16* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];

                g00[4] = (__fp16)k01[k];
                g00[5] = (__fp16)k11[k];
                g00[6] = (__fp16)k21[k];
                g00[7] = (__fp16)k31[k];

                g00[8] = (__fp16)k02[k];
                g00[9] = (__fp16)k12[k];
                g00[10] = (__fp16)k22[k];
                g00[11] = (__fp16)k32[k];

                g00[12] = (__fp16)k03[k];
                g00[13] = (__fp16)k13[k];
                g00[14] = (__fp16)k23[k];
                g00[15] = (__fp16)k33[k];

                g00 += 16;
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 8u, 4, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            float16x4_t _val0 = vld1_f16(sptr);
                            float16x4_t _val1 = vld1_f16(sptr + stride_w * 4);
                            float16x4_t _val2 = vld1_f16(sptr + stride_w * 8);
                            float16x4_t _val3 = vld1_f16(sptr + stride_w * 12);
                            vst1_f16(ptr, _val0);
                            vst1_f16(ptr + 4, _val1);
                            vst1_f16(ptr + 8, _val2);
                            vst1_f16(ptr + 12, _val3);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            float16x4_t _val0 = vld1_f16(sptr);
                            float16x4_t _val1 = vld1_f16(sptr + stride_w * 4);
                            vst1_f16(ptr, _val0);
                            vst1_f16(ptr + 4, _val1);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }
                        for (; j < outw; j++)
                        {
                            float16x4_t _val = vld1_f16(sptr);
                            vst1_f16(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
