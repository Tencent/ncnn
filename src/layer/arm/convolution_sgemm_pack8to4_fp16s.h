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

static void im2col_sgemm_pack8to4_fp16sa_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 16u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 16u, 8, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 16u, 8, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 16u, 8, opt.workspace_allocator);
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
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 8;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 8x8
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]   \n"
                        "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                        "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0] \n"
                        "sub    %0, %0, #64             \n"

                        "uzp1   v16.8h, v0.8h, v4.8h    \n"
                        "uzp2   v20.8h, v0.8h, v4.8h    \n"
                        "uzp1   v17.8h, v1.8h, v5.8h    \n"
                        "uzp2   v21.8h, v1.8h, v5.8h    \n"
                        "uzp1   v18.8h, v2.8h, v6.8h    \n"
                        "uzp2   v22.8h, v2.8h, v6.8h    \n"
                        "uzp1   v19.8h, v3.8h, v7.8h    \n"
                        "uzp2   v23.8h, v3.8h, v7.8h    \n"

                        "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
                        "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                    img0 += size * 8;
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
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 8;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 8x4
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]   \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                        "st4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3");
                    img0 += size * 8;
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
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 8;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]   \n"
                        "ld1    {v0.8h}, [%0]           \n"
                        "st1    {v0.8h}, [%1], #16      \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0");
                    img0 += size * 8;
                }
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 1;

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

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"

                "fmla   v24.8h, v16.8h, v0.h[0]     \n"
                "fmla   v25.8h, v16.8h, v0.h[1]     \n"
                "fmla   v26.8h, v16.8h, v0.h[2]     \n"
                "fmla   v27.8h, v16.8h, v0.h[3]     \n"
                "fmla   v28.8h, v16.8h, v0.h[4]     \n"
                "fmla   v29.8h, v16.8h, v0.h[5]     \n"
                "fmla   v30.8h, v16.8h, v0.h[6]     \n"
                "fmla   v31.8h, v16.8h, v0.h[7]     \n"

                "fmla   v24.8h, v17.8h, v1.h[0]     \n"
                "fmla   v25.8h, v17.8h, v1.h[1]     \n"
                "fmla   v26.8h, v17.8h, v1.h[2]     \n"
                "fmla   v27.8h, v17.8h, v1.h[3]     \n"
                "fmla   v28.8h, v17.8h, v1.h[4]     \n"
                "fmla   v29.8h, v17.8h, v1.h[5]     \n"
                "fmla   v30.8h, v17.8h, v1.h[6]     \n"
                "fmla   v31.8h, v17.8h, v1.h[7]     \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                "fmla   v24.8h, v18.8h, v2.h[0]     \n"
                "fmla   v25.8h, v18.8h, v2.h[1]     \n"
                "fmla   v26.8h, v18.8h, v2.h[2]     \n"
                "fmla   v27.8h, v18.8h, v2.h[3]     \n"
                "fmla   v28.8h, v18.8h, v2.h[4]     \n"
                "fmla   v29.8h, v18.8h, v2.h[5]     \n"
                "fmla   v30.8h, v18.8h, v2.h[6]     \n"
                "fmla   v31.8h, v18.8h, v2.h[7]     \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n"

                "fmla   v24.8h, v19.8h, v3.h[0]     \n"
                "fmla   v25.8h, v19.8h, v3.h[1]     \n"
                "fmla   v26.8h, v19.8h, v3.h[2]     \n"
                "fmla   v27.8h, v19.8h, v3.h[3]     \n"
                "fmla   v28.8h, v19.8h, v3.h[4]     \n"
                "fmla   v29.8h, v19.8h, v3.h[5]     \n"
                "fmla   v30.8h, v19.8h, v3.h[6]     \n"
                "fmla   v31.8h, v19.8h, v3.h[7]     \n"

                "fmla   v24.8h, v20.8h, v4.h[0]     \n"
                "fmla   v25.8h, v20.8h, v4.h[1]     \n"
                "fmla   v26.8h, v20.8h, v4.h[2]     \n"
                "fmla   v27.8h, v20.8h, v4.h[3]     \n"
                "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                "fmla   v29.8h, v20.8h, v4.h[5]     \n"
                "fmla   v30.8h, v20.8h, v4.h[6]     \n"
                "fmla   v31.8h, v20.8h, v4.h[7]     \n"

                "fmla   v24.8h, v21.8h, v5.h[0]     \n"
                "fmla   v25.8h, v21.8h, v5.h[1]     \n"
                "fmla   v26.8h, v21.8h, v5.h[2]     \n"
                "fmla   v27.8h, v21.8h, v5.h[3]     \n"
                "fmla   v28.8h, v21.8h, v5.h[4]     \n"
                "fmla   v29.8h, v21.8h, v5.h[5]     \n"
                "fmla   v30.8h, v21.8h, v5.h[6]     \n"
                "fmla   v31.8h, v21.8h, v5.h[7]     \n"

                "fmla   v24.8h, v22.8h, v6.h[0]     \n"
                "fmla   v25.8h, v22.8h, v6.h[1]     \n"
                "fmla   v26.8h, v22.8h, v6.h[2]     \n"
                "fmla   v27.8h, v22.8h, v6.h[3]     \n"
                "fmla   v28.8h, v22.8h, v6.h[4]     \n"
                "fmla   v29.8h, v22.8h, v6.h[5]     \n"
                "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                "fmla   v31.8h, v22.8h, v6.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.8h, v23.8h, v7.h[0]     \n"
                "fmla   v25.8h, v23.8h, v7.h[1]     \n"
                "fmla   v26.8h, v23.8h, v7.h[2]     \n"
                "fmla   v27.8h, v23.8h, v7.h[3]     \n"
                "fmla   v28.8h, v23.8h, v7.h[4]     \n"
                "fmla   v29.8h, v23.8h, v7.h[5]     \n"
                "fmla   v30.8h, v23.8h, v7.h[6]     \n"
                "fmla   v31.8h, v23.8h, v7.h[7]     \n"

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
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "mov    v24.16b, %10.16b            \n"
                "mov    v25.16b, %10.16b            \n"
                "mov    v26.16b, %10.16b            \n"
                "mov    v27.16b, %10.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"

                "fmla   v24.8h, v16.8h, v0.h[0]     \n"
                "fmla   v25.8h, v16.8h, v0.h[1]     \n"
                "fmla   v26.8h, v16.8h, v0.h[2]     \n"
                "fmla   v27.8h, v16.8h, v0.h[3]     \n"
                "fmla   v24.8h, v17.8h, v0.h[4]     \n"
                "fmla   v25.8h, v17.8h, v0.h[5]     \n"
                "fmla   v26.8h, v17.8h, v0.h[6]     \n"
                "fmla   v27.8h, v17.8h, v0.h[7]     \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                "fmla   v24.8h, v18.8h, v1.h[0]     \n"
                "fmla   v25.8h, v18.8h, v1.h[1]     \n"
                "fmla   v26.8h, v18.8h, v1.h[2]     \n"
                "fmla   v27.8h, v18.8h, v1.h[3]     \n"
                "fmla   v24.8h, v19.8h, v1.h[4]     \n"
                "fmla   v25.8h, v19.8h, v1.h[5]     \n"
                "fmla   v26.8h, v19.8h, v1.h[6]     \n"
                "fmla   v27.8h, v19.8h, v1.h[7]     \n"

                "fmla   v24.8h, v20.8h, v2.h[0]     \n"
                "fmla   v25.8h, v20.8h, v2.h[1]     \n"
                "fmla   v26.8h, v20.8h, v2.h[2]     \n"
                "fmla   v27.8h, v20.8h, v2.h[3]     \n"
                "fmla   v24.8h, v21.8h, v2.h[4]     \n"
                "fmla   v25.8h, v21.8h, v2.h[5]     \n"
                "fmla   v26.8h, v21.8h, v2.h[6]     \n"
                "fmla   v27.8h, v21.8h, v2.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.8h, v22.8h, v3.h[0]     \n"
                "fmla   v25.8h, v22.8h, v3.h[1]     \n"
                "fmla   v26.8h, v22.8h, v3.h[2]     \n"
                "fmla   v27.8h, v22.8h, v3.h[3]     \n"
                "fmla   v24.8h, v23.8h, v3.h[4]     \n"
                "fmla   v25.8h, v23.8h, v3.h[5]     \n"
                "fmla   v26.8h, v23.8h, v3.h[6]     \n"
                "fmla   v27.8h, v23.8h, v3.h[7]     \n"

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
                : "cc", "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum0 = _bias0;

            for (int q = 0; q < nn; q++)
            {
                float16x8_t _r0 = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);
                float16x8_t _k1 = vld1q_f16(kptr + 8);
                float16x8_t _k2 = vld1q_f16(kptr + 16);
                float16x8_t _k3 = vld1q_f16(kptr + 24);
                float16x8_t _k4 = vld1q_f16(kptr + 32);
                float16x8_t _k5 = vld1q_f16(kptr + 40);
                float16x8_t _k6 = vld1q_f16(kptr + 48);
                float16x8_t _k7 = vld1q_f16(kptr + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _k0, _r0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _k1, _r0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _k2, _r0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _k3, _r0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _k4, _r0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _k5, _r0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _k6, _r0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _k7, _r0, 7);

                kptr += 64;
                tmpptr += 8;
            }

            vst1_f16(outptr0, vget_low_f16(_sum0));
            vst1_f16(outptr1, vget_high_f16(_sum0));

            outptr0 += 4;
            outptr1 += 4;
        }
    }

    remain_outch_start += nn_outch << 1;

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

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

                "fmla   v24.4h, v16.4h, v0.h[0]     \n"
                "fmla   v25.4h, v16.4h, v0.h[1]     \n"
                "fmla   v26.4h, v16.4h, v0.h[2]     \n"
                "fmla   v27.4h, v16.4h, v0.h[3]     \n"
                "fmla   v28.4h, v16.4h, v0.h[4]     \n"
                "fmla   v29.4h, v16.4h, v0.h[5]     \n"
                "fmla   v30.4h, v16.4h, v0.h[6]     \n"
                "fmla   v31.4h, v16.4h, v0.h[7]     \n"

                "fmla   v24.4h, v17.4h, v1.h[0]     \n"
                "fmla   v25.4h, v17.4h, v1.h[1]     \n"
                "fmla   v26.4h, v17.4h, v1.h[2]     \n"
                "fmla   v27.4h, v17.4h, v1.h[3]     \n"
                "fmla   v28.4h, v17.4h, v1.h[4]     \n"
                "fmla   v29.4h, v17.4h, v1.h[5]     \n"
                "fmla   v30.4h, v17.4h, v1.h[6]     \n"
                "fmla   v31.4h, v17.4h, v1.h[7]     \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%3], #32 \n"

                "fmla   v24.4h, v18.4h, v2.h[0]     \n"
                "fmla   v25.4h, v18.4h, v2.h[1]     \n"
                "fmla   v26.4h, v18.4h, v2.h[2]     \n"
                "fmla   v27.4h, v18.4h, v2.h[3]     \n"
                "fmla   v28.4h, v18.4h, v2.h[4]     \n"
                "fmla   v29.4h, v18.4h, v2.h[5]     \n"
                "fmla   v30.4h, v18.4h, v2.h[6]     \n"
                "fmla   v31.4h, v18.4h, v2.h[7]     \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"

                "fmla   v24.4h, v19.4h, v3.h[0]     \n"
                "fmla   v25.4h, v19.4h, v3.h[1]     \n"
                "fmla   v26.4h, v19.4h, v3.h[2]     \n"
                "fmla   v27.4h, v19.4h, v3.h[3]     \n"
                "fmla   v28.4h, v19.4h, v3.h[4]     \n"
                "fmla   v29.4h, v19.4h, v3.h[5]     \n"
                "fmla   v30.4h, v19.4h, v3.h[6]     \n"
                "fmla   v31.4h, v19.4h, v3.h[7]     \n"

                "fmla   v24.4h, v20.4h, v4.h[0]     \n"
                "fmla   v25.4h, v20.4h, v4.h[1]     \n"
                "fmla   v26.4h, v20.4h, v4.h[2]     \n"
                "fmla   v27.4h, v20.4h, v4.h[3]     \n"
                "fmla   v28.4h, v20.4h, v4.h[4]     \n"
                "fmla   v29.4h, v20.4h, v4.h[5]     \n"
                "fmla   v30.4h, v20.4h, v4.h[6]     \n"
                "fmla   v31.4h, v20.4h, v4.h[7]     \n"

                "fmla   v24.4h, v21.4h, v5.h[0]     \n"
                "fmla   v25.4h, v21.4h, v5.h[1]     \n"
                "fmla   v26.4h, v21.4h, v5.h[2]     \n"
                "fmla   v27.4h, v21.4h, v5.h[3]     \n"
                "fmla   v28.4h, v21.4h, v5.h[4]     \n"
                "fmla   v29.4h, v21.4h, v5.h[5]     \n"
                "fmla   v30.4h, v21.4h, v5.h[6]     \n"
                "fmla   v31.4h, v21.4h, v5.h[7]     \n"

                "fmla   v24.4h, v22.4h, v6.h[0]     \n"
                "fmla   v25.4h, v22.4h, v6.h[1]     \n"
                "fmla   v26.4h, v22.4h, v6.h[2]     \n"
                "fmla   v27.4h, v22.4h, v6.h[3]     \n"
                "fmla   v28.4h, v22.4h, v6.h[4]     \n"
                "fmla   v29.4h, v22.4h, v6.h[5]     \n"
                "fmla   v30.4h, v22.4h, v6.h[6]     \n"
                "fmla   v31.4h, v22.4h, v6.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.4h, v23.4h, v7.h[0]     \n"
                "fmla   v25.4h, v23.4h, v7.h[1]     \n"
                "fmla   v26.4h, v23.4h, v7.h[2]     \n"
                "fmla   v27.4h, v23.4h, v7.h[3]     \n"
                "fmla   v28.4h, v23.4h, v7.h[4]     \n"
                "fmla   v29.4h, v23.4h, v7.h[5]     \n"
                "fmla   v30.4h, v23.4h, v7.h[6]     \n"
                "fmla   v31.4h, v23.4h, v7.h[7]     \n"

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
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "mov    v24.16b, %8.16b             \n"
                "mov    v25.16b, %8.16b             \n"
                "mov    v26.16b, %8.16b             \n"
                "mov    v27.16b, %8.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n"

                "fmla   v24.4h, v16.4h, v0.h[0]     \n"
                "fmla   v25.4h, v16.4h, v0.h[1]     \n"
                "fmla   v26.4h, v16.4h, v0.h[2]     \n"
                "fmla   v27.4h, v16.4h, v0.h[3]     \n"
                "fmla   v24.4h, v17.4h, v0.h[4]     \n"
                "fmla   v25.4h, v17.4h, v0.h[5]     \n"
                "fmla   v26.4h, v17.4h, v0.h[6]     \n"
                "fmla   v27.4h, v17.4h, v0.h[7]     \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%3], #32 \n"

                "fmla   v24.4h, v18.4h, v1.h[0]     \n"
                "fmla   v25.4h, v18.4h, v1.h[1]     \n"
                "fmla   v26.4h, v18.4h, v1.h[2]     \n"
                "fmla   v27.4h, v18.4h, v1.h[3]     \n"
                "fmla   v24.4h, v19.4h, v1.h[4]     \n"
                "fmla   v25.4h, v19.4h, v1.h[5]     \n"
                "fmla   v26.4h, v19.4h, v1.h[6]     \n"
                "fmla   v27.4h, v19.4h, v1.h[7]     \n"

                "fmla   v24.4h, v20.4h, v2.h[0]     \n"
                "fmla   v25.4h, v20.4h, v2.h[1]     \n"
                "fmla   v26.4h, v20.4h, v2.h[2]     \n"
                "fmla   v27.4h, v20.4h, v2.h[3]     \n"
                "fmla   v24.4h, v21.4h, v2.h[4]     \n"
                "fmla   v25.4h, v21.4h, v2.h[5]     \n"
                "fmla   v26.4h, v21.4h, v2.h[6]     \n"
                "fmla   v27.4h, v21.4h, v2.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.4h, v22.4h, v3.h[0]     \n"
                "fmla   v25.4h, v22.4h, v3.h[1]     \n"
                "fmla   v26.4h, v22.4h, v3.h[2]     \n"
                "fmla   v27.4h, v22.4h, v3.h[3]     \n"
                "fmla   v24.4h, v23.4h, v3.h[4]     \n"
                "fmla   v25.4h, v23.4h, v3.h[5]     \n"
                "fmla   v26.4h, v23.4h, v3.h[6]     \n"
                "fmla   v27.4h, v23.4h, v3.h[7]     \n"

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
                : "cc", "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float16x4_t _sum0 = _bias0;

            for (int q = 0; q < nn; q++)
            {
                float16x8_t _r0 = vld1q_f16(tmpptr);

                float16x4_t _k0 = vld1_f16(kptr);
                float16x4_t _k1 = vld1_f16(kptr + 4);
                float16x4_t _k2 = vld1_f16(kptr + 8);
                float16x4_t _k3 = vld1_f16(kptr + 12);
                float16x4_t _k4 = vld1_f16(kptr + 16);
                float16x4_t _k5 = vld1_f16(kptr + 20);
                float16x4_t _k6 = vld1_f16(kptr + 24);
                float16x4_t _k7 = vld1_f16(kptr + 28);

                _sum0 = vfma_laneq_f16(_sum0, _k0, _r0, 0);
                _sum0 = vfma_laneq_f16(_sum0, _k1, _r0, 1);
                _sum0 = vfma_laneq_f16(_sum0, _k2, _r0, 2);
                _sum0 = vfma_laneq_f16(_sum0, _k3, _r0, 3);
                _sum0 = vfma_laneq_f16(_sum0, _k4, _r0, 4);
                _sum0 = vfma_laneq_f16(_sum0, _k5, _r0, 5);
                _sum0 = vfma_laneq_f16(_sum0, _k6, _r0, 6);
                _sum0 = vfma_laneq_f16(_sum0, _k7, _r0, 7);

                kptr += 32;
                tmpptr += 8;
            }

            vst1_f16(outptr0, _sum0);

            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack8to4_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-8a-maxk-inch/8a-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(64 * maxk, inch / 8, outch / 8 + (outch % 8) / 4, (size_t)2u);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        __fp16* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = (__fp16)k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        __fp16* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = (__fp16)k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack8to4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 16u, 8, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 8;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            float16x8_t _val0 = vld1q_f16(sptr);
                            float16x8_t _val1 = vld1q_f16(sptr + stride_w * 8);
                            float16x8_t _val2 = vld1q_f16(sptr + stride_w * 16);
                            float16x8_t _val3 = vld1q_f16(sptr + stride_w * 24);
                            vst1q_f16(ptr, _val0);
                            vst1q_f16(ptr + 8, _val1);
                            vst1q_f16(ptr + 16, _val2);
                            vst1q_f16(ptr + 24, _val3);

                            sptr += stride_w * 32;
                            ptr += 32;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            float16x8_t _val0 = vld1q_f16(sptr);
                            float16x8_t _val1 = vld1q_f16(sptr + stride_w * 8);
                            vst1q_f16(ptr, _val0);
                            vst1q_f16(ptr + 8, _val1);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }
                        for (; j < outw; j++)
                        {
                            float16x8_t _val = vld1q_f16(sptr);
                            vst1q_f16(ptr, _val);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack8to4_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
