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

static void im2col_sgemm_pack4_bf16s_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
#if __aarch64__
    if (size >= 12)
        tmp.create(12 * maxk, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, 8u, 4, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 4, opt.workspace_allocator);
#else
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 8u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 4, opt.workspace_allocator);
#endif
    {
#if __aarch64__
        int nn_size = size / 12;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 12;

            unsigned short* tmpptr = tmp.channel(i / 12);

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                        "ld4    {v4.4h, v5.4h, v6.4h, v7.4h}, [%0]      \n"
                        "st1    {v0.8h}, [%1], #16          \n"
                        "st1    {v4.4h}, [%1], #8           \n"
                        "st1    {v1.8h}, [%1], #16          \n"
                        "st1    {v5.4h}, [%1], #8           \n"
                        "sub    %0, %0, #64                 \n"
                        "st1    {v2.8h}, [%1], #16          \n"
                        "st1    {v6.4h}, [%1], #8           \n"
                        "st1    {v3.8h}, [%1], #16          \n"
                        "st1    {v7.4h}, [%1], #8           \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;
#else
        int nn_size = size >> 3;
        int remain_size_start = 0;
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

#if __aarch64__
            unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
#else
            unsigned short* tmpptr = tmp.channel(i / 8);
#endif

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                        "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld4.u16   {d0-d3}, [%0]!      \n"
                        "pld        [%0, #256]          \n"
                        "vld4.u16   {d4-d7}, [%0]       \n"
                        "sub        %0, %0, #32         \n"
                        "vst1.u16   {d0}, [%1 :64]!     \n"
                        "vst1.u16   {d4}, [%1 :64]!     \n"
                        "vst1.u16   {d1}, [%1 :64]!     \n"
                        "vst1.u16   {d5}, [%1 :64]!     \n"
                        "vst1.u16   {d2}, [%1 :64]!     \n"
                        "vst1.u16   {d6}, [%1 :64]!     \n"
                        "vst1.u16   {d3}, [%1 :64]!     \n"
                        "vst1.u16   {d7}, [%1 :64]!     \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
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

#if __aarch64__
            unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%0]        \n"
                        "st1    {v0.8h, v1.8h}, [%1], #32   \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1");
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.u16   {d0-d3}, [%0 :128]  \n"
                        "vst1.u16   {d0-d3}, [%1 :128]! \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1");
#endif // __aarch64__
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

#if __aarch64__
            unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
#endif

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]   \n"
                        "ld1    {v0.8h}, [%0]           \n"
                        "st1    {v0.8h}, [%1], #16      \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0");
#else
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.u16   {d0-d1}, [%0 :128]  \n"
                        "vst1.u16   {d0-d1}, [%1 :128]! \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0");
#endif // __aarch64__
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __aarch64__
            unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
#endif

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #64]    \n"
                        "ld1    {v0.4h}, [%0]           \n"
                        "st1    {v0.4h}, [%1], #8       \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0");
#else
                    asm volatile(
                        "pld        [%0, #64]           \n"
                        "vld1.u16   {d0}, [%0 :64]      \n"
                        "vst1.u16   {d0}, [%1 :64]!     \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0");
#endif // __aarch64__
                    img0 += size * 4;
                }
            }
        }
    }

    int remain_outch_start = 0;

#if __aarch64__
    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        unsigned short* outptr0 = top_blob.channel(p);
        unsigned short* outptr1 = top_blob.channel(p + 1);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%10]       \n"
                "mov    v8.16b, v0.16b              \n"
                "mov    v9.16b, v0.16b              \n"
                "mov    v10.16b, v0.16b             \n"
                "mov    v11.16b, v0.16b             \n"
                "mov    v12.16b, v0.16b             \n"
                "mov    v13.16b, v0.16b             \n"
                "mov    v14.16b, v0.16b             \n"
                "mov    v15.16b, v0.16b             \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"
                "mov    v20.16b, v1.16b             \n"
                "mov    v21.16b, v1.16b             \n"
                "mov    v22.16b, v1.16b             \n"
                "mov    v23.16b, v1.16b             \n"
                "mov    v24.16b, v1.16b             \n"
                "mov    v25.16b, v1.16b             \n"
                "mov    v26.16b, v1.16b             \n"
                "mov    v27.16b, v1.16b             \n"
                "mov    v28.16b, v1.16b             \n"
                "mov    v29.16b, v1.16b             \n"
                "mov    v30.16b, v1.16b             \n"
                "mov    v31.16b, v1.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%4], #32 \n" // w0011_01

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v4.4s, v4.4h, #16           \n"
                "shll   v5.4s, v5.4h, #16           \n"
                "shll   v6.4s, v6.4h, #16           \n"
                "shll   v7.4s, v7.4h, #16           \n"

                "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                "fmla   v11.4s, v6.4s, v3.s[3]      \n"

                "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n"

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                "fmla   v19.4s, v6.4s, v1.s[3]      \n"

                "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%4], #32 \n" // w2233_01

                "shll   v4.4s, v4.4h, #16           \n"
                "shll   v5.4s, v5.4h, #16           \n"
                "shll   v6.4s, v6.4h, #16           \n"
                "shll   v7.4s, v7.4h, #16           \n"

                "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n"

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                "fmla   v19.4s, v4.4s, v0.s[3]      \n"

                "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                "fmla   v19.4s, v6.4s, v3.s[3]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                "bne    0b                          \n"

                "shrn   v8.4h, v8.4s, #16           \n"
                "shrn   v9.4h, v9.4s, #16           \n"
                "shrn   v10.4h, v10.4s, #16         \n"
                "shrn   v11.4h, v11.4s, #16         \n"

                "shrn   v12.4h, v12.4s, #16         \n"
                "shrn   v13.4h, v13.4s, #16         \n"
                "shrn   v14.4h, v14.4s, #16         \n"
                "shrn   v15.4h, v15.4s, #16         \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "shrn   v20.4h, v20.4s, #16         \n"
                "shrn   v21.4h, v21.4s, #16         \n"
                "shrn   v22.4h, v22.4s, #16         \n"
                "shrn   v23.4h, v23.4s, #16         \n"

                "shrn   v24.4h, v24.4s, #16         \n"
                "shrn   v25.4h, v25.4s, #16         \n"
                "shrn   v26.4h, v26.4s, #16         \n"
                "shrn   v27.4h, v27.4s, #16         \n"

                "shrn   v28.4h, v28.4s, #16         \n"
                "shrn   v29.4h, v29.4s, #16         \n"
                "shrn   v30.4h, v30.4s, #16         \n"
                "shrn   v31.4h, v31.4s, #16         \n"

                "st1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%1], #32 \n"
                "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"
                "st1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%1], #32 \n"
                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"
                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr0)    // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr0),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%10]       \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"
                "mov    v20.16b, v0.16b             \n"
                "mov    v21.16b, v0.16b             \n"
                "mov    v22.16b, v0.16b             \n"
                "mov    v23.16b, v0.16b             \n"
                "mov    v24.16b, v1.16b             \n"
                "mov    v25.16b, v1.16b             \n"
                "mov    v26.16b, v1.16b             \n"
                "mov    v27.16b, v1.16b             \n"
                "mov    v28.16b, v1.16b             \n"
                "mov    v29.16b, v1.16b             \n"
                "mov    v30.16b, v1.16b             \n"
                "mov    v31.16b, v1.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%4], #32 \n" // w0011_01

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // r4 r5 r6 r7

                "shll   v4.4s, v4.4h, #16           \n"
                "shll   v5.4s, v5.4h, #16           \n"
                "shll   v6.4s, v6.4h, #16           \n"
                "shll   v7.4s, v7.4h, #16           \n"

                "fmla   v20.4s, v8.4s, v4.s[0]      \n"
                "fmla   v21.4s, v8.4s, v5.s[0]      \n"
                "fmla   v22.4s, v8.4s, v6.s[0]      \n"
                "fmla   v23.4s, v8.4s, v7.s[0]      \n"

                "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                "fmla   v25.4s, v9.4s, v1.s[0]      \n"
                "fmla   v26.4s, v9.4s, v2.s[0]      \n"
                "fmla   v27.4s, v9.4s, v3.s[0]      \n"
                "fmla   v28.4s, v9.4s, v4.s[0]      \n"
                "fmla   v29.4s, v9.4s, v5.s[0]      \n"
                "fmla   v30.4s, v9.4s, v6.s[0]      \n"
                "fmla   v31.4s, v9.4s, v7.s[0]      \n"

                "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                "fmla   v19.4s, v10.4s, v3.s[1]     \n"
                "fmla   v20.4s, v10.4s, v4.s[1]     \n"
                "fmla   v21.4s, v10.4s, v5.s[1]     \n"
                "fmla   v22.4s, v10.4s, v6.s[1]     \n"
                "fmla   v23.4s, v10.4s, v7.s[1]     \n"

                "fmla   v24.4s, v11.4s, v0.s[1]     \n"
                "fmla   v25.4s, v11.4s, v1.s[1]     \n"
                "fmla   v26.4s, v11.4s, v2.s[1]     \n"
                "fmla   v27.4s, v11.4s, v3.s[1]     \n"
                "fmla   v28.4s, v11.4s, v4.s[1]     \n"
                "fmla   v29.4s, v11.4s, v5.s[1]     \n"
                "fmla   v30.4s, v11.4s, v6.s[1]     \n"
                "fmla   v31.4s, v11.4s, v7.s[1]     \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%4], #32 \n" // w2233_01

                "shll   v12.4s, v12.4h, #16         \n"
                "shll   v13.4s, v13.4h, #16         \n"
                "shll   v14.4s, v14.4h, #16         \n"
                "shll   v15.4s, v15.4h, #16         \n"

                "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                "fmla   v18.4s, v12.4s, v2.s[2]     \n"
                "fmla   v19.4s, v12.4s, v3.s[2]     \n"
                "fmla   v20.4s, v12.4s, v4.s[2]     \n"
                "fmla   v21.4s, v12.4s, v5.s[2]     \n"
                "fmla   v22.4s, v12.4s, v6.s[2]     \n"
                "fmla   v23.4s, v12.4s, v7.s[2]     \n"

                "fmla   v24.4s, v13.4s, v0.s[2]     \n"
                "fmla   v25.4s, v13.4s, v1.s[2]     \n"
                "fmla   v26.4s, v13.4s, v2.s[2]     \n"
                "fmla   v27.4s, v13.4s, v3.s[2]     \n"
                "fmla   v28.4s, v13.4s, v4.s[2]     \n"
                "fmla   v29.4s, v13.4s, v5.s[2]     \n"
                "fmla   v30.4s, v13.4s, v6.s[2]     \n"
                "fmla   v31.4s, v13.4s, v7.s[2]     \n"

                "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                "fmla   v18.4s, v14.4s, v2.s[3]     \n"
                "fmla   v19.4s, v14.4s, v3.s[3]     \n"
                "fmla   v20.4s, v14.4s, v4.s[3]     \n"
                "fmla   v21.4s, v14.4s, v5.s[3]     \n"
                "fmla   v22.4s, v14.4s, v6.s[3]     \n"
                "fmla   v23.4s, v14.4s, v7.s[3]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v24.4s, v15.4s, v0.s[3]     \n"
                "fmla   v25.4s, v15.4s, v1.s[3]     \n"
                "fmla   v26.4s, v15.4s, v2.s[3]     \n"
                "fmla   v27.4s, v15.4s, v3.s[3]     \n"
                "fmla   v28.4s, v15.4s, v4.s[3]     \n"
                "fmla   v29.4s, v15.4s, v5.s[3]     \n"
                "fmla   v30.4s, v15.4s, v6.s[3]     \n"
                "fmla   v31.4s, v15.4s, v7.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "shrn   v20.4h, v20.4s, #16         \n"
                "shrn   v21.4h, v21.4s, #16         \n"
                "shrn   v22.4h, v22.4s, #16         \n"
                "shrn   v23.4h, v23.4s, #16         \n"

                "shrn   v24.4h, v24.4s, #16         \n"
                "shrn   v25.4h, v25.4s, #16         \n"
                "shrn   v26.4h, v26.4s, #16         \n"
                "shrn   v27.4h, v27.4s, #16         \n"

                "shrn   v28.4h, v28.4s, #16         \n"
                "shrn   v29.4h, v29.4s, #16         \n"
                "shrn   v30.4h, v30.4s, #16         \n"
                "shrn   v31.4h, v31.4s, #16         \n"

                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "st1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"
                "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%1], #32 \n"
                "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr0)    // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr0),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%10]       \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"
                "mov    v20.16b, v1.16b             \n"
                "mov    v21.16b, v1.16b             \n"
                "mov    v22.16b, v1.16b             \n"
                "mov    v23.16b, v1.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%4], #32 \n" // w0011_01

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "fmla   v20.4s, v9.4s, v0.s[0]      \n"
                "fmla   v21.4s, v9.4s, v1.s[0]      \n"
                "fmla   v22.4s, v9.4s, v2.s[0]      \n"
                "fmla   v23.4s, v9.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%4], #32 \n" // w2233_01

                "shll   v12.4s, v12.4h, #16         \n"
                "shll   v13.4s, v13.4h, #16         \n"
                "shll   v14.4s, v14.4h, #16         \n"
                "shll   v15.4s, v15.4h, #16         \n"

                "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                "fmla   v19.4s, v10.4s, v3.s[1]     \n"

                "fmla   v20.4s, v11.4s, v0.s[1]     \n"
                "fmla   v21.4s, v11.4s, v1.s[1]     \n"
                "fmla   v22.4s, v11.4s, v2.s[1]     \n"
                "fmla   v23.4s, v11.4s, v3.s[1]     \n"

                "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                "fmla   v18.4s, v12.4s, v2.s[2]     \n"
                "fmla   v19.4s, v12.4s, v3.s[2]     \n"

                "fmla   v20.4s, v13.4s, v0.s[2]     \n"
                "fmla   v21.4s, v13.4s, v1.s[2]     \n"
                "fmla   v22.4s, v13.4s, v2.s[2]     \n"
                "fmla   v23.4s, v13.4s, v3.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                "fmla   v18.4s, v14.4s, v2.s[3]     \n"
                "fmla   v19.4s, v14.4s, v3.s[3]     \n"

                "fmla   v20.4s, v15.4s, v0.s[3]     \n"
                "fmla   v21.4s, v15.4s, v1.s[3]     \n"
                "fmla   v22.4s, v15.4s, v2.s[3]     \n"
                "fmla   v23.4s, v15.4s, v3.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "shrn   v20.4h, v20.4s, #16         \n"
                "shrn   v21.4h, v21.4s, #16         \n"
                "shrn   v22.4h, v22.4s, #16         \n"
                "shrn   v23.4h, v23.4s, #16         \n"

                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr0)    // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr0),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i + 1 < size; i += 2)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%10]       \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v1.16b             \n"
                "mov    v19.16b, v1.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%3], #16   \n" // r0 r1

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%4], #32 \n" // w0011_01

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v9.4s, v0.s[0]      \n"
                "fmla   v19.4s, v9.4s, v1.s[0]      \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%4], #32 \n" // w2233_01

                "shll   v12.4s, v12.4h, #16         \n"
                "shll   v13.4s, v13.4h, #16         \n"
                "shll   v14.4s, v14.4h, #16         \n"
                "shll   v15.4s, v15.4h, #16         \n"

                "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                "fmla   v18.4s, v11.4s, v0.s[1]     \n"
                "fmla   v19.4s, v11.4s, v1.s[1]     \n"

                "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                "fmla   v18.4s, v13.4s, v0.s[2]     \n"
                "fmla   v19.4s, v13.4s, v1.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                "fmla   v18.4s, v15.4s, v0.s[3]     \n"
                "fmla   v19.4s, v15.4s, v1.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "st1    {v16.4h, v17.4h}, [%1], #16 \n"
                "st1    {v18.4h, v19.4h}, [%2], #16 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr0)    // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr0),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
        }
        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v16.4s, v17.4s}, [%10]     \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #64]        \n"
                "ld1    {v0.4h}, [%3], #8           \n" // r0

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%4], #32 \n" // w0011_01

                "shll   v0.4s, v0.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v9.4s, v0.s[0]      \n"

                "prfm   pldl1keep, [%4, #256]       \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%4], #32 \n" // w2233_01

                "shll   v12.4s, v12.4h, #16         \n"
                "shll   v13.4s, v13.4h, #16         \n"
                "shll   v14.4s, v14.4h, #16         \n"
                "shll   v15.4s, v15.4h, #16         \n"

                "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                "fmla   v17.4s, v11.4s, v0.s[1]     \n"

                "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                "fmla   v17.4s, v13.4s, v0.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                "fmla   v17.4s, v15.4s, v0.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"

                "st1    {v16.4h}, [%1], #8          \n"
                "st1    {v17.4h}, [%2], #8          \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr0)    // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr0),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
        }
    }
#endif // __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        unsigned short* outptr0 = top_blob.channel(p);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
#if __aarch64__
        for (; i + 11 < size; i += 12)
        {
            const unsigned short* tmpptr = tmp.channel(i / 12);
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v8.16b, v0.16b              \n"
                "mov    v9.16b, v0.16b              \n"
                "mov    v10.16b, v0.16b             \n"
                "mov    v11.16b, v0.16b             \n"
                "mov    v12.16b, v0.16b             \n"
                "mov    v13.16b, v0.16b             \n"
                "mov    v14.16b, v0.16b             \n"
                "mov    v15.16b, v0.16b             \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3], #32 \n" // w0123_0

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v4.4s, v4.4h, #16           \n"
                "shll   v5.4s, v5.4h, #16           \n"
                "shll   v6.4s, v6.4h, #16           \n"
                "shll   v7.4s, v7.4h, #16           \n"

                "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"

                "shll   v20.4s, v20.4h, #16         \n"
                "shll   v21.4s, v21.4h, #16         \n"
                "shll   v22.4s, v22.4h, #16         \n"
                "shll   v23.4s, v23.4h, #16         \n"

                "fmla   v8.4s, v5.4s, v3.s[0]       \n"
                "fmla   v9.4s, v5.4s, v3.s[1]       \n"
                "fmla   v10.4s, v5.4s, v3.s[2]      \n"
                "fmla   v11.4s, v5.4s, v3.s[3]      \n"
                "fmla   v12.4s, v5.4s, v20.s[0]     \n"
                "fmla   v13.4s, v5.4s, v20.s[1]     \n"
                "fmla   v14.4s, v5.4s, v20.s[2]     \n"
                "fmla   v15.4s, v5.4s, v20.s[3]     \n"
                "fmla   v16.4s, v5.4s, v21.s[0]     \n"
                "fmla   v17.4s, v5.4s, v21.s[1]     \n"
                "fmla   v18.4s, v5.4s, v21.s[2]     \n"
                "fmla   v19.4s, v5.4s, v21.s[3]     \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%2], #32 \n"

                "shll   v24.4s, v24.4h, #16         \n"
                "shll   v25.4s, v25.4h, #16         \n"
                "shll   v26.4s, v26.4h, #16         \n"
                "shll   v27.4s, v27.4h, #16         \n"

                "fmla   v8.4s, v6.4s, v22.s[0]      \n"
                "fmla   v9.4s, v6.4s, v22.s[1]      \n"
                "fmla   v10.4s, v6.4s, v22.s[2]     \n"
                "fmla   v11.4s, v6.4s, v22.s[3]     \n"
                "fmla   v12.4s, v6.4s, v23.s[0]     \n"
                "fmla   v13.4s, v6.4s, v23.s[1]     \n"
                "fmla   v14.4s, v6.4s, v23.s[2]     \n"
                "fmla   v15.4s, v6.4s, v23.s[3]     \n"
                "fmla   v16.4s, v6.4s, v24.s[0]     \n"
                "fmla   v17.4s, v6.4s, v24.s[1]     \n"
                "fmla   v18.4s, v6.4s, v24.s[2]     \n"
                "fmla   v19.4s, v6.4s, v24.s[3]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v8.4s, v7.4s, v25.s[0]      \n"
                "fmla   v9.4s, v7.4s, v25.s[1]      \n"
                "fmla   v10.4s, v7.4s, v25.s[2]     \n"
                "fmla   v11.4s, v7.4s, v25.s[3]     \n"
                "fmla   v12.4s, v7.4s, v26.s[0]     \n"
                "fmla   v13.4s, v7.4s, v26.s[1]     \n"
                "fmla   v14.4s, v7.4s, v26.s[2]     \n"
                "fmla   v15.4s, v7.4s, v26.s[3]     \n"
                "fmla   v16.4s, v7.4s, v27.s[0]     \n"
                "fmla   v17.4s, v7.4s, v27.s[1]     \n"
                "fmla   v18.4s, v7.4s, v27.s[2]     \n"
                "fmla   v19.4s, v7.4s, v27.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v8.4h, v8.4s, #16           \n"
                "shrn   v9.4h, v9.4s, #16           \n"
                "shrn   v10.4h, v10.4s, #16         \n"
                "shrn   v11.4h, v11.4s, #16         \n"

                "shrn   v12.4h, v12.4s, #16         \n"
                "shrn   v13.4h, v13.4s, #16         \n"
                "shrn   v14.4h, v14.4s, #16         \n"
                "shrn   v15.4h, v15.4s, #16         \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "st1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%1], #32 \n"
                "st1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%1], #32 \n"
                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
#if __aarch64__
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* tmpptr = tmp.channel(i / 8);
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"
                "mov    v20.16b, v0.16b             \n"
                "mov    v21.16b, v0.16b             \n"
                "mov    v22.16b, v0.16b             \n"
                "mov    v23.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%3], #32 \n" // w0123

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%2], #32 \n" // r4 r5 r6 r7

                "shll   v4.4s, v4.4h, #16           \n"
                "shll   v5.4s, v5.4h, #16           \n"
                "shll   v6.4s, v6.4h, #16           \n"
                "shll   v7.4s, v7.4h, #16           \n"

                "fmla   v20.4s, v8.4s, v4.s[0]      \n"
                "fmla   v21.4s, v8.4s, v5.s[0]      \n"
                "fmla   v22.4s, v8.4s, v6.s[0]      \n"
                "fmla   v23.4s, v8.4s, v7.s[0]      \n"

                "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                "fmla   v17.4s, v9.4s, v1.s[1]      \n"
                "fmla   v18.4s, v9.4s, v2.s[1]      \n"
                "fmla   v19.4s, v9.4s, v3.s[1]      \n"
                "fmla   v20.4s, v9.4s, v4.s[1]      \n"
                "fmla   v21.4s, v9.4s, v5.s[1]      \n"
                "fmla   v22.4s, v9.4s, v6.s[1]      \n"
                "fmla   v23.4s, v9.4s, v7.s[1]      \n"

                "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                "fmla   v17.4s, v10.4s, v1.s[2]     \n"
                "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                "fmla   v19.4s, v10.4s, v3.s[2]     \n"
                "fmla   v20.4s, v10.4s, v4.s[2]     \n"
                "fmla   v21.4s, v10.4s, v5.s[2]     \n"
                "fmla   v22.4s, v10.4s, v6.s[2]     \n"
                "fmla   v23.4s, v10.4s, v7.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                "fmla   v17.4s, v11.4s, v1.s[3]     \n"
                "fmla   v18.4s, v11.4s, v2.s[3]     \n"
                "fmla   v19.4s, v11.4s, v3.s[3]     \n"
                "fmla   v20.4s, v11.4s, v4.s[3]     \n"
                "fmla   v21.4s, v11.4s, v5.s[3]     \n"
                "fmla   v22.4s, v11.4s, v6.s[3]     \n"
                "fmla   v23.4s, v11.4s, v7.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "shrn   v20.4h, v20.4s, #16         \n"
                "shrn   v21.4h, v21.4s, #16         \n"
                "shrn   v22.4h, v22.4s, #16         \n"
                "shrn   v23.4h, v23.4s, #16         \n"

                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%8]   \n"
                "vmov       q8, q0          \n"
                "vmov       q9, q0          \n"
                "vmov       q10, q0         \n"
                "vmov       q11, q0         \n"
                "vmov       q12, q0         \n"
                "vmov       q13, q0         \n"
                "vmov       q14, q0         \n"
                "vmov       q15, q0         \n"

                "0:                         \n"

                "pld        [%2, #256]      \n"
                "vld1.u16   {d4-d7}, [%2]!  \n"

                "pld        [%3, #256]      \n"
                "vld1.u16   {d12-d15}, [%3]! \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q9, q4, d0[1]   \n"
                "vmla.f32   q10, q4, d1[0]  \n"
                "vmla.f32   q11, q4, d1[1]  \n"
                "vmla.f32   q12, q4, d2[0]  \n"
                "vmla.f32   q13, q4, d2[1]  \n"
                "vmla.f32   q14, q4, d3[0]  \n"
                "vmla.f32   q15, q4, d3[1]  \n"

                "vmla.f32   q8, q5, d4[0]   \n"
                "vmla.f32   q9, q5, d4[1]   \n"
                "vmla.f32   q10, q5, d5[0]  \n"
                "vmla.f32   q11, q5, d5[1]  \n"
                "vmla.f32   q12, q5, d6[0]  \n"
                "vmla.f32   q13, q5, d6[1]  \n"
                "vmla.f32   q14, q5, d7[0]  \n"
                "vmla.f32   q15, q5, d7[1]  \n"

                "pld        [%2, #256]      \n"
                "vld1.u16   {d4-d7}, [%2]!  \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"

                "vmla.f32   q8, q6, d0[0]   \n"
                "vmla.f32   q9, q6, d0[1]   \n"
                "vmla.f32   q10, q6, d1[0]  \n"
                "vmla.f32   q11, q6, d1[1]  \n"
                "vmla.f32   q12, q6, d2[0]  \n"
                "vmla.f32   q13, q6, d2[1]  \n"
                "vmla.f32   q14, q6, d3[0]  \n"
                "vmla.f32   q15, q6, d3[1]  \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q7, d4[0]   \n"
                "vmla.f32   q9, q7, d4[1]   \n"
                "vmla.f32   q10, q7, d5[0]  \n"
                "vmla.f32   q11, q7, d5[1]  \n"
                "vmla.f32   q12, q7, d6[0]  \n"
                "vmla.f32   q13, q7, d6[1]  \n"
                "vmla.f32   q14, q7, d7[0]  \n"
                "vmla.f32   q15, q7, d7[1]  \n"

                "bne        0b              \n"

                "vshrn.u32  d16, q8, #16    \n"
                "vshrn.u32  d17, q9, #16    \n"
                "vshrn.u32  d18, q10, #16   \n"
                "vshrn.u32  d19, q11, #16   \n"

                "vshrn.u32  d20, q12, #16   \n"
                "vshrn.u32  d21, q13, #16   \n"
                "vshrn.u32  d22, q14, #16   \n"
                "vshrn.u32  d23, q15, #16   \n"

                "vstm       %1!, {d16-d23}  \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif
        }
        for (; i + 3 < size; i += 4)
        {
#if __aarch64__
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%3], #32 \n" // w0123

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                "fmla   v17.4s, v9.4s, v1.s[1]      \n"
                "fmla   v18.4s, v9.4s, v2.s[1]      \n"
                "fmla   v19.4s, v9.4s, v3.s[1]      \n"

                "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                "fmla   v17.4s, v10.4s, v1.s[2]     \n"
                "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                "fmla   v19.4s, v10.4s, v3.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                "fmla   v17.4s, v11.4s, v1.s[3]     \n"
                "fmla   v18.4s, v11.4s, v2.s[3]     \n"
                "fmla   v19.4s, v11.4s, v3.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"

                "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%8]   \n"
                "vmov       q8, q0          \n"
                "vmov       q9, q0          \n"
                "vmov       q10, q0         \n"
                "vmov       q11, q0         \n"

                "0:                         \n"

                "pld        [%2, #256]      \n"
                "vld1.u16   {d4-d7}, [%2]!  \n"

                "pld        [%3, #256]      \n"
                "vld1.u16   {d12-d15}, [%3]! \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q9, q4, d2[0]   \n"
                "vmla.f32   q10, q4, d4[0]  \n"
                "vmla.f32   q11, q4, d6[0]  \n"

                "vmla.f32   q8, q5, d0[1]   \n"
                "vmla.f32   q9, q5, d2[1]   \n"
                "vmla.f32   q10, q5, d4[1]  \n"
                "vmla.f32   q11, q5, d6[1]  \n"

                "vmla.f32   q8, q6, d1[0]   \n"
                "vmla.f32   q9, q6, d3[0]   \n"
                "vmla.f32   q10, q6, d5[0]  \n"
                "vmla.f32   q11, q6, d7[0]  \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q7, d1[1]   \n"
                "vmla.f32   q9, q7, d3[1]   \n"
                "vmla.f32   q10, q7, d5[1]  \n"
                "vmla.f32   q11, q7, d7[1]  \n"

                "bne        0b              \n"

                "vshrn.u32  d16, q8, #16    \n"
                "vshrn.u32  d17, q9, #16    \n"
                "vshrn.u32  d18, q10, #16   \n"
                "vshrn.u32  d19, q11, #16   \n"

                "vst1.u16   {d16-d19}, [%1]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
        }
        for (; i + 1 < size; i += 2)
        {
#if __aarch64__
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n" // r0 r1

                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%3], #32 \n" // w0123

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"

                "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                "fmla   v17.4s, v9.4s, v1.s[1]      \n"

                "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                "fmla   v17.4s, v10.4s, v1.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                "fmla   v17.4s, v11.4s, v1.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"

                "st1    {v16.4h, v17.4h}, [%1], #16 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17");
#else
            asm volatile(
                "vld1.f32   {d0-d1}, [%8]   \n"
                "vmov       q8, q0          \n"
                "vmov       q9, q0          \n"

                "0:                         \n"

                "pld        [%2, #128]      \n"
                "vld1.u16   {d4-d5}, [%2 :128]! \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"

                "pld        [%3, #256]      \n"
                "vld1.u16   {d12-d15}, [%3]! \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q9, q4, d2[0]   \n"

                "vmla.f32   q8, q5, d0[1]   \n"
                "vmla.f32   q9, q5, d2[1]   \n"

                "vmla.f32   q8, q6, d1[0]   \n"
                "vmla.f32   q9, q6, d3[0]   \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q7, d1[1]   \n"
                "vmla.f32   q9, q7, d3[1]   \n"

                "bne        0b              \n"

                "vshrn.u32  d16, q8, #16    \n"
                "vshrn.u32  d17, q9, #16    \n"

                "vst1.u16   {d16-d17}, [%1 :128]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9");
#endif
        }
        for (; i < size; i++)
        {
#if __aarch64__
            const unsigned short* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v16.4s}, [%8]              \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #64]        \n"
                "ld1    {v0.4h}, [%2], #8           \n" // r0

                "shll   v0.4s, v0.4h, #16           \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%3], #32 \n" // w0123

                "shll   v8.4s, v8.4h, #16           \n"
                "shll   v9.4s, v9.4h, #16           \n"
                "shll   v10.4s, v10.4h, #16         \n"
                "shll   v11.4s, v11.4h, #16         \n"

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v16.4s, v9.4s, v0.s[1]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                "fmla   v16.4s, v11.4s, v0.s[3]     \n"

                "bne    0b                          \n"

                "shrn   v16.4h, v16.4s, #16         \n"

                "st1    {v16.4h}, [%1], #8          \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v16");
#else
            asm volatile(
                "vld1.f32   {d16-d17}, [%8] \n"

                "0:                         \n"

                "pld        [%2, #64]       \n"
                "vld1.u16   {d1}, [%2 :64]! \n"

                "vshll.u16  q0, d1, #16     \n"

                "pld        [%3, #256]      \n"
                "vld1.u16   {d12-d15}, [%3]! \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q8, q5, d0[1]   \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q6, d1[0]   \n"
                "vmla.f32   q8, q7, d1[1]   \n"

                "bne        0b              \n"

                "vshrn.u32  d16, q8, #16    \n"

                "vst1.u16   {d16}, [%1 :64]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8");
#endif
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4_bf16s_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __aarch64__
    kernel_tm.create(32 * maxk, inch / 4, outch / 8 + (outch % 8) / 4, (size_t)2u);
#else
    kernel_tm.create(16 * maxk, inch / 4, outch / 4, (size_t)2u);
#endif

    int q = 0;
#if __aarch64__
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

        unsigned short* g00 = kernel_tm.channel(q / 8);

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
                g00[0] = float32_to_bfloat16(k00[k]);
                g00[1] = float32_to_bfloat16(k10[k]);
                g00[2] = float32_to_bfloat16(k20[k]);
                g00[3] = float32_to_bfloat16(k30[k]);
                g00[4] = float32_to_bfloat16(k40[k]);
                g00[5] = float32_to_bfloat16(k50[k]);
                g00[6] = float32_to_bfloat16(k60[k]);
                g00[7] = float32_to_bfloat16(k70[k]);

                g00[8] = float32_to_bfloat16(k01[k]);
                g00[9] = float32_to_bfloat16(k11[k]);
                g00[10] = float32_to_bfloat16(k21[k]);
                g00[11] = float32_to_bfloat16(k31[k]);
                g00[12] = float32_to_bfloat16(k41[k]);
                g00[13] = float32_to_bfloat16(k51[k]);
                g00[14] = float32_to_bfloat16(k61[k]);
                g00[15] = float32_to_bfloat16(k71[k]);

                g00[16] = float32_to_bfloat16(k02[k]);
                g00[17] = float32_to_bfloat16(k12[k]);
                g00[18] = float32_to_bfloat16(k22[k]);
                g00[19] = float32_to_bfloat16(k32[k]);
                g00[20] = float32_to_bfloat16(k42[k]);
                g00[21] = float32_to_bfloat16(k52[k]);
                g00[22] = float32_to_bfloat16(k62[k]);
                g00[23] = float32_to_bfloat16(k72[k]);

                g00[24] = float32_to_bfloat16(k03[k]);
                g00[25] = float32_to_bfloat16(k13[k]);
                g00[26] = float32_to_bfloat16(k23[k]);
                g00[27] = float32_to_bfloat16(k33[k]);
                g00[28] = float32_to_bfloat16(k43[k]);
                g00[29] = float32_to_bfloat16(k53[k]);
                g00[30] = float32_to_bfloat16(k63[k]);
                g00[31] = float32_to_bfloat16(k73[k]);

                g00 += 32;
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

#if __aarch64__
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4);
#endif

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
                g00[0] = float32_to_bfloat16(k00[k]);
                g00[1] = float32_to_bfloat16(k10[k]);
                g00[2] = float32_to_bfloat16(k20[k]);
                g00[3] = float32_to_bfloat16(k30[k]);

                g00[4] = float32_to_bfloat16(k01[k]);
                g00[5] = float32_to_bfloat16(k11[k]);
                g00[6] = float32_to_bfloat16(k21[k]);
                g00[7] = float32_to_bfloat16(k31[k]);

                g00[8] = float32_to_bfloat16(k02[k]);
                g00[9] = float32_to_bfloat16(k12[k]);
                g00[10] = float32_to_bfloat16(k22[k]);
                g00[11] = float32_to_bfloat16(k32[k]);

                g00[12] = float32_to_bfloat16(k03[k]);
                g00[13] = float32_to_bfloat16(k13[k]);
                g00[14] = float32_to_bfloat16(k23[k]);
                g00[15] = float32_to_bfloat16(k33[k]);

                g00 += 16;
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
            unsigned short* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const unsigned short* sptr = img.row<const unsigned short>(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            uint16x4_t _val0 = vld1_u16(sptr);
                            uint16x4_t _val1 = vld1_u16(sptr + stride_w * 4);
                            uint16x4_t _val2 = vld1_u16(sptr + stride_w * 8);
                            uint16x4_t _val3 = vld1_u16(sptr + stride_w * 12);
                            vst1_u16(ptr, _val0);
                            vst1_u16(ptr + 4, _val1);
                            vst1_u16(ptr + 8, _val2);
                            vst1_u16(ptr + 12, _val3);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            uint16x4_t _val0 = vld1_u16(sptr);
                            uint16x4_t _val1 = vld1_u16(sptr + stride_w * 4);
                            vst1_u16(ptr, _val0);
                            vst1_u16(ptr + 4, _val1);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }
                        for (; j < outw; j++)
                        {
                            uint16x4_t _val = vld1_u16(sptr);
                            vst1_u16(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4_bf16s_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
