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

static void im2col_sgemm_pack4to1_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
#if __aarch64__
    if (size >= 12)
        tmp.create(12 * maxk, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + size % 12 % 4, 16u, 4, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 16u, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 16u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 16u, 4, opt.workspace_allocator);
#else
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 16u, 4, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 16u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 16u, 4, opt.workspace_allocator);
#endif
    {
#if __aarch64__
        int nn_size = size / 12;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 12;

            float* tmpptr = tmp.channel(i / 12);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x12
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        "st1    {v4.4s}, [%1], #16          \n"
                        "st1    {v8.4s}, [%1], #16          \n"
                        "sub    %0, %0, #128                \n"
                        "st1    {v1.4s}, [%1], #16          \n"
                        "st1    {v5.4s}, [%1], #16          \n"
                        "st1    {v9.4s}, [%1], #16          \n"
                        "st1    {v2.4s}, [%1], #16          \n"
                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v10.4s}, [%1], #16         \n"
                        "st1    {v3.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%1], #16          \n"
                        "st1    {v11.4s}, [%1], #16         \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
#else
            float* tmpptr = tmp.channel(i / 8);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x8
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                        "sub    %0, %0, #64                 \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        "st1    {v4.4s}, [%1], #16          \n"
                        "st1    {v1.4s}, [%1], #16          \n"
                        "st1    {v5.4s}, [%1], #16          \n"
                        "st1    {v2.4s}, [%1], #16          \n"
                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v3.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%1], #16          \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0!, {d0-d7}        \n"
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d16-d23}       \n"

                        // transpose 8x4
                        "vtrn.32    q0, q1              \n"
                        "vtrn.32    q2, q3              \n"
                        "vtrn.32    q8, q9              \n"
                        "vtrn.32    q10, q11            \n"
                        "vswp       d1, d4              \n"
                        "vswp       d3, d6              \n"
                        "vswp       d17, d20            \n"
                        "vswp       d19, d22            \n"
                        "vswp       q1, q8              \n"
                        "vswp       q3, q10             \n"

                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "vst1.f32   {d16-d19}, [%1 :128]! \n"
                        "sub        %0, %0, #64         \n"
                        "vst1.f32   {d4-d7}, [%1 :128]! \n"
                        "vst1.f32   {d20-d23}, [%1 :128]! \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
                    // transpose 4x4
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3");
#else
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d0-d3}, [%0 :128]! \n"
                        "pld        [%0, #256]          \n"
                        "vld4.f32   {d4-d7}, [%0 :128]  \n"
                        "sub        %0, %0, #32         \n"
                        "vswp       d1, d4              \n"
                        "vswp       d3, d6              \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        "vst1.f32   {d4-d5}, [%1 :128]! \n"
                        "vst1.f32   {d2-d3}, [%1 :128]! \n"
                        "vst1.f32   {d6-d7}, [%1 :128]! \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "q0", "q1", "q2", "q3");
#endif
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v0.4s}, [%0]               \n"
                        "st1    {v0.4s}, [%1], #16          \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0");
#else
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d0-d1}, [%0 :128]  \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
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

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __aarch64__
    nn_outch = outch >> 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);
        float* outptr4 = top_blob.channel(p + 4);
        float* outptr5 = top_blob.channel(p + 5);
        float* outptr6 = top_blob.channel(p + 6);
        float* outptr7 = top_blob.channel(p + 7);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            float* tmpptr = tmp.channel(i / 12);
            const float* kptr = (const float*)kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v30.4s, v31.4s}, [%22] \n"
                "dup    v8.4s, v30.s[0]         \n"
                "dup    v9.4s, v30.s[0]         \n"
                "dup    v10.4s, v30.s[0]        \n"
                "dup    v11.4s, v30.s[1]        \n"
                "dup    v12.4s, v30.s[1]        \n"
                "dup    v13.4s, v30.s[1]        \n"
                "dup    v14.4s, v30.s[2]        \n"
                "dup    v15.4s, v30.s[2]        \n"
                "dup    v16.4s, v30.s[2]        \n"
                "dup    v17.4s, v30.s[3]        \n"
                "dup    v18.4s, v30.s[3]        \n"
                "dup    v19.4s, v30.s[3]        \n"
                "dup    v20.4s, v31.s[0]        \n"
                "dup    v21.4s, v31.s[0]        \n"
                "dup    v22.4s, v31.s[0]        \n"
                "dup    v23.4s, v31.s[1]        \n"
                "dup    v24.4s, v31.s[1]        \n"
                "dup    v25.4s, v31.s[1]        \n"
                "dup    v26.4s, v31.s[2]        \n"
                "dup    v27.4s, v31.s[2]        \n"
                "dup    v28.4s, v31.s[2]        \n"
                "dup    v29.4s, v31.s[3]        \n"
                "dup    v30.4s, v31.s[3]        \n"
                "dup    v31.4s, v31.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                "fmla   v23.4s, v0.4s, v5.s[1]  \n"
                "fmla   v26.4s, v0.4s, v5.s[2]  \n"
                "fmla   v29.4s, v0.4s, v5.s[3]  \n"

                "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                "fmla   v21.4s, v1.4s, v5.s[0]  \n"
                "fmla   v24.4s, v1.4s, v5.s[1]  \n"
                "fmla   v27.4s, v1.4s, v5.s[2]  \n"
                "fmla   v30.4s, v1.4s, v5.s[3]  \n"

                "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                "fmla   v19.4s, v2.4s, v4.s[3]  \n"
                "fmla   v22.4s, v2.4s, v5.s[0]  \n"
                "fmla   v25.4s, v2.4s, v5.s[1]  \n"
                "fmla   v28.4s, v2.4s, v5.s[2]  \n"
                "fmla   v31.4s, v2.4s, v5.s[3]  \n"

                "fmla   v8.4s, v3.4s, v6.s[0]   \n"
                "fmla   v11.4s, v3.4s, v6.s[1]  \n"
                "fmla   v14.4s, v3.4s, v6.s[2]  \n"
                "fmla   v17.4s, v3.4s, v6.s[3]  \n"
                "fmla   v20.4s, v3.4s, v7.s[0]  \n"
                "fmla   v23.4s, v3.4s, v7.s[1]  \n"
                "fmla   v26.4s, v3.4s, v7.s[2]  \n"
                "fmla   v29.4s, v3.4s, v7.s[3]  \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                "fmla   v9.4s, v0.4s, v6.s[0]   \n"
                "fmla   v12.4s, v0.4s, v6.s[1]  \n"
                "fmla   v15.4s, v0.4s, v6.s[2]  \n"
                "fmla   v18.4s, v0.4s, v6.s[3]  \n"
                "fmla   v21.4s, v0.4s, v7.s[0]  \n"
                "fmla   v24.4s, v0.4s, v7.s[1]  \n"
                "fmla   v27.4s, v0.4s, v7.s[2]  \n"
                "fmla   v30.4s, v0.4s, v7.s[3]  \n"

                "fmla   v10.4s, v1.4s, v6.s[0]  \n"
                "fmla   v13.4s, v1.4s, v6.s[1]  \n"
                "fmla   v16.4s, v1.4s, v6.s[2]  \n"
                "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                "fmla   v22.4s, v1.4s, v7.s[0]  \n"
                "fmla   v25.4s, v1.4s, v7.s[1]  \n"
                "fmla   v28.4s, v1.4s, v7.s[2]  \n"
                "fmla   v31.4s, v1.4s, v7.s[3]  \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                "fmla   v8.4s, v2.4s, v4.s[0]   \n"
                "fmla   v11.4s, v2.4s, v4.s[1]  \n"
                "fmla   v14.4s, v2.4s, v4.s[2]  \n"
                "fmla   v17.4s, v2.4s, v4.s[3]  \n"
                "fmla   v20.4s, v2.4s, v5.s[0]  \n"
                "fmla   v23.4s, v2.4s, v5.s[1]  \n"
                "fmla   v26.4s, v2.4s, v5.s[2]  \n"
                "fmla   v29.4s, v2.4s, v5.s[3]  \n"

                "fmla   v9.4s, v3.4s, v4.s[0]   \n"
                "fmla   v12.4s, v3.4s, v4.s[1]  \n"
                "fmla   v15.4s, v3.4s, v4.s[2]  \n"
                "fmla   v18.4s, v3.4s, v4.s[3]  \n"
                "fmla   v21.4s, v3.4s, v5.s[0]  \n"
                "fmla   v24.4s, v3.4s, v5.s[1]  \n"
                "fmla   v27.4s, v3.4s, v5.s[2]  \n"
                "fmla   v30.4s, v3.4s, v5.s[3]  \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                "fmla   v10.4s, v0.4s, v4.s[0]  \n"
                "fmla   v13.4s, v0.4s, v4.s[1]  \n"
                "fmla   v16.4s, v0.4s, v4.s[2]  \n"
                "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                "fmla   v22.4s, v0.4s, v5.s[0]  \n"
                "fmla   v25.4s, v0.4s, v5.s[1]  \n"
                "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                "fmla   v31.4s, v0.4s, v5.s[3]  \n"

                "fmla   v8.4s, v1.4s, v6.s[0]   \n"
                "fmla   v11.4s, v1.4s, v6.s[1]  \n"
                "fmla   v14.4s, v1.4s, v6.s[2]  \n"
                "fmla   v17.4s, v1.4s, v6.s[3]  \n"
                "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                "fmla   v23.4s, v1.4s, v7.s[1]  \n"
                "fmla   v26.4s, v1.4s, v7.s[2]  \n"
                "fmla   v29.4s, v1.4s, v7.s[3]  \n"

                "fmla   v9.4s, v2.4s, v6.s[0]   \n"
                "fmla   v12.4s, v2.4s, v6.s[1]  \n"
                "fmla   v15.4s, v2.4s, v6.s[2]  \n"
                "fmla   v18.4s, v2.4s, v6.s[3]  \n"
                "fmla   v21.4s, v2.4s, v7.s[0]  \n"
                "fmla   v24.4s, v2.4s, v7.s[1]  \n"
                "fmla   v27.4s, v2.4s, v7.s[2]  \n"
                "fmla   v30.4s, v2.4s, v7.s[3]  \n"

                "fmla   v10.4s, v3.4s, v6.s[0]  \n"
                "fmla   v13.4s, v3.4s, v6.s[1]  \n"
                "fmla   v16.4s, v3.4s, v6.s[2]  \n"
                "fmla   v19.4s, v3.4s, v6.s[3]  \n"
                "fmla   v22.4s, v3.4s, v7.s[0]  \n"
                "fmla   v25.4s, v3.4s, v7.s[1]  \n"
                "fmla   v28.4s, v3.4s, v7.s[2]  \n"
                "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                "bne    0b                      \n"

                "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"
                "st1    {v20.4s, v21.4s, v22.4s}, [%5], #48 \n"
                "st1    {v23.4s, v24.4s, v25.4s}, [%6], #48 \n"
                "st1    {v26.4s, v27.4s, v28.4s}, [%7], #48 \n"
                "st1    {v29.4s, v30.4s, v31.4s}, [%8], #48 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(outptr4), // %5
                "=r"(outptr5), // %6
                "=r"(outptr6), // %7
                "=r"(outptr7), // %8
                "=r"(tmpptr),  // %9
                "=r"(kptr)     // %10
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(outptr6),
                "8"(outptr7),
                "9"(tmpptr),
                "10"(kptr),
                "r"(biasptr) // %22
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr = (const float*)kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v30.4s, v31.4s}, [%22] \n"
                "dup    v16.4s, v30.s[0]        \n"
                "dup    v17.4s, v30.s[0]        \n"
                "dup    v18.4s, v30.s[1]        \n"
                "dup    v19.4s, v30.s[1]        \n"
                "dup    v20.4s, v30.s[2]        \n"
                "dup    v21.4s, v30.s[2]        \n"
                "dup    v22.4s, v30.s[3]        \n"
                "dup    v23.4s, v30.s[3]        \n"
                "dup    v24.4s, v31.s[0]        \n"
                "dup    v25.4s, v31.s[0]        \n"
                "dup    v26.4s, v31.s[1]        \n"
                "dup    v27.4s, v31.s[1]        \n"
                "dup    v28.4s, v31.s[2]        \n"
                "dup    v29.4s, v31.s[2]        \n"
                "dup    v30.4s, v31.s[3]        \n"
                "dup    v31.4s, v31.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                "fmla   v18.4s, v0.4s, v4.s[1]  \n"
                "fmla   v20.4s, v0.4s, v4.s[2]  \n"
                "fmla   v22.4s, v0.4s, v4.s[3]  \n"
                "fmla   v24.4s, v0.4s, v5.s[0]  \n"
                "fmla   v26.4s, v0.4s, v5.s[1]  \n"
                "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                "fmla   v30.4s, v0.4s, v5.s[3]  \n"
                "fmla   v17.4s, v1.4s, v4.s[0]  \n"
                "fmla   v19.4s, v1.4s, v4.s[1]  \n"
                "fmla   v21.4s, v1.4s, v4.s[2]  \n"
                "fmla   v23.4s, v1.4s, v4.s[3]  \n"
                "fmla   v25.4s, v1.4s, v5.s[0]  \n"
                "fmla   v27.4s, v1.4s, v5.s[1]  \n"
                "fmla   v29.4s, v1.4s, v5.s[2]  \n"
                "fmla   v31.4s, v1.4s, v5.s[3]  \n"

                "fmla   v16.4s, v2.4s, v6.s[0]  \n"
                "fmla   v18.4s, v2.4s, v6.s[1]  \n"
                "fmla   v20.4s, v2.4s, v6.s[2]  \n"
                "fmla   v22.4s, v2.4s, v6.s[3]  \n"
                "fmla   v24.4s, v2.4s, v7.s[0]  \n"
                "fmla   v26.4s, v2.4s, v7.s[1]  \n"
                "fmla   v28.4s, v2.4s, v7.s[2]  \n"
                "fmla   v30.4s, v2.4s, v7.s[3]  \n"
                "fmla   v17.4s, v3.4s, v6.s[0]  \n"
                "fmla   v19.4s, v3.4s, v6.s[1]  \n"
                "fmla   v21.4s, v3.4s, v6.s[2]  \n"
                "fmla   v23.4s, v3.4s, v6.s[3]  \n"
                "fmla   v25.4s, v3.4s, v7.s[0]  \n"
                "fmla   v27.4s, v3.4s, v7.s[1]  \n"
                "fmla   v29.4s, v3.4s, v7.s[2]  \n"
                "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%9], #64 \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                "fmla   v16.4s, v12.4s, v8.s[0] \n"
                "fmla   v18.4s, v12.4s, v8.s[1] \n"
                "fmla   v20.4s, v12.4s, v8.s[2] \n"
                "fmla   v22.4s, v12.4s, v8.s[3] \n"
                "fmla   v24.4s, v12.4s, v9.s[0] \n"
                "fmla   v26.4s, v12.4s, v9.s[1] \n"
                "fmla   v28.4s, v12.4s, v9.s[2] \n"
                "fmla   v30.4s, v12.4s, v9.s[3] \n"
                "fmla   v17.4s, v13.4s, v8.s[0] \n"
                "fmla   v19.4s, v13.4s, v8.s[1] \n"
                "fmla   v21.4s, v13.4s, v8.s[2] \n"
                "fmla   v23.4s, v13.4s, v8.s[3] \n"
                "fmla   v25.4s, v13.4s, v9.s[0] \n"
                "fmla   v27.4s, v13.4s, v9.s[1] \n"
                "fmla   v29.4s, v13.4s, v9.s[2] \n"
                "fmla   v31.4s, v13.4s, v9.s[3] \n"

                "fmla   v16.4s, v14.4s, v10.s[0] \n"
                "fmla   v18.4s, v14.4s, v10.s[1] \n"
                "fmla   v20.4s, v14.4s, v10.s[2] \n"
                "fmla   v22.4s, v14.4s, v10.s[3] \n"
                "fmla   v24.4s, v14.4s, v11.s[0] \n"
                "fmla   v26.4s, v14.4s, v11.s[1] \n"
                "fmla   v28.4s, v14.4s, v11.s[2] \n"
                "fmla   v30.4s, v14.4s, v11.s[3] \n"
                "fmla   v17.4s, v15.4s, v10.s[0] \n"
                "fmla   v19.4s, v15.4s, v10.s[1] \n"
                "fmla   v21.4s, v15.4s, v10.s[2] \n"
                "fmla   v23.4s, v15.4s, v10.s[3] \n"
                "fmla   v25.4s, v15.4s, v11.s[0] \n"
                "fmla   v27.4s, v15.4s, v11.s[1] \n"
                "fmla   v29.4s, v15.4s, v11.s[2] \n"
                "fmla   v31.4s, v15.4s, v11.s[3] \n"

                "bne    0b                      \n"

                "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                "st1    {v18.4s, v19.4s}, [%2], #32 \n"
                "st1    {v20.4s, v21.4s}, [%3], #32 \n"
                "st1    {v22.4s, v23.4s}, [%4], #32 \n"
                "st1    {v24.4s, v25.4s}, [%5], #32 \n"
                "st1    {v26.4s, v27.4s}, [%6], #32 \n"
                "st1    {v28.4s, v29.4s}, [%7], #32 \n"
                "st1    {v30.4s, v31.4s}, [%8], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(outptr4), // %5
                "=r"(outptr5), // %6
                "=r"(outptr6), // %7
                "=r"(outptr7), // %8
                "=r"(tmpptr),  // %9
                "=r"(kptr)     // %10
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(outptr6),
                "8"(outptr7),
                "9"(tmpptr),
                "10"(kptr),
                "r"(biasptr) // %22
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr = (const float*)kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v22.4s, v23.4s}, [%22] \n"
                "dup    v16.4s, v22.s[0]        \n"
                "dup    v17.4s, v22.s[1]        \n"
                "dup    v18.4s, v22.s[2]        \n"
                "dup    v19.4s, v22.s[3]        \n"
                "dup    v20.4s, v23.s[0]        \n"
                "dup    v21.4s, v23.s[1]        \n"
                "dup    v22.4s, v23.s[2]        \n"
                "dup    v23.4s, v23.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                "fmla   v17.4s, v0.4s, v4.s[1]  \n"
                "fmla   v18.4s, v0.4s, v4.s[2]  \n"
                "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                "fmla   v21.4s, v0.4s, v5.s[1]  \n"
                "fmla   v22.4s, v0.4s, v5.s[2]  \n"
                "fmla   v23.4s, v0.4s, v5.s[3]  \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                "fmla   v16.4s, v1.4s, v6.s[0]  \n"
                "fmla   v17.4s, v1.4s, v6.s[1]  \n"
                "fmla   v18.4s, v1.4s, v6.s[2]  \n"
                "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                "fmla   v21.4s, v1.4s, v7.s[1]  \n"
                "fmla   v22.4s, v1.4s, v7.s[2]  \n"
                "fmla   v23.4s, v1.4s, v7.s[3]  \n"

                "fmla   v16.4s, v2.4s, v8.s[0]  \n"
                "fmla   v17.4s, v2.4s, v8.s[1]  \n"
                "fmla   v18.4s, v2.4s, v8.s[2]  \n"
                "fmla   v19.4s, v2.4s, v8.s[3]  \n"
                "fmla   v20.4s, v2.4s, v9.s[0]  \n"
                "fmla   v21.4s, v2.4s, v9.s[1]  \n"
                "fmla   v22.4s, v2.4s, v9.s[2]  \n"
                "fmla   v23.4s, v2.4s, v9.s[3]  \n"

                "fmla   v16.4s, v3.4s, v10.s[0] \n"
                "fmla   v17.4s, v3.4s, v10.s[1] \n"
                "fmla   v18.4s, v3.4s, v10.s[2] \n"
                "fmla   v19.4s, v3.4s, v10.s[3] \n"
                "fmla   v20.4s, v3.4s, v11.s[0] \n"
                "fmla   v21.4s, v3.4s, v11.s[1] \n"
                "fmla   v22.4s, v3.4s, v11.s[2] \n"
                "fmla   v23.4s, v3.4s, v11.s[3] \n"

                "bne    0b                      \n"

                "st1    {v16.4s}, [%1], #16     \n"
                "st1    {v17.4s}, [%2], #16     \n"
                "st1    {v18.4s}, [%3], #16     \n"
                "st1    {v19.4s}, [%4], #16     \n"
                "st1    {v20.4s}, [%5], #16     \n"
                "st1    {v21.4s}, [%6], #16     \n"
                "st1    {v22.4s}, [%7], #16     \n"
                "st1    {v23.4s}, [%8], #16     \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(outptr4), // %5
                "=r"(outptr5), // %6
                "=r"(outptr6), // %7
                "=r"(outptr7), // %8
                "=r"(tmpptr),  // %9
                "=r"(kptr)     // %10
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(outptr6),
                "8"(outptr7),
                "9"(tmpptr),
                "10"(kptr),
                "r"(biasptr) // %22
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr = (const float*)kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v16.4s, v17.4s}, [%22] \n"
                "eor    v18.16b, v18.16b, v18.16b \n"
                "eor    v19.16b, v19.16b, v19.16b \n"

                "0:                             \n"

                "prfm   pldl1keep, [%9, #128]   \n"
                "ld1    {v0.4s}, [%9], #16      \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v16.4s, v4.4s, v0.s[0]  \n"
                "fmla   v17.4s, v5.4s, v0.s[0]  \n"
                "fmla   v18.4s, v6.4s, v0.s[1]  \n"
                "fmla   v19.4s, v7.4s, v0.s[1]  \n"

                "prfm   pldl1keep, [%10, #512]  \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                "fmla   v16.4s, v8.4s, v0.s[2]  \n"
                "fmla   v17.4s, v9.4s, v0.s[2]  \n"
                "fmla   v18.4s, v10.4s, v0.s[3] \n"
                "fmla   v19.4s, v11.4s, v0.s[3] \n"

                "bne    0b                      \n"

                "fadd   v16.4s, v16.4s, v18.4s  \n"
                "fadd   v17.4s, v17.4s, v19.4s  \n"

                "st1    {v16.s}[0], [%1], #4    \n"
                "st1    {v16.s}[1], [%2], #4    \n"
                "st1    {v16.s}[2], [%3], #4    \n"
                "st1    {v16.s}[3], [%4], #4    \n"
                "st1    {v17.s}[0], [%5], #4    \n"
                "st1    {v17.s}[1], [%6], #4    \n"
                "st1    {v17.s}[2], [%7], #4    \n"
                "st1    {v17.s}[3], [%8], #4    \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(outptr4), // %5
                "=r"(outptr5), // %6
                "=r"(outptr6), // %7
                "=r"(outptr7), // %8
                "=r"(tmpptr),  // %9
                "=r"(kptr)     // %10
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(outptr4),
                "6"(outptr5),
                "7"(outptr6),
                "8"(outptr7),
                "9"(tmpptr),
                "10"(kptr),
                "r"(biasptr) // %22
                : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
        }
    }

    remain_outch_start += nn_outch << 3;
    nn_outch = (outch - remain_outch_start) >> 2;
#else  // __aarch64__
    nn_outch = outch >> 2;
#endif // __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
#if __aarch64__
        for (; i + 11 < size; i += 12)
        {
            float* tmpptr = tmp.channel(i / 12);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v19.4s}, [%14]         \n"
                "dup    v8.4s, v19.s[0]         \n"
                "dup    v9.4s, v19.s[0]         \n"
                "dup    v10.4s, v19.s[0]        \n"
                "dup    v11.4s, v19.s[1]        \n"
                "dup    v12.4s, v19.s[1]        \n"
                "dup    v13.4s, v19.s[1]        \n"
                "dup    v14.4s, v19.s[2]        \n"
                "dup    v15.4s, v19.s[2]        \n"
                "dup    v16.4s, v19.s[2]        \n"
                "dup    v17.4s, v19.s[3]        \n"
                "dup    v18.4s, v19.s[3]        \n"
                "dup    v19.4s, v19.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                "prfm   pldl1keep, [%6, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                "fmla   v19.4s, v2.4s, v4.s[3]  \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%5], #64 \n"

                "fmla   v8.4s, v3.4s, v5.s[0]   \n"
                "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                "fmla   v14.4s, v3.4s, v5.s[2]  \n"
                "fmla   v17.4s, v3.4s, v5.s[3]  \n"
                "fmla   v9.4s, v20.4s, v5.s[0]  \n"
                "fmla   v12.4s, v20.4s, v5.s[1] \n"
                "fmla   v15.4s, v20.4s, v5.s[2] \n"
                "fmla   v18.4s, v20.4s, v5.s[3] \n"
                "fmla   v10.4s, v21.4s, v5.s[0] \n"
                "fmla   v13.4s, v21.4s, v5.s[1] \n"
                "fmla   v16.4s, v21.4s, v5.s[2] \n"
                "fmla   v19.4s, v21.4s, v5.s[3] \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%5], #64 \n"

                "fmla   v8.4s, v22.4s, v6.s[0]  \n"
                "fmla   v11.4s, v22.4s, v6.s[1] \n"
                "fmla   v14.4s, v22.4s, v6.s[2] \n"
                "fmla   v17.4s, v22.4s, v6.s[3] \n"
                "fmla   v9.4s, v23.4s, v6.s[0]  \n"
                "fmla   v12.4s, v23.4s, v6.s[1] \n"
                "fmla   v15.4s, v23.4s, v6.s[2] \n"
                "fmla   v18.4s, v23.4s, v6.s[3] \n"
                "fmla   v10.4s, v24.4s, v6.s[0] \n"
                "fmla   v13.4s, v24.4s, v6.s[1] \n"
                "fmla   v16.4s, v24.4s, v6.s[2] \n"
                "fmla   v19.4s, v24.4s, v6.s[3] \n"

                "fmla   v8.4s, v25.4s, v7.s[0]  \n"
                "fmla   v11.4s, v25.4s, v7.s[1] \n"
                "fmla   v14.4s, v25.4s, v7.s[2] \n"
                "fmla   v17.4s, v25.4s, v7.s[3] \n"
                "fmla   v9.4s, v26.4s, v7.s[0]  \n"
                "fmla   v12.4s, v26.4s, v7.s[1] \n"
                "fmla   v15.4s, v26.4s, v7.s[2] \n"
                "fmla   v18.4s, v26.4s, v7.s[3] \n"
                "fmla   v10.4s, v27.4s, v7.s[0] \n"
                "fmla   v13.4s, v27.4s, v7.s[1] \n"
                "fmla   v16.4s, v27.4s, v7.s[2] \n"
                "fmla   v19.4s, v27.4s, v7.s[3] \n"

                "bne    0b                      \n"

                "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr = (const float*)kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v15.4s}, [%14]         \n"
                "dup    v8.4s, v15.s[0]         \n"
                "dup    v9.4s, v15.s[0]         \n"
                "dup    v10.4s, v15.s[1]        \n"
                "dup    v11.4s, v15.s[1]        \n"
                "dup    v12.4s, v15.s[2]        \n"
                "dup    v13.4s, v15.s[2]        \n"
                "dup    v14.4s, v15.s[3]        \n"
                "dup    v15.4s, v15.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                "prfm   pldl1keep, [%6, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v10.4s, v0.4s, v4.s[1]  \n"
                "fmla   v12.4s, v0.4s, v4.s[2]  \n"
                "fmla   v14.4s, v0.4s, v4.s[3]  \n"
                "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                "fmla   v11.4s, v1.4s, v4.s[1]  \n"
                "fmla   v13.4s, v1.4s, v4.s[2]  \n"
                "fmla   v15.4s, v1.4s, v4.s[3]  \n"

                "fmla   v8.4s, v2.4s, v5.s[0]   \n"
                "fmla   v10.4s, v2.4s, v5.s[1]  \n"
                "fmla   v12.4s, v2.4s, v5.s[2]  \n"
                "fmla   v14.4s, v2.4s, v5.s[3]  \n"
                "fmla   v9.4s, v3.4s, v5.s[0]   \n"
                "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                "fmla   v13.4s, v3.4s, v5.s[2]  \n"
                "fmla   v15.4s, v3.4s, v5.s[3]  \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%5], #64 \n"

                "fmla   v8.4s, v16.4s, v6.s[0]  \n"
                "fmla   v10.4s, v16.4s, v6.s[1] \n"
                "fmla   v12.4s, v16.4s, v6.s[2] \n"
                "fmla   v14.4s, v16.4s, v6.s[3] \n"
                "fmla   v9.4s, v17.4s, v6.s[0]  \n"
                "fmla   v11.4s, v17.4s, v6.s[1] \n"
                "fmla   v13.4s, v17.4s, v6.s[2] \n"
                "fmla   v15.4s, v17.4s, v6.s[3] \n"

                "fmla   v8.4s, v18.4s, v7.s[0]  \n"
                "fmla   v10.4s, v18.4s, v7.s[1] \n"
                "fmla   v12.4s, v18.4s, v7.s[2] \n"
                "fmla   v14.4s, v18.4s, v7.s[3] \n"
                "fmla   v9.4s, v19.4s, v7.s[0]  \n"
                "fmla   v11.4s, v19.4s, v7.s[1] \n"
                "fmla   v13.4s, v19.4s, v7.s[2] \n"
                "fmla   v15.4s, v19.4s, v7.s[3] \n"

                "bne    0b                      \n"

                "st1    {v8.4s, v9.4s}, [%1], #32 \n"
                "st1    {v10.4s, v11.4s}, [%2], #32 \n"
                "st1    {v12.4s, v13.4s}, [%3], #32 \n"
                "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d30-d31}, [%14] \n"
                "vdup.f32   q8, d30[0]      \n"
                "vdup.f32   q9, d30[0]      \n"
                "vdup.f32   q10, d30[1]     \n"
                "vdup.f32   q11, d30[1]     \n"
                "vdup.f32   q12, d31[0]     \n"
                "vdup.f32   q13, d31[0]     \n"
                "vdup.f32   q14, d31[1]     \n"
                "vdup.f32   q15, d31[1]     \n"

                "0:                         \n"

                "pld        [%5, #512]      \n"
                "vldm       %5!, {d0-d7}    \n"

                "pld        [%6, #512]      \n"
                "vldm       %6!, {d8-d15}   \n"

                "vmla.f32   q8, q0, d8[0]   \n"
                "vmla.f32   q10, q0, d8[1]  \n"
                "vmla.f32   q12, q0, d9[0]  \n"
                "vmla.f32   q14, q0, d9[1]  \n"
                "vmla.f32   q9, q1, d8[0]   \n"
                "vmla.f32   q11, q1, d8[1]  \n"
                "vmla.f32   q13, q1, d9[0]  \n"
                "vmla.f32   q15, q1, d9[1]  \n"

                "vmla.f32   q8, q2, d10[0]  \n"
                "vmla.f32   q10, q2, d10[1] \n"
                "vmla.f32   q12, q2, d11[0] \n"
                "vmla.f32   q14, q2, d11[1] \n"
                "vmla.f32   q9, q3, d10[0]  \n"
                "vmla.f32   q11, q3, d10[1] \n"
                "vmla.f32   q13, q3, d11[0] \n"
                "vmla.f32   q15, q3, d11[1] \n"

                "pld        [%5, #512]      \n"
                "vldm       %5!, {d0-d7}    \n"

                "vmla.f32   q8, q0, d12[0]  \n"
                "vmla.f32   q10, q0, d12[1] \n"
                "vmla.f32   q12, q0, d13[0] \n"
                "vmla.f32   q14, q0, d13[1] \n"
                "vmla.f32   q9, q1, d12[0]  \n"
                "vmla.f32   q11, q1, d12[1] \n"
                "vmla.f32   q13, q1, d13[0] \n"
                "vmla.f32   q15, q1, d13[1] \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q2, d14[0]  \n"
                "vmla.f32   q10, q2, d14[1] \n"
                "vmla.f32   q12, q2, d15[0] \n"
                "vmla.f32   q14, q2, d15[1] \n"
                "vmla.f32   q9, q3, d14[0]  \n"
                "vmla.f32   q11, q3, d14[1] \n"
                "vmla.f32   q13, q3, d15[0] \n"
                "vmla.f32   q15, q3, d15[1] \n"

                "bne        0b              \n"

                "vst1.f32   {d16-d19}, [%1 :128]! \n"
                "vst1.f32   {d20-d23}, [%2 :128]! \n"
                "vst1.f32   {d24-d27}, [%3 :128]! \n"
                "vst1.f32   {d28-d31}, [%4 :128]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr = (const float*)kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v11.4s}, [%14]         \n"
                "dup    v8.4s, v11.s[0]         \n"
                "dup    v9.4s, v11.s[1]         \n"
                "dup    v10.4s, v11.s[2]        \n"
                "dup    v11.4s, v11.s[3]        \n"

                "0:                             \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                "prfm   pldl1keep, [%6, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v9.4s, v0.4s, v4.s[1]   \n"
                "fmla   v10.4s, v0.4s, v4.s[2]  \n"
                "fmla   v11.4s, v0.4s, v4.s[3]  \n"

                "fmla   v8.4s, v1.4s, v5.s[0]   \n"
                "fmla   v9.4s, v1.4s, v5.s[1]   \n"
                "fmla   v10.4s, v1.4s, v5.s[2]  \n"
                "fmla   v11.4s, v1.4s, v5.s[3]  \n"

                "fmla   v8.4s, v2.4s, v6.s[0]   \n"
                "fmla   v9.4s, v2.4s, v6.s[1]   \n"
                "fmla   v10.4s, v2.4s, v6.s[2]  \n"
                "fmla   v11.4s, v2.4s, v6.s[3]  \n"

                "fmla   v8.4s, v3.4s, v7.s[0]   \n"
                "fmla   v9.4s, v3.4s, v7.s[1]   \n"
                "fmla   v10.4s, v3.4s, v7.s[2]  \n"
                "fmla   v11.4s, v3.4s, v7.s[3]  \n"

                "bne    0b                      \n"

                "st1    {v8.4s}, [%1], #16      \n"
                "st1    {v9.4s}, [%2], #16      \n"
                "st1    {v10.4s}, [%3], #16     \n"
                "st1    {v11.4s}, [%4], #16     \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d22-d23}, [%14] \n"
                "vdup.f32   q8, d22[0]      \n"
                "vdup.f32   q9, d22[1]      \n"
                "vdup.f32   q10, d23[0]     \n"
                "vdup.f32   q11, d23[1]     \n"

                "0:                         \n"

                "pld        [%5, #512]      \n"
                "vldm       %5!, {d0-d7}    \n"

                "pld        [%6, #512]      \n"
                "vldm       %6!, {d8-d15}   \n"

                "vmla.f32   q8, q0, d8[0]   \n"
                "vmla.f32   q9, q0, d8[1]   \n"
                "vmla.f32   q10, q0, d9[0]  \n"
                "vmla.f32   q11, q0, d9[1]  \n"

                "vmla.f32   q8, q1, d10[0]  \n"
                "vmla.f32   q9, q1, d10[1]  \n"
                "vmla.f32   q10, q1, d11[0] \n"
                "vmla.f32   q11, q1, d11[1] \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q2, d12[0]  \n"
                "vmla.f32   q9, q2, d12[1]  \n"
                "vmla.f32   q10, q2, d13[0] \n"
                "vmla.f32   q11, q2, d13[1] \n"

                "vmla.f32   q8, q3, d14[0]  \n"
                "vmla.f32   q9, q3, d14[1]  \n"
                "vmla.f32   q10, q3, d15[0] \n"
                "vmla.f32   q11, q3, d15[1] \n"

                "bne        0b              \n"

                "vst1.f32   {d16-d17}, [%1 :128]! \n"
                "vst1.f32   {d18-d19}, [%2 :128]! \n"
                "vst1.f32   {d20-d21}, [%3 :128]! \n"
                "vst1.f32   {d22-d23}, [%4 :128]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
        }
        for (; i < size; i++)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr = (const float*)kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v8.4s}, [%14]          \n"
                "eor    v9.16b, v9.16b, v9.16b  \n"
                "eor    v10.16b, v10.16b, v10.16b \n"
                "eor    v11.16b, v11.16b, v11.16b \n"

                "0:                             \n"

                "prfm   pldl1keep, [%5, #128]   \n"
                "ld1    {v0.4s}, [%5], #16      \n"

                "prfm   pldl1keep, [%6, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v5.4s, v0.s[1]   \n"
                "fmla   v10.4s, v6.4s, v0.s[2]  \n"
                "fmla   v11.4s, v7.4s, v0.s[3]  \n"

                "bne    0b                      \n"

                "fadd   v8.4s, v8.4s, v9.4s     \n"
                "fadd   v10.4s, v10.4s, v11.4s  \n"
                "fadd   v8.4s, v8.4s, v10.4s    \n"

                "st1    {v8.s}[0], [%1], #4     \n"
                "st1    {v8.s}[1], [%2], #4     \n"
                "st1    {v8.s}[2], [%3], #4     \n"
                "st1    {v8.s}[3], [%4], #4     \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d16-d17}, [%14] \n"
                "veor       q9, q9          \n"
                "veor       q10, q10        \n"
                "veor       q11, q11        \n"

                "0:                         \n"

                "pld        [%5, #128]      \n"
                "vld1.f32   {d0-d1}, [%5]!  \n"

                "pld        [%6, #512]      \n"
                "vldm       %6!, {d8-d15}   \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q9, q5, d0[1]   \n"
                "vmla.f32   q10, q6, d1[0]  \n"
                "vmla.f32   q11, q7, d1[1]  \n"

                "bne        0b              \n"

                "vadd.f32   q8, q8, q9      \n"
                "vadd.f32   q10, q10, q11   \n"
                "vadd.f32   q8, q8, q10     \n"

                "vst1.f32   {d16[0]}, [%1]! \n"
                "vst1.f32   {d16[1]}, [%2]! \n"
                "vst1.f32   {d17[0]}, [%3]! \n"
                "vst1.f32   {d17[1]}, [%4]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(outptr2), // %3
                "=r"(outptr3), // %4
                "=r"(tmpptr),  // %5
                "=r"(kptr)     // %6
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(outptr2),
                "4"(outptr3),
                "5"(tmpptr),
                "6"(kptr),
                "r"(biasptr) // %14
                : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
#if __aarch64__
        for (; i + 11 < size; i += 12)
        {
            float* tmpptr = tmp.channel(i / 12);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "dup    v8.4s, %w8              \n"
                "dup    v9.4s, %w8              \n"
                "dup    v10.4s, %w8             \n"
                "eor    v5.16b, v5.16b, v5.16b  \n"
                "eor    v6.16b, v6.16b, v6.16b  \n"
                "eor    v7.16b, v7.16b, v7.16b  \n"

                "0:                             \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "prfm   pldl1keep, [%3, #128]   \n"
                "ld1    {v4.4s}, [%3], #16      \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                "fmla   v10.4s, v2.4s, v4.s[0]  \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                "fmla   v5.4s, v3.4s, v4.s[1]   \n"
                "fmla   v6.4s, v12.4s, v4.s[1]  \n"
                "fmla   v7.4s, v13.4s, v4.s[1]  \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n"

                "fmla   v8.4s, v14.4s, v4.s[2]  \n"
                "fmla   v9.4s, v15.4s, v4.s[2]  \n"
                "fmla   v10.4s, v16.4s, v4.s[2] \n"

                "fmla   v5.4s, v17.4s, v4.s[3]  \n"
                "fmla   v6.4s, v18.4s, v4.s[3]  \n"
                "fmla   v7.4s, v19.4s, v4.s[3]  \n"

                "bne    0b                      \n"

                "fadd   v8.4s, v8.4s, v5.4s     \n"
                "fadd   v9.4s, v9.4s, v6.4s     \n"
                "fadd   v10.4s, v10.4s, v7.4s   \n"

                "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(bias0) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr = (const float*)kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w8              \n"
                "dup    v9.4s, %w8              \n"
                "eor    v10.16b, v10.16b, v10.16b \n"
                "eor    v11.16b, v11.16b, v11.16b \n"

                "0:                             \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "prfm   pldl1keep, [%3, #128]   \n"
                "ld1    {v4.4s}, [%3], #16      \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                "fmla   v10.4s, v2.4s, v4.s[1]  \n"
                "fmla   v11.4s, v3.4s, v4.s[1]  \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                "fmla   v8.4s, v12.4s, v4.s[2]  \n"
                "fmla   v9.4s, v13.4s, v4.s[2]  \n"
                "fmla   v10.4s, v14.4s, v4.s[3] \n"
                "fmla   v11.4s, v15.4s, v4.s[3] \n"

                "bne    0b                      \n"

                "fadd   v8.4s, v8.4s, v10.4s    \n"
                "fadd   v9.4s, v9.4s, v11.4s    \n"

                "st1    {v8.4s, v9.4s}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(bias0) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %8          \n"
                "vdup.f32   q9, %8          \n"
                "veor       q10, q10        \n"
                "veor       q11, q11        \n"

                "0:                         \n"

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d0-d7}    \n"

                "pld        [%3, #128]      \n"
                "vld1.f32   {d8-d9}, [%3]!  \n"

                "vmla.f32   q8, q0, d8[0]   \n"
                "vmla.f32   q9, q1, d8[0]   \n"
                "vmla.f32   q10, q2, d8[1]  \n"
                "vmla.f32   q11, q3, d8[1]  \n"

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d24-d31}  \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q12, d9[0]  \n"
                "vmla.f32   q9, q13, d9[0]  \n"
                "vmla.f32   q10, q14, d9[1] \n"
                "vmla.f32   q11, q15, d9[1] \n"

                "bne        0b              \n"

                "vadd.f32   q8, q8, q10     \n"
                "vadd.f32   q9, q9, q11     \n"

                "vst1.f32   {d16-d19}, [%1 :128]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(bias0) // %8
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr = (const float*)kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w8              \n"
                "eor    v9.16b, v9.16b, v9.16b  \n"
                "eor    v10.16b, v10.16b, v10.16b \n"
                "eor    v11.16b, v11.16b, v11.16b \n"

                "0:                             \n"

                "prfm   pldl1keep, [%2, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "prfm   pldl1keep, [%3, #128]   \n"
                "ld1    {v4.4s}, [%3], #16      \n"

                "subs   %w0, %w0, #1            \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v9.4s, v1.4s, v4.s[1]   \n"
                "fmla   v10.4s, v2.4s, v4.s[2]  \n"
                "fmla   v11.4s, v3.4s, v4.s[3]  \n"

                "bne    0b                      \n"

                "fadd   v8.4s, v8.4s, v9.4s     \n"
                "fadd   v10.4s, v10.4s, v11.4s  \n"
                "fadd   v8.4s, v8.4s, v10.4s    \n"

                "st1    {v8.4s}, [%1], #16      \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(bias0) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %8          \n"
                "veor       q9, q9          \n"
                "veor       q10, q10        \n"
                "veor       q11, q11        \n"

                "0:                         \n"

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d0-d7}    \n"

                "pld        [%3, #128]      \n"
                "vld1.f32   {d8-d9}, [%3]!  \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q0, d8[0]   \n"
                "vmla.f32   q9, q1, d8[1]   \n"
                "vmla.f32   q10, q2, d9[0]  \n"
                "vmla.f32   q11, q3, d9[1]  \n"

                "bne        0b              \n"

                "vadd.f32   q8, q8, q9      \n"
                "vadd.f32   q10, q10, q11   \n"
                "vadd.f32   q8, q8, q10     \n"

                "vst1.f32   {d16-d17}, [%1]! \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(bias0) // %8
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11");
#endif // __aarch64__
        }
        for (; i < size; i++)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4);
            const float* kptr = (const float*)kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr = (const float*)kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

            float32x4_t _sum0 = vdupq_n_f32(0.f);

            for (int q = 0; q < nn; q++)
            {
                float32x4_t _r0 = vld1q_f32(tmpptr);

                float32x4_t _k0 = vld1q_f32(kptr);

                _sum0 = vmlaq_f32(_sum0, _r0, _k0);

                kptr += 4;
                tmpptr += 4;
            }

#if __aarch64__
            float sum0 = vaddvq_f32(_sum0);
#else
            float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
            float32x2_t _ss2 = vpadd_f32(_ss, _ss);
            float sum0 = vget_lane_f32(_ss2, 0);
#endif

            outptr0[0] = bias0 + sum0;

            outptr0++;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack4to1_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __aarch64__
    kernel_tm.create(32 * maxk, inch / 4, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm.create(16 * maxk, inch / 4, outch / 4 + outch % 4);
#endif

    int q = 0;
#if __aarch64__
    for (; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

#if __aarch64__
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        float* g00 = kernel_tm.channel(q / 4 + q % 4);
#endif

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const float* k00 = k0.row(p + j);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4to1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            float32x4_t _val0 = vld1q_f32(sptr);
                            float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                            float32x4_t _val2 = vld1q_f32(sptr + stride_w * 8);
                            float32x4_t _val3 = vld1q_f32(sptr + stride_w * 12);
                            vst1q_f32(ptr, _val0);
                            vst1q_f32(ptr + 4, _val1);
                            vst1q_f32(ptr + 8, _val2);
                            vst1q_f32(ptr + 12, _val3);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            float32x4_t _val0 = vld1q_f32(sptr);
                            float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                            vst1q_f32(ptr, _val0);
                            vst1q_f32(ptr + 4, _val1);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }
                        for (; j < outw; j++)
                        {
                            float32x4_t _val = vld1q_f32(sptr);
                            vst1q_f32(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4to1_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
