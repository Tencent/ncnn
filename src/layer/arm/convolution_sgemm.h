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

static void im2col_sgemm_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
#if __ARM_NEON
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    vst1q_f32(tmpptr, vld1q_f32(img0));
                    vst1q_f32(tmpptr + 4, vld1q_f32(img0 + 4));
                    img0 += size;
                    tmpptr += 8;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    vst1q_f32(tmpptr, vld1q_f32(img0));
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }
#else // __ARM_NEON
    tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            float* tmpptr = tmp.channel(i);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }
#endif // __ARM_NEON

#if __ARM_NEON
    int nn_outch = 0;
    int remain_outch_start = 0;

#if __aarch64__
    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

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
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%20]   \n"
                "dup    v16.4s, v0.s[0]         \n"
                "dup    v17.4s, v0.s[0]         \n"
                "dup    v18.4s, v0.s[1]         \n"
                "dup    v19.4s, v0.s[1]         \n"
                "dup    v20.4s, v0.s[2]         \n"
                "dup    v21.4s, v0.s[2]         \n"
                "dup    v22.4s, v0.s[3]         \n"
                "dup    v23.4s, v0.s[3]         \n"
                "dup    v24.4s, v1.s[0]         \n"
                "dup    v25.4s, v1.s[0]         \n"
                "dup    v26.4s, v1.s[1]         \n"
                "dup    v27.4s, v1.s[1]         \n"
                "dup    v28.4s, v1.s[2]         \n"
                "dup    v29.4s, v1.s[2]         \n"
                "dup    v30.4s, v1.s[3]         \n"
                "dup    v31.4s, v1.s[3]         \n"

                // inch loop
                "lsr    w4, %w21, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%8, #512]   \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

                "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                "fmla   v22.4s, v8.4s, v0.s[3]  \n"

                "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                "fmla   v30.4s, v8.4s, v1.s[3]  \n"

                "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                "prfm   pldl1keep, [%8, #512]   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

                "fmla   v16.4s, v10.4s, v2.s[0] \n"
                "fmla   v18.4s, v10.4s, v2.s[1] \n"
                "fmla   v20.4s, v10.4s, v2.s[2] \n"
                "fmla   v22.4s, v10.4s, v2.s[3] \n"

                "fmla   v17.4s, v11.4s, v2.s[0] \n"
                "fmla   v19.4s, v11.4s, v2.s[1] \n"
                "fmla   v21.4s, v11.4s, v2.s[2] \n"
                "fmla   v23.4s, v11.4s, v2.s[3] \n"

                "fmla   v24.4s, v10.4s, v3.s[0] \n"
                "fmla   v26.4s, v10.4s, v3.s[1] \n"
                "fmla   v28.4s, v10.4s, v3.s[2] \n"
                "fmla   v30.4s, v10.4s, v3.s[3] \n"

                "fmla   v25.4s, v11.4s, v3.s[0] \n"
                "fmla   v27.4s, v11.4s, v3.s[1] \n"
                "fmla   v29.4s, v11.4s, v3.s[2] \n"
                "fmla   v31.4s, v11.4s, v3.s[3] \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

                "fmla   v16.4s, v12.4s, v4.s[0] \n"
                "fmla   v18.4s, v12.4s, v4.s[1] \n"
                "fmla   v20.4s, v12.4s, v4.s[2] \n"
                "fmla   v22.4s, v12.4s, v4.s[3] \n"

                "fmla   v17.4s, v13.4s, v4.s[0] \n"
                "fmla   v19.4s, v13.4s, v4.s[1] \n"
                "fmla   v21.4s, v13.4s, v4.s[2] \n"
                "fmla   v23.4s, v13.4s, v4.s[3] \n"

                "fmla   v24.4s, v12.4s, v5.s[0] \n"
                "fmla   v26.4s, v12.4s, v5.s[1] \n"
                "fmla   v28.4s, v12.4s, v5.s[2] \n"
                "fmla   v30.4s, v12.4s, v5.s[3] \n"

                "fmla   v25.4s, v13.4s, v5.s[0] \n"
                "fmla   v27.4s, v13.4s, v5.s[1] \n"
                "fmla   v29.4s, v13.4s, v5.s[2] \n"
                "fmla   v31.4s, v13.4s, v5.s[3] \n"

                "subs   w4, w4, #1              \n"

                "fmla   v16.4s, v14.4s, v6.s[0] \n"
                "fmla   v18.4s, v14.4s, v6.s[1] \n"
                "fmla   v20.4s, v14.4s, v6.s[2] \n"
                "fmla   v22.4s, v14.4s, v6.s[3] \n"

                "fmla   v17.4s, v15.4s, v6.s[0] \n"
                "fmla   v19.4s, v15.4s, v6.s[1] \n"
                "fmla   v21.4s, v15.4s, v6.s[2] \n"
                "fmla   v23.4s, v15.4s, v6.s[3] \n"

                "fmla   v24.4s, v14.4s, v7.s[0] \n"
                "fmla   v26.4s, v14.4s, v7.s[1] \n"
                "fmla   v28.4s, v14.4s, v7.s[2] \n"
                "fmla   v30.4s, v14.4s, v7.s[3] \n"

                "fmla   v25.4s, v15.4s, v7.s[0] \n"
                "fmla   v27.4s, v15.4s, v7.s[1] \n"
                "fmla   v29.4s, v15.4s, v7.s[2] \n"
                "fmla   v31.4s, v15.4s, v7.s[3] \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w21, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #256]   \n"
                "ld1    {v8.4s, v9.4s}, [%8], #32   \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                "fmla   v22.4s, v8.4s, v0.s[3]  \n"

                "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                "fmla   v30.4s, v8.4s, v1.s[3]  \n"

                "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                "st1    {v18.4s, v19.4s}, [%1], #32 \n"
                "st1    {v20.4s, v21.4s}, [%2], #32 \n"
                "st1    {v22.4s, v23.4s}, [%3], #32 \n"
                "st1    {v24.4s, v25.4s}, [%4], #32 \n"
                "st1    {v26.4s, v27.4s}, [%5], #32 \n"
                "st1    {v28.4s, v29.4s}, [%6], #32 \n"
                "st1    {v30.4s, v31.4s}, [%7], #32 \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(outptr4), // %4
                "=r"(outptr5), // %5
                "=r"(outptr6), // %6
                "=r"(outptr7), // %7
                "=r"(tmpptr),  // %8
                "=r"(kptr)     // %9
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(outptr4),
                "5"(outptr5),
                "6"(outptr6),
                "7"(outptr7),
                "8"(tmpptr),
                "9"(kptr),
                "r"(biasptr), // %20
                "r"(nn)       // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%20]   \n"
                "dup    v16.4s, v0.s[0]         \n"
                "dup    v17.4s, v0.s[1]         \n"
                "dup    v18.4s, v0.s[2]         \n"
                "dup    v19.4s, v0.s[3]         \n"
                "dup    v20.4s, v1.s[0]         \n"
                "dup    v21.4s, v1.s[1]         \n"
                "dup    v22.4s, v1.s[2]         \n"
                "dup    v23.4s, v1.s[3]         \n"

                // inch loop
                "lsr    w4, %w21, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%8, #512]   \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

                "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                "fmla   v19.4s, v8.4s, v0.s[3]  \n"
                "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

                "fmla   v16.4s, v9.4s, v2.s[0]  \n"
                "fmla   v17.4s, v9.4s, v2.s[1]  \n"
                "fmla   v18.4s, v9.4s, v2.s[2]  \n"
                "fmla   v19.4s, v9.4s, v2.s[3]  \n"
                "fmla   v20.4s, v9.4s, v3.s[0]  \n"
                "fmla   v21.4s, v9.4s, v3.s[1]  \n"
                "fmla   v22.4s, v9.4s, v3.s[2]  \n"
                "fmla   v23.4s, v9.4s, v3.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v16.4s, v10.4s, v4.s[0] \n"
                "fmla   v17.4s, v10.4s, v4.s[1] \n"
                "fmla   v18.4s, v10.4s, v4.s[2] \n"
                "fmla   v19.4s, v10.4s, v4.s[3] \n"
                "fmla   v20.4s, v10.4s, v5.s[0] \n"
                "fmla   v21.4s, v10.4s, v5.s[1] \n"
                "fmla   v22.4s, v10.4s, v5.s[2] \n"
                "fmla   v23.4s, v10.4s, v5.s[3] \n"

                "fmla   v16.4s, v11.4s, v6.s[0] \n"
                "fmla   v17.4s, v11.4s, v6.s[1] \n"
                "fmla   v18.4s, v11.4s, v6.s[2] \n"
                "fmla   v19.4s, v11.4s, v6.s[3] \n"
                "fmla   v20.4s, v11.4s, v7.s[0] \n"
                "fmla   v21.4s, v11.4s, v7.s[1] \n"
                "fmla   v22.4s, v11.4s, v7.s[2] \n"
                "fmla   v23.4s, v11.4s, v7.s[3] \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w21, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #128]   \n"
                "ld1    {v8.4s}, [%8], #16      \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                "fmla   v19.4s, v8.4s, v0.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v16.4s}, [%0], #16     \n"
                "st1    {v17.4s}, [%1], #16     \n"
                "st1    {v18.4s}, [%2], #16     \n"
                "st1    {v19.4s}, [%3], #16     \n"
                "st1    {v20.4s}, [%4], #16     \n"
                "st1    {v21.4s}, [%5], #16     \n"
                "st1    {v22.4s}, [%6], #16     \n"
                "st1    {v23.4s}, [%7], #16     \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(outptr4), // %4
                "=r"(outptr5), // %5
                "=r"(outptr6), // %6
                "=r"(outptr7), // %7
                "=r"(tmpptr),  // %8
                "=r"(kptr)     // %9
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(outptr4),
                "5"(outptr5),
                "6"(outptr6),
                "7"(outptr7),
                "8"(tmpptr),
                "9"(kptr),
                "r"(biasptr), // %20
                "r"(nn)       // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            asm volatile(
                "ld1    {v24.4s, v25.4s}, [%20] \n"

                // inch loop
                "lsr    w4, %w21, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "eor    v16.16b, v16.16b, v16.16b  \n"
                "eor    v17.16b, v17.16b, v17.16b  \n"
                "eor    v18.16b, v18.16b, v18.16b  \n"
                "eor    v19.16b, v19.16b, v19.16b  \n"
                "eor    v20.16b, v20.16b, v20.16b  \n"
                "eor    v21.16b, v21.16b, v21.16b  \n"
                "eor    v22.16b, v22.16b, v22.16b  \n"
                "eor    v23.16b, v23.16b, v23.16b  \n"

                "0:                             \n"

                "prfm   pldl1keep, [%8, #128]   \n"
                "ld1    {v8.4s}, [%8], #16      \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

                "fmla   v16.4s, v0.4s, v8.s[0]  \n"
                "fmla   v17.4s, v1.4s, v8.s[0]  \n"
                "fmla   v18.4s, v2.4s, v8.s[1]  \n"
                "fmla   v19.4s, v3.4s, v8.s[1]  \n"

                "prfm   pldl1keep, [%9, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

                "subs   w4, w4, #1              \n"

                "fmla   v20.4s, v4.4s, v8.s[2]  \n"
                "fmla   v21.4s, v5.4s, v8.s[2]  \n"
                "fmla   v22.4s, v6.4s, v8.s[3]  \n"
                "fmla   v23.4s, v7.4s, v8.s[3]  \n"

                "bne    0b                      \n"

                "fadd   v16.4s, v16.4s, v18.4s  \n"
                "fadd   v17.4s, v17.4s, v19.4s  \n"
                "fadd   v20.4s, v20.4s, v22.4s  \n"
                "fadd   v21.4s, v21.4s, v23.4s  \n"
                "fadd   v16.4s, v16.4s, v20.4s  \n"
                "fadd   v17.4s, v17.4s, v21.4s  \n"
                "fadd   v24.4s, v24.4s, v16.4s  \n"
                "fadd   v25.4s, v25.4s, v17.4s  \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w21, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #32]    \n"
                "ld1r   {v8.4s}, [%8], #4       \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                "subs   w4, w4, #1              \n"

                "fmla   v24.4s, v8.4s, v0.4s    \n"
                "fmla   v25.4s, v8.4s, v1.4s    \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v24.s}[0],[%0], #4     \n"
                "st1    {v24.s}[1],[%1], #4     \n"
                "st1    {v24.s}[2],[%2], #4     \n"
                "st1    {v24.s}[3],[%3], #4     \n"
                "st1    {v25.s}[0],[%4], #4     \n"
                "st1    {v25.s}[1],[%5], #4     \n"
                "st1    {v25.s}[2],[%6], #4     \n"
                "st1    {v25.s}[3],[%7], #4     \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(outptr4), // %4
                "=r"(outptr5), // %5
                "=r"(outptr6), // %6
                "=r"(outptr7), // %7
                "=r"(tmpptr),  // %8
                "=r"(kptr)     // %9
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(outptr4),
                "5"(outptr5),
                "6"(outptr6),
                "7"(outptr7),
                "8"(tmpptr),
                "9"(kptr),
                "r"(biasptr), // %20
                "r"(nn)       // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
        }
    }
#endif // __aarch64__

    nn_outch = (outch - remain_outch_start) >> 2;

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
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const float* kptr = kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%12]          \n"
                "dup    v8.4s, v0.s[0]          \n"
                "dup    v9.4s, v0.s[0]          \n"
                "dup    v10.4s, v0.s[1]         \n"
                "dup    v11.4s, v0.s[1]         \n"
                "dup    v12.4s, v0.s[2]         \n"
                "dup    v13.4s, v0.s[2]         \n"
                "dup    v14.4s, v0.s[3]         \n"
                "dup    v15.4s, v0.s[3]         \n"

                // inch loop
                "lsr    w4, %w13, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                "fmla   v14.4s, v4.4s, v0.s[3]  \n"

                "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                "prfm   pldl1keep, [%4, #512]   \n"
                "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                "fmla   v8.4s, v6.4s, v1.s[0]   \n"
                "fmla   v10.4s, v6.4s, v1.s[1]  \n"
                "fmla   v12.4s, v6.4s, v1.s[2]  \n"
                "fmla   v14.4s, v6.4s, v1.s[3]  \n"

                "fmla   v9.4s, v7.4s, v1.s[0]   \n"
                "fmla   v11.4s, v7.4s, v1.s[1]  \n"
                "fmla   v13.4s, v7.4s, v1.s[2]  \n"
                "fmla   v15.4s, v7.4s, v1.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v16.4s, v2.s[0]  \n"
                "fmla   v10.4s, v16.4s, v2.s[1] \n"
                "fmla   v12.4s, v16.4s, v2.s[2] \n"
                "fmla   v14.4s, v16.4s, v2.s[3] \n"

                "fmla   v9.4s, v17.4s, v2.s[0]  \n"
                "fmla   v11.4s, v17.4s, v2.s[1] \n"
                "fmla   v13.4s, v17.4s, v2.s[2] \n"
                "fmla   v15.4s, v17.4s, v2.s[3] \n"

                "fmla   v8.4s, v18.4s, v3.s[0]  \n"
                "fmla   v10.4s, v18.4s, v3.s[1] \n"
                "fmla   v12.4s, v18.4s, v3.s[2] \n"
                "fmla   v14.4s, v18.4s, v3.s[3] \n"

                "fmla   v9.4s, v19.4s, v3.s[0]  \n"
                "fmla   v11.4s, v19.4s, v3.s[1] \n"
                "fmla   v13.4s, v19.4s, v3.s[2] \n"
                "fmla   v15.4s, v19.4s, v3.s[3] \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w13, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #256]   \n"
                "ld1    {v4.4s, v5.4s}, [%4], #32   \n"

                "prfm   pldl1keep, [%5, #128]   \n"
                "ld1    {v0.4s}, [%5], #16      \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                "fmla   v14.4s, v4.4s, v0.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v8.4s, v9.4s}, [%0], #32   \n"
                "st1    {v10.4s, v11.4s}, [%1], #32 \n"
                "st1    {v12.4s, v13.4s}, [%2], #32 \n"
                "st1    {v14.4s, v15.4s}, [%3], #32 \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d0-d1}, [%12]      \n"
                "vdup.f32   q8, d0[0]           \n"
                "vdup.f32   q9, d0[0]           \n"
                "vdup.f32   q10, d0[1]          \n"
                "vdup.f32   q11, d0[1]          \n"
                "vdup.f32   q12, d1[0]          \n"
                "vdup.f32   q13, d1[0]          \n"
                "vdup.f32   q14, d1[1]          \n"
                "vdup.f32   q15, d1[1]          \n"

                // inch loop
                "lsr        r4, %13, #2         \n" // r4 = nn >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%4, #512]          \n"
                "vldm       %4!, {d8-d15}       \n"
                //                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
                //                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

                "pld        [%5, #512]          \n"
                "vldm       %5!, {d0-d7}       \n"
                //                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
                //                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q10, q4, d0[1]      \n"
                "vmla.f32   q12, q4, d1[0]      \n"
                "vmla.f32   q14, q4, d1[1]      \n"

                "vmla.f32   q9, q5, d0[0]       \n"
                "vmla.f32   q11, q5, d0[1]      \n"
                "vmla.f32   q13, q5, d1[0]      \n"
                "vmla.f32   q15, q5, d1[1]      \n"

                "vmla.f32   q8, q6, d2[0]       \n"
                "vmla.f32   q10, q6, d2[1]      \n"
                "vmla.f32   q12, q6, d3[0]      \n"
                "vmla.f32   q14, q6, d3[1]      \n"

                "vmla.f32   q9, q7, d2[0]       \n"
                "vmla.f32   q11, q7, d2[1]      \n"
                "vmla.f32   q13, q7, d3[0]      \n"
                "vmla.f32   q15, q7, d3[1]      \n"

                "pld        [%4, #512]          \n"
                "vldm       %4!, {d8-d15}       \n"
                //                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
                //                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

                "vmla.f32   q8, q4, d4[0]       \n"
                "vmla.f32   q10, q4, d4[1]      \n"
                "vmla.f32   q12, q4, d5[0]      \n"
                "vmla.f32   q14, q4, d5[1]      \n"

                "vmla.f32   q9, q5, d4[0]       \n"
                "vmla.f32   q11, q5, d4[1]      \n"
                "vmla.f32   q13, q5, d5[0]      \n"
                "vmla.f32   q15, q5, d5[1]      \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q6, d6[0]       \n"
                "vmla.f32   q10, q6, d6[1]      \n"
                "vmla.f32   q12, q6, d7[0]      \n"
                "vmla.f32   q14, q6, d7[1]      \n"

                "vmla.f32   q9, q7, d6[0]       \n"
                "vmla.f32   q11, q7, d6[1]      \n"
                "vmla.f32   q13, q7, d7[0]      \n"
                "vmla.f32   q15, q7, d7[1]      \n"

                "bne        0b                  \n"

                "1:                             \n"

                // remain loop
                "and        r4, %13, #3         \n" // r4 = remain = nn & 3
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #256]          \n"
                "vld1.f32   {d8-d11}, [%4 :128]!    \n"

                "pld        [%5, #128]          \n"
                "vld1.f32   {d0-d1}, [%5 :128]!     \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q10, q4, d0[1]      \n"
                "vmla.f32   q12, q4, d1[0]      \n"
                "vmla.f32   q14, q4, d1[1]      \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q9, q5, d0[0]       \n"
                "vmla.f32   q11, q5, d0[1]      \n"
                "vmla.f32   q13, q5, d1[0]      \n"
                "vmla.f32   q15, q5, d1[1]      \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vst1.f32   {d16-d19}, [%0 :128]!   \n"
                "vst1.f32   {d20-d23}, [%1 :128]!   \n"
                "vst1.f32   {d24-d27}, [%2 :128]!   \n"
                "vst1.f32   {d28-d31}, [%3 :128]!   \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const float* kptr = kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%12]          \n"
                "dup    v8.4s, v0.s[0]          \n"
                "dup    v9.4s, v0.s[1]          \n"
                "dup    v10.4s, v0.s[2]         \n"
                "dup    v11.4s, v0.s[3]         \n"

                // inch loop
                "lsr    w4, %w13, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                "fmla   v8.4s, v5.4s, v1.s[0]   \n"
                "fmla   v9.4s, v5.4s, v1.s[1]   \n"
                "fmla   v10.4s, v5.4s, v1.s[2]  \n"
                "fmla   v11.4s, v5.4s, v1.s[3]  \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v6.4s, v2.s[0]   \n"
                "fmla   v9.4s, v6.4s, v2.s[1]   \n"
                "fmla   v10.4s, v6.4s, v2.s[2]  \n"
                "fmla   v11.4s, v6.4s, v2.s[3]  \n"

                "fmla   v8.4s, v7.4s, v3.s[0]   \n"
                "fmla   v9.4s, v7.4s, v3.s[1]   \n"
                "fmla   v10.4s, v7.4s, v3.s[2]  \n"
                "fmla   v11.4s, v7.4s, v3.s[3]  \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w13, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #128]   \n"
                "ld1    {v4.4s}, [%4], #16      \n"

                "prfm   pldl1keep, [%5, #128]   \n"
                "ld1    {v0.4s}, [%5], #16      \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v8.4s}, [%0], #16      \n"
                "st1    {v9.4s}, [%1], #16      \n"
                "st1    {v10.4s}, [%2], #16     \n"
                "st1    {v11.4s}, [%3], #16     \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d0-d1}, [%12]      \n"
                "vdup.f32   q8, d0[0]           \n"
                "vdup.f32   q9, d0[1]           \n"
                "vdup.f32   q10, d1[0]          \n"
                "vdup.f32   q11, d1[1]          \n"

                // inch loop
                "lsr        r4, %13, #2         \n" // r4 = nn >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%4, #512]          \n"
                "vldm       %4!, {d8-d15}       \n"
                //                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
                //                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

                "pld        [%5, #512]          \n"
                "vldm       %5!, {d0-d7}       \n"
                //                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
                //                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"

                "vmla.f32   q8, q5, d2[0]       \n"
                "vmla.f32   q9, q5, d2[1]       \n"
                "vmla.f32   q10, q5, d3[0]      \n"
                "vmla.f32   q11, q5, d3[1]      \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q6, d4[0]       \n"
                "vmla.f32   q9, q6, d4[1]       \n"
                "vmla.f32   q10, q6, d5[0]      \n"
                "vmla.f32   q11, q6, d5[1]      \n"

                "vmla.f32   q8, q7, d6[0]       \n"
                "vmla.f32   q9, q7, d6[1]       \n"
                "vmla.f32   q10, q7, d7[0]      \n"
                "vmla.f32   q11, q7, d7[1]      \n"

                "bne        0b                  \n"

                "1:                             \n"

                // remain loop
                "and        r4, %13, #3         \n" // r4 = remain = nn & 3
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #128]          \n"
                "vld1.f32   {d8-d9}, [%4 :128]! \n"

                "pld        [%5, #128]          \n"
                "vld1.f32   {d0-d1}, [%5 :128]! \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vst1.f32   {d16-d17}, [%0 :128]!   \n"
                "vst1.f32   {d18-d19}, [%1 :128]!   \n"
                "vst1.f32   {d20-d21}, [%2 :128]!   \n"
                "vst1.f32   {d22-d23}, [%3 :128]!   \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const float* kptr = kernel.channel(p / 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v12.4s}, [%12]         \n"

                // inch loop
                "lsr    w4, %w13, #2            \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "eor    v8.16b, v8.16b, v8.16b  \n"
                "eor    v9.16b, v9.16b, v9.16b  \n"
                "eor    v10.16b, v10.16b, v10.16b  \n"
                "eor    v11.16b, v11.16b, v11.16b  \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #128]   \n"
                "ld1    {v4.4s}, [%4], #16      \n"

                "prfm   pldl1keep, [%5, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                "fmla   v9.4s, v1.4s, v4.s[1]   \n"
                "fmla   v10.4s, v2.4s, v4.s[2]  \n"
                "fmla   v11.4s, v3.4s, v4.s[3]  \n"

                "bne    0b                      \n"

                "fadd   v8.4s, v8.4s, v9.4s     \n"
                "fadd   v10.4s, v10.4s, v11.4s  \n"
                "fadd   v8.4s, v8.4s, v10.4s    \n"
                "fadd   v12.4s, v12.4s, v8.4s   \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w13, #3            \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #32]    \n"
                "ld1r   {v4.4s}, [%4], #4       \n"

                "prfm   pldl1keep, [%5, #128]   \n"
                "ld1    {v0.4s}, [%5], #16      \n"

                "subs   w4, w4, #1              \n"

                "fmla   v12.4s, v4.4s, v0.4s    \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v12.s}[0], [%0], #4    \n"
                "st1    {v12.s}[1], [%1], #4    \n"
                "st1    {v12.s}[2], [%2], #4    \n"
                "st1    {v12.s}[3], [%3], #4    \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d24-d25}, [%12]    \n"

                // inch loop
                "lsr        r4, %13, #2         \n" // r4 = nn >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "veor       q8, q8, q8          \n"
                "veor       q9, q9, q9          \n"
                "veor       q10, q10, q10       \n"
                "veor       q11, q11, q11       \n"

                "0:                             \n"

                "pld        [%4, #128]          \n"
                "vld1.f32   {d8-d9}, [%4 :128]! \n"

                "pld        [%5, #512]          \n"
                "vldm       %5!, {d0-d7}       \n"
                //                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
                //                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q0, d8[0]       \n"
                "vmla.f32   q9, q1, d8[1]       \n"
                "vmla.f32   q10, q2, d9[0]      \n"
                "vmla.f32   q11, q3, d9[1]      \n"

                "bne        0b                  \n"

                "vadd.f32   q8, q8, q9          \n"
                "vadd.f32   q10, q10, q11       \n"
                "vadd.f32   q8, q8, q10         \n"
                "vadd.f32   q12, q12, q8        \n"

                "1:                             \n"

                // remain loop
                "and        r4, %13, #3         \n" // r4 = remain = nn & 3
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #32]           \n"
                "vld1.f32   {d8[],d9[]}, [%4]!  \n"

                "pld        [%5, #128]          \n"
                "vld1.f32   {d0-d1}, [%5 :128]! \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q12, q4, q0         \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vst1.f32   {d24[0]}, [%0]!     \n"
                "vst1.f32   {d24[1]}, [%1]!     \n"
                "vst1.f32   {d25[0]}, [%2]!     \n"
                "vst1.f32   {d25[1]}, [%3]!     \n"

                : "=r"(outptr0), // %0
                "=r"(outptr1), // %1
                "=r"(outptr2), // %2
                "=r"(outptr3), // %3
                "=r"(tmpptr),  // %4
                "=r"(kptr)     // %5
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(tmpptr),
                "5"(kptr),
                "r"(biasptr), // %12
                "r"(nn)       // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12");
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
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 8);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const float* kptr = kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w6              \n"
                "dup    v9.4s, %w6              \n"

                // inch loop
                "lsr    w4, %w7, #2             \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%1, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v0.4s}, [%2], #16      \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v5.4s, v0.s[0]   \n"

                "prfm   pldl1keep, [%1, #512]   \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

                "fmla   v8.4s, v6.4s, v0.s[1]   \n"
                "fmla   v9.4s, v7.4s, v0.s[1]   \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v12.4s, v0.s[2]  \n"
                "fmla   v9.4s, v13.4s, v0.s[2]  \n"

                "fmla   v8.4s, v14.4s, v0.s[3]  \n"
                "fmla   v9.4s, v15.4s, v0.s[3]  \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w7, #3             \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%1, #256]   \n"
                "ld1    {v4.4s, v5.4s}, [%1], #32   \n"

                "prfm   pldl1keep, [%2, #32]    \n"
                "ld1r   {v0.4s}, [%2], #4       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.4s     \n"
                "fmla   v9.4s, v5.4s, v0.4s     \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v8.4s, v9.4s}, [%0], #32   \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(nn)     // %7
                : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %6              \n"
                "vdup.f32   q9, %6              \n"

                // inch loop
                "lsr        r4, %7, #2          \n" // r4 = nn >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%1, #512]          \n"
                "vldm       %1!, {d8-d15}       \n"
                //                 "vld1.f32   {d8-d11}, [%1 :128]!    \n"
                //                 "vld1.f32   {d12-d15}, [%1 :128]!   \n"

                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%2 :128]! \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q5, d0[0]       \n"

                "pld        [%1, #512]          \n"
                "vldm       %1!, {d24-d31}      \n"
                //                 "vld1.f32   {d24-d27}, [%1 :128]!   \n"
                //                 "vld1.f32   {d28-d31}, [%1 :128]!   \n"

                "vmla.f32   q8, q6, d0[1]       \n"
                "vmla.f32   q9, q7, d0[1]       \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q12, d1[0]      \n"
                "vmla.f32   q9, q13, d1[0]      \n"

                "vmla.f32   q8, q14, d1[1]      \n"
                "vmla.f32   q9, q15, d1[1]      \n"

                "bne        0b                  \n"

                "1:                             \n"

                // remain loop
                "and        r4, %7, #3          \n" // r4 = remain = nn & 3
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%1, #256]          \n"
                "vld1.f32   {d8-d11}, [%1 :128]!    \n"

                "pld        [%2, #32]           \n"
                "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, q0          \n"
                "vmla.f32   q9, q5, q0          \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vst1.f32   {d16-d19}, [%0 :128]!   \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(nn)     // %7
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const float* kptr = kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w6              \n"

                // inch loop
                "lsr    w4, %w7, #2             \n" // w4 = nn >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%1, #512]   \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

                "prfm   pldl1keep, [%2, #128]   \n"
                "ld1    {v0.4s}, [%2], #16      \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v8.4s, v5.4s, v0.s[1]   \n"
                "fmla   v8.4s, v6.4s, v0.s[2]   \n"
                "fmla   v8.4s, v7.4s, v0.s[3]   \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w7, #3             \n" // w4 = remain = nn & 3
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v4.4s}, [%1], #16      \n"

                "prfm   pldl1keep, [%2, #32]    \n"
                "ld1r   {v0.4s}, [%2], #4       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.4s     \n"

                "bne    2b                      \n"

                "3:                             \n"

                "st1    {v8.4s}, [%0], #16      \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(nn)     // %7
                : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %6              \n"

                // inch loop
                "lsr        r4, %7, #2          \n" // r4 = nn >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%1, #512]          \n"
                "vldm       %1!, {d8-d15}       \n"
                //                 "vld1.f32   {d8-d11}, [%1 :128]!    \n"
                //                 "vld1.f32   {d12-d15}, [%1 :128]!   \n"

                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%2]!      \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q8, q5, d0[1]       \n"
                "vmla.f32   q8, q6, d1[0]       \n"
                "vmla.f32   q8, q7, d1[1]       \n"

                "bne        0b                  \n"

                "1:                             \n"

                // remain loop
                "and        r4, %7, #3          \n" // r4 = remain = nn & 3
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%1, #128]          \n"
                "vld1.f32   {d8-d9}, [%1 :128]! \n"

                "pld        [%2, #32]           \n"
                "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, q0          \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vst1.f32   {d16-d17}, [%0 :128]!   \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(nn)     // %7
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8");
#endif // __aarch64__
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __aarch64__
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const float* kptr = kernel.channel(p / 4 + p % 4);
#endif

            int nn = inch * maxk; // inch always > 0

            float32x4_t _sum0 = vdupq_n_f32(0.f);

            int q = 0;
            for (; q + 3 < nn; q += 4)
            {
                float32x4_t _p0 = vld1q_f32(tmpptr);
                tmpptr += 4;

                float32x4_t _k0 = vld1q_f32(kptr);
                kptr += 4;

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _p0, _k0);
#else
                _sum0 = vmlaq_f32(_sum0, _p0, _k0);
#endif
            }

#if __aarch64__
            float sum0 = bias0 + vaddvq_f32(_sum0);
#else
            float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
            float sum0 = bias0 + vget_lane_f32(vpadd_f32(_ss, _ss), 0);
#endif

            for (; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
#else // __ARM_NEON
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        for (int i = 0; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i);
            const float* kptr = kernel.channel(p);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
#endif // __ARM_NEON
}

static void convolution_im2col_sgemm_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __ARM_NEON
#if __aarch64__
    kernel_tm.create(32 * maxk, inch / 4 + inch % 4, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4);
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

        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);
            const float* k40 = k4.row(p);
            const float* k50 = k5.row(p);
            const float* k60 = k6.row(p);
            const float* k70 = k7.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00 += 8;
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
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        float* g00 = kernel_tm.channel(q / 4);
#endif

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
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

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];

                g00 += 1;
            }
        }
    }
#else
    kernel_tm = kernel;
#endif // __ARM_NEON
}

static void convolution_im2col_sgemm_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];
                            ptr[2] = sptr[stride_w * 2];
                            ptr[3] = sptr[stride_w * 3];

                            sptr += stride_w * 4;
                            ptr += 4;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            ptr[0] = sptr[0];
                            ptr[1] = sptr[stride_w];

                            sptr += stride_w * 2;
                            ptr += 2;
                        }
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
