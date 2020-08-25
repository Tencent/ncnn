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

static void conv1x1s1_sgemm_transform_kernel_pack8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 8b-8a-inch/8a-outch/8b
    kernel_tm_pack8.create(1, inch / 8, outch / 8, (size_t)2u * 64, 64);

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

        __fp16* g0 = kernel_tm_pack8.channel(q / 8);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int i = 0; i < 8; i++)
            {
                g0[0] = (__fp16)k0[i];
                g0[1] = (__fp16)k1[i];
                g0[2] = (__fp16)k2[i];
                g0[3] = (__fp16)k3[i];
                g0[4] = (__fp16)k4[i];
                g0[5] = (__fp16)k5[i];
                g0[6] = (__fp16)k6[i];
                g0[7] = (__fp16)k7[i];

                g0 += 8;
            }

            k0 += 8;
            k1 += 8;
            k2 += 8;
            k3 += 8;
            k4 += 8;
            k5 += 8;
            k6 += 8;
            k7 += 8;
        }
    }
}

static void conv1x1s1_sgemm_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
    if (size >= 12)
        tmp.create(12, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4, inch, size / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2, inch, size / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else // if (size >= 1)
        tmp.create(1, inch, size, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size;
        int remain_size_start;

        nn_size = size / 12;
        remain_size_start = nn_size * 12;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 12;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 12);

            for (int q = 0; q < inch; q++)
            {
                // transpose 12x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
                    "ld4    {v4.8h, v5.8h, v6.8h, v7.8h}, [%0], #64 \n"
                    "ld4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%0] \n"

                    "sub    %0, %0, #128            \n"

                    "uzp1   v20.8h, v0.8h, v4.8h    \n" // 0
                    "uzp1   v21.8h, v16.8h, v1.8h   \n" // 1
                    "uzp1   v22.8h, v5.8h, v17.8h   \n" // 2
                    "uzp1   v23.8h, v2.8h, v6.8h    \n" // 3
                    "uzp1   v24.8h, v18.8h, v3.8h   \n" // 4
                    "uzp1   v25.8h, v7.8h, v19.8h   \n" // 5
                    "uzp2   v26.8h, v0.8h, v4.8h    \n" // 6
                    "uzp2   v27.8h, v16.8h, v1.8h   \n" // 7
                    "uzp2   v28.8h, v5.8h, v17.8h   \n" // 8
                    "uzp2   v29.8h, v2.8h, v6.8h    \n" // 9
                    "uzp2   v30.8h, v18.8h, v3.8h   \n" // 10
                    "uzp2   v31.8h, v7.8h, v19.8h   \n" // 11

                    "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%1], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                img0 += bottom_blob.cstep * 8;
            }
        }

        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);

            for (int q = 0; q < inch; q++)
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

                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");

                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]   \n"
                    "ld1    {v0.8h, v1.8h}, [%0]    \n"
                    "st1    {v0.8h, v1.8h}, [%1], #32 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1");

                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

            for (int q = 0; q < inch; q++)
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

                img0 += bottom_blob.cstep * 8;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        const __fp16 zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 8 : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            __fp16* tmpptr = tmp.channel(i / 12);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v20.8h}, [%8]              \n"
                "mov    v21.16b, v20.16b            \n"
                "mov    v22.16b, v20.16b            \n"
                "mov    v23.16b, v20.16b            \n"
                "mov    v24.16b, v20.16b            \n"
                "mov    v25.16b, v20.16b            \n"
                "mov    v26.16b, v20.16b            \n"
                "mov    v27.16b, v20.16b            \n"
                "mov    v28.16b, v20.16b            \n"
                "mov    v29.16b, v20.16b            \n"
                "mov    v30.16b, v20.16b            \n"
                "mov    v31.16b, v20.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w0123

                "fmla   v20.8h, v12.8h, v0.h[0]     \n"
                "fmla   v21.8h, v12.8h, v0.h[1]     \n"
                "fmla   v22.8h, v12.8h, v0.h[2]     \n"
                "fmla   v23.8h, v12.8h, v0.h[3]     \n"
                "fmla   v24.8h, v12.8h, v0.h[4]     \n"
                "fmla   v25.8h, v12.8h, v0.h[5]     \n"
                "fmla   v26.8h, v12.8h, v0.h[6]     \n"
                "fmla   v27.8h, v12.8h, v0.h[7]     \n"
                "fmla   v28.8h, v12.8h, v1.h[0]     \n"
                "fmla   v29.8h, v12.8h, v1.h[1]     \n"
                "fmla   v30.8h, v12.8h, v1.h[2]     \n"
                "fmla   v31.8h, v12.8h, v1.h[3]     \n"

                "fmla   v20.8h, v13.8h, v1.h[4]     \n"
                "fmla   v21.8h, v13.8h, v1.h[5]     \n"
                "fmla   v22.8h, v13.8h, v1.h[6]     \n"
                "fmla   v23.8h, v13.8h, v1.h[7]     \n"
                "fmla   v24.8h, v13.8h, v2.h[0]     \n"
                "fmla   v25.8h, v13.8h, v2.h[1]     \n"
                "fmla   v26.8h, v13.8h, v2.h[2]     \n"
                "fmla   v27.8h, v13.8h, v2.h[3]     \n"
                "fmla   v28.8h, v13.8h, v2.h[4]     \n"
                "fmla   v29.8h, v13.8h, v2.h[5]     \n"
                "fmla   v30.8h, v13.8h, v2.h[6]     \n"
                "fmla   v31.8h, v13.8h, v2.h[7]     \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r4567

                "fmla   v20.8h, v14.8h, v3.h[0]     \n"
                "fmla   v21.8h, v14.8h, v3.h[1]     \n"
                "fmla   v22.8h, v14.8h, v3.h[2]     \n"
                "fmla   v23.8h, v14.8h, v3.h[3]     \n"
                "fmla   v24.8h, v14.8h, v3.h[4]     \n"
                "fmla   v25.8h, v14.8h, v3.h[5]     \n"
                "fmla   v26.8h, v14.8h, v3.h[6]     \n"
                "fmla   v27.8h, v14.8h, v3.h[7]     \n"
                "fmla   v28.8h, v14.8h, v4.h[0]     \n"
                "fmla   v29.8h, v14.8h, v4.h[1]     \n"
                "fmla   v30.8h, v14.8h, v4.h[2]     \n"
                "fmla   v31.8h, v14.8h, v4.h[3]     \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3], #64 \n" // w4567

                "fmla   v20.8h, v15.8h, v4.h[4]     \n"
                "fmla   v21.8h, v15.8h, v4.h[5]     \n"
                "fmla   v22.8h, v15.8h, v4.h[6]     \n"
                "fmla   v23.8h, v15.8h, v4.h[7]     \n"
                "fmla   v24.8h, v15.8h, v5.h[0]     \n"
                "fmla   v25.8h, v15.8h, v5.h[1]     \n"
                "fmla   v26.8h, v15.8h, v5.h[2]     \n"
                "fmla   v27.8h, v15.8h, v5.h[3]     \n"
                "fmla   v28.8h, v15.8h, v5.h[4]     \n"
                "fmla   v29.8h, v15.8h, v5.h[5]     \n"
                "fmla   v30.8h, v15.8h, v5.h[6]     \n"
                "fmla   v31.8h, v15.8h, v5.h[7]     \n"

                "fmla   v20.8h, v16.8h, v6.h[0]     \n"
                "fmla   v21.8h, v16.8h, v6.h[1]     \n"
                "fmla   v22.8h, v16.8h, v6.h[2]     \n"
                "fmla   v23.8h, v16.8h, v6.h[3]     \n"
                "fmla   v24.8h, v16.8h, v6.h[4]     \n"
                "fmla   v25.8h, v16.8h, v6.h[5]     \n"
                "fmla   v26.8h, v16.8h, v6.h[6]     \n"
                "fmla   v27.8h, v16.8h, v6.h[7]     \n"
                "fmla   v28.8h, v16.8h, v7.h[0]     \n"
                "fmla   v29.8h, v16.8h, v7.h[1]     \n"
                "fmla   v30.8h, v16.8h, v7.h[2]     \n"
                "fmla   v31.8h, v16.8h, v7.h[3]     \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r891011

                "fmla   v20.8h, v17.8h, v7.h[4]     \n"
                "fmla   v21.8h, v17.8h, v7.h[5]     \n"
                "fmla   v22.8h, v17.8h, v7.h[6]     \n"
                "fmla   v23.8h, v17.8h, v7.h[7]     \n"
                "fmla   v24.8h, v17.8h, v8.h[0]     \n"
                "fmla   v25.8h, v17.8h, v8.h[1]     \n"
                "fmla   v26.8h, v17.8h, v8.h[2]     \n"
                "fmla   v27.8h, v17.8h, v8.h[3]     \n"
                "fmla   v28.8h, v17.8h, v8.h[4]     \n"
                "fmla   v29.8h, v17.8h, v8.h[5]     \n"
                "fmla   v30.8h, v17.8h, v8.h[6]     \n"
                "fmla   v31.8h, v17.8h, v8.h[7]     \n"

                "fmla   v20.8h, v18.8h, v9.h[0]     \n"
                "fmla   v21.8h, v18.8h, v9.h[1]     \n"
                "fmla   v22.8h, v18.8h, v9.h[2]     \n"
                "fmla   v23.8h, v18.8h, v9.h[3]     \n"
                "fmla   v24.8h, v18.8h, v9.h[4]     \n"
                "fmla   v25.8h, v18.8h, v9.h[5]     \n"
                "fmla   v26.8h, v18.8h, v9.h[6]     \n"
                "fmla   v27.8h, v18.8h, v9.h[7]     \n"
                "fmla   v28.8h, v18.8h, v10.h[0]    \n"
                "fmla   v29.8h, v18.8h, v10.h[1]    \n"
                "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                "fmla   v31.8h, v18.8h, v10.h[3]    \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v20.8h, v19.8h, v10.h[4]    \n"
                "fmla   v21.8h, v19.8h, v10.h[5]    \n"
                "fmla   v22.8h, v19.8h, v10.h[6]    \n"
                "fmla   v23.8h, v19.8h, v10.h[7]    \n"
                "fmla   v24.8h, v19.8h, v11.h[0]    \n"
                "fmla   v25.8h, v19.8h, v11.h[1]    \n"
                "fmla   v26.8h, v19.8h, v11.h[2]    \n"
                "fmla   v27.8h, v19.8h, v11.h[3]    \n"
                "fmla   v28.8h, v19.8h, v11.h[4]    \n"
                "fmla   v29.8h, v19.8h, v11.h[5]    \n"
                "fmla   v30.8h, v19.8h, v11.h[6]    \n"
                "fmla   v31.8h, v19.8h, v11.h[7]    \n"

                "bne    0b                          \n"

                "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"
                "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%1], #64 \n"
                "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 7 < size; i += 8)
        {
            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v16.8h}, [%8]              \n"
                "mov    v17.16b, v16.16b            \n"
                "mov    v18.16b, v16.16b            \n"
                "mov    v19.16b, v16.16b            \n"
                "mov    v20.16b, v16.16b            \n"
                "mov    v21.16b, v16.16b            \n"
                "mov    v22.16b, v16.16b            \n"
                "mov    v23.16b, v16.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                "fmla   v17.8h, v8.8h, v0.h[1]      \n"
                "fmla   v18.8h, v8.8h, v0.h[2]      \n"
                "fmla   v19.8h, v8.8h, v0.h[3]      \n"
                "fmla   v20.8h, v8.8h, v0.h[4]      \n"
                "fmla   v21.8h, v8.8h, v0.h[5]      \n"
                "fmla   v22.8h, v8.8h, v0.h[6]      \n"
                "fmla   v23.8h, v8.8h, v0.h[7]      \n"

                "fmla   v16.8h, v9.8h, v1.h[0]      \n"
                "fmla   v17.8h, v9.8h, v1.h[1]      \n"
                "fmla   v18.8h, v9.8h, v1.h[2]      \n"
                "fmla   v19.8h, v9.8h, v1.h[3]      \n"
                "fmla   v20.8h, v9.8h, v1.h[4]      \n"
                "fmla   v21.8h, v9.8h, v1.h[5]      \n"
                "fmla   v22.8h, v9.8h, v1.h[6]      \n"
                "fmla   v23.8h, v9.8h, v1.h[7]      \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r4567

                "fmla   v16.8h, v10.8h, v2.h[0]     \n"
                "fmla   v17.8h, v10.8h, v2.h[1]     \n"
                "fmla   v18.8h, v10.8h, v2.h[2]     \n"
                "fmla   v19.8h, v10.8h, v2.h[3]     \n"
                "fmla   v20.8h, v10.8h, v2.h[4]     \n"
                "fmla   v21.8h, v10.8h, v2.h[5]     \n"
                "fmla   v22.8h, v10.8h, v2.h[6]     \n"
                "fmla   v23.8h, v10.8h, v2.h[7]     \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                "fmla   v16.8h, v11.8h, v3.h[0]     \n"
                "fmla   v17.8h, v11.8h, v3.h[1]     \n"
                "fmla   v18.8h, v11.8h, v3.h[2]     \n"
                "fmla   v19.8h, v11.8h, v3.h[3]     \n"
                "fmla   v20.8h, v11.8h, v3.h[4]     \n"
                "fmla   v21.8h, v11.8h, v3.h[5]     \n"
                "fmla   v22.8h, v11.8h, v3.h[6]     \n"
                "fmla   v23.8h, v11.8h, v3.h[7]     \n"

                "fmla   v16.8h, v12.8h, v4.h[0]     \n"
                "fmla   v17.8h, v12.8h, v4.h[1]     \n"
                "fmla   v18.8h, v12.8h, v4.h[2]     \n"
                "fmla   v19.8h, v12.8h, v4.h[3]     \n"
                "fmla   v20.8h, v12.8h, v4.h[4]     \n"
                "fmla   v21.8h, v12.8h, v4.h[5]     \n"
                "fmla   v22.8h, v12.8h, v4.h[6]     \n"
                "fmla   v23.8h, v12.8h, v4.h[7]     \n"

                "fmla   v16.8h, v13.8h, v5.h[0]     \n"
                "fmla   v17.8h, v13.8h, v5.h[1]     \n"
                "fmla   v18.8h, v13.8h, v5.h[2]     \n"
                "fmla   v19.8h, v13.8h, v5.h[3]     \n"
                "fmla   v20.8h, v13.8h, v5.h[4]     \n"
                "fmla   v21.8h, v13.8h, v5.h[5]     \n"
                "fmla   v22.8h, v13.8h, v5.h[6]     \n"
                "fmla   v23.8h, v13.8h, v5.h[7]     \n"

                "fmla   v16.8h, v14.8h, v6.h[0]     \n"
                "fmla   v17.8h, v14.8h, v6.h[1]     \n"
                "fmla   v18.8h, v14.8h, v6.h[2]     \n"
                "fmla   v19.8h, v14.8h, v6.h[3]     \n"
                "fmla   v20.8h, v14.8h, v6.h[4]     \n"
                "fmla   v21.8h, v14.8h, v6.h[5]     \n"
                "fmla   v22.8h, v14.8h, v6.h[6]     \n"
                "fmla   v23.8h, v14.8h, v6.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.8h, v15.8h, v7.h[0]     \n"
                "fmla   v17.8h, v15.8h, v7.h[1]     \n"
                "fmla   v18.8h, v15.8h, v7.h[2]     \n"
                "fmla   v19.8h, v15.8h, v7.h[3]     \n"
                "fmla   v20.8h, v15.8h, v7.h[4]     \n"
                "fmla   v21.8h, v15.8h, v7.h[5]     \n"
                "fmla   v22.8h, v15.8h, v7.h[6]     \n"
                "fmla   v23.8h, v15.8h, v7.h[7]     \n"

                "bne    0b                          \n"

                "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
                "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v16.8h}, [%8]              \n"
                "mov    v17.16b, v16.16b            \n"
                "mov    v18.16b, v16.16b            \n"
                "mov    v19.16b, v16.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%2], #64 \n" // r0123

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                "fmla   v17.8h, v8.8h, v1.h[0]      \n"
                "fmla   v18.8h, v8.8h, v2.h[0]      \n"
                "fmla   v19.8h, v8.8h, v3.h[0]      \n"

                "fmla   v16.8h, v9.8h, v0.h[1]      \n"
                "fmla   v17.8h, v9.8h, v1.h[1]      \n"
                "fmla   v18.8h, v9.8h, v2.h[1]      \n"
                "fmla   v19.8h, v9.8h, v3.h[1]      \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                "fmla   v17.8h, v10.8h, v1.h[2]     \n"
                "fmla   v18.8h, v10.8h, v2.h[2]     \n"
                "fmla   v19.8h, v10.8h, v3.h[2]     \n"

                "fmla   v16.8h, v11.8h, v0.h[3]     \n"
                "fmla   v17.8h, v11.8h, v1.h[3]     \n"
                "fmla   v18.8h, v11.8h, v2.h[3]     \n"
                "fmla   v19.8h, v11.8h, v3.h[3]     \n"

                "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                "fmla   v17.8h, v12.8h, v1.h[4]     \n"
                "fmla   v18.8h, v12.8h, v2.h[4]     \n"
                "fmla   v19.8h, v12.8h, v3.h[4]     \n"

                "fmla   v16.8h, v13.8h, v0.h[5]     \n"
                "fmla   v17.8h, v13.8h, v1.h[5]     \n"
                "fmla   v18.8h, v13.8h, v2.h[5]     \n"
                "fmla   v19.8h, v13.8h, v3.h[5]     \n"

                "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                "fmla   v17.8h, v14.8h, v1.h[6]     \n"
                "fmla   v18.8h, v14.8h, v2.h[6]     \n"
                "fmla   v19.8h, v14.8h, v3.h[6]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.8h, v15.8h, v0.h[7]     \n"
                "fmla   v17.8h, v15.8h, v1.h[7]     \n"
                "fmla   v18.8h, v15.8h, v2.h[7]     \n"
                "fmla   v19.8h, v15.8h, v3.h[7]     \n"

                "bne    0b                          \n"

                "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
        }
        for (; i + 1 < size; i += 2)
        {
            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v16.8h}, [%8]              \n"
                "mov    v17.16b, v16.16b            \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.8h, v1.8h}, [%2], #32   \n" // r01

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                "fmla   v17.8h, v8.8h, v1.h[0]      \n"

                "fmla   v16.8h, v9.8h, v0.h[1]      \n"
                "fmla   v17.8h, v9.8h, v1.h[1]      \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                "fmla   v17.8h, v10.8h, v1.h[2]     \n"

                "fmla   v16.8h, v11.8h, v0.h[3]     \n"
                "fmla   v17.8h, v11.8h, v1.h[3]     \n"

                "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                "fmla   v17.8h, v12.8h, v1.h[4]     \n"

                "fmla   v16.8h, v13.8h, v0.h[5]     \n"
                "fmla   v17.8h, v13.8h, v1.h[5]     \n"

                "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                "fmla   v17.8h, v14.8h, v1.h[6]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.8h, v15.8h, v0.h[7]     \n"
                "fmla   v17.8h, v15.8h, v1.h[7]     \n"

                "bne    0b                          \n"

                "st1    {v16.8h, v17.8h}, [%1], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const __fp16* kptr0 = kernel.channel(p);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v16.8h}, [%8]              \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.8h}, [%2], #16          \n" // r0

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // w0123

                "fmla   v16.8h, v8.8h, v0.h[0]      \n"
                "fmla   v16.8h, v9.8h, v0.h[1]      \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // w4567

                "fmla   v16.8h, v10.8h, v0.h[2]     \n"
                "fmla   v16.8h, v11.8h, v0.h[3]     \n"

                "fmla   v16.8h, v12.8h, v0.h[4]     \n"
                "fmla   v16.8h, v13.8h, v0.h[5]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.8h, v14.8h, v0.h[6]     \n"
                "fmla   v16.8h, v15.8h, v0.h[7]     \n"

                "bne    0b                          \n"

                "st1    {v16.8h}, [%1], #16         \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr0)    // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr0),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
        }
    }

    //     // NOTE sgemm
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         const __fp16 bias0 = bias ? bias[p] : 0.f;
    //
    //         __fp16* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             __fp16 sum = bias0;
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

static void conv1x1s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 8;

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
                float16x8_t _v0 = vld1q_f16(r0);
                float16x8_t _v1 = vld1q_f16(r0 + 16);
                float16x8_t _v2 = vld1q_f16(r0 + 32);
                float16x8_t _v3 = vld1q_f16(r0 + 48);
                vst1q_f16(outptr, _v0);
                vst1q_f16(outptr + 8, _v1);
                vst1q_f16(outptr + 16, _v2);
                vst1q_f16(outptr + 24, _v3);

                r0 += 64;
                outptr += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                float16x8_t _v0 = vld1q_f16(r0);
                float16x8_t _v1 = vld1q_f16(r0 + 16);
                vst1q_f16(outptr, _v0);
                vst1q_f16(outptr + 8, _v1);

                r0 += 32;
                outptr += 16;
            }
            for (; j < outw; j++)
            {
                float16x8_t _v = vld1q_f16(r0);
                vst1q_f16(outptr, _v);

                r0 += 16;
                outptr += 8;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack8_fp16sa_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
