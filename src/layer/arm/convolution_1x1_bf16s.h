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

static void conv1x1s1_sgemm_transform_kernel_bf16s_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const float* kernel = _kernel;

    // interleave
#if __ARM_NEON && __aarch64__
    kernel_tm.create(4 * 8, inch / 4 + inch % 4, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)2u, 1);
#else
    kernel_tm.create(4 * 4, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)2u, 1);
#endif // __ARM_NEON && __aarch64__

    int p = 0;
#if __ARM_NEON && __aarch64__
    for (; p + 7 < outch; p += 8)
    {
        const float* kernel0 = kernel + (p + 0) * inch;
        const float* kernel1 = kernel + (p + 1) * inch;
        const float* kernel2 = kernel + (p + 2) * inch;
        const float* kernel3 = kernel + (p + 3) * inch;
        const float* kernel4 = kernel + (p + 4) * inch;
        const float* kernel5 = kernel + (p + 5) * inch;
        const float* kernel6 = kernel + (p + 6) * inch;
        const float* kernel7 = kernel + (p + 7) * inch;

        unsigned short* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            // kernel0...7 0
            ktmp[0] = float32_to_bfloat16(kernel0[0]);
            ktmp[1] = float32_to_bfloat16(kernel1[0]);
            ktmp[2] = float32_to_bfloat16(kernel2[0]);
            ktmp[3] = float32_to_bfloat16(kernel3[0]);
            ktmp[4] = float32_to_bfloat16(kernel4[0]);
            ktmp[5] = float32_to_bfloat16(kernel5[0]);
            ktmp[6] = float32_to_bfloat16(kernel6[0]);
            ktmp[7] = float32_to_bfloat16(kernel7[0]);

            ktmp += 8;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
            kernel4 += 1;
            kernel5 += 1;
            kernel6 += 1;
            kernel7 += 1;
        }
    }
#endif // __ARM_NEON && __aarch64__
    for (; p + 3 < outch; p += 4)
    {
        const float* kernel0 = kernel + (p + 0) * inch;
        const float* kernel1 = kernel + (p + 1) * inch;
        const float* kernel2 = kernel + (p + 2) * inch;
        const float* kernel3 = kernel + (p + 3) * inch;

#if __ARM_NEON && __aarch64__
        unsigned short* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4);
#else
        unsigned short* ktmp = kernel_tm.channel(p / 4);
#endif // __ARM_NEON && __aarch64__

        for (int q = 0; q < inch; q++)
        {
            // kernel0...3 0
            ktmp[0] = float32_to_bfloat16(kernel0[0]);
            ktmp[1] = float32_to_bfloat16(kernel1[0]);
            ktmp[2] = float32_to_bfloat16(kernel2[0]);
            ktmp[3] = float32_to_bfloat16(kernel3[0]);

            ktmp += 4;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
        }
    }
    for (; p < outch; p++)
    {
        const float* kernel0 = kernel + p * inch;

#if __ARM_NEON && __aarch64__
        unsigned short* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
        unsigned short* ktmp = kernel_tm.channel(p / 4 + p % 4);
#endif // __ARM_NEON && __aarch64__

        for (int q = 0; q < inch; q++)
        {
            ktmp[0] = float32_to_bfloat16(kernel0[0]);
            ktmp++;
            kernel0++;
        }
    }
}

static void conv1x1s1_sgemm_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    const float* bias = _bias;

    // interleave
    Mat tmp(8 * 4, inch / 4 + inch % 4, size / 8 + (size % 8) / 4 + size % 4, 2u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const unsigned short* img0 = bottom_blob.channel(0);
            img0 += i;

            unsigned short* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
#if __ARM_NEON
#if __aarch64__
                vst1q_u16(tmpptr, vld1q_u16(img0));

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#else
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.u16   {d0-d1}, [%0 :64]   \n"
                    "vst1.u16   {d0-d1}, [%1 :64]!  \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "q0");

                img0 += bottom_blob.cstep;
#endif // __aarch64__
#else
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON
            }
        }

        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const unsigned short* img0 = bottom_blob.channel(0);
            img0 += i;

            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
#if __ARM_NEON
#if __aarch64__
                vst1_u16(tmpptr, vld1_u16(img0));

                tmpptr += 4;
                img0 += bottom_blob.cstep;
#else
                asm volatile(
                    "pld        [%0, #64]       \n"
                    "vld1.u16   {d0}, [%0 :64]  \n"
                    "vst1.u16   {d0}, [%1 :64]! \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "d0");

                img0 += bottom_blob.cstep;
#endif // __aarch64__
#else
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const unsigned short* img0 = bottom_blob.channel(0);
            img0 += i;

            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        unsigned short* outptr0 = top_blob.channel(p);
        unsigned short* outptr1 = top_blob.channel(p + 1);
        unsigned short* outptr2 = top_blob.channel(p + 2);
        unsigned short* outptr3 = top_blob.channel(p + 3);
        unsigned short* outptr4 = top_blob.channel(p + 4);
        unsigned short* outptr5 = top_blob.channel(p + 5);
        unsigned short* outptr6 = top_blob.channel(p + 6);
        unsigned short* outptr7 = top_blob.channel(p + 7);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8);
            const unsigned short* kptr = kernel.channel(p / 8);

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
                "lsr    w4, %w21, #2            \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%8, #256]   \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%8], #32   \n"

                "shll   v8.4s, v8.4h, #16       \n"
                "shll   v9.4s, v9.4h, #16       \n"
                "shll   v10.4s, v10.4h, #16     \n"
                "shll   v11.4s, v11.4h, #16     \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%9], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

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

                "prfm   pldl1keep, [%8, #256]   \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%8], #32 \n"

                "shll   v12.4s, v12.4h, #16     \n"
                "shll   v13.4s, v13.4h, #16     \n"
                "shll   v14.4s, v14.4h, #16     \n"
                "shll   v15.4s, v15.4h, #16     \n"

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

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%9], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

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
                "and    w4, %w21, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #128]   \n"
                "ld1    {v8.4h, v9.4h}, [%8], #16   \n"

                "shll   v8.4s, v8.4h, #16       \n"
                "shll   v9.4s, v9.4h, #16       \n"

                "prfm   pldl1keep, [%9, #128]   \n"
                "ld1    {v0.4h, v1.4h}, [%9], #16   \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"

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

                "st1    {v16.4h, v17.4h}, [%0], #16 \n"
                "st1    {v18.4h, v19.4h}, [%1], #16 \n"
                "st1    {v20.4h, v21.4h}, [%2], #16 \n"
                "st1    {v22.4h, v23.4h}, [%3], #16 \n"
                "st1    {v24.4h, v25.4h}, [%4], #16 \n"
                "st1    {v26.4h, v27.4h}, [%5], #16 \n"
                "st1    {v28.4h, v29.4h}, [%6], #16 \n"
                "st1    {v30.4h, v31.4h}, [%7], #16 \n"

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
                "r"(inch)     // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }

        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const unsigned short* kptr = kernel.channel(p / 8);

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
                "lsr    w4, %w21, #2            \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%8, #256]   \n"
                "ld1    {v8.4h, v9.4h, v10.4h, v11.4h}, [%8], #32   \n"

                "shll   v8.4s, v8.4h, #16       \n"
                "shll   v9.4s, v9.4h, #16       \n"
                "shll   v10.4s, v10.4h, #16     \n"
                "shll   v11.4s, v11.4h, #16     \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%9], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

                "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                "fmla   v19.4s, v8.4s, v0.s[3]  \n"
                "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%9], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

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
                "and    w4, %w21, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #64]    \n"
                "ld1    {v8.4h}, [%8], #8       \n"

                "shll   v8.4s, v8.4h, #16       \n"

                "prfm   pldl1keep, [%9, #128]   \n"
                "ld1    {v0.4h, v1.4h}, [%9], #16   \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"

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

                "shrn   v16.4h, v16.4s, #16         \n"
                "shrn   v17.4h, v17.4s, #16         \n"
                "shrn   v18.4h, v18.4s, #16         \n"
                "shrn   v19.4h, v19.4s, #16         \n"
                "shrn   v20.4h, v20.4s, #16         \n"
                "shrn   v21.4h, v21.4s, #16         \n"
                "shrn   v22.4h, v22.4s, #16         \n"
                "shrn   v23.4h, v23.4s, #16         \n"

                "st1    {v16.4h}, [%0], #8      \n"
                "st1    {v17.4h}, [%1], #8      \n"
                "st1    {v18.4h}, [%2], #8      \n"
                "st1    {v19.4h}, [%3], #8      \n"
                "st1    {v20.4h}, [%4], #8      \n"
                "st1    {v21.4h}, [%5], #8      \n"
                "st1    {v22.4h}, [%6], #8      \n"
                "st1    {v23.4h}, [%7], #8      \n"

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
                "r"(inch)     // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }

        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const unsigned short* kptr = kernel.channel(p / 8);

            asm volatile(
                "ld1    {v24.4s, v25.4s}, [%20] \n"

                // inch loop
                "lsr    w4, %w21, #2            \n" // w4 = nn = inch >> 2
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

                "prfm   pldl1keep, [%8, #64]    \n"
                "ld1    {v8.4h}, [%8], #8       \n"

                "shll   v8.4s, v8.4h, #16       \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%9], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

                "fmla   v16.4s, v0.4s, v8.s[0]  \n"
                "fmla   v17.4s, v1.4s, v8.s[0]  \n"
                "fmla   v18.4s, v2.4s, v8.s[1]  \n"
                "fmla   v19.4s, v3.4s, v8.s[1]  \n"

                "prfm   pldl1keep, [%9, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%9], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

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
                "and    w4, %w21, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%8, #16]    \n"
                "ld1r   {v8.4h}, [%8], #2       \n"

                "shll   v8.4s, v8.4h, #16       \n"

                "prfm   pldl1keep, [%9, #128]   \n"
                "ld1    {v0.4h, v1.4h}, [%9], #16   \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v24.4s, v8.4s, v0.4s    \n"
                "fmla   v25.4s, v8.4s, v1.4s    \n"

                "bne    2b                      \n"

                "3:                             \n"

                "shrn   v24.4h, v24.4s, #16         \n"
                "shrn   v25.4h, v25.4s, #16         \n"

                "st1    {v24.h}[0],[%0], #2     \n"
                "st1    {v24.h}[1],[%1], #2     \n"
                "st1    {v24.h}[2],[%2], #2     \n"
                "st1    {v24.h}[3],[%3], #2     \n"
                "st1    {v25.h}[0],[%4], #2     \n"
                "st1    {v25.h}[1],[%5], #2     \n"
                "st1    {v25.h}[2],[%6], #2     \n"
                "st1    {v25.h}[3],[%7], #2     \n"

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
                "r"(inch)     // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
        }
    }
#endif // __ARM_NEON && __aarch64__

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        unsigned short* outptr0 = top_blob.channel(p);
        unsigned short* outptr1 = top_blob.channel(p + 1);
        unsigned short* outptr2 = top_blob.channel(p + 2);
        unsigned short* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
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
                "lsr    w4, %w13, #2            \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%4], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

                "prfm   pldl1keep, [%5, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%5], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                "fmla   v14.4s, v4.4s, v0.s[3]  \n"

                "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                "prfm   pldl1keep, [%4, #256]   \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%4], #32 \n"

                "shll   v16.4s, v16.4h, #16     \n"
                "shll   v17.4s, v17.4h, #16     \n"
                "shll   v18.4s, v18.4h, #16     \n"
                "shll   v19.4s, v19.4h, #16     \n"

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
                "and    w4, %w13, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #128]   \n"
                "ld1    {v4.4h, v5.4h}, [%4], #16   \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"

                "prfm   pldl1keep, [%5, #64]    \n"
                "ld1    {v0.4h}, [%5], #8       \n"

                "shll   v0.4s, v0.4h, #16       \n"

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

                "shrn   v8.4h, v8.4s, #16           \n"
                "shrn   v9.4h, v9.4s, #16           \n"
                "shrn   v10.4h, v10.4s, #16         \n"
                "shrn   v11.4h, v11.4s, #16         \n"
                "shrn   v12.4h, v12.4s, #16         \n"
                "shrn   v13.4h, v13.4s, #16         \n"
                "shrn   v14.4h, v14.4s, #16         \n"
                "shrn   v15.4h, v15.4s, #16         \n"

                "st1    {v8.4h, v9.4h}, [%0], #16   \n"
                "st1    {v10.4h, v11.4h}, [%1], #16 \n"
                "st1    {v12.4h, v13.4h}, [%2], #16 \n"
                "st1    {v14.4h, v15.4h}, [%3], #16 \n"

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
                "r"(inch)     // %13
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
                "lsr        r4, %13, #2         \n" // r4 = nn = inch >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%4, #256]          \n"
                "vld1.u16   {d12-d15}, [%4 :64]! \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "pld        [%5, #256]          \n"
                "vld1.u16   {d4-d7}, [%5 :64]!  \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"

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

                "pld        [%4, #256]          \n"
                "vld1.u16   {d12-d15}, [%4 :64]! \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

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
                "and        r4, %13, #3         \n" // r4 = remain = inch & 3;
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #128]          \n"
                "vld1.u16   {d10-d11}, [%4 :64]! \n"

                "vshll.u16  q4, d10, #16        \n"
                "vshll.u16  q5, d11, #16        \n"

                "pld        [%5, #64]           \n"
                "vld1.u16   {d1}, [%5 :64]!     \n"

                "vshll.u16  q0, d1, #16         \n"

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

                "vshrn.u32  d16, q8, #16        \n"
                "vshrn.u32  d17, q9, #16        \n"
                "vshrn.u32  d20, q10, #16       \n"
                "vshrn.u32  d21, q11, #16       \n"
                "vshrn.u32  d24, q12, #16       \n"
                "vshrn.u32  d25, q13, #16       \n"
                "vshrn.u32  d28, q14, #16       \n"
                "vshrn.u32  d29, q15, #16       \n"

                "vst1.u16   {d16-d17}, [%0 :64]!   \n"
                "vst1.u16   {d20-d21}, [%1 :64]!   \n"
                "vst1.u16   {d24-d25}, [%2 :64]!   \n"
                "vst1.u16   {d28-d29}, [%3 :64]!   \n"

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
                "r"(inch)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else
            float sum0_0 = biasptr[0];
            float sum0_1 = biasptr[0];
            float sum0_2 = biasptr[0];
            float sum0_3 = biasptr[0];
            float sum0_4 = biasptr[0];
            float sum0_5 = biasptr[0];
            float sum0_6 = biasptr[0];
            float sum0_7 = biasptr[0];

            float sum1_0 = biasptr[1];
            float sum1_1 = biasptr[1];
            float sum1_2 = biasptr[1];
            float sum1_3 = biasptr[1];
            float sum1_4 = biasptr[1];
            float sum1_5 = biasptr[1];
            float sum1_6 = biasptr[1];
            float sum1_7 = biasptr[1];

            float sum2_0 = biasptr[2];
            float sum2_1 = biasptr[2];
            float sum2_2 = biasptr[2];
            float sum2_3 = biasptr[2];
            float sum2_4 = biasptr[2];
            float sum2_5 = biasptr[2];
            float sum2_6 = biasptr[2];
            float sum2_7 = biasptr[2];

            float sum3_0 = biasptr[3];
            float sum3_1 = biasptr[3];
            float sum3_2 = biasptr[3];
            float sum3_3 = biasptr[3];
            float sum3_4 = biasptr[3];
            float sum3_5 = biasptr[3];
            float sum3_6 = biasptr[3];
            float sum3_7 = biasptr[3];

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum0_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum0_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum0_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);
                sum0_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[0]);
                sum0_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[0]);
                sum0_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[0]);
                sum0_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[0]);

                sum1_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum1_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[1]);
                sum1_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[1]);
                sum1_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[1]);
                sum1_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[1]);
                sum1_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[1]);
                sum1_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[1]);
                sum1_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[1]);

                sum2_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum2_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[2]);
                sum2_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[2]);
                sum2_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[2]);
                sum2_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[2]);
                sum2_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[2]);
                sum2_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[2]);
                sum2_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[2]);

                sum3_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);
                sum3_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[3]);
                sum3_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[3]);
                sum3_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[3]);
                sum3_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[3]);
                sum3_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[3]);
                sum3_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[3]);
                sum3_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[3]);

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0_0);
            outptr0[1] = float32_to_bfloat16(sum0_1);
            outptr0[2] = float32_to_bfloat16(sum0_2);
            outptr0[3] = float32_to_bfloat16(sum0_3);
            outptr0[4] = float32_to_bfloat16(sum0_4);
            outptr0[5] = float32_to_bfloat16(sum0_5);
            outptr0[6] = float32_to_bfloat16(sum0_6);
            outptr0[7] = float32_to_bfloat16(sum0_7);

            outptr1[0] = float32_to_bfloat16(sum1_0);
            outptr1[1] = float32_to_bfloat16(sum1_1);
            outptr1[2] = float32_to_bfloat16(sum1_2);
            outptr1[3] = float32_to_bfloat16(sum1_3);
            outptr1[4] = float32_to_bfloat16(sum1_4);
            outptr1[5] = float32_to_bfloat16(sum1_5);
            outptr1[6] = float32_to_bfloat16(sum1_6);
            outptr1[7] = float32_to_bfloat16(sum1_7);

            outptr2[0] = float32_to_bfloat16(sum2_0);
            outptr2[1] = float32_to_bfloat16(sum2_1);
            outptr2[2] = float32_to_bfloat16(sum2_2);
            outptr2[3] = float32_to_bfloat16(sum2_3);
            outptr2[4] = float32_to_bfloat16(sum2_4);
            outptr2[5] = float32_to_bfloat16(sum2_5);
            outptr2[6] = float32_to_bfloat16(sum2_6);
            outptr2[7] = float32_to_bfloat16(sum2_7);

            outptr3[0] = float32_to_bfloat16(sum3_0);
            outptr3[1] = float32_to_bfloat16(sum3_1);
            outptr3[2] = float32_to_bfloat16(sum3_2);
            outptr3[3] = float32_to_bfloat16(sum3_3);
            outptr3[4] = float32_to_bfloat16(sum3_4);
            outptr3[5] = float32_to_bfloat16(sum3_5);
            outptr3[6] = float32_to_bfloat16(sum3_6);
            outptr3[7] = float32_to_bfloat16(sum3_7);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%12]          \n"
                "dup    v8.4s, v0.s[0]          \n"
                "dup    v9.4s, v0.s[1]          \n"
                "dup    v10.4s, v0.s[2]         \n"
                "dup    v11.4s, v0.s[3]         \n"

                // inch loop
                "lsr    w4, %w13, #2            \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%4], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

                "prfm   pldl1keep, [%5, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%5], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

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
                "and    w4, %w13, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #64]    \n"
                "ld1    {v4.4h}, [%4], #8       \n"

                "shll   v4.4s, v4.4h, #16       \n"

                "prfm   pldl1keep, [%5, #64]    \n"
                "ld1    {v0.4h}, [%5], #8       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                "bne    2b                      \n"

                "3:                             \n"

                "shrn   v8.4h, v8.4s, #16           \n"
                "shrn   v9.4h, v9.4s, #16           \n"
                "shrn   v10.4h, v10.4s, #16         \n"
                "shrn   v11.4h, v11.4s, #16         \n"

                "st1    {v8.4h}, [%0], #8       \n"
                "st1    {v9.4h}, [%1], #8       \n"
                "st1    {v10.4h}, [%2], #8      \n"
                "st1    {v11.4h}, [%3], #8      \n"

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
                "r"(inch)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d0-d1}, [%12]      \n"
                "vdup.f32   q8, d0[0]           \n"
                "vdup.f32   q9, d0[1]           \n"
                "vdup.f32   q10, d1[0]          \n"
                "vdup.f32   q11, d1[1]          \n"

                // inch loop
                "lsr        r4, %13, #2         \n" // r4 = nn = inch >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%4, #256]          \n"
                "vld1.u16   {d12-d15}, [%4 :64]! \n"

                "vshll.u16  q4, d12, #16    \n"
                "vshll.u16  q5, d13, #16    \n"
                "vshll.u16  q6, d14, #16    \n"
                "vshll.u16  q7, d15, #16    \n"

                "pld        [%5, #256]          \n"
                "vld1.u16   {d4-d7}, [%5 :64]!  \n"

                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"

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
                "and        r4, %13, #3         \n" // r4 = remain = inch & 3;
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #64]           \n"
                "vld1.u16   {d9}, [%4 :64]!     \n"

                "vshll.u16  q4, d9, #16         \n"

                "pld        [%5, #64]           \n"
                "vld1.u16   {d1}, [%5 :64]!     \n"

                "vshll.u16  q0, d1, #16         \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vshrn.u32  d16, q8, #16        \n"
                "vshrn.u32  d18, q9, #16        \n"
                "vshrn.u32  d20, q10, #16       \n"
                "vshrn.u32  d22, q11, #16       \n"

                "vst1.u16   {d16}, [%0 :64]!    \n"
                "vst1.u16   {d18}, [%1 :64]!    \n"
                "vst1.u16   {d20}, [%2 :64]!    \n"
                "vst1.u16   {d22}, [%3 :64]!    \n"

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
                "r"(inch)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif // __aarch64__
#else
            float sum0_0 = biasptr[0];
            float sum0_1 = biasptr[0];
            float sum0_2 = biasptr[0];
            float sum0_3 = biasptr[0];

            float sum1_0 = biasptr[1];
            float sum1_1 = biasptr[1];
            float sum1_2 = biasptr[1];
            float sum1_3 = biasptr[1];

            float sum2_0 = biasptr[2];
            float sum2_1 = biasptr[2];
            float sum2_2 = biasptr[2];
            float sum2_3 = biasptr[2];

            float sum3_0 = biasptr[3];
            float sum3_1 = biasptr[3];
            float sum3_2 = biasptr[3];
            float sum3_3 = biasptr[3];

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum0_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum0_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum0_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);

                sum1_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum1_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[1]);
                sum1_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[1]);
                sum1_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[1]);

                sum2_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum2_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[2]);
                sum2_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[2]);
                sum2_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[2]);

                sum3_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);
                sum3_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[3]);
                sum3_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[3]);
                sum3_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[3]);

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0_0);
            outptr0[1] = float32_to_bfloat16(sum0_1);
            outptr0[2] = float32_to_bfloat16(sum0_2);
            outptr0[3] = float32_to_bfloat16(sum0_3);

            outptr1[0] = float32_to_bfloat16(sum1_0);
            outptr1[1] = float32_to_bfloat16(sum1_1);
            outptr1[2] = float32_to_bfloat16(sum1_2);
            outptr1[3] = float32_to_bfloat16(sum1_3);

            outptr2[0] = float32_to_bfloat16(sum2_0);
            outptr2[1] = float32_to_bfloat16(sum2_1);
            outptr2[2] = float32_to_bfloat16(sum2_2);
            outptr2[3] = float32_to_bfloat16(sum2_3);

            outptr3[0] = float32_to_bfloat16(sum3_0);
            outptr3[1] = float32_to_bfloat16(sum3_1);
            outptr3[2] = float32_to_bfloat16(sum3_2);
            outptr3[3] = float32_to_bfloat16(sum3_3);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "ld1    {v12.4s}, [%12]         \n"

                // inch loop
                "lsr    w4, %w13, #2            \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "eor    v8.16b, v8.16b, v8.16b  \n"
                "eor    v9.16b, v9.16b, v9.16b  \n"
                "eor    v10.16b, v10.16b, v10.16b  \n"
                "eor    v11.16b, v11.16b, v11.16b  \n"

                "0:                             \n"

                "prfm   pldl1keep, [%4, #64]    \n"
                "ld1    {v4.4h}, [%4], #8       \n"

                "shll   v4.4s, v4.4h, #16       \n"

                "prfm   pldl1keep, [%5, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%5], #32     \n"

                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"

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
                "and    w4, %w13, #3            \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%4, #16]    \n"
                "ld1r   {v4.4h}, [%4], #2       \n"

                "shll   v4.4s, v4.4h, #16       \n"

                "prfm   pldl1keep, [%5, #64]    \n"
                "ld1    {v0.4h}, [%5], #8       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v12.4s, v4.4s, v0.4s    \n"

                "bne    2b                      \n"

                "3:                             \n"

                "shrn   v12.4h, v12.4s, #16     \n"

                "st1    {v12.h}[0], [%0], #2    \n"
                "st1    {v12.h}[1], [%1], #2    \n"
                "st1    {v12.h}[2], [%2], #2    \n"
                "st1    {v12.h}[3], [%3], #2    \n"

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
                "r"(inch)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d24-d25}, [%12]    \n"

                // inch loop
                "lsr        r4, %13, #2         \n" // r4 = nn = inch >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "veor       q8, q8, q8          \n"
                "veor       q9, q9, q9          \n"
                "veor       q10, q10, q10       \n"
                "veor       q11, q11, q11       \n"

                "0:                             \n"

                "pld        [%4, #64]           \n"
                "vld1.u16   {d9}, [%4 :64]!     \n"

                "vshll.u16  q4, d9, #16         \n"

                "pld        [%5, #256]          \n"
                "vld1.u16   {d4-d7}, [%5 :64]!  \n"

                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"

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
                "and        r4, %13, #3         \n" // r4 = remain = inch & 3;
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%4, #16]           \n"
                "vld1.u16   {d9[]}, [%4]!       \n"

                "vshll.u16  q4, d9, #16         \n"

                "pld        [%5, #64]           \n"
                "vld1.u16   {d1}, [%5 :64]!     \n"

                "vshll.u16  q0, d1, #16         \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q12, q4, q0         \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vshrn.u32  d24, q12, #16       \n"

                "vst1.u16   {d24[0]}, [%0]!     \n"
                "vst1.u16   {d24[1]}, [%1]!     \n"
                "vst1.u16   {d24[2]}, [%2]!     \n"
                "vst1.u16   {d24[3]}, [%3]!     \n"

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
                "r"(inch)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
            float sum0 = biasptr[0];
            float sum1 = biasptr[1];
            float sum2 = biasptr[2];
            float sum3 = biasptr[3];

            for (int q = 0; q < inch; q++)
            {
                sum0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum1 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum2 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum3 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0);
            outptr1[0] = float32_to_bfloat16(sum1);
            outptr2[0] = float32_to_bfloat16(sum2);
            outptr3[0] = float32_to_bfloat16(sum3);

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
#endif // __ARM_NEON
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        unsigned short* outptr0 = out0;

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4 + p % 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w6              \n"
                "dup    v9.4s, %w6              \n"

                // inch loop
                "lsr    w4, %w7, #2             \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%1, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

                "prfm   pldl1keep, [%2, #64]    \n"
                "ld1    {v0.4h}, [%2], #8       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v9.4s, v5.4s, v0.s[0]   \n"

                "prfm   pldl1keep, [%1, #256]   \n"
                "ld1    {v12.4h, v13.4h, v14.4h, v15.4h}, [%1], #32 \n"

                "shll   v12.4s, v12.4h, #16     \n"
                "shll   v13.4s, v13.4h, #16     \n"
                "shll   v14.4s, v14.4h, #16     \n"
                "shll   v15.4s, v15.4h, #16     \n"

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
                "and    w4, %w7, #3             \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%1, #128]   \n"
                "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"

                "prfm   pldl1keep, [%2, #16]    \n"
                "ld1r   {v0.4h}, [%2], #2       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.4s     \n"
                "fmla   v9.4s, v5.4s, v0.4s     \n"

                "bne    2b                      \n"

                "3:                             \n"

                "shrn   v8.4h, v8.4s, #16           \n"
                "shrn   v9.4h, v9.4s, #16           \n"

                "st1    {v8.4h, v9.4h}, [%0], #16   \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(inch)   // %7
                : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %6              \n"
                "vdup.f32   q9, %6              \n"

                // inch loop
                "lsr        r4, %7, #2          \n" // r4 = nn = inch >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"

                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"

                "pld        [%2, #64]           \n"
                "vld1.u16   {d1}, [%2 :64]!     \n"

                "vshll.u16  q0, d1, #16         \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q5, d0[0]       \n"

                "pld        [%1, #256]          \n"
                "vld1.u16   {d28-d31}, [%1 :64]! \n"

                "vshll.u16  q12, d28, #16       \n"
                "vshll.u16  q13, d29, #16       \n"
                "vshll.u16  q14, d30, #16       \n"
                "vshll.u16  q15, d31, #16       \n"

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
                "and        r4, %7, #3          \n" // r4 = remain = inch & 3;
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%1, #128]          \n"
                "vld1.u16   {d10-d11}, [%1 :64]! \n"

                "vshll.u16  q4, d10, #16        \n"
                "vshll.u16  q5, d11, #16        \n"

                "pld        [%2, #16]           \n"
                "vld1.u16   {d1[]}, [%2]!       \n"

                "vshll.u16  q0, d1, #16         \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, q0          \n"
                "vmla.f32   q9, q5, q0          \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vshrn.u32  d16, q8, #16        \n"
                "vshrn.u32  d17, q9, #16        \n"

                "vst1.u16   {d16-d17}, [%0 :64]! \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(inch)   // %7
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else
            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;
            float sum4 = bias0;
            float sum5 = bias0;
            float sum6 = bias0;
            float sum7 = bias0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);
                sum4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[0]);
                sum5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[0]);
                sum6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[0]);
                sum7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[0]);

                tmpptr += 8;
                kptr++;
            }

            outptr0[0] = float32_to_bfloat16(sum0);
            outptr0[1] = float32_to_bfloat16(sum1);
            outptr0[2] = float32_to_bfloat16(sum2);
            outptr0[3] = float32_to_bfloat16(sum3);
            outptr0[4] = float32_to_bfloat16(sum4);
            outptr0[5] = float32_to_bfloat16(sum5);
            outptr0[6] = float32_to_bfloat16(sum6);
            outptr0[7] = float32_to_bfloat16(sum7);

            outptr0 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4 + p % 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
            asm volatile(
                "dup    v8.4s, %w6              \n"

                // inch loop
                "lsr    w4, %w7, #2             \n" // w4 = nn = inch >> 2
                "cmp    w4, #0                  \n"
                "beq    1f                      \n"

                "0:                             \n"

                "prfm   pldl1keep, [%1, #256]   \n"
                "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32     \n"

                "shll   v4.4s, v4.4h, #16       \n"
                "shll   v5.4s, v5.4h, #16       \n"
                "shll   v6.4s, v6.4h, #16       \n"
                "shll   v7.4s, v7.4h, #16       \n"

                "prfm   pldl1keep, [%2, #64]    \n"
                "ld1    {v0.4h}, [%2], #8       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                "fmla   v8.4s, v5.4s, v0.s[1]   \n"
                "fmla   v8.4s, v6.4s, v0.s[2]   \n"
                "fmla   v8.4s, v7.4s, v0.s[3]   \n"

                "bne    0b                      \n"

                "1:                             \n"

                // remain loop
                "and    w4, %w7, #3             \n" // w4 = remain = inch & 3;
                "cmp    w4, #0                  \n"
                "beq    3f                      \n"

                "2:                             \n"

                "prfm   pldl1keep, [%1, #64]    \n"
                "ld1    {v4.4h}, [%1], #8       \n"

                "shll   v4.4s, v4.4h, #16       \n"

                "prfm   pldl1keep, [%2, #16]    \n"
                "ld1r   {v0.4h}, [%2], #2       \n"

                "shll   v0.4s, v0.4h, #16       \n"

                "subs   w4, w4, #1              \n"

                "fmla   v8.4s, v4.4s, v0.4s     \n"

                "bne    2b                      \n"

                "3:                             \n"

                "shrn   v8.4h, v8.4s, #16           \n"

                "st1    {v8.4h}, [%0], #8       \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(inch)   // %7
                : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8");
#else  // __aarch64__
            asm volatile(
                "vdup.f32   q8, %6              \n"

                // inch loop
                "lsr        r4, %7, #2          \n" // r4 = nn = inch >> 2
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "0:                             \n"

                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"

                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"

                "pld        [%2, #64]           \n"
                "vld1.u16   {d1}, [%2]!         \n"

                "vshll.u16  q0, d1, #16         \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q8, q5, d0[1]       \n"
                "vmla.f32   q8, q6, d1[0]       \n"
                "vmla.f32   q8, q7, d1[1]       \n"

                "bne        0b                  \n"

                "1:                             \n"

                // remain loop
                "and        r4, %7, #3          \n" // r4 = remain = inch & 3;
                "cmp        r4, #0              \n"
                "beq        3f                  \n"

                "2:                             \n"

                "pld        [%1, #64]           \n"
                "vld1.u16   {d9}, [%1 :64]!     \n"

                "vshll.u16  q4, d9, #16         \n"

                "pld        [%2, #16]           \n"
                "vld1.u16   {d1[]}, [%2]!       \n"

                "vshll.u16  q0, d1, #16         \n"

                "subs       r4, r4, #1          \n"

                "vmla.f32   q8, q4, q0          \n"

                "bne        2b                  \n"

                "3:                             \n"

                "vshrn.u32  d16, q8, #16        \n"

                "vst1.u16   {d16}, [%0 :64]!    \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(bias0), // %6
                "r"(inch)   // %7
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8");
#endif // __aarch64__
#else
            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = float32_to_bfloat16(sum0);
            outptr0[1] = float32_to_bfloat16(sum1);
            outptr0[2] = float32_to_bfloat16(sum2);
            outptr0[3] = float32_to_bfloat16(sum3);

            outptr0 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const unsigned short* kptr = kernel.channel(p / 4 + p % 4);
#endif // __ARM_NEON && __aarch64__

            int q = 0;

#if __ARM_NEON
            float32x4_t _sum0 = vdupq_n_f32(0.f);

            for (; q + 3 < inch; q += 4)
            {
                float32x4_t _p0 = vcvt_f32_bf16(vld1_u16(tmpptr));
                tmpptr += 4;

                float32x4_t _k0 = vcvt_f32_bf16(vld1_u16(kptr));
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
#else
            float sum0 = bias0;
#endif // __ARM_NEON

            for (; q < inch; q++)
            {
                sum0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                tmpptr++;
                kptr++;
            }

            outptr0[0] = float32_to_bfloat16(sum0);

            outptr0++;
        }
    }

    //     // NOTE sgemm
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         const float bias0 = bias ? bias[p] : 0.f;
    //
    //         float* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             float sum = bias0;
    //
    //             const float* kptr = _kernel.channel(p/8 + p%8);
    //
    //             for (int q=0; q<inch; q++)
    //             {
    //                 const float* img0 = bottom_blob.channel(q);
    //
    //                 sum += img0[i] * kptr[0];
    //                 kptr ++;
    //             }
    //
    //             outptr0[i] = sum;
    //         }
    //     }
}
