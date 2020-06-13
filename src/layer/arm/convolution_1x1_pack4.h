// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sgemm_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4b-4a-inch/4a-outch/4b
#if __aarch64__
    kernel_tm_pack4.create(2 * 1, inch / 4, (outch / 4) / 2 + (outch / 4) % 2, (size_t)4u * 16, 16);
#else
    kernel_tm_pack4.create(1, inch / 4, outch / 4, (size_t)4u * 16, 16);
#endif

    int q = 0;
#if __aarch64__
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

        float* g0 = kernel_tm_pack4.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = k0[0];
            g0[1] = k1[0];
            g0[2] = k2[0];
            g0[3] = k3[0];

            g0[4] = k4[0];
            g0[5] = k5[0];
            g0[6] = k6[0];
            g0[7] = k7[0];

            g0[8] = k0[1];
            g0[9] = k1[1];
            g0[10] = k2[1];
            g0[11] = k3[1];

            g0[12] = k4[1];
            g0[13] = k5[1];
            g0[14] = k6[1];
            g0[15] = k7[1];

            g0[16] = k0[2];
            g0[17] = k1[2];
            g0[18] = k2[2];
            g0[19] = k3[2];

            g0[20] = k4[2];
            g0[21] = k5[2];
            g0[22] = k6[2];
            g0[23] = k7[2];

            g0[24] = k0[3];
            g0[25] = k1[3];
            g0[26] = k2[3];
            g0[27] = k3[3];

            g0[28] = k4[3];
            g0[29] = k5[3];
            g0[30] = k6[3];
            g0[31] = k7[3];

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
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;

#if __aarch64__
        float* g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);
#else
        float* g0 = kernel_tm_pack4.channel(q / 4);
#endif

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = k0[0];
            g0[1] = k1[0];
            g0[2] = k2[0];
            g0[3] = k3[0];

            g0[4] = k0[1];
            g0[5] = k1[1];
            g0[6] = k2[1];
            g0[7] = k3[1];

            g0[8] = k0[2];
            g0[9] = k1[2];
            g0[10] = k2[2];
            g0[11] = k3[2];

            g0[12] = k0[3];
            g0[13] = k1[3];
            g0[14] = k2[3];
            g0[15] = k3[3];

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            g0 += 16;
        }
    }
}

static void conv1x1s1_sgemm_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const float* bias = _bias;

    // interleave
#if __aarch64__
    Mat tmp(12, inch, size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, elemsize, elempack, opt.workspace_allocator);
#else
    Mat tmp(8, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
#endif
    {
        int nn_size;
        int remain_size_start;

#if __aarch64__
        nn_size = size / 12;
        remain_size_start = nn_size * 12;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 12;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            float* tmpptr = tmp.channel(i / 12);

            for (int q = 0; q < inch; q++)
            {
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
                img0 += bottom_blob.cstep * 4;
            }
        }
#else
        remain_size_start = 0;
#endif
        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
#else
            float* tmpptr = tmp.channel(i / 8);
#endif

            for (int q = 0; q < inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    "sub    %0, %0, #64                 \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
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
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#endif

            for (int q = 0; q < inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");
#else
                asm volatile(
                    "pld        [%0, #512]          \n"
                    "vldm       %0, {d0-d7}         \n"
                    "vstm       %1!, {d0-d7}        \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
#endif

            for (int q = 0; q < inch; q++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%0]        \n"
                    "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0 :128]  \n"
                    "vst1.f32   {d0-d3}, [%1 :128]! \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "q0", "q1");
#endif // __aarch64__
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i * 4;

#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
#endif

            for (int q = 0; q < inch; q++)
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
                img0 += bottom_blob.cstep * 4;
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 11 < size; i += 12)
        {
            const float* tmpptr = tmp.channel(i / 12);

            const float* kptr01 = (const float*)kernel.channel(pp);

            int nn = inch; // inch always > 0

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

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64 \n" // w0011_01

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

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

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

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64 \n" // w2233_01

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

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

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

                "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"
                "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr01)   // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr01),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 7 < size; i += 8)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);

            const float* kptr01 = (const float*)kernel.channel(pp);

            int nn = inch; // inch always > 0

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

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r4 r5 r6 r7

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

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

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

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr01)   // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr01),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

            const float* kptr01 = (const float*)kernel.channel(pp);

            int nn = inch; // inch always > 0

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

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "fmla   v20.4s, v9.4s, v0.s[0]      \n"
                "fmla   v21.4s, v9.4s, v1.s[0]      \n"
                "fmla   v22.4s, v9.4s, v2.s[0]      \n"
                "fmla   v23.4s, v9.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

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

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr01)   // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr01),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i + 1 < size; i += 2)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

            const float* kptr01 = (const float*)kernel.channel(pp);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%10]       \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v1.16b             \n"
                "mov    v19.16b, v1.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%3], #32   \n" // r0 r1

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v9.4s, v0.s[0]     \n"
                "fmla   v19.4s, v9.4s, v1.s[0]     \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                "fmla   v16.4s, v10.4s, v0.s[1]      \n"
                "fmla   v17.4s, v10.4s, v1.s[1]      \n"
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

                "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                "st1    {v18.4s, v19.4s}, [%2], #32 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr01)   // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr01),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
        }
        for (; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

            const float* kptr01 = (const float*)kernel.channel(pp);

            int nn = inch; // inch always > 0

            asm volatile(
                "ld1    {v16.4s, v17.4s}, [%10]     \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%3, #128]       \n"
                "ld1    {v0.4s}, [%3], #16          \n" // r0

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v9.4s, v0.s[0]      \n"

                "prfm   pldl1keep, [%4, #512]       \n"
                "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                "fmla   v17.4s, v11.4s, v0.s[1]     \n"

                "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                "fmla   v17.4s, v13.4s, v0.s[2]     \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                "fmla   v17.4s, v15.4s, v0.s[3]     \n"

                "bne    0b                          \n"

                "st1    {v16.4s}, [%1], #16         \n"
                "st1    {v17.4s}, [%2], #16         \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(outptr1), // %2
                "=r"(tmpptr),  // %3
                "=r"(kptr01)   // %4
                : "0"(nn),
                "1"(outptr0),
                "2"(outptr1),
                "3"(tmpptr),
                "4"(kptr01),
                "r"(biasptr) // %10
                : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
        }
    }

#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
#if __aarch64__
        for (; i + 11 < size; i += 12)
        {
            float* tmpptr = tmp.channel(i / 12);

            const float* kptr0 = (const float*)kernel.channel(p / 2 + p % 2);

            int nn = inch; // inch always > 0

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

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // w0123_0

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

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

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

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"

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

                "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

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
#endif
        for (; i + 7 < size; i += 8)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8);
            const float* kptr0 = (const float*)kernel.channel(p / 2 + p % 2);
#else
            float* tmpptr = tmp.channel(i / 8);
            const float* kptr0 = (const float*)kernel.channel(p);
#endif

            int nn = inch; // inch always > 0

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

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r4 r5 r6 r7

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

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

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

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d0-d7}    \n"

                "pld        [%3, #512]      \n"
                "vldm       %3!, {d8-d15}   \n"

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

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d0-d7}    \n"

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

                "vstm       %1!, {d16-d23}  \n"
                "vstm       %1!, {d24-d31}  \n"

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
            const float* kptr0 = (const float*)kernel.channel(p / 2 + p % 2);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const float* kptr0 = (const float*)kernel.channel(p);
#endif

            int nn = inch; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"
                "mov    v18.16b, v0.16b             \n"
                "mov    v19.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

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

                "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

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

                "pld        [%2, #512]      \n"
                "vldm       %2!, {d0-d7}    \n"

                "pld        [%3, #512]      \n"
                "vldm       %3!, {d8-d15}   \n"

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
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
        }
        for (; i + 1 < size; i += 2)
        {
#if __aarch64__
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
            const float* kptr0 = (const float*)kernel.channel(p / 2 + p % 2);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const float* kptr0 = (const float*)kernel.channel(p);
#endif

            int nn = inch; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%8]               \n"
                "mov    v16.16b, v0.16b             \n"
                "mov    v17.16b, v0.16b             \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%2], #32   \n" // r0 r1

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

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

                "st1    {v16.4s, v17.4s}, [%1], #32 \n"

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

                "pld        [%2, #256]      \n"
                "vld1.f32   {d0-d3}, [%2 :128]! \n"

                "pld        [%3, #512]      \n"
                "vldm       %3!, {d8-d15}   \n"

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

                "vst1.f32   {d16-d19}, [%1 :128]! \n"

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
            float* tmpptr = tmp.channel(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
            const float* kptr0 = (const float*)kernel.channel(p / 2 + p % 2);
#else
            float* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = (const float*)kernel.channel(p);
#endif

            int nn = inch; // inch always > 0

#if __aarch64__
            asm volatile(
                "ld1    {v16.4s}, [%8]              \n"

                "0:                                 \n"

                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4s}, [%2], #16          \n" // r0

                "prfm   pldl1keep, [%3, #512]       \n"
                "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                "fmla   v16.4s, v9.4s, v0.s[1]      \n"

                "subs   %w0, %w0, #1                \n"

                "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                "fmla   v16.4s, v11.4s, v0.s[3]     \n"

                "bne    0b                          \n"

                "st1    {v16.4s}, [%1], #16         \n"

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

                "pld        [%2, #128]      \n"
                "vld1.f32   {d0-d1}, [%2 :128]! \n"

                "pld        [%3, #512]      \n"
                "vldm       %3!, {d8-d15}   \n"

                "vmla.f32   q8, q4, d0[0]   \n"
                "vmla.f32   q8, q5, d0[1]   \n"

                "subs       %0, %0, #1      \n"

                "vmla.f32   q8, q6, d1[0]   \n"
                "vmla.f32   q8, q7, d1[1]   \n"

                "bne        0b              \n"

                "vst1.f32   {d16-d17}, [%1 :128]! \n"

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
    //             const float* kptr = _kernel.channel(p);
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

static void conv1x1s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
        const float* r0 = bottom_blob.channel(p);
        float* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _v = vld1q_f32(r0);
                vst1q_f32(outptr, _v);

                r0 += 8;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
