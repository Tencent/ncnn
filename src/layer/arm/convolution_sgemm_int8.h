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

#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void im2col_sgemm_int8_neon_i8mm(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
void convolution_im2col_sgemm_transform_kernel_int8_neon_i8mm(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
void im2col_sgemm_int8_neon_asimddp(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt);
void convolution_im2col_sgemm_transform_kernel_int8_neon_asimddp(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);
#endif
#endif

static void im2col_sgemm_int8_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        im2col_sgemm_int8_neon_i8mm(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        im2col_sgemm_int8_neon_asimddp(bottom_im2col, top_blob, kernel, opt);
        return;
    }
#endif
#endif

    // Mat bottom_im2col(size, maxk, inch, 8u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    // permute
    Mat tmp;
#if __ARM_NEON
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
    if (inch >= 8)
    {
        if (size >= 8)
            tmp.create(8 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else if (size >= 4)
            tmp.create(4 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 4 + (size % 4) / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size, 8u, 8, opt.workspace_allocator);
    }
    else if (inch >= 4)
    {
        if (size >= 8)
            tmp.create(8 * maxk, inch / 4 + inch % 4, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else if (size >= 4)
            tmp.create(4 * maxk, inch / 4 + inch % 4, size / 4 + (size % 4) / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
    {
        if (size >= 8)
            tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else if (size >= 4)
            tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 8u, 1, opt.workspace_allocator);
    }
#else  // __ARM_FEATURE_DOTPROD
    if (inch >= 8)
    {
        if (size >= 4)
            tmp.create(4 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 4 + (size % 4) / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size, 8u, 8, opt.workspace_allocator);
    }
    else if (inch >= 4)
    {
        if (size >= 4)
            tmp.create(4 * maxk, inch / 4 + inch % 4, size / 4 + (size % 4) / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
    {
        if (size >= 4)
            tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    }
#endif // __ARM_FEATURE_DOTPROD
#else  // __aarch64__
    if (inch >= 8)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size / 2 + size % 2, 8u, 8, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 8 + (inch % 8) / 4 + inch % 4, size, 8u, 8, opt.workspace_allocator);
    }
    else if (inch >= 4)
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch / 4 + inch % 4, size / 2 + size % 2, 4u, 4, opt.workspace_allocator);
        else
            tmp.create(maxk, inch / 4 + inch % 4, size, 4u, 4, opt.workspace_allocator);
    }
    else
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    }
#endif // __aarch64__
#else  // __ARM_NEON
    {
        if (size >= 2)
            tmp.create(2 * maxk, inch, size / 2 + size % 2, 1u, 1, opt.workspace_allocator);
        else
            tmp.create(maxk, inch, size, 1u, 1, opt.workspace_allocator);
    }
#endif // __ARM_NEON
    {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        int nn_size = size >> 3;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            signed char* tmpptr = tmp.channel(i / 8);

            int q = 0;
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    asm volatile(
                        "ld1    {v0.8b}, [%0]               \n"
                        "ld1    {v1.8b}, [%1]               \n"
                        "ld1    {v2.8b}, [%2]               \n"
                        "ld1    {v3.8b}, [%3]               \n"
                        "ld1    {v4.8b}, [%4]               \n"
                        "ld1    {v5.8b}, [%5]               \n"
                        "ld1    {v6.8b}, [%6]               \n"
                        "ld1    {v7.8b}, [%7]               \n"
                        "zip1   v8.8b, v0.8b, v4.8b         \n"
                        "zip1   v9.8b, v1.8b, v5.8b         \n"
                        "zip1   v10.8b, v2.8b, v6.8b        \n"
                        "zip1   v11.8b, v3.8b, v7.8b        \n"
                        "zip2   v0.8b, v0.8b, v4.8b         \n"
                        "zip2   v1.8b, v1.8b, v5.8b         \n"
                        "zip2   v2.8b, v2.8b, v6.8b         \n"
                        "zip2   v3.8b, v3.8b, v7.8b         \n"
                        "st4    {v8.8b, v9.8b, v10.8b, v11.8b}, [%8], #32 \n"
                        "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%8], #32 \n"
                        : "=r"(img0), // %0
                        "=r"(img1),
                        "=r"(img2),
                        "=r"(img3),
                        "=r"(img4),
                        "=r"(img5),
                        "=r"(img6),
                        "=r"(img7),
                        "=r"(tmpptr) // %8
                        : "0"(img0),
                        "1"(img1),
                        "2"(img2),
                        "3"(img3),
                        "4"(img4),
                        "5"(img5),
                        "6"(img6),
                        "7"(img7),
                        "8"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else  // __ARM_FEATURE_MATMUL_INT8
                    asm volatile(
                        "ld1    {v0.8b}, [%0]               \n"
                        "ld1    {v1.8b}, [%1]               \n"
                        "ld1    {v2.8b}, [%2]               \n"
                        "ld1    {v3.8b}, [%3]               \n"
                        "ld1    {v4.8b}, [%4]               \n"
                        "ld1    {v5.8b}, [%5]               \n"
                        "ld1    {v6.8b}, [%6]               \n"
                        "ld1    {v7.8b}, [%7]               \n"
                        "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%8], #32 \n"
                        "st4    {v4.8b, v5.8b, v6.8b, v7.8b}, [%8], #32 \n"
                        : "=r"(img0), // %0
                        "=r"(img1),
                        "=r"(img2),
                        "=r"(img3),
                        "=r"(img4),
                        "=r"(img5),
                        "=r"(img6),
                        "=r"(img7),
                        "=r"(tmpptr) // %8
                        : "0"(img0),
                        "1"(img1),
                        "2"(img2),
                        "3"(img3),
                        "4"(img4),
                        "5"(img5),
                        "6"(img6),
                        "7"(img7),
                        "8"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#endif // __ARM_FEATURE_MATMUL_INT8
                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "ld1    {v0.8b}, [%0]               \n"
                        "ld1    {v1.8b}, [%1]               \n"
                        "ld1    {v2.8b}, [%2]               \n"
                        "ld1    {v3.8b}, [%3]               \n"
                        "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%4], #32 \n"
                        : "=r"(img0), // %0
                        "=r"(img1),
                        "=r"(img2),
                        "=r"(img3),
                        "=r"(tmpptr) // %4
                        : "0"(img0),
                        "1"(img1),
                        "2"(img2),
                        "3"(img3),
                        "4"(tmpptr)
                        : "memory", "v0", "v1", "v2", "v3");
                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #64]    \n"
                        "ld1    {v0.8b}, [%0]           \n"
                        "st1    {v0.8b}, [%1], #8       \n"
                        : "=r"(img0),  // %0
                        "=r"(tmpptr) // %1
                        : "0"(img0),
                        "1"(tmpptr)
                        : "memory", "v0");
                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;
#else  // __ARM_FEATURE_DOTPROD
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 2;
#endif // __ARM_FEATURE_DOTPROD

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

#if __ARM_FEATURE_DOTPROD
            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#else
            signed char* tmpptr = tmp.channel(i / 4);
#endif

            int q = 0;
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    tmpptr[0] = img0[1];
                    tmpptr[1] = img1[1];
                    tmpptr[2] = img2[1];
                    tmpptr[3] = img3[1];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;

                    tmpptr[0] = img0[2];
                    tmpptr[1] = img1[2];
                    tmpptr[2] = img2[2];
                    tmpptr[3] = img3[2];
                    tmpptr[4] = img4[2];
                    tmpptr[5] = img5[2];
                    tmpptr[6] = img6[2];
                    tmpptr[7] = img7[2];
                    tmpptr += 8;

                    tmpptr[0] = img0[3];
                    tmpptr[1] = img1[3];
                    tmpptr[2] = img2[3];
                    tmpptr[3] = img3[3];
                    tmpptr[4] = img4[3];
                    tmpptr[5] = img5[3];
                    tmpptr[6] = img6[3];
                    tmpptr[7] = img7[3];
                    tmpptr += 8;
#elif __ARM_FEATURE_DOTPROD
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    tmpptr[0] = img0[2];
                    tmpptr[1] = img1[2];
                    tmpptr[2] = img2[2];
                    tmpptr[3] = img3[2];
                    tmpptr[4] = img0[3];
                    tmpptr[5] = img1[3];
                    tmpptr[6] = img2[3];
                    tmpptr[7] = img3[3];
                    tmpptr += 8;

                    tmpptr[0] = img4[0];
                    tmpptr[1] = img5[0];
                    tmpptr[2] = img6[0];
                    tmpptr[3] = img7[0];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;

                    tmpptr[0] = img4[2];
                    tmpptr[1] = img5[2];
                    tmpptr[2] = img6[2];
                    tmpptr[3] = img7[2];
                    tmpptr[4] = img4[3];
                    tmpptr[5] = img5[3];
                    tmpptr[6] = img6[3];
                    tmpptr[7] = img7[3];
                    tmpptr += 8;
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    tmpptr[0] = img0[1];
                    tmpptr[1] = img1[1];
                    tmpptr[2] = img2[1];
                    tmpptr[3] = img3[1];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;

                    tmpptr[0] = img0[2];
                    tmpptr[1] = img1[2];
                    tmpptr[2] = img2[2];
                    tmpptr[3] = img3[2];
                    tmpptr[4] = img4[2];
                    tmpptr[5] = img5[2];
                    tmpptr[6] = img6[2];
                    tmpptr[7] = img7[2];
                    tmpptr += 8;

                    tmpptr[0] = img0[3];
                    tmpptr[1] = img1[3];
                    tmpptr[2] = img2[3];
                    tmpptr[3] = img3[3];
                    tmpptr[4] = img4[3];
                    tmpptr[5] = img5[3];
                    tmpptr[6] = img6[3];
                    tmpptr[7] = img7[3];
                    tmpptr += 8;
#endif

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    tmpptr[0] = img0[2];
                    tmpptr[1] = img1[2];
                    tmpptr[2] = img2[2];
                    tmpptr[3] = img3[2];
                    tmpptr[4] = img0[3];
                    tmpptr[5] = img1[3];
                    tmpptr[6] = img2[3];
                    tmpptr[7] = img3[3];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];

                    tmpptr += 4;

                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
#else  // __aarch64__
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;
#endif // __aarch64__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
#else
            signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
#endif
#else
            signed char* tmpptr = tmp.channel(i / 2);
#endif

            int q = 0;
#if __ARM_NEON
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __ARM_FEATURE_MATMUL_INT8
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    tmpptr[0] = img0[1];
                    tmpptr[1] = img1[1];
                    tmpptr[2] = img2[1];
                    tmpptr[3] = img3[1];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;
#elif __ARM_FEATURE_DOTPROD
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    tmpptr[0] = img4[0];
                    tmpptr[1] = img5[0];
                    tmpptr[2] = img6[0];
                    tmpptr[3] = img7[0];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    tmpptr[0] = img0[1];
                    tmpptr[1] = img1[1];
                    tmpptr[2] = img2[1];
                    tmpptr[3] = img3[1];
                    tmpptr[4] = img4[1];
                    tmpptr[5] = img5[1];
                    tmpptr[6] = img6[1];
                    tmpptr[7] = img7[1];
                    tmpptr += 8;
#endif

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img0[1];
                    tmpptr[5] = img1[1];
                    tmpptr[6] = img2[1];
                    tmpptr[7] = img3[1];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];

                    tmpptr += 2;

                    img0 += size;
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
#else
            signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
#endif
#else
            signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#endif

            int q = 0;
#if __ARM_NEON
            for (; q + 7 < inch; q += 8)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;
                const signed char* img4 = (const signed char*)bottom_im2col.channel(q + 4) + i;
                const signed char* img5 = (const signed char*)bottom_im2col.channel(q + 5) + i;
                const signed char* img6 = (const signed char*)bottom_im2col.channel(q + 6) + i;
                const signed char* img7 = (const signed char*)bottom_im2col.channel(q + 7) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr[4] = img4[0];
                    tmpptr[5] = img5[0];
                    tmpptr[6] = img6[0];
                    tmpptr[7] = img7[0];
                    tmpptr += 8;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                    img4 += size;
                    img5 += size;
                    img6 += size;
                    img7 += size;
                }
            }
            for (; q + 3 < inch; q += 4)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;
                const signed char* img1 = (const signed char*)bottom_im2col.channel(q + 1) + i;
                const signed char* img2 = (const signed char*)bottom_im2col.channel(q + 2) + i;
                const signed char* img3 = (const signed char*)bottom_im2col.channel(q + 3) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img1[0];
                    tmpptr[2] = img2[0];
                    tmpptr[3] = img3[0];
                    tmpptr += 4;

                    img0 += size;
                    img1 += size;
                    img2 += size;
                    img3 += size;
                }
            }
#endif // __ARM_NEON
            for (; q < inch; q++)
            {
                const signed char* img0 = (const signed char*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];

                    tmpptr += 1;

                    img0 += size;
                }
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
    nn_outch = outch / 8;
    remain_outch_start = nn_outch * 8;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);
        int* outptr4 = top_blob.channel(p + 4);
        int* outptr5 = top_blob.channel(p + 5);
        int* outptr6 = top_blob.channel(p + 6);
        int* outptr7 = top_blob.channel(p + 7);

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr0 = kernel.channel(p / 8);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b   \n"
                "eor    v17.16b, v17.16b, v17.16b   \n"
                "eor    v18.16b, v18.16b, v18.16b   \n"
                "eor    v19.16b, v19.16b, v19.16b   \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"
                "eor    v21.16b, v21.16b, v21.16b   \n"
                "eor    v22.16b, v22.16b, v22.16b   \n"
                "eor    v23.16b, v23.16b, v23.16b   \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"
                "eor    v25.16b, v25.16b, v25.16b   \n"
                "eor    v26.16b, v26.16b, v26.16b   \n"
                "eor    v27.16b, v27.16b, v27.16b   \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "cmp    %w8, #0                     \n"
                "beq    1f                          \n"

#if __ARM_FEATURE_MATMUL_INT8
                "eor    v4.16b, v4.16b, v4.16b      \n"
                "eor    v5.16b, v5.16b, v5.16b      \n"
                "eor    v6.16b, v6.16b, v6.16b      \n"
                "eor    v7.16b, v7.16b, v7.16b      \n"
                "eor    v12.16b, v12.16b, v12.16b   \n"
                "eor    v13.16b, v13.16b, v13.16b   \n"
                "eor    v14.16b, v14.16b, v14.16b   \n"
                "eor    v15.16b, v15.16b, v15.16b   \n"

                "0:                                 \n"

                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%11], #64 \n"   // _val0 _val1 _val2 _val3
                "ld1    {v8.16b, v9.16b, v10.16b, v11.16b}, [%12], #64 \n" // _w01 _w23 _w45 _w67

                "smmla  v4.4s, v0.16b, v8.16b       \n"
                "smmla  v17.4s, v0.16b, v9.16b      \n"
                "smmla  v5.4s, v1.16b, v8.16b       \n"
                "smmla  v19.4s, v1.16b, v9.16b      \n"
                "smmla  v6.4s, v2.16b, v8.16b       \n"
                "smmla  v21.4s, v2.16b, v9.16b      \n"
                "smmla  v7.4s, v3.16b, v8.16b       \n"
                "smmla  v23.4s, v3.16b, v9.16b      \n"

                "subs   %w8, %w8, #1                \n"

                "smmla  v12.4s, v0.16b, v10.16b     \n"
                "smmla  v25.4s, v0.16b, v11.16b     \n"
                "smmla  v13.4s, v1.16b, v10.16b     \n"
                "smmla  v27.4s, v1.16b, v11.16b     \n"
                "smmla  v14.4s, v2.16b, v10.16b     \n"
                "smmla  v29.4s, v2.16b, v11.16b     \n"
                "smmla  v15.4s, v3.16b, v10.16b     \n"
                "smmla  v31.4s, v3.16b, v11.16b     \n"

                "bne    0b                          \n"

                "trn1  v16.2d, v4.2d, v17.2d        \n"
                "trn2  v17.2d, v4.2d, v17.2d        \n"
                "trn1  v18.2d, v5.2d, v19.2d        \n"
                "trn2  v19.2d, v5.2d, v19.2d        \n"
                "trn1  v20.2d, v6.2d, v21.2d        \n"
                "trn2  v21.2d, v6.2d, v21.2d        \n"
                "trn1  v22.2d, v7.2d, v23.2d        \n"
                "trn2  v23.2d, v7.2d, v23.2d        \n"

                "trn1  v24.2d, v12.2d, v25.2d       \n"
                "trn2  v25.2d, v12.2d, v25.2d       \n"
                "trn1  v26.2d, v13.2d, v27.2d       \n"
                "trn2  v27.2d, v13.2d, v27.2d       \n"
                "trn1  v28.2d, v14.2d, v29.2d       \n"
                "trn2  v29.2d, v14.2d, v29.2d       \n"
                "trn1  v30.2d, v15.2d, v31.2d       \n"
                "trn2  v31.2d, v15.2d, v31.2d       \n"
#else  // __ARM_FEATURE_MATMUL_INT8
                "0:                                 \n"

                "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%11], #64 \n"   // _val0123_l _val4567_l _val0123_h _val4567_h
                "ld1    {v8.16b, v9.16b, v10.16b, v11.16b}, [%12], #64 \n" // _w0123_l _w0123_h _w4567_l _w4567_h

                "sdot   v16.4s, v8.16b, v0.4b[0]    \n"
                "sdot   v17.4s, v8.16b, v0.4b[1]    \n"
                "sdot   v18.4s, v8.16b, v0.4b[2]    \n"
                "sdot   v19.4s, v8.16b, v0.4b[3]    \n"
                "sdot   v20.4s, v8.16b, v1.4b[0]    \n"
                "sdot   v21.4s, v8.16b, v1.4b[1]    \n"
                "sdot   v22.4s, v8.16b, v1.4b[2]    \n"
                "sdot   v23.4s, v8.16b, v1.4b[3]    \n"

                "sdot   v16.4s, v9.16b, v2.4b[0]    \n"
                "sdot   v17.4s, v9.16b, v2.4b[1]    \n"
                "sdot   v18.4s, v9.16b, v2.4b[2]    \n"
                "sdot   v19.4s, v9.16b, v2.4b[3]    \n"
                "sdot   v20.4s, v9.16b, v3.4b[0]    \n"
                "sdot   v21.4s, v9.16b, v3.4b[1]    \n"
                "sdot   v22.4s, v9.16b, v3.4b[2]    \n"
                "sdot   v23.4s, v9.16b, v3.4b[3]    \n"

                "subs   %w8, %w8, #1                \n"

                "sdot   v24.4s, v10.16b, v0.4b[0]   \n"
                "sdot   v25.4s, v10.16b, v0.4b[1]   \n"
                "sdot   v26.4s, v10.16b, v0.4b[2]   \n"
                "sdot   v27.4s, v10.16b, v0.4b[3]   \n"
                "sdot   v28.4s, v10.16b, v1.4b[0]   \n"
                "sdot   v29.4s, v10.16b, v1.4b[1]   \n"
                "sdot   v30.4s, v10.16b, v1.4b[2]   \n"
                "sdot   v31.4s, v10.16b, v1.4b[3]   \n"

                "sdot   v24.4s, v11.16b, v2.4b[0]   \n"
                "sdot   v25.4s, v11.16b, v2.4b[1]   \n"
                "sdot   v26.4s, v11.16b, v2.4b[2]   \n"
                "sdot   v27.4s, v11.16b, v2.4b[3]   \n"
                "sdot   v28.4s, v11.16b, v3.4b[0]   \n"
                "sdot   v29.4s, v11.16b, v3.4b[1]   \n"
                "sdot   v30.4s, v11.16b, v3.4b[2]   \n"
                "sdot   v31.4s, v11.16b, v3.4b[3]   \n"

                "bne    0b                          \n"
#endif // __ARM_FEATURE_MATMUL_INT8
                "1:                                 \n"

                "cmp    %w9, #0                     \n"
                "beq    3f                          \n"

                "2:                                 \n"

                "ld1    {v0.16b, v1.16b}, [%11], #32 \n" // _val0123 _val4567
                "ld1    {v8.16b, v9.16b}, [%12], #32 \n" // _w0 _w1

                "sdot   v16.4s, v8.16b, v0.4b[0]    \n"
                "sdot   v17.4s, v8.16b, v0.4b[1]    \n"
                "sdot   v18.4s, v8.16b, v0.4b[2]    \n"
                "sdot   v19.4s, v8.16b, v0.4b[3]    \n"
                "sdot   v20.4s, v8.16b, v1.4b[0]    \n"
                "sdot   v21.4s, v8.16b, v1.4b[1]    \n"
                "sdot   v22.4s, v8.16b, v1.4b[2]    \n"
                "sdot   v23.4s, v8.16b, v1.4b[3]    \n"

                "subs   %w9, %w9, #1                \n"

                "sdot   v24.4s, v9.16b, v0.4b[0]    \n"
                "sdot   v25.4s, v9.16b, v0.4b[1]    \n"
                "sdot   v26.4s, v9.16b, v0.4b[2]    \n"
                "sdot   v27.4s, v9.16b, v0.4b[3]    \n"
                "sdot   v28.4s, v9.16b, v1.4b[0]    \n"
                "sdot   v29.4s, v9.16b, v1.4b[1]    \n"
                "sdot   v30.4s, v9.16b, v1.4b[2]    \n"
                "sdot   v31.4s, v9.16b, v1.4b[3]    \n"

                "bne    2b                          \n"

                "3:                                 \n"

                "lsr    w4, %w10, #2                 \n" // w4 = nn1 >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"

                "ld2    {v0.4s, v1.4s}, [%11], #32   \n"
                "ld2    {v8.4s, v9.4s}, [%12], #32   \n"

                "uzp1   v2.16b, v0.16b, v1.16b      \n"
                "uzp2   v3.16b, v0.16b, v1.16b      \n"
                "uzp1   v0.16b, v2.16b, v3.16b      \n"
                "uzp2   v1.16b, v2.16b, v3.16b      \n"
                "uzp1   v2.4s, v0.4s, v1.4s         \n" // _val0123
                "uzp2   v3.4s, v0.4s, v1.4s         \n" // _val4567

                "uzp1   v10.16b, v8.16b, v9.16b     \n"
                "uzp2   v11.16b, v8.16b, v9.16b     \n"
                "uzp1   v8.16b, v10.16b, v11.16b    \n"
                "uzp2   v9.16b, v10.16b, v11.16b    \n"
                "uzp1   v10.4s, v8.4s, v9.4s        \n" // _w0123f
                "uzp2   v11.4s, v8.4s, v9.4s        \n" // _w4567f

                "sdot   v16.4s, v10.16b, v2.4b[0]   \n"
                "sdot   v17.4s, v10.16b, v2.4b[1]   \n"
                "sdot   v18.4s, v10.16b, v2.4b[2]   \n"
                "sdot   v19.4s, v10.16b, v2.4b[3]   \n"
                "sdot   v20.4s, v10.16b, v3.4b[0]   \n"
                "sdot   v21.4s, v10.16b, v3.4b[1]   \n"
                "sdot   v22.4s, v10.16b, v3.4b[2]   \n"
                "sdot   v23.4s, v10.16b, v3.4b[3]   \n"

                "subs   w4, w4, #1                  \n"

                "sdot   v24.4s, v11.16b, v2.4b[0]   \n"
                "sdot   v25.4s, v11.16b, v2.4b[1]   \n"
                "sdot   v26.4s, v11.16b, v2.4b[2]   \n"
                "sdot   v27.4s, v11.16b, v2.4b[3]   \n"
                "sdot   v28.4s, v11.16b, v3.4b[0]   \n"
                "sdot   v29.4s, v11.16b, v3.4b[1]   \n"
                "sdot   v30.4s, v11.16b, v3.4b[2]   \n"
                "sdot   v31.4s, v11.16b, v3.4b[3]   \n"

                "bne    4b                          \n"

                "5:                                 \n"

                "and    w4, %w10, #3                 \n" // w4 = remain = nn1 & 3
                "cmp    w4, #0                      \n"  // w4 > 0
                "beq    7f                          \n"

                "6:                                 \n"

                "ld1    {v0.8b}, [%11], #8           \n"
                "ld1    {v1.8b}, [%12], #8           \n"

                "sshll  v0.8h, v0.8b, #0            \n"

                "sshll  v1.8h, v1.8b, #0            \n"

                "smlal  v16.4s, v1.4h, v0.h[0]      \n"
                "smlal  v17.4s, v1.4h, v0.h[1]      \n"
                "smlal  v18.4s, v1.4h, v0.h[2]      \n"
                "smlal  v19.4s, v1.4h, v0.h[3]      \n"
                "smlal  v20.4s, v1.4h, v0.h[4]      \n"
                "smlal  v21.4s, v1.4h, v0.h[5]      \n"
                "smlal  v22.4s, v1.4h, v0.h[6]      \n"
                "smlal  v23.4s, v1.4h, v0.h[7]      \n"

                "subs   w4, w4, #1                  \n"

                "smlal2 v24.4s, v1.8h, v0.h[0]      \n"
                "smlal2 v25.4s, v1.8h, v0.h[1]      \n"
                "smlal2 v26.4s, v1.8h, v0.h[2]      \n"
                "smlal2 v27.4s, v1.8h, v0.h[3]      \n"
                "smlal2 v28.4s, v1.8h, v0.h[4]      \n"
                "smlal2 v29.4s, v1.8h, v0.h[5]      \n"
                "smlal2 v30.4s, v1.8h, v0.h[6]      \n"
                "smlal2 v31.4s, v1.8h, v0.h[7]      \n"

                "bne    6b                          \n"

                "7:                                 \n"

                "trn1   v0.4s, v16.4s, v17.4s      \n"
                "trn2   v1.4s, v16.4s, v17.4s      \n"
                "trn1   v2.4s, v18.4s, v19.4s      \n"
                "trn2   v3.4s, v18.4s, v19.4s      \n"
                "trn1   v4.4s, v20.4s, v21.4s      \n"
                "trn2   v5.4s, v20.4s, v21.4s      \n"
                "trn1   v6.4s, v22.4s, v23.4s      \n"
                "trn2   v7.4s, v22.4s, v23.4s      \n"

                "trn1   v16.2d, v0.2d, v2.2d       \n"
                "trn1   v18.2d, v1.2d, v3.2d       \n"
                "trn2   v20.2d, v0.2d, v2.2d       \n"
                "trn2   v22.2d, v1.2d, v3.2d       \n"
                "trn1   v17.2d, v4.2d, v6.2d       \n"
                "trn1   v19.2d, v5.2d, v7.2d       \n"
                "trn2   v21.2d, v4.2d, v6.2d       \n"
                "trn2   v23.2d, v5.2d, v7.2d       \n"

                "trn1   v0.4s, v24.4s, v25.4s      \n"
                "trn2   v1.4s, v24.4s, v25.4s      \n"
                "trn1   v2.4s, v26.4s, v27.4s      \n"
                "trn2   v3.4s, v26.4s, v27.4s      \n"
                "trn1   v4.4s, v28.4s, v29.4s      \n"
                "trn2   v5.4s, v28.4s, v29.4s      \n"
                "trn1   v6.4s, v30.4s, v31.4s      \n"
                "trn2   v7.4s, v30.4s, v31.4s      \n"

                "trn1   v24.2d, v0.2d, v2.2d       \n"
                "trn1   v26.2d, v1.2d, v3.2d       \n"
                "trn2   v28.2d, v0.2d, v2.2d       \n"
                "trn2   v30.2d, v1.2d, v3.2d       \n"
                "trn1   v25.2d, v4.2d, v6.2d       \n"
                "trn1   v27.2d, v5.2d, v7.2d       \n"
                "trn2   v29.2d, v4.2d, v6.2d       \n"
                "trn2   v31.2d, v5.2d, v7.2d       \n"

                "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                "st1    {v18.4s, v19.4s}, [%1], #32 \n"
                "st1    {v20.4s, v21.4s}, [%2], #32 \n"
                "st1    {v22.4s, v23.4s}, [%3], #32 \n"
                "st1    {v24.4s, v25.4s}, [%4], #32 \n"
                "st1    {v26.4s, v27.4s}, [%5], #32 \n"
                "st1    {v28.4s, v29.4s}, [%6], #32 \n"
                "st1    {v30.4s, v31.4s}, [%7], #32 \n"
                : "=r"(outptr0),
                "=r"(outptr1),
                "=r"(outptr2),
                "=r"(outptr3),
                "=r"(outptr4),
                "=r"(outptr5),
                "=r"(outptr6),
                "=r"(outptr7),
                "=r"(nn),
                "=r"(nn4),
                "=r"(nn1),
                "=r"(tmpptr),
                "=r"(kptr0)
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(outptr4),
                "5"(outptr5),
                "6"(outptr6),
                "7"(outptr7),
                "8"(nn),
                "9"(nn4),
                "10"(nn1),
                "11"(tmpptr),
                "12"(kptr0)
                : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        }
        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr0 = kernel.channel(p / 8);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);
            int32x4_t _sum4 = vdupq_n_s32(0);
            int32x4_t _sum5 = vdupq_n_s32(0);
            int32x4_t _sum6 = vdupq_n_s32(0);
            int32x4_t _sum7 = vdupq_n_s32(0);

#if __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0 = vld1q_s8(tmpptr);
                int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);
                int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                _sum0 = vmmlaq_s32(_sum0, _val0, _w01);
                _sum1 = vmmlaq_s32(_sum1, _val0, _w23);
                _sum2 = vmmlaq_s32(_sum2, _val1, _w01);
                _sum3 = vmmlaq_s32(_sum3, _val1, _w23);

                _sum4 = vmmlaq_s32(_sum4, _val0, _w45);
                _sum5 = vmmlaq_s32(_sum5, _val0, _w67);
                _sum6 = vmmlaq_s32(_sum6, _val1, _w45);
                _sum7 = vmmlaq_s32(_sum7, _val1, _w67);

                tmpptr += 32;
                kptr0 += 64;
            }

            int32x4_t _sum0x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum1x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum2x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum3x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum4x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum4), vreinterpretq_s64_s32(_sum5)));
            int32x4_t _sum5x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum4), vreinterpretq_s64_s32(_sum5)));
            int32x4_t _sum6x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum6), vreinterpretq_s64_s32(_sum7)));
            int32x4_t _sum7x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum6), vreinterpretq_s64_s32(_sum7)));

            _sum0 = _sum0x;
            _sum1 = _sum1x;
            _sum2 = _sum2x;
            _sum3 = _sum3x;
            _sum4 = _sum4x;
            _sum5 = _sum5x;
            _sum6 = _sum6x;
            _sum7 = _sum7x;
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0123_l = vld1q_s8(tmpptr);
                int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                int8x16_t _w0123_l = vld1q_s8(kptr0);
                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);
                int8x16_t _w4567_l = vld1q_s8(kptr0 + 32);
                int8x16_t _w4567_h = vld1q_s8(kptr0 + 48);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
                _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

                _sum4 = vdotq_laneq_s32(_sum4, _w4567_l, _val0123_l, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w4567_l, _val0123_l, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w4567_l, _val0123_l, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w4567_l, _val0123_l, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _w4567_h, _val0123_h, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w4567_h, _val0123_h, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w4567_h, _val0123_h, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w4567_h, _val0123_h, 3);

                tmpptr += 32;
                kptr0 += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            for (int j = 0; j < nn4; j++)
            {
                int8x16_t _val0123 = vld1q_s8(tmpptr);
                int8x16_t _w0 = vld1q_s8(kptr0);
                int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);

                _sum4 = vdotq_laneq_s32(_sum4, _w1, _val0123, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w1, _val0123, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w1, _val0123, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w1, _val0123, 3);

                tmpptr += 16;
                kptr0 += 32;
            }

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                // 0123 0123 0123 0123  ->  0000111122223333
                int8x16_t _val = vld1q_s8(tmpptr);

                int8x8x2_t _val01 = vuzp_s8(vget_low_s8(_val), vget_high_s8(_val));
                int8x8x2_t _val0123 = vuzp_s8(_val01.val[0], _val01.val[1]);
                int8x16_t _val0123f = vcombine_s8(_val0123.val[0], _val0123.val[1]);

                // 0123 4567 0123 4567 0123 4567 0123 4567  ->  0000111122223333
                int32x4x2_t _w = vld2q_s32((const int*)kptr0);

                int8x16_t _w0 = vreinterpretq_s8_s32(_w.val[0]);
                int8x16_t _w1 = vreinterpretq_s8_s32(_w.val[1]);

                int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w0), vget_high_s8(_w0));
                int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
                int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

                int8x8x2_t _w45 = vuzp_s8(vget_low_s8(_w1), vget_high_s8(_w1));
                int8x8x2_t _w4567 = vuzp_s8(_w45.val[0], _w45.val[1]);
                int8x16_t _w4567f = vcombine_s8(_w4567.val[0], _w4567.val[1]);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123f, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123f, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123f, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123f, 3);

                _sum4 = vdotq_laneq_s32(_sum4, _w4567f, _val0123f, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w4567f, _val0123f, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w4567f, _val0123f, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w4567f, _val0123f, 3);

                tmpptr += 16;
                kptr0 += 32;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
                int16x4_t _val3 = vdup_n_s16(tmpptr[3]);

                int16x8_t _w01 = vmovl_s8(vld1_s8(kptr0));

                _sum0 = vmlal_s16(_sum0, _val0, vget_low_s16(_w01));
                _sum1 = vmlal_s16(_sum1, _val1, vget_low_s16(_w01));
                _sum2 = vmlal_s16(_sum2, _val2, vget_low_s16(_w01));
                _sum3 = vmlal_s16(_sum3, _val3, vget_low_s16(_w01));

                _sum4 = vmlal_s16(_sum4, _val0, vget_high_s16(_w01));
                _sum5 = vmlal_s16(_sum5, _val1, vget_high_s16(_w01));
                _sum6 = vmlal_s16(_sum6, _val2, vget_high_s16(_w01));
                _sum7 = vmlal_s16(_sum7, _val3, vget_high_s16(_w01));

                tmpptr += 4;
                kptr0 += 8;
            }

            // transpose 4x4
            int32x4_t _sum01_0 = vtrn1q_s32(_sum0, _sum1);
            int32x4_t _sum01_1 = vtrn2q_s32(_sum0, _sum1);
            int32x4_t _sum23_0 = vtrn1q_s32(_sum2, _sum3);
            int32x4_t _sum23_1 = vtrn2q_s32(_sum2, _sum3);
            int32x4_t _sum45_0 = vtrn1q_s32(_sum4, _sum5);
            int32x4_t _sum45_1 = vtrn2q_s32(_sum4, _sum5);
            int32x4_t _sum67_0 = vtrn1q_s32(_sum6, _sum7);
            int32x4_t _sum67_1 = vtrn2q_s32(_sum6, _sum7);
            _sum0 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum01_0), vreinterpretq_s64_s32(_sum23_0)));
            _sum1 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum01_1), vreinterpretq_s64_s32(_sum23_1)));
            _sum2 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum01_0), vreinterpretq_s64_s32(_sum23_0)));
            _sum3 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum01_1), vreinterpretq_s64_s32(_sum23_1)));
            _sum4 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum45_0), vreinterpretq_s64_s32(_sum67_0)));
            _sum5 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum45_1), vreinterpretq_s64_s32(_sum67_1)));
            _sum6 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum45_0), vreinterpretq_s64_s32(_sum67_0)));
            _sum7 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum45_1), vreinterpretq_s64_s32(_sum67_1)));

            vst1q_s32(outptr0, _sum0);
            vst1q_s32(outptr1, _sum1);
            vst1q_s32(outptr2, _sum2);
            vst1q_s32(outptr3, _sum3);
            vst1q_s32(outptr4, _sum4);
            vst1q_s32(outptr5, _sum5);
            vst1q_s32(outptr6, _sum6);
            vst1q_s32(outptr7, _sum7);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const signed char* kptr0 = kernel.channel(p / 8);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);

#if __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val = vld1q_s8(tmpptr);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);
                int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                _sum0 = vmmlaq_s32(_sum0, _val, _w01);
                _sum1 = vmmlaq_s32(_sum1, _val, _w23);
                _sum2 = vmmlaq_s32(_sum2, _val, _w45);
                _sum3 = vmmlaq_s32(_sum3, _val, _w67);

                tmpptr += 16;
                kptr0 += 64;
            }

            int32x4_t _sum0x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum1x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum2x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum3x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));

            _sum0 = _sum0x;
            _sum1 = _sum1x;
            _sum2 = _sum2x;
            _sum3 = _sum3x;
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val01_l_h = vld1q_s8(tmpptr);
                int8x16_t _w0123_l = vld1q_s8(kptr0);
                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);
                int8x16_t _w4567_l = vld1q_s8(kptr0 + 32);
                int8x16_t _w4567_h = vld1q_s8(kptr0 + 48);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val01_l_h, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val01_l_h, 1);
                _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val01_l_h, 2);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val01_l_h, 3);

                _sum2 = vdotq_laneq_s32(_sum2, _w4567_l, _val01_l_h, 0);
                _sum3 = vdotq_laneq_s32(_sum3, _w4567_l, _val01_l_h, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w4567_h, _val01_l_h, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w4567_h, _val01_l_h, 3);

                tmpptr += 16;
                kptr0 += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            if (nn4 > 0)
            {
                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x16_t _val0123 = vld1q_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);
                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);
                    int8x16_t _w2 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w3 = vld1q_s8(kptr0 + 48);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w1, _val0123, 0);
                    _sum3 = vdotq_laneq_s32(_sum3, _w1, _val0123, 1);

                    _sum0 = vdotq_laneq_s32(_sum0, _w2, _val0123, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _w2, _val0123, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _w3, _val0123, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w3, _val0123, 3);

                    tmpptr += 16;
                    kptr0 += 64;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);
                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w0, _val01, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _w0, _val01, 1);
                    _sum2 = vdotq_lane_s32(_sum2, _w1, _val01, 0);
                    _sum3 = vdotq_lane_s32(_sum3, _w1, _val01, 1);

                    tmpptr += 8;
                    kptr0 += 32;
                }
            }

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int16x8_t _val01234567 = vmovl_s8(vld1_s8(tmpptr));

                int8x16_t _w0 = vld1q_s8(kptr0);
                int8x16_t _w1 = vld1q_s8(kptr0 + 16);
                int16x8_t _w0l = vmovl_s8(vget_low_s8(_w0));
                int16x8_t _w0h = vmovl_s8(vget_high_s8(_w0));
                int16x8_t _w1l = vmovl_s8(vget_low_s8(_w1));
                int16x8_t _w1h = vmovl_s8(vget_high_s8(_w1));

                _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_w0l), _val01234567, 0);
                _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_w0l), _val01234567, 1);
                _sum2 = vmlal_laneq_s16(_sum2, vget_high_s16(_w0l), _val01234567, 0);
                _sum3 = vmlal_laneq_s16(_sum3, vget_high_s16(_w0l), _val01234567, 1);

                _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_w0h), _val01234567, 2);
                _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_w0h), _val01234567, 3);
                _sum2 = vmlal_laneq_s16(_sum2, vget_high_s16(_w0h), _val01234567, 2);
                _sum3 = vmlal_laneq_s16(_sum3, vget_high_s16(_w0h), _val01234567, 3);

                _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_w1l), _val01234567, 4);
                _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_w1l), _val01234567, 5);
                _sum2 = vmlal_laneq_s16(_sum2, vget_high_s16(_w1l), _val01234567, 4);
                _sum3 = vmlal_laneq_s16(_sum3, vget_high_s16(_w1l), _val01234567, 5);

                _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_w1h), _val01234567, 6);
                _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_w1h), _val01234567, 7);
                _sum2 = vmlal_laneq_s16(_sum2, vget_high_s16(_w1h), _val01234567, 6);
                _sum3 = vmlal_laneq_s16(_sum3, vget_high_s16(_w1h), _val01234567, 7);

                tmpptr += 8;
                kptr0 += 32;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                int16x8_t _w01 = vmovl_s8(vld1_s8(kptr0));

                _sum0 = vmlal_s16(_sum0, _val0, vget_low_s16(_w01));
                _sum1 = vmlal_s16(_sum1, _val1, vget_low_s16(_w01));
                _sum2 = vmlal_s16(_sum2, _val0, vget_high_s16(_w01));
                _sum3 = vmlal_s16(_sum3, _val1, vget_high_s16(_w01));

                tmpptr += 2;
                kptr0 += 8;
            }

            int32x4x2_t _sum01 = vzipq_s32(_sum0, _sum1);
            int32x4x2_t _sum23 = vzipq_s32(_sum2, _sum3);

            vst1_s32(outptr0, vget_low_s32(_sum01.val[0]));
            vst1_s32(outptr1, vget_high_s32(_sum01.val[0]));
            vst1_s32(outptr2, vget_low_s32(_sum01.val[1]));
            vst1_s32(outptr3, vget_high_s32(_sum01.val[1]));
            vst1_s32(outptr4, vget_low_s32(_sum23.val[0]));
            vst1_s32(outptr5, vget_high_s32(_sum23.val[0]));
            vst1_s32(outptr6, vget_low_s32(_sum23.val[1]));
            vst1_s32(outptr7, vget_high_s32(_sum23.val[1]));
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
            outptr4 += 2;
            outptr5 += 2;
            outptr6 += 2;
            outptr7 += 2;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 8);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

#if __ARM_FEATURE_MATMUL_INT8
            int32x4_t _sum01 = vdupq_n_s32(0);
            int32x4_t _sum23 = vdupq_n_s32(0);
            int32x4_t _sum45 = vdupq_n_s32(0);
            int32x4_t _sum67 = vdupq_n_s32(0);

            for (int j = 0; j < nn; j++)
            {
                int8x8_t _val0 = vld1_s8(tmpptr);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);
                int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                int8x16_t _val = vcombine_s8(_val0, _val0);

                _sum01 = vdotq_s32(_sum01, _val, _w01);
                _sum23 = vdotq_s32(_sum23, _val, _w23);
                _sum45 = vdotq_s32(_sum45, _val, _w45);
                _sum67 = vdotq_s32(_sum67, _val, _w67);

                tmpptr += 8;
                kptr0 += 64;
            }

            int32x4_t _sum0 = vpaddq_s32(_sum01, _sum23);
            int32x4_t _sum1 = vpaddq_s32(_sum45, _sum67);
#else  // __ARM_FEATURE_MATMUL_INT8
            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);

            for (int j = 0; j < nn; j++)
            {
                int8x8_t _val0_l_h = vld1_s8(tmpptr);
                int8x16_t _w0123_l = vld1q_s8(kptr0);
                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);
                int8x16_t _w4567_l = vld1q_s8(kptr0 + 32);
                int8x16_t _w4567_h = vld1q_s8(kptr0 + 48);

                _sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);
                _sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);
                _sum1 = vdotq_lane_s32(_sum1, _w4567_l, _val0_l_h, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w4567_h, _val0_l_h, 1);

                tmpptr += 8;
                kptr0 += 64;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            if (nn4 > 0)
            {
                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);
                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);
                    int8x16_t _w2 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w3 = vld1q_s8(kptr0 + 48);

                    _sum0 = vdotq_lane_s32(_sum0, _w0, _val01, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _w1, _val01, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _w2, _val01, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _w3, _val01, 1);

                    tmpptr += 8;
                    kptr0 += 64;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val_xxx = vld1_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);
                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w0, _val_xxx, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _w1, _val_xxx, 0);

                    tmpptr += 4;
                    kptr0 += 32;
                }
            }

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int16x4_t _val0123 = vget_low_s16(vmovl_s8(vld1_s8(tmpptr)));

                int8x16_t _w0 = vld1q_s8(kptr0);
                int8x16_t _w1 = vld1q_s8(kptr0 + 16);
                int16x8_t _w0l = vmovl_s8(vget_low_s8(_w0));
                int16x8_t _w0h = vmovl_s8(vget_high_s8(_w0));
                int16x8_t _w1l = vmovl_s8(vget_low_s8(_w1));
                int16x8_t _w1h = vmovl_s8(vget_high_s8(_w1));

                _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0l), _val0123, 0);
                _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0l), _val0123, 0);
                _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w0h), _val0123, 1);
                _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w0h), _val0123, 1);

                _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1l), _val0123, 2);
                _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1l), _val0123, 2);
                _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_w1h), _val0123, 3);
                _sum1 = vmlal_lane_s16(_sum1, vget_high_s16(_w1h), _val0123, 3);

                tmpptr += 4;
                kptr0 += 32;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val = vdup_n_s16(tmpptr[0]);
                int16x8_t _w01 = vmovl_s8(vld1_s8(kptr0));

                _sum0 = vmlal_s16(_sum0, _val, vget_low_s16(_w01));
                _sum1 = vmlal_s16(_sum1, _val, vget_high_s16(_w01));

                tmpptr += 1;
                kptr0 += 8;
            }

            outptr0[0] = vgetq_lane_s32(_sum0, 0);
            outptr1[0] = vgetq_lane_s32(_sum0, 1);
            outptr2[0] = vgetq_lane_s32(_sum0, 2);
            outptr3[0] = vgetq_lane_s32(_sum0, 3);
            outptr4[0] = vgetq_lane_s32(_sum1, 0);
            outptr5[0] = vgetq_lane_s32(_sum1, 1);
            outptr6[0] = vgetq_lane_s32(_sum1, 2);
            outptr7[0] = vgetq_lane_s32(_sum1, 3);
            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
            outptr4 += 1;
            outptr5 += 1;
            outptr6 += 1;
            outptr7 += 1;
        }
    }
#endif // __ARM_FEATURE_DOTPROD

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);

        int i = 0;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);
            int32x4_t _sum4 = vdupq_n_s32(0);
            int32x4_t _sum5 = vdupq_n_s32(0);
            int32x4_t _sum6 = vdupq_n_s32(0);
            int32x4_t _sum7 = vdupq_n_s32(0);

#if __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0 = vld1q_s8(tmpptr);
                int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                int8x16_t _val2 = vld1q_s8(tmpptr + 32);
                int8x16_t _val3 = vld1q_s8(tmpptr + 48);

                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                _sum0 = vmmlaq_s32(_sum0, _val0, _w01);
                _sum1 = vmmlaq_s32(_sum1, _val0, _w23);
                _sum2 = vmmlaq_s32(_sum2, _val1, _w01);
                _sum3 = vmmlaq_s32(_sum3, _val1, _w23);
                _sum4 = vmmlaq_s32(_sum4, _val2, _w01);
                _sum5 = vmmlaq_s32(_sum5, _val2, _w23);
                _sum6 = vmmlaq_s32(_sum6, _val3, _w01);
                _sum7 = vmmlaq_s32(_sum7, _val3, _w23);

                tmpptr += 64;
                kptr0 += 32;
            }

            int32x4_t _sum0x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum1x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum2x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum3x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum4x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum4), vreinterpretq_s64_s32(_sum5)));
            int32x4_t _sum5x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum4), vreinterpretq_s64_s32(_sum5)));
            int32x4_t _sum6x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum6), vreinterpretq_s64_s32(_sum7)));
            int32x4_t _sum7x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum6), vreinterpretq_s64_s32(_sum7)));

            _sum0 = _sum0x;
            _sum1 = _sum1x;
            _sum2 = _sum2x;
            _sum3 = _sum3x;
            _sum4 = _sum4x;
            _sum5 = _sum5x;
            _sum6 = _sum6x;
            _sum7 = _sum7x;
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0123_l = vld1q_s8(tmpptr);
                int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);

                int8x16_t _w0123_l = vld1q_s8(kptr0);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _w0123_l, _val4567_l, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w0123_l, _val4567_l, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w0123_l, _val4567_l, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w0123_l, _val4567_l, 3);

                int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);

                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _w0123_h, _val4567_h, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w0123_h, _val4567_h, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w0123_h, _val4567_h, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w0123_h, _val4567_h, 3);

                tmpptr += 64;
                kptr0 += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            for (int j = 0; j < nn4; j++)
            {
                int8x16_t _val0123 = vld1q_s8(tmpptr);
                int8x16_t _val4567 = vld1q_s8(tmpptr + 16);
                int8x16_t _w0 = vld1q_s8(kptr0);

                _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _w0, _val4567, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w0, _val4567, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w0, _val4567, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w0, _val4567, 3);

                tmpptr += 32;
                kptr0 += 16;
            }

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int8x8x4_t _val4 = vld4_s8(tmpptr);

                int8x8x2_t _val0145 = vuzp_s8(_val4.val[0], _val4.val[1]);
                int8x8x2_t _val2367 = vuzp_s8(_val4.val[2], _val4.val[3]);

                int8x16_t _val0123 = vcombine_s8(_val0145.val[0], _val2367.val[0]);
                int8x16_t _val4567 = vcombine_s8(_val0145.val[1], _val2367.val[1]);

                int8x16_t _w = vld1q_s8(kptr0);

                int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
                int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
                int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123, 3);
                _sum4 = vdotq_laneq_s32(_sum4, _w0123f, _val4567, 0);
                _sum5 = vdotq_laneq_s32(_sum5, _w0123f, _val4567, 1);
                _sum6 = vdotq_laneq_s32(_sum6, _w0123f, _val4567, 2);
                _sum7 = vdotq_laneq_s32(_sum7, _w0123f, _val4567, 3);

                tmpptr += 32;
                kptr0 += 16;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
                int16x4_t _val3 = vdup_n_s16(tmpptr[3]);
                int16x4_t _val4 = vdup_n_s16(tmpptr[4]);
                int16x4_t _val5 = vdup_n_s16(tmpptr[5]);
                int16x4_t _val6 = vdup_n_s16(tmpptr[6]);
                int16x4_t _val7 = vdup_n_s16(tmpptr[7]);

                int16x4_t _w0123;
                _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                _sum0 = vmlal_s16(_sum0, _val0, _w0123);
                _sum1 = vmlal_s16(_sum1, _val1, _w0123);
                _sum2 = vmlal_s16(_sum2, _val2, _w0123);
                _sum3 = vmlal_s16(_sum3, _val3, _w0123);
                _sum4 = vmlal_s16(_sum4, _val4, _w0123);
                _sum5 = vmlal_s16(_sum5, _val5, _w0123);
                _sum6 = vmlal_s16(_sum6, _val6, _w0123);
                _sum7 = vmlal_s16(_sum7, _val7, _w0123);

                tmpptr += 8;
                kptr0 += 4;
            }

            // transpose 4x8
            int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
            int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
            int32x4x2_t _s45 = vtrnq_s32(_sum4, _sum5);
            int32x4x2_t _s67 = vtrnq_s32(_sum6, _sum7);
            _sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
            _sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
            _sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
            _sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));
            _sum4 = vcombine_s32(vget_low_s32(_s45.val[0]), vget_low_s32(_s67.val[0]));
            _sum5 = vcombine_s32(vget_low_s32(_s45.val[1]), vget_low_s32(_s67.val[1]));
            _sum6 = vcombine_s32(vget_high_s32(_s45.val[0]), vget_high_s32(_s67.val[0]));
            _sum7 = vcombine_s32(vget_high_s32(_s45.val[1]), vget_high_s32(_s67.val[1]));

            vst1q_s32(outptr0, _sum0);
            vst1q_s32(outptr1, _sum1);
            vst1q_s32(outptr2, _sum2);
            vst1q_s32(outptr3, _sum3);
            vst1q_s32(outptr0 + 4, _sum4);
            vst1q_s32(outptr1 + 4, _sum5);
            vst1q_s32(outptr2 + 4, _sum6);
            vst1q_s32(outptr3 + 4, _sum7);
            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
        }
#endif
        for (; i + 3 < size; i += 4)
        {
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4);
            const signed char* kptr0 = kernel.channel(p / 4);
#endif

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;
#if __ARM_FEATURE_DOTPROD
            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);

#if __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0 = vld1q_s8(tmpptr);
                int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                _sum0 = vmmlaq_s32(_sum0, _val0, _w01);
                _sum1 = vmmlaq_s32(_sum1, _val0, _w23);
                _sum2 = vmmlaq_s32(_sum2, _val1, _w01);
                _sum3 = vmmlaq_s32(_sum3, _val1, _w23);

                tmpptr += 32;
                kptr0 += 32;
            }

            int32x4_t _sum0x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum1x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum0), vreinterpretq_s64_s32(_sum1)));
            int32x4_t _sum2x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));
            int32x4_t _sum3x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum2), vreinterpretq_s64_s32(_sum3)));

            _sum0 = _sum0x;
            _sum1 = _sum1x;
            _sum2 = _sum2x;
            _sum3 = _sum3x;
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val0123_l = vld1q_s8(tmpptr);
                int8x16_t _w0123_l = vld1q_s8(kptr0);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);

                int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

                tmpptr += 32;
                kptr0 += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            for (int j = 0; j < nn4; j++)
            {
                int8x16_t _val0123 = vld1q_s8(tmpptr);
                int8x16_t _w0 = vld1q_s8(kptr0);

                _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);

                tmpptr += 16;
                kptr0 += 16;
            }

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int8x16_t _val = vld1q_s8(tmpptr);

                int8x8x2_t _val01 = vuzp_s8(vget_low_s8(_val), vget_high_s8(_val));
                int8x8x2_t _val0123 = vuzp_s8(_val01.val[0], _val01.val[1]);
                int8x16_t _val0123f = vcombine_s8(_val0123.val[0], _val0123.val[1]);

                int8x16_t _w = vld1q_s8(kptr0);

                int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
                int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
                int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

                _sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123f, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123f, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123f, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123f, 3);

                tmpptr += 16;
                kptr0 += 16;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
                int16x4_t _val3 = vdup_n_s16(tmpptr[3]);

                int16x4_t _w0123;
                _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                _sum0 = vmlal_s16(_sum0, _val0, _w0123);
                _sum1 = vmlal_s16(_sum1, _val1, _w0123);
                _sum2 = vmlal_s16(_sum2, _val2, _w0123);
                _sum3 = vmlal_s16(_sum3, _val3, _w0123);

                tmpptr += 4;
                kptr0 += 4;
            }

            // transpose 4x4
            int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
            int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
            _sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
            _sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
            _sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
            _sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));

            vst1q_s32(outptr0, _sum0);
            vst1q_s32(outptr1, _sum1);
            vst1q_s32(outptr2, _sum2);
            vst1q_s32(outptr3, _sum3);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
#else  // __ARM_FEATURE_DOTPROD
            asm volatile(
                "eor    v0.16b, v0.16b, v0.16b      \n"
                "eor    v1.16b, v1.16b, v1.16b      \n"
                "eor    v2.16b, v2.16b, v2.16b      \n"
                "eor    v3.16b, v3.16b, v3.16b      \n"

                "cmp    %w4, #0                     \n"
                "beq    3f                          \n"

                "eor    v4.16b, v4.16b, v4.16b      \n"
                "eor    v5.16b, v5.16b, v5.16b      \n"
                "eor    v6.16b, v6.16b, v6.16b      \n"
                "eor    v7.16b, v7.16b, v7.16b      \n"
                "eor    v8.16b, v8.16b, v8.16b      \n"
                "eor    v9.16b, v9.16b, v9.16b      \n"
                "eor    v10.16b, v10.16b, v10.16b   \n"
                "eor    v11.16b, v11.16b, v11.16b   \n"
                "eor    v12.16b, v12.16b, v12.16b   \n"
                "eor    v13.16b, v13.16b, v13.16b   \n"
                "eor    v14.16b, v14.16b, v14.16b   \n"
                "eor    v15.16b, v15.16b, v15.16b   \n"

                "prfm   pldl1keep, [%7, #128]       \n"

                "prfm   pldl1keep, [%8, #256]       \n"

                "lsr    w4, %w4, #1                 \n" // w4 = nn >> 1
                "cmp    w4, #0                      \n"
                "beq    1f                          \n"

                "prfm   pldl1keep, [%8, #512]       \n"

                "add    x5, %7, #16                 \n"

                "prfm   pldl1keep, [x5, #128]       \n"

                "ld1    {v16.16b}, [%7]             \n" // val L H
                "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%8], #64 \n"
                "add    %7, %7, #32                 \n"
                "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                "ld1    {v18.16b}, [%7]             \n"
                "add    %7, %7, #32                 \n"

                "0:                                 \n"

                "smull  v24.8h, v16.8b,  v20.8b     \n"
                "prfm   pldl1keep, [%8, #256]       \n"
                "smull2 v25.8h, v17.16b, v20.16b    \n"
                "prfm   pldl1keep, [%8, #512]       \n"
                "smull  v26.8h, v16.8b,  v21.8b     \n"
                "subs   w4, w4, #1                  \n"
                "smull2 v27.8h, v17.16b, v21.16b    \n"
                "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                "smlal  v24.8h, v18.8b,  v22.8b     \n"
                "smlal2 v25.8h, v19.16b, v22.16b    \n"
                "smlal  v26.8h, v18.8b,  v23.8b     \n"
                "smlal2 v27.8h, v19.16b, v23.16b    \n"

                "smull2 v29.8h, v16.16b, v20.16b    \n"
                "sadalp v0.4s, v24.8h               \n"
                "smull  v28.8h, v17.8b,  v20.8b     \n"
                "sadalp v1.4s, v25.8h               \n"
                "smull2 v31.8h, v16.16b, v21.16b    \n"
                "ld1    {v16.16b}, [x5]             \n" // val L H
                "smull  v30.8h, v17.8b,  v21.8b     \n"
                "add    x5, x5, #32                 \n"
                "smlal2 v29.8h, v18.16b, v22.16b    \n"
                "sadalp v2.4s, v26.8h               \n"
                "smlal  v28.8h, v19.8b,  v22.8b     \n"
                "sadalp v3.4s, v27.8h               \n"
                "smlal2 v31.8h, v18.16b, v23.16b    \n"
                "ld1    {v18.16b}, [x5]             \n"
                "smlal  v30.8h, v19.8b,  v23.8b     \n"
                "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                "smull  v24.8h, v16.8b,  v20.8b     \n"
                "add    x5, x5, #32                 \n"
                "smull2 v25.8h, v17.16b, v20.16b    \n"
                "prfm   pldl1keep, [x5, #128]       \n"
                "smull  v26.8h, v16.8b,  v21.8b     \n"
                "prfm   pldl1keep, [x5, #384]       \n"
                "smull2 v27.8h, v17.16b, v21.16b    \n"
                "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                "smlal  v24.8h, v18.8b,  v22.8b     \n"
                "sadalp v5.4s, v29.8h               \n"
                "smlal2 v25.8h, v19.16b, v22.16b    \n"
                "sadalp v4.4s, v28.8h               \n"
                "smlal  v26.8h, v18.8b,  v23.8b     \n"
                "sadalp v7.4s, v31.8h               \n"
                "smlal2 v27.8h, v19.16b, v23.16b    \n"
                "sadalp v6.4s, v30.8h               \n"

                "smull2 v29.8h, v16.16b, v20.16b    \n"
                "sadalp v8.4s, v24.8h               \n"
                "smull  v28.8h, v17.8b,  v20.8b     \n"
                "sadalp v9.4s, v25.8h               \n"
                "smull2 v31.8h, v16.16b, v21.16b    \n"
                "ld1    {v16.16b}, [%7]             \n" // val L H
                "smull  v30.8h, v17.8b,  v21.8b     \n"
                "add    %7, %7, #32                 \n"
                "smlal2 v29.8h, v18.16b, v22.16b    \n"
                "sadalp v10.4s, v26.8h              \n"
                "smlal  v28.8h, v19.8b,  v22.8b     \n"
                "sadalp v11.4s, v27.8h              \n"
                "smlal2 v31.8h, v18.16b, v23.16b    \n"
                "ld1    {v18.16b}, [%7]             \n"
                "smlal  v30.8h, v19.8b,  v23.8b     \n"
                "add    %7, %7, #32                 \n"
                "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%8], #64 \n"

                "sadalp v13.4s, v29.8h              \n"
                "prfm   pldl1keep, [%7, #128]       \n"
                "sadalp v12.4s, v28.8h              \n"
                "prfm   pldl1keep, [%7, #384]       \n"
                "sadalp v15.4s, v31.8h              \n"
                "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                "sadalp v14.4s, v30.8h              \n"

                "bne    0b                          \n"

                "sub    %7, %7, #64                 \n"
                "sub    %8, %8, #64                 \n"

                "1:                                 \n"
                "and    w4, %w4, #1                 \n" // w4 = remain = nn & 1
                "cmp    w4, #0                      \n" // w4 > 0
                "beq    2f                          \n"

                "ld1    {v16.8b, v17.8b}, [%7], #16 \n"
                "ld1    {v20.8b, v21.8b, v22.8b, v23.8b}, [%8], #32 \n"

                "smull  v24.8h, v16.8b, v20.8b      \n"
                "smull  v25.8h, v16.8b, v21.8b      \n"
                "smull  v26.8h, v16.8b, v22.8b      \n"
                "ld1    {v18.8b, v19.8b}, [%7], #16 \n"
                "smull  v27.8h, v16.8b, v23.8b      \n"
                "sadalp v0.4s, v24.8h               \n"
                "smull  v28.8h, v17.8b, v20.8b      \n"
                "sadalp v1.4s, v25.8h               \n"
                "smull  v29.8h, v17.8b, v21.8b      \n"
                "sadalp v2.4s, v26.8h               \n"
                "smull  v30.8h, v17.8b, v22.8b      \n"
                "sadalp v3.4s, v27.8h               \n"
                "smull  v31.8h, v17.8b, v23.8b      \n"
                "sadalp v4.4s, v28.8h               \n"
                "smull  v24.8h, v18.8b, v20.8b      \n"
                "sadalp v5.4s, v29.8h               \n"
                "smull  v25.8h, v18.8b, v21.8b      \n"
                "sadalp v6.4s, v30.8h               \n"
                "smull  v26.8h, v18.8b, v22.8b      \n"
                "sadalp v7.4s, v31.8h               \n"
                "smull  v27.8h, v18.8b, v23.8b      \n"
                "sadalp v8.4s, v24.8h               \n"
                "smull  v28.8h, v19.8b, v20.8b      \n"
                "sadalp v9.4s, v25.8h               \n"
                "smull  v29.8h, v19.8b, v21.8b      \n"
                "sadalp v10.4s, v26.8h              \n"
                "smull  v30.8h, v19.8b, v22.8b      \n"
                "sadalp v11.4s, v27.8h              \n"
                "smull  v31.8h, v19.8b, v23.8b      \n"

                "sadalp v12.4s, v28.8h              \n"
                "sadalp v13.4s, v29.8h              \n"
                "sadalp v14.4s, v30.8h              \n"
                "sadalp v15.4s, v31.8h              \n"

                "2:                                 \n"

                "addp   v0.4s, v0.4s, v1.4s         \n"
                "addp   v2.4s, v2.4s, v3.4s         \n"
                "addp   v4.4s, v4.4s, v5.4s         \n"
                "addp   v6.4s, v6.4s, v7.4s         \n"
                "addp   v8.4s, v8.4s, v9.4s         \n"
                "addp   v10.4s, v10.4s, v11.4s      \n"
                "addp   v12.4s, v12.4s, v13.4s      \n"
                "addp   v14.4s, v14.4s, v15.4s      \n"

                "addp   v0.4s, v0.4s, v2.4s         \n"
                "addp   v1.4s, v4.4s, v6.4s         \n"
                "addp   v2.4s, v8.4s, v10.4s        \n"
                "addp   v3.4s, v12.4s, v14.4s       \n"

                "3:                                 \n"

                "cmp    %w5, #0                     \n"
                "beq    7f                          \n"

                "eor    v8.16b, v8.16b, v8.16b      \n"
                "eor    v9.16b, v9.16b, v9.16b      \n"
                "eor    v10.16b, v10.16b, v10.16b   \n"
                "eor    v11.16b, v11.16b, v11.16b   \n"
                "eor    v12.16b, v12.16b, v12.16b   \n"
                "eor    v13.16b, v13.16b, v13.16b   \n"
                "eor    v14.16b, v14.16b, v14.16b   \n"
                "eor    v15.16b, v15.16b, v15.16b   \n"

                "lsr    w4, %w5, #1                 \n" // w4 = nn4 >> 1
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"

                "ld1    {v16.8b, v17.8b}, [%7], #16 \n"
                "ld1    {v22.8b, v23.8b}, [%8], #16 \n"

                "zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
                "zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

                "smull  v24.8h, v18.8b, v22.8b      \n"
                "smull  v25.8h, v18.8b, v23.8b      \n"

                "zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

                "smull  v26.8h, v19.8b, v22.8b      \n"
                "smull  v27.8h, v19.8b, v23.8b      \n"

                "zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

                "smull  v28.8h, v20.8b, v22.8b      \n"
                "smull  v29.8h, v20.8b, v23.8b      \n"

                "ld1    {v16.8b, v17.8b}, [%7], #16 \n"

                "smull  v30.8h, v21.8b, v22.8b      \n"
                "smull  v31.8h, v21.8b, v23.8b      \n"

                "ld1    {v22.8b, v23.8b}, [%8], #16 \n"

                "zip1   v18.2s, v16.2s, v16.2s      \n" // _val44
                "zip2   v19.2s, v16.2s, v16.2s      \n" // _val55

                "smlal  v24.8h, v18.8b, v22.8b      \n"
                "smlal  v25.8h, v18.8b, v23.8b      \n"

                "zip1   v20.2s, v17.2s, v17.2s      \n" // _val66

                "smlal  v26.8h, v19.8b, v22.8b      \n"
                "smlal  v27.8h, v19.8b, v23.8b      \n"

                "zip2   v21.2s, v17.2s, v17.2s      \n" // _val77

                "sadalp v8.4s, v24.8h               \n"
                "smlal  v28.8h, v20.8b, v22.8b      \n"
                "sadalp v9.4s, v25.8h               \n"
                "smlal  v29.8h, v20.8b, v23.8b      \n"
                "sadalp v10.4s, v26.8h              \n"
                "smlal  v30.8h, v21.8b, v22.8b      \n"
                "sadalp v11.4s, v27.8h              \n"
                "smlal  v31.8h, v21.8b, v23.8b      \n"
                "sadalp v12.4s, v28.8h              \n"
                "sadalp v13.4s, v29.8h              \n"

                "subs   w4, w4, #1                  \n"

                "sadalp v14.4s, v30.8h              \n"
                "sadalp v15.4s, v31.8h              \n"

                "bne    4b                          \n"

                "5:                                 \n"

                "and    w4, %w5, #1                 \n" // w4 = remain = nn4 & 1
                "cmp    w4, #0                      \n" // w4 > 0
                "beq    6f                          \n"

                "ld1    {v16.8b, v17.8b}, [%7], #16 \n"
                "ld1    {v22.8b, v23.8b}, [%8], #16 \n"

                "zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
                "zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

                "smull  v24.8h, v18.8b, v22.8b      \n"
                "smull  v25.8h, v18.8b, v23.8b      \n"

                "zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

                "smull  v26.8h, v19.8b, v22.8b      \n"
                "smull  v27.8h, v19.8b, v23.8b      \n"

                "zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

                "sadalp v8.4s, v24.8h               \n"
                "smull  v28.8h, v20.8b, v22.8b      \n"
                "sadalp v9.4s, v25.8h               \n"
                "smull  v29.8h, v20.8b, v23.8b      \n"
                "sadalp v10.4s, v26.8h              \n"
                "smull  v30.8h, v21.8b, v22.8b      \n"
                "sadalp v11.4s, v27.8h              \n"
                "smull  v31.8h, v21.8b, v23.8b      \n"
                "sadalp v12.4s, v28.8h              \n"
                "sadalp v13.4s, v29.8h              \n"
                "sadalp v14.4s, v30.8h              \n"
                "sadalp v15.4s, v31.8h              \n"

                "6:                                 \n"

                "addp   v8.4s, v8.4s, v9.4s         \n"
                "addp   v10.4s, v10.4s, v11.4s      \n"
                "addp   v12.4s, v12.4s, v13.4s      \n"
                "addp   v14.4s, v14.4s, v15.4s      \n"

                "add    v0.4s, v0.4s, v8.4s         \n"
                "add    v1.4s, v1.4s, v10.4s        \n"
                "add    v2.4s, v2.4s, v12.4s        \n"
                "add    v3.4s, v3.4s, v14.4s        \n"

                "7:                                 \n"

                "lsr    w4, %w6, #2                 \n" // w4 = nn1 >> 2
                "cmp    w4, #0                      \n"
                "beq    9f                          \n"

                "8:                                 \n"

                "ld1    {v8.16b}, [%7], #16         \n"
                "ld1    {v9.16b}, [%8], #16         \n"

                "sshll  v4.8h, v8.8b, #0            \n"
                "sshll2 v5.8h, v8.16b, #0           \n"
                "sshll  v6.8h, v9.8b, #0            \n"
                "sshll2 v7.8h, v9.16b, #0           \n"

                "smlal  v0.4s, v6.4h, v4.h[0]       \n"
                "smlal  v1.4s, v6.4h, v4.h[1]       \n"
                "smlal  v2.4s, v6.4h, v4.h[2]       \n"
                "smlal  v3.4s, v6.4h, v4.h[3]       \n"
                "smlal2 v0.4s, v6.8h, v4.h[4]       \n"
                "smlal2 v1.4s, v6.8h, v4.h[5]       \n"
                "smlal2 v2.4s, v6.8h, v4.h[6]       \n"
                "smlal2 v3.4s, v6.8h, v4.h[7]       \n"
                "smlal  v0.4s, v7.4h, v5.h[0]       \n"
                "smlal  v1.4s, v7.4h, v5.h[1]       \n"
                "smlal  v2.4s, v7.4h, v5.h[2]       \n"
                "smlal  v3.4s, v7.4h, v5.h[3]       \n"
                "smlal2 v0.4s, v7.8h, v5.h[4]       \n"
                "smlal2 v1.4s, v7.8h, v5.h[5]       \n"
                "smlal2 v2.4s, v7.8h, v5.h[6]       \n"
                "smlal2 v3.4s, v7.8h, v5.h[7]       \n"

                "subs   w4, w4, #1                  \n"
                "bne    8b                          \n"

                "9:                                 \n"

                "and    w4, %w6, #3                 \n" // w4 = nn1 & 3
                "cmp    w4, #0                      \n" // w4 > 0
                "beq    11f                         \n"

                "10:                                \n"

                "ld1    {v4.8b}, [%7]               \n"
                "ld1    {v6.8b}, [%8]               \n"

                "sshll  v4.8h, v4.8b, #0            \n"
                "sshll  v6.8h, v6.8b, #0            \n"

                "smlal  v0.4s, v6.4h, v4.h[0]       \n"
                "smlal  v1.4s, v6.4h, v4.h[1]       \n"
                "smlal  v2.4s, v6.4h, v4.h[2]       \n"
                "smlal  v3.4s, v6.4h, v4.h[3]       \n"

                "add    %7, %7, #4                  \n"
                "add    %8, %8, #4                  \n"

                "subs   w4, w4, #1                  \n"
                "bne    10b                         \n"

                "11:                                 \n"

                // transpose 4x4
                "trn1   v4.4s, v0.4s, v1.4s         \n"
                "trn2   v5.4s, v0.4s, v1.4s         \n"
                "trn1   v6.4s, v2.4s, v3.4s         \n"
                "trn2   v7.4s, v2.4s, v3.4s         \n"

                "trn1   v0.2d, v4.2d, v6.2d         \n"
                "trn2   v2.2d, v4.2d, v6.2d         \n"
                "trn1   v1.2d, v5.2d, v7.2d         \n"
                "trn2   v3.2d, v5.2d, v7.2d         \n"

                "st1    {v0.4s}, [%0], #16          \n"
                "st1    {v1.4s}, [%1], #16          \n"
                "st1    {v2.4s}, [%2], #16          \n"
                "st1    {v3.4s}, [%3], #16          \n"

                : "=r"(outptr0),
                "=r"(outptr1),
                "=r"(outptr2),
                "=r"(outptr3),
                "=r"(nn),
                "=r"(nn4),
                "=r"(nn1),
                "=r"(tmpptr),
                "=r"(kptr0)
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(nn),
                "5"(nn4),
                "6"(nn1),
                "7"(tmpptr),
                "8"(kptr0)
                : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#endif // __ARM_FEATURE_DOTPROD
        }
#endif // __aarch64__
        for (; i + 1 < size; i += 2)
        {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
            const signed char* kptr0 = kernel.channel(p / 4);
#endif
#else
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr0 = kernel.channel(p / 4);
#endif

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;
#if __aarch64__
            int32x4_t _sum00 = vdupq_n_s32(0);
            int32x4_t _sum10 = vdupq_n_s32(0);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val = vld1q_s8(tmpptr);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                _sum00 = vmmlaq_s32(_sum00, _val, _w01);
                _sum10 = vmmlaq_s32(_sum10, _val, _w23);

                tmpptr += 16;
                kptr0 += 32;
            }

            int32x4_t _sum00x = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(_sum00), vreinterpretq_s64_s32(_sum10)));
            int32x4_t _sum10x = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(_sum00), vreinterpretq_s64_s32(_sum10)));

            _sum00 = _sum00x;
            _sum10 = _sum10x;
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x16_t _val01_l_h = vld1q_s8(tmpptr);
                int8x16_t _w0123_l = vld1q_s8(kptr0);

                _sum00 = vdotq_laneq_s32(_sum00, _w0123_l, _val01_l_h, 0);
                _sum10 = vdotq_laneq_s32(_sum10, _w0123_l, _val01_l_h, 1);

                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                _sum00 = vdotq_laneq_s32(_sum00, _w0123_h, _val01_l_h, 2);
                _sum10 = vdotq_laneq_s32(_sum10, _w0123_h, _val01_l_h, 3);

                tmpptr += 16;
                kptr0 += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            if (nn4 > 0)
            {
                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x16_t _val0123 = vld1q_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum00 = vdotq_laneq_s32(_sum00, _w0, _val0123, 0);
                    _sum10 = vdotq_laneq_s32(_sum10, _w0, _val0123, 1);

                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                    _sum00 = vdotq_laneq_s32(_sum00, _w1, _val0123, 2);
                    _sum10 = vdotq_laneq_s32(_sum10, _w1, _val0123, 3);

                    tmpptr += 16;
                    kptr0 += 32;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum00 = vdotq_lane_s32(_sum00, _w0, _val01, 0);
                    _sum10 = vdotq_lane_s32(_sum10, _w0, _val01, 1);

                    tmpptr += 8;
                    kptr0 += 16;
                }
            }
#else  // __ARM_FEATURE_DOTPROD
            if (nn > 0)
            {
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum02 = vdupq_n_s32(0);
                int32x4_t _sum03 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum12 = vdupq_n_s32(0);
                int32x4_t _sum13 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w23));

                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv00 = vmlal_s8(_wv00, vget_low_s8(_val1), vget_low_s8(_w45));
                    _wv01 = vmlal_s8(_wv01, vget_low_s8(_val1), vget_high_s8(_w45));
                    _wv02 = vmlal_s8(_wv02, vget_low_s8(_val1), vget_low_s8(_w67));
                    _wv03 = vmlal_s8(_wv03, vget_low_s8(_val1), vget_high_s8(_w67));

                    _wv10 = vmlal_s8(_wv10, vget_high_s8(_val1), vget_low_s8(_w45));
                    _wv11 = vmlal_s8(_wv11, vget_high_s8(_val1), vget_high_s8(_w45));
                    _wv12 = vmlal_s8(_wv12, vget_high_s8(_val1), vget_low_s8(_w67));
                    _wv13 = vmlal_s8(_wv13, vget_high_s8(_val1), vget_high_s8(_w67));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 32;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w23));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                int32x4_t _s001 = vpaddq_s32(_sum00, _sum01);
                int32x4_t _s023 = vpaddq_s32(_sum02, _sum03);
                int32x4_t _s101 = vpaddq_s32(_sum10, _sum11);
                int32x4_t _s123 = vpaddq_s32(_sum12, _sum13);

                _sum00 = vpaddq_s32(_s001, _s023);
                _sum10 = vpaddq_s32(_s101, _s123);
            }

            if (nn4 > 0)
            {
                int32x4_t _sum100 = vdupq_n_s32(0);
                int32x4_t _sum101 = vdupq_n_s32(0);
                int32x4_t _sum110 = vdupq_n_s32(0);
                int32x4_t _sum111 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x16_t _val0123 = vld1q_s8(tmpptr);

                    int32x4x2_t _val00221133 = vzipq_s32(vreinterpretq_s32_s8(_val0123), vreinterpretq_s32_s8(_val0123));
                    int8x8_t _val00 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[0]));
                    int8x8_t _val11 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[0]));
                    int8x8_t _val22 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[1]));
                    int8x8_t _val33 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[1]));

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
                    int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

                    _wv00 = vmlal_s8(_wv00, _val22, vget_low_s8(_w23));
                    _wv01 = vmlal_s8(_wv01, _val22, vget_high_s8(_w23));
                    _wv10 = vmlal_s8(_wv10, _val33, vget_low_s8(_w23));
                    _wv11 = vmlal_s8(_wv11, _val33, vget_high_s8(_w23));

                    _sum100 = vpadalq_s16(_sum100, _wv00);
                    _sum101 = vpadalq_s16(_sum101, _wv01);
                    _sum110 = vpadalq_s16(_sum110, _wv10);
                    _sum111 = vpadalq_s16(_sum111, _wv11);

                    tmpptr += 16;
                    kptr0 += 32;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);
                    int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
                    int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
                    int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

                    int8x16_t _w01 = vld1q_s8(kptr0);

                    int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
                    int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

                    _sum100 = vpadalq_s16(_sum100, _wv00);
                    _sum101 = vpadalq_s16(_sum101, _wv01);
                    _sum110 = vpadalq_s16(_sum110, _wv10);
                    _sum111 = vpadalq_s16(_sum111, _wv11);

                    tmpptr += 8;
                    kptr0 += 16;
                }

                int32x4_t _s001 = vpaddq_s32(_sum100, _sum101);
                int32x4_t _s101 = vpaddq_s32(_sum110, _sum111);

                _sum00 = vaddq_s32(_sum00, _s001);
                _sum10 = vaddq_s32(_sum10, _s101);
            }
#endif // __ARM_FEATURE_DOTPROD

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int16x8_t _val01234567 = vmovl_s8(vld1_s8(tmpptr));

                int8x16_t _w = vld1q_s8(kptr0);
                int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
                int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
                int16x4_t _w0123 = vget_low_s16(_w01234567);
                int16x4_t _w4567 = vget_high_s16(_w01234567);
                int16x4_t _w89ab = vget_low_s16(_w89abcdef);
                int16x4_t _wcdef = vget_high_s16(_w89abcdef);

                _sum00 = vmlal_laneq_s16(_sum00, _w0123, _val01234567, 0);
                _sum10 = vmlal_laneq_s16(_sum10, _w0123, _val01234567, 1);
                _sum00 = vmlal_laneq_s16(_sum00, _w4567, _val01234567, 2);
                _sum10 = vmlal_laneq_s16(_sum10, _w4567, _val01234567, 3);
                _sum00 = vmlal_laneq_s16(_sum00, _w89ab, _val01234567, 4);
                _sum10 = vmlal_laneq_s16(_sum10, _w89ab, _val01234567, 5);
                _sum00 = vmlal_laneq_s16(_sum00, _wcdef, _val01234567, 6);
                _sum10 = vmlal_laneq_s16(_sum10, _wcdef, _val01234567, 7);

                tmpptr += 8;
                kptr0 += 16;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                int16x4_t _val1 = vdup_n_s16(tmpptr[1]);

                int16x4_t _w0123;
                _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                _sum00 = vmlal_s16(_sum00, _val0, _w0123);
                _sum10 = vmlal_s16(_sum10, _val1, _w0123);

                tmpptr += 2;
                kptr0 += 4;
            }

            vst1q_lane_s32(outptr0, _sum00, 0);
            vst1q_lane_s32(outptr1, _sum00, 1);
            vst1q_lane_s32(outptr2, _sum00, 2);
            vst1q_lane_s32(outptr3, _sum00, 3);
            vst1q_lane_s32(outptr0 + 1, _sum10, 0);
            vst1q_lane_s32(outptr1 + 1, _sum10, 1);
            vst1q_lane_s32(outptr2 + 1, _sum10, 2);
            vst1q_lane_s32(outptr3 + 1, _sum10, 3);
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
#else  // __aarch64__
            asm volatile(
                "veor       q0, q0              \n"
                "veor       q1, q1              \n"
                "veor       q2, q2              \n"
                "veor       q3, q3              \n"
                "veor       q4, q4              \n"
                "veor       q5, q5              \n"
                "veor       q6, q6              \n"
                "veor       q7, q7              \n"

                "cmp        %4, #0              \n"
                "beq        3f                  \n"

                "pld        [%7, #256]          \n"

                "lsr        r4, %4, #1          \n" // r4 = nn = size >> 1
                "cmp        r4, #0              \n"
                "beq        1f                  \n"

                "add        r5, %8, #16         \n"
                "pld        [%8, #128]          \n"
                "mov        r6, #32             \n"
                "pld        [%8, #384]          \n"

                "vld1.s8    {d20-d21}, [%8 :128], r6 \n" // _w01

                "vld1.s8    {d16-d19}, [%7 :128]! \n" // _val0 _val1

                "vld1.s8    {d22-d23}, [%8 :128], r6 \n" // _w45

                "0:                             \n"

                "vmull.s8   q12, d16, d20       \n"
                "pld        [%7, #256]          \n"
                "vmull.s8   q13, d16, d21       \n"
                "pld        [%8, #384]          \n"
                "vmull.s8   q14, d17, d20       \n"
                "vmull.s8   q15, d17, d21       \n"
                "vld1.s8    {d20-d21}, [r5 :128], r6 \n" // _w23

                "vmlal.s8   q12, d18, d22       \n"
                "vmlal.s8   q13, d18, d23       \n"
                "subs       r4, r4, #1          \n"
                "vmlal.s8   q14, d19, d22       \n"
                "vmlal.s8   q15, d19, d23       \n"
                "vld1.s8    {d22-d23}, [r5 :128], r6 \n" // _w67

                "vpadal.s16 q0, q12             \n"
                "vmull.s8   q12, d16, d20       \n"
                "vpadal.s16 q1, q13             \n"
                "vmull.s8   q13, d16, d21       \n"
                "vpadal.s16 q4, q14             \n"
                "vmull.s8   q14, d17, d20       \n"
                "vpadal.s16 q5, q15             \n"
                "vmull.s8   q15, d17, d21       \n"
                "vld1.s8    {d16-d17}, [%7 :128]! \n" // _val0

                "vmlal.s8   q12, d18, d22       \n"
                "vld1.s8    {d20-d21}, [%8 :128], r6 \n" // _w01
                "vmlal.s8   q13, d18, d23       \n"
                "pld        [r5, #128]          \n"
                "vmlal.s8   q14, d19, d22       \n"
                "pld        [r5, #384]          \n"
                "vmlal.s8   q15, d19, d23       \n"
                "vld1.s8    {d18-d19}, [%7 :128]! \n" // _val1

                "vpadal.s16 q2, q12             \n"
                "vld1.s8    {d22-d23}, [%8 :128], r6 \n" // _w45
                "vpadal.s16 q3, q13             \n"
                "pld        [%7, #128]          \n"
                "vpadal.s16 q6, q14             \n"
                "pld        [%8, #128]          \n"
                "vpadal.s16 q7, q15             \n"

                "bne        0b                  \n"

                "sub        %7, %7, #32         \n"
                "sub        %8, %8, #64         \n"

                "1:                             \n"
                "and        r4, %4, #1          \n" // r4 = remain = size & 1
                "cmp        r4, #0              \n" // r4 > 0
                "beq        2f                  \n"

                "vld1.s8    {d16-d17}, [%7 :128]! \n" // _val
                "vld1.s8    {d20-d21}, [%8 :128]! \n" // _w01

                "vmull.s8   q12, d16, d20       \n"

                "vld1.s8    {d22-d23}, [%8 :128]! \n" // _w23
                "vmull.s8   q13, d16, d21       \n"
                "vmull.s8   q14, d17, d20       \n"
                "vmull.s8   q15, d17, d21       \n"

                "vpadal.s16 q0, q12             \n"
                "vmull.s8   q12, d16, d22       \n"
                "vpadal.s16 q1, q13             \n"
                "vmull.s8   q13, d16, d23       \n"
                "vpadal.s16 q4, q14             \n"
                "vmull.s8   q14, d17, d22       \n"
                "vpadal.s16 q5, q15             \n"
                "vmull.s8   q15, d17, d23       \n"

                "vpadal.s16 q2, q12             \n"
                "vpadal.s16 q3, q13             \n"
                "vpadal.s16 q6, q14             \n"
                "vpadal.s16 q7, q15             \n"

                "2:                             \n"

                "vpadd.s32  d16, d0, d1         \n"
                "vpadd.s32  d17, d2, d3         \n"
                "vpadd.s32  d18, d4, d5         \n"
                "vpadd.s32  d19, d6, d7         \n"
                "vpadd.s32  d20, d8, d9         \n"
                "vpadd.s32  d21, d10, d11       \n"
                "vpadd.s32  d22, d12, d13       \n"
                "vpadd.s32  d23, d14, d15       \n"

                "vpadd.s32  d0, d16, d17        \n"
                "vpadd.s32  d1, d18, d19        \n"
                "vpadd.s32  d2, d20, d21        \n"
                "vpadd.s32  d3, d22, d23        \n"

                "3:                             \n"

                "cmp        %5, #0              \n"
                "beq        7f                  \n"

                "veor       q2, q2              \n"
                "veor       q3, q3              \n"
                "veor       q4, q4              \n"
                "veor       q5, q5              \n"

                "lsr        r4, %5, #1          \n" // r4 = nn4 >> 1
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "4:                             \n"

                "vld1.s8    {d16-d17}, [%7]!    \n" // _val0123
                "vld1.s8    {d20-d23}, [%8]!    \n" // _w01 _w23

                "vmov.s8    q9, q8              \n"
                "vtrn.s32   q8, q9              \n" // _val00 _val22 _val11 _val33

                "vmull.s8   q12, d16, d20       \n"
                "vmull.s8   q13, d16, d21       \n"
                "vmull.s8   q14, d18, d20       \n"
                "vmull.s8   q15, d18, d21       \n"

                "vmlal.s8   q12, d17, d22       \n"
                "vmlal.s8   q13, d17, d23       \n"
                "vmlal.s8   q14, d19, d22       \n"
                "vmlal.s8   q15, d19, d23       \n"

                "vpadal.s16 q2, q12             \n"
                "vpadal.s16 q3, q13             \n"
                "vpadal.s16 q4, q14             \n"
                "vpadal.s16 q5, q15             \n"

                "subs       r4, r4, #1          \n"
                "bne        4b                  \n"

                "5:                             \n"

                "and        r4, %5, #1          \n" // r4 = nn4 & 1
                "cmp        r4, #0              \n" // r4 > 0
                "beq        6f                  \n"

                "vld1.s8    {d16}, [%7]!        \n" // _val01
                "vld1.s8    {d18-d19}, [%8]!    \n" // _w01

                "vmov.s8    d17, d16            \n"
                "vtrn.s32   d16, d17            \n" // _val00 _val11

                "vmull.s8   q12, d16, d18       \n"
                "vmull.s8   q13, d16, d19       \n"
                "vmull.s8   q14, d17, d18       \n"
                "vmull.s8   q15, d17, d19       \n"

                "vpadal.s16 q2, q12             \n"
                "vpadal.s16 q3, q13             \n"
                "vpadal.s16 q4, q14             \n"
                "vpadal.s16 q5, q15             \n"

                "6:                             \n"

                "vpadd.s32  d16, d4, d5         \n"
                "vpadd.s32  d17, d6, d7         \n"
                "vpadd.s32  d18, d8, d9         \n"
                "vpadd.s32  d19, d10, d11       \n"

                "vadd.s32   q0, q0, q8          \n"
                "vadd.s32   q1, q1, q9          \n"

                "7:                             \n"

                "lsr        r4, %6, #2          \n" // r4 = nn1 >> 2
                "cmp        r4, #0              \n"
                "beq        9f                  \n"

                "8:                             \n"

                "vld1.s8    {d4}, [%7]!         \n"
                "vmovl.s8   q2, d4              \n"

                "vld1.s8    {d10-d11}, [%8]!    \n"
                "vmovl.s8   q3, d10             \n"
                "vmovl.s8   q4, d11             \n"

                "vmlal.s16  q0, d6, d4[0]       \n"
                "vmlal.s16  q1, d6, d4[1]       \n"
                "vmlal.s16  q0, d7, d4[2]       \n"
                "vmlal.s16  q1, d7, d4[3]       \n"
                "vmlal.s16  q0, d8, d5[0]       \n"
                "vmlal.s16  q1, d8, d5[1]       \n"
                "vmlal.s16  q0, d9, d5[2]       \n"
                "vmlal.s16  q1, d9, d5[3]       \n"

                "subs       r4, r4, #1          \n"
                "bne        8b                  \n"

                "9:                             \n"

                "and        r4, %6, #3          \n" // r4 = nn1 & 3
                "cmp        r4, #0              \n" // w4 > 0
                "beq        11f                 \n"

                "10:                            \n"

                "vld1.s8    {d4[]}, [%7]!       \n"
                "vld1.s8    {d6[]}, [%7]!       \n"
                "vmovl.s8   q2, d4              \n"
                "vmovl.s8   q3, d6              \n"

                "vld1.s8    {d8}, [%8]          \n"
                "vmovl.s8   q4, d8              \n"

                "vmlal.s16  q0, d4, d8          \n"
                "vmlal.s16  q1, d6, d8          \n"

                "add        %8, %8, #4          \n"

                "subs       r4, r4, #1          \n"
                "bne        10b                 \n"

                "11:                            \n"

                "vst1.s32   {d0[0]}, [%0]!      \n"
                "vst1.s32   {d0[1]}, [%1]!      \n"
                "vst1.s32   {d1[0]}, [%2]!      \n"
                "vst1.s32   {d1[1]}, [%3]!      \n"
                "vst1.s32   {d2[0]}, [%0]!      \n"
                "vst1.s32   {d2[1]}, [%1]!      \n"
                "vst1.s32   {d3[0]}, [%2]!      \n"
                "vst1.s32   {d3[1]}, [%3]!      \n"

                : "=r"(outptr0),
                "=r"(outptr1),
                "=r"(outptr2),
                "=r"(outptr3),
                "=r"(nn),
                "=r"(nn4),
                "=r"(nn1),
                "=r"(tmpptr),
                "=r"(kptr0)
                : "0"(outptr0),
                "1"(outptr1),
                "2"(outptr2),
                "3"(outptr3),
                "4"(nn),
                "5"(nn4),
                "6"(nn1),
                "7"(tmpptr),
                "8"(kptr0)
                : "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
        }
        for (; i < size; i++)
        {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 4);
#endif
#else
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 4);
#endif

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            int32x4_t _sum23 = vdupq_n_s32(0);

            for (int j = 0; j < nn; j++)
            {
                int8x8_t _val0 = vld1_s8(tmpptr);
                int8x16_t _w01 = vld1q_s8(kptr0);
                int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                int8x16_t _val = vcombine_s8(_val0, _val0);

                _sum0 = vdotq_s32(_sum0, _val, _w01);
                _sum23 = vdotq_s32(_sum23, _val, _w23);

                tmpptr += 8;
                kptr0 += 32;
            }

            _sum0 = vpaddq_s32(_sum0, _sum23);
#else  // __ARM_FEATURE_MATMUL_INT8
            for (int j = 0; j < nn; j++)
            {
                int8x8_t _val0_l_h = vld1_s8(tmpptr);

                int8x16_t _w0123_l = vld1q_s8(kptr0);

                _sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);

                int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                _sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);

                tmpptr += 8;
                kptr0 += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8

            if (nn4 > 0)
            {
                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);

                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _w0, _val01, 0);

                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w1, _val01, 1);

                    tmpptr += 8;
                    kptr0 += 32;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val_xxx = vld1_s8(tmpptr);

                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _w0, _val_xxx, 0);

                    tmpptr += 4;
                    kptr0 += 16;
                }
            }
#else // __ARM_FEATURE_DOTPROD
            if (nn > 0)
            {
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
                    _wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
                    _wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
                    _wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 16;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(_val, vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(_val, vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(_val, vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(_val, vget_high_s8(_w23));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 8;
                    kptr0 += 32;
                }

#if __aarch64__
                int32x4_t _s01 = vpaddq_s32(_sum0, _sum1);
                int32x4_t _s23 = vpaddq_s32(_sum2, _sum3);

                _sum0 = vpaddq_s32(_s01, _s23);
#else
                int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                int32x2_t _s23_low = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s23_high = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));

                _sum0 = vcombine_s32(vpadd_s32(_s01_low, _s01_high), vpadd_s32(_s23_low, _s23_high));
#endif
            }

            if (nn4 > 0)
            {
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn4; j += 2)
                {
                    int8x8_t _val01 = vld1_s8(tmpptr);
                    int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
                    int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
                    int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

                    int8x16_t _w0 = vld1q_s8(kptr0);
                    int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(_val00, vget_low_s8(_w0));
                    int16x8_t _wv1 = vmull_s8(_val00, vget_high_s8(_w0));

                    _wv0 = vmlal_s8(_wv0, _val11, vget_low_s8(_w1));
                    _wv1 = vmlal_s8(_wv1, _val11, vget_high_s8(_w1));

                    _sum10 = vpadalq_s16(_sum10, _wv0);
                    _sum11 = vpadalq_s16(_sum11, _wv1);

                    tmpptr += 8;
                    kptr0 += 32;
                }
                for (; j < nn4; j++)
                {
                    int8x8_t _val_xxx = vld1_s8(tmpptr);
                    int8x8_t _val_val = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_val_xxx), vreinterpret_s32_s8(_val_xxx)).val[0]);

                    int8x16_t _w0 = vld1q_s8(kptr0);

                    int16x8_t _wv0 = vmull_s8(_val_val, vget_low_s8(_w0));
                    int16x8_t _wv1 = vmull_s8(_val_val, vget_high_s8(_w0));

                    _sum10 = vpadalq_s16(_sum10, _wv0);
                    _sum11 = vpadalq_s16(_sum11, _wv1);

                    tmpptr += 4;
                    kptr0 += 16;
                }

#if __aarch64__
                int32x4_t _s01 = vpaddq_s32(_sum10, _sum11);
#else
                int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum10), vget_high_s32(_sum10));
                int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum11), vget_high_s32(_sum11));

                int32x4_t _s01 = vcombine_s32(_s01_low, _s01_high);
#endif

                _sum0 = vaddq_s32(_sum0, _s01);
            }
#endif // __ARM_FEATURE_DOTPROD

            int32x4_t _sum1 = vdupq_n_s32(0);

            int j = 0;
            for (; j + 3 < nn1; j += 4)
            {
                int16x4_t _val0123 = vget_low_s16(vmovl_s8(vld1_s8(tmpptr)));

                int8x16_t _w = vld1q_s8(kptr0);
                int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
                int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
                int16x4_t _w0123 = vget_low_s16(_w01234567);
                int16x4_t _w4567 = vget_high_s16(_w01234567);
                int16x4_t _w89ab = vget_low_s16(_w89abcdef);
                int16x4_t _wcdef = vget_high_s16(_w89abcdef);

                _sum0 = vmlal_lane_s16(_sum0, _w0123, _val0123, 0);
                _sum1 = vmlal_lane_s16(_sum1, _w4567, _val0123, 1);
                _sum0 = vmlal_lane_s16(_sum0, _w89ab, _val0123, 2);
                _sum1 = vmlal_lane_s16(_sum1, _wcdef, _val0123, 3);

                tmpptr += 4;
                kptr0 += 16;
            }
            for (; j < nn1; j++)
            {
                int16x4_t _val = vdup_n_s16(tmpptr[0]);

                int16x4_t _w0123;
                _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                _sum0 = vmlal_s16(_sum0, _val, _w0123);

                tmpptr += 1;
                kptr0 += 4;
            }

            _sum0 = vaddq_s32(_sum0, _sum1);

            vst1q_lane_s32(outptr0, _sum0, 0);
            vst1q_lane_s32(outptr1, _sum0, 1);
            vst1q_lane_s32(outptr2, _sum0, 2);
            vst1q_lane_s32(outptr3, _sum0, 3);
            outptr0 += 1;
            outptr1 += 1;
            outptr2 += 1;
            outptr3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;
#else // __ARM_NEON
    nn_outch = outch >> 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int i = 0;
        for (; i + 1 < size; i += 2)
        {
            const signed char* tmpptr = tmp.channel(i / 2);
            const signed char* kptr = kernel.channel(p / 2);

            int sum00 = 0;
            int sum01 = 0;
            int sum10 = 0;
            int sum11 = 0;

            int nn1 = inch * maxk;

            int j = 0;
#if __ARM_FEATURE_SIMD32
            for (; j + 1 < nn1; j += 2)
            {
                // fomit-frame-pointer implied in optimized flag spare one register
                // let us stay away from error: ‘asm’ operand has impossible constraints   --- nihui
#if __OPTIMIZE__
                asm volatile(
                    "ldr    r2, [%0], #4    \n" // int8x4_t _val = *((int8x4_t*)tmpptr); tmpptr += 4;
                    "ldr    r3, [%1], #4    \n" // int8x4_t _k = *((int8x4_t*)kptr); kptr += 4;
                    "ror    r4, r2, #8      \n" // int8x4_t _val_r8 = __ror(_val, 8);
                    "ror    r5, r3, #8      \n" // int8x4_t _k_r8 = __ror(_k, 8);
                    "sxtb16 r2, r2          \n" // int16x2_t _val02 = __sxtb16(_val);
                    "sxtb16 r3, r3          \n" // int16x2_t _w02 = __sxtb16(_k);
                    "sxtb16 r4, r4          \n" // int16x2_t _val13 = __sxtb16(_val_r8);
                    "sxtb16 r5, r5          \n" // int16x2_t _w13 = __sxtb16(_k_r8);
                    "smlad  %2, r2, r3, %2  \n" // sum00 = __smlad(_val02, _w02, sum00);
                    "smlad  %3, r4, r3, %3  \n" // sum01 = __smlad(_val13, _w02, sum01);
                    "smlad  %4, r2, r5, %4  \n" // sum10 = __smlad(_val02, _w13, sum10);
                    "smlad  %5, r4, r5, %5  \n" // sum11 = __smlad(_val13, _w13, sum11);
                    : "=r"(tmpptr),
                    "=r"(kptr),
                    "=r"(sum00),
                    "=r"(sum01),
                    "=r"(sum10),
                    "=r"(sum11)
                    : "0"(tmpptr),
                    "1"(kptr),
                    "2"(sum00),
                    "3"(sum01),
                    "4"(sum10),
                    "5"(sum11)
                    : "memory", "r2", "r3", "r4", "r5");
#else
                int _val = *((int*)tmpptr);
                int _k = *((int*)kptr);
                int _val_r8;
                int _k_r8;
                asm volatile("ror %0, %2, #8"
                             : "=r"(_val_r8)
                             : "0"(_val_r8), "r"(_val)
                             :);
                asm volatile("ror %0, %2, #8"
                             : "=r"(_k_r8)
                             : "0"(_k_r8), "r"(_k)
                             :);
                int _val02;
                int _w02;
                int _val13;
                int _w13;
                asm volatile("sxtb16 %0, %2"
                             : "=r"(_val02)
                             : "0"(_val02), "r"(_val)
                             :);
                asm volatile("sxtb16 %0, %2"
                             : "=r"(_w02)
                             : "0"(_w02), "r"(_k)
                             :);
                asm volatile("sxtb16 %0, %2"
                             : "=r"(_val13)
                             : "0"(_val13), "r"(_val_r8)
                             :);
                asm volatile("sxtb16 %0, %2"
                             : "=r"(_w13)
                             : "0"(_w13), "r"(_k_r8)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum00)
                             : "0"(sum00), "r"(_val02), "r"(_w02)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum01)
                             : "0"(sum01), "r"(_val13), "r"(_w02)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum10)
                             : "0"(sum10), "r"(_val02), "r"(_w13)
                             :);
                asm volatile("smlad %0, %2, %3, %0"
                             : "=r"(sum11)
                             : "0"(sum11), "r"(_val13), "r"(_w13)
                             :);
                tmpptr += 4;
                kptr += 4;
#endif
            }
#endif // __ARM_FEATURE_SIMD32
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char w0 = kptr[0];
                signed char w1 = kptr[1];

                sum00 += val0 * w0;
                sum01 += val1 * w0;
                sum10 += val0 * w1;
                sum11 += val1 * w1;

                tmpptr += 2;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
        }
        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
            const signed char* kptr = kernel.channel(p / 2);

            int sum00 = 0;
            int sum10 = 0;

            int nn1 = inch * maxk;

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char w0 = kptr[0];
                signed char w1 = kptr[1];

                sum00 += val0 * w0;
                sum10 += val0 * w1;

                tmpptr += 1;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr1[0] = sum10;
            outptr0 += 1;
            outptr1 += 1;
        }
    }

    remain_outch_start += nn_outch << 1;
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr0 = top_blob.channel(p);

        int i = 0;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            if (nn > 0)
            {
#if __ARM_FEATURE_MATMUL_INT8
                int32x2_t _s0 = vdup_n_s32(0);
                int32x2_t _s1 = vdup_n_s32(0);
                int32x2_t _s2 = vdup_n_s32(0);
                int32x2_t _s3 = vdup_n_s32(0);
                int32x2_t _s4 = vdup_n_s32(0);
                int32x2_t _s5 = vdup_n_s32(0);
                int32x2_t _s6 = vdup_n_s32(0);
                int32x2_t _s7 = vdup_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x16_t _val2 = vld1q_s8(tmpptr + 32);
                    int8x16_t _val3 = vld1q_s8(tmpptr + 48);
                    int8x8_t _w = vld1_s8(kptr0);

                    _s0 = vdot_s32(_s0, vget_low_s8(_val0), _w);
                    _s1 = vdot_s32(_s1, vget_high_s8(_val0), _w);
                    _s2 = vdot_s32(_s2, vget_low_s8(_val1), _w);
                    _s3 = vdot_s32(_s3, vget_high_s8(_val1), _w);
                    _s4 = vdot_s32(_s4, vget_low_s8(_val2), _w);
                    _s5 = vdot_s32(_s5, vget_high_s8(_val2), _w);
                    _s6 = vdot_s32(_s6, vget_low_s8(_val3), _w);
                    _s7 = vdot_s32(_s7, vget_high_s8(_val3), _w);

                    tmpptr += 64;
                    kptr0 += 8;
                }

                _sum0 = vpaddq_s32(vcombine_s32(_s0, _s1), vcombine_s32(_s2, _s3));
                _sum1 = vpaddq_s32(vcombine_s32(_s4, _s5), vcombine_s32(_s6, _s7));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);
                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _val4567_l, _w_lh, 0);
                    _sum2 = vdotq_lane_s32(_sum2, _val0123_h, _w_lh, 1);
                    _sum3 = vdotq_lane_s32(_sum3, _val4567_h, _w_lh, 1);

                    tmpptr += 64;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);
#endif // __ARM_FEATURE_MATMUL_INT8
            }

            if (nn4 > 0)
            {
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int j = 0; j < nn4; j++)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);

                    int8x8_t _w_0123_xxxx = vld1_s8(kptr0);

                    _sum2 = vdotq_lane_s32(_sum2, _val0, _w_0123_xxxx, 0);
                    _sum3 = vdotq_lane_s32(_sum3, _val1, _w_0123_xxxx, 0);

                    tmpptr += 32;
                    kptr0 += 4;
                }

                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                int8x8_t _val = vld1_s8(tmpptr);
                int8x8_t _w = vld1_dup_s8(kptr0);

                int16x8_t _s = vmull_s8(_val, _w);

                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s));

                tmpptr += 8;
                kptr0 += 1;
            }

            vst1q_s32(outptr0, _sum0);
            vst1q_s32(outptr0 + 4, _sum1);
            outptr0 += 8;
        }
#endif // __ARM_FEATURE_DOTPROD
        for (; i + 3 < size; i += 4)
        {
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);
#endif

            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x4_t _sum0 = vdupq_n_s32(0);
            if (nn > 0)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x2_t _s0 = vdup_n_s32(0);
                int32x2_t _s1 = vdup_n_s32(0);
                int32x2_t _s2 = vdup_n_s32(0);
                int32x2_t _s3 = vdup_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x8_t _w = vld1_s8(kptr0);

                    _s0 = vdot_s32(_s0, vget_low_s8(_val0), _w);
                    _s1 = vdot_s32(_s1, vget_high_s8(_val0), _w);
                    _s2 = vdot_s32(_s2, vget_low_s8(_val1), _w);
                    _s3 = vdot_s32(_s3, vget_high_s8(_val1), _w);

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0 = vpaddq_s32(vcombine_s32(_s0, _s1), vcombine_s32(_s2, _s3));
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _val0123_h, _w_lh, 1);

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x16_t _val2 = vld1q_s8(tmpptr + 32);
                    int8x16_t _val3 = vld1q_s8(tmpptr + 48);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), vget_low_s8(_w));
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), vget_low_s8(_w));

                    _s0 = vmlal_s8(_s0, vget_low_s8(_val2), vget_high_s8(_w));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_val2), vget_high_s8(_w));
                    _s2 = vmlal_s8(_s2, vget_low_s8(_val3), vget_high_s8(_w));
                    _s3 = vmlal_s8(_s3, vget_high_s8(_val3), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
                    _sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

                    tmpptr += 64;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), _w);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), _w);
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), _w);
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
                    _sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);
                _sum4 = vaddq_s32(_sum4, _sum5);
                _sum6 = vaddq_s32(_sum6, _sum7);

                int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s4 = vadd_s32(vget_low_s32(_sum4), vget_high_s32(_sum4));
                int32x2_t _s6 = vadd_s32(vget_low_s32(_sum6), vget_high_s32(_sum6));
                int32x2_t _ss0 = vpadd_s32(_s0, _s2);
                int32x2_t _ss1 = vpadd_s32(_s4, _s6);
                _sum0 = vcombine_s32(_ss0, _ss1);
#endif // __ARM_FEATURE_DOTPROD
            }

            int sum0123[4] = {0, 0, 0, 0};

            if (nn4 > 0)
            {
#if __ARM_FEATURE_DOTPROD
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j < nn4; j++)
                {
                    int8x16_t _val0123_lh = vld1q_s8(tmpptr);
                    int8x8_t _w_lh_xx = vld1_s8(kptr0);

                    _sum1 = vdotq_lane_s32(_sum1, _val0123_lh, _w_lh_xx, 0);

                    tmpptr += 16;
                    kptr0 += 4;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
#else  // __ARM_FEATURE_DOTPROD
                int j = 0;
                for (; j < nn4; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];
                    signed char val4 = tmpptr[4];
                    signed char val5 = tmpptr[5];
                    signed char val6 = tmpptr[6];
                    signed char val7 = tmpptr[7];
                    signed char val8 = tmpptr[8];
                    signed char val9 = tmpptr[9];
                    signed char val10 = tmpptr[10];
                    signed char val11 = tmpptr[11];
                    signed char val12 = tmpptr[12];
                    signed char val13 = tmpptr[13];
                    signed char val14 = tmpptr[14];
                    signed char val15 = tmpptr[15];

                    signed char w0 = kptr0[0];
                    signed char w1 = kptr0[1];
                    signed char w2 = kptr0[2];
                    signed char w3 = kptr0[3];

                    sum0123[0] += val0 * w0;
                    sum0123[0] += val1 * w1;
                    sum0123[0] += val2 * w2;
                    sum0123[0] += val3 * w3;
                    sum0123[1] += val4 * w0;
                    sum0123[1] += val5 * w1;
                    sum0123[1] += val6 * w2;
                    sum0123[1] += val7 * w3;
                    sum0123[2] += val8 * w0;
                    sum0123[2] += val9 * w1;
                    sum0123[2] += val10 * w2;
                    sum0123[2] += val11 * w3;
                    sum0123[3] += val12 * w0;
                    sum0123[3] += val13 * w1;
                    sum0123[3] += val14 * w2;
                    sum0123[3] += val15 * w3;

                    tmpptr += 16;
                    kptr0 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
            }

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char val2 = tmpptr[2];
                signed char val3 = tmpptr[3];
                signed char w = kptr0[0];

                sum0123[0] += val0 * w;
                sum0123[1] += val1 * w;
                sum0123[2] += val2 * w;
                sum0123[3] += val3 * w;

                tmpptr += 4;
                kptr0 += 1;
            }

            _sum0 = vaddq_s32(_sum0, vld1q_s32(sum0123));

            vst1q_s32(outptr0, _sum0);
            outptr0 += 4;
        }
#endif // __aarch64__
        for (; i + 1 < size; i += 2)
        {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);
#endif
#else
            const signed char* tmpptr = tmp.channel(i / 2);
#if __ARM_NEON
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr0 = kernel.channel(p / 2 + p % 2);
#endif
#endif

#if __ARM_NEON
            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            int32x2_t _sum = vdup_n_s32(0);
            if (nn > 0)
            {
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                int32x2_t _s0 = vdup_n_s32(0);
                int32x2_t _s1 = vdup_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    _s0 = vdot_s32(_s0, vget_low_s8(_val), _w);
                    _s1 = vdot_s32(_s1, vget_high_s8(_val), _w);

                    tmpptr += 16;
                    kptr0 += 8;
                }

                _sum = vpadd_s32(_s0, _s1);
#else  // __ARM_FEATURE_MATMUL_INT8
                int32x2_t _sum0 = vdup_n_s32(0);
                int32x2_t _sum1 = vdup_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val01_lh = vld1q_s8(tmpptr);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdot_lane_s32(_sum0, vget_low_s8(_val01_lh), _w_lh, 0);
                    _sum1 = vdot_lane_s32(_sum1, vget_high_s8(_val01_lh), _w_lh, 1);

                    tmpptr += 16;
                    kptr0 += 8;
                }

                _sum = vadd_s32(_sum0, _sum1);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));

                    _s0 = vmlal_s8(_s0, vget_low_s8(_val1), vget_high_s8(_w));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_val1), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                    tmpptr += 32;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val), _w);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val), _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                    tmpptr += 16;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);

                int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                _sum = vpadd_s32(_s0, _s2);
#endif // __ARM_FEATURE_DOTPROD
            }

            int sum0 = 0;
            int sum1 = 0;

            if (nn4 > 0)
            {
                int j = 0;
                for (; j < nn4; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];
                    signed char val4 = tmpptr[4];
                    signed char val5 = tmpptr[5];
                    signed char val6 = tmpptr[6];
                    signed char val7 = tmpptr[7];

                    signed char w0 = kptr0[0];
                    signed char w1 = kptr0[1];
                    signed char w2 = kptr0[2];
                    signed char w3 = kptr0[3];

                    sum0 += val0 * w0;
                    sum0 += val1 * w1;
                    sum0 += val2 * w2;
                    sum0 += val3 * w3;
                    sum1 += val4 * w0;
                    sum1 += val5 * w1;
                    sum1 += val6 * w2;
                    sum1 += val7 * w3;

                    tmpptr += 8;
                    kptr0 += 4;
                }
            }
#else  // __ARM_NEON
            int nn1 = inch * maxk;
            int sum0 = 0;
            int sum1 = 0;
#endif // __ARM_NEON

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val0 = tmpptr[0];
                signed char val1 = tmpptr[1];
                signed char w = kptr0[0];

                sum0 += val0 * w;
                sum1 += val1 * w;

                tmpptr += 2;
                kptr0 += 1;
            }

#if __ARM_NEON
            int sum01[2] = {sum0, sum1};
            _sum = vadd_s32(_sum, vld1_s32(sum01));
            vst1_s32(outptr0, _sum);
#else  // __ARM_NEON
            outptr0[0] = sum0;
            outptr0[1] = sum1;
#endif // __ARM_NEON
            outptr0 += 2;
        }
        for (; i < size; i++)
        {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const signed char* tmpptr = tmp.channel(i / 4 + (i % 4) / 2 + i % 2);
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);
#endif
#else
            const signed char* tmpptr = tmp.channel(i / 2 + i % 2);
#if __ARM_NEON
            const signed char* kptr0 = kernel.channel(p / 4 + p % 4);
#else
            const signed char* kptr0 = kernel.channel(p / 2 + p % 2);
#endif
#endif

            int sum = 0;
#if __ARM_NEON
            int nn = (inch / 8) * maxk;
            int nn4 = ((inch % 8) / 4) * maxk;
            int nn1 = (inch % 4) * maxk;

            if (nn > 0)
            {
#if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x2_t _sum1 = vdup_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x16_t _w = vld1q_s8(kptr0);

                    _sum0 = vdotq_s32(_sum0, _val, _w);

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    _sum1 = vdot_s32(_sum1, _val, _w);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                sum = vaddvq_s32(_sum0) + vaddv_s32(_sum1);
#else // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s8 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w));
                    _s8 = vmlal_s8(_s8, vget_high_s8(_val), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s8 = vmull_s8(_val, _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

                    tmpptr += 8;
                    kptr0 += 8;
                }

                int32x4_t _sum = vaddq_s32(_sum0, _sum1);
#if __aarch64__
                sum = vaddvq_s32(_sum); // dot
#else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                _ss = vpadd_s32(_ss, _ss);
                sum = vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_FEATURE_DOTPROD
            }

            if (nn4 > 0)
            {
                int j = 0;
                for (; j < nn4; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];

                    signed char w0 = kptr0[0];
                    signed char w1 = kptr0[1];
                    signed char w2 = kptr0[2];
                    signed char w3 = kptr0[3];

                    sum += val0 * w0;
                    sum += val1 * w1;
                    sum += val2 * w2;
                    sum += val3 * w3;

                    tmpptr += 4;
                    kptr0 += 4;
                }
            }
#else  // __ARM_NEON
            int nn1 = inch * maxk;
#endif // __ARM_NEON

            int j = 0;
            for (; j < nn1; j++)
            {
                signed char val = tmpptr[0];
                signed char w = kptr0[0];

                sum += val * w;

                tmpptr += 1;
                kptr0 += 1;
            }

            outptr0[0] = sum;
            outptr0 += 1;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
#if !(__ARM_FEATURE_MATMUL_INT8 || __ARM_FEATURE_DOTPROD)
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        convolution_im2col_sgemm_transform_kernel_int8_neon_i8mm(_kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        convolution_im2col_sgemm_transform_kernel_int8_neon_asimddp(_kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
        return;
    }
#endif
#endif

    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    // dst = 4a-4b-2aa-2bb-maxk-inch/8a-outch/8b (arm82)
    // dst = 8a-8b-maxk-inch/8a-outch/8b (arm84)
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
    if (outch >= 8)
    {
        if (inch >= 8)
            kernel_tm.create(64 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(32 * maxk, inch / 4 + inch % 4, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + outch % 4, (size_t)1u);
    }
    else if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(32 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + outch % 4, (size_t)1u);
    }
#else
    if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(32 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(16 * maxk, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + outch % 4, (size_t)1u);
    }
#endif // __ARM_FEATURE_DOTPROD
#else  // __ARM_NEON
    if (outch >= 2)
    {
        {
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2, (size_t)1u);
        }
    }
#endif // __ARM_NEON
    else
    {
#if __ARM_NEON
        if (inch >= 8)
            kernel_tm.create(8 * maxk, inch / 8 + (inch % 8) / 4 + inch % 4, outch, (size_t)1u);
        else if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + inch % 4, outch, (size_t)1u);
        else
#endif // __ARM_NEON
        {
            kernel_tm.create(1 * maxk, inch, outch, (size_t)1u);
        }
    }

    int q = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
    for (; q + 7 < outch; q += 8)
    {
        signed char* g00 = kernel_tm.channel(q / 8);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
#if __ARM_FEATURE_MATMUL_INT8
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
#else  // __ARM_FEATURE_MATMUL_INT8
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 4; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
                for (int i = 4; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
                for (int i = 4; i < 8; i++)
                {
                    for (int j = 4; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
#endif // __ARM_FEATURE_MATMUL_INT8
            }
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __ARM_FEATURE_DOTPROD
    for (; q + 3 < outch; q += 4)
    {
#if __ARM_FEATURE_DOTPROD
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        signed char* g00 = kernel_tm.channel(q / 4);
#endif

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
#if __ARM_FEATURE_MATMUL_INT8
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
#elif __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 4; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
#else
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
#endif
            }
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = kernel.channel(q + i).row<const signed char>(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __ARM_NEON
    for (; q + 1 < outch; q += 2)
    {
        signed char* g00 = kernel_tm.channel(q / 2);

        int p = 0;
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const signed char* k00 = kernel.channel(q + i).row<const signed char>(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __ARM_NEON
    for (; q < outch; q++)
    {
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
        signed char* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        signed char* g00 = kernel_tm.channel(q / 4 + q % 4);
#endif
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __ARM_NEON
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const signed char* k00 = kernel.channel(q).row<const signed char>(p + j);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const signed char* k00 = kernel.channel(q).row<const signed char>(p + j);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
#endif // __ARM_NEON
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k00 = kernel.channel(q).row<const signed char>(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void convolution_im2col_sgemm_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 1u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            signed char* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const signed char* sptr = img.row<const signed char>(dilation_h * u) + dilation_w * v;

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

    im2col_sgemm_int8_neon(bottom_im2col, top_blob, kernel, opt);
}
