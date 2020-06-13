// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static void conv_im2col_sgemm_transform_kernel_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_size)
{
    const float* kernel = _kernel;

#if __ARM_NEON && __aarch64__
    // kernel memory packed 8 x 8
    kernel_tm.create(8 * kernel_size, inch, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    // kernel memory packed 4 x 8
    kernel_tm.create(4 * kernel_size, inch, outch / 4 + outch % 4);
#endif

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        const float* k0 = kernel + (p + 0) * inch * kernel_size;
        const float* k1 = kernel + (p + 1) * inch * kernel_size;
        const float* k2 = kernel + (p + 2) * inch * kernel_size;
        const float* k3 = kernel + (p + 3) * inch * kernel_size;
        const float* k4 = kernel + (p + 4) * inch * kernel_size;
        const float* k5 = kernel + (p + 5) * inch * kernel_size;
        const float* k6 = kernel + (p + 6) * inch * kernel_size;
        const float* k7 = kernel + (p + 7) * inch * kernel_size;

        float* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp[4] = k4[0];
            ktmp[5] = k5[0];
            ktmp[6] = k6[0];
            ktmp[7] = k7[0];
            ktmp += 8;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
            k4 += 1;
            k5 += 1;
            k6 += 1;
            k7 += 1;
        }
    }
#endif

    nn_outch = (outch - remain_outch_start) >> 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        const float* k0 = kernel + (p + 0) * inch * kernel_size;
        const float* k1 = kernel + (p + 1) * inch * kernel_size;
        const float* k2 = kernel + (p + 2) * inch * kernel_size;
        const float* k3 = kernel + (p + 3) * inch * kernel_size;

#if __ARM_NEON && __aarch64__
        float* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4);
#else
        float* ktmp = kernel_tm.channel(p / 4);
#endif // __ARM_NEON && __aarch64__

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k1[0];
            ktmp[2] = k2[0];
            ktmp[3] = k3[0];
            ktmp += 4;

            k0 += 1;
            k1 += 1;
            k2 += 1;
            k3 += 1;
        }
    }

    remain_outch_start += nn_outch << 2;

    for (int p = remain_outch_start; p < outch; p++)
    {
        const float* k0 = kernel + (p + 0) * inch * kernel_size;

#if __ARM_NEON && __aarch64__
        float* ktmp = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
        float* ktmp = kernel_tm.channel(p / 4 + p % 4);
#endif // __ARM_NEON && __aarch64__

        for (int q = 0; q < inch * kernel_size; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

static void conv_im2col_sgemm_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias,
                                   const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    // im2col
    Mat bottom_im2col(outw * outh, kernel_h * kernel_w * inch, elemsize, opt.workspace_allocator);
    {
        const int stride = kernel_h * kernel_w * outw * outh;
        float* ret = (float*)bottom_im2col;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const float* input = bottom_blob.channel(p);
            int retID = stride * p;
            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // bottom_im2col memory packed 8 x 8
    Mat bottom_tm(8 * kernel_size, inch, out_size / 8 + out_size % 8, elemsize, opt.workspace_allocator);
    {
        int nn_size = out_size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 8);

            for (int q = 0; q < inch * kernel_size; q++)
            {
#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "prfm    pldl1keep, [%0, #256]   \n"
                    "ld1     {v0.4s, v1.4s}, [%0]    \n"
                    "st1     {v0.4s, v1.4s}, [%1]    \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "cc", "memory", "v0", "v1");
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0]       \n"
                    "vst1.f32   {d0-d3}, [%1]       \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "q0", "q1");
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
#endif // __ARM_NEON
                tmpptr += 8;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const float* img0 = bottom_im2col.channel(0);
            img0 += i;

            float* tmpptr = bottom_tm.channel(i / 8 + i % 8);

            for (int q = 0; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += out_size;
            }
        }
    }

    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        //int M = outch;                    // outch
        int N = outw * outh;                // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __aarch64__
        nn_outch = outch >> 3;
        remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 8;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);
            float* output4 = top_blob.channel(i + 4);
            float* output5 = top_blob.channel(i + 5);
            float* output6 = top_blob.channel(i + 6);
            float* output7 = top_blob.channel(i + 7);

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
                const float* va = kernel_tm.channel(i / 8);
#if __ARM_NEON
                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%21]   \n"
                    "dup    v16.4s, v0.s[0]         \n" // sum0
                    "dup    v17.4s, v0.s[0]         \n"
                    "dup    v18.4s, v0.s[1]         \n" // sum1
                    "dup    v19.4s, v0.s[1]         \n"
                    "dup    v20.4s, v0.s[2]         \n" // sum2
                    "dup    v21.4s, v0.s[2]         \n"
                    "dup    v22.4s, v0.s[3]         \n" // sum3
                    "dup    v23.4s, v0.s[3]         \n"
                    "dup    v24.4s, v1.s[0]         \n" // sum4
                    "dup    v25.4s, v1.s[0]         \n"
                    "dup    v26.4s, v1.s[1]         \n" // sum5
                    "dup    v27.4s, v1.s[1]         \n"
                    "dup    v28.4s, v1.s[2]         \n" // sum6
                    "dup    v29.4s, v1.s[2]         \n"
                    "dup    v30.4s, v1.s[3]         \n" // sum7
                    "dup    v31.4s, v1.s[3]         \n"

                    "lsr         w4, %w20, #2            \n" // r4 = nn = L >> 2
                    "cmp         w4, #0                  \n"
                    "beq         1f                      \n"

                    "0:                                  \n" // for (; k+3<L; k=k+4)

                    "prfm   pldl1keep, [%9, #512]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n" // kernel
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

                    "prfm   pldl1keep, [%8, #512]                       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n" // data
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"
                    // k0
                    "fmla    v16.4s, v8.4s, v0.s[0]      \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v9.4s, v0.s[0]      \n" //
                    "fmla    v18.4s, v8.4s, v0.s[1]      \n" // sum1 += (a00-a70) * k10
                    "fmla    v19.4s, v9.4s, v0.s[1]      \n" //
                    "fmla    v20.4s, v8.4s, v0.s[2]      \n" // sum2 += (a00-a70) * k20
                    "fmla    v21.4s, v9.4s, v0.s[2]      \n" //
                    "fmla    v22.4s, v8.4s, v0.s[3]      \n" // sum3 += (a00-a70) * k30
                    "fmla    v23.4s, v9.4s, v0.s[3]      \n" //
                    "fmla    v24.4s, v8.4s, v1.s[0]      \n" // sum4 += (a00-a70) * k40
                    "fmla    v25.4s, v9.4s, v1.s[0]      \n" //
                    "fmla    v26.4s, v8.4s, v1.s[1]      \n" // sum5 += (a00-a70) * k50
                    "fmla    v27.4s, v9.4s, v1.s[1]      \n" //
                    "fmla    v28.4s, v8.4s, v1.s[2]      \n" // sum6 += (a00-a70) * k60
                    "fmla    v29.4s, v9.4s, v1.s[2]      \n" //
                    "fmla    v30.4s, v8.4s, v1.s[3]      \n" // sum7 += (a00-a70) * k70
                    "fmla    v31.4s, v9.4s, v1.s[3]      \n" //
                    // k1
                    "fmla    v16.4s, v10.4s, v2.s[0]     \n" // sum0 += (a01-a71) * k01
                    "fmla    v17.4s, v11.4s, v2.s[0]     \n" //
                    "fmla    v18.4s, v10.4s, v2.s[1]     \n" // sum1 += (a01-a71) * k11
                    "fmla    v19.4s, v11.4s, v2.s[1]     \n" //
                    "fmla    v20.4s, v10.4s, v2.s[2]     \n" // sum2 += (a01-a71) * k21
                    "fmla    v21.4s, v11.4s, v2.s[2]     \n" //
                    "fmla    v22.4s, v10.4s, v2.s[3]     \n" // sum3 += (a01-a71) * k31
                    "fmla    v23.4s, v11.4s, v2.s[3]     \n" //
                    "fmla    v24.4s, v10.4s, v3.s[0]     \n" // sum4 += (a01-a71) * k41
                    "fmla    v25.4s, v11.4s, v3.s[0]     \n" //
                    "fmla    v26.4s, v10.4s, v3.s[1]     \n" // sum5 += (a01-a71) * k51
                    "fmla    v27.4s, v11.4s, v3.s[1]     \n" //
                    "fmla    v28.4s, v10.4s, v3.s[2]     \n" // sum6 += (a01-a71) * k61
                    "fmla    v29.4s, v11.4s, v3.s[2]     \n" //
                    "fmla    v30.4s, v10.4s, v3.s[3]     \n" // sum7 += (a01-a71) * k71
                    "fmla    v31.4s, v11.4s, v3.s[3]     \n" //
                    // k2
                    "fmla    v16.4s, v12.4s, v4.s[0]     \n" // sum0 += (a02-a72) * k02
                    "fmla    v17.4s, v13.4s, v4.s[0]     \n" //
                    "fmla    v18.4s, v12.4s, v4.s[1]     \n" // sum1 += (a02-a72) * k12
                    "fmla    v19.4s, v13.4s, v4.s[1]     \n" //
                    "fmla    v20.4s, v12.4s, v4.s[2]     \n" // sum2 += (a02-a72) * k22
                    "fmla    v21.4s, v13.4s, v4.s[2]     \n" //
                    "fmla    v22.4s, v12.4s, v4.s[3]     \n" // sum3 += (a02-a72) * k32
                    "fmla    v23.4s, v13.4s, v4.s[3]     \n" //
                    "fmla    v24.4s, v12.4s, v5.s[0]     \n" // sum4 += (a02-a72) * k42
                    "fmla    v25.4s, v13.4s, v5.s[0]     \n" //
                    "fmla    v26.4s, v12.4s, v5.s[1]     \n" // sum5 += (a02-a72) * k52
                    "fmla    v27.4s, v13.4s, v5.s[1]     \n" //
                    "fmla    v28.4s, v12.4s, v5.s[2]     \n" // sum6 += (a02-a72) * k62
                    "fmla    v29.4s, v13.4s, v5.s[2]     \n" //
                    "fmla    v30.4s, v12.4s, v5.s[3]     \n" // sum7 += (a02-a72) * k72
                    "fmla    v31.4s, v13.4s, v5.s[3]     \n" //
                    // k3
                    "fmla    v16.4s, v14.4s, v6.s[0]     \n" // sum0 += (a03-a73) * k03
                    "fmla    v17.4s, v15.4s, v6.s[0]     \n" //
                    "fmla    v18.4s, v14.4s, v6.s[1]     \n" // sum1 += (a03-a73) * k13
                    "fmla    v19.4s, v15.4s, v6.s[1]     \n" //
                    "fmla    v20.4s, v14.4s, v6.s[2]     \n" // sum2 += (a03-a73) * k23
                    "fmla    v21.4s, v15.4s, v6.s[2]     \n" //
                    "fmla    v22.4s, v14.4s, v6.s[3]     \n" // sum3 += (a03-a73) * k33
                    "fmla    v23.4s, v15.4s, v6.s[3]     \n" //
                    "fmla    v24.4s, v14.4s, v7.s[0]     \n" // sum4 += (a03-a73) * k43
                    "fmla    v25.4s, v15.4s, v7.s[0]     \n" //
                    "fmla    v26.4s, v14.4s, v7.s[1]     \n" // sum5 += (a03-a73) * k53
                    "fmla    v27.4s, v15.4s, v7.s[1]     \n" //
                    "fmla    v28.4s, v14.4s, v7.s[2]     \n" // sum6 += (a03-a73) * k63
                    "fmla    v29.4s, v15.4s, v7.s[2]     \n" //
                    "fmla    v30.4s, v14.4s, v7.s[3]     \n" // sum7 += (a03-a73) * k73
                    "fmla    v31.4s, v15.4s, v7.s[3]     \n" //

                    "subs   w4, w4, #1                   \n"
                    "bne    0b                           \n"

                    "1:                                  \n"

                    // remain loop
                    "and    w4, %w20, #3                 \n" // w4 = remain = inch & 3;
                    "cmp    w4, #0                       \n"
                    "beq    3f                           \n"

                    "2:                                  \n"

                    "prfm   pldl1keep, [%9, #256]        \n"
                    "ld1    {v0.4s, v1.4s}, [%9], #32    \n"

                    "prfm   pldl1keep, [%8, #256]        \n"
                    "ld1    {v8.4s, v9.4s}, [%8], #32    \n"

                    // k0
                    "fmla    v16.4s, v8.4s, v0.s[0]      \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v9.4s, v0.s[0]      \n" //
                    "fmla    v18.4s, v8.4s, v0.s[1]      \n" // sum1 += (a00-a70) * k10
                    "fmla    v19.4s, v9.4s, v0.s[1]      \n" //
                    "fmla    v20.4s, v8.4s, v0.s[2]      \n" // sum2 += (a00-a70) * k20
                    "fmla    v21.4s, v9.4s, v0.s[2]      \n" //
                    "fmla    v22.4s, v8.4s, v0.s[3]      \n" // sum3 += (a00-a70) * k30
                    "fmla    v23.4s, v9.4s, v0.s[3]      \n" //
                    "fmla    v24.4s, v8.4s, v1.s[0]      \n" // sum4 += (a00-a70) * k40
                    "fmla    v25.4s, v9.4s, v1.s[0]      \n" //
                    "fmla    v26.4s, v8.4s, v1.s[1]      \n" // sum5 += (a00-a70) * k50
                    "fmla    v27.4s, v9.4s, v1.s[1]      \n" //
                    "fmla    v28.4s, v8.4s, v1.s[2]      \n" // sum6 += (a00-a70) * k60
                    "fmla    v29.4s, v9.4s, v1.s[2]      \n" //
                    "fmla    v30.4s, v8.4s, v1.s[3]      \n" // sum7 += (a00-a70) * k70
                    "fmla    v31.4s, v9.4s, v1.s[3]      \n" //

                    "subs   w4, w4, #1                   \n"

                    "bne    2b                           \n"

                    "3:                                  \n"

                    "st1    {v16.4s, v17.4s}, [%0]       \n"
                    "st1    {v18.4s, v19.4s}, [%1]       \n"
                    "st1    {v20.4s, v21.4s}, [%2]       \n"
                    "st1    {v22.4s, v23.4s}, [%3]       \n"
                    "st1    {v24.4s, v25.4s}, [%4]       \n"
                    "st1    {v26.4s, v27.4s}, [%5]       \n"
                    "st1    {v28.4s, v29.4s}, [%6]       \n"
                    "st1    {v30.4s, v31.4s}, [%7]       \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(output4), // %4
                    "=r"(output5), // %5
                    "=r"(output6), // %6
                    "=r"(output7), // %7
                    "=r"(vb),      // %8
                    "=r"(va)       // %9
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(output4),
                    "5"(output5),
                    "6"(output6),
                    "7"(output7),
                    "8"(vb),
                    "9"(va),
                    "r"(L),      // %20
                    "r"(biasptr) // %21
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                float sum4[8] = {0};
                float sum5[8] = {0};
                float sum6[8] = {0};
                float sum7[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        sum4[n] += va[4] * vb[n];
                        sum5[n] += va[5] * vb[n];
                        sum6[n] += va[6] * vb[n];
                        sum7[n] += va[7] * vb[n];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 8];
                        sum1[n] += va[1] * vb[n + 8];
                        sum2[n] += va[2] * vb[n + 8];
                        sum3[n] += va[3] * vb[n + 8];
                        sum4[n] += va[4] * vb[n + 8];
                        sum5[n] += va[5] * vb[n + 8];
                        sum6[n] += va[6] * vb[n + 8];
                        sum7[n] += va[7] * vb[n + 8];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 16];
                        sum1[n] += va[1] * vb[n + 16];
                        sum2[n] += va[2] * vb[n + 16];
                        sum3[n] += va[3] * vb[n + 16];
                        sum4[n] += va[4] * vb[n + 16];
                        sum5[n] += va[5] * vb[n + 16];
                        sum6[n] += va[6] * vb[n + 16];
                        sum7[n] += va[7] * vb[n + 16];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 24];
                        sum1[n] += va[1] * vb[n + 24];
                        sum2[n] += va[2] * vb[n + 24];
                        sum3[n] += va[3] * vb[n + 24];
                        sum4[n] += va[4] * vb[n + 24];
                        sum5[n] += va[5] * vb[n + 24];
                        sum6[n] += va[6] * vb[n + 24];
                        sum7[n] += va[7] * vb[n + 24];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 32];
                        sum1[n] += va[1] * vb[n + 32];
                        sum2[n] += va[2] * vb[n + 32];
                        sum3[n] += va[3] * vb[n + 32];
                        sum4[n] += va[4] * vb[n + 32];
                        sum5[n] += va[5] * vb[n + 32];
                        sum6[n] += va[6] * vb[n + 32];
                        sum7[n] += va[7] * vb[n + 32];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 40];
                        sum1[n] += va[1] * vb[n + 40];
                        sum2[n] += va[2] * vb[n + 40];
                        sum3[n] += va[3] * vb[n + 40];
                        sum4[n] += va[4] * vb[n + 40];
                        sum5[n] += va[5] * vb[n + 40];
                        sum6[n] += va[6] * vb[n + 40];
                        sum7[n] += va[7] * vb[n + 40];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 48];
                        sum1[n] += va[1] * vb[n + 48];
                        sum2[n] += va[2] * vb[n + 48];
                        sum3[n] += va[3] * vb[n + 48];
                        sum4[n] += va[4] * vb[n + 48];
                        sum5[n] += va[5] * vb[n + 48];
                        sum6[n] += va[6] * vb[n + 48];
                        sum7[n] += va[7] * vb[n + 48];
                        va += 8;

                        sum0[n] += va[0] * vb[n + 56];
                        sum1[n] += va[1] * vb[n + 56];
                        sum2[n] += va[2] * vb[n + 56];
                        sum3[n] += va[3] * vb[n + 56];
                        sum4[n] += va[4] * vb[n + 56];
                        sum5[n] += va[5] * vb[n + 56];
                        sum6[n] += va[6] * vb[n + 56];
                        sum7[n] += va[7] * vb[n + 56];
                        va -= 56;
                    }

                    va += 64;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        sum4[n] += va[4] * vb[n];
                        sum5[n] += va[5] * vb[n];
                        sum6[n] += va[6] * vb[n];
                        sum7[n] += va[7] * vb[n];
                    }

                    va += 8;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                    output4[n] = sum4[n] + biasptr[4];
                    output5[n] = sum5[n] + biasptr[5];
                    output6[n] = sum6[n] + biasptr[6];
                    output7[n] = sum7[n] + biasptr[7];
                }
#endif // __ARM_NEON
                output0 += 8;
                output1 += 8;
                output2 += 8;
                output3 += 8;
                output4 += 8;
                output5 += 8;
                output6 += 8;
                output7 += 8;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 8 + j % 8);
                const float* va = kernel_tm.channel(i / 8);

#if __ARM_NEON
                asm volatile(
                    "ld1    {v14.4s, v15.4s}, [%21]      \n" // sum0_7 inital with bias
                    "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                    "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                    "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                    "eor    v19.16b, v19.16b, v19.16b    \n" // sum3
                    "eor    v20.16b, v20.16b, v20.16b    \n" // sum4
                    "eor    v21.16b, v21.16b, v21.16b    \n" // sum5
                    "eor    v22.16b, v22.16b, v22.16b    \n" // sum6
                    "eor    v23.16b, v23.16b, v23.16b    \n" // sum7

                    "lsr         w4, %w20, #2            \n" // r4 = nn = L >> 2
                    "cmp         w4, #0                  \n"
                    "beq         1f                      \n"

                    "0:                                  \n" // for (; k+3<L; k=k+4)

                    "prfm   pldl1keep, [%9, #256]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n" // k
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

                    "prfm   pldl1keep, [%8, #128]        \n"
                    "ld1    {v8.4s}, [%8], #16           \n" // d

                    // k0
                    "fmla    v16.4s, v0.4s, v8.s[0]      \n" // sum0 += (k00-k70) * a00
                    "fmla    v17.4s, v1.4s, v8.s[0]      \n" //
                    "fmla    v18.4s, v2.4s, v8.s[1]      \n" // sum1 += (k01-k71) * a10
                    "fmla    v19.4s, v3.4s, v8.s[1]      \n" //
                    "fmla    v20.4s, v4.4s, v8.s[2]      \n" // sum2 += (k02-k72) * a20
                    "fmla    v21.4s, v5.4s, v8.s[2]      \n" //
                    "fmla    v22.4s, v6.4s, v8.s[3]      \n" // sum3 += (k03-k73) * a30
                    "fmla    v23.4s, v7.4s, v8.s[3]      \n" //

                    "subs   w4, w4, #1                   \n"
                    "bne    0b                           \n"

                    "fadd   v16.4s, v16.4s, v18.4s       \n"
                    "fadd   v17.4s, v17.4s, v19.4s       \n"
                    "fadd   v20.4s, v20.4s, v22.4s       \n"
                    "fadd   v21.4s, v21.4s, v23.4s       \n"
                    "fadd   v16.4s, v16.4s, v20.4s       \n"
                    "fadd   v17.4s, v17.4s, v21.4s       \n"
                    "fadd   v14.4s, v14.4s, v16.4s       \n"
                    "fadd   v15.4s, v15.4s, v17.4s       \n"

                    "1:                                  \n"

                    // remain loop
                    "and    w4, %w20, #3                 \n" // w4 = remain = inch & 3;
                    "cmp    w4, #0                       \n"
                    "beq    3f                           \n"

                    "2:                                  \n"

                    "prfm   pldl1keep, [%9, #256]        \n"
                    "ld1    {v0.4s, v1.4s}, [%9], #32    \n"
                    "prfm   pldl1keep, [%8, #32]         \n"
                    "ld1r   {v8.4s}, [%8], #4            \n"

                    // k0
                    "fmla   v14.4s, v8.4s, v0.4s         \n" // sum0 += (k00-k70) * a00
                    "fmla   v15.4s, v8.4s, v1.4s         \n" //

                    "subs   w4, w4, #1                   \n"

                    "bne    2b                           \n"

                    "3:                                  \n"

                    "st1    {v14.s}[0], [%0]             \n"
                    "st1    {v14.s}[1], [%1]             \n"
                    "st1    {v14.s}[2], [%2]             \n"
                    "st1    {v14.s}[3], [%3]             \n"
                    "st1    {v15.s}[0], [%4]             \n"
                    "st1    {v15.s}[1], [%5]             \n"
                    "st1    {v15.s}[2], [%6]             \n"
                    "st1    {v15.s}[3], [%7]             \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(output4), // %4
                    "=r"(output5), // %5
                    "=r"(output6), // %6
                    "=r"(output7), // %7
                    "=r"(vb),      // %8
                    "=r"(va)       // %9
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(output4),
                    "5"(output5),
                    "6"(output6),
                    "7"(output7),
                    "8"(vb),
                    "9"(va),
                    "r"(L),      // %20
                    "r"(biasptr) // %21
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];
                float sum4 = biasptr[4];
                float sum5 = biasptr[5];
                float sum6 = biasptr[6];
                float sum7 = biasptr[7];

                for (int k = 0; k < L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];
                    sum4 += va[4] * vb[0];
                    sum5 += va[5] * vb[0];
                    sum6 += va[6] * vb[0];
                    sum7 += va[7] * vb[0];

                    va += 8;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
                output4[0] = sum4;
                output5[0] = sum5;
                output6[0] = sum6;
                output7[0] = sum7;
#endif // __ARM_NEON
                output0++;
                output1++;
                output2++;
                output3++;
                output4++;
                output5++;
                output6++;
                output7++;
            }
        }
#endif // __aarch64__

        nn_outch = (outch - remain_outch_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = remain_outch_start + pp * 4;

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + i : zeros;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
#if __ARM_NEON && __aarch64__
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4);
#else
                const float* va = kernel_tm.channel(i / 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "ld1    {v0.4s}, [%13]               \n"
                    "dup    v16.4s, v0.s[0]              \n" // sum0
                    "dup    v17.4s, v0.s[0]              \n"
                    "dup    v18.4s, v0.s[1]              \n" // sum1
                    "dup    v19.4s, v0.s[1]              \n"
                    "dup    v20.4s, v0.s[2]              \n" // sum2
                    "dup    v21.4s, v0.s[2]              \n"
                    "dup    v22.4s, v0.s[3]              \n" // sum3
                    "dup    v23.4s, v0.s[3]              \n"

                    "lsr         w4, %w12, #2            \n" // r4 = nn = L >> 2
                    "cmp         w4, #0                  \n"
                    "beq         1f                      \n"

                    "0:                                  \n" // for (; k+3<L; k=k+4)

                    "prfm   pldl1keep, [%5, #512]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n" // kernel

                    "prfm   pldl1keep, [%4, #512]                       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64   \n" // data
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n"

                    "subs   w4, w4, #1                   \n"
                    // k0
                    "fmla    v16.4s, v8.4s, v0.s[0]      \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v9.4s, v0.s[0]      \n" //
                    "fmla    v18.4s, v8.4s, v0.s[1]      \n" // sum1 += (a00-a70) * k10
                    "fmla    v19.4s, v9.4s, v0.s[1]      \n" //
                    "fmla    v20.4s, v8.4s, v0.s[2]      \n" // sum2 += (a00-a70) * k20
                    "fmla    v21.4s, v9.4s, v0.s[2]      \n" //
                    "fmla    v22.4s, v8.4s, v0.s[3]      \n" // sum3 += (a00-a70) * k30
                    "fmla    v23.4s, v9.4s, v0.s[3]      \n" //
                    // k1
                    "fmla    v16.4s, v10.4s, v1.s[0]     \n" // sum0 += (a01-a71) * k01
                    "fmla    v17.4s, v11.4s, v1.s[0]     \n" //
                    "fmla    v18.4s, v10.4s, v1.s[1]     \n" // sum1 += (a01-a71) * k11
                    "fmla    v19.4s, v11.4s, v1.s[1]     \n" //
                    "fmla    v20.4s, v10.4s, v1.s[2]     \n" // sum2 += (a01-a71) * k21
                    "fmla    v21.4s, v11.4s, v1.s[2]     \n" //
                    "fmla    v22.4s, v10.4s, v1.s[3]     \n" // sum3 += (a01-a71) * k31
                    "fmla    v23.4s, v11.4s, v1.s[3]     \n" //
                    // k2
                    "fmla    v16.4s, v12.4s, v2.s[0]     \n" // sum0 += (a02-a72) * k02
                    "fmla    v17.4s, v13.4s, v2.s[0]     \n" //
                    "fmla    v18.4s, v12.4s, v2.s[1]     \n" // sum1 += (a02-a72) * k12
                    "fmla    v19.4s, v13.4s, v2.s[1]     \n" //
                    "fmla    v20.4s, v12.4s, v2.s[2]     \n" // sum2 += (a02-a72) * k22
                    "fmla    v21.4s, v13.4s, v2.s[2]     \n" //
                    "fmla    v22.4s, v12.4s, v2.s[3]     \n" // sum3 += (a02-a72) * k32
                    "fmla    v23.4s, v13.4s, v2.s[3]     \n" //
                    // k3
                    "fmla    v16.4s, v14.4s, v3.s[0]     \n" // sum0 += (a03-a73) * k03
                    "fmla    v17.4s, v15.4s, v3.s[0]     \n" //
                    "fmla    v18.4s, v14.4s, v3.s[1]     \n" // sum1 += (a03-a73) * k13
                    "fmla    v19.4s, v15.4s, v3.s[1]     \n" //
                    "fmla    v20.4s, v14.4s, v3.s[2]     \n" // sum2 += (a03-a73) * k23
                    "fmla    v21.4s, v15.4s, v3.s[2]     \n" //
                    "fmla    v22.4s, v14.4s, v3.s[3]     \n" // sum3 += (a03-a73) * k33
                    "fmla    v23.4s, v15.4s, v3.s[3]     \n" //

                    "bne    0b                           \n"

                    "1:                                  \n"

                    // remain loop
                    "and    w4, %w12, #3                 \n" // w4 = remain = inch & 3;
                    "cmp    w4, #0                       \n"
                    "beq    3f                           \n"

                    "2:                                  \n"

                    "prfm   pldl1keep, [%5, #256]        \n"
                    "ld1    {v0.4s}, [%5], #16           \n"
                    "prfm   pldl1keep, [%4, #256]        \n"
                    "ld1    {v8.4s, v9.4s}, [%4], #32    \n"
                    // k0
                    "fmla    v16.4s, v8.4s, v0.s[0]      \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v9.4s, v0.s[0]      \n" //
                    "fmla    v18.4s, v8.4s, v0.s[1]      \n" // sum1 += (a00-a70) * k10
                    "fmla    v19.4s, v9.4s, v0.s[1]      \n" //
                    "fmla    v20.4s, v8.4s, v0.s[2]      \n" // sum2 += (a00-a70) * k20
                    "fmla    v21.4s, v9.4s, v0.s[2]      \n" //
                    "fmla    v22.4s, v8.4s, v0.s[3]      \n" // sum3 += (a00-a70) * k30
                    "fmla    v23.4s, v9.4s, v0.s[3]      \n" //

                    "subs   w4, w4, #1                   \n"

                    "bne    2b                           \n"

                    "3:                                  \n"

                    "st1    {v16.4s, v17.4s}, [%0]       \n"
                    "st1    {v18.4s, v19.4s}, [%1]       \n"
                    "st1    {v20.4s, v21.4s}, [%2]       \n"
                    "st1    {v22.4s, v23.4s}, [%3]       \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(vb),      // %4
                    "=r"(va)       // %5
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(vb),
                    "5"(va),
                    "r"(L),      // %12
                    "r"(biasptr) // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
                asm volatile(
                    "vld1.f32   {d0-d1}, [%13]      \n"
                    "vdup.f32   q8, d0[0]           \n"
                    "vdup.f32   q9, d0[0]           \n"
                    "vdup.f32   q10, d0[1]          \n"
                    "vdup.f32   q11, d0[1]          \n"
                    "vdup.f32   q12, d1[0]          \n"
                    "vdup.f32   q13, d1[0]          \n"
                    "vdup.f32   q14, d1[1]          \n"
                    "vdup.f32   q15, d1[1]          \n"

                    "lsr         r4, %12, #2        \n" // r4 = nn = L >> 2
                    "cmp         r4, #0             \n"
                    "beq         1f                 \n"

                    "0:                             \n" // for(; nn != 0; nn--)
                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n" // kernel
                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n" // data

                    "vmla.f32   q8, q4, d0[0]       \n" // sum0 = (a00-a07) * k00
                    "vmla.f32   q9, q5, d0[0]       \n"
                    "vmla.f32   q10, q4, d0[1]      \n" // sum1 = (a00-a07) * k10
                    "vmla.f32   q11, q5, d0[1]      \n"
                    "vmla.f32   q12, q4, d1[0]      \n" // sum2 = (a00-a07) * k20
                    "vmla.f32   q13, q5, d1[0]      \n"
                    "vmla.f32   q14, q4, d1[1]      \n" // sum3 = (a00-a07) * k30
                    "vmla.f32   q15, q5, d1[1]      \n"

                    "vmla.f32   q8, q6, d2[0]       \n" // sum0 += (a10-a17) * k01
                    "vmla.f32   q9, q7, d2[0]       \n"
                    "vmla.f32   q10, q6, d2[1]      \n" // sum1 += (a10-a17) * k11
                    "vmla.f32   q11, q7, d2[1]      \n"
                    "vmla.f32   q12, q6, d3[0]      \n" // sum2 += (a10-a17) * k21
                    "vmla.f32   q13, q7, d3[0]      \n"
                    "vmla.f32   q14, q6, d3[1]      \n" // sum3 += (a10-a17) * k31
                    "vmla.f32   q15, q7, d3[1]      \n"

                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n" // data

                    "vmla.f32   q8, q4, d4[0]       \n" // sum0 += (a20-a27) * k02
                    "vmla.f32   q9, q5, d4[0]       \n"
                    "vmla.f32   q10, q4, d4[1]      \n" // sum1 += (a20-a27) * k12
                    "vmla.f32   q11, q5, d4[1]      \n"
                    "vmla.f32   q12, q4, d5[0]      \n" // sum2 += (a20-a27) * k22
                    "vmla.f32   q13, q5, d5[0]      \n"
                    "vmla.f32   q14, q4, d5[1]      \n" // sum3 += (a20-a27) * k32
                    "vmla.f32   q15, q5, d5[1]      \n"

                    "vmla.f32   q8, q6, d6[0]       \n" // sum0 += (a30-a37) * k03
                    "vmla.f32   q9, q7, d6[0]       \n"
                    "vmla.f32   q10, q6, d6[1]      \n" // sum1 += (a30-a37) * k13
                    "vmla.f32   q11, q7, d6[1]      \n"
                    "vmla.f32   q12, q6, d7[0]      \n" // sum2 += (a30-a37) * k23
                    "vmla.f32   q13, q7, d7[0]      \n"
                    "vmla.f32   q14, q6, d7[1]      \n" // sum3 += (a30-a37) * k33
                    "vmla.f32   q15, q7, d7[1]      \n"

                    "subs        r4, r4, #1         \n"
                    "bne         0b                 \n" // end for

                    "1:                             \n"
                    // remain loop
                    "and         r4, %12, #3        \n" // r4 = remain = inch & 3
                    "cmp         r4, #0             \n"
                    "beq         3f                 \n"

                    "2:                             \n" // for(; remain != 0; remain--)

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"
                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d8-d11}, [%4]!     \n"

                    "vmla.f32   q8, q4, d0[0]       \n" // sum0 += (a00-a70) * k00
                    "vmla.f32   q9, q5, d0[0]       \n"
                    "vmla.f32   q10, q4, d0[1]      \n" // sum1 += (a00-a70) * k10
                    "vmla.f32   q11, q5, d0[1]      \n"
                    "vmla.f32   q12, q4, d1[0]      \n" // sum2 += (a00-a70) * k20
                    "vmla.f32   q13, q5, d1[0]      \n"
                    "vmla.f32   q14, q4, d1[1]      \n" // sum3 += (a00-a70) * k30
                    "vmla.f32   q15, q5, d1[1]      \n"

                    "subs        r4, r4, #1         \n"
                    "bne         2b                 \n"

                    "3:                             \n" // store the result to memory
                    "vst1.f32    {d16-d19}, [%0]    \n"
                    "vst1.f32    {d20-d23}, [%1]    \n"
                    "vst1.f32    {d24-d27}, [%2]    \n"
                    "vst1.f32    {d28-d31}, [%3]    \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(vb),      // %4
                    "=r"(va)       // %5
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(vb),
                    "5"(va),
                    "r"(L),      // %12
                    "r"(biasptr) // %13
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 8];
                        sum1[n] += va[1] * vb[n + 8];
                        sum2[n] += va[2] * vb[n + 8];
                        sum3[n] += va[3] * vb[n + 8];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 16];
                        sum1[n] += va[1] * vb[n + 16];
                        sum2[n] += va[2] * vb[n + 16];
                        sum3[n] += va[3] * vb[n + 16];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 24];
                        sum1[n] += va[1] * vb[n + 24];
                        sum2[n] += va[2] * vb[n + 24];
                        sum3[n] += va[3] * vb[n + 24];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 32];
                        sum1[n] += va[1] * vb[n + 32];
                        sum2[n] += va[2] * vb[n + 32];
                        sum3[n] += va[3] * vb[n + 32];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 40];
                        sum1[n] += va[1] * vb[n + 40];
                        sum2[n] += va[2] * vb[n + 40];
                        sum3[n] += va[3] * vb[n + 40];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 48];
                        sum1[n] += va[1] * vb[n + 48];
                        sum2[n] += va[2] * vb[n + 48];
                        sum3[n] += va[3] * vb[n + 48];
                        va += 4;

                        sum0[n] += va[0] * vb[n + 56];
                        sum1[n] += va[1] * vb[n + 56];
                        sum2[n] += va[2] * vb[n + 56];
                        sum3[n] += va[3] * vb[n + 56];
                        va -= 28;
                    }

                    va += 32;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum0[n] += va[0] * vb[n];
                        sum1[n] += va[1] * vb[n];
                        sum2[n] += va[2] * vb[n];
                        sum3[n] += va[3] * vb[n];
                    }

                    va += 4;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output0[n] = sum0[n] + biasptr[0];
                    output1[n] = sum1[n] + biasptr[1];
                    output2[n] = sum2[n] + biasptr[2];
                    output3[n] = sum3[n] + biasptr[3];
                }
#endif // __ARM_NEON
                output0 += 8;
                output1 += 8;
                output2 += 8;
                output3 += 8;
            }

            for (; j < N; j++)
            {
                float* vb = bottom_tm.channel(j / 8 + j % 8);
#if __ARM_NEON && __aarch64__
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4);
#else
                const float* va = kernel_tm.channel(i / 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "ld1    {v14.4s}, [%13]              \n" // sum0_3 inital with bias

                    "lsr         w4, %w12, #2            \n" // r4 = nn = L >> 2
                    "cmp         w4, #0                  \n"
                    "beq         1f                      \n"

                    "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                    "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                    "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                    "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                    "0:                                  \n" // for (; k+3<L; k=k+4)

                    "prfm   pldl1keep, [%5, #256]                       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n" // k

                    "prfm   pldl1keep, [%4, #128]        \n"
                    "ld1    {v8.4s}, [%4], #16           \n" // d

                    "subs   w4, w4, #1                   \n"
                    "fmla    v16.4s, v0.4s, v8.s[0]      \n" // sum0 += (k00-k30) * a00
                    "fmla    v17.4s, v1.4s, v8.s[1]      \n" // sum1 += (k01-k31) * a10
                    "fmla    v18.4s, v2.4s, v8.s[2]      \n" // sum2 += (k02-k32) * a20
                    "fmla    v19.4s, v3.4s, v8.s[3]      \n" // sum3 += (k03-k33) * a30

                    "bne    0b                           \n"

                    "fadd      v16.4s, v16.4s, v18.4s    \n"
                    "fadd      v17.4s, v17.4s, v19.4s    \n"
                    "fadd      v14.4s, v14.4s, v16.4s    \n"
                    "fadd      v14.4s, v14.4s, v17.4s    \n"

                    "1:                                  \n"

                    // remain loop
                    "and    w4, %w12, #3                 \n" // w4 = remain = inch & 3;
                    "cmp    w4, #0                       \n"
                    "beq    3f                           \n"

                    "2:                                  \n"

                    "prfm   pldl1keep, [%5, #128]        \n"
                    "ld1    {v0.4s}, [%5], #16           \n"
                    "prfm   pldl1keep, [%4, #32]         \n"
                    "ld1r   {v8.4s}, [%4], #4            \n"

                    "subs   w4, w4, #1                   \n"
                    // k0
                    "fmla   v14.4s, v8.4s, v0.4s         \n" // sum0 += (k00-k30) * a00
                    "bne    2b                           \n"

                    "3:                                  \n"

                    "st1    {v14.s}[0], [%0]             \n"
                    "st1    {v14.s}[1], [%1]             \n"
                    "st1    {v14.s}[2], [%2]             \n"
                    "st1    {v14.s}[3], [%3]             \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(vb),      // %4
                    "=r"(va)       // %5
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(vb),
                    "5"(va),
                    "r"(L),      // %12
                    "r"(biasptr) // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
#else
                asm volatile(
                    // inch loop
                    "vld1.f32   {d24-d25}, [%13]    \n"

                    "lsr         r4, %12, #2        \n" // r4 = nn = L >> 2
                    "cmp         r4, #0             \n"
                    "beq         1f                 \n"

                    "veor       q8, q8, q8          \n"
                    "veor       q9, q9, q9          \n"
                    "veor       q10, q10, q10       \n"
                    "veor       q11, q11, q11       \n"

                    "0:                             \n" // for(; nn != 0; nn--)
                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n" // kernel
                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d8-d9}, [%4]!      \n" // data

                    "vmla.f32   q8, q0, d8[0]       \n" // (k00-k30) * a00
                    "vmla.f32   q9, q1, d8[1]       \n" // (k01-k31) * a01
                    "vmla.f32   q10, q2, d9[0]      \n" // (k02-k32) * a02
                    "vmla.f32   q11, q3, d9[1]      \n" // (k03-k33) * a03

                    "subs        r4, r4, #1         \n"
                    "bne         0b                 \n" // end for

                    "vadd.f32   q8, q8, q9          \n"
                    "vadd.f32   q10, q10, q11       \n"
                    "vadd.f32   q8, q8, q10         \n"
                    "vadd.f32   q12, q12, q8        \n"

                    "1:                             \n"
                    // remain loop
                    "and         r4, %12, #3        \n" // r4 = remain = inch & 3
                    "cmp         r4, #0             \n"
                    "beq         3f                 \n"

                    "2:                             \n" // for(; remain != 0; remain--)
                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"
                    "pld        [%4, #32]           \n"
                    "vld1.f32   {d8[],d9[]}, [%4]!  \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q12, q0, q4         \n"
                    "bne         2b                 \n"

                    "3:                             \n" // store the result to memory
                    "vst1.f32    {d24[0]}, [%0]     \n"
                    "vst1.f32    {d24[1]}, [%1]     \n"
                    "vst1.f32    {d25[0]}, [%2]     \n"
                    "vst1.f32    {d25[1]}, [%3]     \n"

                    : "=r"(output0), // %0
                    "=r"(output1), // %1
                    "=r"(output2), // %2
                    "=r"(output3), // %3
                    "=r"(vb),      // %4
                    "=r"(va)       // %5
                    : "0"(output0),
                    "1"(output1),
                    "2"(output2),
                    "3"(output3),
                    "4"(vb),
                    "5"(va),
                    "r"(L),      // %12
                    "r"(biasptr) // %13
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
                float sum0 = biasptr[0];
                float sum1 = biasptr[1];
                float sum2 = biasptr[2];
                float sum3 = biasptr[3];

                for (int k = 0; k < L; k++)
                {
                    sum0 += va[0] * vb[0];
                    sum1 += va[1] * vb[0];
                    sum2 += va[2] * vb[0];
                    sum3 += va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;
#endif // __ARM_NEON
                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        remain_outch_start += nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            int j = 0;
            for (; j + 7 < N; j = j + 8)
            {
                const float* vb = bottom_tm.channel(j / 8);
#if __ARM_NEON && __aarch64__
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
                const float* va = kernel_tm.channel(i / 4 + i % 4);
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
                asm volatile(
                    "dup    v16.4s, %w7                  \n" // sum0
                    "dup    v17.4s, %w7                  \n" // sum0n

                    "lsr         w4, %w6, #2             \n" // r4 = nn = L >> 2
                    "cmp         w4, #0                  \n"
                    "beq         1f                      \n"

                    "0:                                  \n" // for (; k+3<L; k=k+4)

                    "prfm   pldl1keep, [%2, #128]        \n"
                    "ld1    {v0.4s}, [%2], #16           \n"

                    "prfm   pldl1keep, [%1, #512]        \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64   \n" // data
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

                    // k0
                    "fmla    v16.4s, v8.4s, v0.s[0]      \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v9.4s, v0.s[0]      \n" //
                    // k1
                    "fmla    v16.4s, v10.4s, v0.s[1]     \n" // sum0 += (a01-a71) * k01
                    "fmla    v17.4s, v11.4s, v0.s[1]     \n" //
                    // k2
                    "fmla    v16.4s, v12.4s, v0.s[2]     \n" // sum0 += (a02-a72) * k02
                    "fmla    v17.4s, v13.4s, v0.s[2]     \n" //
                    // k3
                    "fmla    v16.4s, v14.4s, v0.s[3]     \n" // sum0 += (a03-a73) * k03
                    "fmla    v17.4s, v15.4s, v0.s[3]     \n" //

                    "subs   w4, w4, #1                   \n"
                    "bne    0b                           \n"

                    "1:                                  \n"

                    // remain loop
                    "and    w4, %w6, #3                  \n" // w4 = remain = inch & 3;
                    "cmp    w4, #0                       \n"
                    "beq    3f                           \n"

                    "2:                                  \n"
                    "prfm   pldl1keep, [%2, #32]         \n"
                    "ld1r   {v0.4s}, [%2], #4            \n"
                    "prfm   pldl1keep, [%1, #256]        \n"
                    "ld1    {v8.4s, v9.4s}, [%1], #32    \n"

                    "subs   w4, w4, #1                   \n"
                    // k0
                    "fmla    v16.4s, v0.4s, v8.4s        \n" // sum0 += (a00-a70) * k00
                    "fmla    v17.4s, v0.4s, v9.4s        \n" //

                    "bne    2b                           \n"

                    "3:                                  \n"
                    "st1    {v16.4s, v17.4s}, [%0]       \n"

                    : "=r"(output), // %0
                    "=r"(vb),     // %1
                    "=r"(va)      // %2
                    : "0"(output),
                    "1"(vb),
                    "2"(va),
                    "r"(L),    // %6
                    "r"(bias0) // %7
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
#else
                asm volatile(
                    "vdup.f32   q8, %7              \n"
                    "vdup.f32   q9, %7              \n"
                    // inch loop
                    "lsr        r4, %6, #2          \n" // r4 = nn = inch >> 2
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "0:                             \n"

                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d8-d15}       \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%2]!      \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q5, d0[0]       \n"

                    "pld        [%1, #512]          \n"
                    "vldm       %1!, {d24-d31}      \n"

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
                    "and        r4, %6, #3          \n" // r4 = remain = inch & 3;
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d8-d11}, [%1]!     \n"
                    "pld        [%2, #32]           \n"
                    "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q8, q4, q0          \n"
                    "vmla.f32   q9, q5, q0          \n"
                    "bne        2b                  \n"

                    "3:                             \n"
                    "vst1.f32   {d16-d19}, [%0]     \n"

                    : "=r"(output), // %0
                    "=r"(vb),     // %1
                    "=r"(va)      // %2
                    : "0"(output),
                    "1"(vb),
                    "2"(va),
                    "r"(L),    // %6
                    "r"(bias0) // %7
                    : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else
                float sum[8] = {0};

                int k = 0;
                for (; k + 7 < L; k = k + 8)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum[n] += va[0] * vb[n];
                        sum[n] += va[1] * vb[n + 8];
                        sum[n] += va[2] * vb[n + 16];
                        sum[n] += va[3] * vb[n + 24];
                        sum[n] += va[4] * vb[n + 32];
                        sum[n] += va[5] * vb[n + 40];
                        sum[n] += va[6] * vb[n + 48];
                        sum[n] += va[7] * vb[n + 56];
                    }

                    va += 8;
                    vb += 64;
                }

                for (; k < L; k++)
                {
                    for (int n = 0; n < 8; n++)
                    {
                        sum[n] += va[0] * vb[n];
                    }

                    va += 1;
                    vb += 8;
                }

                for (int n = 0; n < 8; n++)
                {
                    output[n] = sum[n] + bias0;
                }
#endif // __ARM_NEON
                output += 8;
            }

            for (; j < N; j++)
            {
                const float* vb = bottom_tm.channel(j / 8 + j % 8);
#if __ARM_NEON && __aarch64__
                const float* va = kernel_tm.channel(i / 8 + (i % 8) / 4 + i % 4);
#else
                const float* va = kernel_tm.channel(i / 4 + i % 4);
#endif // __ARM_NEON && __aarch64__

                int k = 0;
#if __ARM_NEON
                float32x4_t _sum0 = vdupq_n_f32(0.f);

                for (; k + 3 < L; k += 4)
                {
                    float32x4_t _p0 = vld1q_f32(vb);
                    vb += 4;

                    float32x4_t _k0 = vld1q_f32(va);
                    va += 4;

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
                for (; k < L; k++)
                {
                    sum0 += va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = sum0;

                output++;
            }
        }
    }
}
