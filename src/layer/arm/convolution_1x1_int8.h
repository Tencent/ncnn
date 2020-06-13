// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// author:BUG1989 (https://github.com/BUG1989/) Long-term support.
// author:FuGuangping (https://github.com/fu1899) Implemented the first version of INT8 quantization on ARMv7.
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

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

#if __aarch64__
#if 1
#include "gemm_symm_int8.h"
static void conv1x1s1_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(outch, inch, (size_t)1u);
    const int8_t* a = _kernel;
    int8_t* sa = kernel_tm;
    reorder_a((int8_t*)a, sa, outch, inch, inch);
}

static void conv1x1s1_sgemm_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const size_t n = bottom_blob.w * bottom_blob.h;
    const size_t k = bottom_blob.c;
    const size_t m = top_blob.c;

    ncnn::Mat bottom_tm(k * n, (size_t)1u, opt.workspace_allocator);
    {
        const int8_t* pData = bottom_blob;
        int8_t* pReorder = bottom_tm;
        reorder_b(pData, pReorder, k, n, bottom_blob.cstep);
    }

    // GEMM
    int32_t* pc = top_blob;
    const int8_t* pa = kernel;
    const int8_t* pb = bottom_tm;
    const size_t ldc = top_blob.cstep;

    int8kernel((void*)pc, pa, pb, m, k, n, ldc, 0, 0, opt);
}

static void conv1x1s1_sgemm_int8_requant_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, std::vector<float> scales_requant, const Option& opt)
{
    const size_t n = bottom_blob.w * bottom_blob.h;
    const size_t k = bottom_blob.c;
    const size_t m = top_blob.c;

    ncnn::Mat scales_tm(m);
    ncnn::Mat bias_tm(m);
    float* scales = scales_tm;
    const float* bias = _bias;

    // outptr0[0]  = float2int8(((float)sum0 * scale_requant_in + bias0) * scale_requant_out);
    // the equation could convert to:
    //      out = float2int8( (float)sum * (scale_requant_in * scale_requant_out) + (bias * scale_requant_out) )
    // prebuild the list of (scales_requant_in*scale_requant_out)
    for (size_t i = 0; i < m; ++i)
    {
        scales_tm[i] = scales_requant[2 * i] * scales_requant[2 * i + 1];
    }
    if (!_bias.empty())
    {
        for (size_t i = 0; i < m; ++i)
        {
            bias_tm[i] = bias[i] * scales_requant[2 * i + 1];
        }
        bias = bias_tm;
    }

    ncnn::Mat bottom_tm(k * n, (size_t)1u, opt.workspace_allocator);
    {
        const int8_t* pData = bottom_blob;
        int8_t* pReorder = bottom_tm;
        reorder_b(pData, pReorder, k, n, bottom_blob.cstep);
    }

    // GEMM
    int8_t* pc = top_blob;
    const int8_t* pa = kernel;
    const int8_t* pb = bottom_tm;
    const size_t ldc = top_blob.cstep;
    int8kernel((void*)pc, pa, pb, m, k, n, ldc, scales, (float*)bias, opt);
}
#else
static void conv1x1s1_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const signed char* kernel = _kernel;

    // kernel memory packed 4 x 4
    kernel_tm.create(4 * 4, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;

    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        const signed char* k0 = kernel + (p + 0) * inch;
        const signed char* k1 = kernel + (p + 1) * inch;
        const signed char* k2 = kernel + (p + 2) * inch;
        const signed char* k3 = kernel + (p + 3) * inch;

        signed char* ktmp = kernel_tm.channel(p / 4);

        int q = 0;
        for (; q + 1 < inch; q += 2)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k0[1];
            ktmp[2] = k1[0];
            ktmp[3] = k1[1];
            ktmp[4] = k2[0];
            ktmp[5] = k2[1];
            ktmp[6] = k3[0];
            ktmp[7] = k3[1];

            ktmp += 8;

            k0 += 2;
            k1 += 2;
            k2 += 2;
            k3 += 2;
        }

        for (; q < inch; q++)
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

    for (int p = remain_outch_start; p < outch; p++)
    {
        const signed char* k0 = kernel + (p + 0) * inch;

        signed char* ktmp = kernel_tm.channel(p / 4 + p % 4);

        int q = 0;
        for (; q + 1 < inch; q = q + 2)
        {
            ktmp[0] = k0[0];
            ktmp[1] = k0[1];
            ktmp += 2;
            k0 += 2;
        }

        for (; q < inch; q++)
        {
            ktmp[0] = k0[0];
            ktmp++;
            k0++;
        }
    }
}

static void conv1x1s1_sgemm_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    // bottom_tm memory packed 4 x 4
    ncnn::Mat bottom_tm(4, inch, size / 4 + size % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const signed char* img0 = bottom_blob.channel(0);
            const signed char* img1 = bottom_blob.channel(1);
            img0 += i;
            img1 += i;

            signed char* tmpptr = bottom_tm.channel(i / 4);

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img1[0];
                tmpptr[2] = img0[1];
                tmpptr[3] = img1[1];
                tmpptr[4] = img0[2];
                tmpptr[5] = img1[2];
                tmpptr[6] = img0[3];
                tmpptr[7] = img1[3];

                tmpptr += 8;
                img0 += bottom_blob.cstep;
                img0 += bottom_blob.cstep;
                img1 += bottom_blob.cstep;
                img1 += bottom_blob.cstep;
            }

            for (; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);
        int* outptr2 = top_blob.channel(p + 2);
        int* outptr3 = top_blob.channel(p + 3);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4);
            const signed char* kptr = kernel.channel(p / 4);
#if __ARM_NEON
            asm volatile(
                "prfm   pldl1keep, [%4, #128]        \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                "lsr    w4, %w12, #2                 \n" // r4 = nn = L >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n" // for (; k+3<L; k=k+4)
                "ld1    {v0.16b}, [%4]               \n" // i0, i1, i2, i3
                "ld1    {v4.16b}, [%5]               \n" // k0, k1, k2, k3
                "add    %4, %4, #16                  \n"
                "add    %5, %5, #16                  \n"

                "rev32    v1.8h, v0.8h               \n" // i1, i0, i3, i2
                "rev64    v2.4s, v0.4s               \n" // i2, i3, i0, i1
                "rev64    v3.8h, v0.8h               \n" // i3, i2, i1, i0

                "smull	  v8.8h, v4.8b, v0.8b        \n"
                "smull	  v9.8h, v4.8b, v1.8b        \n"
                "smull	  v10.8h, v4.8b, v2.8b       \n"
                "smull	  v11.8h, v4.8b, v3.8b       \n"

                "prfm     pldl1keep, [%4, #1024]     \n"
                "prfm     pldl1keep, [%5, #1024]     \n"

                "smlal2	  v8.8h, v4.16b, v0.16b      \n"
                "smlal2	  v9.8h, v4.16b, v1.16b      \n"
                "smlal2	  v10.8h, v4.16b, v2.16b     \n"
                "smlal2	  v11.8h, v4.16b, v3.16b     \n"

                "sadalp	  v16.4s, v8.8h              \n" // i0k0, i1k1, i2k2, i3k3
                "sadalp	  v17.4s, v9.8h              \n" // i1k0, i0k1, i3k2, i2k3
                "sadalp	  v18.4s, v10.8h             \n" // i2k0, i3k1, i0k2, i1k3
                "sadalp	  v19.4s, v11.8h             \n" // i3k0, i2k1, i1k2, i0k3

                "subs     w4, w4, #1                 \n"
                "bne      0b                         \n"

                "1:                                  \n" // for (; k+1<L; k=k+2)

                // remain loop
                "and      w4, %w12, #3               \n" // w4 = remain = K & 3;
                "cmp      w4, #0                     \n"
                "beq      3f                         \n"

                "lsr      w4, w4, #1                 \n" // r4 = nn = L >> 1
                "cmp      w4, #0                     \n"
                "beq      3f                         \n"

                "2:                                  \n" // for (; k+1<L; k=k+2)

                "ld1      {v0.8b}, [%4]              \n" // i0, i1, i2, i3
                "ld1      {v4.8b}, [%5]              \n" // k0, k1, k2, k3
                "add      %4, %4, #8                 \n"
                "add      %5, %5, #8                 \n"

                "rev32	  v1.4h, v0.4h               \n" // i2, i3, i0, i1
                "rev64    v2.2s, v0.2s               \n" // i1, i0, i3, i2
                "rev64    v3.4h, v0.4h               \n" // i0, i1, i2, i3

                "smull	  v8.8h, v4.8b, v0.8b        \n"
                "smull	  v9.8h, v4.8b, v1.8b        \n"
                "smull    v10.8h, v4.8b, v2.8b       \n"
                "smull	  v11.8h, v4.8b, v3.8b       \n"
                "sadalp	  v16.4s, v8.8h              \n"
                "sadalp	  v17.4s, v9.8h              \n"
                "sadalp	  v18.4s,v10.8h              \n"
                "sadalp	  v19.4s,v11.8h              \n"

                "subs     w4, w4, #1                 \n"
                "bne      2b                         \n"

                "3:                                  \n" // realloc

                "mov      v20.s[0], v16.s[0]         \n"
                "mov      v20.s[1], v17.s[0]         \n"
                "mov      v20.s[2], v18.s[0]         \n"
                "mov      v20.s[3], v19.s[0]         \n"

                "mov      v21.s[0], v17.s[1]         \n"
                "mov      v21.s[1], v16.s[1]         \n"
                "mov      v21.s[2], v19.s[1]         \n"
                "mov      v21.s[3], v18.s[1]         \n"

                "mov      v22.s[0], v18.s[2]         \n"
                "mov      v22.s[1], v19.s[2]         \n"
                "mov      v22.s[2], v16.s[2]         \n"
                "mov      v22.s[3], v17.s[2]         \n"

                "mov      v23.s[0], v19.s[3]         \n"
                "mov      v23.s[1], v18.s[3]         \n"
                "mov      v23.s[2], v17.s[3]         \n"
                "mov      v23.s[3], v16.s[3]         \n"

                "and      w4, %w12, #1               \n" // w4 = remain = K & 1;
                "cmp      w4, #0                     \n"
                "beq      5f                         \n"

                "4:                                  \n"
                "ld1      {v0.8b}, [%4]              \n"
                "ld1      {v1.8b}, [%5]              \n"
                "add      %4, %4, #4                 \n"
                "add      %5, %5, #4                 \n"

                "sshll    v0.8h, v0.8b, #0           \n" // i0[0], i1[0], i2[0], i3[0]
                "sshll    v1.8h, v1.8b, #0           \n" // k0[0], k1[0], k2[0], k3[0]

                "smlal    v20.4s, v0.4h, v1.h[0]     \n" // i0k0, i1k0, i2k0, i3k0
                "smlal    v21.4s, v0.4h, v1.h[1]     \n" // i0k1, i1k1, i2k1, i3k1
                "smlal    v22.4s, v0.4h, v1.h[2]     \n" // i0k2, i1k2, i2k2, i3k2
                "smlal    v23.4s, v0.4h, v1.h[3]     \n" // i0k3, i1k3, i2k3, i3k3

                "subs     w4, w4, #1                 \n"

                "bne      2b                         \n"

                "5:                                  \n"

                "st1      {v20.4s}, [%0]             \n"
                "st1      {v21.4s}, [%1]             \n"
                "st1      {v22.4s}, [%2]             \n"
                "st1      {v23.4s}, [%3]             \n"

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
                "r"(inch) // %12
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_0 += tmpptr[1] * kptr[1];
                sum0_1 += tmpptr[2] * kptr[0];
                sum0_1 += tmpptr[3] * kptr[1];
                sum0_2 += tmpptr[4] * kptr[0];
                sum0_2 += tmpptr[5] * kptr[1];
                sum0_3 += tmpptr[6] * kptr[0];
                sum0_3 += tmpptr[7] * kptr[1];

                sum1_0 += tmpptr[0] * kptr[2];
                sum1_0 += tmpptr[1] * kptr[3];
                sum1_1 += tmpptr[2] * kptr[2];
                sum1_1 += tmpptr[3] * kptr[3];
                sum1_2 += tmpptr[4] * kptr[2];
                sum1_2 += tmpptr[5] * kptr[3];
                sum1_3 += tmpptr[6] * kptr[2];
                sum1_3 += tmpptr[7] * kptr[3];

                sum2_0 += tmpptr[0] * kptr[4];
                sum2_0 += tmpptr[1] * kptr[5];
                sum2_1 += tmpptr[2] * kptr[4];
                sum2_1 += tmpptr[3] * kptr[5];
                sum2_2 += tmpptr[4] * kptr[4];
                sum2_2 += tmpptr[5] * kptr[5];
                sum2_3 += tmpptr[6] * kptr[4];
                sum2_3 += tmpptr[7] * kptr[5];

                sum3_0 += tmpptr[0] * kptr[6];
                sum3_0 += tmpptr[1] * kptr[7];
                sum3_1 += tmpptr[2] * kptr[6];
                sum3_1 += tmpptr[3] * kptr[7];
                sum3_2 += tmpptr[4] * kptr[6];
                sum3_2 += tmpptr[5] * kptr[7];
                sum3_3 += tmpptr[6] * kptr[6];
                sum3_3 += tmpptr[7] * kptr[7];

                tmpptr += 8;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
#endif
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }

        for (; i < size; i++)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4);
#if 0 //__ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);

            int q=0;
            for (; q+3<inch; q=q+4)
            {
                int8x8_t _r0 = vld1_s8(tmpptr);   // i0[0-3]
                int8x8x2_t _k = vld2_s8(kptr);    // k0[0-1], k1[0-1], k2[0-1], k3[0-1];k0[2-3], k1[2-3], k2[2-3], k3[2-3]

                int16x8_t _r0_s16 = vmovl_s8(_r0);          // i0[0],i0[1],i0[2],i0[3]
                int16x8_t _k02_s16 = vmovl_s8(_k.val[0]);   // k0[0],k1[0],k2[0],k3[0],k0[2],k1[2],k2[2],k3[2]
                int16x8_t _k13_s16 = vmovl_s8(_k.val[1]);   // k0[1],k1[1],k2[1],k3[1],k0[3],k1[3],k2[3],k3[3]

                _sum = vmlal_lane_s16(_sum, vget_low_s16(_k02_s16), vget_low_s16(_r0_s16), 0);    // i0[0]*k[0-3][0]
                _sum = vmlal_lane_s16(_sum, vget_low_s16(_k13_s16), vget_low_s16(_r0_s16), 1);    // i0[1]*k[0-3][1]
                _sum = vmlal_lane_s16(_sum, vget_high_s16(_k02_s16), vget_low_s16(_r0_s16), 2);   // i0[2]*k[0-3][2]
                _sum = vmlal_lane_s16(_sum, vget_high_s16(_k13_s16), vget_low_s16(_r0_s16), 3);   // i0[3]*k[0-3][3]

                tmpptr += 4;
                kptr += 16;
            }

            for (; q+1<inch; q=q+2)
            {
                int8x8_t _r0 = vld1_s8(tmpptr);   // i0[0-3]
                int8x8_t _k = vld1_s8(kptr);      // k0[0-1], k1[0-1], k2[0-1], k3[0-1]

                _r0[2] = _r0[0];
                _r0[3] = _r0[1];
                _r0[4] = _r0[0];
                _r0[5] = _r0[1];
                _r0[6] = _r0[0];
                _r0[7] = _r0[1];

                int16x8_t _tp0 = vmull_s8(_k, _r0);
                _sum = vpadalq_s16(_sum, _tp0);

                tmpptr += 2;
                kptr += 8;
            }

            for (; q<inch; q++)
            {
                int8x8_t _r0 = vld1_s8(tmpptr);   // i0[0-3]
                int8x8_t _k = vld1_s8(kptr);      // k[0-3][0]

                int16x8_t _tp0 = vmull_s8(_k, _r0);

                _sum = vaddw_s16(_sum, vget_low_s16(_tp0));

                tmpptr += 1;
                kptr += 4;
            }

            vst1q_lane_s32(outptr0, _sum, 0);
            vst1q_lane_s32(outptr1, _sum, 1);
            vst1q_lane_s32(outptr2, _sum, 2);
            vst1q_lane_s32(outptr3, _sum, 3);
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum0 += tmpptr[1] * kptr[1];

                sum1 += tmpptr[0] * kptr[2];
                sum1 += tmpptr[1] * kptr[3];

                sum2 += tmpptr[0] * kptr[4];
                sum2 += tmpptr[1] * kptr[5];

                sum3 += tmpptr[0] * kptr[6];
                sum3 += tmpptr[1] * kptr[7];

                tmpptr += 2;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];

                tmpptr += 1;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;
#endif
            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        int* outptr0 = out0;

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if __ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0-1], i1[0-1], i2[0-1], i3[0-1]
                int8x8_t _k = vld1_s8(kptr);    // k0[0-1]

                _k[2] = _k[0];
                _k[3] = _k[1];
                _k[4] = _k[0];
                _k[5] = _k[1];
                _k[6] = _k[0];
                _k[7] = _k[1];

                int16x8_t _tp0 = vmull_s8(_k, _r0);
                _sum = vpadalq_s16(_sum, _tp0);

                tmpptr += 8;
                kptr += 2;
            }

            for (; q < inch; q++)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0], i1[0], i2[0], i3[0]
                int8x8_t _k = vld1_s8(kptr);    // k[0][0]

                int16x8_t _r0_s16 = vmovl_s8(_r0);
                int16x8_t _k_s16 = vmovl_s8(_k);

                _sum = vmlal_lane_s16(_sum, vget_low_s16(_r0_s16), vget_low_s16(_k_s16), 0); // i0k0, i1k0, i2k0, i3k0

                tmpptr += 4;
                kptr += 1;
            }

            vst1q_s32(outptr0, _sum);
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum0 += tmpptr[1] * kptr[1];

                sum1 += tmpptr[2] * kptr[0];
                sum1 += tmpptr[3] * kptr[1];

                sum2 += tmpptr[4] * kptr[0];
                sum2 += tmpptr[5] * kptr[1];

                sum3 += tmpptr[6] * kptr[0];
                sum3 += tmpptr[7] * kptr[1];

                tmpptr += 8;
                kptr += 2;
            }

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
#endif
            outptr0 += 4;
        }

        for (; i < size; i++)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

            int q = 0;
            int sum0 = 0;

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
}

static void conv1x1s1_sgemm_int8_requant_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, std::vector<float> scales_requant, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;
    const float* bias = _bias;

    // bottom_tm memory packed 4 x 4
    ncnn::Mat bottom_tm(4, inch, size / 4 + size % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const signed char* img0 = bottom_blob.channel(0);
            const signed char* img1 = bottom_blob.channel(1);
            img0 += i;
            img1 += i;

            signed char* tmpptr = bottom_tm.channel(i / 4);

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img1[0];
                tmpptr[2] = img0[1];
                tmpptr[3] = img1[1];
                tmpptr[4] = img0[2];
                tmpptr[5] = img1[2];
                tmpptr[6] = img0[3];
                tmpptr[7] = img1[3];

                tmpptr += 8;
                img0 += bottom_blob.cstep;
                img0 += bottom_blob.cstep;
                img1 += bottom_blob.cstep;
                img1 += bottom_blob.cstep;
            }

            for (; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;
    remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        signed char* outptr0 = top_blob.channel(p);
        signed char* outptr1 = top_blob.channel(p + 1);
        signed char* outptr2 = top_blob.channel(p + 2);
        signed char* outptr3 = top_blob.channel(p + 3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;

        const float scale_requant_in0 = scales_requant[2 * p];
        const float scale_requant_out0 = scales_requant[2 * p + 1];
        const float scale_requant_in1 = scales_requant[2 * (p + 1)];
        const float scale_requant_out1 = scales_requant[2 * (p + 1) + 1];
        const float scale_requant_in2 = scales_requant[2 * (p + 2)];
        const float scale_requant_out2 = scales_requant[2 * (p + 2) + 1];
        const float scale_requant_in3 = scales_requant[2 * (p + 3)];
        const float scale_requant_out3 = scales_requant[2 * (p + 3) + 1];

        float32x4_t _bias03, _scale_in03, _scale_out03;
        float32x4_t _bias0 = vdupq_n_f32(bias0);
        float32x4_t _bias1 = vdupq_n_f32(bias1);
        float32x4_t _bias2 = vdupq_n_f32(bias2);
        float32x4_t _bias3 = vdupq_n_f32(bias3);

        _bias03[0] = bias0;
        _bias03[1] = bias1;
        _bias03[2] = bias2;
        _bias03[3] = bias3;

        _scale_in03[0] = scale_requant_in0;
        _scale_in03[1] = scale_requant_in1;
        _scale_in03[2] = scale_requant_in2;
        _scale_in03[3] = scale_requant_in3;

        _scale_out03[0] = scale_requant_out0;
        _scale_out03[1] = scale_requant_out1;
        _scale_out03[2] = scale_requant_out2;
        _scale_out03[3] = scale_requant_out3;

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4);
            const signed char* kptr = kernel.channel(p / 4);
#if 1 //__ARM_NEON
            asm volatile(
                "prfm   pldl1keep, [%4, #128]        \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                "lsr    w4, %w12, #2                 \n" // r4 = nn = L >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n" // for (; k+3<L; k=k+4)
                "ld1    {v0.16b}, [%4]               \n" // i0, i1, i2, i3
                "ld1    {v4.16b}, [%5]               \n" // k0, k1, k2, k3
                "add    %4, %4, #16                  \n"
                "add    %5, %5, #16                  \n"

                "rev32    v1.8h, v0.8h               \n" // i1, i0, i3, i2
                "rev64    v2.4s, v0.4s               \n" // i2, i3, i0, i1
                "rev64    v3.8h, v0.8h               \n" // i3, i2, i1, i0

                "smull	  v8.8h, v4.8b, v0.8b        \n"
                "smull	  v9.8h, v4.8b, v1.8b        \n"
                "smull	  v10.8h, v4.8b, v2.8b       \n"
                "smull	  v11.8h, v4.8b, v3.8b       \n"

                "prfm     pldl1keep, [%4, #1024]     \n"
                "prfm     pldl1keep, [%5, #1024]     \n"

                "smlal2	  v8.8h, v4.16b, v0.16b      \n"
                "smlal2	  v9.8h, v4.16b, v1.16b      \n"
                "smlal2	  v10.8h, v4.16b, v2.16b     \n"
                "smlal2	  v11.8h, v4.16b, v3.16b     \n"

                "sadalp	  v16.4s, v8.8h              \n" // i0k0, i1k1, i2k2, i3k3
                "sadalp	  v17.4s, v9.8h              \n" // i1k0, i0k1, i3k2, i2k3
                "sadalp	  v18.4s, v10.8h             \n" // i2k0, i3k1, i0k2, i1k3
                "sadalp	  v19.4s, v11.8h             \n" // i3k0, i2k1, i1k2, i0k3

                "subs     w4, w4, #1                 \n"
                "bne      0b                         \n"

                "1:                                  \n" // for (; k+1<L; k=k+2)

                // remain loop
                "and      w4, %w12, #3               \n" // w4 = remain = K & 3;
                "cmp      w4, #0                     \n"
                "beq      3f                         \n"

                "lsr      w4, w4, #1                 \n" // r4 = nn = L >> 1
                "cmp      w4, #0                     \n"
                "beq      3f                         \n"

                "2:                                  \n" // for (; k+1<L; k=k+2)

                "ld1      {v0.8b}, [%4]              \n" // i0, i1, i2, i3
                "ld1      {v4.8b}, [%5]              \n" // k0, k1, k2, k3
                "add      %4, %4, #8                 \n"
                "add      %5, %5, #8                 \n"

                "rev32	  v1.4h, v0.4h               \n" // i2, i3, i0, i1
                "rev64    v2.2s, v0.2s               \n" // i1, i0, i3, i2
                "rev64    v3.4h, v0.4h               \n" // i0, i1, i2, i3

                "smull	  v8.8h, v4.8b, v0.8b        \n"
                "smull	  v9.8h, v4.8b, v1.8b        \n"
                "smull    v10.8h, v4.8b, v2.8b       \n"
                "smull	  v11.8h, v4.8b, v3.8b       \n"
                "sadalp	  v16.4s, v8.8h              \n"
                "sadalp	  v17.4s, v9.8h              \n"
                "sadalp	  v18.4s,v10.8h              \n"
                "sadalp	  v19.4s,v11.8h              \n"

                "subs     w4, w4, #1                 \n"
                "bne      2b                         \n"

                "3:                                  \n" // realloc

                "mov      v20.s[0], v16.s[0]         \n"
                "mov      v20.s[1], v17.s[0]         \n"
                "mov      v20.s[2], v18.s[0]         \n"
                "mov      v20.s[3], v19.s[0]         \n"

                "mov      v21.s[0], v17.s[1]         \n"
                "mov      v21.s[1], v16.s[1]         \n"
                "mov      v21.s[2], v19.s[1]         \n"
                "mov      v21.s[3], v18.s[1]         \n"

                "mov      v22.s[0], v18.s[2]         \n"
                "mov      v22.s[1], v19.s[2]         \n"
                "mov      v22.s[2], v16.s[2]         \n"
                "mov      v22.s[3], v17.s[2]         \n"

                "mov      v23.s[0], v19.s[3]         \n"
                "mov      v23.s[1], v18.s[3]         \n"
                "mov      v23.s[2], v17.s[3]         \n"
                "mov      v23.s[3], v16.s[3]         \n"

                "and      w4, %w12, #1               \n" // w4 = remain = K & 1;
                "cmp      w4, #0                     \n"
                "beq      5f                         \n"

                "4:                                  \n"
                "ld1      {v0.8b}, [%4]              \n"
                "ld1      {v1.8b}, [%5]              \n"
                "add      %4, %4, #4                 \n"
                "add      %5, %5, #4                 \n"

                "sshll    v0.8h, v0.8b, #0           \n" // i0[0], i1[0], i2[0], i3[0]
                "sshll    v1.8h, v1.8b, #0           \n" // k0[0], k1[0], k2[0], k3[0]

                "smlal    v20.4s, v0.4h, v1.h[0]     \n" // i0k0, i1k0, i2k0, i3k0
                "smlal    v21.4s, v0.4h, v1.h[1]     \n" // i0k1, i1k1, i2k1, i3k1
                "smlal    v22.4s, v0.4h, v1.h[2]     \n" // i0k2, i1k2, i2k2, i3k2
                "smlal    v23.4s, v0.4h, v1.h[3]     \n" // i0k3, i1k3, i2k3, i3k3

                "subs     w4, w4, #1                 \n"

                "bne      2b                         \n"

                "5:                                  \n"
                // top_s32 -> top_f32
                "scvtf  v20.4s, v20.4s               \n"
                "scvtf  v21.4s, v21.4s               \n"
                "scvtf  v22.4s, v22.4s               \n"
                "scvtf  v23.4s, v23.4s               \n"
                // top_f32 = top_f32 * scale_in
                "fmul   v20.4s, v20.4s, %17.s[0]     \n"
                "fmul   v21.4s, v21.4s, %17.s[1]     \n"
                "fmul   v22.4s, v22.4s, %17.s[2]     \n"
                "fmul   v23.4s, v23.4s, %17.s[3]     \n"
                // top_f32 = top_f32 + bias
                "fadd   v20.4s, v20.4s, %13.4s       \n"
                "fadd   v21.4s, v21.4s, %14.4s       \n"
                "fadd   v22.4s, v22.4s, %15.4s       \n"
                "fadd   v23.4s, v23.4s, %16.4s       \n"
                // top_f32 = top_f32 * scale_out
                "fmul   v20.4s, v20.4s, %18.s[0]     \n"
                "fmul   v21.4s, v21.4s, %18.s[1]     \n"
                "fmul   v22.4s, v22.4s, %18.s[2]     \n"
                "fmul   v23.4s, v23.4s, %18.s[3]     \n"
                // top_f32 -> top_s32
                "fcvtas v20.4s, v20.4s               \n"
                "fcvtas v21.4s, v21.4s               \n"
                "fcvtas v22.4s, v22.4s               \n"
                "fcvtas v23.4s, v23.4s               \n"
                // top_s32 -> top_s16
                "sqxtn  v7.4h, v20.4s                \n"
                "sqxtn2 v7.8h, v21.4s                 \n"
                "sqxtn  v8.4h, v22.4s                 \n"
                "sqxtn2 v8.8h, v23.4s                 \n"
                // top_s16 -> top_s8
                "sqxtn  v0.8b, v7.8h                 \n"
                "sqxtn  v1.8b, v8.8h                 \n"
                // save top_s8
                "st1    {v0.s}[0], [%0]              \n"
                "st1    {v0.s}[1], [%1]              \n"
                "st1    {v1.s}[0], [%2]              \n"
                "st1    {v1.s}[1], [%3]              \n"

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
                "r"(inch),        // %12
                "w"(_bias0),      // %13
                "w"(_bias1),      // %14
                "w"(_bias2),      // %15
                "w"(_bias3),      // %16
                "w"(_scale_in03), // %17
                "w"(_scale_out03) // %18
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_0 += tmpptr[1] * kptr[1];
                sum0_1 += tmpptr[2] * kptr[0];
                sum0_1 += tmpptr[3] * kptr[1];
                sum0_2 += tmpptr[4] * kptr[0];
                sum0_2 += tmpptr[5] * kptr[1];
                sum0_3 += tmpptr[6] * kptr[0];
                sum0_3 += tmpptr[7] * kptr[1];

                sum1_0 += tmpptr[0] * kptr[2];
                sum1_0 += tmpptr[1] * kptr[3];
                sum1_1 += tmpptr[2] * kptr[2];
                sum1_1 += tmpptr[3] * kptr[3];
                sum1_2 += tmpptr[4] * kptr[2];
                sum1_2 += tmpptr[5] * kptr[3];
                sum1_3 += tmpptr[6] * kptr[2];
                sum1_3 += tmpptr[7] * kptr[3];

                sum2_0 += tmpptr[0] * kptr[4];
                sum2_0 += tmpptr[1] * kptr[5];
                sum2_1 += tmpptr[2] * kptr[4];
                sum2_1 += tmpptr[3] * kptr[5];
                sum2_2 += tmpptr[4] * kptr[4];
                sum2_2 += tmpptr[5] * kptr[5];
                sum2_3 += tmpptr[6] * kptr[4];
                sum2_3 += tmpptr[7] * kptr[5];

                sum3_0 += tmpptr[0] * kptr[6];
                sum3_0 += tmpptr[1] * kptr[7];
                sum3_1 += tmpptr[2] * kptr[6];
                sum3_1 += tmpptr[3] * kptr[7];
                sum3_2 += tmpptr[4] * kptr[6];
                sum3_2 += tmpptr[5] * kptr[7];
                sum3_3 += tmpptr[6] * kptr[6];
                sum3_3 += tmpptr[7] * kptr[7];

                tmpptr += 8;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = float2int8(((float)sum0_0 * scale_requant_in0 + bias0) * scale_requant_out0);
            outptr0[1] = float2int8(((float)sum0_1 * scale_requant_in0 + bias0) * scale_requant_out0);
            outptr0[2] = float2int8(((float)sum0_2 * scale_requant_in0 + bias0) * scale_requant_out0);
            outptr0[3] = float2int8(((float)sum0_3 * scale_requant_in0 + bias0) * scale_requant_out0);

            outptr1[0] = float2int8(((float)sum1_0 * scale_requant_in1 + bias1) * scale_requant_out1);
            outptr1[1] = float2int8(((float)sum1_1 * scale_requant_in1 + bias1) * scale_requant_out1);
            outptr1[2] = float2int8(((float)sum1_2 * scale_requant_in1 + bias1) * scale_requant_out1);
            outptr1[3] = float2int8(((float)sum1_3 * scale_requant_in1 + bias1) * scale_requant_out1);

            outptr2[0] = float2int8(((float)sum2_0 * scale_requant_in2 + bias2) * scale_requant_out2);
            outptr2[1] = float2int8(((float)sum2_1 * scale_requant_in2 + bias2) * scale_requant_out2);
            outptr2[2] = float2int8(((float)sum2_2 * scale_requant_in2 + bias2) * scale_requant_out2);
            outptr2[3] = float2int8(((float)sum2_3 * scale_requant_in2 + bias2) * scale_requant_out2);

            outptr3[0] = float2int8(((float)sum3_0 * scale_requant_in3 + bias3) * scale_requant_out3);
            outptr3[1] = float2int8(((float)sum3_1 * scale_requant_in3 + bias3) * scale_requant_out3);
            outptr3[2] = float2int8(((float)sum3_2 * scale_requant_in3 + bias3) * scale_requant_out3);
            outptr3[3] = float2int8(((float)sum3_3 * scale_requant_in3 + bias3) * scale_requant_out3);
#endif
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }

        for (; i < size; i++)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4);
#if 1 //__ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);

            int q = 0;
            for (; q + 3 < inch; q = q + 4)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0-3]
                int8x8x2_t _k = vld2_s8(kptr);  // k0[0-1], k1[0-1], k2[0-1], k3[0-1];k0[2-3], k1[2-3], k2[2-3], k3[2-3]

                int16x8_t _r0_s16 = vmovl_s8(_r0);        // i0[0],i0[1],i0[2],i0[3]
                int16x8_t _k02_s16 = vmovl_s8(_k.val[0]); // k0[0],k1[0],k2[0],k3[0],k0[2],k1[2],k2[2],k3[2]
                int16x8_t _k13_s16 = vmovl_s8(_k.val[1]); // k0[1],k1[1],k2[1],k3[1],k0[3],k1[3],k2[3],k3[3]

                _sum = vmlal_lane_s16(_sum, vget_low_s16(_k02_s16), vget_low_s16(_r0_s16), 0);  // i0[0]*k[0-3][0]
                _sum = vmlal_lane_s16(_sum, vget_low_s16(_k13_s16), vget_low_s16(_r0_s16), 1);  // i0[1]*k[0-3][1]
                _sum = vmlal_lane_s16(_sum, vget_high_s16(_k02_s16), vget_low_s16(_r0_s16), 2); // i0[2]*k[0-3][2]
                _sum = vmlal_lane_s16(_sum, vget_high_s16(_k13_s16), vget_low_s16(_r0_s16), 3); // i0[3]*k[0-3][3]

                tmpptr += 4;
                kptr += 16;
            }

            for (; q + 1 < inch; q = q + 2)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0-3]
                int8x8_t _k = vld1_s8(kptr);    // k0[0-1], k1[0-1], k2[0-1], k3[0-1]

                _r0[2] = _r0[0];
                _r0[3] = _r0[1];
                _r0[4] = _r0[0];
                _r0[5] = _r0[1];
                _r0[6] = _r0[0];
                _r0[7] = _r0[1];

                int16x8_t _tp0 = vmull_s8(_k, _r0);
                _sum = vpadalq_s16(_sum, _tp0);

                tmpptr += 2;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0-3]
                int8x8_t _k = vld1_s8(kptr);    // k[0-3][0]

                int16x8_t _tp0 = vmull_s8(_k, _r0);

                _sum = vaddw_s16(_sum, vget_low_s16(_tp0));

                tmpptr += 1;
                kptr += 4;
            }

            // top_s32 -> top_f32
            float32x4_t _sum_f32 = vcvtq_f32_s32(_sum);
            // top_f32 = top_f32 * scale_in
            _sum_f32 = vmulq_f32(_sum_f32, _scale_in03);
            // top_f32 = top_f32 + bias
            _sum_f32 = vaddq_f32(_sum_f32, _bias03);
            // top_f32 = top_f32 * scale_out
            _sum_f32 = vmulq_f32(_sum_f32, _scale_out03);
            // top_f32 -> top_s32
            _sum = vcvtaq_s32_f32(_sum_f32);
            // top_s32 -> top_s16
            int16x4_t _sum_s16 = vqmovn_s32(_sum);
            int16x8_t _sum_s16_tp = vcombine_s16(_sum_s16, _sum_s16);
            // top_s16 -> top_s8
            int8x8_t _sum_s8 = vqmovn_s16(_sum_s16_tp);
            // save top_s8

            vst1_lane_s8(outptr0, _sum_s8, 0);
            vst1_lane_s8(outptr1, _sum_s8, 1);
            vst1_lane_s8(outptr2, _sum_s8, 2);
            vst1_lane_s8(outptr3, _sum_s8, 3);
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum0 += tmpptr[1] * kptr[1];

                sum1 += tmpptr[0] * kptr[2];
                sum1 += tmpptr[1] * kptr[3];

                sum2 += tmpptr[0] * kptr[4];
                sum2 += tmpptr[1] * kptr[5];

                sum3 += tmpptr[0] * kptr[6];
                sum3 += tmpptr[1] * kptr[7];

                tmpptr += 2;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];

                tmpptr += 1;
                kptr += 4;
            }

            outptr0[0] = float2int8(((float)sum0 * scale_requant_in0 + bias0) * scale_requant_out0);
            outptr1[0] = float2int8(((float)sum1 * scale_requant_in1 + bias1) * scale_requant_out1);
            outptr2[0] = float2int8(((float)sum2 * scale_requant_in2 + bias2) * scale_requant_out2);
            outptr3[0] = float2int8(((float)sum3 * scale_requant_in3 + bias3) * scale_requant_out3);
#endif
            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        signed char* outptr0 = out0;

        const float bias0 = bias ? bias[p] : 0.f;
        const float scale_requant_in = scales_requant[2 * p];
        const float scale_requant_out = scales_requant[2 * p + 1];

        float32x4_t _bias0 = vdupq_n_f32(bias0);
        float32x4_t _scale_in = vdupq_n_f32(scale_requant_in);
        float32x4_t _scale_out = vdupq_n_f32(scale_requant_out);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if 1 //__ARM_NEON
            int32x4_t _sum = vdupq_n_s32(0);

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0-1], i1[0-1], i2[0-1], i3[0-1]
                int8x8_t _k = vld1_s8(kptr);    // k0[0-1]

                _k[2] = _k[0];
                _k[3] = _k[1];
                _k[4] = _k[0];
                _k[5] = _k[1];
                _k[6] = _k[0];
                _k[7] = _k[1];

                int16x8_t _tp0 = vmull_s8(_k, _r0);
                _sum = vpadalq_s16(_sum, _tp0);

                tmpptr += 8;
                kptr += 2;
            }

            for (; q < inch; q++)
            {
                int8x8_t _r0 = vld1_s8(tmpptr); // i0[0], i1[0], i2[0], i3[0]
                int8x8_t _k = vld1_s8(kptr);    // k[0][0]

                int16x8_t _r0_s16 = vmovl_s8(_r0);
                int16x8_t _k_s16 = vmovl_s8(_k);

                _sum = vmlal_lane_s16(_sum, vget_low_s16(_r0_s16), vget_low_s16(_k_s16), 0); // i0k0, i1k0, i2k0, i3k0

                tmpptr += 4;
                kptr += 1;
            }

            // top_s32 -> top_f32
            float32x4_t _sum_f32 = vcvtq_f32_s32(_sum);
            // top_f32 = top_f32 * scale_in
            _sum_f32 = vmulq_f32(_sum_f32, _scale_in);
            // top_f32 = top_f32 + bias
            _sum_f32 = vaddq_f32(_sum_f32, _bias0);
            // top_f32 = top_f32 * scale_out
            _sum_f32 = vmulq_f32(_sum_f32, _scale_out);
            // top_f32 -> top_s32
            _sum = vcvtaq_s32_f32(_sum_f32);
            // top_s32 -> top_s16
            int16x4_t _sum_s16 = vqmovn_s32(_sum);
            int16x8_t _sum_s16_tp = vcombine_s16(_sum_s16, _sum_s16);
            // top_s16 -> top_s8
            int8x8_t _sum_s8 = vqmovn_s16(_sum_s16_tp);
            // save top_s8

            vst1_s8(outptr0, _sum_s8);
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            int q = 0;
            for (; q + 1 < inch; q = q + 2)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum0 += tmpptr[1] * kptr[1];

                sum1 += tmpptr[2] * kptr[0];
                sum1 += tmpptr[3] * kptr[1];

                sum2 += tmpptr[4] * kptr[0];
                sum2 += tmpptr[5] * kptr[1];

                sum3 += tmpptr[6] * kptr[0];
                sum3 += tmpptr[7] * kptr[1];

                tmpptr += 8;
                kptr += 2;
            }

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = float2int8(((float)sum0 * scale_requant_in + bias0) * scale_requant_out);
            outptr0[1] = float2int8(((float)sum1 * scale_requant_in + bias0) * scale_requant_out);
            outptr0[2] = float2int8(((float)sum2 * scale_requant_in + bias0) * scale_requant_out);
            outptr0[3] = float2int8(((float)sum3 * scale_requant_in + bias0) * scale_requant_out);
#endif
            outptr0 += 4;
        }

        for (; i < size; i++)
        {
            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

            int q = 0;
            int sum0 = 0;

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = float2int8(((float)sum0 * scale_requant_in + bias0) * scale_requant_out);

            outptr0++;
        }
    }
}

#endif
#else

static void conv1x1s1_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const signed char* kernel = _kernel;

    kernel_tm.create(4 * 4, inch / 4 + inch % 4, outch / 4 + outch % 4, (size_t)1u);

    int p = 0;
    for (; p + 3 < outch; p += 4)
    {
        const signed char* kernel0 = kernel + (p + 0) * inch;
        const signed char* kernel1 = kernel + (p + 1) * inch;
        const signed char* kernel2 = kernel + (p + 2) * inch;
        const signed char* kernel3 = kernel + (p + 3) * inch;

        signed char* ktmp = kernel_tm.channel(p / 4);

        for (int q = 0; q < inch; q++)
        {
            // kernel0...3 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];

            ktmp += 4;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
        }
    }

    for (; p < outch; p++)
    {
        const signed char* kernel0 = kernel + p * inch;
        signed char* ktmp = kernel_tm.channel(p / 4 + p % 4);

        for (int q = 0; q < inch; q++)
        {
            ktmp[0] = kernel0[0];
            ktmp++;
            kernel0++;
        }
    }
}

/*
 * Convolution 1x1 quantized with sgemm int8
 */
static void conv1x1s1_sgemm_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    // interleave
    Mat tmp(8 * 4, inch / 4 + inch % 4, size / 8 + (size % 8) / 4 + size % 4, 1u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
#if __ARM_NEON
                asm volatile(
                    "pld        [%0, #64]     \n"
                    "vld1.s8   {d0}, [%0]     \n"
                    "vst1.s8   {d0}, [%1]!    \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "d0");
                img0 += bottom_blob.cstep;
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

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

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

        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"
                "vmov.s32    q10, #0           \n"
                "vmov.s32    q11, #0           \n"
                "vmov.s32    q12, #0           \n"
                "vmov.s32    q13, #0           \n"

                "lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4-d7}, [%4]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n" // a30-a37
                "vmovl.s8    q4, d6            \n" // a20-a27
                "vmovl.s8    q3, d5            \n" // a10-a17
                "vmovl.s8    q2, d4            \n" // a00-a07

                "vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d4, d0[0]     \n" // sum0 = (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q8, d4, d0[1]     \n" // sum1 = (a00-a07) * k10
                "vmlal.s16   q9, d5, d0[1]     \n"
                "vmlal.s16   q10, d4, d0[2]    \n" // sum2 = (a00-a07) * k20
                "vmlal.s16   q11, d5, d0[2]    \n"
                "vmlal.s16   q12, d4, d0[3]    \n" // sum3 = (a00-a07) * k30
                "vmlal.s16   q13, d5, d0[3]    \n"

                "vmlal.s16   q6, d6, d1[0]     \n" // sum0 += (a10-a17) * k01
                "vmlal.s16   q7, d7, d1[0]     \n"
                "vmlal.s16   q8, d6, d1[1]     \n" // sum1 += (a10-a17) * k11
                "vmlal.s16   q9, d7, d1[1]     \n"
                "vmlal.s16   q10, d6, d1[2]    \n" // sum2 += (a10-a17) * k21
                "vmlal.s16   q11, d7, d1[2]    \n"
                "vmlal.s16   q12, d6, d1[3]    \n" // sum3 += (a10-a17) * k31
                "vmlal.s16   q13, d7, d1[3]    \n"

                "vmlal.s16   q6, d8, d2[0]     \n" // sum0 += (a20-a27) * k02
                "vmlal.s16   q7, d9, d2[0]     \n"
                "vmlal.s16   q8, d8, d2[1]     \n" // sum1 += (a20-a27) * k12
                "vmlal.s16   q9, d9, d2[1]     \n"
                "vmlal.s16   q10, d8, d2[2]    \n" // sum2 += (a20-a27) * k22
                "vmlal.s16   q11, d9, d2[2]    \n"
                "vmlal.s16   q12, d8, d2[3]    \n" // sum3 += (a20-a27) * k32
                "vmlal.s16   q13, d9, d2[3]    \n"

                "vmlal.s16   q6, d10, d3[0]    \n" // sum0 += (a30-a37) * k03
                "vmlal.s16   q7, d11, d3[0]    \n"
                "vmlal.s16   q8, d10, d3[1]    \n" // sum1 += (a30-a37) * k13
                "vmlal.s16   q9, d11, d3[1]    \n"
                "vmlal.s16   q10, d10, d3[2]   \n" // sum2 += (a30-a37) * k23
                "vmlal.s16   q11, d11, d3[2]   \n"
                "vmlal.s16   q12, d10, d3[3]   \n" // sum3 += (a30-a37) * k33
                "vmlal.s16   q13, d11, d3[3]   \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]!       \n" // tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // sum0 += (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"
                "vmlal.s16   q8, d2, d0[1]     \n" // sum1 += (a00-a07) * k10
                "vmlal.s16   q9, d3, d0[1]     \n"
                "vmlal.s16   q10, d2, d0[2]    \n" // sum2 += (a00-a07) * k20
                "vmlal.s16   q11, d3, d0[2]    \n"
                "vmlal.s16   q12, d2, d0[3]    \n" // sum3 += (a00-a07) * k30
                "vmlal.s16   q13, d3, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"
                "vst1.s32    {d16-d19}, [%1]!  \n"
                "vst1.s32    {d20-d23}, [%2]!  \n"
                "vst1.s32    {d24-d27}, [%3]!  \n"

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
                "r"(inch) // %12
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;
            int sum0_4 = 0;
            int sum0_5 = 0;
            int sum0_6 = 0;
            int sum0_7 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;
            int sum1_4 = 0;
            int sum1_5 = 0;
            int sum1_6 = 0;
            int sum1_7 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;
            int sum2_4 = 0;
            int sum2_5 = 0;
            int sum2_6 = 0;
            int sum2_7 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;
            int sum3_4 = 0;
            int sum3_5 = 0;
            int sum3_6 = 0;
            int sum3_7 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];
                sum0_4 += tmpptr[4] * kptr[0];
                sum0_5 += tmpptr[5] * kptr[0];
                sum0_6 += tmpptr[6] * kptr[0];
                sum0_7 += tmpptr[7] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];
                sum1_4 += tmpptr[4] * kptr[1];
                sum1_5 += tmpptr[5] * kptr[1];
                sum1_6 += tmpptr[6] * kptr[1];
                sum1_7 += tmpptr[7] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];
                sum2_4 += tmpptr[4] * kptr[2];
                sum2_5 += tmpptr[5] * kptr[2];
                sum2_6 += tmpptr[6] * kptr[2];
                sum2_7 += tmpptr[7] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];
                sum3_4 += tmpptr[4] * kptr[3];
                sum3_5 += tmpptr[5] * kptr[3];
                sum3_6 += tmpptr[6] * kptr[3];
                sum3_7 += tmpptr[7] * kptr[3];

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"

                "lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4-d5}, [%4]!    \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q3, d5            \n" // a20-a23,a30-a33
                "vmovl.s8    q2, d4            \n" // a00-a04,a10-a14

                "vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d4, d0[0]     \n" // sum0 = (a00-a03) * k00
                "vmlal.s16   q7, d4, d0[1]     \n" // sum1 = (a00-a03) * k10
                "vmlal.s16   q8, d4, d0[2]     \n" // sum2 = (a00-a03) * k20
                "vmlal.s16   q9, d4, d0[3]     \n" // sum3 = (a00-a03) * k30

                "vmlal.s16   q6, d5, d1[0]     \n" // sum0 += (a10-a13) * k01
                "vmlal.s16   q7, d5, d1[1]     \n" // sum1 += (a10-a13) * k11
                "vmlal.s16   q8, d5, d1[2]     \n" // sum2 += (a10-a13) * k21
                "vmlal.s16   q9, d5, d1[3]     \n" // sum3 += (a10-a13) * k31

                "vmlal.s16   q6, d6, d2[0]     \n" // sum0 += (a20-a23) * k02
                "vmlal.s16   q7, d6, d2[1]     \n" // sum1 += (a20-a23) * k12
                "vmlal.s16   q8, d6, d2[2]     \n" // sum2 += (a20-a23) * k22
                "vmlal.s16   q9, d6, d2[3]     \n" // sum3 += (a20-a23) * k32

                "vmlal.s16   q6, d7, d3[0]     \n" // sum0 += (a30-a33) * k03
                "vmlal.s16   q7, d7, d3[1]     \n" // sum1 += (a30-a33) * k13
                "vmlal.s16   q8, d7, d3[2]     \n" // sum2 += (a30-a33) * k23
                "vmlal.s16   q9, d7, d3[3]     \n" // sum3 += (a30-a33) * k33

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n" // tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #4            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // sum0 += (a00-a03) * k00
                "vmlal.s16   q7, d2, d0[1]     \n" // sum1 += (a00-a03) * k10
                "vmlal.s16   q8, d2, d0[2]     \n" // sum2 += (a00-a03) * k20
                "vmlal.s16   q9, d2, d0[3]     \n" // sum3 += (a00-a03) * k30

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory
                "vst1.s32    {d12-d13}, [%0]!  \n"
                "vst1.s32    {d14-d15}, [%1]!  \n"
                "vst1.s32    {d16-d17}, [%2]!  \n"
                "vst1.s32    {d18-d19}, [%3]!  \n"

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
                "r"(inch) // %12
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "veor        q6, q6, q6        \n"
                "veor        q7, q7, q7        \n"
                "veor        q8, q8, q8        \n"
                "veor        q9, q9, q9        \n"
                "vmov.s32    q10, #0           \n"

                "lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4}, [%4]        \n" // tmpr a00,a10,a20,a30    a(inch)(data)
                "add         %4, #4            \n"
                "vmovl.s8    q2, d4            \n" // a00,a10,a20,a30

                "vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d0, d4[0]     \n" // (k00-k30) * a00
                "vmlal.s16   q7, d1, d4[1]     \n" // (k01-k31) * a10
                "vmlal.s16   q8, d2, d4[2]     \n" // (k02-k32) * a20
                "vmlal.s16   q9, d3, d4[3]     \n" // (k03-k33) * a30

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "vadd.s32    q6, q6, q7        \n"
                "vadd.s32    q9, q9, q8        \n"
                "vadd.s32    q10, q6, q9       \n"

                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n" // tmpr a00        a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #1            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q10, d0, d2[0]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory
                "vst1.s32    {d20[0]}, [%0]!   \n"
                "vst1.s32    {d20[1]}, [%1]!   \n"
                "vst1.s32    {d21[0]}, [%2]!   \n"
                "vst1.s32    {d21[1]}, [%3]!   \n"

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
                "r"(inch) // %12
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

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

        int* outptr0 = out0;

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"

                "lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%1, #128]        \n"
                "vld1.s8     {d4-d7}, [%1]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n" // a30-a37
                "vmovl.s8    q4, d6            \n" // a20-a27
                "vmovl.s8    q3, d5            \n" // a10-a17
                "vmovl.s8    q2, d4            \n" // a00-a07

                "vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n" // (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q6, d6, d0[1]     \n" // (a10-a17) * k01
                "vmlal.s16   q7, d7, d0[1]     \n"
                "vmlal.s16   q6, d8, d0[2]     \n" // (a20-a27) * k02
                "vmlal.s16   q7, d9, d0[2]     \n"
                "vmlal.s16   q6, d10, d0[3]    \n" // (a30-a37) * k03
                "vmlal.s16   q7, d11, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]!       \n" // tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(inch) // %6
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];
                sum4 += tmpptr[4] * kptr[0];
                sum5 += tmpptr[5] * kptr[0];
                sum6 += tmpptr[6] * kptr[0];
                sum7 += tmpptr[7] * kptr[0];

                tmpptr += 8;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0[4] = sum4;
            outptr0[5] = sum5;
            outptr0[6] = sum6;
            outptr0[7] = sum7;

            outptr0 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"

                "lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%2, #128]        \n"
                "vld1.s8     {d4-d5}, [%1]!    \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q3, d5            \n" // a20-a23,a30-a33
                "vmovl.s8    q2, d4            \n" // a00-a03,a10-a13

                "vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n" // (a00-a03) * k00
                "vmlal.s16   q6, d5, d0[1]     \n" // (a10-a13) * k01
                "vmlal.s16   q6, d6, d0[2]     \n" // (a20-a23) * k02
                "vmlal.s16   q6, d7, d0[3]     \n" // (a30-a33) * k03

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]        \n" // tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %1, #4            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // (a00-a03) * k00

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory
                "vst1.s32    {d12-d13}, [%0]!  \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(inch) // %6
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

            int q = 0;
            int sum0 = 0;

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }

    //     // NOTE sgemm int8
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         int* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             int sum = 0;
    //
    //             const signed char* kptr = _kernel.channel(p/8 + p%8);
    //
    //             for (int q=0; q<inch; q++)
    //             {
    //                 const signed char* img0 = bottom_blob.channel(q);
    //
    //                 sum += img0[i] * kptr[0];
    //                 kptr ++;
    //             }
    //
    //             outptr0[i] = sum;
    //         }
    //     }
}

static void conv1x1s1_sgemm_int8_requant_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, std::vector<float> scales_requant, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    const float* bias = _bias;

    // interleave
    Mat tmp(8 * 4, inch / 4 + inch % 4, size / 8 + (size % 8) / 4 + size % 4, 1u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
#if __ARM_NEON
                asm volatile(
                    "pld        [%0, #64]     \n"
                    "vld1.s8   {d0}, [%0]     \n"
                    "vst1.s8   {d0}, [%1]!    \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "d0");
                img0 += bottom_blob.cstep;
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

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];

                tmpptr += 4;
                img0 += bottom_blob.cstep;
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        signed char* outptr0 = top_blob.channel(p);
        signed char* outptr1 = top_blob.channel(p + 1);
        signed char* outptr2 = top_blob.channel(p + 2);
        signed char* outptr3 = top_blob.channel(p + 3);

        const float bias0 = bias ? bias[p] : 0.f;
        const float bias1 = bias ? bias[p + 1] : 0.f;
        const float bias2 = bias ? bias[p + 2] : 0.f;
        const float bias3 = bias ? bias[p + 3] : 0.f;

        const float scale_requant_in0 = scales_requant[2 * p];
        const float scale_requant_out0 = scales_requant[2 * p + 1];
        const float scale_requant_in1 = scales_requant[2 * (p + 1)];
        const float scale_requant_out1 = scales_requant[2 * (p + 1) + 1];
        const float scale_requant_in2 = scales_requant[2 * (p + 2)];
        const float scale_requant_out2 = scales_requant[2 * (p + 2) + 1];
        const float scale_requant_in3 = scales_requant[2 * (p + 3)];
        const float scale_requant_out3 = scales_requant[2 * (p + 3) + 1];

#if __ARM_NEON
        float32x4_t _bias03, _scale_in03, _scale_out03;

        _bias03[0] = bias0;
        _bias03[1] = bias1;
        _bias03[2] = bias2;
        _bias03[3] = bias3;

        _scale_in03[0] = scale_requant_in0;
        _scale_in03[1] = scale_requant_in1;
        _scale_in03[2] = scale_requant_in2;
        _scale_in03[3] = scale_requant_in3;

        _scale_out03[0] = scale_requant_out0;
        _scale_out03[1] = scale_requant_out1;
        _scale_out03[2] = scale_requant_out2;
        _scale_out03[3] = scale_requant_out3;
#endif // __ARM_NEON

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"
                "vmov.s32    q10, #0           \n"
                "vmov.s32    q11, #0           \n"
                "vmov.s32    q12, #0           \n"
                "vmov.s32    q13, #0           \n"

                "lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d28-d31}, [%4]!  \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d31           \n" // a30-a37
                "vmovl.s8    q4, d30           \n" // a20-a27
                "vmovl.s8    q15, d29          \n" // a10-a17
                "vmovl.s8    q14, d28          \n" // a00-a07

                "vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d28, d0[0]    \n" // sum0 = (a00-a07) * k00
                "vmlal.s16   q7, d29, d0[0]    \n"
                "vmlal.s16   q8, d28, d0[1]    \n" // sum1 = (a00-a07) * k10
                "vmlal.s16   q9, d29, d0[1]    \n"
                "vmlal.s16   q10, d28, d0[2]   \n" // sum2 = (a00-a07) * k20
                "vmlal.s16   q11, d29, d0[2]   \n"
                "vmlal.s16   q12, d28, d0[3]   \n" // sum3 = (a00-a07) * k30
                "vmlal.s16   q13, d29, d0[3]   \n"

                "vmlal.s16   q6, d30, d1[0]    \n" // sum0 += (a10-a17) * k01
                "vmlal.s16   q7, d31, d1[0]    \n"
                "vmlal.s16   q8, d30, d1[1]    \n" // sum1 += (a10-a17) * k11
                "vmlal.s16   q9, d31, d1[1]    \n"
                "vmlal.s16   q10, d30, d1[2]   \n" // sum2 += (a10-a17) * k21
                "vmlal.s16   q11, d31, d1[2]   \n"
                "vmlal.s16   q12, d30, d1[3]   \n" // sum3 += (a10-a17) * k31
                "vmlal.s16   q13, d31, d1[3]   \n"

                "vmlal.s16   q6, d8, d2[0]     \n" // sum0 += (a20-a27) * k02
                "vmlal.s16   q7, d9, d2[0]     \n"
                "vmlal.s16   q8, d8, d2[1]     \n" // sum1 += (a20-a27) * k12
                "vmlal.s16   q9, d9, d2[1]     \n"
                "vmlal.s16   q10, d8, d2[2]    \n" // sum2 += (a20-a27) * k22
                "vmlal.s16   q11, d9, d2[2]    \n"
                "vmlal.s16   q12, d8, d2[3]    \n" // sum3 += (a20-a27) * k32
                "vmlal.s16   q13, d9, d2[3]    \n"

                "vmlal.s16   q6, d10, d3[0]    \n" // sum0 += (a30-a37) * k03
                "vmlal.s16   q7, d11, d3[0]    \n"
                "vmlal.s16   q8, d10, d3[1]    \n" // sum1 += (a30-a37) * k13
                "vmlal.s16   q9, d11, d3[1]    \n"
                "vmlal.s16   q10, d10, d3[2]   \n" // sum2 += (a30-a37) * k23
                "vmlal.s16   q11, d11, d3[2]   \n"
                "vmlal.s16   q12, d10, d3[3]   \n" // sum3 += (a30-a37) * k33
                "vmlal.s16   q13, d11, d3[3]   \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]!       \n" // tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // sum0 += (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"
                "vmlal.s16   q8, d2, d0[1]     \n" // sum1 += (a00-a07) * k10
                "vmlal.s16   q9, d3, d0[1]     \n"
                "vmlal.s16   q10, d2, d0[2]    \n" // sum2 += (a00-a07) * k20
                "vmlal.s16   q11, d3, d0[2]    \n"
                "vmlal.s16   q12, d2, d0[3]    \n" // sum3 += (a00-a07) * k30
                "vmlal.s16   q13, d3, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                             \n" // store the result to memory

                "vdup.f32   q14, %13            \n" // bias
                "vdup.f32   q15, %14            \n" // bias
                "vdup.f32   q4, %15             \n" // bias
                "vdup.f32   q5, %16             \n" // bias

                // sum0
                // top_s32 -> top_f32
                "vcvt.f32.s32 q6, q6            \n"
                "vcvt.f32.s32 q7, q7            \n"
                "vcvt.f32.s32 q8, q8            \n"
                "vcvt.f32.s32 q9, q9            \n"
                // top_f32 = top_f32 * scale_int
                "vmul.f32   q6, q6, %e17[0]     \n"
                "vmul.f32   q7, q7, %e17[0]     \n"
                "vmul.f32   q8, q8, %e17[1]     \n"
                "vmul.f32   q9, q9, %e17[1]     \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q6, q6, q14         \n"
                "vadd.f32   q7, q7, q14         \n"
                "vadd.f32   q8, q8, q15         \n"
                "vadd.f32   q9, q9, q15         \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q6, %e18[0]     \n"
                "vmul.f32   q1, q7, %e18[0]     \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d12, q0             \n"
                "vqmovn.s32 d13, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d12, q6           \n"
                // save top_s8
                "vst1.8     {d12}, [%0]!        \n"
                // sum1
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q8, %e18[1]     \n"
                "vmul.f32   q1, q9, %e18[1]     \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d16, q0             \n"
                "vqmovn.s32 d17, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d16, q8           \n"
                // save top_s8
                "vst1.8     {d16}, [%1]!        \n"

                // sum2
                // top_s32 -> top_f32
                "vcvt.f32.s32 q10, q10          \n"
                "vcvt.f32.s32 q11, q11          \n"
                "vcvt.f32.s32 q12, q12          \n"
                "vcvt.f32.s32 q13, q13          \n"
                // top_f32 = top_f32 * scale_int
                "vmul.f32   q10, q10, %f17[0]   \n"
                "vmul.f32   q11, q11, %f17[0]   \n"
                "vmul.f32   q12, q12, %f17[1]   \n"
                "vmul.f32   q13, q13, %f17[1]   \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q10, q10, q4        \n"
                "vadd.f32   q11, q11, q4        \n"
                "vadd.f32   q12, q12, q5        \n"
                "vadd.f32   q13, q13, q5        \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q10, %f18[0]    \n"
                "vmul.f32   q1, q11, %f18[0]    \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d20, q0             \n"
                "vqmovn.s32 d21, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d20, q10          \n"
                // save top_s8
                "vst1.8     {d20}, [%2]!        \n"
                // sum3
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q12, %f18[1]    \n"
                "vmul.f32   q1, q13, %f18[1]    \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d24, q0             \n"
                "vqmovn.s32 d25, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d24, q12          \n"
                // save top_s8
                "vst1.8     {d24}, [%3]!        \n"

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
                "r"(inch),        // %12
                "r"(bias0),       // %13
                "r"(bias1),       // %14
                "r"(bias2),       // %15
                "r"(bias3),       // %16
                "w"(_scale_in03), // %17
                "w"(_scale_out03) // %18
                : "cc", "memory", "r4", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;
            int sum0_4 = 0;
            int sum0_5 = 0;
            int sum0_6 = 0;
            int sum0_7 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;
            int sum1_4 = 0;
            int sum1_5 = 0;
            int sum1_6 = 0;
            int sum1_7 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;
            int sum2_4 = 0;
            int sum2_5 = 0;
            int sum2_6 = 0;
            int sum2_7 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;
            int sum3_4 = 0;
            int sum3_5 = 0;
            int sum3_6 = 0;
            int sum3_7 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];
                sum0_4 += tmpptr[4] * kptr[0];
                sum0_5 += tmpptr[5] * kptr[0];
                sum0_6 += tmpptr[6] * kptr[0];
                sum0_7 += tmpptr[7] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];
                sum1_4 += tmpptr[4] * kptr[1];
                sum1_5 += tmpptr[5] * kptr[1];
                sum1_6 += tmpptr[6] * kptr[1];
                sum1_7 += tmpptr[7] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];
                sum2_4 += tmpptr[4] * kptr[2];
                sum2_5 += tmpptr[5] * kptr[2];
                sum2_6 += tmpptr[6] * kptr[2];
                sum2_7 += tmpptr[7] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];
                sum3_4 += tmpptr[4] * kptr[3];
                sum3_5 += tmpptr[5] * kptr[3];
                sum3_6 += tmpptr[6] * kptr[3];
                sum3_7 += tmpptr[7] * kptr[3];

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0             \n"
                "vmov.s32    q7, #0             \n"
                "vmov.s32    q8, #0             \n"
                "vmov.s32    q9, #0             \n"

                "lsr         r4, %12, #2        \n" // r4 = nn = inch >> 2
                "cmp         r4, #0             \n"
                "beq         1f                 \n"

                "0:                             \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]         \n"
                "vld1.s8     {d28-d29}, [%4]!   \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q15, d29           \n" // a20-a23,a30-a33
                "vmovl.s8    q14, d28           \n" // a00-a04,a10-a14

                "vld1.s8     {d0-d1}, [%5]!     \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1             \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0             \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d28, d0[0]     \n" // sum0 = (a00-a03) * k00
                "vmlal.s16   q7, d28, d0[1]     \n" // sum1 = (a00-a03) * k10
                "vmlal.s16   q8, d28, d0[2]     \n" // sum2 = (a00-a03) * k20
                "vmlal.s16   q9, d28, d0[3]     \n" // sum3 = (a00-a03) * k30

                "vmlal.s16   q6, d29, d1[0]     \n" // sum0 += (a10-a13) * k01
                "vmlal.s16   q7, d29, d1[1]     \n" // sum1 += (a10-a13) * k11
                "vmlal.s16   q8, d29, d1[2]     \n" // sum2 += (a10-a13) * k21
                "vmlal.s16   q9, d29, d1[3]     \n" // sum3 += (a10-a13) * k31

                "vmlal.s16   q6, d30, d2[0]     \n" // sum0 += (a20-a23) * k02
                "vmlal.s16   q7, d30, d2[1]     \n" // sum1 += (a20-a23) * k12
                "vmlal.s16   q8, d30, d2[2]     \n" // sum2 += (a20-a23) * k22
                "vmlal.s16   q9, d30, d2[3]     \n" // sum3 += (a20-a23) * k32

                "vmlal.s16   q6, d31, d3[0]     \n" // sum0 += (a30-a33) * k03
                "vmlal.s16   q7, d31, d3[1]     \n" // sum1 += (a30-a33) * k13
                "vmlal.s16   q8, d31, d3[2]     \n" // sum2 += (a30-a33) * k23
                "vmlal.s16   q9, d31, d3[3]     \n" // sum3 += (a30-a33) * k33

                "subs        r4, r4, #1         \n"
                "bne         0b                 \n" // end for

                "1:                             \n"
                // remain loop
                "and         r4, %12, #3        \n" // r4 = remain = inch & 3
                "cmp         r4, #0             \n"
                "beq         3f                 \n"

                "2:                             \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]         \n" // tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%5]         \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2             \n"
                "vmovl.s8    q0, d0             \n"
                "add         %4, #4             \n"
                "add         %5, #4             \n"

                "vmlal.s16   q6, d2, d0[0]      \n" // sum0 += (a00-a03) * k00
                "vmlal.s16   q7, d2, d0[1]      \n" // sum1 += (a00-a03) * k10
                "vmlal.s16   q8, d2, d0[2]      \n" // sum2 += (a00-a03) * k20
                "vmlal.s16   q9, d2, d0[3]      \n" // sum3 += (a00-a03) * k30

                "subs        r4, r4, #1         \n"
                "bne         2b                 \n"

                "3:                             \n" // store the result to memory

                "vdup.f32   q14, %13            \n" // bias
                "vdup.f32   q15, %14            \n" // bias
                "vdup.f32   q4, %15             \n" // bias
                "vdup.f32   q5, %16             \n" // bias

                // sum0-1
                // top_s32 -> top_f32
                "vcvt.f32.s32 q6, q6            \n"
                "vcvt.f32.s32 q7, q7            \n"
                "vcvt.f32.s32 q8, q8            \n"
                "vcvt.f32.s32 q9, q9            \n"
                // top_f32 = top_f32 * scale_int
                "vmul.f32   q6, q6, %e17[0]     \n"
                "vmul.f32   q7, q7, %e17[1]     \n"
                "vmul.f32   q8, q8, %f17[0]     \n"
                "vmul.f32   q9, q9, %f17[1]     \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q6, q6, q14         \n"
                "vadd.f32   q7, q7, q15         \n"
                "vadd.f32   q8, q8, q4          \n"
                "vadd.f32   q9, q9, q5          \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q6, %e18[0]     \n"
                "vmul.f32   q1, q7, %e18[1]     \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d12, q0             \n"
                "vqmovn.s32 d13, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d12, q6           \n"
                // save top_s8
                "vst1.s32  {d12[0]}, [%0]!      \n"
                "vst1.s32  {d12[1]}, [%1]!      \n"

                // sum1-2
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q8, %f18[0]     \n"
                "vmul.f32   q1, q9, %f18[1]     \n"

                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0           \n"
                "vcvtr.s32.f32 s1, s1           \n"
                "vcvtr.s32.f32 s2, s2           \n"
                "vcvtr.s32.f32 s3, s3           \n"
                "vcvtr.s32.f32 s4, s4           \n"
                "vcvtr.s32.f32 s5, s5           \n"
                "vcvtr.s32.f32 s6, s6           \n"
                "vcvtr.s32.f32 s7, s7           \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d16, q0             \n"
                "vqmovn.s32 d17, q1             \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d16, q8           \n"
                // save top_s8
                "vst1.s32     {d16[0]}, [%2]!   \n"
                "vst1.s32     {d16[1]}, [%3]!   \n"

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
                "r"(inch),        // %12
                "r"(bias0),       // %13
                "r"(bias1),       // %14
                "r"(bias2),       // %15
                "r"(bias3),       // %16
                "w"(_scale_in03), // %17
                "w"(_scale_out03) // %18
                : "cc", "memory", "r4", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
            int sum0_0 = 0;
            int sum0_1 = 0;
            int sum0_2 = 0;
            int sum0_3 = 0;

            int sum1_0 = 0;
            int sum1_1 = 0;
            int sum1_2 = 0;
            int sum1_3 = 0;

            int sum2_0 = 0;
            int sum2_1 = 0;
            int sum2_2 = 0;
            int sum2_3 = 0;

            int sum3_0 = 0;
            int sum3_1 = 0;
            int sum3_2 = 0;
            int sum3_3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0_0 += tmpptr[0] * kptr[0];
                sum0_1 += tmpptr[1] * kptr[0];
                sum0_2 += tmpptr[2] * kptr[0];
                sum0_3 += tmpptr[3] * kptr[0];

                sum1_0 += tmpptr[0] * kptr[1];
                sum1_1 += tmpptr[1] * kptr[1];
                sum1_2 += tmpptr[2] * kptr[1];
                sum1_3 += tmpptr[3] * kptr[1];

                sum2_0 += tmpptr[0] * kptr[2];
                sum2_1 += tmpptr[1] * kptr[2];
                sum2_2 += tmpptr[2] * kptr[2];
                sum2_3 += tmpptr[3] * kptr[2];

                sum3_0 += tmpptr[0] * kptr[3];
                sum3_1 += tmpptr[1] * kptr[3];
                sum3_2 += tmpptr[2] * kptr[3];
                sum3_3 += tmpptr[3] * kptr[3];

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "veor        q6, q6, q6        \n"
                "veor        q7, q7, q7        \n"
                "veor        q8, q8, q8        \n"
                "veor        q9, q9, q9        \n"
                "vmov.s32    q10, #0           \n"

                "lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4}, [%4]        \n" // tmpr a00,a10,a20,a30    a(inch)(data)
                "add         %4, #4            \n"
                "vmovl.s8    q2, d4            \n" // a00,a10,a20,a30

                "vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

                "vmlal.s16   q6, d0, d4[0]     \n" // (k00-k30) * a00
                "vmlal.s16   q7, d1, d4[1]     \n" // (k01-k31) * a10
                "vmlal.s16   q8, d2, d4[2]     \n" // (k02-k32) * a20
                "vmlal.s16   q9, d3, d4[3]     \n" // (k03-k33) * a30

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "vadd.s32    q6, q6, q7        \n"
                "vadd.s32    q9, q9, q8        \n"
                "vadd.s32    q10, q6, q9       \n"

                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n" // tmpr a00        a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #1            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q10, d0, d2[0]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory

                // top_s32 -> top_f32
                "vcvt.f32.s32 q10, q10         \n"
                // top_f32 = top_f32 * scale_int
                "vmul.f32   q10, q10, %q14     \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q10, q10, %q13     \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q10, %q15      \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0          \n"
                "vcvtr.s32.f32 s1, s1          \n"
                "vcvtr.s32.f32 s2, s2          \n"
                "vcvtr.s32.f32 s3, s3          \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d12, q0            \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d12, q6          \n"
                // save top_s8
                "vst1.8     {d12[0]}, [%0]!    \n"
                "vst1.8     {d12[1]}, [%1]!    \n"
                "vst1.8     {d12[2]}, [%2]!    \n"
                "vst1.8     {d12[3]}, [%3]!    \n"

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
                "r"(inch),        // %12
                "w"(_bias03),     // %13
                "w"(_scale_in03), // %14
                "w"(_scale_out03) // %15
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

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

        signed char* outptr0 = out0;

        const float bias0 = bias ? bias[p] : 0.f;
        const float scale_requant_in = scales_requant[2 * p];
        const float scale_requant_out = scales_requant[2 * p + 1];

#if __ARM_NEON
        float32x4_t _bias0 = vdupq_n_f32(bias0);
        float32x4_t _scale_in = vdupq_n_f32(scale_requant_in);
        float32x4_t _scale_out = vdupq_n_f32(scale_requant_out);
#endif // __ARM_NEON

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const signed char* tmpptr = tmp.channel(i / 8);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"

                "lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%1, #128]        \n"
                "vld1.s8     {d4-d7}, [%1]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n" // a30-a37
                "vmovl.s8    q4, d6            \n" // a20-a27
                "vmovl.s8    q3, d5            \n" // a10-a17
                "vmovl.s8    q2, d4            \n" // a00-a07

                "vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n" // (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q6, d6, d0[1]     \n" // (a10-a17) * k01
                "vmlal.s16   q7, d7, d0[1]     \n"
                "vmlal.s16   q6, d8, d0[2]     \n" // (a20-a27) * k02
                "vmlal.s16   q7, d9, d0[2]     \n"
                "vmlal.s16   q6, d10, d0[3]    \n" // (a30-a37) * k03
                "vmlal.s16   q7, d11, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]!       \n" // tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory

                // top_s32 -> top_f32
                "vcvt.f32.s32 q6, q6           \n"
                "vcvt.f32.s32 q7, q7           \n"
                // top_f32 = top_f32 * scale_in
                "vmul.f32   q6, q6, %q8        \n"
                "vmul.f32   q7, q7, %q8        \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q6, q6, %q7        \n"
                "vadd.f32   q7, q7, %q7        \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q6, %q9        \n"
                "vmul.f32   q1, q7, %q9        \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0          \n"
                "vcvtr.s32.f32 s1, s1          \n"
                "vcvtr.s32.f32 s2, s2          \n"
                "vcvtr.s32.f32 s3, s3          \n"
                "vcvtr.s32.f32 s4, s4          \n"
                "vcvtr.s32.f32 s5, s5          \n"
                "vcvtr.s32.f32 s6, s6          \n"
                "vcvtr.s32.f32 s7, s7          \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d12, q0            \n"
                "vqmovn.s32 d13, q1            \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d12, q6          \n"
                // save top_s8
                "vst1.8     {d12}, [%0]!       \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(inch),      // %6
                "w"(_bias0),    // %7
                "w"(_scale_in), // %8
                "w"(_scale_out) // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;
            int sum4 = 0;
            int sum5 = 0;
            int sum6 = 0;
            int sum7 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];
                sum4 += tmpptr[4] * kptr[0];
                sum5 += tmpptr[5] * kptr[0];
                sum6 += tmpptr[6] * kptr[0];
                sum7 += tmpptr[7] * kptr[0];

                tmpptr += 8;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0[4] = sum4;
            outptr0[5] = sum5;
            outptr0[6] = sum6;
            outptr0[7] = sum7;

            outptr0 += 8;
#endif // __ARM_NEON
        }

        for (; i + 3 < size; i += 4)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

#if __ARM_NEON
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"

                "lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"

                "0:                            \n" // for(; nn != 0; nn--)
                "pld         [%2, #128]        \n"
                "vld1.s8     {d4-d5}, [%1]!    \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
                "vmovl.s8    q3, d5            \n" // a20-a23,a30-a33
                "vmovl.s8    q2, d4            \n" // a00-a03,a10-a13

                "vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n" // (a00-a03) * k00
                "vmlal.s16   q6, d5, d0[1]     \n" // (a10-a13) * k01
                "vmlal.s16   q6, d6, d0[2]     \n" // (a20-a23) * k02
                "vmlal.s16   q6, d7, d0[3]     \n" // (a30-a33) * k03

                "subs        r4, r4, #1        \n"
                "bne         0b                \n" // end for

                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n" // r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n" // for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]        \n" // tmpr a00-a03    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %1, #4            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n" // (a00-a03) * k00

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n" // store the result to memory

                // top_s32 -> top_f32
                "vcvt.f32.s32 q6, q6           \n"
                // top_f32 = top_f32 * scale_in
                "vmul.f32   q6, q6, %q8        \n"
                // top_f32 = top_f32 + bias
                "vadd.f32   q6, q6, %q7        \n"
                // top_f32 = top_f32 * scale_out
                "vmul.f32   q0, q6, %q9        \n"
                // top_f32 -> top_s32
                "vcvtr.s32.f32 s0, s0          \n"
                "vcvtr.s32.f32 s1, s1          \n"
                "vcvtr.s32.f32 s2, s2          \n"
                "vcvtr.s32.f32 s3, s3          \n"
                // top_s32 -> top_s16
                "vqmovn.s32 d12, q0            \n"
                // top_s16 -> top_s8
                "vqmovn.s16   d12, q6          \n"

                "vst1.s32    {d12[0]}, [%0]!   \n"

                : "=r"(outptr0), // %0
                "=r"(tmpptr),  // %1
                "=r"(kptr)     // %2
                : "0"(outptr0),
                "1"(tmpptr),
                "2"(kptr),
                "r"(inch),      // %6
                "w"(_bias0),    // %7
                "w"(_scale_in), // %8
                "w"(_scale_out) // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#else
            int sum0 = 0;
            int sum1 = 0;
            int sum2 = 0;
            int sum3 = 0;

            for (int q = 0; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];

                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
#endif // __ARM_NEON
        }

        for (; i < size; i++)
        {
            const signed char* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const signed char* kptr = kernel.channel(p / 4 + p % 4);

            int q = 0;
            int sum0 = 0;

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
}
#endif
