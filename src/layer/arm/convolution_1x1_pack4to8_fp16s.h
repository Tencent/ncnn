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

static void conv1x1s1_sgemm_transform_kernel_pack4to8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack4to8, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 8b-4a-inch/4a-outch/8b
    kernel_tm_pack4to8.create(8 * 4, inch / 4, outch / 8, (size_t)2u, 1);

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

        __fp16* g0 = kernel_tm_pack4to8.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int i = 0; i < 4; i++)
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

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            k4 += 4;
            k5 += 4;
            k6 += 4;
            k7 += 4;
        }
    }
}

static void conv1x1s1_sgemm_pack4to8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
    if (size >= 8)
        tmp.create(8, inch, size / 8 + size % 8, elemsize, elempack, opt.workspace_allocator);
    else // if (size >= 1)
        tmp.create(1, inch, size, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size;
        int remain_size_start = 0;

        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                // transpose 4x8
                asm volatile(
                    "prfm   pldl1keep, [%0, #512]   \n"
                    "ld4    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0] \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0", "v1", "v2", "v3");

                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + i % 8);

            for (int q = 0; q < inch; q++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%0, #64]    \n"
                    "ld1    {v0.4h}, [%0]           \n"
                    "st1    {v0.4h}, [%1], #8       \n"
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "v0");

                img0 += bottom_blob.cstep * 4;
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
        for (; i + 7 < size; i += 8)
        {
            __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p);

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

                "fmla   v16.8h, v10.8h, v2.h[0]     \n"
                "fmla   v17.8h, v10.8h, v2.h[1]     \n"
                "fmla   v18.8h, v10.8h, v2.h[2]     \n"
                "fmla   v19.8h, v10.8h, v2.h[3]     \n"
                "fmla   v20.8h, v10.8h, v2.h[4]     \n"
                "fmla   v21.8h, v10.8h, v2.h[5]     \n"
                "fmla   v22.8h, v10.8h, v2.h[6]     \n"
                "fmla   v23.8h, v10.8h, v2.h[7]     \n"

                "fmla   v16.8h, v11.8h, v3.h[0]     \n"
                "fmla   v17.8h, v11.8h, v3.h[1]     \n"
                "fmla   v18.8h, v11.8h, v3.h[2]     \n"
                "fmla   v19.8h, v11.8h, v3.h[3]     \n"
                "fmla   v20.8h, v11.8h, v3.h[4]     \n"
                "fmla   v21.8h, v11.8h, v3.h[5]     \n"
                "fmla   v22.8h, v11.8h, v3.h[6]     \n"
                "fmla   v23.8h, v11.8h, v3.h[7]     \n"

                "subs   %w0, %w0, #1                \n"

                "bne    0b                          \n"

                "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
                "st1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%1], #64 \n"

                : "=r"(nn),      // %0
                "=r"(outptr0), // %1
                "=r"(tmpptr),  // %2
                "=r"(kptr)     // %3
                : "0"(nn),
                "1"(outptr0),
                "2"(tmpptr),
                "3"(kptr),
                "r"(biasptr) // %8
                : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + i % 8);
            const __fp16* kptr = kernel.channel(p);

            float16x8_t _sum0 = vld1q_f16(biasptr);

            int q = 0;
            for (; q < inch; q++)
            {
                float16x4_t _r0 = vld1_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);
                float16x8_t _k1 = vld1q_f16(kptr + 8);
                float16x8_t _k2 = vld1q_f16(kptr + 16);
                float16x8_t _k3 = vld1q_f16(kptr + 24);

                _sum0 = vfmaq_lane_f16(_sum0, _k0, _r0, 0);
                _sum0 = vfmaq_lane_f16(_sum0, _k1, _r0, 1);
                _sum0 = vfmaq_lane_f16(_sum0, _k2, _r0, 2);
                _sum0 = vfmaq_lane_f16(_sum0, _k3, _r0, 3);

                kptr += 32;
                tmpptr += 4;
            }

            vst1q_f16(outptr0, _sum0);

            outptr0 += 8;
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

static void conv1x1s2_pack4to8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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
        const __fp16* r0 = bottom_blob.channel(p);
        __fp16* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                float16x4_t _v0 = vld1_f16(r0);
                float16x4_t _v1 = vld1_f16(r0 + 8);
                float16x4_t _v2 = vld1_f16(r0 + 16);
                float16x4_t _v3 = vld1_f16(r0 + 24);
                vst1_f16(outptr, _v0);
                vst1_f16(outptr + 4, _v1);
                vst1_f16(outptr + 8, _v2);
                vst1_f16(outptr + 12, _v3);

                r0 += 32;
                outptr += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                float16x4_t _v0 = vld1_f16(r0);
                float16x4_t _v1 = vld1_f16(r0 + 8);
                vst1_f16(outptr, _v0);
                vst1_f16(outptr + 4, _v1);

                r0 += 16;
                outptr += 8;
            }
            for (; j < outw; j++)
            {
                float16x4_t _v = vld1_f16(r0);
                vst1_f16(outptr, _v);

                r0 += 8;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4to8_fp16sa_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
