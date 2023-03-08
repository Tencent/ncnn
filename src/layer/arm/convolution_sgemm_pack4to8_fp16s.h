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

static void im2col_sgemm_pack4to8_fp16sa_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 8u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + size % 8, 8u, 4, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 8u, 4, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
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
                    img0 += size * 4;
                }
            }
        }

        remain_size_start += nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + i % 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i * 4;

                for (int k = 0; k < maxk; k++)
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
                    img0 += size * 4;
                }
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

            int nn = inch * maxk; // inch always > 0

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

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum0 = vld1q_f16(biasptr);

            int q = 0;
            for (; q < nn; q++)
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
}

static void convolution_im2col_sgemm_transform_kernel_pack4to8_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-4a-maxk-inch/4a-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(32 * maxk, inch / 4, outch / 8, (size_t)2u);

    for (int q = 0; q + 7 < outch; q += 8)
    {
        __fp16* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel.channel(q + j).row(p + i);

                        g00[0] = (__fp16)k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

static void convolution_im2col_sgemm_pack4to8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 3 < outw; j += 4)
                        {
                            float16x4_t _val0 = vld1_f16(sptr);
                            float16x4_t _val1 = vld1_f16(sptr + stride_w * 4);
                            float16x4_t _val2 = vld1_f16(sptr + stride_w * 8);
                            float16x4_t _val3 = vld1_f16(sptr + stride_w * 12);
                            vst1_f16(ptr, _val0);
                            vst1_f16(ptr + 4, _val1);
                            vst1_f16(ptr + 8, _val2);
                            vst1_f16(ptr + 12, _val3);

                            sptr += stride_w * 16;
                            ptr += 16;
                        }
                        for (; j + 1 < outw; j += 2)
                        {
                            float16x4_t _val0 = vld1_f16(sptr);
                            float16x4_t _val1 = vld1_f16(sptr + stride_w * 4);
                            vst1_f16(ptr, _val0);
                            vst1_f16(ptr + 4, _val1);

                            sptr += stride_w * 8;
                            ptr += 8;
                        }
                        for (; j < outw; j++)
                        {
                            float16x4_t _val = vld1_f16(sptr);
                            vst1_f16(ptr, _val);

                            sptr += stride_w * 4;
                            ptr += 4;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack4to8_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
