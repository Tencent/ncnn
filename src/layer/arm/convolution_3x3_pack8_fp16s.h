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

static void conv3x3s1_winograd63_transform_kernel_pack8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd63 transform kernel
    Mat kernel_tm;
    kernel_tm.create(8 * 8, inch, outch);

    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i = 0; i < 8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j = 0; j < 8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++)
                {
                    kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 64-inch-outch
    // dst = 8b-8a-inch/8a-64-outch/8b
    kernel_tm_pack8.create(inch / 8, 64, outch / 8, (size_t)2u * 64, 64);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

                    g00[0] = (__fp16)k00[k];
                    g00[1] = (__fp16)k10[k];
                    g00[2] = (__fp16)k20[k];
                    g00[3] = (__fp16)k30[k];
                    g00[4] = (__fp16)k40[k];
                    g00[5] = (__fp16)k50[k];
                    g00[6] = (__fp16)k60[k];
                    g00[7] = (__fp16)k70[k];

                    g00 += 8;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd63_transform_input_pack8_fp16sa_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_fp16sa_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd63_transform_output_pack8_fp16sa_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_pack8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd43 transform kernel
    Mat kernel_tm(6 * 6, inch, outch);

    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 36-inch-outch
    // dst = 8b-8a-inch/8a-36-outch/8b
    kernel_tm_pack8.create(inch / 8, 36, outch / 8, (size_t)2u * 64, 64);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

                    g00[0] = (__fp16)k00[k];
                    g00[1] = (__fp16)k10[k];
                    g00[2] = (__fp16)k20[k];
                    g00[3] = (__fp16)k30[k];
                    g00[4] = (__fp16)k40[k];
                    g00[5] = (__fp16)k50[k];
                    g00[6] = (__fp16)k60[k];
                    g00[7] = (__fp16)k70[k];

                    g00 += 8;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_pack8_fp16sa_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_fp16sa_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd43_transform_output_pack8_fp16sa_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd23_transform_kernel_pack8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd23 transform kernel
    Mat kernel_tm(4 * 4, inch, outch);

    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 16-inch-outch
    // dst = 8b-8a-inch/8a-16-outch/8b
    kernel_tm_pack8.create(inch / 8, 16, outch / 8, (size_t)2u * 64, 64);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

                    g00[0] = (__fp16)k00[k];
                    g00[1] = (__fp16)k10[k];
                    g00[2] = (__fp16)k20[k];
                    g00[3] = (__fp16)k30[k];
                    g00[4] = (__fp16)k40[k];
                    g00[5] = (__fp16)k50[k];
                    g00[6] = (__fp16)k60[k];
                    g00[7] = (__fp16)k70[k];

                    g00 += 8;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_pack8_fp16sa_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_fp16sa_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd23_transform_output_pack8_fp16sa_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            __fp16* outptr0 = out0.row<__fp16>(0);

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n" // sum0

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v4.8h, v5.8h}, [%1]        \n" // r04 r05

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v4.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v4.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v4.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v5.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v12.8h, v13.8h}, [%2]      \n" // r14 r15

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v11.h[0]    \n"

                        "fmla   v28.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v11.h[2]    \n"

                        "fmla   v28.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v11.h[4]    \n"

                        "fmla   v28.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v11.h[6]    \n"

                        "fmla   v28.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v12.h[0]    \n"

                        "fmla   v28.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v12.h[2]    \n"

                        "fmla   v28.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v12.h[4]    \n"

                        "fmla   v28.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v12.h[6]    \n"

                        "fmla   v28.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v13.h[0]    \n"

                        "fmla   v28.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v13.h[2]    \n"

                        "fmla   v28.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v13.h[4]    \n"

                        "fmla   v28.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v13.h[6]    \n"

                        "fmla   v28.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v4.8h, v5.8h}, [%3]        \n" // r24 r25

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v4.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v4.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v4.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v5.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1] \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v30.8h, v31.8h}, [%0]      \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2] \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v7.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v7.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3] \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v7.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v7.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "add    %1, %1, #32                 \n"

                        "add    %2, %2, #32                 \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"
                        "fadd   v29.8h, v29.8h, v31.8h      \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h, v29.8h}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%1] \n" // r00 r01 r02

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v31.8h}, [%0]              \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmul   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.8h, v4.8h, v5.8h}, [%2] \n" // r10 r11 r12

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%3] \n" // r20 r21 r22

                        "fmla   v28.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"

                        "add    %1, %1, #16                 \n"

                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "add    %2, %2, #16                 \n"

                        "fadd   v28.8h, v28.8h, v29.8h      \n"
                        "fadd   v30.8h, v30.8h, v31.8h      \n"

                        "add    %3, %3, #16                 \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h}, [%0], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }

                r0 += 16;
                r1 += 16;
                r2 += 16;
            }
        }
    }
}

static void conv3x3s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n" // sum0

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1], #64 \n" // r04 r05 r06 r07

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v7.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v7.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v7.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v7.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.8h}, [%1]               \n" // r08

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v0.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v0.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v0.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v0.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v14.h[0]    \n"

                        "fmla   v28.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v14.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v14.h[2]    \n"

                        "fmla   v28.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v14.h[3]    \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v14.h[4]    \n"

                        "fmla   v28.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v14.h[5]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v14.h[6]    \n"

                        "fmla   v28.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v14.h[7]    \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v15.h[0]    \n"

                        "fmla   v28.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v15.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v15.h[2]    \n"

                        "fmla   v28.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v13.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v15.h[3]    \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v15.h[4]    \n"

                        "fmla   v28.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v15.h[5]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v15.h[6]    \n"

                        "fmla   v28.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v13.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v15.h[7]    \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v8.8h}, [%2]               \n" // r18

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v8.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v14.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v8.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v8.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v14.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v8.h[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v8.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v8.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v8.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v14.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v8.h[7]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v7.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v7.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v7.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v7.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3]               \n" // r28

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v0.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v0.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v0.h[2]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v0.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v0.h[5]     \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v0.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v30.8h, v31.8h}, [%0]      \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.8h}, [%1]               \n" // r04

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v7.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v7.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v4.8h}, [%2]               \n" // r14

                        "fmla   v28.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v7.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v7.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3]               \n" // r24

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v0.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"
                        "fadd   v29.8h, v29.8h, v31.8h      \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h, v29.8h}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%1] \n" // r00 r01 r02

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v31.8h}, [%0]              \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmul   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.8h, v4.8h, v5.8h}, [%2] \n" // r10 r11 r12

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%3] \n" // r20 r21 r22

                        "fmla   v28.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"

                        // "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n"

                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "add    %2, %2, #32                 \n"

                        "fadd   v28.8h, v28.8h, v29.8h      \n"
                        "fadd   v30.8h, v30.8h, v31.8h      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"

                        "sub    %4, %4, #1088               \n" // kptr -= 8.5 * 64;

                        "st1    {v28.8h}, [%0], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(kptr)     // %4
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    }
}
