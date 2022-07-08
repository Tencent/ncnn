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

static void conv3x3s1_winograd63_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = 4b-4a-inch/4a-64-outch/4b;
#if __aarch64__
    kernel_tm_pack4.create(2 * inch / 4, 64, (outch / 4) / 2 + (outch / 4) % 2, (size_t)4u * 16, 16);
#else
    kernel_tm_pack4.create(inch / 4, 64, outch / 4, (size_t)4u * 16, 16);
#endif

    int q = 0;
#if __aarch64__
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

        Mat g0 = kernel_tm_pack4.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

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
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);
#else
        Mat g0 = kernel_tm_pack4.channel(q / 4);
#endif

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

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
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, 16u, 4, opt.workspace_allocator);
        conv3x3s1_winograd63_transform_input_pack4_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack4_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 16u, 4, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd63_transform_output_pack4_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = 4b-4a-inch/4a-36-outch/4b;
#if __aarch64__
    kernel_tm_pack4.create(2 * inch / 4, 36, (outch / 4) / 2 + (outch / 4) % 2, (size_t)4u * 16, 16);
#else
    kernel_tm_pack4.create(inch / 4, 36, outch / 4, (size_t)4u * 16, 16);
#endif

    int q = 0;
#if __aarch64__
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

        Mat g0 = kernel_tm_pack4.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

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
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);
#else
        Mat g0 = kernel_tm_pack4.channel(q / 4);
#endif

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

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
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, 16u, 4, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_pack4_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack4_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 16u, 4, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd43_transform_output_pack4_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd23_transform_kernel_pack4_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = 4b-4a-inch/4a-16-outch/4b;
#if __aarch64__
    kernel_tm_pack4.create(2 * inch / 4, 16, (outch / 4) / 2 + (outch / 4) % 2, (size_t)4u * 16, 16);
#else
    kernel_tm_pack4.create(inch / 4, 16, outch / 4, (size_t)4u * 16, 16);
#endif

    int q = 0;
#if __aarch64__
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

        Mat g0 = kernel_tm_pack4.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);
                    const float* k40 = k4.row(p + i);
                    const float* k50 = k5.row(p + i);
                    const float* k60 = k6.row(p + i);
                    const float* k70 = k7.row(p + i);

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
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

#if __aarch64__
        Mat g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);
#else
        Mat g0 = kernel_tm_pack4.channel(q / 4);
#endif

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = k0.row(p + i);
                    const float* k10 = k1.row(p + i);
                    const float* k20 = k2.row(p + i);
                    const float* k30 = k3.row(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

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
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, 16u, 4, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_pack4_neon(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack4_neon(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 16u, 4, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd23_transform_output_pack4_neon(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = (w - 2 * outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            const float* kptr = (const float*)kernel.channel(p).row(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n" // r04 r05 r06 r07

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v28.4s}, [%1]              \n" // r08

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v28.s[3]    \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v20.4s, v24.4s, v8.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v24.4s, v12.s[0]    \n"
                        "fmla   v23.4s, v24.4s, v14.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v8.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v10.s[1]    \n"
                        "fmla   v22.4s, v25.4s, v12.s[1]    \n"
                        "fmla   v23.4s, v25.4s, v14.s[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v26.4s, v8.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v10.s[2]    \n"
                        "fmla   v22.4s, v26.4s, v12.s[2]    \n"
                        "fmla   v23.4s, v26.4s, v14.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v8.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v10.s[3]    \n"
                        "fmla   v22.4s, v27.4s, v12.s[3]    \n"
                        "fmla   v23.4s, v27.4s, v14.s[3]    \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v28.4s}, [%2]              \n" // r18

                        "fmla   v20.4s, v16.4s, v9.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v11.s[0]    \n"
                        "fmla   v22.4s, v16.4s, v13.s[0]    \n"
                        "fmla   v23.4s, v16.4s, v15.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v11.s[1]    \n"
                        "fmla   v22.4s, v17.4s, v13.s[1]    \n"
                        "fmla   v23.4s, v17.4s, v15.s[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v18.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v11.s[2]    \n"
                        "fmla   v22.4s, v18.4s, v13.s[2]    \n"
                        "fmla   v23.4s, v18.4s, v15.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v9.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v11.s[3]    \n"
                        "fmla   v22.4s, v19.4s, v13.s[3]    \n"
                        "fmla   v23.4s, v19.4s, v15.s[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v20.4s, v24.4s, v10.s[0]    \n"
                        "fmla   v21.4s, v24.4s, v12.s[0]    \n"
                        "fmla   v22.4s, v24.4s, v14.s[0]    \n"
                        "fmla   v23.4s, v24.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v10.s[1]    \n"
                        "fmla   v21.4s, v25.4s, v12.s[1]    \n"
                        "fmla   v22.4s, v25.4s, v14.s[1]    \n"
                        "fmla   v23.4s, v25.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v26.4s, v10.s[2]    \n"
                        "fmla   v21.4s, v26.4s, v12.s[2]    \n"
                        "fmla   v22.4s, v26.4s, v14.s[2]    \n"
                        "fmla   v23.4s, v26.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v10.s[3]    \n"
                        "fmla   v21.4s, v27.4s, v12.s[3]    \n"
                        "fmla   v22.4s, v27.4s, v14.s[3]    \n"
                        "fmla   v23.4s, v27.4s, v28.s[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v28.4s}, [%3]              \n" // r28

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        //                         "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4] \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v28.s[1]    \n"
                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v28.s[3]    \n"

                        "sub    %4, %4, #512                \n" // kptr -= 8 * 16;

                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n" // sum0 sum1 sum2 sum3

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d0-d7}        \n" // r00 r01 r02 r03

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d8-d15}       \n" // r04 r05 r06 r07

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]  \n" // r08

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d8-d15}       \n" // r10 r11 r12 r13

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n" // r14 r15 r16 r17

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d8-d9}, [%2 :128]  \n" // r18

                        "vmla.f32   q12, q8, d10[0]     \n"
                        "vmla.f32   q13, q8, d14[0]     \n"
                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d10[1]     \n"
                        "vmla.f32   q13, q9, d14[1]     \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d11[0]    \n"
                        "vmla.f32   q13, q10, d15[0]    \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d11[1]    \n"
                        "vmla.f32   q13, q11, d15[1]    \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d12[0]     \n"
                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d12[1]     \n"
                        "vmla.f32   q13, q9, d0[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d13[0]    \n"
                        "vmla.f32   q13, q10, d1[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d13[1]    \n"
                        "vmla.f32   q13, q11, d1[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n" // r20 r21 r22 r23

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d8-d15}       \n" // r24 r25 r26 r27

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d0-d1}, [%3 :128]  \n" // r28

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        //                         "pld        [%4, #512]          \n"
                        "vldm       %4, {d16-d23}       \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "sub        %4, %4, #512        \n" // kptr -= 8 * 16;

                        "vstm       %0!, {d24-d31}      \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v20.4s, v21.4s}, [%0]      \n" // sum0 sum1

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmul   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v4.4s}, [%1]               \n" // r04

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v22.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v4.4s}, [%2]               \n" // r14

                        "fmla   v22.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v4.4s}, [%3]               \n" // r24

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        //                         "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4] \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "fadd   v20.4s, v20.4s, v22.4s      \n"
                        "fadd   v21.4s, v21.4s, v23.4s      \n"

                        "sub    %4, %4, #512                \n" // kptr -= 8 * 16;

                        "st1    {v20.4s, v21.4s}, [%0], #32 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128] \n" // sum0 sum1

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d0-d7}        \n" // r00 r01 r02 r03

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmul.f32   q14, q8, d0[0]      \n"
                        "vmul.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d8-d9}, [%1 :128]  \n" // r04

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n" // r10 r11 r12 r13

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d8-d9}, [%2 :128]  \n" // r14

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n" // r20 r21 r22 r23

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d8-d9}, [%3 :128]  \n" // r24

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        //                         "pld        [%4, #512]          \n"
                        "vldm       %4, {d16-d23}       \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "vadd.f32   q12, q12, q14       \n"
                        "vadd.f32   q13, q13, q15       \n"

                        "sub        %4, %4, #512        \n" // kptr -= 8 * 16;

                        "vst1.f32   {d24-d27}, [%0 :128]! \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v20.4s}, [%0]              \n" // sum0

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%1] \n" // r00 r01 r02

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmul   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmul   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v3.4s, v4.4s, v5.4s}, [%2] \n" // r10 r11 r12

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%3] \n" // r20 r21 r22

                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4], #64 \n"

                        "fmla   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        //                         "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4] \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "add    %1, %1, #32                 \n"

                        "fadd   v22.4s, v21.4s, v22.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "fadd   v23.4s, v23.4s, v22.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v20.4s, v20.4s, v23.4s      \n"

                        "sub    %4, %4, #512                \n" // kptr -= 8 * 16;

                        "st1    {v20.4s}, [%0], #16         \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d24-d25}, [%0 :128] \n" // sum0

                        "pld        [%1, #384]          \n"
                        "vldm       %1, {d0-d5}         \n" // r00 r01 r02

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmul.f32   q13, q8, d0[0]      \n"
                        "vmul.f32   q14, q9, d0[1]      \n"
                        "vmul.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%2, #384]          \n"
                        "vldm       %2, {d0-d5}         \n" // r10 r11 r12

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%3, #384]          \n"
                        "vldm       %3, {d0-d5}         \n" // r20 r21 r22

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        //                         "pld        [%4, #512]          \n"
                        "vldm       %4, {d16-d23}       \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "vadd.f32   q14, q14, q13       \n"

                        "add        %1, %1, #32         \n"

                        "vadd.f32   q15, q15, q14       \n"

                        "add        %2, %2, #32         \n"

                        "vadd.f32   q12, q12, q15       \n"

                        "add        %3, %3, #32         \n"

                        "sub        %4, %4, #512        \n" // kptr -= 8 * 16;

                        "vst1.f32   {d24-d25}, [%0 :128]! \n"

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
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    }
}

static void conv3x3s2_im2col_sgemm_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    // im2col
    Mat bottom_im2col(size, 9, inch, 16u, 4, opt.workspace_allocator);
    {
        const int gap = (w * 2 - outw * 2) * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            Mat out = bottom_im2col.channel(p);

            float* ptr0 = out.row(0);
            float* ptr1 = out.row(1);
            float* ptr2 = out.row(2);
            float* ptr3 = out.row(3);
            float* ptr4 = out.row(4);
            float* ptr5 = out.row(5);
            float* ptr6 = out.row(6);
            float* ptr7 = out.row(7);
            float* ptr8 = out.row(8);

            const float* r0 = img.row(0);
            const float* r1 = img.row(1);
            const float* r2 = img.row(2);

            for (int i = 0; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);

                    vst1q_f32(ptr0, _r00);
                    vst1q_f32(ptr0 + 4, _r02);
                    vst1q_f32(ptr1, _r01);
                    vst1q_f32(ptr1 + 4, _r03);
                    vst1q_f32(ptr2, _r02);
                    vst1q_f32(ptr2 + 4, _r04);

                    vst1q_f32(ptr3, _r10);
                    vst1q_f32(ptr3 + 4, _r12);
                    vst1q_f32(ptr4, _r11);
                    vst1q_f32(ptr4 + 4, _r13);
                    vst1q_f32(ptr5, _r12);
                    vst1q_f32(ptr5 + 4, _r14);

                    vst1q_f32(ptr6, _r20);
                    vst1q_f32(ptr6 + 4, _r22);
                    vst1q_f32(ptr7, _r21);
                    vst1q_f32(ptr7 + 4, _r23);
                    vst1q_f32(ptr8, _r22);
                    vst1q_f32(ptr8 + 4, _r24);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    ptr8 += 8;
                }
                for (; j < outw; j++)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);

                    vst1q_f32(ptr0, _r00);
                    vst1q_f32(ptr1, _r01);
                    vst1q_f32(ptr2, _r02);
                    vst1q_f32(ptr3, _r10);
                    vst1q_f32(ptr4, _r11);
                    vst1q_f32(ptr5, _r12);
                    vst1q_f32(ptr6, _r20);
                    vst1q_f32(ptr7, _r21);
                    vst1q_f32(ptr8, _r22);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    ptr8 += 4;
                }

                r0 += gap;
                r1 += gap;
                r2 += gap;
            }
        }
    }

    im2col_sgemm_pack4_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
