// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd23_transform_kernel_rvv(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
{
    Mat kernel_tm(4 * 4, inch, outch);

    // G
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
    // dst = inch-16-outch
#if __riscv_vector
    kernel_tm2.create(8 * inch, 16, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(2 * inch, 16, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __riscv_vector
    for (; q + 1 < outch; q += 2)
    {
        Mat g0 = kernel_tm2.channel(q / 2);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
#if __riscv_vector
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q / 2 + q % 2);
#endif

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd23_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_rvv(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd23_transform_output_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_rvv(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
{
    Mat kernel_tm(6 * 6, inch, outch);

    // G
    const float sq2 = 1.41421356237f;
    const float ktm[6][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 3, -sq2 / 3, -1.0f / 3},
        {-2.0f / 3, sq2 / 3, -1.0f / 3},
        {1.0f / 6, sq2 / 6, 1.0f / 3},
        {1.0f / 6, -sq2 / 6, 1.0f / 3},
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
    // dst = inch-36-outch
#if __riscv_vector
    kernel_tm2.create(8 * inch, 36, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(2 * inch, 36, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __riscv_vector
    for (; q + 1 < outch; q += 2)
    {
        Mat g0 = kernel_tm2.channel(q / 2);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
#if __riscv_vector
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q / 2 + q % 2);
#endif

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd43_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2, winograd F(4,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;

    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_rvv(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd43_transform_output_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
