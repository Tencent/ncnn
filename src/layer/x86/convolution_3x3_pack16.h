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

static void conv3x3s1_winograd63_transform_kernel_pack16_avx512(const Mat& kernel, Mat& kernel_tm_pack16, int inch, int outch, const Option& opt)
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
    // dst = 16b-16a-inch/16a-64-outch/16b
    kernel_tm_pack16.create(inch / 16, 64, outch / 16, (size_t)4u * 16 * 16, 16 * 16);
    for (int q = 0; q + 15 < outch; q += 16)
    {
        Mat g0 = kernel_tm_pack16.channel(q / 16);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 15 < inch; p += 16)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        conv3x3s1_winograd63_transform_input_pack16_avx512(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack16_avx512(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd63_transform_output_pack16_avx512(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_pack16_avx512(const Mat& kernel, Mat& kernel_tm_pack16, int inch, int outch, const Option& opt)
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
    // dst = 16b-16a-inch/16a-36-outch/16b
    kernel_tm_pack16.create(inch / 16, 36, outch / 16, (size_t)4u * 16 * 16, 16 * 16);
    for (int q = 0; q + 15 < outch; q += 16)
    {
        Mat g0 = kernel_tm_pack16.channel(q / 16);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + 15 < inch; p += 16)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        conv3x3s1_winograd43_transform_input_pack16_avx512(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack16_avx512(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd43_transform_output_pack16_avx512(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd23_transform_kernel_pack16_avx512(const Mat& kernel, Mat& kernel_tm_pack16, int inch, int outch, const Option& opt)
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
    // dst = pb-pa-inch/pa-16-outch/pb
    kernel_tm_pack16.create(inch / 16, 16, outch / 16, (size_t)4u * 16 * 16, 16 * 16);
    for (int q = 0; q + 15 < outch; q += 16)
    {
        Mat g0 = kernel_tm_pack16.channel(q / 16);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + 15 < inch; p += 16)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_pack16_avx512(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack16_avx512(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd23_transform_output_pack16_avx512(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
