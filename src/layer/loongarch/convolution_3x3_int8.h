// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void conv3x3s1_winograd43_transform_kernel_int8_lsx(const Mat& kernel, Mat& kernel_tm_packed, int inch, int outch, const Option& opt)
{
    // winograd43 transform kernel
    Mat kernel_tm(6 * 6, inch, outch, (size_t)2u);

    const short ktm[6][3] = {
        {6, 0, 0},
        {-4, -4, -4},
        {-4, 4, -4},
        {1, 2, 4},
        {1, -2, 4},
        {0, 0, 6}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const signed char* kernel0 = (const signed char*)kernel + p * inch * 9 + q * 9;
            short* kernel_tm0 = kernel_tm.channel(p).row<short>(q);

            // transform kernel
            const signed char* k0 = kernel0;
            const signed char* k1 = kernel0 + 3;
            const signed char* k2 = kernel0 + 6;

            // h
            short tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                short* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 36-inch-outch
    // dst = 2b-inch-36-outch/2b
#if __loongarch_sx
    if (outch >= 4)
    {
        if (inch >= 4)
            kernel_tm_packed.create(inch / 4 + inch % 4, 36, outch / 4 + outch % 4, (size_t)2u * 16, 16);
        else
            kernel_tm_packed.create(inch, 36, outch / 4 + outch % 4, (size_t)2u * 4, 4);
    }
#else  // __loongarch_sx
    if (outch >= 2)
    {
        kernel_tm_packed.create(inch, 36, outch / 2 + outch % 2, (size_t)2u * 2, 2);
    }
#endif // __loongarch_sx
    else
    {
#if __loongarch_sx
        if (inch >= 4)
            kernel_tm_packed.create(inch / 4 + inch % 4, 36, outch, (size_t)2u * 4, 4);
        else
#endif // __loongarch_sx
        {
            kernel_tm_packed.create(inch, 36, outch, (size_t)2u, 1);
        }
    }

    int p = 0;
#if __loongarch_sx
    for (; p + 3 < outch; p += 4)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);
        const Mat k2 = kernel_tm.channel(p + 2);
        const Mat k3 = kernel_tm.channel(p + 3);

        Mat g0 = kernel_tm_packed.channel(p / 4);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            int q = 0;
            for (; q + 3 < inch; q += 4)
            {
                g00[0] = k0.row<const short>(q)[k];
                g00[1] = k0.row<const short>(q + 1)[k];
                g00[2] = k0.row<const short>(q + 2)[k];
                g00[3] = k0.row<const short>(q + 3)[k];
                g00[4] = k1.row<const short>(q)[k];
                g00[5] = k1.row<const short>(q + 1)[k];
                g00[6] = k1.row<const short>(q + 2)[k];
                g00[7] = k1.row<const short>(q + 3)[k];
                g00[8] = k2.row<const short>(q)[k];
                g00[9] = k2.row<const short>(q + 1)[k];
                g00[10] = k2.row<const short>(q + 2)[k];
                g00[11] = k2.row<const short>(q + 3)[k];
                g00[12] = k3.row<const short>(q)[k];
                g00[13] = k3.row<const short>(q + 1)[k];
                g00[14] = k3.row<const short>(q + 2)[k];
                g00[15] = k3.row<const short>(q + 3)[k];
                g00 += 16;
            }
            for (; q < inch; q++)
            {
                g00[0] = k0.row<const short>(q)[k];
                g00[1] = k1.row<const short>(q)[k];
                g00[2] = k2.row<const short>(q)[k];
                g00[3] = k3.row<const short>(q)[k];
                g00 += 4;
            }
        }
    }
#else  // __loongarch_sx
    for (; p + 1 < outch; p += 2)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);

        Mat g0 = kernel_tm_packed.channel(p / 2);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            int q = 0;
            for (; q < inch; q++)
            {
                g00[0] = k0.row<const short>(q)[k];
                g00[1] = k1.row<const short>(q)[k];
                g00 += 2;
            }
        }
    }
#endif // __loongarch_sx
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

#if __loongarch_sx
        Mat g0 = kernel_tm_packed.channel(p / 4 + p % 4);
#else
        Mat g0 = kernel_tm_packed.channel(p / 2 + p % 2);
#endif

        for (int k = 0; k < 36; k++)
        {
            short* g00 = g0.row<short>(k);

            int q = 0;
#if __loongarch_sx
            for (; q + 3 < inch; q += 4)
            {
                g00[0] = k0.row<const short>(q)[k];
                g00[1] = k0.row<const short>(q + 1)[k];
                g00[2] = k0.row<const short>(q + 2)[k];
                g00[3] = k0.row<const short>(q + 3)[k];
                g00 += 4;
            }
#endif // __loongarch_sx
            for (; q < inch; q++)
            {
                g00[0] = k0.row<const short>(q)[k];
                g00 += 1;
            }
        }
    }
}

static void conv3x3s1_winograd43_int8_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    //     size_t elemsize = bottom_blob.elemsize;
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

        bottom_blob_tm.create(tiles, 36, inch, 2u * elempack, elempack, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_int8_lsx(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_int8_lsx(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, 1, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd43_transform_output_int8_lsx(top_blob_tm, top_blob_bordered, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
