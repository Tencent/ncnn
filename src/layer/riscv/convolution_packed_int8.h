// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
    if (outch >= 2)
    {
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)4u, 4);
        else
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch, (size_t)2u, 2);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
    for (; q + 1 < outch; q += 2)
    {
        const signed char* kptr0 = (const signed char*)kernel + q * inch * maxk;
        const signed char* kptr1 = (const signed char*)kernel + (q + 1) * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / 2);

        int p = 0;
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k0[maxk];
                g00[3] = k1[maxk];
                g00 += 4;
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr0 + k;
                const signed char* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);

        int p = 0;
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00 += 2;
            }

            kptr += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                g00[0] = k0[0];
                g00++;
            }
        }
    }
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int N = bottom_blob.cstep * elempack;

        int* outptr0 = top_blob.channel(p);
        int* outptr1 = top_blob.channel(p + 1);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum00 = 0;
            int sum01 = 0;
            int sum10 = 0;
            int sum11 = 0;

            const signed char* kptr = weight_data_tm.channel(p / 2);

            int q = 0;
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum00 += r0s[0] * kptr[0];
                        sum10 += r0s[0] * kptr[1];
                        sum00 += r0s[N] * kptr[2];
                        sum10 += r0s[N] * kptr[3];
                        sum01 += r1s[0] * kptr[0];
                        sum11 += r1s[0] * kptr[1];
                        sum01 += r1s[N] * kptr[2];
                        sum11 += r1s[N] * kptr[3];

                        kptr += 4;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum00 += r0s[0] * kptr[0];
                        sum10 += r0s[0] * kptr[1];
                        sum01 += r1s[0] * kptr[0];
                        sum11 += r1s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr0 += 2;
            outptr1 += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum0 = 0;
            int sum1 = 0;

            const signed char* kptr = weight_data_tm.channel(p / 2);

            int q = 0;
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];
                        sum0 += r0s[N] * kptr[2];
                        sum1 += r0s[N] * kptr[3];

                        kptr += 4;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r0s[0] * kptr[1];

                        kptr += 2;
                    }
                }
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int i1 = (ij + 1) / outw;
            const int j0 = ij % outw;
            const int j1 = (ij + 1) % outw;

            int sum0 = 0;
            int sum1 = 0;

            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);

            int q = 0;
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum0 += r0s[N] * kptr[1];
                        sum1 += r1s[0] * kptr[0];
                        sum1 += r1s[N] * kptr[1];

                        kptr += 2;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum0 += r0s[0] * kptr[0];
                        sum1 += r1s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;

            const signed char* kptr = weight_data_tm.channel(p / 2 + p % 2);

            int q = 0;
            for (; q + 1 < inch; q += 2)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];
                        sum += r0s[N] * kptr[1];

                        kptr += 2;
                    }
                }
            }
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];

                    // if (elempack == 1)
                    {
                        sum += r0s[0] * kptr[0];

                        kptr += 1;
                    }
                }
            }

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
