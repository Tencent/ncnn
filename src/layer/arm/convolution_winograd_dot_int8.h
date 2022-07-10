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

static void convolution_winograd_dot_int8_neon(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 2u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    {
        if (tiles >= 2)
            bottom_blob_tm2.create(inch, tiles / 2 + tiles % 2, batch, 4u, 2, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(inch, tiles, batch, 2u, 1, opt.workspace_allocator);
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 1 < tiles; i += 2)
        {
            short* tmpptr = tm2.row<short>(i / 2);

            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r0[1];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 2;
            }
        }
        for (; i < tiles; i++)
        {
            short* tmpptr = tm2.row<short>(i / 2 + i % 2);

            const short* r0 = (const short*)bottom_blob_tm + r * tiles + i;

            int q = 0;
            for (; q < inch; q++)
            {
                tmpptr[0] = r0[0];
                r0 += bottom_blob_tm.cstep;
                tmpptr += 1;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u, 1, opt.workspace_allocator);

#if __ARM_NEON
    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);
        int* output2_tm = top_blob_tm.channel(p + 2);
        int* output3_tm = top_blob_tm.channel(p + 3);

        const Mat kernel0_tm = kernel_tm.channel(p / 4);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn = inch;

                int sum00 = 0;
                int sum10 = 0;
                int sum20 = 0;
                int sum30 = 0;
                int sum01 = 0;
                int sum11 = 0;
                int sum21 = 0;
                int sum31 = 0;

                for (int j = 0; j < nn; j++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];

                    signed short k0 = k0[0];
                    signed short k1 = k0[1];
                    signed short k2 = k0[2];
                    signed short k3 = k0[3];

                    sum00 += val0 * k0;
                    sum10 += val0 * k1;
                    sum20 += val0 * k2;
                    sum30 += val0 * k3;
                    sum01 += val1 * k0;
                    sum11 += val1 * k1;
                    sum21 += val1 * k2;
                    sum31 += val1 * k3;

                    r0 += 2;
                    k0 += 4;
                }

                output0_tm[0] = sum00;
                output1_tm[0] = sum10;
                output2_tm[0] = sum20;
                output3_tm[0] = sum30;
                output0_tm[1] = sum01;
                output1_tm[1] = sum11;
                output2_tm[1] = sum21;
                output3_tm[1] = sum31;
                output0_tm += 2;
                output1_tm += 2;
                output2_tm += 2;
                output3_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn = inch;

                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                for (int j = 0; j < nn; j++)
                {
                    signed short val = r0[0];

                    sum0 += val * k0[0];
                    sum1 += val * k0[1];
                    sum2 += val * k0[2];
                    sum3 += val * k0[3];

                    r0 += 1;
                    k0 += 4;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output2_tm[0] = sum2;
                output3_tm[0] = sum3;
                output0_tm += 1;
                output1_tm += 1;
                output2_tm += 1;
                output3_tm += 1;
            }
        }
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

#if __ARM_NEON
        const Mat kernel0_tm = kernel_tm.channel(p / 4 + p % 4);
#else
        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

                int nn = inch;

                for (int q = 0; q < nn; q++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];
                    signed short w = k0[0];

                    sum0 += val0 * w;
                    sum1 += val1 * w;

                    k0 += 1;
                    r0 += 2;
                }

                output0_tm[0] = sum0;
                output0_tm[1] = sum1;
                output0_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum = 0;

                int nn = inch;

                for (int q = 0; q < nn; q++)
                {
                    signed short val = r0[0];
                    signed short w = k0[0];

                    sum += val * w;

                    k0 += 1;
                    r0 += 1;
                }

                output0_tm[0] = sum;
                output0_tm++;
            }
        }
    }
}
