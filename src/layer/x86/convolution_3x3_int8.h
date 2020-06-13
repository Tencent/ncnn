// BUG1989 is pleased to support the open source community by supporting ncnn available.
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

static void conv3x3s1_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_int8_sse(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(4 * 4, inch, outch, 2ul);

    // G
    const short ktm[4][3] = {
        {2, 0, 0},
        {1, 1, 1},
        {1, -1, 1},
        {0, 0, 2}
    };

    #pragma omp parallel for
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
            short tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = (short)k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = (short)k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = (short)k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++)
            {
                short* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
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
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4 * 4, tiles, inch, 2u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const signed char* img = bottom_blob_bordered.channel(q);
            short* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const signed char* r0 = img + w * j * 2;
                const signed char* r1 = r0 + w;
                const signed char* r2 = r1 + w;
                const signed char* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    short d0[4], d1[4], d2[4], d3[4];
                    short w0[4], w1[4], w2[4], w3[4];
                    short t0[4], t1[4], t2[4], t3[4];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                    }
                    // U = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm0[n + 4] = d1[n];
                        out_tm0[n + 8] = d2[n];
                        out_tm0[n + 12] = d3[n];
                    }

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p + 1);
            Mat out2_tm = top_blob_tm.channel(p + 2);
            Mat out3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p + 1);
            const Mat kernel2_tm = kernel_tm.channel(p + 2);
            const Mat kernel3_tm = kernel_tm.channel(p + 3);

            for (int i = 0; i < tiles; i++)
            {
                int* output0_tm = out0_tm.row<int>(i);
                int* output1_tm = out1_tm.row<int>(i);
                int* output2_tm = out2_tm.row<int>(i);
                int* output3_tm = out3_tm.row<int>(i);

                int sum0[16] = {0};
                int sum1[16] = {0};
                int sum2[16] = {0};
                int sum3[16] = {0};

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
                    const short* r1 = bottom_blob_tm.channel(q + 1).row<short>(i);
                    const short* r2 = bottom_blob_tm.channel(q + 2).row<short>(i);
                    const short* r3 = bottom_blob_tm.channel(q + 3).row<short>(i);

                    const short* k0 = kernel0_tm.row<short>(q);
                    const short* k1 = kernel1_tm.row<short>(q);
                    const short* k2 = kernel2_tm.row<short>(q);
                    const short* k3 = kernel3_tm.row<short>(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += (int)r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += (int)r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += (int)r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += (int)r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += (int)r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += (int)r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += (int)r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += (int)r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += (int)r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += (int)r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += (int)r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += (int)r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += (int)r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += (int)r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += (int)r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += (int)r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q < inch; q++)
                {
                    const short* r0 = bottom_blob_tm.channel(q).row<short>(i);

                    const short* k0 = kernel0_tm.row<short>(q);
                    const short* k1 = kernel1_tm.row<short>(q);
                    const short* k2 = kernel2_tm.row<short>(q);
                    const short* k3 = kernel3_tm.row<short>(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += (int)r0[n] * k0[n];
                        sum1[n] += (int)r0[n] * k1[n];
                        sum2[n] += (int)r0[n] * k2[n];
                        sum3[n] += (int)r0[n] * k3[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i = 0; i < tiles; i++)
            {
                int* output0_tm = out0_tm.row<int>(i);

                int sum0[16] = {0};

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
                    const short* r1 = bottom_blob_tm.channel(q + 1).row<short>(i);
                    const short* r2 = bottom_blob_tm.channel(q + 2).row<short>(i);
                    const short* r3 = bottom_blob_tm.channel(q + 3).row<short>(i);

                    const short* k0 = kernel0_tm.row<short>(q);
                    const short* k1 = kernel0_tm.row<short>(q + 1);
                    const short* k2 = kernel0_tm.row<short>(q + 2);
                    const short* k3 = kernel0_tm.row<short>(q + 3);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += (int)r0[n] * k0[n];
                        sum0[n] += (int)r1[n] * k1[n];
                        sum0[n] += (int)r2[n] * k2[n];
                        sum0[n] += (int)r3[n] * k3[n];
                    }
                }

                for (; q < inch; q++)
                {
                    const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
                    const short* k0 = kernel0_tm.row<short>(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += (int)r0[n] * k0[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // };

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

            for (int j = 0; j < nColBlocks; j++)
            {
                int* outRow0 = out.row<int>(j * 2);
                int* outRow1 = out.row<int>(j * 2 + 1);

                for (int i = 0; i < nRowBlocks; i++)
                {
                    int* out_tile = out_tm.row<int>(j * nRowBlocks + i);

                    int s0[4], s1[4], s2[4], s3[4];
                    int w0[4], w1[4];
                    int d0[2], d1[2], d2[2], d3[2];
                    int o0[2], o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 4];
                        s2[n] = out_tile[n + 8];
                        s3[n] = out_tile[n + 12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n];
                        o1[n] = d1[n] - d2[n] + d3[n];
                    }
                    // save to top blob tm,why right 2,because the G' = G*2
                    outRow0[0] = o0[0] >> 2;
                    outRow0[1] = o0[1] >> 2;
                    outRow1[0] = o1[0] >> 2;
                    outRow1[1] = o1[1] >> 2;

                    outRow0 += 2;
                    outRow1 += 2;
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_int8_sse(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(6 * 6, inch, outch, 2ul);

    // G
    // const float ktm[6][3] = {
    //     {  1.0f/4,     0.0f,    0.0f},
    //     { -1.0f/6,  -1.0f/6, -1.0f/6},
    //     { -1.0f/6,   1.0f/6, -1.0f/6},
    //     { 1.0f/24,  1.0f/12,  1.0f/6},
    //     { 1.0f/24, -1.0f/12,  1.0f/6},
    //     {    0.0f,     0.0f,    1.0f}
    // };
    const short ktm[6][3] = {
        {6, 0, 0},
        {-4, -4, -4},
        {-4, 4, -4},
        {1, 2, 4},
        {1, -2, 4},
        {0, 0, 24}
    };

    #pragma omp parallel for
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
}

static void conv3x3s1_winograd43_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
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
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(6 * 6, tiles, inch, 2u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =	4 * r00  - 5 * r02	+ r04
        // 1 = -4 * (r01 + r02)  + r03 + r04
        // 2 =	4 * (r01 - r02)  - r03 + r04
        // 3 = -2 * r01 - r02 + 2 * r03 + r04
        // 4 =	2 * r01 - r02 - 2 * r03 + r04
        // 5 =	4 * r01 - 5 * r03 + r05

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const signed char* img = bottom_blob_bordered.channel(q);
            short* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const signed char* r0 = img + w * j * 4;
                const signed char* r1 = r0 + w;
                const signed char* r2 = r1 + w;
                const signed char* r3 = r2 + w;
                const signed char* r4 = r3 + w;
                const signed char* r5 = r4 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    short d0[6], d1[6], d2[6], d3[6], d4[6], d5[6];
                    short w0[6], w1[6], w2[6], w3[6], w4[6], w5[6];
                    short t0[6], t1[6], t2[6], t3[6], t4[6], t5[6];

                    // load
                    for (int n = 0; n < 6; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                        d4[n] = r4[n];
                        d5[n] = r5[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 6; n++)
                    {
                        w0[n] = 4 * d0[n] - 5 * d2[n] + d4[n];
                        w1[n] = -4 * d1[n] - 4 * d2[n] + d3[n] + d4[n];
                        w2[n] = 4 * d1[n] - 4 * d2[n] - d3[n] + d4[n];
                        w3[n] = -2 * d1[n] - d2[n] + 2 * d3[n] + d4[n];
                        w4[n] = 2 * d1[n] - d2[n] - 2 * d3[n] + d4[n];
                        w5[n] = 4 * d1[n] - 5 * d3[n] + d5[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t4[0] = w0[4];
                        t5[0] = w0[5];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t4[1] = w1[4];
                        t5[1] = w1[5];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t4[2] = w2[4];
                        t5[2] = w2[5];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                        t4[3] = w3[4];
                        t5[3] = w3[5];
                        t0[4] = w4[0];
                        t1[4] = w4[1];
                        t2[4] = w4[2];
                        t3[4] = w4[3];
                        t4[4] = w4[4];
                        t5[4] = w4[5];
                        t0[5] = w5[0];
                        t1[5] = w5[1];
                        t2[5] = w5[2];
                        t3[5] = w5[3];
                        t4[5] = w5[4];
                        t5[5] = w5[5];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 6; n++)
                    {
                        d0[n] = 4 * t0[n] - 5 * t2[n] + t4[n];
                        d1[n] = -4 * t1[n] - 4 * t2[n] + t3[n] + t4[n];
                        d2[n] = 4 * t1[n] - 4 * t2[n] - t3[n] + t4[n];
                        d3[n] = -2 * t1[n] - t2[n] + 2 * t3[n] + t4[n];
                        d4[n] = 2 * t1[n] - t2[n] - 2 * t3[n] + t4[n];
                        d5[n] = 4 * t1[n] - 5 * t3[n] + t5[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 6; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm0[n + 6] = d1[n];
                        out_tm0[n + 12] = d2[n];
                        out_tm0[n + 18] = d3[n];
                        out_tm0[n + 24] = d4[n];
                        out_tm0[n + 30] = d5[n];
                    }

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;

                    out_tm0 += 36;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(36, tiles, outch, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i = 0; i < tiles; i++)
            {
                int* output0_tm = out0_tm.row<int>(i);

                int sum0[36] = {0};

                for (int q = 0; q < inch; q++)
                {
                    const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
                    const short* k0 = kernel0_tm.row<short>(q);

                    for (int n = 0; n < 36; n++)
                    {
                        sum0[n] += (int)r0[n] * k0[n];
                    }
                }

                for (int n = 0; n < 36; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
        // AT
        // const float itm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 =	r00 + r01 + r02 + r03 +	r04
        // 1 =		  r01 - r02 + 2 * (r03 - r04)
        // 2 =		  r01 + r02 + 4 * (r03 + r04)
        // 3 =		  r01 - r02 + 8 * (r03 - r04)  + r05

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

            for (int j = 0; j < nColBlocks; j++)
            {
                int* outRow0 = out.row<int>(j * 4);
                int* outRow1 = out.row<int>(j * 4 + 1);
                int* outRow2 = out.row<int>(j * 4 + 2);
                int* outRow3 = out.row<int>(j * 4 + 3);

                for (int i = 0; i < nRowBlocks; i++)
                {
                    int* out_tile = out_tm.row<int>(j * nRowBlocks + i);

                    int s0[6], s1[6], s2[6], s3[6], s4[6], s5[6];
                    int w0[6], w1[6], w2[6], w3[6];
                    int d0[4], d1[4], d2[4], d3[4], d4[4], d5[4];
                    int o0[4], o1[4], o2[4], o3[4];
                    // load
                    for (int n = 0; n < 6; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 6];
                        s2[n] = out_tile[n + 12];
                        s3[n] = out_tile[n + 18];
                        s4[n] = out_tile[n + 24];
                        s5[n] = out_tile[n + 30];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 6; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + s5[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d0[2] = w2[0];
                        d0[3] = w3[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d1[2] = w2[1];
                        d1[3] = w3[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d2[2] = w2[2];
                        d2[3] = w3[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                        d3[2] = w2[3];
                        d3[3] = w3[3];
                        d4[0] = w0[4];
                        d4[1] = w1[4];
                        d4[2] = w2[4];
                        d4[3] = w3[4];
                        d5[0] = w0[5];
                        d5[1] = w1[5];
                        d5[2] = w2[5];
                        d5[3] = w3[5];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 4; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n] + d3[n] + d4[n];
                        o1[n] = d1[n] - d2[n] + 2 * d3[n] - 2 * d4[n];
                        o2[n] = d1[n] + d2[n] + 4 * d3[n] + 4 * d4[n];
                        o3[n] = d1[n] - d2[n] + 8 * d3[n] - 8 * d4[n] + d5[n];
                    }
                    // save to top blob tm
                    for (int n = 0; n < 4; n++)
                    {
                        outRow0[n] = o0[n] / 576;
                        outRow1[n] = o1[n] / 576;
                        outRow2[n] = o2[n] / 576;
                        outRow3[n] = o3[n] / 576;
                    }

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s2_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char* kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}
