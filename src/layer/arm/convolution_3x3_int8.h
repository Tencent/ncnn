// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// author:BUG1989 (https://github.com/BUG1989/) Long-term support.
// author:FuGuangping (https://github.com/fu1899) Implemented the first version of INT8 quantization on ARMv7.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static void conv3x3s1_winograd23_transform_kernel_int8_neon(const Mat& kernel, std::vector<Mat>& kernel_tm2, int inch, int outch)
{
    Mat kernel_tm(4 * 4, inch, outch, 2ul);

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

    for (int r = 0; r < 4; r++)
    {
        Mat kernel_tm_test(4 * 8, inch, outch / 8 + (outch % 8) / 4 + outch % 4, 2u);

        int p = 0;
        for (; p + 7 < outch; p += 8)
        {
            const short* kernel0 = (const short*)kernel_tm + (p + 0) * inch * 16;
            const short* kernel1 = (const short*)kernel_tm + (p + 1) * inch * 16;
            const short* kernel2 = (const short*)kernel_tm + (p + 2) * inch * 16;
            const short* kernel3 = (const short*)kernel_tm + (p + 3) * inch * 16;
            const short* kernel4 = (const short*)kernel_tm + (p + 4) * inch * 16;
            const short* kernel5 = (const short*)kernel_tm + (p + 5) * inch * 16;
            const short* kernel6 = (const short*)kernel_tm + (p + 6) * inch * 16;
            const short* kernel7 = (const short*)kernel_tm + (p + 7) * inch * 16;

            short* ktmp = kernel_tm_test.channel(p / 8);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp[16] = kernel4[r * 4 + 0];
                ktmp[17] = kernel4[r * 4 + 1];
                ktmp[18] = kernel4[r * 4 + 2];
                ktmp[19] = kernel4[r * 4 + 3];

                ktmp[20] = kernel5[r * 4 + 0];
                ktmp[21] = kernel5[r * 4 + 1];
                ktmp[22] = kernel5[r * 4 + 2];
                ktmp[23] = kernel5[r * 4 + 3];

                ktmp[24] = kernel6[r * 4 + 0];
                ktmp[25] = kernel6[r * 4 + 1];
                ktmp[26] = kernel6[r * 4 + 2];
                ktmp[27] = kernel6[r * 4 + 3];

                ktmp[28] = kernel7[r * 4 + 0];
                ktmp[29] = kernel7[r * 4 + 1];
                ktmp[30] = kernel7[r * 4 + 2];
                ktmp[31] = kernel7[r * 4 + 3];

                ktmp += 32;
                kernel0 += 16;
                kernel1 += 16;
                kernel2 += 16;
                kernel3 += 16;
                kernel4 += 16;
                kernel5 += 16;
                kernel6 += 16;
                kernel7 += 16;
            }
        }

        for (; p + 3 < outch; p += 4)
        {
            const short* kernel0 = (const short*)kernel_tm + (p + 0) * inch * 16;
            const short* kernel1 = (const short*)kernel_tm + (p + 1) * inch * 16;
            const short* kernel2 = (const short*)kernel_tm + (p + 2) * inch * 16;
            const short* kernel3 = (const short*)kernel_tm + (p + 3) * inch * 16;

            short* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp += 16;
                kernel0 += 16;
                kernel1 += 16;
                kernel2 += 16;
                kernel3 += 16;
            }
        }

        for (; p < outch; p++)
        {
            const short* kernel0 = (const short*)kernel_tm + p * inch * 16;

            short* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4 + p % 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp += 4;
                kernel0 += 16;
            }
        }
        kernel_tm2.push_back(kernel_tm_test);
    }
}

static void conv3x3s1_winograd23_int8_neon(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat>& kernel_tm_test, const Option& opt)
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

        int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4, inch, tiles * 4, 2u, opt.workspace_allocator);

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

            for (int j = 0; j < nColBlocks; j++)
            {
                const signed char* r0 = img + w * j * 2;
                const signed char* r1 = r0 + w;
                const signed char* r2 = r1 + w;
                const signed char* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    short* out_tm0 = bottom_blob_tm.channel(tiles * 0 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm1 = bottom_blob_tm.channel(tiles * 1 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm2 = bottom_blob_tm.channel(tiles * 2 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm3 = bottom_blob_tm.channel(tiles * 3 + j * nRowBlocks + i).row<short>(q);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // load
                        "prfm   pldl1keep, [%0, #64]    \n"
                        "ld1    {v0.8b}, [%0]           \n"
                        "prfm   pldl1keep, [%1, #64]    \n"
                        "ld1    {v1.8b}, [%1]           \n"
                        "prfm   pldl1keep, [%2, #64]    \n"
                        "ld1    {v2.8b}, [%2]           \n"
                        "prfm   pldl1keep, [%3, #64]    \n"
                        "ld1    {v3.8b}, [%3]           \n"
                        // w = B_t * d, trans int8 to int16
                        "ssubl    v4.8h, v0.8b, v2.8b   \n" // d4
                        "saddl    v5.8h, v1.8b, v2.8b   \n" // d6
                        "ssubl    v6.8h, v2.8b, v1.8b   \n" // d8
                        "ssubl    v7.8h, v3.8b, v1.8b   \n" // d10
                        // transpose w to w_t
                        "trn1   v8.4h, v4.4h, v5.4h    \n"
                        "trn2   v9.4h, v4.4h, v5.4h    \n"
                        "trn1   v10.4h, v6.4h, v7.4h    \n"
                        "trn2   v11.4h, v6.4h, v7.4h    \n"

                        "trn1   v0.2s, v8.2s, v10.2s    \n"
                        "trn2   v2.2s, v8.2s, v10.2s    \n"
                        "trn1   v1.2s, v9.2s, v11.2s    \n"
                        "trn2   v3.2s, v9.2s, v11.2s    \n"
                        // U = B_t * d_t
                        "sub    v4.4h, v0.4h, v2.4h   \n"
                        "add    v5.4h, v1.4h, v2.4h   \n"
                        "sub    v6.4h, v2.4h, v1.4h   \n"
                        "sub    v7.4h, v3.4h, v1.4h   \n"
                        // save
                        "st1    {v4.4h}, [%4]   \n"
                        "st1    {v5.4h}, [%5]   \n"
                        "st1    {v6.4h}, [%6]   \n"
                        "st1    {v7.4h}, [%7]   \n"
                        : "=r"(r0),      // %0
                        "=r"(r1),      // %1
                        "=r"(r2),      // %2
                        "=r"(r3),      // %3
                        "=r"(out_tm0), // %4
                        "=r"(out_tm1), // %5
                        "=r"(out_tm2), // %6
                        "=r"(out_tm3)  // %7
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(out_tm0),
                        "5"(out_tm1),
                        "6"(out_tm2),
                        "7"(out_tm3)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
                    asm volatile(
                        // load
                        "pld         [%0, #64]     \n"
                        "vld1.s8     {d0}, [%0]    \n"
                        "pld         [%1, #64]     \n"
                        "vld1.s8     {d1}, [%1]    \n"
                        "pld         [%2, #64]     \n"
                        "vld1.s8     {d2}, [%2]    \n"
                        "pld         [%3, #64]     \n"
                        "vld1.s8     {d3}, [%3]    \n"
                        // w = B_t * d, trans int8 to int16
                        "vsubl.s8    q2, d0, d2    \n" // d4
                        "vaddl.s8    q3, d1, d2    \n" // d6
                        "vsubl.s8    q4, d2, d1    \n" // d8
                        "vsubl.s8    q5, d3, d1    \n" // d10
                        // transpose w to w_t
                        "vtrn.s16    d4, d6        \n"
                        "vtrn.s16    d8, d10       \n"
                        "vtrn.s32    d4, d8        \n"
                        "vtrn.s32    d6, d10       \n"
                        // U = B_t * d_t
                        "vsub.s16    d11, d4, d8   \n"
                        "vadd.s16    d12, d6, d8   \n"
                        "vsub.s16    d13, d8, d6   \n"
                        "vsub.s16    d14, d10, d6  \n"
                        // save
                        "vst1.s32    {d11}, [%4]   \n"
                        "vst1.s32    {d12}, [%5]   \n"
                        "vst1.s32    {d13}, [%6]   \n"
                        "vst1.s32    {d14}, [%7]   \n"
                        : "=r"(r0),      // %0
                        "=r"(r1),      // %1
                        "=r"(r2),      // %2
                        "=r"(r3),      // %3
                        "=r"(out_tm0), // %4
                        "=r"(out_tm1), // %5
                        "=r"(out_tm2), // %6
                        "=r"(out_tm3)  // %7
                        : "0"(r0),
                        "1"(r1),
                        "2"(r2),
                        "3"(r3),
                        "4"(out_tm0),
                        "5"(out_tm1),
                        "6"(out_tm2),
                        "7"(out_tm3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif // __aarch64__
#else
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
                        out_tm1[n] = d1[n];
                        out_tm2[n] = d2[n];
                        out_tm3[n] = d3[n];
                    }
#endif
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
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

        int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 4; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);
                int* output4_tm = top_blob_tm.channel(p + 4);
                int* output5_tm = top_blob_tm.channel(p + 5);
                int* output6_tm = top_blob_tm.channel(p + 6);
                int* output7_tm = top_blob_tm.channel(p + 7);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "eor    v4.16b, v4.16b, v4.16b    \n"
                        "eor    v5.16b, v5.16b, v5.16b    \n"
                        "eor    v6.16b, v6.16b, v6.16b    \n"
                        "eor    v7.16b, v7.16b, v7.16b    \n"
                        "mov    w4, %w20                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%9, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v8.4h}, [%8]            \n"
                        "ld1     {v9.4h, v10.4h}, [%9]    \n" // _k0 = vld1q_s16(kptr);
                        "add     %9, %9, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%9]   \n" // _k0n = vld1q_s16(kptr+8);
                        "add     %9, %9, #16              \n"
                        "ld1     {v13.4h, v14.4h}, [%9]   \n" // _k1 = vld1q_s16(kptr+16);
                        "add     %9, %9, #16              \n"
                        "ld1     {v15.4h, v16.4h}, [%9]   \n" // _k1n = vld1q_s16(kptr+24);
                        "add     %8, %8, #8               \n"
                        "add     %9, %9, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)
                        "smlal   v4.4s, v8.4h, v13.4h     \n" // sum4 += (a00-a03) * (k40-k43)
                        "smlal   v5.4s, v8.4h, v14.4h     \n" // sum5 += (a00-a03) * (k50-k53)
                        "smlal   v6.4s, v8.4h, v15.4h     \n" // sum6 += (a00-a03) * (k60-k63)
                        "smlal   v7.4s, v8.4h, v16.4h     \n" // sum7 += (a00-a03) * (k70-k73)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //
                        "st1     {v4.4s}, [%4]            \n" //
                        "st1     {v5.4s}, [%5]            \n" //
                        "st1     {v6.4s}, [%6]            \n" //
                        "st1     {v7.4s}, [%7]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "vmov.s32    q4, #0           \n"
                        "vmov.s32    q5, #0           \n"
                        "vmov.s32    q6, #0           \n"
                        "vmov.s32    q7, #0           \n"
                        "mov         r4, %20          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%8]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%9]  \n" // _k0 = vld1q_s16(kptr);
                        "add         %9, #16          \n"
                        "vld1.s16    {d20-d21}, [%9]  \n" // _k0n = vld1q_s16(kptr+8);
                        "add         %9, #16          \n"
                        "vld1.s16    {d22-d23}, [%9]  \n" // _k1 = vld1q_s16(kptr+16);
                        "add         %9, #16          \n"
                        "vld1.s16    {d24-d25}, [%9]  \n" // _k1n = vld1q_s16(kptr+24);
                        "add         %9, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)
                        "vmlal.s16   q4, d16, d22     \n" // sum4 += (a00-a03) * (k40-k43)
                        "vmlal.s16   q5, d16, d23     \n" // sum5 += (a00-a03) * (k50-k53)
                        "vmlal.s16   q6, d16, d24     \n" // sum6 += (a00-a03) * (k60-k63)
                        "vmlal.s16   q7, d16, d25     \n" // sum7 += (a00-a03) * (k70-k73)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"
                        "vst1.s32    {d8-d9}, [%4]    \n"
                        "vst1.s32    {d10-d11}, [%5]  \n"
                        "vst1.s32    {d12-d13}, [%6]  \n"
                        "vst1.s32    {d14-d15}, [%7]  \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};
                    int sum4[4] = {0};
                    int sum5[4] = {0};
                    int sum6[4] = {0};
                    int sum7[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                            sum4[n] += (int)r0[n] * kptr[n + 16];
                            sum5[n] += (int)r0[n] * kptr[n + 20];
                            sum6[n] += (int)r0[n] * kptr[n + 24];
                            sum7[n] += (int)r0[n] * kptr[n + 28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                    output4_tm += 16;
                    output5_tm += 16;
                    output6_tm += 16;
                    output7_tm += 16;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "mov    w4, %w12                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%5, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v8.4h}, [%4]            \n"
                        "ld1     {v9.4h, v10.4h}, [%5]    \n" // _k0 = vld1q_s16(kptr);
                        "add     %5, %5, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%5]   \n" // _k0n = vld1q_s16(kptr+8);
                        "add     %4, %4, #8               \n"
                        "add     %5, %5, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "mov         r4, %12          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%4]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%5]  \n" // _k0 = vld1q_s16(kptr);
                        "add         %5, #16          \n"
                        "vld1.s16    {d20-d21}, [%5]  \n" // _k0n = vld1q_s16(kptr+8);
                        "add         %5, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                }
            }

            remain_outch_start += nn_outch << 2;

            for (int p = remain_outch_start; p < outch; p++)
            {
                int* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4 + p % 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "mov    w4, %w6                   \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        //"prfm    pldl1keep, [%2, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v8.4h}, [%1]            \n"
                        "ld1     {v9.4h}, [%2]            \n" // _k0 = vld1q_s16(kptr);
                        "add     %1, %1, #8               \n"
                        "add     %2, %2, #8               \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "mov         r4, %6           \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%1]      \n" // _r0 = vld1_s16(r0);  // input inch0
                        "add         %1, #8           \n"
                        "vld1.s16    {d18}, [%2]      \n" // _k0 = vld1q_s16(kptr);
                        "add         %2, #8           \n"
                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "r4", "q0", "q8", "q9");
#endif // __aarch64__
#else
                    int sum0[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
#endif
                    output0_tm += 16;
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

        int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm / 4;

#if __ARM_NEON
        int32x2_t _shift = vdup_n_s32(-2);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            int* out_tile = top_blob_tm.channel(p);
            int* outRow0 = top_blob_bordered.channel(p);
            int* outRow1 = outRow0 + outw;

            for (int j = 0; j < nColBlocks; j++)
            {
                for (int i = 0; i < nRowBlocks; i++)
                {
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]  \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"

                        "add    v0.4s, v0.4s, v1.4s    \n" // s0 = s0 + s1 + s2;
                        "sub    v1.4s, v1.4s, v2.4s    \n"
                        "add    v0.4s, v0.4s, v2.4s    \n" // s1 = s1 - s2 + s3;
                        "add    v1.4s, v1.4s, v3.4s    \n"

                        "trn1   v4.4s, v0.4s, v1.4s    \n"
                        "trn2   v5.4s, v0.4s, v1.4s    \n"

                        "dup    v6.2d, v4.d[1]         \n"
                        "dup    v7.2d, v5.d[1]         \n"

                        "add    v0.2s, v4.2s, v5.2s    \n" // o0 = d0 + d1 + d2;
                        "sub    v1.2s, v5.2s, v6.2s    \n"
                        "add    v0.2s, v0.2s, v6.2s    \n" // o1 = d1 - d2 + d3;
                        "add    v1.2s, v1.2s, v7.2s    \n"

                        "sshl    v0.2s, v0.2s, %6.2s   \n" // o0 = o0 >> 2
                        "sshl    v1.2s, v1.2s, %6.2s   \n" // o1 = o1 >> 2

                        "st1     {v0.2s}, [%1], #8     \n"
                        "st1     {v1.2s}, [%2], #8     \n"
                        : "=r"(out_tile), // %0
                        "=r"(outRow0),  // %1
                        "=r"(outRow1)   // %2
                        : "0"(out_tile),
                        "1"(outRow0),
                        "2"(outRow1),
                        "w"(_shift) // %6
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
                    asm volatile(
                        "pld        [%0, #512]      \n"
                        "vldm        %0!, {d0-d7}   \n"

                        "vaddq.s32    q0, q0, q1    \n" // s0 = s0 + s1 + s2;
                        "vsubq.s32    q1, q1, q2    \n"
                        "vaddq.s32    q0, q0, q2    \n" // s1 = s1 - s2 + s3;
                        "vaddq.s32    q1, q1, q3    \n"

                        "vtrn.s32    q0, q1         \n"

                        "vadd.s32    d8, d0, d2     \n" // o0 = d0 + d1 + d2;
                        "vsub.s32    d9, d2, d1     \n"
                        "vadd.s32    d8, d8, d1     \n" // o1 = d1 - d2 + d3;
                        "vadd.s32    d9, d9, d3     \n"

                        "vshl.s32    d8, d8, %P6    \n" // o0 = o0 >> 2
                        "vshl.s32    d9, d9, %P6    \n" // o1 = o1 >> 2

                        "vst1.s32    {d8}, [%1]!    \n"
                        "vst1.s32    {d9}, [%2]!    \n"
                        : "=r"(out_tile), // %0
                        "=r"(outRow0),  // %1
                        "=r"(outRow1)   // %2
                        : "0"(out_tile),
                        "1"(outRow0),
                        "2"(outRow1),
                        "w"(_shift) // %6
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
#endif // __aarch64__
#else
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

                    out_tile += 16;

                    outRow0 += 2;
                    outRow1 += 2;
#endif // __ARM_NEON
                }

                outRow0 += outw;
                outRow1 += outw;
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_int8_neon(const Mat& kernel, std::vector<Mat>& kernel_tm2, int inch, int outch)
{
    Mat kernel_tm(6 * 6, inch, outch, 2ul);

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
        {0, 0, 6}
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

    for (int r = 0; r < 9; r++)
    {
        Mat kernel_tm_test(4 * 8, inch, outch / 8 + (outch % 8) / 4 + outch % 4, 2u);

        int p = 0;
        for (; p + 7 < outch; p += 8)
        {
            const short* kernel0 = (const short*)kernel_tm.channel(p);
            const short* kernel1 = (const short*)kernel_tm.channel(p + 1);
            const short* kernel2 = (const short*)kernel_tm.channel(p + 2);
            const short* kernel3 = (const short*)kernel_tm.channel(p + 3);
            const short* kernel4 = (const short*)kernel_tm.channel(p + 4);
            const short* kernel5 = (const short*)kernel_tm.channel(p + 5);
            const short* kernel6 = (const short*)kernel_tm.channel(p + 6);
            const short* kernel7 = (const short*)kernel_tm.channel(p + 7);

            short* ktmp = kernel_tm_test.channel(p / 8);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp[16] = kernel4[r * 4 + 0];
                ktmp[17] = kernel4[r * 4 + 1];
                ktmp[18] = kernel4[r * 4 + 2];
                ktmp[19] = kernel4[r * 4 + 3];

                ktmp[20] = kernel5[r * 4 + 0];
                ktmp[21] = kernel5[r * 4 + 1];
                ktmp[22] = kernel5[r * 4 + 2];
                ktmp[23] = kernel5[r * 4 + 3];

                ktmp[24] = kernel6[r * 4 + 0];
                ktmp[25] = kernel6[r * 4 + 1];
                ktmp[26] = kernel6[r * 4 + 2];
                ktmp[27] = kernel6[r * 4 + 3];

                ktmp[28] = kernel7[r * 4 + 0];
                ktmp[29] = kernel7[r * 4 + 1];
                ktmp[30] = kernel7[r * 4 + 2];
                ktmp[31] = kernel7[r * 4 + 3];

                ktmp += 32;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
                kernel4 += 36;
                kernel5 += 36;
                kernel6 += 36;
                kernel7 += 36;
            }
        }

        for (; p + 3 < outch; p += 4)
        {
            const short* kernel0 = (const short*)kernel_tm.channel(p);
            const short* kernel1 = (const short*)kernel_tm.channel(p + 1);
            const short* kernel2 = (const short*)kernel_tm.channel(p + 2);
            const short* kernel3 = (const short*)kernel_tm.channel(p + 3);

            short* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp[4] = kernel1[r * 4 + 0];
                ktmp[5] = kernel1[r * 4 + 1];
                ktmp[6] = kernel1[r * 4 + 2];
                ktmp[7] = kernel1[r * 4 + 3];

                ktmp[8] = kernel2[r * 4 + 0];
                ktmp[9] = kernel2[r * 4 + 1];
                ktmp[10] = kernel2[r * 4 + 2];
                ktmp[11] = kernel2[r * 4 + 3];

                ktmp[12] = kernel3[r * 4 + 0];
                ktmp[13] = kernel3[r * 4 + 1];
                ktmp[14] = kernel3[r * 4 + 2];
                ktmp[15] = kernel3[r * 4 + 3];

                ktmp += 16;
                kernel0 += 36;
                kernel1 += 36;
                kernel2 += 36;
                kernel3 += 36;
            }
        }

        for (; p < outch; p++)
        {
            const short* kernel0 = (const short*)kernel_tm.channel(p);

            short* ktmp = kernel_tm_test.channel(p / 8 + (p % 8) / 4 + p % 4);

            for (int q = 0; q < inch; q++)
            {
                ktmp[0] = kernel0[r * 4 + 0];
                ktmp[1] = kernel0[r * 4 + 1];
                ktmp[2] = kernel0[r * 4 + 2];
                ktmp[3] = kernel0[r * 4 + 3];

                ktmp += 4;
                kernel0 += 36;
            }
        }
        kernel_tm2.push_back(kernel_tm_test);
    }
}

static void conv3x3s1_winograd43_int8_neon(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat>& kernel_tm_test, const Option& opt)
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

        bottom_blob_tm.create(4, inch, tiles * 9, 2u, opt.workspace_allocator);

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
                    short* out_tm0 = bottom_blob_tm.channel(tiles * 0 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm1 = bottom_blob_tm.channel(tiles * 1 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm2 = bottom_blob_tm.channel(tiles * 2 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm3 = bottom_blob_tm.channel(tiles * 3 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm4 = bottom_blob_tm.channel(tiles * 4 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm5 = bottom_blob_tm.channel(tiles * 5 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm6 = bottom_blob_tm.channel(tiles * 6 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm7 = bottom_blob_tm.channel(tiles * 7 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm8 = bottom_blob_tm.channel(tiles * 8 + j * nRowBlocks + i).row<short>(q);
#if __ARM_NEON
                    int8x8_t _d0, _d1, _d2, _d3, _d4, _d5;
                    int16x8_t _w0, _w1, _w2, _w3, _w4, _w5;
                    int16x8_t _t0, _t1, _t2, _t3, _t4, _t5;
                    int16x8_t _n0, _n1, _n2, _n3, _n4, _n5;
                    // load
                    _d0 = vld1_s8(r0);
                    _d1 = vld1_s8(r1);
                    _d2 = vld1_s8(r2);
                    _d3 = vld1_s8(r3);
                    _d4 = vld1_s8(r4);
                    _d5 = vld1_s8(r5);

                    int8x8_t _1_n = vdup_n_s8(-1);
                    int8x8_t _2_p = vdup_n_s8(2);
                    int8x8_t _2_n = vdup_n_s8(-2);
                    int8x8_t _4_p = vdup_n_s8(4);
                    int8x8_t _4_n = vdup_n_s8(-4);
                    int8x8_t _5_n = vdup_n_s8(-5);

                    int16x8_t _1_n_s16 = vdupq_n_s16(-1);
                    int16x8_t _2_p_s16 = vdupq_n_s16(2);
                    int16x8_t _2_n_s16 = vdupq_n_s16(-2);
                    int16x8_t _4_p_s16 = vdupq_n_s16(4);
                    int16x8_t _4_n_s16 = vdupq_n_s16(-4);
                    int16x8_t _5_n_s16 = vdupq_n_s16(-5);
                    // w = B_t * d
                    _w0 = vmull_s8(_d0, _4_p);
                    _w0 = vmlal_s8(_w0, _d2, _5_n);
                    _w0 = vaddw_s8(_w0, _d4);

                    _w1 = vmull_s8(_d1, _4_n);
                    _w1 = vmlal_s8(_w1, _d2, _4_n);
                    _w1 = vaddw_s8(_w1, _d3);
                    _w1 = vaddw_s8(_w1, _d4);

                    _w2 = vmull_s8(_d1, _4_p);
                    _w2 = vmlal_s8(_w2, _d2, _4_n);
                    _w2 = vmlal_s8(_w2, _d3, _1_n);
                    _w2 = vaddw_s8(_w2, _d4);

                    _w3 = vmull_s8(_d1, _2_n);
                    _w3 = vmlal_s8(_w3, _d2, _1_n);
                    _w3 = vmlal_s8(_w3, _d3, _2_p);
                    _w3 = vaddw_s8(_w3, _d4);

                    _w4 = vmull_s8(_d1, _2_p);
                    _w4 = vmlal_s8(_w4, _d2, _1_n);
                    _w4 = vmlal_s8(_w4, _d3, _2_n);
                    _w4 = vaddw_s8(_w4, _d4);

                    _w5 = vmull_s8(_d1, _4_p);
                    _w5 = vmlal_s8(_w5, _d3, _5_n);
                    _w5 = vaddw_s8(_w5, _d5);
                    // transpose d to d_t
                    {
                        _t0[0] = _w0[0];
                        _t1[0] = _w0[1];
                        _t2[0] = _w0[2];
                        _t3[0] = _w0[3];
                        _t4[0] = _w0[4];
                        _t5[0] = _w0[5];
                        _t0[1] = _w1[0];
                        _t1[1] = _w1[1];
                        _t2[1] = _w1[2];
                        _t3[1] = _w1[3];
                        _t4[1] = _w1[4];
                        _t5[1] = _w1[5];
                        _t0[2] = _w2[0];
                        _t1[2] = _w2[1];
                        _t2[2] = _w2[2];
                        _t3[2] = _w2[3];
                        _t4[2] = _w2[4];
                        _t5[2] = _w2[5];
                        _t0[3] = _w3[0];
                        _t1[3] = _w3[1];
                        _t2[3] = _w3[2];
                        _t3[3] = _w3[3];
                        _t4[3] = _w3[4];
                        _t5[3] = _w3[5];
                        _t0[4] = _w4[0];
                        _t1[4] = _w4[1];
                        _t2[4] = _w4[2];
                        _t3[4] = _w4[3];
                        _t4[4] = _w4[4];
                        _t5[4] = _w4[5];
                        _t0[5] = _w5[0];
                        _t1[5] = _w5[1];
                        _t2[5] = _w5[2];
                        _t3[5] = _w5[3];
                        _t4[5] = _w5[4];
                        _t5[5] = _w5[5];
                    }
                    // d = B_t * d_t
                    _n0 = vmulq_s16(_t0, _4_p_s16);
                    _n0 = vmlaq_s16(_n0, _t2, _5_n_s16);
                    _n0 = vaddq_s16(_n0, _t4);

                    _n1 = vmulq_s16(_t1, _4_n_s16);
                    _n1 = vmlaq_s16(_n1, _t2, _4_n_s16);
                    _n1 = vaddq_s16(_n1, _t3);
                    _n1 = vaddq_s16(_n1, _t4);

                    _n2 = vmulq_s16(_t1, _4_p_s16);
                    _n2 = vmlaq_s16(_n2, _t2, _4_n_s16);
                    _n2 = vmlaq_s16(_n2, _t3, _1_n_s16);
                    _n2 = vaddq_s16(_n2, _t4);

                    _n3 = vmulq_s16(_t1, _2_n_s16);
                    _n3 = vmlaq_s16(_n3, _t2, _1_n_s16);
                    _n3 = vmlaq_s16(_n3, _t3, _2_p_s16);
                    _n3 = vaddq_s16(_n3, _t4);

                    _n4 = vmulq_s16(_t1, _2_p_s16);
                    _n4 = vmlaq_s16(_n4, _t2, _1_n_s16);
                    _n4 = vmlaq_s16(_n4, _t3, _2_n_s16);
                    _n4 = vaddq_s16(_n4, _t4);

                    _n5 = vmulq_s16(_t1, _4_p_s16);
                    _n5 = vmlaq_s16(_n5, _t3, _5_n_s16);
                    _n5 = vaddq_s16(_n5, _t5);
                    // save to out_tm
                    out_tm0[0] = _n0[0];
                    out_tm0[1] = _n0[1];
                    out_tm0[2] = _n0[2];
                    out_tm0[3] = _n0[3];
                    out_tm1[0] = _n0[4];
                    out_tm1[1] = _n0[5];
                    out_tm1[2] = _n1[0];
                    out_tm1[3] = _n1[1];
                    out_tm2[0] = _n1[2];
                    out_tm2[1] = _n1[3];
                    out_tm2[2] = _n1[4];
                    out_tm2[3] = _n1[5];

                    out_tm3[0] = _n2[0];
                    out_tm3[1] = _n2[1];
                    out_tm3[2] = _n2[2];
                    out_tm3[3] = _n2[3];
                    out_tm4[0] = _n2[4];
                    out_tm4[1] = _n2[5];
                    out_tm4[2] = _n3[0];
                    out_tm4[3] = _n3[1];
                    out_tm5[0] = _n3[2];
                    out_tm5[1] = _n3[3];
                    out_tm5[2] = _n3[4];
                    out_tm5[3] = _n3[5];

                    out_tm6[0] = _n4[0];
                    out_tm6[1] = _n4[1];
                    out_tm6[2] = _n4[2];
                    out_tm6[3] = _n4[3];
                    out_tm7[0] = _n4[4];
                    out_tm7[1] = _n4[5];
                    out_tm7[2] = _n5[0];
                    out_tm7[3] = _n5[1];
                    out_tm8[0] = _n5[2];
                    out_tm8[1] = _n5[3];
                    out_tm8[2] = _n5[4];
                    out_tm8[3] = _n5[5];
#else
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
                    {
                        out_tm0[0] = d0[0];
                        out_tm0[1] = d0[1];
                        out_tm0[2] = d0[2];
                        out_tm0[3] = d0[3];
                        out_tm1[0] = d0[4];
                        out_tm1[1] = d0[5];
                        out_tm1[2] = d1[0];
                        out_tm1[3] = d1[1];
                        out_tm2[0] = d1[2];
                        out_tm2[1] = d1[3];
                        out_tm2[2] = d1[4];
                        out_tm2[3] = d1[5];

                        out_tm3[0] = d2[0];
                        out_tm3[1] = d2[1];
                        out_tm3[2] = d2[2];
                        out_tm3[3] = d2[3];
                        out_tm4[0] = d2[4];
                        out_tm4[1] = d2[5];
                        out_tm4[2] = d3[0];
                        out_tm4[3] = d3[1];
                        out_tm5[0] = d3[2];
                        out_tm5[1] = d3[3];
                        out_tm5[2] = d3[4];
                        out_tm5[3] = d3[5];

                        out_tm6[0] = d4[0];
                        out_tm6[1] = d4[1];
                        out_tm6[2] = d4[2];
                        out_tm6[3] = d4[3];
                        out_tm7[0] = d4[4];
                        out_tm7[1] = d4[5];
                        out_tm7[2] = d5[0];
                        out_tm7[3] = d5[1];
                        out_tm8[0] = d5[2];
                        out_tm8[1] = d5[3];
                        out_tm8[2] = d5[4];
                        out_tm8[3] = d5[5];
                    }
#endif // __ARM_NEON
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
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
        for (int r = 0; r < 9; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);
                int* output4_tm = top_blob_tm.channel(p + 4);
                int* output5_tm = top_blob_tm.channel(p + 5);
                int* output6_tm = top_blob_tm.channel(p + 6);
                int* output7_tm = top_blob_tm.channel(p + 7);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "eor    v4.16b, v4.16b, v4.16b    \n"
                        "eor    v5.16b, v5.16b, v5.16b    \n"
                        "eor    v6.16b, v6.16b, v6.16b    \n"
                        "eor    v7.16b, v7.16b, v7.16b    \n"
                        "mov    w4, %w20                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%9, #128]    \n" // _r0 = vld1_s16(r0);
                        "ld1     {v8.4h}, [%8]            \n"
                        "ld1     {v9.4h, v10.4h}, [%9]    \n" // _k01 = vld1q_s16(kptr);
                        "add     %9, %9, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%9]   \n" // _k23 = vld1q_s16(kptr+8);
                        "add     %9, %9, #16              \n"
                        "ld1     {v13.4h, v14.4h}, [%9]   \n" // _k45 = vld1q_s16(kptr+16);
                        "add     %9, %9, #16              \n"
                        "ld1     {v15.4h, v16.4h}, [%9]   \n" // _k67 = vld1q_s16(kptr+24);
                        "add     %8, %8, #8               \n"
                        "add     %9, %9, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)
                        "smlal   v4.4s, v8.4h, v13.4h     \n" // sum4 += (a00-a03) * (k40-k43)
                        "smlal   v5.4s, v8.4h, v14.4h     \n" // sum5 += (a00-a03) * (k50-k53)
                        "smlal   v6.4s, v8.4h, v15.4h     \n" // sum6 += (a00-a03) * (k60-k63)
                        "smlal   v7.4s, v8.4h, v16.4h     \n" // sum7 += (a00-a03) * (k70-k73)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //
                        "st1     {v4.4s}, [%4]            \n" //
                        "st1     {v5.4s}, [%5]            \n" //
                        "st1     {v6.4s}, [%6]            \n" //
                        "st1     {v7.4s}, [%7]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "vmov.s32    q4, #0           \n"
                        "vmov.s32    q5, #0           \n"
                        "vmov.s32    q6, #0           \n"
                        "vmov.s32    q7, #0           \n"
                        "mov         r4, %20          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%8]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%9]  \n" // _k01 = vld1q_s16(kptr);
                        "add         %9, #16          \n"
                        "vld1.s16    {d20-d21}, [%9]  \n" // _k23 = vld1q_s16(kptr+8);
                        "add         %9, #16          \n"
                        "vld1.s16    {d22-d23}, [%9]  \n" // _k45 = vld1q_s16(kptr+16);
                        "add         %9, #16          \n"
                        "vld1.s16    {d24-d25}, [%9]  \n" // _k67 = vld1q_s16(kptr+24);
                        "add         %9, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)
                        "vmlal.s16   q4, d16, d22     \n" // sum4 += (a00-a03) * (k40-k43)
                        "vmlal.s16   q5, d16, d23     \n" // sum5 += (a00-a03) * (k50-k53)
                        "vmlal.s16   q6, d16, d24     \n" // sum6 += (a00-a03) * (k60-k63)
                        "vmlal.s16   q7, d16, d25     \n" // sum7 += (a00-a03) * (k70-k73)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"
                        "vst1.s32    {d8-d9}, [%4]    \n"
                        "vst1.s32    {d10-d11}, [%5]  \n"
                        "vst1.s32    {d12-d13}, [%6]  \n"
                        "vst1.s32    {d14-d15}, [%7]  \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};
                    int sum4[4] = {0};
                    int sum5[4] = {0};
                    int sum6[4] = {0};
                    int sum7[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                            sum4[n] += (int)r0[n] * kptr[n + 16];
                            sum5[n] += (int)r0[n] * kptr[n + 20];
                            sum6[n] += (int)r0[n] * kptr[n + 24];
                            sum7[n] += (int)r0[n] * kptr[n + 28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                    output4_tm += 36;
                    output5_tm += 36;
                    output6_tm += 36;
                    output7_tm += 36;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "mov    w4, %w12                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%5, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v8.4h}, [%4]            \n"
                        "ld1     {v9.4h, v10.4h}, [%5]    \n" // _k01 = vld1q_s16(kptr);
                        "add     %5, %5, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%5]   \n" // _k23 = vld1q_s16(kptr+8);
                        "add     %4, %4, #8               \n"
                        "add     %5, %5, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "mov         r4, %12          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%4]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%5]  \n" // _k01 = vld1q_s16(kptr);
                        "add         %5, #16          \n"
                        "vld1.s16    {d20-d21}, [%5]  \n" // _k23 = vld1q_s16(kptr+8);
                        "add         %5, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                }
            }

            remain_outch_start += nn_outch << 2;

            for (int p = remain_outch_start; p < outch; p++)
            {
                int* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4 + p % 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "mov    w4, %w6                   \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "ld1     {v8.4h}, [%1]            \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v9.4h}, [%2]            \n" // _k0 = vld1q_s16(kptr);
                        "add     %1, %1, #8               \n"
                        "add     %2, %2, #8               \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "mov         r4, %6           \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%1]      \n" // _r0 = vld1_s16(r0);  // input inch0
                        "add         %1, #8           \n"
                        "vld1.s16    {d18}, [%2]      \n" // _k0 = vld1q_s16(kptr);
                        "add         %2, #8           \n"
                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "r4", "q0", "q8", "q9");
#endif // __aarch64__
#else  // __ARM_NEON
                    int sum0[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                }
            }

            // for (int p=0; p<outch; p++)
            // {
            //     Mat out0_tm = top_blob_tm.channel(p);
            //     const Mat kernel0_tm = kernel_tm.channel(p);

            //     for (int i=0; i<tiles; i++)
            //     {
            //         int* output0_tm = out0_tm.row<int>(i);

            //         int sum0[36] = {0};

            //         for (int q=0; q<inch; q++)
            //         {
            //             const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
            //             const short* k0 = kernel0_tm.row<short>(q);

            //             for (int n=0; n<36; n++)
            //             {
            //                 sum0[n] += (int)r0[n] * k0[n];
            //             }
            //         }

            //         for (int n=0; n<36; n++)
            //         {
            //             output0_tm[n] = sum0[n];
            //         }
            //     }
            // }
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
            int* out_tile = top_blob_tm.channel(p);
            int* outRow0 = top_blob_bordered.channel(p);
            int* outRow1 = outRow0 + outw;
            int* outRow2 = outRow0 + outw * 2;
            int* outRow3 = outRow0 + outw * 3;

            for (int j = 0; j < nColBlocks; j++)
            {
                for (int i = 0; i < nRowBlocks; i++)
                {
#if __ARM_NEON
                    int32x4_t _s0, _s1, _s2, _s3, _s4, _s5;
                    int32x2_t _s0n, _s1n, _s2n, _s3n, _s4n, _s5n;
                    int32x4_t _w0, _w3;
                    int32x2_t _w0n, _w3n;
                    int32x4_t _d0, _d1, _d2, _d3, _d4, _d5;
                    int32x4_t _o0, _o1, _o2, _o3;
                    // load
                    _s0 = vld1q_s32(out_tile);
                    _s0n = vld1_s32(out_tile + 4);
                    _s1 = vld1q_s32(out_tile + 6);
                    _s1n = vld1_s32(out_tile + 10);
                    _s2 = vld1q_s32(out_tile + 12);
                    _s2n = vld1_s32(out_tile + 16);
                    _s3 = vld1q_s32(out_tile + 18);
                    _s3n = vld1_s32(out_tile + 22);
                    _s4 = vld1q_s32(out_tile + 24);
                    _s4n = vld1_s32(out_tile + 28);
                    _s5 = vld1q_s32(out_tile + 30);
                    _s5n = vld1_s32(out_tile + 34);
                    // w = A_T * W
                    int32x2_t _tp0 = {1, 4};
                    int32x2_t _tp1 = {2, 8};

                    // 4*s5[n]
                    int32x4_t _s5x4 = vshlq_n_s32(_s5, 2);
                    int32x2_t _s5x4n = vshl_n_s32(_s5n, 2);

                    int32x4_t _t1p2 = vaddq_s32(_s1, _s2);
                    int32x2_t _t1p2n = vadd_s32(_s1n, _s2n);
                    int32x4_t _t3p4 = vaddq_s32(_s3, _s4);
                    int32x2_t _t3p4n = vadd_s32(_s3n, _s4n);
                    int32x4_t _t1s2 = vsubq_s32(_s1, _s2);
                    int32x2_t _t1s2n = vsub_s32(_s1n, _s2n);
                    int32x4_t _t3s4 = vsubq_s32(_s3, _s4);
                    int32x2_t _t3s4n = vsub_s32(_s3n, _s4n);

                    _w0 = vaddq_s32(_s0, _t1p2);
                    _w0n = vadd_s32(_s0n, _t1p2n);
                    _w0 = vaddq_s32(_w0, _t3p4);
                    _w0n = vadd_s32(_w0n, _t3p4n);
                    _w0n = vmul_s32(_w0n, _tp0);

                    // _w2,_w2n
                    _t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);
                    _t1p2n = vmla_lane_s32(_t1p2n, _t3p4n, _tp0, 1);
                    _t1p2n = vmul_s32(_t1p2n, _tp0);

                    _w3 = vaddq_s32(_s5x4, _t1s2);
                    _w3n = vadd_s32(_s5x4n, _t1s2n);
                    _w3 = vmlaq_lane_s32(_w3, _t3s4, _tp1, 1);
                    _w3n = vmla_lane_s32(_w3n, _t3s4n, _tp1, 1);
                    _w3n = vmul_s32(_w3n, _tp0);

                    // _w1, _w1n
                    _t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);
                    _t1s2n = vmla_lane_s32(_t1s2n, _t3s4n, _tp1, 0);
                    _t1s2n = vmul_s32(_t1s2n, _tp0);

                    int32x4_t _w02n = vcombine_s32(_w0n, _t1p2n);
                    int32x4_t _w13n = vcombine_s32(_t1s2n, _w3n);

                    // transpose w to w_t
#if __aarch64__
                    int32x4_t _wt0 = vtrn1q_s32(_w0, _t1s2);
                    int32x4_t _wt1 = vtrn2q_s32(_w0, _t1s2);
                    int32x4_t _wt2 = vtrn1q_s32(_t1p2, _w3);
                    int32x4_t _wt3 = vtrn2q_s32(_t1p2, _w3);
                    int64x2_t _dt0 = vtrn1q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
                    int64x2_t _dt2 = vtrn2q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
                    int64x2_t _dt1 = vtrn1q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
                    int64x2_t _dt3 = vtrn2q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
                    _d0 = vreinterpretq_s32_s64(_dt0);
                    _d1 = vreinterpretq_s32_s64(_dt1);
                    _d2 = vreinterpretq_s32_s64(_dt2);
                    _d3 = vreinterpretq_s32_s64(_dt3);
                    _d4 = vtrn1q_s32(_w02n, _w13n);
                    _d5 = vtrn2q_s32(_w02n, _w13n);
#else
                    asm volatile(
                        "vtrn.32    %q[_w0], %q[_w1]        \n"
                        "vtrn.32    %q[_w2], %q[_w3]        \n"
                        "vswp       %f[_w0], %e[_w2]        \n"
                        "vswp       %f[_w1], %e[_w3]        \n"
                        "vtrn.32    %q[_w02n], %q[_w13n]    \n"
                        : [_w0] "+w"(_w0),
                        [_w1] "+w"(_t1s2),
                        [_w2] "+w"(_t1p2),
                        [_w3] "+w"(_w3),
                        [_w02n] "+w"(_w02n),
                        [_w13n] "+w"(_w13n)
                        :
                        : "cc", "memory");
                    _d0 = _w0;
                    _d1 = _t1s2;
                    _d2 = _t1p2;
                    _d3 = _w3;
                    _d4 = _w02n;
                    _d5 = _w13n;
#endif
                    // Y = A_T * w_t
                    _t1p2 = vaddq_s32(_d1, _d2);
                    _t3p4 = vaddq_s32(_d3, _d4);
                    _t1s2 = vsubq_s32(_d1, _d2);
                    _t3s4 = vsubq_s32(_d3, _d4);

                    _o0 = vaddq_s32(_d0, _t1p2);
                    _o0 = vaddq_s32(_o0, _t3p4);

                    // _o2
                    _t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);

                    _o3 = vaddq_s32(_d5, _t1s2);
                    _o3 = vmlaq_lane_s32(_o3, _t3s4, _tp1, 1);

                    // _o1
                    _t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);

                    // save to top blob tm
                    float32x4_t _ot0 = vcvtq_f32_s32(_o0);
                    float32x4_t _ot1 = vcvtq_f32_s32(_t1s2);
                    float32x4_t _ot2 = vcvtq_f32_s32(_t1p2);
                    float32x4_t _ot3 = vcvtq_f32_s32(_o3);

                    _ot0 = vmulq_n_f32(_ot0, 0.0017361112);
                    _ot1 = vmulq_n_f32(_ot1, 0.0017361112);
                    _ot2 = vmulq_n_f32(_ot2, 0.0017361112);
                    _ot3 = vmulq_n_f32(_ot3, 0.0017361112);

                    _o0 = vcvtq_s32_f32(_ot0);
                    _o1 = vcvtq_s32_f32(_ot1);
                    _o2 = vcvtq_s32_f32(_ot2);
                    _o3 = vcvtq_s32_f32(_ot3);

                    vst1q_s32(outRow0, _o0);
                    vst1q_s32(outRow1, _o1);
                    vst1q_s32(outRow2, _o2);
                    vst1q_s32(outRow3, _o3);
#else
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
                    for (int n = 0; n < 5; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n];
                    }
                    for (int n = 5; n < 6; n++)
                    {
                        w0[n] = 4 * (s0[n] + s1[n] + s2[n] + s3[n] + s4[n]);
                        w1[n] = 4 * (s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n]);
                        w2[n] = 4 * (s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n]);
                        w3[n] = 4 * (s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n]);
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
#endif // __ARM_NEON
                    out_tile += 36;

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }

                outRow0 += outw * 3;
                outRow1 += outw * 3;
                outRow2 += outw * 3;
                outRow3 += outw * 3;
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_dequant_int8_neon(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat>& kernel_tm_test, const Mat& _bias, std::vector<float> scales_dequant, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

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

        bottom_blob_tm.create(4, inch, tiles * 9, 2u, opt.workspace_allocator);

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
                    short* out_tm0 = bottom_blob_tm.channel(tiles * 0 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm1 = bottom_blob_tm.channel(tiles * 1 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm2 = bottom_blob_tm.channel(tiles * 2 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm3 = bottom_blob_tm.channel(tiles * 3 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm4 = bottom_blob_tm.channel(tiles * 4 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm5 = bottom_blob_tm.channel(tiles * 5 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm6 = bottom_blob_tm.channel(tiles * 6 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm7 = bottom_blob_tm.channel(tiles * 7 + j * nRowBlocks + i).row<short>(q);
                    short* out_tm8 = bottom_blob_tm.channel(tiles * 8 + j * nRowBlocks + i).row<short>(q);
#if __ARM_NEON
                    int8x8_t _d0, _d1, _d2, _d3, _d4, _d5;
                    int16x8_t _w0, _w1, _w2, _w3, _w4, _w5;
                    int16x8_t _t0, _t1, _t2, _t3, _t4, _t5;
                    int16x8_t _n0, _n1, _n2, _n3, _n4, _n5;
                    // load
                    _d0 = vld1_s8(r0);
                    _d1 = vld1_s8(r1);
                    _d2 = vld1_s8(r2);
                    _d3 = vld1_s8(r3);
                    _d4 = vld1_s8(r4);
                    _d5 = vld1_s8(r5);

                    int8x8_t _1_n = vdup_n_s8(-1);
                    int8x8_t _2_p = vdup_n_s8(2);
                    int8x8_t _2_n = vdup_n_s8(-2);
                    int8x8_t _4_p = vdup_n_s8(4);
                    int8x8_t _4_n = vdup_n_s8(-4);
                    int8x8_t _5_n = vdup_n_s8(-5);

                    int16x8_t _1_n_s16 = vdupq_n_s16(-1);
                    int16x8_t _2_p_s16 = vdupq_n_s16(2);
                    int16x8_t _2_n_s16 = vdupq_n_s16(-2);
                    int16x8_t _4_p_s16 = vdupq_n_s16(4);
                    int16x8_t _4_n_s16 = vdupq_n_s16(-4);
                    int16x8_t _5_n_s16 = vdupq_n_s16(-5);
                    // w = B_t * d
                    _w0 = vmull_s8(_d0, _4_p);
                    _w0 = vmlal_s8(_w0, _d2, _5_n);
                    _w0 = vaddw_s8(_w0, _d4);

                    _w1 = vmull_s8(_d1, _4_n);
                    _w1 = vmlal_s8(_w1, _d2, _4_n);
                    _w1 = vaddw_s8(_w1, _d3);
                    _w1 = vaddw_s8(_w1, _d4);

                    _w2 = vmull_s8(_d1, _4_p);
                    _w2 = vmlal_s8(_w2, _d2, _4_n);
                    _w2 = vmlal_s8(_w2, _d3, _1_n);
                    _w2 = vaddw_s8(_w2, _d4);

                    _w3 = vmull_s8(_d1, _2_n);
                    _w3 = vmlal_s8(_w3, _d2, _1_n);
                    _w3 = vmlal_s8(_w3, _d3, _2_p);
                    _w3 = vaddw_s8(_w3, _d4);

                    _w4 = vmull_s8(_d1, _2_p);
                    _w4 = vmlal_s8(_w4, _d2, _1_n);
                    _w4 = vmlal_s8(_w4, _d3, _2_n);
                    _w4 = vaddw_s8(_w4, _d4);

                    _w5 = vmull_s8(_d1, _4_p);
                    _w5 = vmlal_s8(_w5, _d3, _5_n);
                    _w5 = vaddw_s8(_w5, _d5);
                    // transpose d to d_t
                    {
                        _t0[0] = _w0[0];
                        _t1[0] = _w0[1];
                        _t2[0] = _w0[2];
                        _t3[0] = _w0[3];
                        _t4[0] = _w0[4];
                        _t5[0] = _w0[5];
                        _t0[1] = _w1[0];
                        _t1[1] = _w1[1];
                        _t2[1] = _w1[2];
                        _t3[1] = _w1[3];
                        _t4[1] = _w1[4];
                        _t5[1] = _w1[5];
                        _t0[2] = _w2[0];
                        _t1[2] = _w2[1];
                        _t2[2] = _w2[2];
                        _t3[2] = _w2[3];
                        _t4[2] = _w2[4];
                        _t5[2] = _w2[5];
                        _t0[3] = _w3[0];
                        _t1[3] = _w3[1];
                        _t2[3] = _w3[2];
                        _t3[3] = _w3[3];
                        _t4[3] = _w3[4];
                        _t5[3] = _w3[5];
                        _t0[4] = _w4[0];
                        _t1[4] = _w4[1];
                        _t2[4] = _w4[2];
                        _t3[4] = _w4[3];
                        _t4[4] = _w4[4];
                        _t5[4] = _w4[5];
                        _t0[5] = _w5[0];
                        _t1[5] = _w5[1];
                        _t2[5] = _w5[2];
                        _t3[5] = _w5[3];
                        _t4[5] = _w5[4];
                        _t5[5] = _w5[5];
                    }
                    // d = B_t * d_t
                    _n0 = vmulq_s16(_t0, _4_p_s16);
                    _n0 = vmlaq_s16(_n0, _t2, _5_n_s16);
                    _n0 = vaddq_s16(_n0, _t4);

                    _n1 = vmulq_s16(_t1, _4_n_s16);
                    _n1 = vmlaq_s16(_n1, _t2, _4_n_s16);
                    _n1 = vaddq_s16(_n1, _t3);
                    _n1 = vaddq_s16(_n1, _t4);

                    _n2 = vmulq_s16(_t1, _4_p_s16);
                    _n2 = vmlaq_s16(_n2, _t2, _4_n_s16);
                    _n2 = vmlaq_s16(_n2, _t3, _1_n_s16);
                    _n2 = vaddq_s16(_n2, _t4);

                    _n3 = vmulq_s16(_t1, _2_n_s16);
                    _n3 = vmlaq_s16(_n3, _t2, _1_n_s16);
                    _n3 = vmlaq_s16(_n3, _t3, _2_p_s16);
                    _n3 = vaddq_s16(_n3, _t4);

                    _n4 = vmulq_s16(_t1, _2_p_s16);
                    _n4 = vmlaq_s16(_n4, _t2, _1_n_s16);
                    _n4 = vmlaq_s16(_n4, _t3, _2_n_s16);
                    _n4 = vaddq_s16(_n4, _t4);

                    _n5 = vmulq_s16(_t1, _4_p_s16);
                    _n5 = vmlaq_s16(_n5, _t3, _5_n_s16);
                    _n5 = vaddq_s16(_n5, _t5);
                    // save to out_tm
                    out_tm0[0] = _n0[0];
                    out_tm0[1] = _n0[1];
                    out_tm0[2] = _n0[2];
                    out_tm0[3] = _n0[3];
                    out_tm1[0] = _n0[4];
                    out_tm1[1] = _n0[5];
                    out_tm1[2] = _n1[0];
                    out_tm1[3] = _n1[1];
                    out_tm2[0] = _n1[2];
                    out_tm2[1] = _n1[3];
                    out_tm2[2] = _n1[4];
                    out_tm2[3] = _n1[5];

                    out_tm3[0] = _n2[0];
                    out_tm3[1] = _n2[1];
                    out_tm3[2] = _n2[2];
                    out_tm3[3] = _n2[3];
                    out_tm4[0] = _n2[4];
                    out_tm4[1] = _n2[5];
                    out_tm4[2] = _n3[0];
                    out_tm4[3] = _n3[1];
                    out_tm5[0] = _n3[2];
                    out_tm5[1] = _n3[3];
                    out_tm5[2] = _n3[4];
                    out_tm5[3] = _n3[5];

                    out_tm6[0] = _n4[0];
                    out_tm6[1] = _n4[1];
                    out_tm6[2] = _n4[2];
                    out_tm6[3] = _n4[3];
                    out_tm7[0] = _n4[4];
                    out_tm7[1] = _n4[5];
                    out_tm7[2] = _n5[0];
                    out_tm7[3] = _n5[1];
                    out_tm8[0] = _n5[2];
                    out_tm8[1] = _n5[3];
                    out_tm8[2] = _n5[4];
                    out_tm8[3] = _n5[5];
#else
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
                    {
                        out_tm0[0] = d0[0];
                        out_tm0[1] = d0[1];
                        out_tm0[2] = d0[2];
                        out_tm0[3] = d0[3];
                        out_tm1[0] = d0[4];
                        out_tm1[1] = d0[5];
                        out_tm1[2] = d1[0];
                        out_tm1[3] = d1[1];
                        out_tm2[0] = d1[2];
                        out_tm2[1] = d1[3];
                        out_tm2[2] = d1[4];
                        out_tm2[3] = d1[5];

                        out_tm3[0] = d2[0];
                        out_tm3[1] = d2[1];
                        out_tm3[2] = d2[2];
                        out_tm3[3] = d2[3];
                        out_tm4[0] = d2[4];
                        out_tm4[1] = d2[5];
                        out_tm4[2] = d3[0];
                        out_tm4[3] = d3[1];
                        out_tm5[0] = d3[2];
                        out_tm5[1] = d3[3];
                        out_tm5[2] = d3[4];
                        out_tm5[3] = d3[5];

                        out_tm6[0] = d4[0];
                        out_tm6[1] = d4[1];
                        out_tm6[2] = d4[2];
                        out_tm6[3] = d4[3];
                        out_tm7[0] = d4[4];
                        out_tm7[1] = d4[5];
                        out_tm7[2] = d5[0];
                        out_tm7[3] = d5[1];
                        out_tm8[0] = d5[2];
                        out_tm8[1] = d5[3];
                        out_tm8[2] = d5[4];
                        out_tm8[3] = d5[5];
                    }
#endif // __ARM_NEON
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
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
        for (int r = 0; r < 9; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);
                int* output4_tm = top_blob_tm.channel(p + 4);
                int* output5_tm = top_blob_tm.channel(p + 5);
                int* output6_tm = top_blob_tm.channel(p + 6);
                int* output7_tm = top_blob_tm.channel(p + 7);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;
                output4_tm = output4_tm + r * 4;
                output5_tm = output5_tm + r * 4;
                output6_tm = output6_tm + r * 4;
                output7_tm = output7_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "eor    v4.16b, v4.16b, v4.16b    \n"
                        "eor    v5.16b, v5.16b, v5.16b    \n"
                        "eor    v6.16b, v6.16b, v6.16b    \n"
                        "eor    v7.16b, v7.16b, v7.16b    \n"
                        "mov    w4, %w20                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%9, #128]    \n" // _r0 = vld1_s16(r0);
                        "ld1     {v8.4h}, [%8]            \n"
                        "ld1     {v9.4h, v10.4h}, [%9]    \n" // _k01 = vld1q_s16(kptr);
                        "add     %9, %9, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%9]   \n" // _k23 = vld1q_s16(kptr+8);
                        "add     %9, %9, #16              \n"
                        "ld1     {v13.4h, v14.4h}, [%9]   \n" // _k45 = vld1q_s16(kptr+16);
                        "add     %9, %9, #16              \n"
                        "ld1     {v15.4h, v16.4h}, [%9]   \n" // _k67 = vld1q_s16(kptr+24);
                        "add     %8, %8, #8               \n"
                        "add     %9, %9, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)
                        "smlal   v4.4s, v8.4h, v13.4h     \n" // sum4 += (a00-a03) * (k40-k43)
                        "smlal   v5.4s, v8.4h, v14.4h     \n" // sum5 += (a00-a03) * (k50-k53)
                        "smlal   v6.4s, v8.4h, v15.4h     \n" // sum6 += (a00-a03) * (k60-k63)
                        "smlal   v7.4s, v8.4h, v16.4h     \n" // sum7 += (a00-a03) * (k70-k73)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //
                        "st1     {v4.4s}, [%4]            \n" //
                        "st1     {v5.4s}, [%5]            \n" //
                        "st1     {v6.4s}, [%6]            \n" //
                        "st1     {v7.4s}, [%7]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "vmov.s32    q4, #0           \n"
                        "vmov.s32    q5, #0           \n"
                        "vmov.s32    q6, #0           \n"
                        "vmov.s32    q7, #0           \n"
                        "mov         r4, %20          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%8]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%9]  \n" // _k01 = vld1q_s16(kptr);
                        "add         %9, #16          \n"
                        "vld1.s16    {d20-d21}, [%9]  \n" // _k23 = vld1q_s16(kptr+8);
                        "add         %9, #16          \n"
                        "vld1.s16    {d22-d23}, [%9]  \n" // _k45 = vld1q_s16(kptr+16);
                        "add         %9, #16          \n"
                        "vld1.s16    {d24-d25}, [%9]  \n" // _k67 = vld1q_s16(kptr+24);
                        "add         %9, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)
                        "vmlal.s16   q4, d16, d22     \n" // sum4 += (a00-a03) * (k40-k43)
                        "vmlal.s16   q5, d16, d23     \n" // sum5 += (a00-a03) * (k50-k53)
                        "vmlal.s16   q6, d16, d24     \n" // sum6 += (a00-a03) * (k60-k63)
                        "vmlal.s16   q7, d16, d25     \n" // sum7 += (a00-a03) * (k70-k73)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"
                        "vst1.s32    {d8-d9}, [%4]    \n"
                        "vst1.s32    {d10-d11}, [%5]  \n"
                        "vst1.s32    {d12-d13}, [%6]  \n"
                        "vst1.s32    {d14-d15}, [%7]  \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(output4_tm), // %4
                        "=r"(output5_tm), // %5
                        "=r"(output6_tm), // %6
                        "=r"(output7_tm), // %7
                        "=r"(r0),         // %8
                        "=r"(kptr)        // %9
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(output4_tm),
                        "5"(output5_tm),
                        "6"(output6_tm),
                        "7"(output7_tm),
                        "8"(r0),
                        "9"(kptr),
                        "r"(inch) // %20
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};
                    int sum4[4] = {0};
                    int sum5[4] = {0};
                    int sum6[4] = {0};
                    int sum7[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                            sum4[n] += (int)r0[n] * kptr[n + 16];
                            sum5[n] += (int)r0[n] * kptr[n + 20];
                            sum6[n] += (int)r0[n] * kptr[n + 24];
                            sum7[n] += (int)r0[n] * kptr[n + 28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                    output4_tm += 36;
                    output5_tm += 36;
                    output6_tm += 36;
                    output7_tm += 36;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p + 1);
                int* output2_tm = top_blob_tm.channel(p + 2);
                int* output3_tm = top_blob_tm.channel(p + 3);

                output0_tm = output0_tm + r * 4;
                output1_tm = output1_tm + r * 4;
                output2_tm = output2_tm + r * 4;
                output3_tm = output3_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "eor    v1.16b, v1.16b, v1.16b    \n"
                        "eor    v2.16b, v2.16b, v2.16b    \n"
                        "eor    v3.16b, v3.16b, v3.16b    \n"
                        "mov    w4, %w12                  \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "prfm    pldl1keep, [%5, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v8.4h}, [%4]            \n"
                        "ld1     {v9.4h, v10.4h}, [%5]    \n" // _k01 = vld1q_s16(kptr);
                        "add     %5, %5, #16              \n"
                        "ld1     {v11.4h, v12.4h}, [%5]   \n" // _k23 = vld1q_s16(kptr+8);
                        "add     %4, %4, #8               \n"
                        "add     %5, %5, #16              \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
                        "smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
                        "smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
                        "smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory
                        "st1     {v1.4s}, [%1]            \n" //
                        "st1     {v2.4s}, [%2]            \n" //
                        "st1     {v3.4s}, [%3]            \n" //

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "mov         r4, %12          \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%4]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%5]  \n" // _k01 = vld1q_s16(kptr);
                        "add         %5, #16          \n"
                        "vld1.s16    {d20-d21}, [%5]  \n" // _k23 = vld1q_s16(kptr+8);
                        "add         %5, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"

                        : "=r"(output0_tm), // %0
                        "=r"(output1_tm), // %1
                        "=r"(output2_tm), // %2
                        "=r"(output3_tm), // %3
                        "=r"(r0),         // %4
                        "=r"(kptr)        // %5
                        : "0"(output0_tm),
                        "1"(output1_tm),
                        "2"(output2_tm),
                        "3"(output3_tm),
                        "4"(r0),
                        "5"(kptr),
                        "r"(inch) // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
#endif // __aarch64__
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n + 4];
                            sum2[n] += (int)r0[n] * kptr[n + 8];
                            sum3[n] += (int)r0[n] * kptr[n + 12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                    output1_tm += 36;
                    output2_tm += 36;
                    output3_tm += 36;
                }
            }

            remain_outch_start += nn_outch << 2;

            for (int p = remain_outch_start; p < outch; p++)
            {
                int* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r * 4;

                for (int i = 0; i < tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p / 8 + (p % 8) / 4 + p % 4);
                    const short* r0 = bottom_blob_tm.channel(tiles * r + i);
#if __ARM_NEON
#if __aarch64__
                    asm volatile(
                        // inch loop
                        "eor    v0.16b, v0.16b, v0.16b    \n"
                        "mov    w4, %w6                   \n"

                        "0:                               \n" // for (int q=0; q<inch; q++)
                        "ld1     {v8.4h}, [%1]            \n" // _r0 = vld1_s16(r0);  // input inch0
                        "ld1     {v9.4h}, [%2]            \n" // _k0 = vld1q_s16(kptr);
                        "add     %1, %1, #8               \n"
                        "add     %2, %2, #8               \n"

                        "subs    w4, w4, #1               \n"

                        "smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)

                        "bne     0b                       \n" // end for

                        "st1     {v0.4s}, [%0]            \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "mov         r4, %6           \n"

                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%1]      \n" // _r0 = vld1_s16(r0);  // input inch0
                        "add         %1, #8           \n"
                        "vld1.s16    {d18}, [%2]      \n" // _k0 = vld1q_s16(kptr);
                        "add         %2, #8           \n"
                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                        "=r"(r0),         // %1
                        "=r"(kptr)        // %2
                        : "0"(output0_tm),
                        "1"(r0),
                        "2"(kptr),
                        "r"(inch) // %6
                        : "cc", "memory", "r4", "q0", "q8", "q9");
#endif // __aarch64__
#else  // __ARM_NEON
                    int sum0[4] = {0};

                    for (int q = 0; q < inch; q++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4;
                        r0 += 4;
                    }

                    for (int n = 0; n < 4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 36;
                }
            }

            // for (int p=0; p<outch; p++)
            // {
            //     Mat out0_tm = top_blob_tm.channel(p);
            //     const Mat kernel0_tm = kernel_tm.channel(p);

            //     for (int i=0; i<tiles; i++)
            //     {
            //         int* output0_tm = out0_tm.row<int>(i);

            //         int sum0[36] = {0};

            //         for (int q=0; q<inch; q++)
            //         {
            //             const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
            //             const short* k0 = kernel0_tm.row<short>(q);

            //             for (int n=0; n<36; n++)
            //             {
            //                 sum0[n] += (int)r0[n] * k0[n];
            //             }
            //         }

            //         for (int n=0; n<36; n++)
            //         {
            //             output0_tm[n] = sum0[n];
            //         }
            //     }
            // }
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
            int* out_tile = top_blob_tm.channel(p);
            float* outRow0 = top_blob_bordered.channel(p);
            float* outRow1 = outRow0 + outw;
            float* outRow2 = outRow0 + outw * 2;
            float* outRow3 = outRow0 + outw * 3;

            const float bias0 = bias ? bias[p] : 0.f;

            const float scale_dequant0 = scales_dequant[p];

            const float scale0 = scale_dequant0 / 576.0;

            for (int j = 0; j < nColBlocks; j++)
            {
                for (int i = 0; i < nRowBlocks; i++)
                {
#if __ARM_NEON
                    int32x4_t _s0, _s1, _s2, _s3, _s4, _s5;
                    int32x2_t _s0n, _s1n, _s2n, _s3n, _s4n, _s5n;
                    int32x4_t _w0, _w3;
                    int32x2_t _w0n, _w3n;
                    int32x4_t _d0, _d1, _d2, _d3, _d4, _d5;
                    int32x4_t _o0, _o3;
                    // load
                    _s0 = vld1q_s32(out_tile);
                    _s0n = vld1_s32(out_tile + 4);
                    _s1 = vld1q_s32(out_tile + 6);
                    _s1n = vld1_s32(out_tile + 10);
                    _s2 = vld1q_s32(out_tile + 12);
                    _s2n = vld1_s32(out_tile + 16);
                    _s3 = vld1q_s32(out_tile + 18);
                    _s3n = vld1_s32(out_tile + 22);
                    _s4 = vld1q_s32(out_tile + 24);
                    _s4n = vld1_s32(out_tile + 28);
                    _s5 = vld1q_s32(out_tile + 30);
                    _s5n = vld1_s32(out_tile + 34);
                    // w = A_T * W
                    int32x2_t _tp0 = {1, 4};
                    int32x2_t _tp1 = {2, 8};

                    // 4*s5[n]
                    int32x4_t _s5x4 = vshlq_n_s32(_s5, 2);
                    int32x2_t _s5x4n = vshl_n_s32(_s5n, 2);

                    int32x4_t _t1p2 = vaddq_s32(_s1, _s2);
                    int32x2_t _t1p2n = vadd_s32(_s1n, _s2n);
                    int32x4_t _t3p4 = vaddq_s32(_s3, _s4);
                    int32x2_t _t3p4n = vadd_s32(_s3n, _s4n);
                    int32x4_t _t1s2 = vsubq_s32(_s1, _s2);
                    int32x2_t _t1s2n = vsub_s32(_s1n, _s2n);
                    int32x4_t _t3s4 = vsubq_s32(_s3, _s4);
                    int32x2_t _t3s4n = vsub_s32(_s3n, _s4n);

                    _w0 = vaddq_s32(_s0, _t1p2);
                    _w0n = vadd_s32(_s0n, _t1p2n);
                    _w0 = vaddq_s32(_w0, _t3p4);
                    _w0n = vadd_s32(_w0n, _t3p4n);
                    _w0n = vmul_s32(_w0n, _tp0);

                    // _w2,_w2n
                    _t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);
                    _t1p2n = vmla_lane_s32(_t1p2n, _t3p4n, _tp0, 1);
                    _t1p2n = vmul_s32(_t1p2n, _tp0);

                    _w3 = vaddq_s32(_s5x4, _t1s2);
                    _w3n = vadd_s32(_s5x4n, _t1s2n);
                    _w3 = vmlaq_lane_s32(_w3, _t3s4, _tp1, 1);
                    _w3n = vmla_lane_s32(_w3n, _t3s4n, _tp1, 1);
                    _w3n = vmul_s32(_w3n, _tp0);

                    // _w1, _w1n
                    _t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);
                    _t1s2n = vmla_lane_s32(_t1s2n, _t3s4n, _tp1, 0);
                    _t1s2n = vmul_s32(_t1s2n, _tp0);

                    int32x4_t _w02n = vcombine_s32(_w0n, _t1p2n);
                    int32x4_t _w13n = vcombine_s32(_t1s2n, _w3n);

                    // transpose w to w_t
#if __aarch64__
                    int32x4_t _wt0 = vtrn1q_s32(_w0, _t1s2);
                    int32x4_t _wt1 = vtrn2q_s32(_w0, _t1s2);
                    int32x4_t _wt2 = vtrn1q_s32(_t1p2, _w3);
                    int32x4_t _wt3 = vtrn2q_s32(_t1p2, _w3);
                    int64x2_t _dt0 = vtrn1q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
                    int64x2_t _dt2 = vtrn2q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
                    int64x2_t _dt1 = vtrn1q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
                    int64x2_t _dt3 = vtrn2q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
                    _d0 = vreinterpretq_s32_s64(_dt0);
                    _d1 = vreinterpretq_s32_s64(_dt1);
                    _d2 = vreinterpretq_s32_s64(_dt2);
                    _d3 = vreinterpretq_s32_s64(_dt3);
                    _d4 = vtrn1q_s32(_w02n, _w13n);
                    _d5 = vtrn2q_s32(_w02n, _w13n);
#else
                    asm volatile(
                        "vtrn.32    %q[_w0], %q[_w1]        \n"
                        "vtrn.32    %q[_w2], %q[_w3]        \n"
                        "vswp       %f[_w0], %e[_w2]        \n"
                        "vswp       %f[_w1], %e[_w3]        \n"
                        "vtrn.32    %q[_w02n], %q[_w13n]    \n"
                        : [_w0] "+w"(_w0),
                        [_w1] "+w"(_t1s2),
                        [_w2] "+w"(_t1p2),
                        [_w3] "+w"(_w3),
                        [_w02n] "+w"(_w02n),
                        [_w13n] "+w"(_w13n)
                        :
                        : "cc", "memory");
                    _d0 = _w0;
                    _d1 = _t1s2;
                    _d2 = _t1p2;
                    _d3 = _w3;
                    _d4 = _w02n;
                    _d5 = _w13n;
#endif
                    // Y = A_T * w_t
                    _t1p2 = vaddq_s32(_d1, _d2);
                    _t3p4 = vaddq_s32(_d3, _d4);
                    _t1s2 = vsubq_s32(_d1, _d2);
                    _t3s4 = vsubq_s32(_d3, _d4);

                    _o0 = vaddq_s32(_d0, _t1p2);
                    _o0 = vaddq_s32(_o0, _t3p4);

                    // _o2
                    _t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);

                    _o3 = vaddq_s32(_d5, _t1s2);
                    _o3 = vmlaq_lane_s32(_o3, _t3s4, _tp1, 1);

                    // _o1
                    _t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);

                    // save to top blob tm
                    float32x4_t _scale0 = vdupq_n_f32(scale0);
                    float32x4_t _out0_f32 = vdupq_n_f32(bias0);
                    float32x4_t _out1_f32 = vdupq_n_f32(bias0);
                    float32x4_t _out2_f32 = vdupq_n_f32(bias0);
                    float32x4_t _out3_f32 = vdupq_n_f32(bias0);

                    _out0_f32 = vmlaq_f32(_out0_f32, vcvtq_f32_s32(_o0), _scale0);
                    _out1_f32 = vmlaq_f32(_out1_f32, vcvtq_f32_s32(_t1s2), _scale0);
                    _out2_f32 = vmlaq_f32(_out2_f32, vcvtq_f32_s32(_t1p2), _scale0);
                    _out3_f32 = vmlaq_f32(_out3_f32, vcvtq_f32_s32(_o3), _scale0);

                    vst1q_f32(outRow0, _out0_f32);
                    vst1q_f32(outRow1, _out1_f32);
                    vst1q_f32(outRow2, _out2_f32);
                    vst1q_f32(outRow3, _out3_f32);
#else
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
                    for (int n = 0; n < 5; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
                        w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
                        w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
                        w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n];
                    }
                    for (int n = 5; n < 6; n++)
                    {
                        w0[n] = 4 * (s0[n] + s1[n] + s2[n] + s3[n] + s4[n]);
                        w1[n] = 4 * (s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n]);
                        w2[n] = 4 * (s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n]);
                        w3[n] = 4 * (s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n]);
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
                        outRow0[n] = (float)o0[n] * scale0 + bias0;
                        outRow1[n] = (float)o1[n] * scale0 + bias0;
                        outRow2[n] = (float)o2[n] * scale0 + bias0;
                        outRow3[n] = (float)o3[n] * scale0 + bias0;
                    }
#endif // __ARM_NEON
                    out_tile += 36;

                    outRow0 += 4;
                    outRow1 += 4;
                    outRow2 += 4;
                    outRow3 += 4;
                }

                outRow0 += outw * 3;
                outRow1 += outw * 3;
                outRow2 += outw * 3;
                outRow3 += outw * 3;
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s2_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8 * 9, inch, outch / 8 + outch % 8, (size_t)1u);

    const signed char* kernel = _kernel;

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        const signed char* k0 = kernel + (p + 0) * inch * 9;
        const signed char* k1 = kernel + (p + 1) * inch * 9;
        const signed char* k2 = kernel + (p + 2) * inch * 9;
        const signed char* k3 = kernel + (p + 3) * inch * 9;
        const signed char* k4 = kernel + (p + 4) * inch * 9;
        const signed char* k5 = kernel + (p + 5) * inch * 9;
        const signed char* k6 = kernel + (p + 6) * inch * 9;
        const signed char* k7 = kernel + (p + 7) * inch * 9;

        signed char* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p < outch; p++)
    {
        const signed char* k0 = kernel + (p + 0) * inch * 9;

        signed char* ktmp = kernel_tm.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s2_packed_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p + 0);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);
        Mat out4 = top_blob.channel(p + 4);
        Mat out5 = top_blob.channel(p + 5);
        Mat out6 = top_blob.channel(p + 6);
        Mat out7 = top_blob.channel(p + 7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        const signed char* ktmp = _kernel.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
#if __aarch64__
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int nn = outw >> 2;
                int remain = outw & 3;
#endif // __aarch64__
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                   \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%9], #16     \n"      //r0-r2
                        "ld2    {v5.8b, v6.8b}, [%9]          \n"

                        "ld1    {v8.4s, v9.4s}, [%1]          \n" //out0
                        "ld1    {v10.4s, v11.4s}, [%2]        \n" //out1
                        "ld1    {v12.4s, v13.4s}, [%3]        \n" //out2
                        "ld1    {v14.4s, v15.4s}, [%4]        \n" //out3
                        "ld1    {v16.4s, v17.4s}, [%5]        \n" //out4
                        "ld1    {v18.4s, v19.4s}, [%6]        \n" //out5
                        "ld1    {v20.4s, v21.4s}, [%7]        \n" //out6
                        "ld1    {v22.4s, v23.4s}, [%8]        \n" //out7

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k00-k70)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k01-k71)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k02-k72)
                        "sshll  v3.8h, v3.8b, #0              \n" // r0
                        "sshll  v4.8h, v4.8b, #0              \n" // r1
                        "sshll  v7.8h, v7.8b, #0              \n" // r2

                        // r0
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r00-r07)*k00
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r00-r07)*k10
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r00-r07)*k20
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r00-r07)*k30
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r00-r07)*k40
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r00-r07)*k50
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r00-r07)*k60
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r00-r07)*k70
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r1
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r10-r17)*k01
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r10-r17)*k11
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r10-r17)*k21
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r10-r17)*k31
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r10-r17)*k41
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r10-r17)*k51
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r10-r17)*k61
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r10-r17)*k71
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r2
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r20-r27)*k02
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r20-r27)*k12
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r20-r27)*k22
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r20-r27)*k32
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r20-r27)*k42
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r20-r27)*k52
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r20-r27)*k62
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r20-r27)*k72
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%10], #16    \n"      //r3-r5
                        "ld2    {v5.8b, v6.8b}, [%10]         \n"

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k03-k73)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k04-k74)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k05-k75)
                        "sshll  v3.8h, v3.8b, #0              \n" // r3
                        "sshll  v4.8h, v4.8b, #0              \n" // r4
                        "sshll  v7.8h, v7.8b, #0              \n" // r5

                        // r3
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r30-r37)*k03
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r30-r37)*k13
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r30-r37)*k23
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r30-r37)*k33
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r30-r37)*k43
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r30-r37)*k53
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r30-r37)*k63
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r30-r37)*k73
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r4
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r40-r47)*k04
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r40-r47)*k14
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r40-r47)*k24
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r40-r47)*k34
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r40-r47)*k44
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r40-r47)*k54
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r40-r47)*k64
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r40-r47)*k74
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r5
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r50-r57)*k05
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r50-r57)*k15
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r50-r57)*k25
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r50-r57)*k35
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r50-r57)*k45
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r50-r57)*k55
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r50-r57)*k65
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r50-r57)*k75
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%11], #16    \n"      //r6-r8
                        "ld2    {v5.8b, v6.8b}, [%11]         \n"

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k06-k76)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k07-k77)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k08-k78)
                        "sshll  v3.8h, v3.8b, #0              \n" // r6
                        "sshll  v4.8h, v4.8b, #0              \n" // r7
                        "sshll  v7.8h, v7.8b, #0              \n" // r8

                        // r6
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r60-r67)*k06
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r60-r67)*k16
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r60-r67)*k26
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r60-r67)*k36
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r60-r67)*k46
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r60-r67)*k56
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r60-r67)*k66
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r60-r67)*k76
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r7
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r70-r77)*k07
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r70-r77)*k17
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r70-r77)*k27
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r70-r77)*k37
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r70-r77)*k47
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r70-r77)*k57
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r70-r77)*k67
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r70-r77)*k77
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r8
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r80-r87)*k08
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r80-r87)*k18
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r80-r87)*k28
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r80-r87)*k38
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r80-r87)*k48
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r80-r87)*k58
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r80-r87)*k68
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r80-r87)*k78
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32     \n"
                        "st1    {v10.4s, v11.4s}, [%2], #32   \n"
                        "st1    {v12.4s, v13.4s}, [%3], #32   \n"
                        "st1    {v14.4s, v15.4s}, [%4], #32   \n"
                        "st1    {v16.4s, v17.4s}, [%5], #32   \n"
                        "st1    {v18.4s, v19.4s}, [%6], #32   \n"
                        "st1    {v20.4s, v21.4s}, [%7], #32   \n"
                        "st1    {v22.4s, v23.4s}, [%8], #32   \n"

                        "subs   %w0, %w0, #1                  \n"
                        "sub    %12, %12, #72                 \n" // reset ktmp

                        "bne    0b                            \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(ktmp)     // %12
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(ktmp)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
#else  // __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.s32   {d16-d17}, [%1]     \n" // out0
                        "pld        [%2, #128]          \n"
                        "vld1.s32   {d18-d19}, [%2]     \n" // out1
                        "pld        [%3, #128]          \n"
                        "vld1.s32   {d20-d21}, [%3]     \n" // out2
                        "pld        [%4, #128]          \n"
                        "vld1.s32   {d22-d23}, [%4]     \n" // out3

                        // r0
                        "pld        [%9, #64]          \n"
                        "vld2.s8    {d8-d9}, [%9]       \n" // d8(a00 a02 a04 a06 a08 a010 a012 a014), d9(a01 a03 a05 a07 a09 a011 a013 a015)
                        "add        %9, #8              \n"
                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k00-k70) d1(k01-k71) d2(k02-k72)

                        "pld        [%5, #128]          \n"
                        "vld1.s32   {d24-d25}, [%5]     \n" // out4
                        "pld        [%6, #128]          \n"
                        "vld1.s32   {d26-d27}, [%6]     \n" // out5

                        "vmovl.s8   q2, d2              \n" // q2(k02-k72)
                        "vmovl.s8   q1, d1              \n" // q1(k01-k71)
                        "vmovl.s8   q0, d0              \n" // q0(k00-k70)
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a02 a04 a06 a08 x x x x)

                        "pld        [%7, #128]          \n"
                        "vld1.s32   {d28-d29}, [%7]     \n" // out6

                        "vmovl.s8   q5, d9              \n" // q5(a01 a03 a05 a07 a09 a011 a013 a015) d11
                        "vmovl.s8   q4, d8              \n" // q4(a00 a02 a04 a06 a08 a010 a012 a014) d9
                        "vmovl.s8   q6, d12             \n" // q6(a02 a04 a06 a08 a010 a012 a014 a016) d13

                        "pld        [%8, #128]          \n"
                        "vld1.s32   {d30-d31}, [%8]     \n" // out7

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a00 a02 a04 a06) * k00
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a00 a02 a04 a06) * k10
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a00 a02 a04 a06) * k20
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a00 a02 a04 a06) * k30
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a00 a02 a04 a06) * k40
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a00 a02 a04 a06) * k50
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a00 a02 a04 a06) * k60
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a00 a02 a04 a06) * k70

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a01-a07) * k01
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a01-a07) * k11
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a01-a07) * k21
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a01-a07) * k31
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a01-a07) * k41
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a01-a07) * k51
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a01-a07) * k61
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a01-a07) * k71

                        "pld        [%10, #64]         \n"
                        "vld2.s8    {d8-d9}, [%10]      \n" // d8(a10 a12 a14 a16 a18 a110 a112 a114), d9(a11 a13 a15 a17 a19 a111 a113 a115)
                        "add        %10, #8             \n"

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a02-a08) * k02
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a02-a08) * k12
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a02-a08) * k22
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a02-a08) * k32

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k03-k73) d1(k04-k74) d2(k05-k75)

                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a02-a08) * k42
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a02-a08) * k52
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a02-a08) * k62
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a02-a08) * k72

                        // r1
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a12 a14 a16 a18 x x x x)

                        "vmovl.s8   q2, d2              \n" // q2(k05-k75)
                        "vmovl.s8   q1, d1              \n" // q1(k04-k74)
                        "vmovl.s8   q0, d0              \n" // q0(k03-k73)
                        "vmovl.s8   q5, d9              \n" // q5(a11-a115)
                        "vmovl.s8   q4, d8              \n" // q4(a10-a114)
                        "vmovl.s8   q6, d12             \n" // q6(a12-a116)

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a10-a16) * k03
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a10-a16) * k13
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a10-a16) * k23
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a10-a16) * k33
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a10-a16) * k43
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a10-a16) * k53
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a10-a16) * k63
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a10-a16) * k73

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a11-a17) * k04
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a11-a17) * k14
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a11-a17) * k24
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a11-a17) * k34
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a11-a17) * k44
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a11-a17) * k54
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a11-a17) * k64
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a11-a17) * k74

                        "pld        [%11, #64]         \n"
                        "vld2.s8    {d8-d9}, [%11]      \n" // d8(a20 a22 a24 a26 a28 a210 a212 a214), d9(a21 a23 a25 a27 a29 a211 a213 a215)
                        "add        %11, #8             \n"

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a12-a18) * k05
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a12-a18) * k15
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a12-a18) * k25
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a12-a18) * k35

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k06-k76) d1(k07-k77) d2(k08-k78)

                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a12-a18) * k45
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a12-a18) * k55
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a12-a18) * k65
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a12-a18) * k75

                        // r2
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a22 a24 a26 a28 x x x x)

                        "vmovl.s8   q2, d2              \n" // q2(k08-k78)
                        "vmovl.s8   q1, d1              \n" // q1(k07-k77)
                        "vmovl.s8   q0, d0              \n" // q0(k06-k76)
                        "vmovl.s8   q5, d9              \n" // q5(a21-a215)
                        "vmovl.s8   q4, d8              \n" // q4(a20-a214)
                        "vmovl.s8   q6, d12             \n" // q6(a22-a216)

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a20-a26) * k06
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a20-a26) * k16
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a20-a26) * k26
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a20-a26) * k36
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a20-a26) * k46
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a20-a26) * k56
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a20-a26) * k66
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a20-a26) * k76

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a21-a27) * k07
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a21-a27) * k17
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a21-a27) * k27
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a21-a27) * k37
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a21-a27) * k47
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a21-a27) * k57
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a21-a27) * k67
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a21-a27) * k77

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a22-a28) * k08
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a22-a28) * k18
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a22-a28) * k28
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a22-a28) * k38
                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a22-a28) * k48
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a22-a28) * k58
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a22-a28) * k68
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a22-a28) * k78

                        // save s32 to memory
                        "sub        %12, %12, #72       \n"
                        "vst1.s32   {d16-d17}, [%1]!    \n" // out0
                        "vst1.s32   {d18-d19}, [%2]!    \n" // out1
                        "vst1.s32   {d20-d21}, [%3]!    \n" // out2
                        "vst1.s32   {d22-d23}, [%4]!    \n" // out3
                        "subs       %0, #1              \n"
                        "vst1.s32   {d24-d25}, [%5]!    \n" // out4
                        "vst1.s32   {d26-d27}, [%6]!    \n" // out5
                        "vst1.s32   {d28-d29}, [%7]!    \n" // out6
                        "vst1.s32   {d30-d31}, [%8]!    \n" // out7

                        "bne        0b                  \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(ktmp)     // %12
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    int8x8_t _r0_s8 = vld1_s8(r0); // (a00 a01 a02 ....)
                    int8x8_t _r1_s8 = vld1_s8(r1); // (a10 a11 a12 ....)
                    int8x8_t _r2_s8 = vld1_s8(r2); // (a20 a21 a22 ....)

                    int16x8_t _r0 = vmovl_s8(_r0_s8);
                    int16x8_t _r1 = vmovl_s8(_r1_s8);
                    int16x8_t _r2 = vmovl_s8(_r2_s8);

                    int32x4_t _sum03, _sum47;
                    _sum03 = vld1q_lane_s32(outptr0, _sum03, 0); // out0
                    _sum03 = vld1q_lane_s32(outptr1, _sum03, 1); // out1
                    _sum03 = vld1q_lane_s32(outptr2, _sum03, 2); // out2
                    _sum03 = vld1q_lane_s32(outptr3, _sum03, 3); // out3
                    _sum47 = vld1q_lane_s32(outptr4, _sum47, 0); // out4
                    _sum47 = vld1q_lane_s32(outptr5, _sum47, 1); // out5
                    _sum47 = vld1q_lane_s32(outptr6, _sum47, 2); // out6
                    _sum47 = vld1q_lane_s32(outptr7, _sum47, 3); // out7

                    // k0 - k2
                    int8x8_t _k0_8 = vld1_s8(ktmp);      //(k00-k70)
                    int8x8_t _k1_8 = vld1_s8(ktmp + 8);  //(k01-k71)
                    int8x8_t _k2_8 = vld1_s8(ktmp + 16); //(k02-k72)

                    int16x8_t _k0 = vmovl_s8(_k0_8);
                    int16x8_t _k1 = vmovl_s8(_k1_8);
                    int16x8_t _k2 = vmovl_s8(_k2_8);

                    int32x4_t _sum0 = vmull_laneq_s16(vget_low_s16(_k0), _r0, 0);
                    int32x4_t _sum0n = vmull_laneq_s16(vget_high_s16(_k0), _r0, 0);
                    int32x4_t _sum1 = vmull_laneq_s16(vget_low_s16(_k1), _r0, 1);
                    int32x4_t _sum1n = vmull_laneq_s16(vget_high_s16(_k1), _r0, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r0, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r0, 2);

                    // k3 - k5
                    _k0_8 = vld1_s8(ktmp + 24); //(k03-k73)
                    _k1_8 = vld1_s8(ktmp + 32); //(k04-k74)
                    _k2_8 = vld1_s8(ktmp + 40); //(k05-k75)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r1, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r1, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r1, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r1, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r1, 2);

                    // k6 - k8
                    _k0_8 = vld1_s8(ktmp + 48); //(k06-k76)
                    _k1_8 = vld1_s8(ktmp + 56); //(k07-k77)
                    _k2_8 = vld1_s8(ktmp + 64); //(k08-k78)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r2, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r2, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r2, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r2, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r2, 2);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum03 = vaddq_s32(_sum03, _sum0);
                    _sum47 = vaddq_s32(_sum47, _sum0n);

                    vst1q_lane_s32(outptr0, _sum03, 0);
                    vst1q_lane_s32(outptr1, _sum03, 1);
                    vst1q_lane_s32(outptr2, _sum03, 2);
                    vst1q_lane_s32(outptr3, _sum03, 3);
                    vst1q_lane_s32(outptr4, _sum47, 0);
                    vst1q_lane_s32(outptr5, _sum47, 1);
                    vst1q_lane_s32(outptr6, _sum47, 2);
                    vst1q_lane_s32(outptr7, _sum47, 3);

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#else  // __aarch64__
                    asm volatile(
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d0}, [%8]         \n" // d0(a00 a01 a02 ....)
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d2}, [%9]         \n" // d2(a10 a11 a12 ....)
                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d4}, [%10]        \n" // d4(a20 a21 a22 ....)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n" // d6(k00-k70) d7(k01-k71) d8(k02-k72)

                        "vmovl.s8   q0, d0             \n" // d0(a00 a01 a02 x)
                        "vmovl.s8   q1, d2             \n" // d2(a10 a11 a12 x)
                        "vmovl.s8   q2, d4             \n" // d4(a20 a21 a22 x)

                        "vmovl.s8   q5, d8             \n" // d10(k02-k32) d11(k42-k72)
                        "vmovl.s8   q4, d7             \n" // d8(k01-k31) d9(k41-k71)
                        "vmovl.s8   q3, d6             \n" // d6(k00-k30) d7(k40-k70)

                        "vld1.s32   {d20[0]}, [%0]     \n" // out0 q10
                        "vld1.s32   {d20[1]}, [%1]     \n" // out1
                        "vld1.s32   {d21[0]}, [%2]     \n" // out2
                        "vld1.s32   {d21[1]}, [%3]     \n" // out3

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d24-d26}, [%11]!  \n"
                        "vmovl.s8   q14, d26           \n" // d28(k05-k35) d29(k45-k75)
                        "vmovl.s8   q13, d25           \n" // d26(k04-k34) d27(k44-k74)
                        "vmovl.s8   q12, d24           \n" // d24(k03-k33) d25(k43-k73)

                        "vld1.s32   {d22[0]}, [%4]     \n" // out4 q11
                        "vld1.s32   {d22[1]}, [%5]     \n" // out5
                        "vld1.s32   {d23[0]}, [%6]     \n" // out6
                        "vld1.s32   {d23[1]}, [%7]     \n" // out7

                        "vmull.s16  q6, d6, d0[0]      \n" // a00 x (k00-k30)
                        "vmull.s16  q7, d7, d0[0]      \n" // a00 x (k40-k70)
                        "vmull.s16  q8, d8, d0[1]      \n" // a01 x (k01-k31)
                        "vmull.s16  q9, d9, d0[1]      \n" // a01 x (k41-k71)
                        "vmlal.s16  q10, d10, d0[2]    \n" // a02 x (k02-k32)
                        "vmlal.s16  q11, d11, d0[2]    \n" // a02 x (k42-k72)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"
                        "vmovl.s8   q5, d8             \n" // d10(k08-k38) d11(k48-k78)
                        "vmovl.s8   q4, d7             \n" // d8(k07-k37) d9(k47-k77)
                        "vmovl.s8   q3, d6             \n" // d6(k06-k36) d7(k46-k76)

                        "vmlal.s16  q6, d24, d2[0]     \n" // a10 x (k03-k33)
                        "vmlal.s16  q7, d25, d2[0]     \n" // a10 x (k43-k73)
                        "vmlal.s16  q8, d26, d2[1]     \n" // a11 x (k04-k34)
                        "vmlal.s16  q9, d27, d2[1]     \n" // a11 x (k44-k74)
                        "vmlal.s16  q10, d28, d2[2]    \n" // a12 x (k05-k35)
                        "vmlal.s16  q11, d29, d2[2]    \n" // a12 x (k45-k75)

                        "vmlal.s16  q6, d6, d4[0]      \n" // a20 x (k06-k36)
                        "vmlal.s16  q7, d7, d4[0]      \n" // a20 x (k46-k76)
                        "vmlal.s16  q8, d8, d4[1]      \n" // a21 x (k07-k37)
                        "vmlal.s16  q9, d9, d4[1]      \n" // a21 x (k47-k77)
                        "vmlal.s16  q10, d10, d4[2]    \n" // a22 x (k08-k38)
                        "vmlal.s16  q11, d11, d4[2]    \n" // a22 x (k48-k78)

                        "vadd.s32   q8, q8, q6         \n"
                        "vadd.s32   q9, q9, q7         \n"

                        "sub        %11, %11, #72      \n"

                        "vadd.s32   q10, q10, q8       \n"
                        "vadd.s32   q11, q11, q9       \n"

                        "vst1.s32   {d20[0]}, [%0]!    \n" // out0
                        "vst1.s32   {d20[1]}, [%1]!    \n" // out1
                        "vst1.s32   {d21[0]}, [%2]!    \n" // out2
                        "vst1.s32   {d21[1]}, [%3]!    \n" // out3
                        "vst1.s32   {d22[0]}, [%4]!    \n" // out4
                        "vst1.s32   {d22[1]}, [%5]!    \n" // out5
                        "vst1.s32   {d23[0]}, [%6]!    \n" // out6
                        "vst1.s32   {d23[1]}, [%7]!    \n" // out7

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(outptr6), // %6
                        "=r"(outptr7), // %7
                        "=r"(r0),      // %8
                        "=r"(r1),      // %9
                        "=r"(r2),      // %10
                        "=r"(ktmp)     // %11
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(outptr6),
                        "7"(outptr7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(ktmp)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // __ARM_NEON
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;
                    int sum4 = 0;
                    int sum5 = 0;
                    int sum6 = 0;
                    int sum7 = 0;

                    sum0 += (int)r0[0] * ktmp[0];
                    sum1 += (int)r0[0] * ktmp[1];
                    sum2 += (int)r0[0] * ktmp[2];
                    sum3 += (int)r0[0] * ktmp[3];
                    sum4 += (int)r0[0] * ktmp[4];
                    sum5 += (int)r0[0] * ktmp[5];
                    sum6 += (int)r0[0] * ktmp[6];
                    sum7 += (int)r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[1] * ktmp[0];
                    sum1 += (int)r0[1] * ktmp[1];
                    sum2 += (int)r0[1] * ktmp[2];
                    sum3 += (int)r0[1] * ktmp[3];
                    sum4 += (int)r0[1] * ktmp[4];
                    sum5 += (int)r0[1] * ktmp[5];
                    sum6 += (int)r0[1] * ktmp[6];
                    sum7 += (int)r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[2] * ktmp[0];
                    sum1 += (int)r0[2] * ktmp[1];
                    sum2 += (int)r0[2] * ktmp[2];
                    sum3 += (int)r0[2] * ktmp[3];
                    sum4 += (int)r0[2] * ktmp[4];
                    sum5 += (int)r0[2] * ktmp[5];
                    sum6 += (int)r0[2] * ktmp[6];
                    sum7 += (int)r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[0] * ktmp[0];
                    sum1 += (int)r1[0] * ktmp[1];
                    sum2 += (int)r1[0] * ktmp[2];
                    sum3 += (int)r1[0] * ktmp[3];
                    sum4 += (int)r1[0] * ktmp[4];
                    sum5 += (int)r1[0] * ktmp[5];
                    sum6 += (int)r1[0] * ktmp[6];
                    sum7 += (int)r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[1] * ktmp[0];
                    sum1 += (int)r1[1] * ktmp[1];
                    sum2 += (int)r1[1] * ktmp[2];
                    sum3 += (int)r1[1] * ktmp[3];
                    sum4 += (int)r1[1] * ktmp[4];
                    sum5 += (int)r1[1] * ktmp[5];
                    sum6 += (int)r1[1] * ktmp[6];
                    sum7 += (int)r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[2] * ktmp[0];
                    sum1 += (int)r1[2] * ktmp[1];
                    sum2 += (int)r1[2] * ktmp[2];
                    sum3 += (int)r1[2] * ktmp[3];
                    sum4 += (int)r1[2] * ktmp[4];
                    sum5 += (int)r1[2] * ktmp[5];
                    sum6 += (int)r1[2] * ktmp[6];
                    sum7 += (int)r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[0] * ktmp[0];
                    sum1 += (int)r2[0] * ktmp[1];
                    sum2 += (int)r2[0] * ktmp[2];
                    sum3 += (int)r2[0] * ktmp[3];
                    sum4 += (int)r2[0] * ktmp[4];
                    sum5 += (int)r2[0] * ktmp[5];
                    sum6 += (int)r2[0] * ktmp[6];
                    sum7 += (int)r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[1] * ktmp[0];
                    sum1 += (int)r2[1] * ktmp[1];
                    sum2 += (int)r2[1] * ktmp[2];
                    sum3 += (int)r2[1] * ktmp[3];
                    sum4 += (int)r2[1] * ktmp[4];
                    sum5 += (int)r2[1] * ktmp[5];
                    sum6 += (int)r2[1] * ktmp[6];
                    sum7 += (int)r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[2] * ktmp[0];
                    sum1 += (int)r2[2] * ktmp[1];
                    sum2 += (int)r2[2] * ktmp[2];
                    sum3 += (int)r2[2] * ktmp[3];
                    sum4 += (int)r2[2] * ktmp[4];
                    sum5 += (int)r2[2] * ktmp[5];
                    sum6 += (int)r2[2] * ktmp[6];
                    sum7 += (int)r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8 * 9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8 * 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char* ktmp = _kernel.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                   \n"

                        "ld1    {v0.8b, v1.8b}, [%5]          \n" //ktmp
                        "ld2    {v2.8b, v3.8b}, [%2], #16     \n" //r0-r2
                        "ld2    {v4.8b, v5.8b}, [%2]          \n"

                        "ld2    {v6.8b, v7.8b}, [%3], #16     \n" //r3-r5
                        "ld2    {v8.8b, v9.8b}, [%3]          \n"

                        "ld2    {v10.8b, v11.8b}, [%4], #16   \n" //r6-r8
                        "ld2    {v12.8b, v13.8b}, [%4]        \n"

                        "ld1    {v14.4s, v15.4s}, [%1]        \n" //out0

                        "ext    v4.8b, v2.8b, v4.8b, #1       \n"
                        "ext    v8.8b, v6.8b, v8.8b, #1       \n"
                        "ext    v12.8b, v10.8b, v12.8b, #1    \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k0-k7)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k8)
                        "sshll  v2.8h, v2.8b, #0              \n" // r0
                        "sshll  v3.8h, v3.8b, #0              \n" // r1
                        "sshll  v4.8h, v4.8b, #0              \n" // r2
                        "sshll  v6.8h, v6.8b, #0              \n" // r3
                        "sshll  v7.8h, v7.8b, #0              \n" // r4
                        "sshll  v8.8h, v8.8b, #0              \n" // r5
                        "sshll  v10.8h, v10.8b, #0            \n" // r6
                        "sshll  v11.8h, v11.8b, #0            \n" // r7
                        "sshll  v12.8h, v12.8b, #0            \n" // r8

                        // r0
                        "smull  v16.4s, v2.4h, v0.h[0]        \n" // out = r0*k0
                        "smull2  v17.4s, v2.8h, v0.h[0]       \n"
                        "smull  v18.4s, v3.4h, v0.h[1]        \n" // outn = r1*k1
                        "smull2  v19.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v16.4s, v4.4h, v0.h[2]        \n" // out = r2*k2
                        "smlal2  v17.4s, v4.8h, v0.h[2]       \n"
                        "smlal  v18.4s, v6.4h, v0.h[3]        \n" // outn = r3*k3
                        "smlal2  v19.4s, v6.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v0.h[4]        \n" // out = r4*k4
                        "smlal2  v17.4s, v7.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v8.4h, v0.h[5]        \n" // outn = r5*k5
                        "smlal2  v19.4s, v8.8h, v0.h[5]       \n"
                        "smlal  v16.4s, v10.4h, v0.h[6]       \n" // out = r6*k6
                        "smlal2  v17.4s, v10.8h, v0.h[6]      \n"
                        "smlal  v18.4s, v11.4h, v0.h[7]       \n" // outn = r7*k7
                        "smlal2  v19.4s, v11.8h, v0.h[7]      \n"
                        "smlal  v16.4s, v12.4h, v1.h[0]       \n" // out = r8*k8
                        "smlal2  v17.4s, v12.8h, v1.h[0]      \n"

                        "add    v8.4s, v16.4s, v18.4s         \n"
                        "add    v9.4s, v17.4s, v19.4s         \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32     \n"

                        "subs   %w0, %w0, #1                  \n"

                        "bne    0b                            \n"

                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(ktmp)    // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(ktmp)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "vld1.s8    {d0-d1}, [%5]       \n" // d0(k0 - k7) d1(k8 ...)
                        "vmovl.s8   q1, d1              \n" // d2(k8 ...)
                        "vmovl.s8   q0, d0              \n" // d0(k0 - k3) d1(k4 - k7)
                        "0:                             \n"
                        "pld        [%2, #192]          \n"
                        "vld2.s8    {d4-d5}, [%2]!      \n" // r0 d4(a00 a02 ... a014) d5(a01 a03 ... a015)
                        "vld2.s8    {d8-d9}, [%2]       \n" //    d8(a016 ....)
                        "vld2.s8    {d10-d11}, [%3]!    \n" // r1 d10(a10 a12 ... a114) d11(a11 a13 ... a115)
                        "vld2.s8    {d14-d15}, [%3]     \n" //    d14(a116 ....)
                        "vld2.s8    {d16-d17}, [%4]!    \n" // r2 d16(a20 a22 ... a214) d17(a21 a23 ... a215)
                        "vld2.s8    {d20-d21}, [%4]     \n" //    d20(a216 ....)
                        "vld1.s32   {d22-d25}, [%1]     \n" // q11(out0 - out3) q12(out4 - out7)

                        "vext.s8    d8, d4, d8, #1      \n" //  d8(a02 a04 ... a016)
                        "vext.s8    d14, d10, d14, #1   \n" // d14(a12 a14 ... a116)
                        "vext.s8    d20, d16, d20, #1   \n" // d20(a22 a24 ... a216)

                        "vmovl.s8   q3, d5              \n" // q3(a01 a03 ... a015)
                        "vmovl.s8   q2, d4              \n" // q2(a00 a02 ... a014)
                        "vmovl.s8   q4, d8              \n" // q4(a02 a04 ... a016)

                        "vmovl.s8   q6, d11             \n" // q6(a11 a13 ... a115)
                        "vmovl.s8   q5, d10             \n" // q5(a10 a12 ... a114)
                        "vmovl.s8   q7, d14             \n" // q7(a12 a14 ... a116)

                        "vmovl.s8   q9, d17             \n" // q9(a21 a23 ... a215)
                        "vmovl.s8   q8, d16             \n" // q8(a20 a22 ... a214)
                        "vmovl.s8   q10, d20            \n" // q10(a22 a24 ... a216)

                        "vmlal.s16  q11, d4, d0[0]      \n" // k0
                        "vmlal.s16  q12, d5, d0[0]      \n"
                        "vmull.s16  q13, d6, d0[1]      \n" // k1
                        "vmull.s16  q14, d7, d0[1]      \n"
                        "vmlal.s16  q11, d8, d0[2]      \n" // k2
                        "vmlal.s16  q12, d9, d0[2]      \n"

                        "vmlal.s16  q13, d12, d1[0]     \n" // k4
                        "vmlal.s16  q14, d13, d1[0]     \n"
                        "vmlal.s16  q11, d10, d0[3]     \n" // k3
                        "vmlal.s16  q12, d11, d0[3]     \n"
                        "vmlal.s16  q13, d14, d1[1]     \n" // k5
                        "vmlal.s16  q14, d15, d1[1]     \n"

                        "vmlal.s16  q11, d16, d1[2]     \n" // k6
                        "vmlal.s16  q12, d17, d1[2]     \n"
                        "vmlal.s16  q13, d18, d1[3]     \n" // k7
                        "vmlal.s16  q14, d19, d1[3]     \n"
                        "vmlal.s16  q11, d20, d2[0]     \n" // k8
                        "vmlal.s16  q12, d21, d2[0]     \n"

                        "vadd.s32   q11, q11, q13       \n"
                        "vadd.s32   q12, q12, q14       \n"

                        "vst1.32    {d22-d25}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(ktmp)    // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                if (remain > 0)
                {
#if __ARM_NEON
                    int8x8_t _k01234567s8 = vld1_s8(ktmp);
                    int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp + 8);
                    int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
                    int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
                    int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
                    int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
                    int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
                    for (; remain > 0; remain--)
                    {
#if __ARM_NEON
                        int8x8_t _r00s8 = vld1_s8(r0);
                        int8x8_t _r10s8 = vld1_s8(r1);
                        int8x8_t _r20s8 = vld1_s8(r2);

                        int16x8_t _r00s16 = vmovl_s8(_r00s8);
                        int16x8_t _r10s16 = vmovl_s8(_r10s8);
                        int16x8_t _r20s16 = vmovl_s8(_r20s8);

                        int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

                        _sum = vsetq_lane_s32(*outptr, _sum, 3);

#if __aarch64__
                        *outptr = vaddvq_s32(_sum);
#else
                        int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                        _ss = vpadd_s32(_ss, _ss);

                        *outptr = vget_lane_s32(_ss, 0);
#endif // __aarch64__
#else
                        int sum = 0;

                        sum += (int)r0[0] * ktmp[0];
                        sum += (int)r0[1] * ktmp[1];
                        sum += (int)r0[2] * ktmp[2];
                        sum += (int)r1[0] * ktmp[3];
                        sum += (int)r1[1] * ktmp[4];
                        sum += (int)r1[2] * ktmp[5];
                        sum += (int)r2[0] * ktmp[6];
                        sum += (int)r2[1] * ktmp[7];
                        sum += (int)r2[2] * ktmp[8];

                        *outptr += sum;
#endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
