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

static void pack_A_tile_int8(const Mat& A, Mat& AT, int batch, int max_ii, int max_kk)
{
    const int N = max_kk * batch;

    for (int b = 0; b < batch; b++)
    {
        short* pp = AT.row<short>(b);

        int ii = 0;
        for (; ii + 1 < max_ii; ii += 2)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[batch];
                pp[2] = p0[N];
                pp[3] = p0[N + batch];
                p0 += batch * 2;
                pp += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[N];
                p0 += batch;
                pp += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const short* p0 = (const short*)A + ii * N + b;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += batch;
                pp += 1;
            }
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int batch, int max_jj, int max_kk, int nT)
{
    #pragma omp parallel for num_threads(nT)
    for (int b = 0; b < batch; b++)
    {
        short* pp = BT.row<short>(b);

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            const short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                p0 += max_jj * batch * 2;
                pp += 4;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch;
                pp += 2;
            }
        }
        for (; jj < max_jj; jj++)
        {
            const short* p0 = B;

            int kk = 0;
            p0 += (b * max_jj + jj) * 2;
            for (; kk + 1 < max_kk; kk += 2)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                p0 += max_jj * batch * 2;
                pp += 2;
            }
            p0 -= (b * max_jj + jj) * 2;
            p0 += (b * max_jj + jj);
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                p0 += max_jj * batch;
                pp += 1;
            }
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk, bool k_end)
{
    int* outptr = top_blob;

    int ii = 0;
    for (; ii + 1 < max_ii; ii += 2)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;

                if (k == 0)
                {
                    sum00 = 0;
                    sum01 = 0;
                    sum10 = 0;
                    sum11 = 0;
                }
                else
                {
                    sum00 = outptr[0];
                    sum01 = outptr[1];
                    sum10 = outptr[2];
                    sum11 = outptr[3];
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum00 += pA[0] * pB[0];
                    sum00 += pA[1] * pB[1];
                    sum01 += pA[2] * pB[0];
                    sum01 += pA[3] * pB[1];
                    sum10 += pA[0] * pB[2];
                    sum10 += pA[1] * pB[3];
                    sum11 += pA[2] * pB[2];
                    sum11 += pA[3] * pB[3];

                    pA += 4;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum01 += pA[1] * pB[0];
                    sum10 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }

                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
                outptr += 2 * 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

                if (k == 0)
                {
                    sum0 = 0;
                    sum1 = 0;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[2] * pB[0];
                    sum1 += pA[3] * pB[1];
                    pA += 4;
                    pB += 2;
                }
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[1] * pB[0];
                    pA += 2;
                    pB += 1;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        for (int b = 0; b < batch; b++)
        {
            const short* pAT = AT_tile.row<const short>(b) + max_kk * ii;
            const short* pB = BT_tile.row<const short>(b);

            int jj = 0;
            for (; jj + 1 < max_jj; jj += 2)
            {
                const short* pA = pAT;

                int sum0 = 0;
                int sum1 = 0;

                if (k == 0)
                {
                    sum0 = 0;
                    sum1 = 0;
                }
                else
                {
                    sum0 = outptr[0];
                    sum1 = outptr[1];
                }

                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    sum0 += pA[0] * pB[0];
                    sum0 += pA[1] * pB[1];
                    sum1 += pA[0] * pB[2];
                    sum1 += pA[1] * pB[3];
                    pA += 2;
                    pB += 4;
                }
                for (; kk < max_kk; kk++)
                {
                    sum0 += pA[0] * pB[0];
                    sum1 += pA[0] * pB[1];
                    pA += 1;
                    pB += 2;
                }

                outptr[0] = sum0;
                outptr[1] = sum1;
                outptr += 2;
            }
            for (; jj < max_jj; jj++)
            {
                const short* pA = pAT;

                int sum = 0;

                if (k == 0)
                {
                    sum = 0;
                }
                else
                {
                    sum = outptr[0];
                }

                int kk = 0;
                for (; kk < max_kk; kk++)
                {
                    sum += pA[0] * pB[0];
                    pA += 1;
                    pB += 1;
                }

                outptr[0] = sum;
                outptr += 1;
            }
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve M
    {
        int tile_size = (int)sqrt((float)l2_cache_size_int8 / 3);

        TILE_M = std::max(2, tile_size / 2 * 2);

        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;

        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);

        if (nT > 1)
        {
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
        }
    }

    // solve K
    {
        int tile_size = (int)(sqrt((float)l2_cache_size_int8) - TILE_M);

        TILE_K = std::max(2, tile_size / 2 * 2);

        int nn_K = (K + TILE_K - 1) / TILE_K;

        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
    }

    if (N > 0)
    {
        int tile_size = (int)((l2_cache_size_int8 - TILE_M * TILE_K) / (TILE_M * 2 + TILE_K));

        TILE_N = std::max(1, tile_size);

        int nn_N = (N + TILE_N - 1) / TILE_N;

        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
    }
}

static inline void conv3x3s1_winograd23_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const signed char ktm[4][3] = {
    //     {2, 0, 0},
    //     {1, 1, 1},
    //     {1, -1, 1},
    //     {0, 0, 2}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[4][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 2;
                tmp[1][m] = r0 + r1 + r2;
                tmp[2][m] = r0 - r1 + r2;
                tmp[3][m] = r2 * 2;

                k0 += 3;
            }

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 2;
                short z1 = r0 + r1 + r2;
                short z2 = r0 - r1 + r2;
                short z3 = r2 * 2;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp += 4;
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 16;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 2u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 2u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd23_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd23_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const signed char itm[4][4] = {
    //     {1,  0, -1,  0},
    //     {0,  1,  1,  0},
    //     {0, -1,  1,  0},
    //     {0, -1,  0,  1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w - 1) / 2;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[4][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 2 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 2 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 2 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                    }
                }

                tmp[0][m][0] = r00 - r20;
                tmp[0][m][1] = r01 - r21;
                tmp[1][m][0] = r10 + r20;
                tmp[1][m][1] = r11 + r21;
                tmp[2][m][0] = r20 - r10;
                tmp[2][m][1] = r21 - r11;
                tmp[3][m][0] = r30 - r10;
                tmp[3][m][1] = r31 - r11;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];

                p0[0] = r00 - r20;
                p0[1] = r01 - r21;
                p1[0] = r10 + r20;
                p1[1] = r11 + r21;
                p2[0] = r20 - r10;
                p2[1] = r21 - r11;
                p3[0] = r30 - r10;
                p3[1] = r31 - r11;

                p0 += max_jj * 4 * 2;
                p1 += max_jj * 4 * 2;
                p2 += max_jj * 4 * 2;
                p3 += max_jj * 4 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[4][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 2) + (tj * 2);

            for (int m = 0; m < 4; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;

                if (ti * 2 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 2 + 1 < w) r1 = r0123[1];
                        if (tj * 2 + 2 < w) r2 = r0123[2];
                        if (tj * 2 + 3 < w) r3 = r0123[3];
                    }
                }

                tmp[0][m] = r0 - r2;
                tmp[1][m] = r1 + r2;
                tmp[2][m] = r2 - r1;
                tmp[3][m] = r3 - r1;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 16 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];

                p0[0] = r0 - r2;
                p1[0] = r1 + r2;
                p2[0] = r2 - r1;
                p3[0] = r3 - r1;

                p0 += max_jj * 4;
                p1 += max_jj * 4;
                p2 += max_jj * 4;
                p3 += max_jj * 4;
            }
        }
    }
}

static inline void conv3x3s1_winograd23_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[2][4] = {
    //     {1,  1,  1,  0},
    //     {0,  1, -1,  1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 1) / 2;

    int ii = 0;
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[2][4][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m][0] = r0[0] + r1[0] + r2[0];
                tmp[0][m][1] = r0[1] + r1[1] + r2[1];
                tmp[1][m][0] = r1[0] - r2[0] + r3[0];
                tmp[1][m][1] = r1[1] - r2[1] + r3[1];

                r0 += max_jj * 4 * 2;
                r1 += max_jj * 4 * 2;
                r2 += max_jj * 4 * 2;
                r3 += max_jj * 4 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp00 = tmp[m][0][0] + tmp[m][1][0] + tmp[m][2][0];
                int tmp01 = tmp[m][0][1] + tmp[m][1][1] + tmp[m][2][1];
                int tmp10 = tmp[m][1][0] - tmp[m][2][0] + tmp[m][3][0];
                int tmp11 = tmp[m][1][1] - tmp[m][2][1] + tmp[m][3][1];

                tmp00 = tmp00 >> 2;
                tmp01 = tmp01 >> 2;
                tmp10 = tmp10 >> 2;
                tmp11 = tmp11 >> 2;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 2 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[2][4];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 16 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;

            for (int m = 0; m < 4; m++)
            {
                tmp[0][m] = r0[0] + r1[0] + r2[0];
                tmp[1][m] = r1[0] - r2[0] + r3[0];

                r0 += max_jj * 4;
                r1 += max_jj * 4;
                r2 += max_jj * 4;
                r3 += max_jj * 4;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 2) + (tj * 2);

            for (int m = 0; m < 2; m++)
            {
                if (ti * 2 + m >= outh)
                    continue;

                int tmp0 = tmp[m][0] + tmp[m][1] + tmp[m][2];
                int tmp1 = tmp[m][1] - tmp[m][2] + tmp[m][3];

                tmp0 = tmp0 >> 2;
                tmp1 = tmp1 >> 2;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 2 + 1 < outw) outptr0[1] = tmp1;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd23_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 2n+2, winograd F(2,3)
    int w_tiles = (outw + 1) / 2;
    int h_tiles = (outh + 1) / 2;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 16;

    // NCNN_LOGE("conv3x3s1_winograd23_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 2u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 2u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd23_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd23_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_kernel_tile_int8(const Mat& kernel, Mat& A, int inch, int i, int max_ii, int k, int max_kk)
{
    // const short ktm[6][3] = {
    //     {6, 0, 0},
    //     {-4, -4, -4},
    //     {-4, 4, -4},
    //     {1, 2, 4},
    //     {1, -2, 4},
    //     {0, 0, 6}
    // };

    short* ptmp = A;

    int ii = 0;
    for (; ii < max_ii; ii++)
    {
        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            short tmp[6][3];

            const signed char* k0 = (const signed char*)kernel + (i + ii) * inch * 9 + (k + kk) * 9;

            for (int m = 0; m < 3; m++)
            {
                signed char r0 = k0[0];
                signed char r1 = k0[1];
                signed char r2 = k0[2];

                tmp[0][m] = r0 * 6;
                tmp[1][m] = -r0 * 4 - r1 * 4 - r2 * 4;
                tmp[2][m] = -r0 * 4 + r1 * 4 - r2 * 4;
                tmp[3][m] = r0 + r1 * 2 + r2 * 4;
                tmp[4][m] = r0 - r1 * 2 + r2 * 4;
                tmp[5][m] = r2 * 6;

                k0 += 3;
            }

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];

                short z0 = r0 * 6;
                short z1 = -r0 * 4 - r1 * 4 - r2 * 4;
                short z2 = -r0 * 4 + r1 * 4 - r2 * 4;
                short z3 = r0 + r1 * 2 + r2 * 4;
                short z4 = r0 - r1 * 2 + r2 * 4;
                short z5 = r2 * 6;

                ptmp[0] = z0;
                ptmp[1] = z1;
                ptmp[2] = z2;
                ptmp[3] = z3;
                ptmp[4] = z4;
                ptmp[5] = z5;
                ptmp += 6;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_kernel_int8(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    const int M = outch;
    const int K = inch;
    const int B = 36;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat A_tileX(B * TILE_M * TILE_K, 1, opt.num_threads, 4u, (Allocator*)0);

    AT.create(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, 4u, (Allocator*)0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat A_tile = A_tileX.channel(get_omp_thread_num());

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_ii = std::min((M - i), TILE_M);
            const int max_kk = std::min((K - k), TILE_K);

            conv3x3s1_winograd43_transform_kernel_tile_int8(kernel, A_tile, inch, i, max_ii, k, max_kk);

            Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

            pack_A_tile_int8(A_tile, AT_tile, B, max_ii, max_kk);
        }
    }
}

static inline void conv3x3s1_winograd43_transform_input_tile_int8(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    // const float itm[4][4] = {
    //     {4,  0, -5,  0, 1, 0},
    //     {0, -4, -4,  1, 1, 0},
    //     {0,  4, -4, -1, 1, 0},
    //     {0, -2, -1,  2, 1, 0},
    //     {0,  2, -1, -2, 1, 0},
    //     {0,  4,  0, -5, 0, 1}
    // };

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int elempack = bottom_blob.elempack;
    const int N = bottom_blob.cstep * elempack;

    const int w_tiles = (w + 1) / 4;

    int nn_max_kk = 0;
    int remain_max_kk_start = 0;
    nn_max_kk = (max_kk - remain_max_kk_start) / 2;
    #pragma omp parallel for num_threads(nT)
    for (int ppkk = 0; ppkk < nn_max_kk; ppkk++)
    {
        const int kk = remain_max_kk_start + ppkk * 2;

        short tmp[6][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r00 = 0;
                signed char r01 = 0;
                signed char r10 = 0;
                signed char r11 = 0;
                signed char r20 = 0;
                signed char r21 = 0;
                signed char r30 = 0;
                signed char r31 = 0;
                signed char r40 = 0;
                signed char r41 = 0;
                signed char r50 = 0;
                signed char r51 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        const signed char* r1 = r0 + N;

                        r00 = r0[0];
                        r01 = r1[0];
                        if (tj * 4 + 1 < w)
                        {
                            r10 = r0[1];
                            r11 = r1[1];
                        }
                        if (tj * 4 + 2 < w)
                        {
                            r20 = r0[2];
                            r21 = r1[2];
                        }
                        if (tj * 4 + 3 < w)
                        {
                            r30 = r0[3];
                            r31 = r1[3];
                        }
                        if (tj * 4 + 4 < w)
                        {
                            r40 = r0[4];
                            r41 = r1[4];
                        }
                        if (tj * 4 + 5 < w)
                        {
                            r50 = r0[5];
                            r51 = r1[5];
                        }
                    }
                }

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                tmp[0][m][0] = r40 + r00 * 4 - r20 * 5;
                tmp[0][m][1] = r41 + r01 * 4 - r21 * 5;
                tmp[1][m][0] = tmp120b + tmp120a;
                tmp[1][m][1] = tmp121b + tmp121a;
                tmp[2][m][0] = tmp120b - tmp120a;
                tmp[2][m][1] = tmp121b - tmp121a;
                tmp[3][m][0] = tmp340b + tmp340a;
                tmp[3][m][1] = tmp341b + tmp341a;
                tmp[4][m][0] = tmp340b - tmp340a;
                tmp[4][m][1] = tmp341b - tmp341a;
                tmp[5][m][0] = r50 + r10 * 4 - r30 * 5;
                tmp[5][m][1] = r51 + r11 * 4 - r31 * 5;

                r0 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj * 2;
            short* p1 = p0 + max_jj * 2;
            short* p2 = p0 + max_jj * 2 * 2;
            short* p3 = p0 + max_jj * 2 * 3;
            short* p4 = p0 + max_jj * 2 * 4;
            short* p5 = p0 + max_jj * 2 * 5;

            for (int m = 0; m < 6; m++)
            {
                short r00 = tmp[m][0][0];
                short r01 = tmp[m][0][1];
                short r10 = tmp[m][1][0];
                short r11 = tmp[m][1][1];
                short r20 = tmp[m][2][0];
                short r21 = tmp[m][2][1];
                short r30 = tmp[m][3][0];
                short r31 = tmp[m][3][1];
                short r40 = tmp[m][4][0];
                short r41 = tmp[m][4][1];
                short r50 = tmp[m][5][0];
                short r51 = tmp[m][5][1];

                short tmp120a = r30 - r10 * 4;
                short tmp121a = r31 - r11 * 4;
                short tmp120b = r40 - r20 * 4;
                short tmp121b = r41 - r21 * 4;
                short tmp340a = (r30 - r10) * 2;
                short tmp341a = (r31 - r11) * 2;
                short tmp340b = r40 - r20;
                short tmp341b = r41 - r21;

                p0[0] = r40 + r00 * 4 - r20 * 5;
                p0[1] = r41 + r01 * 4 - r21 * 5;
                p1[0] = tmp120b + tmp120a;
                p1[1] = tmp121b + tmp121a;
                p2[0] = tmp120b - tmp120a;
                p2[1] = tmp121b - tmp121a;
                p3[0] = tmp340b + tmp340a;
                p3[1] = tmp341b + tmp341a;
                p4[0] = tmp340b - tmp340a;
                p4[1] = tmp341b - tmp341a;
                p5[0] = r50 + r10 * 4 - r30 * 5;
                p5[1] = r51 + r11 * 4 - r31 * 5;

                p0 += max_jj * 6 * 2;
                p1 += max_jj * 6 * 2;
                p2 += max_jj * 6 * 2;
                p3 += max_jj * 6 * 2;
                p4 += max_jj * 6 * 2;
                p5 += max_jj * 6 * 2;
            }
        }
    }
    remain_max_kk_start += nn_max_kk * 2;
    for (int kk = remain_max_kk_start; kk < max_kk; kk++)
    {
        short tmp[6][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const signed char* r0123 = bottom_blob.channel(k + kk).row<const signed char>(ti * 4) + (tj * 4);

            for (int m = 0; m < 6; m++)
            {
                signed char r0 = 0;
                signed char r1 = 0;
                signed char r2 = 0;
                signed char r3 = 0;
                signed char r4 = 0;
                signed char r5 = 0;

                if (ti * 4 + m < h)
                {
                    // if (elempack == 1)
                    {
                        r0 = r0123[0];
                        if (tj * 4 + 1 < w) r1 = r0123[1];
                        if (tj * 4 + 2 < w) r2 = r0123[2];
                        if (tj * 4 + 3 < w) r3 = r0123[3];
                        if (tj * 4 + 4 < w) r4 = r0123[4];
                        if (tj * 4 + 5 < w) r5 = r0123[5];
                    }
                }

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                tmp[0][m] = r4 + r0 * 4 - r2 * 5;
                tmp[1][m] = tmp12b + tmp12a;
                tmp[2][m] = tmp12b - tmp12a;
                tmp[3][m] = tmp34b + tmp34a;
                tmp[4][m] = tmp34b - tmp34a;
                tmp[5][m] = r5 + r1 * 4 - r3 * 5;

                r0123 += w;
            }

            short* p0 = (short*)B + kk * max_jj * 36 + jj;
            short* p1 = p0 + max_jj;
            short* p2 = p0 + max_jj * 2;
            short* p3 = p0 + max_jj * 3;
            short* p4 = p0 + max_jj * 4;
            short* p5 = p0 + max_jj * 5;

            for (int m = 0; m < 6; m++)
            {
                short r0 = tmp[m][0];
                short r1 = tmp[m][1];
                short r2 = tmp[m][2];
                short r3 = tmp[m][3];
                short r4 = tmp[m][4];
                short r5 = tmp[m][5];

                short tmp12a = r3 - r1 * 4;
                short tmp12b = r4 - r2 * 4;
                short tmp34a = (r3 - r1) * 2;
                short tmp34b = r4 - r2;

                p0[0] = r4 + r0 * 4 - r2 * 5;
                p1[0] = tmp12b + tmp12a;
                p2[0] = tmp12b - tmp12a;
                p3[0] = tmp34b + tmp34a;
                p4[0] = tmp34b - tmp34a;
                p5[0] = r5 + r1 * 4 - r3 * 5;

                p0 += max_jj * 6;
                p1 += max_jj * 6;
                p2 += max_jj * 6;
                p3 += max_jj * 6;
                p4 += max_jj * 6;
                p5 += max_jj * 6;
            }
        }
    }
}

static inline void conv3x3s1_winograd43_transform_output_tile_int8(const Mat& top_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    // const int otm[4][6] = {
    //     {1, 1,  1, 1,  1, 0},
    //     {0, 1, -1, 2, -2, 0},
    //     {0, 1,  1, 4,  4, 0},
    //     {0, 1, -1, 8, -8, 1}
    // };

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int N = top_blob.cstep * out_elempack;

    const int w_tiles = (outw + 3) / 4;

    int ii = 0;
    for (; ii + 1 < max_ii; ii += 2)
    {
        int tmp[4][6][2];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj * 2;
            const int* r1 = r0 + max_jj * 2;
            const int* r2 = r0 + max_jj * 2 * 2;
            const int* r3 = r0 + max_jj * 2 * 3;
            const int* r4 = r0 + max_jj * 2 * 4;
            const int* r5 = r0 + max_jj * 2 * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a0 = r1[0] + r2[0];
                int tmp02a1 = r1[1] + r2[1];
                int tmp02b0 = r3[0] + r4[0];
                int tmp02b1 = r3[1] + r4[1];
                int tmp13a0 = r1[0] - r2[0];
                int tmp13a1 = r1[1] - r2[1];
                int tmp13b0 = r3[0] - r4[0];
                int tmp13b1 = r3[1] - r4[1];

                int tmp00 = tmp02a0 + tmp02b0 + r0[0];
                int tmp01 = tmp02a1 + tmp02b1 + r0[1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + r5[0] * 4;
                int tmp31 = tmp13a1 + tmp13b1 * 8 + r5[1] * 4;

                tmp00 = tmp00 * 4;
                tmp01 = tmp01 * 4;
                tmp10 = tmp10 * 4;
                tmp11 = tmp11 * 4;
                tmp20 = tmp20 * 4;
                tmp21 = tmp21 * 4;
                tmp30 = tmp30 * 4;
                tmp31 = tmp31 * 4;

                tmp[0][m][0] = tmp00;
                tmp[0][m][1] = tmp01;
                tmp[1][m][0] = tmp10;
                tmp[1][m][1] = tmp11;
                tmp[2][m][0] = tmp20;
                tmp[2][m][1] = tmp21;
                tmp[3][m][0] = tmp30;
                tmp[3][m][1] = tmp31;

                r0 += max_jj * 6 * 2;
                r1 += max_jj * 6 * 2;
                r2 += max_jj * 6 * 2;
                r3 += max_jj * 6 * 2;
                r4 += max_jj * 6 * 2;
                r5 += max_jj * 6 * 2;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a0 = tmp[m][1][0] + tmp[m][2][0];
                int tmp02a1 = tmp[m][1][1] + tmp[m][2][1];
                int tmp02b0 = tmp[m][3][0] + tmp[m][4][0];
                int tmp02b1 = tmp[m][3][1] + tmp[m][4][1];
                int tmp13a0 = tmp[m][1][0] - tmp[m][2][0];
                int tmp13a1 = tmp[m][1][1] - tmp[m][2][1];
                int tmp13b0 = tmp[m][3][0] - tmp[m][4][0];
                int tmp13b1 = tmp[m][3][1] - tmp[m][4][1];

                int tmp00 = tmp02a0 + tmp02b0 + tmp[m][0][0];
                int tmp01 = tmp02a1 + tmp02b1 + tmp[m][0][1];
                int tmp10 = tmp13a0 + tmp13b0 * 2;
                int tmp11 = tmp13a1 + tmp13b1 * 2;
                int tmp20 = tmp02a0 + tmp02b0 * 4;
                int tmp21 = tmp02a1 + tmp02b1 * 4;
                int tmp30 = tmp13a0 + tmp13b0 * 8 + tmp[m][5][0];
                int tmp31 = tmp13a1 + tmp13b1 * 8 + tmp[m][5][1];

                tmp00 = tmp00 / 576;
                tmp01 = tmp01 / 576;
                tmp10 = tmp10 / 576;
                tmp11 = tmp11 / 576;
                tmp20 = tmp20 / 576;
                tmp21 = tmp21 / 576;
                tmp30 = tmp30 / 576;
                tmp31 = tmp31 / 576;

                // if (out_elempack == 1)
                {
                    int* outptr1 = outptr0 + N;

                    outptr0[0] = tmp00;
                    outptr1[0] = tmp01;
                    if (tj * 4 + 1 < outw)
                    {
                        outptr0[1] = tmp10;
                        outptr1[1] = tmp11;
                    }
                    if (tj * 4 + 2 < outw)
                    {
                        outptr0[2] = tmp20;
                        outptr1[2] = tmp21;
                    }
                    if (tj * 4 + 3 < outw)
                    {
                        outptr0[3] = tmp30;
                        outptr1[3] = tmp31;
                    }
                }

                outptr0 += outw;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        int tmp[4][6];

        int jj = 0;
        for (; jj < max_jj; jj++)
        {
            int ti = (j + jj) / w_tiles;
            int tj = (j + jj) % w_tiles;

            const int* r0 = (const int*)top_tile + ii * max_jj * 36 + jj;
            const int* r1 = r0 + max_jj;
            const int* r2 = r0 + max_jj * 2;
            const int* r3 = r0 + max_jj * 3;
            const int* r4 = r0 + max_jj * 4;
            const int* r5 = r0 + max_jj * 5;

            for (int m = 0; m < 5; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }
            for (int m = 5; m < 6; m++)
            {
                int tmp02a = r1[0] + r2[0];
                int tmp02b = r3[0] + r4[0];
                int tmp13a = r1[0] - r2[0];
                int tmp13b = r3[0] - r4[0];

                int tmp0 = tmp02a + tmp02b + r0[0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + r5[0] * 4;

                tmp0 = tmp0 * 4;
                tmp1 = tmp1 * 4;
                tmp2 = tmp2 * 4;
                tmp3 = tmp3 * 4;

                tmp[0][m] = tmp0;
                tmp[1][m] = tmp1;
                tmp[2][m] = tmp2;
                tmp[3][m] = tmp3;

                r0 += max_jj * 6;
                r1 += max_jj * 6;
                r2 += max_jj * 6;
                r3 += max_jj * 6;
                r4 += max_jj * 6;
                r5 += max_jj * 6;
            }

            int* outptr0 = top_blob.channel(i + ii).row<int>(ti * 4) + (tj * 4);

            for (int m = 0; m < 4; m++)
            {
                if (ti * 4 + m >= outh)
                    continue;

                int tmp02a = tmp[m][1] + tmp[m][2];
                int tmp02b = tmp[m][3] + tmp[m][4];
                int tmp13a = tmp[m][1] - tmp[m][2];
                int tmp13b = tmp[m][3] - tmp[m][4];

                int tmp0 = tmp02a + tmp02b + tmp[m][0];
                int tmp1 = tmp13a + tmp13b * 2;
                int tmp2 = tmp02a + tmp02b * 4;
                int tmp3 = tmp13a + tmp13b * 8 + tmp[m][5];

                tmp0 = tmp0 / 576;
                tmp1 = tmp1 / 576;
                tmp2 = tmp2 / 576;
                tmp3 = tmp3 / 576;

                // if (out_elempack == 1)
                {
                    outptr0[0] = tmp0;
                    if (tj * 4 + 1 < outw) outptr0[1] = tmp1;
                    if (tj * 4 + 2 < outw) outptr0[2] = tmp2;
                    if (tj * 4 + 3 < outw) outptr0[3] = tmp3;
                }

                outptr0 += outw;
            }
        }
    }
}

static void conv3x3s1_winograd43_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    int outw = top_blob.w;
    int outh = top_blob.h;

    // pad to 4n+2, winograd F(4,3)
    int w_tiles = (outw + 3) / 4;
    int h_tiles = (outh + 3) / 4;
    int tiles = w_tiles * h_tiles;

    const int M = top_blob.c * top_blob.elempack;
    const int N = tiles;
    const int K = bottom_blob.c * bottom_blob.elempack;
    const int B = 36;

    // NCNN_LOGE("conv3x3s1_winograd43_int8 %d %d %d", M, N, K);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 4u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    if (nT > 1 && nn_NK < nT)
    {
        Mat B_tile(TILE_N * B * TILE_K, 4u, opt.workspace_allocator);

        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, nT);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, nT);
        }
    }
    else
    {
        Mat B_tileX(TILE_N * B * TILE_K, 1, nT, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(nT)
        for (int ppjk = 0; ppjk < nn_NK; ppjk++)
        {
            const int ppj = ppjk / nn_K;
            const int ppk = ppjk % nn_K;

            const int j = ppj * TILE_N;
            const int k = ppk * TILE_K;

            const int max_jj = std::min((N - j), TILE_N);
            const int max_kk = std::min((K - k), TILE_K);

            Mat B_tile = B_tileX.channel(get_omp_thread_num());

            // transform input
            conv3x3s1_winograd43_transform_input_tile_int8(bottom_blob, B_tile, j, max_jj, k, max_kk, 1);

            Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

            transpose_pack_B_tile_int8(B_tile, BT_tile, B, max_jj, max_kk, 1);
        }
    }

    Mat top_tileX(TILE_N * B * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat top_tile = top_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).depth(k / TILE_K);

                const Mat BT_tile = BT.channel(j / TILE_N).depth(k / TILE_K);

                bool k_end = k + TILE_K >= K;

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, top_tile, B, max_ii, max_jj, k, max_kk, k_end);
            }

            // transform output
            conv3x3s1_winograd43_transform_output_tile_int8(top_tile, top_blob, i, max_ii, j, max_jj);
        }
    }
}
