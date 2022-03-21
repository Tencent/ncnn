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

static void conv3x3s1_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    int remain_outch_start = 0;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;
        out0.fill(bias0);

        const float* k0 = kernel.channel(p);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            __m256 _k00 = _mm256_loadu_ps(k0);
            __m256 _k01 = _mm256_loadu_ps(k0 + 8);
            __m256 _k02 = _mm256_loadu_ps(k0 + 16);
            __m256 _k10 = _mm256_loadu_ps(k0 + 24);
            __m256 _k11 = _mm256_loadu_ps(k0 + 32);
            __m256 _k12 = _mm256_loadu_ps(k0 + 40);
            __m256 _k20 = _mm256_loadu_ps(k0 + 48);
            __m256 _k21 = _mm256_loadu_ps(k0 + 56);
            __m256 _k22 = _mm256_loadu_ps(k0 + 64);

            int i = 0;

            for (; i < outh; i++)
            {
                const float* r0 = img0.row(i);
                const float* r1 = img0.row(i + 1);
                const float* r2 = img0.row(i + 2);
                int j = 0;
                for (; j < outw; j++)
                {
                    __m256 _r00 = _mm256_loadu_ps(r0);
                    __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                    __m256 _r02 = _mm256_loadu_ps(r0 + 16);

                    __m256 _sum0 = _mm256_mul_ps(_k00, _r00);
                    __m256 _sum1 = _mm256_mul_ps(_k01, _r01);
                    __m256 _sum2 = _mm256_mul_ps(_k02, _r02);

                    __m256 _r10 = _mm256_loadu_ps(r1);
                    __m256 _r11 = _mm256_loadu_ps(r1 + 8);
                    __m256 _r12 = _mm256_loadu_ps(r1 + 16);

                    _sum0 = _mm256_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_k11, _r11, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_k12, _r12, _sum2);

                    __m256 _r20 = _mm256_loadu_ps(r2);
                    __m256 _r21 = _mm256_loadu_ps(r2 + 8);
                    __m256 _r22 = _mm256_loadu_ps(r2 + 16);

                    _sum0 = _mm256_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_k21, _r21, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_k22, _r22, _sum2);
                    __m128 _sum = HorizontalSums(_sum0, _sum1, _sum2);

                    *outptr0 += _mm_reduce_add_ps(_sum); // dot
                    outptr0++;
                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
            }

            k0 += 9 * 8;
        }
    }
}

static void conv3x3s1_winograd64_transform_kernel_pack8to1_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
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
    // dst = 8a-inch/8a-64-outch;
    kernel_tm_pack8.create(8 * inch / 8, 64, outch / 8 + outch % 8, (size_t)4u * 8, 8);

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        Mat g0 = kernel_tm_pack8.channel(p / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_tm.channel(p + j).row(q + i);

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

        Mat g0 = kernel_tm_pack8.channel(p / 8 + p % 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 7 < inch; q += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = k0.row(q + i);

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);

        //         const float itm[8][8] = {
        //             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
        //
        //             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
        //
        //             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        //         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[8][8][8];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 16);
                        __m256 _r03 = _mm256_load_ps(r0 + 24);
                        __m256 _r04 = _mm256_load_ps(r0 + 32);
                        __m256 _r05 = _mm256_load_ps(r0 + 40);
                        __m256 _r06 = _mm256_load_ps(r0 + 48);
                        __m256 _r07 = _mm256_load_ps(r0 + 56);

                        __m256 _tmp0m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r04, _r02), _mm256_sub_ps(_r00, _r06));
                        __m256 _tmp7m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r03, _r05), _mm256_sub_ps(_r07, _r01));
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[7][m], _tmp7m);

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r04, _mm256_add_ps(_r02, _r06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r03, _mm256_add_ps(_r01, _r05));

                        __m256 _tmp1m = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _tmp2m = _mm256_sub_ps(_tmp12a, _tmp12b);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _r02, _r06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(0.5f))));

                        __m256 _tmp3m = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _tmp4m = _mm256_sub_ps(_tmp34a, _tmp34b);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _r02), _r06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(2.f))));

                        __m256 _tmp5m = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _tmp6m = _mm256_sub_ps(_tmp56a, _tmp56b);
                        _mm256_store_ps(tmp[5][m], _tmp5m);
                        _mm256_store_ps(tmp[6][m], _tmp6m);

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 16;
                    float* r0_tm_3 = r0_tm_0 + tiles * 24;
                    float* r0_tm_4 = r0_tm_0 + tiles * 32;
                    float* r0_tm_5 = r0_tm_0 + tiles * 40;
                    float* r0_tm_6 = r0_tm_0 + tiles * 48;
                    float* r0_tm_7 = r0_tm_0 + tiles * 56;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_load_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_load_ps(tmp[m][7]);

                        __m256 _r0tm0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp04, _tmp02), _mm256_sub_ps(_tmp00, _tmp06));
                        __m256 _r0tm7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp03, _tmp05), _mm256_sub_ps(_tmp07, _tmp01));

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp04, _mm256_add_ps(_tmp02, _tmp06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp03, _mm256_add_ps(_tmp01, _tmp05));

                        __m256 _r0tm1 = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _r0tm2 = _mm256_sub_ps(_tmp12a, _tmp12b);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _tmp02, _tmp06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(0.5f))));

                        __m256 _r0tm3 = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _r0tm4 = _mm256_sub_ps(_tmp34a, _tmp34b);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _tmp02), _tmp06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(2.f))));

                        __m256 _r0tm5 = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _r0tm6 = _mm256_sub_ps(_tmp56a, _tmp56b);

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);
                        _mm256_store_ps(r0_tm_4, _r0tm4);
                        _mm256_store_ps(r0_tm_5, _r0tm5);
                        _mm256_store_ps(r0_tm_6, _r0tm6);
                        _mm256_store_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 64;
                        r0_tm_1 += tiles * 64;
                        r0_tm_2 += tiles * 64;
                        r0_tm_3 += tiles * 64;
                        r0_tm_4 += tiles * 64;
                        r0_tm_5 += tiles * 64;
                        r0_tm_6 += tiles * 64;
                        r0_tm_7 += tiles * 64;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = h_tm / 8 * w_tm / 8;

        // permute
        //         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + tiles % 8, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x8
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);

                    tmpptr += 64;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 8 + i % 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _val = _mm256_load_ps(r0);
                    _mm256_store_ps(tmpptr, _val);

                    tmpptr += 8;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 64, outch, 4u, 1, opt.workspace_allocator);

        int nn_outch = outch >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 8;

            float* outptr0_tm = top_blob_tm.channel(p);
            float* outptr1_tm = top_blob_tm.channel(p + 1);
            float* outptr2_tm = top_blob_tm.channel(p + 2);
            float* outptr3_tm = top_blob_tm.channel(p + 3);
            float* outptr4_tm = top_blob_tm.channel(p + 4);
            float* outptr5_tm = top_blob_tm.channel(p + 5);
            float* outptr6_tm = top_blob_tm.channel(p + 6);
            float* outptr7_tm = top_blob_tm.channel(p + 7);

            const Mat kernel01_tm = kernel_tm.channel(p / 8);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _val0 = _mm256_load_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(kptr);
                        __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val0, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                        __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val0, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val0, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                        __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val0, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val0, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                        __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val0, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val0, _w7, _sum7);

                        r0 += 8;
                        kptr += 8;
                    }

                    _mm256_storeu_ps(outptr0_tm, _sum0);
                    _mm256_storeu_ps(outptr1_tm, _sum1);
                    _mm256_storeu_ps(outptr2_tm, _sum2);
                    _mm256_storeu_ps(outptr3_tm, _sum3);
                    _mm256_storeu_ps(outptr4_tm, _sum4);
                    _mm256_storeu_ps(outptr5_tm, _sum5);
                    _mm256_storeu_ps(outptr6_tm, _sum6);
                    _mm256_storeu_ps(outptr7_tm, _sum7);

                    outptr0_tm += 8;
                    outptr1_tm += 8;
                    outptr2_tm += 8;
                    outptr3_tm += 8;
                    outptr4_tm += 8;
                    outptr5_tm += 8;
                    outptr6_tm += 8;
                    outptr7_tm += 8;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 8 + i % 8);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                        r0 += 1;
                        kptr += 8;
                    }

                    float sum[8];
                    _mm256_storeu_ps(sum, _sum);

                    outptr0_tm[0] = sum[0];
                    outptr1_tm[0] = sum[1];
                    outptr2_tm[0] = sum[2];
                    outptr3_tm[0] = sum[3];
                    outptr4_tm[0] = sum[4];
                    outptr5_tm[0] = sum[5];
                    outptr6_tm[0] = sum[6];
                    outptr7_tm[0] = sum[7];

                    outptr0_tm += 1;
                    outptr1_tm += 1;
                    outptr2_tm += 1;
                    outptr3_tm += 1;
                    outptr4_tm += 1;
                    outptr5_tm += 1;
                    outptr6_tm += 1;
                    outptr7_tm += 1;
                }
            }
        }

        int remain_outch_start = nn_outch << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* outptr0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p / 8 + p % 8);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _val0 = _mm256_load_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(kptr);
                        _sum0 = _mm256_comp_fmadd_ps(_w0, _val0, _sum0);

                        r0 += 8;
                        kptr += 1;
                    }

                    _mm256_storeu_ps(outptr0_tm, _sum0);
                    outptr0_tm += 8;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 8 + i % 8);

                    const float* kptr = kernel0_tm.row(r);

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int q = 0; q < inch; q++)
                    {
                        __m256 _val0 = _mm256_load_ps(r0);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 8;
                        kptr += 8;
                    }

                    float sum0 = _mm256_reduce_add_ps(_sum0);

                    outptr0_tm[0] = sum0;
                    outptr0_tm++;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
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
        //         const float otm[6][8] = {
        //             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        //         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            float tmp[6][8];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    // top_blob_tm.create(tiles, 64, outch, 4u, 1, opt.workspace_allocator);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * 1;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 1;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 5;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 6;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 7;

                    // TODO sse optimize
                    for (int m = 0; m < 8; m++)
                    {
                        float tmp024a = output0_tm_1[0] + output0_tm_2[0];
                        float tmp135a = output0_tm_1[0] - output0_tm_2[0];

                        float tmp024b = output0_tm_3[0] + output0_tm_4[0];
                        float tmp135b = output0_tm_3[0] - output0_tm_4[0];

                        float tmp024c = output0_tm_5[0] + output0_tm_6[0];
                        float tmp135c = output0_tm_5[0] - output0_tm_6[0];

                        tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += tiles * 8;
                        output0_tm_1 += tiles * 8;
                        output0_tm_2 += tiles * 8;
                        output0_tm_3 += tiles * 8;
                        output0_tm_4 += tiles * 8;
                        output0_tm_5 += tiles * 8;
                        output0_tm_6 += tiles * 8;
                        output0_tm_7 += tiles * 8;
                    }

                    float* output0 = out0.row(i * 6) + j * 6;

                    for (int m = 0; m < 6; m++)
                    {
                        const float* tmp0 = tmp[m];

                        float tmp024a = tmp0[1] + tmp0[2];
                        float tmp135a = tmp0[1] - tmp0[2];

                        float tmp024b = tmp0[3] + tmp0[4];
                        float tmp135b = tmp0[3] - tmp0[4];

                        float tmp024c = tmp0[5] + tmp0[6];
                        float tmp135c = tmp0[5] - tmp0[6];

                        output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
