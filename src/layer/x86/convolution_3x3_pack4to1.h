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

static void conv3x3s1_winograd64_transform_kernel_pack4to1_sse(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = 4a-inch/4a-64-outch;
    kernel_tm_pack4.create(4 * inch / 4, 64, outch / 4 + outch % 4, (size_t)4u * 4, 4);

    int p = 0;
    for (; p + 3 < outch; p += 4)
    {
        const Mat k0 = kernel_tm.channel(p);
        const Mat k1 = kernel_tm.channel(p + 1);
        const Mat k2 = kernel_tm.channel(p + 2);
        const Mat k3 = kernel_tm.channel(p + 3);

        Mat g0 = kernel_tm_pack4.channel(p / 4);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 3 < inch; q += 4)
            {
                const float* k00 = k0.row(q);
                const float* k01 = k0.row(q + 1);
                const float* k02 = k0.row(q + 2);
                const float* k03 = k0.row(q + 3);

                const float* k10 = k1.row(q);
                const float* k11 = k1.row(q + 1);
                const float* k12 = k1.row(q + 2);
                const float* k13 = k1.row(q + 3);

                const float* k20 = k2.row(q);
                const float* k21 = k2.row(q + 1);
                const float* k22 = k2.row(q + 2);
                const float* k23 = k2.row(q + 3);

                const float* k30 = k3.row(q);
                const float* k31 = k3.row(q + 1);
                const float* k32 = k3.row(q + 2);
                const float* k33 = k3.row(q + 3);

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
    for (; p < outch; p++)
    {
        const Mat k0 = kernel_tm.channel(p);

        Mat g0 = kernel_tm_pack4.channel(p / 4 + p % 4);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 3 < inch; q += 4)
            {
                const float* k00 = k0.row(q);
                const float* k01 = k0.row(q + 1);
                const float* k02 = k0.row(q + 2);
                const float* k03 = k0.row(q + 3);

                g00[0] = k00[k];
                g00[1] = k01[k];
                g00[2] = k02[k];
                g00[3] = k03[k];

                g00 += 4;
            }
        }
    }
}

static void conv3x3s1_winograd64_pack4to1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[8][8][4];

            __m128 _v5_25 = _mm_set1_ps(5.25f);
            __m128 _vm4_25 = _mm_set1_ps(-4.25f);
            __m128 _vm1_25 = _mm_set1_ps(-1.25f);
            __m128 _v0_25 = _mm_set1_ps(0.25f);
            __m128 _vm2_5 = _mm_set1_ps(-2.5f);
            __m128 _v0_5 = _mm_set1_ps(0.5f);
            __m128 _v2 = _mm_set1_ps(2.f);
            __m128 _v4 = _mm_set1_ps(4.f);

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 4;

                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _r00 = _mm_load_ps(r0);
                        __m128 _r01 = _mm_load_ps(r0 + 4);
                        __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                        __m128 _r04 = _mm_load_ps(r0 + 4 * 4);
                        __m128 _r05 = _mm_load_ps(r0 + 4 * 5);
                        __m128 _r06 = _mm_load_ps(r0 + 4 * 6);
                        __m128 _r07 = _mm_load_ps(r0 + 4 * 7);

                        __m128 _tmp0m = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r04, _r02), _mm_sub_ps(_r00, _r06));
                        __m128 _tmp7m = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r03, _r05), _mm_sub_ps(_r07, _r01));
                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[7][m], _tmp7m);

                        __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _r04, _mm_add_ps(_r02, _r06));
                        __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _r03, _mm_add_ps(_r01, _r05));

                        __m128 _tmp1m = _mm_add_ps(_tmp12a, _tmp12b);
                        __m128 _tmp2m = _mm_sub_ps(_tmp12a, _tmp12b);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);

                        __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _r04, _mm_comp_fmadd_ps(_v0_25, _r02, _r06));
                        __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _r05, _mm_comp_fmadd_ps(_vm2_5, _r03, _mm_mul_ps(_r01, _v0_5)));

                        __m128 _tmp3m = _mm_add_ps(_tmp34a, _tmp34b);
                        __m128 _tmp4m = _mm_sub_ps(_tmp34a, _tmp34b);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[4][m], _tmp4m);

                        __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _r04, _r02), _r06);
                        __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _r05, _mm_comp_fmadd_ps(_vm2_5, _r03, _mm_mul_ps(_r01, _v2)));

                        __m128 _tmp5m = _mm_add_ps(_tmp56a, _tmp56b);
                        __m128 _tmp6m = _mm_sub_ps(_tmp56a, _tmp56b);
                        _mm_store_ps(tmp[5][m], _tmp5m);
                        _mm_store_ps(tmp[6][m], _tmp6m);

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;
                    float* r0_tm_6 = r0_tm_0 + tiles * 4 * 6;
                    float* r0_tm_7 = r0_tm_0 + tiles * 4 * 7;

                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);
                        __m128 _tmp06 = _mm_load_ps(tmp[m][6]);
                        __m128 _tmp07 = _mm_load_ps(tmp[m][7]);

                        __m128 _r0tm0 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_tmp04, _tmp02), _mm_sub_ps(_tmp00, _tmp06));
                        __m128 _r0tm7 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_tmp03, _tmp05), _mm_sub_ps(_tmp07, _tmp01));

                        __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _tmp04, _mm_add_ps(_tmp02, _tmp06));
                        __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _tmp03, _mm_add_ps(_tmp01, _tmp05));

                        __m128 _r0tm1 = _mm_add_ps(_tmp12a, _tmp12b);
                        __m128 _r0tm2 = _mm_sub_ps(_tmp12a, _tmp12b);

                        __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _tmp04, _mm_comp_fmadd_ps(_v0_25, _tmp02, _tmp06));
                        __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _tmp05, _mm_comp_fmadd_ps(_vm2_5, _tmp03, _mm_mul_ps(_tmp01, _v0_5)));

                        __m128 _r0tm3 = _mm_add_ps(_tmp34a, _tmp34b);
                        __m128 _r0tm4 = _mm_sub_ps(_tmp34a, _tmp34b);

                        __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _tmp04, _tmp02), _tmp06);
                        __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _tmp05, _mm_comp_fmadd_ps(_vm2_5, _tmp03, _mm_mul_ps(_tmp01, _v2)));

                        __m128 _r0tm5 = _mm_add_ps(_tmp56a, _tmp56b);
                        __m128 _r0tm6 = _mm_sub_ps(_tmp56a, _tmp56b);

                        _mm_store_ps(r0_tm_0, _r0tm0);
                        _mm_store_ps(r0_tm_1, _r0tm1);
                        _mm_store_ps(r0_tm_2, _r0tm2);
                        _mm_store_ps(r0_tm_3, _r0tm3);
                        _mm_store_ps(r0_tm_4, _r0tm4);
                        _mm_store_ps(r0_tm_5, _r0tm5);
                        _mm_store_ps(r0_tm_6, _r0tm6);
                        _mm_store_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 4 * 8;
                        r0_tm_1 += tiles * 4 * 8;
                        r0_tm_2 += tiles * 4 * 8;
                        r0_tm_3 += tiles * 4 * 8;
                        r0_tm_4 += tiles * 4 * 8;
                        r0_tm_5 += tiles * 4 * 8;
                        r0_tm_6 += tiles * 4 * 8;
                        r0_tm_7 += tiles * 4 * 8;
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
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 64, elemsize, elempack, opt.workspace_allocator);
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

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x8
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(r0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(r0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(r0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(r0 + 4 * 7);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r1);
                    _mm_store_ps(tmpptr + 4 * 3, _r5);
                    _mm_store_ps(tmpptr + 4 * 4, _r2);
                    _mm_store_ps(tmpptr + 4 * 5, _r6);
                    _mm_store_ps(tmpptr + 4 * 6, _r3);
                    _mm_store_ps(tmpptr + 4 * 7, _r7);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 32;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r1);
                    _mm_store_ps(tmpptr + 4 * 2, _r2);
                    _mm_store_ps(tmpptr + 4 * 3, _r3);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 16;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 8 + (i % 8) / 4 + i % 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    __m128 _val = _mm_load_ps(r0);
                    _mm_store_ps(tmpptr, _val);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 4;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 64, outch, 4u, 1, opt.workspace_allocator);

        int nn_outch = outch >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            float* output0_tm = top_blob_tm.channel(p);
            float* output1_tm = top_blob_tm.channel(p + 1);
            float* output2_tm = top_blob_tm.channel(p + 2);
            float* output3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel01_tm = kernel_tm.channel(p / 4);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _val0 = _mm_load_ps(r0);
                        __m128 _val1 = _mm_load_ps(r0 + 4);

                        __m128 _w0 = _mm_load1_ps(kptr);
                        __m128 _w1 = _mm_load1_ps(kptr + 1);
                        __m128 _w2 = _mm_load1_ps(kptr + 2);
                        __m128 _w3 = _mm_load1_ps(kptr + 3);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val0, _w1, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val1, _w1, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val0, _w2, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val1, _w2, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val0, _w3, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val1, _w3, _sum7);

                        r0 += 8;
                        kptr += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output0_tm + 4, _sum1);
                    _mm_storeu_ps(output1_tm, _sum2);
                    _mm_storeu_ps(output1_tm + 4, _sum3);
                    _mm_storeu_ps(output2_tm, _sum4);
                    _mm_storeu_ps(output2_tm + 4, _sum5);
                    _mm_storeu_ps(output3_tm, _sum6);
                    _mm_storeu_ps(output3_tm + 4, _sum7);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _val0 = _mm_load_ps(r0);

                        __m128 _w0 = _mm_load1_ps(kptr);
                        __m128 _w1 = _mm_load1_ps(kptr + 1);
                        __m128 _w2 = _mm_load1_ps(kptr + 2);
                        __m128 _w3 = _mm_load1_ps(kptr + 3);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val0, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val0, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val0, _w3, _sum3);

                        r0 += 4;
                        kptr += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_load_ps(kptr);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        r0 += 1;
                        kptr += 4;
                    }

                    float sum[4];
                    _mm_storeu_ps(sum, _sum);

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];

                    output0_tm += 1;
                    output1_tm += 1;
                    output2_tm += 1;
                    output3_tm += 1;
                }
            }
        }

        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p / 4 + p % 4);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 8);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _val0 = _mm_load_ps(r0);
                        __m128 _val1 = _mm_load_ps(r0 + 4);
                        __m128 _w0 = _mm_load1_ps(kptr);
                        _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_w0, _val1, _sum1);

                        r0 += 8;
                        kptr += 1;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output0_tm + 4, _sum1);

                    output0_tm += 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _val0 = _mm_load_ps(r0);
                        __m128 _w0 = _mm_load1_ps(kptr);
                        _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);

                        r0 += 4;
                        kptr += 1;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);

                    output0_tm += 4;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 8 + (i % 8) / 4 + i % 4);

                    const float* kptr = kernel0_tm.row(r);

                    __m128 _sum0 = _mm_setzero_ps();

                    for (int q = 0; q < inch; q++)
                    {
                        __m128 _val0 = _mm_load_ps(r0);
                        __m128 _w0 = _mm_load_ps(kptr);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 4;
                        kptr += 4;
                    }

                    float sum0 = _mm_reduce_add_ps(_sum0);

                    output0_tm[0] = sum0;

                    output0_tm++;
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
