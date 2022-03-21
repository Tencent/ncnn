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

static void conv3x3s1_winograd64_transform_kernel_pack4_sse(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
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
    // dst = pb-pa-inch/pa-64-outch/pb
    kernel_tm_pack4.create(inch / 4, 64, outch / 4, (size_t)4u * 4 * 4, 4 * 4);

    for (int q = 0; q + (4 - 1) < outch; q += 4)
    {
        Mat g0 = kernel_tm_pack4.channel(q / 4);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + (4 - 1) < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = (float)k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

        //         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        bottom_blob_tm.create(tiles, 64, inch, 4u * elempack, elempack, opt.workspace_allocator);

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
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 64, 4u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2.row(i / 12);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x12
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(r0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(r0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(r0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(r0 + 4 * 7);
                    __m128 _r8 = _mm_load_ps(r0 + 4 * 8);
                    __m128 _r9 = _mm_load_ps(r0 + 4 * 9);
                    __m128 _ra = _mm_load_ps(r0 + 4 * 10);
                    __m128 _rb = _mm_load_ps(r0 + 4 * 11);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r8);
                    _mm_store_ps(tmpptr + 4 * 3, _r1);
                    _mm_store_ps(tmpptr + 4 * 4, _r5);
                    _mm_store_ps(tmpptr + 4 * 5, _r9);
                    _mm_store_ps(tmpptr + 4 * 6, _r2);
                    _mm_store_ps(tmpptr + 4 * 7, _r6);
                    _mm_store_ps(tmpptr + 4 * 8, _ra);
                    _mm_store_ps(tmpptr + 4 * 9, _r3);
                    _mm_store_ps(tmpptr + 4 * 10, _r7);
                    _mm_store_ps(tmpptr + 4 * 11, _rb);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 48;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8);

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
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

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
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x2
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);

                    __m128 _r01_0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _r01_1 = _mm_unpackhi_ps(_r0, _r1);

                    _mm_store_ps(tmpptr, _r01_0);
                    _mm_store_ps(tmpptr + 4, _r01_1);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 8;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

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

        top_blob_tm.create(tiles, 64, outch, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();
                    __m128 _sum8 = _mm_setzero_ps();
                    __m128 _sum9 = _mm_setzero_ps();
                    __m128 _suma = _mm_setzero_ps();
                    __m128 _sumb = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);
                        __m128 _val8 = _mm_load1_ps(r0 + 8);
                        __m128 _val9 = _mm_load1_ps(r0 + 9);
                        __m128 _vala = _mm_load1_ps(r0 + 10);
                        __m128 _valb = _mm_load1_ps(r0 + 11);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);
                        _sum8 = _mm_comp_fmadd_ps(_val8, _w0, _sum8);
                        _sum9 = _mm_comp_fmadd_ps(_val9, _w0, _sum9);
                        _suma = _mm_comp_fmadd_ps(_vala, _w0, _suma);
                        _sumb = _mm_comp_fmadd_ps(_valb, _w0, _sumb);

                        r0 += 12;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);
                    _mm_store_ps(output0_tm + 4 * 8, _sum8);
                    _mm_store_ps(output0_tm + 4 * 9, _sum9);
                    _mm_store_ps(output0_tm + 4 * 10, _suma);
                    _mm_store_ps(output0_tm + 4 * 11, _sumb);

                    output0_tm += 4 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                    const float* k0 = kernel0_tm.row(r);

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
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);

                    output0_tm += 4 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);

                    output0_tm += 4 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);

                    output0_tm += 4 * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);
                        __m128 _val0 = _mm_load1_ps(r0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum);

                    output0_tm += 4;
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
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
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

            //             const float bias0 = bias ? bias[p] : 0.f;
            __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_setzero_ps();

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[6][8][4];

            __m128 _v32 = _mm_set1_ps(32.f);
            __m128 _v16 = _mm_set1_ps(16.f);
            __m128 _v8 = _mm_set1_ps(8.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v2 = _mm_set1_ps(2.f);

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 4 * 6;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 4 * 7;

                    float* output0 = out0.row<float>(i * 6) + (j * 6) * 4;

                    // TODO msa optimize
                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _out0tm0 = _mm_load_ps(output0_tm_0);
                        __m128 _out0tm1 = _mm_load_ps(output0_tm_1);
                        __m128 _out0tm2 = _mm_load_ps(output0_tm_2);
                        __m128 _out0tm3 = _mm_load_ps(output0_tm_3);
                        __m128 _out0tm4 = _mm_load_ps(output0_tm_4);
                        __m128 _out0tm5 = _mm_load_ps(output0_tm_5);
                        __m128 _out0tm6 = _mm_load_ps(output0_tm_6);
                        __m128 _out0tm7 = _mm_load_ps(output0_tm_7);

                        __m128 _tmp024a = _mm_add_ps(_out0tm1, _out0tm2);
                        __m128 _tmp135a = _mm_sub_ps(_out0tm1, _out0tm2);

                        __m128 _tmp024b = _mm_add_ps(_out0tm3, _out0tm4);
                        __m128 _tmp135b = _mm_sub_ps(_out0tm3, _out0tm4);

                        __m128 _tmp024c = _mm_add_ps(_out0tm5, _out0tm6);
                        __m128 _tmp135c = _mm_sub_ps(_out0tm5, _out0tm6);

                        __m128 _tmp0m = _mm_add_ps(_mm_add_ps(_out0tm0, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b));
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a));
                        __m128 _tmp4m = _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a));
                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[4][m], _tmp4m);

                        __m128 _tmp1m = _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a));
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a));
                        __m128 _tmp5m = _mm_add_ps(_mm_add_ps(_out0tm7, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c));
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[5][m], _tmp5m);

                        output0_tm_0 += tiles * 4 * 8;
                        output0_tm_1 += tiles * 4 * 8;
                        output0_tm_2 += tiles * 4 * 8;
                        output0_tm_3 += tiles * 4 * 8;
                        output0_tm_4 += tiles * 4 * 8;
                        output0_tm_5 += tiles * 4 * 8;
                        output0_tm_6 += tiles * 4 * 8;
                        output0_tm_7 += tiles * 4 * 8;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);
                        __m128 _tmp06 = _mm_load_ps(tmp[m][6]);
                        __m128 _tmp07 = _mm_load_ps(tmp[m][7]);

                        __m128 _tmp024a = _mm_add_ps(_tmp01, _tmp02);
                        __m128 _tmp135a = _mm_sub_ps(_tmp01, _tmp02);

                        __m128 _tmp024b = _mm_add_ps(_tmp03, _tmp04);
                        __m128 _tmp135b = _mm_sub_ps(_tmp03, _tmp04);

                        __m128 _tmp024c = _mm_add_ps(_tmp05, _tmp06);
                        __m128 _tmp135c = _mm_sub_ps(_tmp05, _tmp06);

                        __m128 _out00 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp00, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b)));
                        __m128 _out02 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a)));
                        __m128 _out04 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a)));
                        _mm_store_ps(output0, _out00);
                        _mm_store_ps(output0 + 4 * 2, _out02);
                        _mm_store_ps(output0 + 4 * 4, _out04);

                        __m128 _out01 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a)));
                        __m128 _out03 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a)));
                        __m128 _out05 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp07, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c)));
                        _mm_store_ps(output0 + 4, _out01);
                        _mm_store_ps(output0 + 4 * 3, _out03);
                        _mm_store_ps(output0 + 4 * 5, _out05);

                        output0 += outw * 4;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd42_transform_kernel_pack4_sse(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
{
    // winograd42 transform kernel
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
    // dst = pb-pa-inch/pa-36-outch/pb
    kernel_tm_pack4.create(inch / 4, 36, outch / 4, (size_t)4u * 4 * 4, 4 * 4);

    for (int q = 0; q + (4 - 1) < outch; q += 4)
    {
        Mat g0 = kernel_tm_pack4.channel(q / 4);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + (4 - 1) < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = (float)k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = w_tm / 6 * h_tm / 6;

        bottom_blob_tm.create(tiles, 36, inch, 4u * elempack, elempack, opt.workspace_allocator);

        // const float itm[6][6] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =  4 * r00 - 5 * r02 + r04
        // 1 = -4 * (r01 + r02) + r04 + r03
        // 2 =  4 * (r01 - r02) + r04 - r03
        // 3 = -2 * (r01 - r03) + r04 - r02
        // 4 =  2 * (r01 - r03) + r04 - r02
        // 5 =  4 * r01 - 5 * r03 + r05

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
            float tmp[6][6][4];

            __m128 _vm5 = _mm_set1_ps(-5.f);
            __m128 _vm4 = _mm_set1_ps(-4.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _vm2 = _mm_set1_ps(-2.f);
            __m128 _v2 = _mm_set1_ps(2.f);

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const float* r0 = img0.row(i * 4) + (j * 4) * 4;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _r00 = _mm_load_ps(r0);
                        __m128 _r01 = _mm_load_ps(r0 + 4);
                        __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                        __m128 _r04 = _mm_load_ps(r0 + 4 * 4);
                        __m128 _r05 = _mm_load_ps(r0 + 4 * 5);

                        __m128 _tmp0m = _mm_comp_fmadd_ps(_vm5, _r02, _mm_comp_fmadd_ps(_v4, _r00, _r04));
                        __m128 _tmp1m = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_r01, _r02), _mm_add_ps(_r04, _r03));
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_r01, _r02), _mm_sub_ps(_r04, _r03));
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_r01, _r03), _mm_sub_ps(_r04, _r02));
                        __m128 _tmp4m = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_r01, _r03), _mm_sub_ps(_r04, _r02));
                        __m128 _tmp5m = _mm_comp_fmadd_ps(_vm5, _r03, _mm_comp_fmadd_ps(_v4, _r01, _r05));

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[4][m], _tmp4m);
                        _mm_store_ps(tmp[5][m], _tmp5m);

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 6 + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);

                        __m128 _r0tm0 = _mm_comp_fmadd_ps(_vm5, _tmp02, _mm_comp_fmadd_ps(_v4, _tmp00, _tmp04));
                        __m128 _r0tm1 = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_tmp01, _tmp02), _mm_add_ps(_tmp04, _tmp03));
                        __m128 _r0tm2 = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_tmp01, _tmp02), _mm_sub_ps(_tmp04, _tmp03));
                        __m128 _r0tm3 = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_tmp01, _tmp03), _mm_sub_ps(_tmp04, _tmp02));
                        __m128 _r0tm4 = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_tmp01, _tmp03), _mm_sub_ps(_tmp04, _tmp02));
                        __m128 _r0tm5 = _mm_comp_fmadd_ps(_vm5, _tmp03, _mm_comp_fmadd_ps(_v4, _tmp01, _tmp05));

                        _mm_store_ps(r0_tm_0, _r0tm0);
                        _mm_store_ps(r0_tm_1, _r0tm1);
                        _mm_store_ps(r0_tm_2, _r0tm2);
                        _mm_store_ps(r0_tm_3, _r0tm3);
                        _mm_store_ps(r0_tm_4, _r0tm4);
                        _mm_store_ps(r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 4 * 6;
                        r0_tm_1 += tiles * 4 * 6;
                        r0_tm_2 += tiles * 4 * 6;
                        r0_tm_3 += tiles * 4 * 6;
                        r0_tm_4 += tiles * 4 * 6;
                        r0_tm_5 += tiles * 4 * 6;
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
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = h_tm / 6 * w_tm / 6;

        // permute
        //         bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2.row(i / 12);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x12
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(r0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(r0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(r0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(r0 + 4 * 7);
                    __m128 _r8 = _mm_load_ps(r0 + 4 * 8);
                    __m128 _r9 = _mm_load_ps(r0 + 4 * 9);
                    __m128 _ra = _mm_load_ps(r0 + 4 * 10);
                    __m128 _rb = _mm_load_ps(r0 + 4 * 11);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                    _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                    _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r4);
                    _mm_store_ps(tmpptr + 4 * 2, _r8);
                    _mm_store_ps(tmpptr + 4 * 3, _r1);
                    _mm_store_ps(tmpptr + 4 * 4, _r5);
                    _mm_store_ps(tmpptr + 4 * 5, _r9);
                    _mm_store_ps(tmpptr + 4 * 6, _r2);
                    _mm_store_ps(tmpptr + 4 * 7, _r6);
                    _mm_store_ps(tmpptr + 4 * 8, _ra);
                    _mm_store_ps(tmpptr + 4 * 9, _r3);
                    _mm_store_ps(tmpptr + 4 * 10, _r7);
                    _mm_store_ps(tmpptr + 4 * 11, _rb);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 48;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8);

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
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

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
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x2
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);

                    __m128 _r01_0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _r01_1 = _mm_unpackhi_ps(_r0, _r1);

                    _mm_store_ps(tmpptr, _r01_0);
                    _mm_store_ps(tmpptr + 4, _r01_1);

                    r0 += bottom_blob_tm.cstep * 4;
                    tmpptr += 8;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

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

        top_blob_tm.create(tiles, 36, outch, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();
                    __m128 _sum8 = _mm_setzero_ps();
                    __m128 _sum9 = _mm_setzero_ps();
                    __m128 _suma = _mm_setzero_ps();
                    __m128 _sumb = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);
                        __m128 _val8 = _mm_load1_ps(r0 + 8);
                        __m128 _val9 = _mm_load1_ps(r0 + 9);
                        __m128 _vala = _mm_load1_ps(r0 + 10);
                        __m128 _valb = _mm_load1_ps(r0 + 11);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);
                        _sum8 = _mm_comp_fmadd_ps(_val8, _w0, _sum8);
                        _sum9 = _mm_comp_fmadd_ps(_val9, _w0, _sum9);
                        _suma = _mm_comp_fmadd_ps(_vala, _w0, _suma);
                        _sumb = _mm_comp_fmadd_ps(_valb, _w0, _sumb);

                        r0 += 12;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);
                    _mm_store_ps(output0_tm + 4 * 8, _sum8);
                    _mm_store_ps(output0_tm + 4 * 9, _sum9);
                    _mm_store_ps(output0_tm + 4 * 10, _suma);
                    _mm_store_ps(output0_tm + 4 * 11, _sumb);

                    output0_tm += 4 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                    const float* k0 = kernel0_tm.row(r);

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
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);

                    output0_tm += 4 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);

                    output0_tm += 4 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);

                    output0_tm += 4 * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row<const float>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);
                        __m128 _val0 = _mm_load1_ps(r0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum);

                    output0_tm += 4;
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
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        // const float otm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 = r00 + (r01 + r02) + (r03 + r04)
        // 1 =       (r01 - r02) + (r03 - r04) * 2
        // 2 =       (r01 + r02) + (r03 + r04) * 4
        // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;
        const int tiles = w_tm / 6 * h_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            // const float bias0 = bias ? bias[p] : 0.f;
            __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + p * 4) : _mm_setzero_ps();

#ifdef _MSC_VER
            __declspec(align(16))
#else
            __attribute__((aligned(16)))
#endif
            float tmp[4][6][4];

            __m128 _v2 = _mm_set1_ps(2.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v8 = _mm_set1_ps(8.f);

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    // top_blob_tm.create(tiles, 36, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 6 + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;

                    float* output0 = out0.row<float>(i * 4) + (j * 4) * 4;

                    // TODO msa optimize
                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _out0tm0 = _mm_load_ps(output0_tm_0);
                        __m128 _out0tm1 = _mm_load_ps(output0_tm_1);
                        __m128 _out0tm2 = _mm_load_ps(output0_tm_2);
                        __m128 _out0tm3 = _mm_load_ps(output0_tm_3);
                        __m128 _out0tm4 = _mm_load_ps(output0_tm_4);
                        __m128 _out0tm5 = _mm_load_ps(output0_tm_5);

                        __m128 _tmp02a = _mm_add_ps(_out0tm1, _out0tm2);
                        __m128 _tmp13a = _mm_sub_ps(_out0tm1, _out0tm2);

                        __m128 _tmp02b = _mm_add_ps(_out0tm3, _out0tm4);
                        __m128 _tmp13b = _mm_sub_ps(_out0tm3, _out0tm4);

                        __m128 _tmp0m = _mm_add_ps(_mm_add_ps(_out0tm0, _tmp02a), _tmp02b);
                        __m128 _tmp1m = _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a);
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a);
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_out0tm5, _tmp13a));

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[3][m], _tmp3m);

                        output0_tm_0 += tiles * 4 * 6;
                        output0_tm_1 += tiles * 4 * 6;
                        output0_tm_2 += tiles * 4 * 6;
                        output0_tm_3 += tiles * 4 * 6;
                        output0_tm_4 += tiles * 4 * 6;
                        output0_tm_5 += tiles * 4 * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);

                        __m128 _tmp02a = _mm_add_ps(_tmp01, _tmp02);
                        __m128 _tmp13a = _mm_sub_ps(_tmp01, _tmp02);

                        __m128 _tmp02b = _mm_add_ps(_tmp03, _tmp04);
                        __m128 _tmp13b = _mm_sub_ps(_tmp03, _tmp04);

                        __m128 _out00 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp00, _tmp02a), _tmp02b));
                        __m128 _out01 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a));
                        __m128 _out02 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a));
                        __m128 _out03 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_tmp05, _tmp13a)));

                        _mm_store_ps(output0, _out00);
                        _mm_store_ps(output0 + 4, _out01);
                        _mm_store_ps(output0 + 4 * 2, _out02);
                        _mm_store_ps(output0 + 4 * 3, _out03);

                        output0 += outw * 4;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
