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

static void conv3x3s1_winograd42_transform_kernel_pack8to4_int8_msa(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd42 transform kernel
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
    // dst = 4b-8a-inch/8a-36-outch/4b
    kernel_tm_pack8.create(inch / 8, 36, outch / 4, (size_t)2u * 32, 32);

    int q = 0;
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);

        Mat kernel_tm = kernel_tm_pack8.channel(q / 4);

        for (int k = 0; k < 36; k++)
        {
            short* g00 = kernel_tm.row<short>(k);

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    const short* k00 = k0.row<const short>(p + i);
                    const short* k10 = k1.row<const short>(p + i);
                    const short* k20 = k2.row<const short>(p + i);
                    const short* k30 = k3.row<const short>(p + i);

                    g00[0] = k00[k];
                    g00[1] = k10[k];
                    g00[2] = k20[k];
                    g00[3] = k30[k];

                    g00 += 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_pack8to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Option& opt)
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
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = w_tm / 6 * h_tm / 6;

        bottom_blob_tm.create(tiles, 36, inch, 2u * elempack, elempack, opt.workspace_allocator);

        // const float itm[4][4] = {
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

            short tmp[6][6][8];

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const signed char* r0 = img0.row<const signed char>(i * 4) + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        v16i8 _r00_01 = __msa_ld_b(r0, 0);
                        v16i8 _r02_03 = __msa_ld_b(r0 + 16, 0);
                        v16i8 _r04_05 = __msa_ld_b(r0 + 32, 0);
                        v16i8 _extr0001 = __msa_clti_s_b(_r00_01, 0);
                        v16i8 _extr0203 = __msa_clti_s_b(_r02_03, 0);
                        v16i8 _extr0405 = __msa_clti_s_b(_r04_05, 0);
                        v8i16 _r00 = (v8i16)__msa_ilvr_b(_extr0001, _r00_01);
                        v8i16 _r01 = (v8i16)__msa_ilvl_b(_extr0001, _r00_01);
                        v8i16 _r02 = (v8i16)__msa_ilvr_b(_extr0203, _r02_03);
                        v8i16 _r03 = (v8i16)__msa_ilvl_b(_extr0203, _r02_03);
                        v8i16 _r04 = (v8i16)__msa_ilvr_b(_extr0405, _r04_05);
                        v8i16 _r05 = (v8i16)__msa_ilvl_b(_extr0405, _r04_05);

                        v8i16 _v5 = __msa_fill_h(5);

                        v8i16 _tmp0m = __msa_subv_h(__msa_addv_h(__msa_slli_h(_r00, 2), _r04), __msa_mulv_h(_r02, _v5));
                        v8i16 _tmp1m = __msa_subv_h(__msa_addv_h(_r04, _r03), __msa_slli_h(__msa_addv_h(_r01, _r02), 2));
                        v8i16 _tmp2m = __msa_addv_h(__msa_subv_h(_r04, _r03), __msa_slli_h(__msa_subv_h(_r01, _r02), 2));
                        v8i16 _tmp3m = __msa_subv_h(__msa_subv_h(_r04, _r02), __msa_slli_h(__msa_subv_h(_r01, _r03), 1));
                        v8i16 _tmp4m = __msa_addv_h(__msa_subv_h(_r04, _r02), __msa_slli_h(__msa_subv_h(_r01, _r03), 1));
                        v8i16 _tmp5m = __msa_subv_h(__msa_addv_h(__msa_slli_h(_r01, 2), _r05), __msa_mulv_h(_r03, _v5));

                        __msa_st_h(_tmp0m, tmp[0][m], 0);
                        __msa_st_h(_tmp1m, tmp[1][m], 0);
                        __msa_st_h(_tmp2m, tmp[2][m], 0);
                        __msa_st_h(_tmp3m, tmp[3][m], 0);
                        __msa_st_h(_tmp4m, tmp[4][m], 0);
                        __msa_st_h(_tmp5m, tmp[5][m], 0);

                        r0 += w * 8;
                    }

                    short* r0_tm_0 = (short*)img0_tm + (i * w_tm / 6 + j) * 8;
                    short* r0_tm_1 = r0_tm_0 + tiles * 8;
                    short* r0_tm_2 = r0_tm_0 + tiles * 16;
                    short* r0_tm_3 = r0_tm_0 + tiles * 24;
                    short* r0_tm_4 = r0_tm_0 + tiles * 32;
                    short* r0_tm_5 = r0_tm_0 + tiles * 40;

                    for (int m = 0; m < 6; m++)
                    {
                        v8i16 _tmp00 = __msa_ld_h(tmp[m][0], 0);
                        v8i16 _tmp01 = __msa_ld_h(tmp[m][1], 0);
                        v8i16 _tmp02 = __msa_ld_h(tmp[m][2], 0);
                        v8i16 _tmp03 = __msa_ld_h(tmp[m][3], 0);
                        v8i16 _tmp04 = __msa_ld_h(tmp[m][4], 0);
                        v8i16 _tmp05 = __msa_ld_h(tmp[m][5], 0);

                        v8i16 _v5 = __msa_fill_h(5);

                        v8i16 _r0tm0 = __msa_subv_h(__msa_addv_h(__msa_slli_h(_tmp00, 2), _tmp04), __msa_mulv_h(_tmp02, _v5));
                        v8i16 _r0tm1 = __msa_subv_h(__msa_addv_h(_tmp04, _tmp03), __msa_slli_h(__msa_addv_h(_tmp01, _tmp02), 2));
                        v8i16 _r0tm2 = __msa_addv_h(__msa_subv_h(_tmp04, _tmp03), __msa_slli_h(__msa_subv_h(_tmp01, _tmp02), 2));
                        v8i16 _r0tm3 = __msa_subv_h(__msa_subv_h(_tmp04, _tmp02), __msa_slli_h(__msa_subv_h(_tmp01, _tmp03), 1));
                        v8i16 _r0tm4 = __msa_addv_h(__msa_subv_h(_tmp04, _tmp02), __msa_slli_h(__msa_subv_h(_tmp01, _tmp03), 1));
                        v8i16 _r0tm5 = __msa_subv_h(__msa_addv_h(__msa_slli_h(_tmp01, 2), _tmp05), __msa_mulv_h(_tmp03, _v5));

                        __msa_st_h(_r0tm0, r0_tm_0, 0);
                        __msa_st_h(_r0tm1, r0_tm_1, 0);
                        __msa_st_h(_r0tm2, r0_tm_2, 0);
                        __msa_st_h(_r0tm3, r0_tm_3, 0);
                        __msa_st_h(_r0tm4, r0_tm_4, 0);
                        __msa_st_h(_r0tm5, r0_tm_5, 0);

                        r0_tm_0 += tiles * 48;
                        r0_tm_1 += tiles * 48;
                        r0_tm_2 += tiles * 48;
                        r0_tm_3 += tiles * 48;
                        r0_tm_4 += tiles * 48;
                        r0_tm_5 += tiles * 48;
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
        if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                short* tmpptr = tm2.row<short>(i / 2);

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    v8i16 _r0 = __msa_ld_h(r0, 0);
                    v8i16 _r1 = __msa_ld_h(r0 + 8, 0);
                    __msa_st_h(_r0, tmpptr, 0);
                    __msa_st_h(_r1, tmpptr + 8, 0);
                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 16;
                }
            }
            for (; i < tiles; i++)
            {
                short* tmpptr = tm2.row<short>(i / 2 + i % 2);

                const short* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    v8i16 _r0 = __msa_ld_h(r0, 0);
                    __msa_st_h(_r0, tmpptr, 0);
                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 4u * 4, 4, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            int* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 1 < tiles; i += 2)
                {
                    const short* r0 = bb2.row<const short>(i / 2);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    v4i32 _sum0 = __msa_fill_w(0);
                    v4i32 _sum1 = __msa_fill_w(0);
                    v4i32 _sum2 = __msa_fill_w(0);
                    v4i32 _sum3 = __msa_fill_w(0);

                    for (int j = 0; j < nn; j++)
                    {
                        __builtin_prefetch(r0 + 64);
                        __builtin_prefetch(k0 + 128);
                        v8i16 _w0 = __msa_ld_h(k0, 0);
                        v8i16 _w1 = __msa_ld_h(k0 + 8, 0);
                        v8i16 _w2 = __msa_ld_h(k0 + 16, 0);
                        v8i16 _w3 = __msa_ld_h(k0 + 24, 0);

                        v8i16 _extw0 = __msa_clti_s_h(_w0, 0);
                        v8i16 _extw1 = __msa_clti_s_h(_w1, 0);
                        v8i16 _extw2 = __msa_clti_s_h(_w2, 0);
                        v8i16 _extw3 = __msa_clti_s_h(_w3, 0);

                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw0, _w0);
                        v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw0, _w0);
                        v4i32 _w1l = (v4i32)__msa_ilvr_h(_extw1, _w1);
                        v4i32 _w1h = (v4i32)__msa_ilvl_h(_extw1, _w1);
                        v4i32 _w2l = (v4i32)__msa_ilvr_h(_extw2, _w2);
                        v4i32 _w2h = (v4i32)__msa_ilvl_h(_extw2, _w2);
                        v4i32 _w3l = (v4i32)__msa_ilvr_h(_extw3, _w3);
                        v4i32 _w3h = (v4i32)__msa_ilvl_h(_extw3, _w3);

                        v4i32 _val0_0 = __msa_fill_w(r0[0]);
                        v4i32 _val0_1 = __msa_fill_w(r0[1]);
                        v4i32 _val0_2 = __msa_fill_w(r0[2]);
                        v4i32 _val0_3 = __msa_fill_w(r0[3]);
                        v4i32 _val0_4 = __msa_fill_w(r0[4]);
                        v4i32 _val0_5 = __msa_fill_w(r0[5]);
                        v4i32 _val0_6 = __msa_fill_w(r0[6]);
                        v4i32 _val0_7 = __msa_fill_w(r0[7]);
                        v4i32 _val1_0 = __msa_fill_w(r0[8]);
                        v4i32 _val1_1 = __msa_fill_w(r0[9]);
                        v4i32 _val1_2 = __msa_fill_w(r0[10]);
                        v4i32 _val1_3 = __msa_fill_w(r0[11]);
                        v4i32 _val1_4 = __msa_fill_w(r0[12]);
                        v4i32 _val1_5 = __msa_fill_w(r0[13]);
                        v4i32 _val1_6 = __msa_fill_w(r0[14]);
                        v4i32 _val1_7 = __msa_fill_w(r0[15]);

                        _sum0 = __msa_maddv_w(_sum0, _w0l, _val0_0);
                        _sum1 = __msa_maddv_w(_sum1, _w0h, _val0_1);
                        _sum2 = __msa_maddv_w(_sum2, _w0l, _val1_0);
                        _sum3 = __msa_maddv_w(_sum3, _w0h, _val1_1);
                        _sum0 = __msa_maddv_w(_sum0, _w1l, _val0_2);
                        _sum1 = __msa_maddv_w(_sum1, _w1h, _val0_3);
                        _sum2 = __msa_maddv_w(_sum2, _w1l, _val1_2);
                        _sum3 = __msa_maddv_w(_sum3, _w1h, _val1_3);
                        _sum0 = __msa_maddv_w(_sum0, _w2l, _val0_4);
                        _sum1 = __msa_maddv_w(_sum1, _w2h, _val0_5);
                        _sum2 = __msa_maddv_w(_sum2, _w2l, _val1_4);
                        _sum3 = __msa_maddv_w(_sum3, _w2h, _val1_5);
                        _sum0 = __msa_maddv_w(_sum0, _w3l, _val0_6);
                        _sum1 = __msa_maddv_w(_sum1, _w3h, _val0_7);
                        _sum2 = __msa_maddv_w(_sum2, _w3l, _val1_6);
                        _sum3 = __msa_maddv_w(_sum3, _w3h, _val1_7);

                        r0 += 16;
                        k0 += 32;
                    }

                    _sum0 = __msa_addv_w(_sum0, _sum1);
                    _sum2 = __msa_addv_w(_sum2, _sum3);

                    __msa_st_w(_sum0, output0_tm, 0);
                    __msa_st_w(_sum2, output0_tm + 4, 0);

                    output0_tm += 8;
                }
                for (; i < tiles; i++)
                {
                    const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                    const short* k0 = kernel0_tm.row<const short>(r);

                    int nn = inch; // inch always > 0

                    v4i32 _sum0 = __msa_fill_w(0);
                    v4i32 _sum1 = __msa_fill_w(0);

                    for (int j = 0; j < nn; j++)
                    {
                        __builtin_prefetch(r0 + 32);
                        __builtin_prefetch(k0 + 128);
                        v8i16 _w0 = __msa_ld_h(k0, 0);
                        v8i16 _w1 = __msa_ld_h(k0 + 8, 0);
                        v8i16 _w2 = __msa_ld_h(k0 + 16, 0);
                        v8i16 _w3 = __msa_ld_h(k0 + 24, 0);

                        v8i16 _extw0 = __msa_clti_s_h(_w0, 0);
                        v8i16 _extw1 = __msa_clti_s_h(_w1, 0);
                        v8i16 _extw2 = __msa_clti_s_h(_w2, 0);
                        v8i16 _extw3 = __msa_clti_s_h(_w3, 0);

                        v4i32 _w0l = (v4i32)__msa_ilvr_h(_extw0, _w0);
                        v4i32 _w0h = (v4i32)__msa_ilvl_h(_extw0, _w0);
                        v4i32 _w1l = (v4i32)__msa_ilvr_h(_extw1, _w1);
                        v4i32 _w1h = (v4i32)__msa_ilvl_h(_extw1, _w1);
                        v4i32 _w2l = (v4i32)__msa_ilvr_h(_extw2, _w2);
                        v4i32 _w2h = (v4i32)__msa_ilvl_h(_extw2, _w2);
                        v4i32 _w3l = (v4i32)__msa_ilvr_h(_extw3, _w3);
                        v4i32 _w3h = (v4i32)__msa_ilvl_h(_extw3, _w3);

                        v4i32 _val0 = __msa_fill_w(r0[0]);
                        v4i32 _val1 = __msa_fill_w(r0[1]);
                        v4i32 _val2 = __msa_fill_w(r0[2]);
                        v4i32 _val3 = __msa_fill_w(r0[3]);
                        v4i32 _val4 = __msa_fill_w(r0[4]);
                        v4i32 _val5 = __msa_fill_w(r0[5]);
                        v4i32 _val6 = __msa_fill_w(r0[6]);
                        v4i32 _val7 = __msa_fill_w(r0[7]);

                        _sum0 = __msa_maddv_w(_sum0, _w0l, _val0);
                        _sum1 = __msa_maddv_w(_sum1, _w0h, _val1);
                        _sum0 = __msa_maddv_w(_sum0, _w1l, _val2);
                        _sum1 = __msa_maddv_w(_sum1, _w1h, _val3);
                        _sum0 = __msa_maddv_w(_sum0, _w2l, _val4);
                        _sum1 = __msa_maddv_w(_sum1, _w2h, _val5);
                        _sum0 = __msa_maddv_w(_sum0, _w3l, _val6);
                        _sum1 = __msa_maddv_w(_sum1, _w3h, _val7);

                        r0 += 8;
                        k0 += 32;
                    }

                    _sum0 = __msa_addv_w(_sum0, _sum1);

                    __msa_st_w(_sum0, output0_tm, 0);
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
        top_blob_bordered.create(outw, outh, outch, 4u * 4, 4, opt.workspace_allocator);
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

            int tmp[4][6][4];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    // top_blob_tm.create(tiles, 36, outch, elemsize, elempack);

                    const int* output0_tm_0 = (const int*)out0_tm + (i * w_tm / 6 + j) * 4;
                    const int* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const int* output0_tm_2 = output0_tm_0 + tiles * 8;
                    const int* output0_tm_3 = output0_tm_0 + tiles * 12;
                    const int* output0_tm_4 = output0_tm_0 + tiles * 16;
                    const int* output0_tm_5 = output0_tm_0 + tiles * 20;

                    int* output0 = out0.row<int>(i * 4) + (j * 4) * 4;

                    for (int m = 0; m < 5; m++)
                    {
                        v4i32 _out0tm0 = __msa_ld_w(output0_tm_0, 0);
                        v4i32 _out0tm1 = __msa_ld_w(output0_tm_1, 0);
                        v4i32 _out0tm2 = __msa_ld_w(output0_tm_2, 0);
                        v4i32 _out0tm3 = __msa_ld_w(output0_tm_3, 0);
                        v4i32 _out0tm4 = __msa_ld_w(output0_tm_4, 0);
                        v4i32 _out0tm5 = __msa_ld_w(output0_tm_5, 0);

                        v4i32 _tmp02a = __msa_addv_w(_out0tm1, _out0tm2);
                        v4i32 _tmp13a = __msa_subv_w(_out0tm1, _out0tm2);

                        v4i32 _tmp02b = __msa_addv_w(_out0tm3, _out0tm4);
                        v4i32 _tmp13b = __msa_subv_w(_out0tm3, _out0tm4);

                        v4i32 _tmp0m = __msa_addv_w(__msa_addv_w(_out0tm0, _tmp02a), _tmp02b);
                        v4i32 _tmp1m = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                        v4i32 _tmp2m = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                        v4i32 _tmp3m = __msa_addv_w(__msa_addv_w(_tmp13a, __msa_slli_w(_out0tm5, 2)), __msa_slli_w(_tmp13b, 3));

                        __msa_st_w(_tmp0m, tmp[0][m], 0);
                        __msa_st_w(_tmp1m, tmp[1][m], 0);
                        __msa_st_w(_tmp2m, tmp[2][m], 0);
                        __msa_st_w(_tmp3m, tmp[3][m], 0);

                        output0_tm_0 += tiles * 24;
                        output0_tm_1 += tiles * 24;
                        output0_tm_2 += tiles * 24;
                        output0_tm_3 += tiles * 24;
                        output0_tm_4 += tiles * 24;
                        output0_tm_5 += tiles * 24;
                    }
                    for (int m = 5; m < 6; m++)
                    {
                        v4i32 _out0tm0 = __msa_ld_w(output0_tm_0, 0);
                        v4i32 _out0tm1 = __msa_ld_w(output0_tm_1, 0);
                        v4i32 _out0tm2 = __msa_ld_w(output0_tm_2, 0);
                        v4i32 _out0tm3 = __msa_ld_w(output0_tm_3, 0);
                        v4i32 _out0tm4 = __msa_ld_w(output0_tm_4, 0);
                        v4i32 _out0tm5 = __msa_ld_w(output0_tm_5, 0);

                        v4i32 _tmp02a = __msa_addv_w(_out0tm1, _out0tm2);
                        v4i32 _tmp13a = __msa_subv_w(_out0tm1, _out0tm2);

                        v4i32 _tmp02b = __msa_addv_w(_out0tm3, _out0tm4);
                        v4i32 _tmp13b = __msa_subv_w(_out0tm3, _out0tm4);

                        v4i32 _tmp0m = __msa_addv_w(__msa_addv_w(_out0tm0, _tmp02a), _tmp02b);
                        v4i32 _tmp1m = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                        v4i32 _tmp2m = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                        v4i32 _tmp3m = __msa_addv_w(__msa_addv_w(_tmp13a, __msa_slli_w(_out0tm5, 2)), __msa_slli_w(_tmp13b, 3));

                        _tmp0m = __msa_slli_w(_tmp0m, 2);
                        _tmp1m = __msa_slli_w(_tmp1m, 2);
                        _tmp2m = __msa_slli_w(_tmp2m, 2);
                        _tmp3m = __msa_slli_w(_tmp3m, 2);

                        __msa_st_w(_tmp0m, tmp[0][m], 0);
                        __msa_st_w(_tmp1m, tmp[1][m], 0);
                        __msa_st_w(_tmp2m, tmp[2][m], 0);
                        __msa_st_w(_tmp3m, tmp[3][m], 0);

                        output0_tm_0 += tiles * 24;
                        output0_tm_1 += tiles * 24;
                        output0_tm_2 += tiles * 24;
                        output0_tm_3 += tiles * 24;
                        output0_tm_4 += tiles * 24;
                        output0_tm_5 += tiles * 24;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        v4i32 _tmp00 = __msa_ld_w(tmp[m][0], 0);
                        v4i32 _tmp01 = __msa_ld_w(tmp[m][1], 0);
                        v4i32 _tmp02 = __msa_ld_w(tmp[m][2], 0);
                        v4i32 _tmp03 = __msa_ld_w(tmp[m][3], 0);
                        v4i32 _tmp04 = __msa_ld_w(tmp[m][4], 0);
                        v4i32 _tmp05 = __msa_ld_w(tmp[m][5], 0);

                        v4i32 _tmp02a = __msa_addv_w(_tmp01, _tmp02);
                        v4i32 _tmp13a = __msa_subv_w(_tmp01, _tmp02);

                        v4i32 _tmp02b = __msa_addv_w(_tmp03, _tmp04);
                        v4i32 _tmp13b = __msa_subv_w(_tmp03, _tmp04);

                        v4i32 _out00 = __msa_addv_w(__msa_addv_w(_tmp00, _tmp02a), _tmp02b);
                        v4i32 _out01 = __msa_addv_w(_tmp13a, __msa_slli_w(_tmp13b, 1));
                        v4i32 _out02 = __msa_addv_w(_tmp02a, __msa_slli_w(_tmp02b, 2));
                        v4i32 _out03 = __msa_addv_w(__msa_addv_w(_tmp05, _tmp13a), __msa_slli_w(_tmp13b, 3));

                        // TODO use integer trick for division by 576
                        v4f32 _v576 = __msa_fill_w_f32(1.0 / 576);
                        _out00 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out00), _v576));
                        _out01 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out01), _v576));
                        _out02 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out02), _v576));
                        _out03 = __msa_ftint_s_w(__msa_fmul_w(__msa_ffint_s_w(_out03), _v576));

                        __msa_st_w(_out00, output0, 0);
                        __msa_st_w(_out01, output0 + 4, 0);
                        __msa_st_w(_out02, output0 + 8, 0);
                        __msa_st_w(_out03, output0 + 12, 0);

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
