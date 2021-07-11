// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd64_transform_kernel_packn_rvv(const Mat& kernel, Mat& kernel_tm_packn, int inch, int outch)
{
    const int packn = csrr_vlenb() / 4;

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

    #pragma omp parallel for
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
    kernel_tm_packn.create(inch / packn, 64, outch / packn, (size_t)4u * packn * packn, packn * packn);

    for (int q = 0; q + (packn - 1) < outch; q += packn)
    {
        Mat g0 = kernel_tm_packn.channel(q / packn);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + (packn - 1) < inch; p += packn)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
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

static void conv3x3s1_winograd64_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);

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

            // NOTE c99 variable length array
            float tmp[8][8][packn];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row<const float>(i * 6) + (j * 6) * packn;

                    for (int m = 0; m < 8; m++)
                    {
                        vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                        vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                        vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                        vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                        vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);
                        vfloat32m1_t _r05 = vle32_v_f32m1(r0 + packn * 5, vl);
                        vfloat32m1_t _r06 = vle32_v_f32m1(r0 + packn * 6, vl);
                        vfloat32m1_t _r07 = vle32_v_f32m1(r0 + packn * 7, vl);

                        vfloat32m1_t _tmp0m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r00, _r06, vl), 5.25f, vfsub_vv_f32m1(_r04, _r02, vl), vl);
                        vfloat32m1_t _tmp7m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r07, _r01, vl), 5.25f, vfsub_vv_f32m1(_r03, _r05, vl), vl);
                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[7][m], _tmp7m, vl);

                        vfloat32m1_t _tmp12a = vfmacc_vf_f32m1(vfadd_vv_f32m1(_r02, _r06, vl), -4.25f, _r04, vl);
                        vfloat32m1_t _tmp12b = vfmacc_vf_f32m1(vfadd_vv_f32m1(_r01, _r05, vl), -4.25f, _r03, vl);

                        vfloat32m1_t _tmp1m = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                        vfloat32m1_t _tmp2m = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);

                        vfloat32m1_t _tmp34a = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r06, 0.25f, _r02, vl), -1.25f, _r04, vl);
                        vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 0.5f, vl), -2.5f, _r03, vl), 2.f, _r05, vl);

                        vfloat32m1_t _tmp3m = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                        vfloat32m1_t _tmp4m = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                        vse32_v_f32m1(tmp[4][m], _tmp4m, vl);

                        vfloat32m1_t _tmp56a = vfmacc_vf_f32m1(_r06, 4.f, vfmacc_vf_f32m1(_r02, -1.25f, _r04, vl), vl);
                        vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmul_vf_f32m1(_r01, 2.f, vl), -2.5f, _r03, vl), 0.5f, _r05, vl);

                        vfloat32m1_t _tmp5m = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                        vfloat32m1_t _tmp6m = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);
                        vse32_v_f32m1(tmp[5][m], _tmp5m, vl);
                        vse32_v_f32m1(tmp[6][m], _tmp6m, vl);

                        r0 += w * packn;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * packn;
                    float* r0_tm_1 = r0_tm_0 + tiles * packn;
                    float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * packn * 5;
                    float* r0_tm_6 = r0_tm_0 + tiles * packn * 6;
                    float* r0_tm_7 = r0_tm_0 + tiles * packn * 7;

                    for (int m = 0; m < 8; m++)
                    {
                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                        vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                        vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                        vfloat32m1_t _r0tm0 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp00, _tmp06, vl), 5.25f, vfsub_vv_f32m1(_tmp04, _tmp02, vl), vl);
                        vfloat32m1_t _r0tm7 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp07, _tmp01, vl), 5.25f, vfsub_vv_f32m1(_tmp03, _tmp05, vl), vl);

                        vfloat32m1_t _tmp12a = vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp02, _tmp06, vl), -4.25f, _tmp04, vl);
                        vfloat32m1_t _tmp12b = vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp01, _tmp05, vl), -4.25f, _tmp03, vl);

                        vfloat32m1_t _r0tm1 = vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                        vfloat32m1_t _r0tm2 = vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                        vfloat32m1_t _tmp34a = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp06, 0.25f, _tmp02, vl), -1.25f, _tmp04, vl);
                        vfloat32m1_t _tmp34b = vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 0.5f, vl), -2.5f, _tmp03, vl), 2.f, _tmp05, vl);

                        vfloat32m1_t _r0tm3 = vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                        vfloat32m1_t _r0tm4 = vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                        vfloat32m1_t _tmp56a = vfmacc_vf_f32m1(_tmp06, 4.f, vfmacc_vf_f32m1(_tmp02, -1.25f, _tmp04, vl), vl);
                        vfloat32m1_t _tmp56b = vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmul_vf_f32m1(_tmp01, 2.f, vl), -2.5f, _tmp03, vl), 0.5f, _tmp05, vl);

                        vfloat32m1_t _r0tm5 = vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                        vfloat32m1_t _r0tm6 = vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                        vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                        vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                        vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                        vse32_v_f32m1(r0_tm_3, _r0tm3, vl);
                        vse32_v_f32m1(r0_tm_4, _r0tm4, vl);
                        vse32_v_f32m1(r0_tm_5, _r0tm5, vl);
                        vse32_v_f32m1(r0_tm_6, _r0tm6, vl);
                        vse32_v_f32m1(r0_tm_7, _r0tm7, vl);

                        r0_tm_0 += tiles * packn * 8;
                        r0_tm_1 += tiles * packn * 8;
                        r0_tm_2 += tiles * packn * 8;
                        r0_tm_3 += tiles * packn * 8;
                        r0_tm_4 += tiles * packn * 8;
                        r0_tm_5 += tiles * packn * 8;
                        r0_tm_6 += tiles * packn * 8;
                        r0_tm_7 += tiles * packn * 8;
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
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row<float>(i / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr[2] = r0[l + packn * 2];
                        tmpptr[3] = r0[l + packn * 3];
                        tmpptr[4] = r0[l + packn * 4];
                        tmpptr[5] = r0[l + packn * 5];
                        tmpptr[6] = r0[l + packn * 6];
                        tmpptr[7] = r0[l + packn * 7];
                        tmpptr += 8;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _val4 = vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _val5 = vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _val6 = vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _val7 = vle32_v_f32m1(r0 + packn * 7, vl);
                    vsseg8e32_v_f32m1x8(tmpptr, vcreate_f32m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr[2] = r0[l + packn * 2];
                        tmpptr[3] = r0[l + packn * 3];
                        tmpptr += 4;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vsseg4e32_v_f32m1x4(tmpptr, vcreate_f32m1x4(_val0, _val1, _val2, _val3), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr += 2;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vsseg2e32_v_f32m1x2(tmpptr, vcreate_f32m1x2(_val0, _val1), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
                    vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                    vse32_v_f32m1(tmpptr, _val, vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn;
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
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row<const float>(i / 8);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        float val2 = *r0++;
                        float val3 = *r0++;
                        float val4 = *r0++;
                        float val5 = *r0++;
                        float val6 = *r0++;
                        float val7 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                        _sum4 = vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                        _sum5 = vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                        _sum6 = vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                        _sum7 = vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);
                    vse32_v_f32m1(output0_tm + packn * 4, _sum4, vl);
                    vse32_v_f32m1(output0_tm + packn * 5, _sum5, vl);
                    vse32_v_f32m1(output0_tm + packn * 6, _sum6, vl);
                    vse32_v_f32m1(output0_tm + packn * 7, _sum7, vl);

                    output0_tm += packn * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        float val2 = *r0++;
                        float val3 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);

                    output0_tm += packn * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);

                    output0_tm += packn * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum = vfmacc_vf_f32m1(_sum, val, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum, vl);

                    output0_tm += packn;
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
            vfloat32m1_t _bias0 = bias ? vle32_v_f32m1((const float*)bias + p * packn, vl) : vfmv_v_f_f32m1(0.f, vl);

            // NOTE c99 variable length array
            float tmp[6][8][packn];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * packn;
                    const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                    const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * packn * 5;
                    const float* output0_tm_6 = output0_tm_0 + tiles * packn * 6;
                    const float* output0_tm_7 = output0_tm_0 + tiles * packn * 7;

                    float* output0 = out0.row<float>(i * 6) + (j * 6) * packn;

                    // TODO rvv optimize
                    for (int m = 0; m < 8; m++)
                    {
                        vfloat32m1_t _out0tm0 = vle32_v_f32m1(output0_tm_0, vl);
                        vfloat32m1_t _out0tm1 = vle32_v_f32m1(output0_tm_1, vl);
                        vfloat32m1_t _out0tm2 = vle32_v_f32m1(output0_tm_2, vl);
                        vfloat32m1_t _out0tm3 = vle32_v_f32m1(output0_tm_3, vl);
                        vfloat32m1_t _out0tm4 = vle32_v_f32m1(output0_tm_4, vl);
                        vfloat32m1_t _out0tm5 = vle32_v_f32m1(output0_tm_5, vl);
                        vfloat32m1_t _out0tm6 = vle32_v_f32m1(output0_tm_6, vl);
                        vfloat32m1_t _out0tm7 = vle32_v_f32m1(output0_tm_7, vl);

                        vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                        vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);

                        vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_out0tm3, _out0tm4, vl);
                        vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_out0tm3, _out0tm4, vl);

                        vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_out0tm5, _out0tm6, vl);
                        vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_out0tm5, _out0tm6, vl);

                        vfloat32m1_t _tmp0m = vfadd_vv_f32m1(vfadd_vv_f32m1(_out0tm0, _tmp024a, vl), vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl);
                        vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl);
                        vfloat32m1_t _tmp4m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl);
                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                        vse32_v_f32m1(tmp[4][m], _tmp4m, vl);

                        vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl);
                        vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl);
                        vfloat32m1_t _tmp5m = vfadd_vv_f32m1(vfadd_vv_f32m1(_out0tm7, _tmp135a, vl), vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                        vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                        output0_tm_0 += tiles * packn * 8;
                        output0_tm_1 += tiles * packn * 8;
                        output0_tm_2 += tiles * packn * 8;
                        output0_tm_3 += tiles * packn * 8;
                        output0_tm_4 += tiles * packn * 8;
                        output0_tm_5 += tiles * packn * 8;
                        output0_tm_6 += tiles * packn * 8;
                        output0_tm_7 += tiles * packn * 8;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);
                        vfloat32m1_t _tmp06 = vle32_v_f32m1(tmp[m][6], vl);
                        vfloat32m1_t _tmp07 = vle32_v_f32m1(tmp[m][7], vl);

                        vfloat32m1_t _tmp024a = vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                        vfloat32m1_t _tmp135a = vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                        vfloat32m1_t _tmp024b = vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                        vfloat32m1_t _tmp135b = vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                        vfloat32m1_t _tmp024c = vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                        vfloat32m1_t _tmp135c = vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                        vfloat32m1_t _out00 = vfadd_vv_f32m1(_bias0, vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp00, _tmp024a, vl), vfmacc_vf_f32m1(_tmp024b, 32.f, _tmp024c, vl), vl), vl);
                        vfloat32m1_t _out02 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 4.f, _tmp024b, vl), 8.f, _tmp024c, vl), vl);
                        vfloat32m1_t _out04 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp024a, 16.f, _tmp024b, vl), 2.f, _tmp024c, vl), vl);
                        vse32_v_f32m1(output0, _out00, vl);
                        vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                        vse32_v_f32m1(output0 + packn * 4, _out04, vl);

                        vfloat32m1_t _out01 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 2.f, _tmp135b, vl), 16.f, _tmp135c, vl), vl);
                        vfloat32m1_t _out03 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp135a, 8.f, _tmp135b, vl), 4.f, _tmp135c, vl), vl);
                        vfloat32m1_t _out05 = vfadd_vv_f32m1(_bias0, vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp07, _tmp135a, vl), vfmacc_vf_f32m1(_tmp135c, 32.f, _tmp135b, vl), vl), vl);
                        vse32_v_f32m1(output0 + packn, _out01, vl);
                        vse32_v_f32m1(output0 + packn * 3, _out03, vl);
                        vse32_v_f32m1(output0 + packn * 5, _out05, vl);

                        output0 += outw * packn;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd42_transform_kernel_packn_rvv(const Mat& kernel, Mat& kernel_tm_packn, int inch, int outch)
{
    const int packn = csrr_vlenb() / 4;

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

    #pragma omp parallel for
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
    kernel_tm_packn.create(inch / packn, 36, outch / packn, (size_t)4u * packn * packn, packn * packn);

    for (int q = 0; q + (packn - 1) < outch; q += packn)
    {
        Mat g0 = kernel_tm_packn.channel(q / packn);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + (packn - 1) < inch; p += packn)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
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

static void conv3x3s1_winograd42_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);

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

            // NOTE c99 variable length array
            float tmp[6][6][packn];

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const float* r0 = img0.row<const float>(i * 4) + (j * 4) * packn;

                    for (int m = 0; m < 6; m++)
                    {
                        vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                        vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                        vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                        vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                        vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);
                        vfloat32m1_t _r05 = vle32_v_f32m1(r0 + packn * 5, vl);

                        vfloat32m1_t _tmp0m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r04, 4.f, _r00, vl), -5.f, _r02, vl);
                        vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(vfadd_vv_f32m1(_r04, _r03, vl), -4.f, vfadd_vv_f32m1(_r01, _r02, vl), vl);
                        vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r03, vl), 4.f, vfsub_vv_f32m1(_r01, _r02, vl), vl);
                        vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r02, vl), -2.f, vfsub_vv_f32m1(_r01, _r03, vl), vl);
                        vfloat32m1_t _tmp4m = vfmacc_vf_f32m1(vfsub_vv_f32m1(_r04, _r02, vl), 2.f, vfsub_vv_f32m1(_r01, _r03, vl), vl);
                        vfloat32m1_t _tmp5m = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_r05, 4.f, _r01, vl), -5.f, _r03, vl);

                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                        vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                        vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                        r0 += w * packn;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 6 + j) * packn;
                    float* r0_tm_1 = r0_tm_0 + tiles * packn;
                    float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * packn * 5;

                    for (int m = 0; m < 6; m++)
                    {
                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);

                        vfloat32m1_t _r0tm0 = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp04, 4.f, _tmp00, vl), -5.f, _tmp02, vl);
                        vfloat32m1_t _r0tm1 = vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp04, _tmp03, vl), -4.f, vfadd_vv_f32m1(_tmp01, _tmp02, vl), vl);
                        vfloat32m1_t _r0tm2 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp03, vl), 4.f, vfsub_vv_f32m1(_tmp01, _tmp02, vl), vl);
                        vfloat32m1_t _r0tm3 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp02, vl), -2.f, vfsub_vv_f32m1(_tmp01, _tmp03, vl), vl);
                        vfloat32m1_t _r0tm4 = vfmacc_vf_f32m1(vfsub_vv_f32m1(_tmp04, _tmp02, vl), 2.f, vfsub_vv_f32m1(_tmp01, _tmp03, vl), vl);
                        vfloat32m1_t _r0tm5 = vfmacc_vf_f32m1(vfmacc_vf_f32m1(_tmp05, 4.f, _tmp01, vl), -5.f, _tmp03, vl);

                        vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                        vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                        vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                        vse32_v_f32m1(r0_tm_3, _r0tm3, vl);
                        vse32_v_f32m1(r0_tm_4, _r0tm4, vl);
                        vse32_v_f32m1(r0_tm_5, _r0tm5, vl);

                        r0_tm_0 += tiles * packn * 6;
                        r0_tm_1 += tiles * packn * 6;
                        r0_tm_2 += tiles * packn * 6;
                        r0_tm_3 += tiles * packn * 6;
                        r0_tm_4 += tiles * packn * 6;
                        r0_tm_5 += tiles * packn * 6;
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
        if (tiles >= 8)
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
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row<float>(i / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr[2] = r0[l + packn * 2];
                        tmpptr[3] = r0[l + packn * 3];
                        tmpptr[4] = r0[l + packn * 4];
                        tmpptr[5] = r0[l + packn * 5];
                        tmpptr[6] = r0[l + packn * 6];
                        tmpptr[7] = r0[l + packn * 7];
                        tmpptr += 8;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _val4 = vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _val5 = vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _val6 = vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _val7 = vle32_v_f32m1(r0 + packn * 7, vl);
                    vsseg8e32_v_f32m1x8(tmpptr, vcreate_f32m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr[2] = r0[l + packn * 2];
                        tmpptr[3] = r0[l + packn * 3];
                        tmpptr += 4;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _val2 = vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _val3 = vle32_v_f32m1(r0 + packn * 3, vl);
                    vsseg4e32_v_f32m1x4(tmpptr, vcreate_f32m1x4(_val0, _val1, _val2, _val3), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if RVV_SPEC_0_7
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr += 2;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat32m1_t _val0 = vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _val1 = vle32_v_f32m1(r0 + packn, vl);
                    vsseg2e32_v_f32m1x2(tmpptr, vcreate_f32m1x2(_val0, _val1), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row<float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
                    vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                    vse32_v_f32m1(tmpptr, _val, vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn;
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
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row<const float>(i / 8);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        float val2 = *r0++;
                        float val3 = *r0++;
                        float val4 = *r0++;
                        float val5 = *r0++;
                        float val6 = *r0++;
                        float val7 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                        _sum4 = vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                        _sum5 = vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                        _sum6 = vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                        _sum7 = vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);
                    vse32_v_f32m1(output0_tm + packn * 4, _sum4, vl);
                    vse32_v_f32m1(output0_tm + packn * 5, _sum5, vl);
                    vse32_v_f32m1(output0_tm + packn * 6, _sum6, vl);
                    vse32_v_f32m1(output0_tm + packn * 7, _sum7, vl);

                    output0_tm += packn * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        float val2 = *r0++;
                        float val3 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);
                    vse32_v_f32m1(output0_tm + packn * 2, _sum2, vl);
                    vse32_v_f32m1(output0_tm + packn * 3, _sum3, vl);

                    output0_tm += packn * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val0 = *r0++;
                        float val1 = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output0_tm + packn, _sum1, vl);

                    output0_tm += packn * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row<const float>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        float val = *r0++;
                        vfloat32m1_t _w0 = vle32_v_f32m1(k0, vl);
                        _sum = vfmacc_vf_f32m1(_sum, val, _w0, vl);

                        k0 += packn;
                    }

                    vse32_v_f32m1(output0_tm, _sum, vl);

                    output0_tm += packn;
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
            vfloat32m1_t _bias0 = bias ? vle32_v_f32m1((const float*)bias + p * packn, vl) : vfmv_v_f_f32m1(0.f, vl);

            // NOTE variable length array
            float tmp[4][6][packn];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    // top_blob_tm.create(tiles, 36, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 6 + j) * packn;
                    const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                    const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                    float* output0 = out0.row<float>(i * 4) + (j * 4) * packn;

                    // TODO rvv optimize
                    for (int m = 0; m < 6; m++)
                    {
                        vfloat32m1_t _out0tm0 = vle32_v_f32m1(output0_tm_0, vl);
                        vfloat32m1_t _out0tm1 = vle32_v_f32m1(output0_tm_1, vl);
                        vfloat32m1_t _out0tm2 = vle32_v_f32m1(output0_tm_2, vl);
                        vfloat32m1_t _out0tm3 = vle32_v_f32m1(output0_tm_3, vl);
                        vfloat32m1_t _out0tm4 = vle32_v_f32m1(output0_tm_4, vl);
                        vfloat32m1_t _out0tm5 = vle32_v_f32m1(output0_tm_5, vl);

                        vfloat32m1_t _tmp02a = vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                        vfloat32m1_t _tmp13a = vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);

                        vfloat32m1_t _tmp02b = vfadd_vv_f32m1(_out0tm3, _out0tm4, vl);
                        vfloat32m1_t _tmp13b = vfsub_vv_f32m1(_out0tm3, _out0tm4, vl);

                        vfloat32m1_t _tmp0m = vfadd_vv_f32m1(vfadd_vv_f32m1(_out0tm0, _tmp02a, vl), _tmp02b, vl);
                        vfloat32m1_t _tmp1m = vfmacc_vf_f32m1(_tmp13a, 2.f, _tmp13b, vl);
                        vfloat32m1_t _tmp2m = vfmacc_vf_f32m1(_tmp02a, 4.f, _tmp02b, vl);
                        vfloat32m1_t _tmp3m = vfmacc_vf_f32m1(vfadd_vv_f32m1(_out0tm5, _tmp13a, vl), 8.f, _tmp13b, vl);

                        vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                        vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                        vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                        vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                        output0_tm_0 += tiles * packn * 6;
                        output0_tm_1 += tiles * packn * 6;
                        output0_tm_2 += tiles * packn * 6;
                        output0_tm_3 += tiles * packn * 6;
                        output0_tm_4 += tiles * packn * 6;
                        output0_tm_5 += tiles * packn * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        vfloat32m1_t _tmp00 = vle32_v_f32m1(tmp[m][0], vl);
                        vfloat32m1_t _tmp01 = vle32_v_f32m1(tmp[m][1], vl);
                        vfloat32m1_t _tmp02 = vle32_v_f32m1(tmp[m][2], vl);
                        vfloat32m1_t _tmp03 = vle32_v_f32m1(tmp[m][3], vl);
                        vfloat32m1_t _tmp04 = vle32_v_f32m1(tmp[m][4], vl);
                        vfloat32m1_t _tmp05 = vle32_v_f32m1(tmp[m][5], vl);

                        vfloat32m1_t _tmp02a = vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                        vfloat32m1_t _tmp13a = vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                        vfloat32m1_t _tmp02b = vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                        vfloat32m1_t _tmp13b = vfsub_vv_f32m1(_tmp03, _tmp04, vl);

                        vfloat32m1_t _out00 = vfadd_vv_f32m1(_bias0, vfadd_vv_f32m1(vfadd_vv_f32m1(_tmp00, _tmp02a, vl), _tmp02b, vl), vl);
                        vfloat32m1_t _out01 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(_tmp13a, 2.f, _tmp13b, vl), vl);
                        vfloat32m1_t _out02 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(_tmp02a, 4.f, _tmp02b, vl), vl);
                        vfloat32m1_t _out03 = vfadd_vv_f32m1(_bias0, vfmacc_vf_f32m1(vfadd_vv_f32m1(_tmp05, _tmp13a, vl), 8.f, _tmp13b, vl), vl);

                        vse32_v_f32m1(output0, _out00, vl);
                        vse32_v_f32m1(output0 + packn, _out01, vl);
                        vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                        vse32_v_f32m1(output0 + packn * 3, _out03, vl);

                        output0 += outw * packn;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
