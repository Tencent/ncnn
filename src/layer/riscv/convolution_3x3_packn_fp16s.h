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

static void conv3x3s1_winograd64_transform_kernel_packn_fp16sa_rvv(const Mat& kernel, Mat& kernel_tm_packn, int inch, int outch, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

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
    kernel_tm_packn.create(inch / packn, 64, outch / packn, (size_t)2u * packn * packn, packn * packn);

    for (int q = 0; q + (packn - 1) < outch; q += packn)
    {
        Mat g0 = kernel_tm_packn.channel(q / packn);

        for (int k = 0; k < 64; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + (packn - 1) < inch; p += packn)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = (__fp16)k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);

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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd64_transform_input_packn_fp16sa_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
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
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 64, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 64, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 64, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
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
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(r0 + packn * 3, vl);
                    vfloat16m1_t _val4 = vle16_v_f16m1(r0 + packn * 4, vl);
                    vfloat16m1_t _val5 = vle16_v_f16m1(r0 + packn * 5, vl);
                    vfloat16m1_t _val6 = vle16_v_f16m1(r0 + packn * 6, vl);
                    vfloat16m1_t _val7 = vle16_v_f16m1(r0 + packn * 7, vl);
                    vsseg8e16_v_f16m1x8(tmpptr, vcreate_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
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
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(r0 + packn * 3, vl);
                    vsseg4e16_v_f16m1x4(tmpptr, vcreate_f16m1x4(_val0, _val1, _val2, _val3), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr += 2;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vsseg2e16_v_f16m1x2(tmpptr, vcreate_f16m1x2(_val0, _val1), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
            for (; i < tiles; i++)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
                    vfloat16m1_t _val = vle16_v_f16m1(r0, vl);
                    vse16_v_f16m1(tmpptr, _val, vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 64, outch, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            __fp16* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum7 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        __fp16 val2 = *r0++;
                        __fp16 val3 = *r0++;
                        __fp16 val4 = *r0++;
                        __fp16 val5 = *r0++;
                        __fp16 val6 = *r0++;
                        __fp16 val7 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);
                        _sum4 = vfmacc_vf_f16m1(_sum4, val4, _w0, vl);
                        _sum5 = vfmacc_vf_f16m1(_sum5, val5, _w0, vl);
                        _sum6 = vfmacc_vf_f16m1(_sum6, val6, _w0, vl);
                        _sum7 = vfmacc_vf_f16m1(_sum7, val7, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);
                    vse16_v_f16m1(output0_tm + packn * 4, _sum4, vl);
                    vse16_v_f16m1(output0_tm + packn * 5, _sum5, vl);
                    vse16_v_f16m1(output0_tm + packn * 6, _sum6, vl);
                    vse16_v_f16m1(output0_tm + packn * 7, _sum7, vl);

                    output0_tm += packn * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        __fp16 val2 = *r0++;
                        __fp16 val3 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);

                    output0_tm += packn * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);

                    output0_tm += packn * 2;
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum = vfmacc_vf_f16m1(_sum, val, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum, vl);

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
        conv3x3s1_winograd64_transform_output_packn_fp16sa_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd42_transform_kernel_packn_fp16sa_rvv(const Mat& kernel, Mat& kernel_tm_packn, int inch, int outch, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

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
    kernel_tm_packn.create(inch / packn, 36, outch / packn, (size_t)2u * packn * packn, packn * packn);

    for (int q = 0; q + (packn - 1) < outch; q += packn)
    {
        Mat g0 = kernel_tm_packn.channel(q / packn);

        for (int k = 0; k < 36; k++)
        {
            __fp16* g00 = g0.row<__fp16>(k);

            for (int p = 0; p + (packn - 1) < inch; p += packn)
            {
                for (int i = 0; i < packn; i++)
                {
                    for (int j = 0; j < packn; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = (__fp16)k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);

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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd42_transform_input_packn_fp16sa_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
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
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 2u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 7 < tiles; i += 8)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
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
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(r0 + packn * 3, vl);
                    vfloat16m1_t _val4 = vle16_v_f16m1(r0 + packn * 4, vl);
                    vfloat16m1_t _val5 = vle16_v_f16m1(r0 + packn * 5, vl);
                    vfloat16m1_t _val6 = vle16_v_f16m1(r0 + packn * 6, vl);
                    vfloat16m1_t _val7 = vle16_v_f16m1(r0 + packn * 7, vl);
                    vsseg8e16_v_f16m1x8(tmpptr, vcreate_f16m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
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
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vfloat16m1_t _val2 = vle16_v_f16m1(r0 + packn * 2, vl);
                    vfloat16m1_t _val3 = vle16_v_f16m1(r0 + packn * 3, vl);
                    vsseg4e16_v_f16m1x4(tmpptr, vcreate_f16m1x4(_val0, _val1, _val2, _val3), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = r0[l];
                        tmpptr[1] = r0[l + packn];
                        tmpptr += 2;
                    }

                    r0 += bottom_blob_tm.cstep * packn;
#else
                    vfloat16m1_t _val0 = vle16_v_f16m1(r0, vl);
                    vfloat16m1_t _val1 = vle16_v_f16m1(r0 + packn, vl);
                    vsseg2e16_v_f16m1x2(tmpptr, vcreate_f16m1x2(_val0, _val1), vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
            for (; i < tiles; i++)
            {
                __fp16* tmpptr = tm2.row<__fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

                const __fp16* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * packn;

                for (int q = 0; q < inch; q++)
                {
                    vfloat16m1_t _val = vle16_v_f16m1(r0, vl);
                    vse16_v_f16m1(tmpptr, _val, vl);

                    r0 += bottom_blob_tm.cstep * packn;
                    tmpptr += packn;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 2u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            __fp16* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum4 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum5 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum6 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum7 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        __fp16 val2 = *r0++;
                        __fp16 val3 = *r0++;
                        __fp16 val4 = *r0++;
                        __fp16 val5 = *r0++;
                        __fp16 val6 = *r0++;
                        __fp16 val7 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);
                        _sum4 = vfmacc_vf_f16m1(_sum4, val4, _w0, vl);
                        _sum5 = vfmacc_vf_f16m1(_sum5, val5, _w0, vl);
                        _sum6 = vfmacc_vf_f16m1(_sum6, val6, _w0, vl);
                        _sum7 = vfmacc_vf_f16m1(_sum7, val7, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);
                    vse16_v_f16m1(output0_tm + packn * 4, _sum4, vl);
                    vse16_v_f16m1(output0_tm + packn * 5, _sum5, vl);
                    vse16_v_f16m1(output0_tm + packn * 6, _sum6, vl);
                    vse16_v_f16m1(output0_tm + packn * 7, _sum7, vl);

                    output0_tm += packn * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum2 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum3 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        __fp16 val2 = *r0++;
                        __fp16 val3 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);
                        _sum2 = vfmacc_vf_f16m1(_sum2, val2, _w0, vl);
                        _sum3 = vfmacc_vf_f16m1(_sum3, val3, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);
                    vse16_v_f16m1(output0_tm + packn * 2, _sum2, vl);
                    vse16_v_f16m1(output0_tm + packn * 3, _sum3, vl);

                    output0_tm += packn * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum0 = vfmv_v_f_f16m1(0.f, vl);
                    vfloat16m1_t _sum1 = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val0 = *r0++;
                        __fp16 val1 = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum0 = vfmacc_vf_f16m1(_sum0, val0, _w0, vl);
                        _sum1 = vfmacc_vf_f16m1(_sum1, val1, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum0, vl);
                    vse16_v_f16m1(output0_tm + packn, _sum1, vl);

                    output0_tm += packn * 2;
                }
                for (; i < tiles; i++)
                {
                    const __fp16* r0 = bb2.row<const __fp16>(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
                    const __fp16* k0 = kernel0_tm.row<const __fp16>(r);

                    int nn = inch * packn; // inch always > 0

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        __fp16 val = *r0++;
                        vfloat16m1_t _w0 = vle16_v_f16m1(k0, vl);
                        _sum = vfmacc_vf_f16m1(_sum, val, _w0, vl);

                        k0 += packn;
                    }

                    vse16_v_f16m1(output0_tm, _sum, vl);

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
        conv3x3s1_winograd42_transform_output_packn_fp16sa_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
