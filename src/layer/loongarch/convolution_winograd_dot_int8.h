// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void convolution_winograd_dot_int8_lsx(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 2u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __loongarch_sx
    if (inch >= 4)
    {
        if (tiles >= 2)
            bottom_blob_tm2.create(inch / 4 + inch % 4, tiles / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(inch / 4 + inch % 4, tiles, batch, 8u, 4, opt.workspace_allocator);
    }
    else
#endif // __loongarch_sx
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
#if __loongarch_sx
            const short* r1 = (const short*)bottom_blob_tm.channel(1) + r * tiles + i;
            const short* r2 = (const short*)bottom_blob_tm.channel(2) + r * tiles + i;
            const short* r3 = (const short*)bottom_blob_tm.channel(3) + r * tiles + i;
            for (; q + 3 < inch; q += 4)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r1[0];
                tmpptr[2] = r2[0];
                tmpptr[3] = r3[0];
                tmpptr[4] = r0[1];
                tmpptr[5] = r1[1];
                tmpptr[6] = r2[1];
                tmpptr[7] = r3[1];
                r0 += bottom_blob_tm.cstep * 4;
                r1 += bottom_blob_tm.cstep * 4;
                r2 += bottom_blob_tm.cstep * 4;
                r3 += bottom_blob_tm.cstep * 4;
                tmpptr += 8;
            }
#endif // __loongarch_sx
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
#if __loongarch_sx
            const short* r1 = (const short*)bottom_blob_tm.channel(1) + r * tiles + i;
            const short* r2 = (const short*)bottom_blob_tm.channel(2) + r * tiles + i;
            const short* r3 = (const short*)bottom_blob_tm.channel(3) + r * tiles + i;
            for (; q + 3 < inch; q += 4)
            {
                tmpptr[0] = r0[0];
                tmpptr[1] = r1[0];
                tmpptr[2] = r2[0];
                tmpptr[3] = r3[0];
                r0 += bottom_blob_tm.cstep * 4;
                r1 += bottom_blob_tm.cstep * 4;
                r2 += bottom_blob_tm.cstep * 4;
                r3 += bottom_blob_tm.cstep * 4;
                tmpptr += 4;
            }
#endif // __loongarch_sx
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

#if __loongarch_sx
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

                int nn4 = inch / 4;
                int nn1 = inch % 4;

                __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                __m128i _sum10 = __lsx_vreplgr2vr_w(0);

                if (nn4 > 0)
                {
                    __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum02 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum03 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum12 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum13 = __lsx_vreplgr2vr_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val01 = __lsx_vld(r0, 0);

                        __m128i _val0 = __lsx_vilvl_d(_val01, _val01);
                        __m128i _val1 = __lsx_vilvh_d(_val01, _val01);

                        __m128i _w0 = __lsx_vld(k0, 0);
                        __m128i _w1 = __lsx_vld(k0 + 8, 0);

                        __m128i _extval0 = __lsx_vslti_h(_val0, 0);
                        __m128i _extval1 = __lsx_vslti_h(_val1, 0);
                        __m128i _extw0 = __lsx_vslti_h(_w0, 0);
                        __m128i _extw1 = __lsx_vslti_h(_w1, 0);

                        __m128i _val0l = __lsx_vilvl_h(_extval0, _val0);
                        __m128i _val0h = __lsx_vilvh_h(_extval0, _val0);
                        __m128i _val1l = __lsx_vilvl_h(_extval1, _val1);
                        __m128i _val1h = __lsx_vilvh_h(_extval1, _val1);

                        __m128i _w0l = __lsx_vilvl_h(_extw0, _w0);
                        __m128i _w0h = __lsx_vilvh_h(_extw0, _w0);
                        __m128i _w1l = __lsx_vilvl_h(_extw1, _w1);
                        __m128i _w1h = __lsx_vilvh_h(_extw1, _w1);

                        _sum00 = __lsx_vmadd_w(_sum00, _val0l, _w0l);
                        _sum01 = __lsx_vmadd_w(_sum01, _val0h, _w0h);
                        _sum02 = __lsx_vmadd_w(_sum02, _val0l, _w1l);
                        _sum03 = __lsx_vmadd_w(_sum03, _val0h, _w1h);
                        _sum10 = __lsx_vmadd_w(_sum10, _val1l, _w0l);
                        _sum11 = __lsx_vmadd_w(_sum11, _val1h, _w0h);
                        _sum12 = __lsx_vmadd_w(_sum12, _val1l, _w1l);
                        _sum13 = __lsx_vmadd_w(_sum13, _val1h, _w1h);

                        r0 += 8;
                        k0 += 16;
                    }

                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = __lsx_vilvl_w(_sum01, _sum00);
                        _tmp1 = __lsx_vilvl_w(_sum03, _sum02);
                        _tmp2 = __lsx_vilvh_w(_sum01, _sum00);
                        _tmp3 = __lsx_vilvh_w(_sum03, _sum02);
                        _sum00 = __lsx_vilvl_d(_tmp1, _tmp0);
                        _sum01 = __lsx_vilvh_d(_tmp1, _tmp0);
                        _sum02 = __lsx_vilvl_d(_tmp3, _tmp2);
                        _sum03 = __lsx_vilvh_d(_tmp3, _tmp2);
                    }
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = __lsx_vilvl_w(_sum11, _sum10);
                        _tmp1 = __lsx_vilvl_w(_sum13, _sum12);
                        _tmp2 = __lsx_vilvh_w(_sum11, _sum10);
                        _tmp3 = __lsx_vilvh_w(_sum13, _sum12);
                        _sum10 = __lsx_vilvl_d(_tmp1, _tmp0);
                        _sum11 = __lsx_vilvh_d(_tmp1, _tmp0);
                        _sum12 = __lsx_vilvl_d(_tmp3, _tmp2);
                        _sum13 = __lsx_vilvh_d(_tmp3, _tmp2);
                    }

                    _sum00 = __lsx_vadd_w(_sum00, _sum01);
                    _sum02 = __lsx_vadd_w(_sum02, _sum03);
                    _sum10 = __lsx_vadd_w(_sum10, _sum11);
                    _sum12 = __lsx_vadd_w(_sum12, _sum13);

                    _sum00 = __lsx_vadd_w(_sum00, _sum02);
                    _sum10 = __lsx_vadd_w(_sum10, _sum12);
                }

                for (int j = 0; j < nn1; j++)
                {
                    __m128i _val0 = __lsx_vreplgr2vr_h(r0[0]);
                    __m128i _val1 = __lsx_vreplgr2vr_h(r0[1]);
                    __m128i _val = __lsx_vilvl_d(_val1, _val0);

                    __m128i _w16 = __lsx_vld(k0, 0);

                    _w16 = __lsx_vilvl_d(_w16, _w16);

                    __m128i _extval = __lsx_vslti_h(_val, 0);
                    __m128i _extw16 = __lsx_vslti_h(_w16, 0);

                    __m128i _vall = __lsx_vilvl_h(_extval, _val);
                    __m128i _valh = __lsx_vilvh_h(_extval, _val);
                    __m128i _w0l = __lsx_vilvl_h(_extw16, _w16);
                    __m128i _w0h = __lsx_vilvh_h(_extw16, _w16);

                    _sum00 = __lsx_vmadd_w(_sum00, _vall, _w0l);
                    _sum10 = __lsx_vmadd_w(_sum10, _valh, _w0h);

                    r0 += 2;
                    k0 += 4;
                }

                int sum[8];
                __lsx_vst(_sum00, sum, 0);
                __lsx_vst(_sum10, sum + 4, 0);

                output0_tm[0] = sum[0];
                output1_tm[0] = sum[1];
                output2_tm[0] = sum[2];
                output3_tm[0] = sum[3];
                output0_tm[1] = sum[4];
                output1_tm[1] = sum[5];
                output2_tm[1] = sum[6];
                output3_tm[1] = sum[7];
                output0_tm += 2;
                output1_tm += 2;
                output2_tm += 2;
                output3_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn4 = inch / 4;
                int nn1 = inch % 4;

                __m128i _sum0 = __lsx_vreplgr2vr_w(0);

                if (nn4 > 0)
                {
                    __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum3 = __lsx_vreplgr2vr_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val16 = __lsx_vld(r0, 0);

                        _val16 = __lsx_vilvl_d(_val16, _val16);

                        __m128i _w0 = __lsx_vld(k0, 0);
                        __m128i _w1 = __lsx_vld(k0 + 8, 0);

                        __m128i _extval16 = __lsx_vslti_h(_val16, 0);
                        __m128i _extw0 = __lsx_vslti_h(_w0, 0);
                        __m128i _extw1 = __lsx_vslti_h(_w1, 0);

                        __m128i _val0l = __lsx_vilvl_h(_extval16, _val16);
                        __m128i _val0h = __lsx_vilvh_h(_extval16, _val16);

                        __m128i _w0l = __lsx_vilvl_h(_extw0, _w0);
                        __m128i _w0h = __lsx_vilvh_h(_extw0, _w0);
                        __m128i _w1l = __lsx_vilvl_h(_extw1, _w1);
                        __m128i _w1h = __lsx_vilvh_h(_extw1, _w1);

                        _sum0 = __lsx_vmadd_w(_sum0, _val0l, _w0l);
                        _sum1 = __lsx_vmadd_w(_sum1, _val0h, _w0h);
                        _sum2 = __lsx_vmadd_w(_sum2, _val0l, _w1l);
                        _sum3 = __lsx_vmadd_w(_sum3, _val0h, _w1h);

                        r0 += 4;
                        k0 += 16;
                    }

                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = __lsx_vilvl_w(_sum1, _sum0);
                        _tmp1 = __lsx_vilvl_w(_sum3, _sum2);
                        _tmp2 = __lsx_vilvh_w(_sum1, _sum0);
                        _tmp3 = __lsx_vilvh_w(_sum3, _sum2);
                        _sum0 = __lsx_vilvl_d(_tmp1, _tmp0);
                        _sum1 = __lsx_vilvh_d(_tmp1, _tmp0);
                        _sum2 = __lsx_vilvl_d(_tmp3, _tmp2);
                        _sum3 = __lsx_vilvh_d(_tmp3, _tmp2);
                    }

                    _sum0 = __lsx_vadd_w(_sum0, _sum1);
                    _sum2 = __lsx_vadd_w(_sum2, _sum3);
                    _sum0 = __lsx_vadd_w(_sum0, _sum2);
                }

                for (int j = 0; j < nn1; j++)
                {
                    __m128i _val = __lsx_vreplgr2vr_w(r0[0]);
                    __m128i _w16 = __lsx_vld(k0, 0);

                    __m128i _extw16 = __lsx_vslti_h(_w16, 0);
                    __m128i _w0l = __lsx_vilvl_h(_extw16, _w16);

                    _sum0 = __lsx_vmadd_w(_sum0, _val, _w0l);

                    r0 += 1;
                    k0 += 4;
                }

                int sum[4];
                __lsx_vst(_sum0, sum, 0);

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
#else // __loongarch_sx
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        int* output0_tm = top_blob_tm.channel(p);
        int* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel0_tm = kernel_tm.channel(p / 2);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;

                int nn1 = inch;

                for (int j = 0; j < nn1; j++)
                {
                    signed short val0 = r0[0];
                    signed short val1 = r0[1];
                    signed short w0 = k0[0];
                    signed short w1 = k0[1];

                    sum00 += val0 * w0;
                    sum01 += val1 * w0;
                    sum10 += val0 * w1;
                    sum11 += val1 * w1;

                    r0 += 2;
                    k0 += 2;
                }

                output0_tm[0] = sum00;
                output0_tm[1] = sum01;
                output1_tm[0] = sum10;
                output1_tm[1] = sum11;
                output0_tm += 2;
                output1_tm += 2;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int sum0 = 0;
                int sum1 = 0;

                int nn1 = inch;

                for (int j = 0; j < nn1; j++)
                {
                    signed short val0 = r0[0];
                    signed short w0 = k0[0];
                    signed short w1 = k0[1];

                    sum0 += val0 * w0;
                    sum1 += val0 * w1;

                    r0 += 1;
                    k0 += 2;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output0_tm += 1;
                output1_tm += 1;
            }
        }
    }
#endif // __loongarch_sx

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

#if __loongarch_sx
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

#if __loongarch_sx
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum1 = __lsx_vreplgr2vr_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val16 = __lsx_vld(r0, 0);

                        __m128i _w16 = __lsx_vld(k0, 0);

                        _w16 = __lsx_vilvl_d(_w16, _w16);

                        __m128i _extval16 = __lsx_vslti_h(_val16, 0);
                        __m128i _extw16 = __lsx_vslti_h(_w16, 0);

                        __m128i _val0l = __lsx_vilvl_h(_extval16, _val16);
                        __m128i _val0h = __lsx_vilvh_h(_extval16, _val16);

                        __m128i _w0l = __lsx_vilvl_h(_extw16, _w16);
                        __m128i _w0h = __lsx_vilvh_h(_extw16, _w16);

                        _sum0 = __lsx_vmadd_w(_sum0, _val0l, _w0l);
                        _sum1 = __lsx_vmadd_w(_sum1, _val0h, _w0h);

                        r0 += 8;
                        k0 += 4;
                    }

                    sum0 = __lsx_reduce_add_w(_sum0);
                    sum1 = __lsx_reduce_add_w(_sum1);
                }
#else  // __loongarch_sx
                int nn1 = inch;
#endif // __loongarch_sx

                for (int q = 0; q < nn1; q++)
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

#if __loongarch_sx
                int nn4 = inch / 4;
                int nn1 = inch % 4;

                if (nn4 > 0)
                {
                    __m128i _sum = __lsx_vreplgr2vr_w(0);

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val16 = __lsx_vld(r0, 0);
                        __m128i _w16 = __lsx_vld(k0, 0);

                        __m128i _extval16 = __lsx_vslti_h(_val16, 0);
                        __m128i _extw16 = __lsx_vslti_h(_w16, 0);

                        __m128i _val0l = __lsx_vilvl_h(_extval16, _val16);
                        __m128i _w0l = __lsx_vilvl_h(_extw16, _w16);

                        _sum = __lsx_vmadd_w(_sum, _val0l, _w0l);

                        r0 += 4;
                        k0 += 4;
                    }

                    sum = __lsx_reduce_add_w(_sum);
                }
#else  // __loongarch_sx
                int nn1 = inch;
#endif // __loongarch_sx

                for (int q = 0; q < nn1; q++)
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
