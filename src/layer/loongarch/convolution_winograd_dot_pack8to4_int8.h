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

static void convolution_winograd_dot_pack8to4_int8_lsx(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 8, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 8, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 8, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
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
                __m128i _r0 = __lsx_vld(r0, 0);
                __m128i _r1 = __lsx_vld(r0 + 8, 0);
                __lsx_vst(_r0, tmpptr, 0);
                __lsx_vst(_r1, tmpptr + 8, 0);
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
                __m128i _r0 = __lsx_vld(r0, 0);
                __lsx_vst(_r0, tmpptr, 0);
                r0 += bottom_blob_tm.cstep * 8;
                tmpptr += 8;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 16u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 1 < tiles; i += 2)
            {
                const short* r0 = bb2.row<const short>(i / 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn = inch; // inch always > 0

                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 64);
                    __builtin_prefetch(k0 + 128);
                    __m128i _w0 = __lsx_vld(k0, 0);
                    __m128i _w1 = __lsx_vld(k0 + 8, 0);
                    __m128i _w2 = __lsx_vld(k0 + 16, 0);
                    __m128i _w3 = __lsx_vld(k0 + 24, 0);

                    __m128i _extw0 = __lsx_vslti_h(_w0, 0);
                    __m128i _extw1 = __lsx_vslti_h(_w1, 0);
                    __m128i _extw2 = __lsx_vslti_h(_w2, 0);
                    __m128i _extw3 = __lsx_vslti_h(_w3, 0);

                    __m128i _w0l = __lsx_vilvl_h(_extw0, _w0);
                    __m128i _w0h = __lsx_vilvh_h(_extw0, _w0);
                    __m128i _w1l = __lsx_vilvl_h(_extw1, _w1);
                    __m128i _w1h = __lsx_vilvh_h(_extw1, _w1);
                    __m128i _w2l = __lsx_vilvl_h(_extw2, _w2);
                    __m128i _w2h = __lsx_vilvh_h(_extw2, _w2);
                    __m128i _w3l = __lsx_vilvl_h(_extw3, _w3);
                    __m128i _w3h = __lsx_vilvh_h(_extw3, _w3);

                    __m128i _val0_0 = __lsx_vreplgr2vr_w(r0[0]);
                    __m128i _val0_1 = __lsx_vreplgr2vr_w(r0[1]);
                    __m128i _val0_2 = __lsx_vreplgr2vr_w(r0[2]);
                    __m128i _val0_3 = __lsx_vreplgr2vr_w(r0[3]);
                    __m128i _val0_4 = __lsx_vreplgr2vr_w(r0[4]);
                    __m128i _val0_5 = __lsx_vreplgr2vr_w(r0[5]);
                    __m128i _val0_6 = __lsx_vreplgr2vr_w(r0[6]);
                    __m128i _val0_7 = __lsx_vreplgr2vr_w(r0[7]);
                    __m128i _val1_0 = __lsx_vreplgr2vr_w(r0[8]);
                    __m128i _val1_1 = __lsx_vreplgr2vr_w(r0[9]);
                    __m128i _val1_2 = __lsx_vreplgr2vr_w(r0[10]);
                    __m128i _val1_3 = __lsx_vreplgr2vr_w(r0[11]);
                    __m128i _val1_4 = __lsx_vreplgr2vr_w(r0[12]);
                    __m128i _val1_5 = __lsx_vreplgr2vr_w(r0[13]);
                    __m128i _val1_6 = __lsx_vreplgr2vr_w(r0[14]);
                    __m128i _val1_7 = __lsx_vreplgr2vr_w(r0[15]);

                    _sum0 = __lsx_vmadd_w(_sum0, _w0l, _val0_0);
                    _sum1 = __lsx_vmadd_w(_sum1, _w0h, _val0_1);
                    _sum2 = __lsx_vmadd_w(_sum2, _w0l, _val1_0);
                    _sum3 = __lsx_vmadd_w(_sum3, _w0h, _val1_1);
                    _sum0 = __lsx_vmadd_w(_sum0, _w1l, _val0_2);
                    _sum1 = __lsx_vmadd_w(_sum1, _w1h, _val0_3);
                    _sum2 = __lsx_vmadd_w(_sum2, _w1l, _val1_2);
                    _sum3 = __lsx_vmadd_w(_sum3, _w1h, _val1_3);
                    _sum0 = __lsx_vmadd_w(_sum0, _w2l, _val0_4);
                    _sum1 = __lsx_vmadd_w(_sum1, _w2h, _val0_5);
                    _sum2 = __lsx_vmadd_w(_sum2, _w2l, _val1_4);
                    _sum3 = __lsx_vmadd_w(_sum3, _w2h, _val1_5);
                    _sum0 = __lsx_vmadd_w(_sum0, _w3l, _val0_6);
                    _sum1 = __lsx_vmadd_w(_sum1, _w3h, _val0_7);
                    _sum2 = __lsx_vmadd_w(_sum2, _w3l, _val1_6);
                    _sum3 = __lsx_vmadd_w(_sum3, _w3h, _val1_7);

                    r0 += 16;
                    k0 += 32;
                }

                _sum0 = __lsx_vadd_w(_sum0, _sum1);
                _sum2 = __lsx_vadd_w(_sum2, _sum3);

                __lsx_vst(_sum0, output0_tm, 0);
                __lsx_vst(_sum2, output0_tm + 4, 0);

                output0_tm += 8;
            }
            for (; i < tiles; i++)
            {
                const short* r0 = bb2.row<const short>(i / 2 + i % 2);
                const short* k0 = kernel0_tm.row<const short>(r);

                int nn = inch; // inch always > 0

                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 32);
                    __builtin_prefetch(k0 + 128);
                    __m128i _w0 = __lsx_vld(k0, 0);
                    __m128i _w1 = __lsx_vld(k0 + 8, 0);
                    __m128i _w2 = __lsx_vld(k0 + 16, 0);
                    __m128i _w3 = __lsx_vld(k0 + 24, 0);

                    __m128i _extw0 = __lsx_vslti_h(_w0, 0);
                    __m128i _extw1 = __lsx_vslti_h(_w1, 0);
                    __m128i _extw2 = __lsx_vslti_h(_w2, 0);
                    __m128i _extw3 = __lsx_vslti_h(_w3, 0);

                    __m128i _w0l = __lsx_vilvl_h(_extw0, _w0);
                    __m128i _w0h = __lsx_vilvh_h(_extw0, _w0);
                    __m128i _w1l = __lsx_vilvl_h(_extw1, _w1);
                    __m128i _w1h = __lsx_vilvh_h(_extw1, _w1);
                    __m128i _w2l = __lsx_vilvl_h(_extw2, _w2);
                    __m128i _w2h = __lsx_vilvh_h(_extw2, _w2);
                    __m128i _w3l = __lsx_vilvl_h(_extw3, _w3);
                    __m128i _w3h = __lsx_vilvh_h(_extw3, _w3);

                    __m128i _val0 = __lsx_vreplgr2vr_w(r0[0]);
                    __m128i _val1 = __lsx_vreplgr2vr_w(r0[1]);
                    __m128i _val2 = __lsx_vreplgr2vr_w(r0[2]);
                    __m128i _val3 = __lsx_vreplgr2vr_w(r0[3]);
                    __m128i _val4 = __lsx_vreplgr2vr_w(r0[4]);
                    __m128i _val5 = __lsx_vreplgr2vr_w(r0[5]);
                    __m128i _val6 = __lsx_vreplgr2vr_w(r0[6]);
                    __m128i _val7 = __lsx_vreplgr2vr_w(r0[7]);

                    _sum0 = __lsx_vmadd_w(_sum0, _w0l, _val0);
                    _sum1 = __lsx_vmadd_w(_sum1, _w0h, _val1);
                    _sum0 = __lsx_vmadd_w(_sum0, _w1l, _val2);
                    _sum1 = __lsx_vmadd_w(_sum1, _w1h, _val3);
                    _sum0 = __lsx_vmadd_w(_sum0, _w2l, _val4);
                    _sum1 = __lsx_vmadd_w(_sum1, _w2h, _val5);
                    _sum0 = __lsx_vmadd_w(_sum0, _w3l, _val6);
                    _sum1 = __lsx_vmadd_w(_sum1, _w3h, _val7);

                    r0 += 8;
                    k0 += 32;
                }

                _sum0 = __lsx_vadd_w(_sum0, _sum1);

                __lsx_vst(_sum0, output0_tm, 0);
                output0_tm += 4;
            }
        }
    }
}
