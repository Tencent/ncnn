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

static void convolution_winograd_dot_pack8to4_int8_msa(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
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
