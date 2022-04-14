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

static void conv3x3s1_pack16to1_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

            __m512 _k00 = _mm512_loadu_ps(k0);
            __m512 _k01 = _mm512_loadu_ps(k0 + 16);
            __m512 _k02 = _mm512_loadu_ps(k0 + 16 * 2);
            __m512 _k10 = _mm512_loadu_ps(k0 + 16 * 3);
            __m512 _k11 = _mm512_loadu_ps(k0 + 16 * 4);
            __m512 _k12 = _mm512_loadu_ps(k0 + 16 * 5);
            __m512 _k20 = _mm512_loadu_ps(k0 + 16 * 6);
            __m512 _k21 = _mm512_loadu_ps(k0 + 16 * 7);
            __m512 _k22 = _mm512_loadu_ps(k0 + 16 * 8);

            int i = 0;

            for (; i < outh; i++)
            {
                const float* r0 = img0.row(i);
                const float* r1 = img0.row(i + 1);
                const float* r2 = img0.row(i + 2);

                int j = 0;
                for (; j < outw; j++)
                {
                    __m512 _r00 = _mm512_loadu_ps(r0);
                    __m512 _r01 = _mm512_loadu_ps(r0 + 16);
                    __m512 _r02 = _mm512_loadu_ps(r0 + 32);

                    __m512 _sum0 = _mm512_mul_ps(_k00, _r00);
                    __m512 _sum1 = _mm512_mul_ps(_k01, _r01);
                    __m512 _sum2 = _mm512_mul_ps(_k02, _r02);

                    __m512 _r10 = _mm512_loadu_ps(r1);
                    __m512 _r11 = _mm512_loadu_ps(r1 + 16);
                    __m512 _r12 = _mm512_loadu_ps(r1 + 32);

                    _sum0 = _mm512_fmadd_ps(_k10, _r10, _sum0);
                    _sum1 = _mm512_fmadd_ps(_k11, _r11, _sum1);
                    _sum2 = _mm512_fmadd_ps(_k12, _r12, _sum2);

                    __m512 _r20 = _mm512_loadu_ps(r2);
                    __m512 _r21 = _mm512_loadu_ps(r2 + 16);
                    __m512 _r22 = _mm512_loadu_ps(r2 + 32);

                    _sum0 = _mm512_fmadd_ps(_k20, _r20, _sum0);
                    _sum1 = _mm512_fmadd_ps(_k21, _r21, _sum1);
                    _sum2 = _mm512_fmadd_ps(_k22, _r22, _sum2);

                    __m512 _sum = _mm512_add_ps(_sum0, _mm512_add_ps(_sum1, _sum2));

                    *outptr0 += _mm512_comp_reduce_add_ps(_sum);
                    outptr0++;
                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
            }

            k0 += 9 * 16;
        }
    }
}

static void conv3x3s1_winograd64_transform_kernel_pack16to1_avx512(const Mat& kernel, Mat& kernel_tm_packed, int inch, int outch, const Option& opt)
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
    // dst = 16a-inch/16a-64-outch;
    kernel_tm_packed.create(16 * inch / 16, 64, outch / 8 + outch % 8, (size_t)4u * 8, 8);

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        Mat g0 = kernel_tm_packed.channel(p / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 15 < inch; q += 16)
            {
                for (int i = 0; i < 16; i++)
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

        Mat g0 = kernel_tm_packed.channel(p / 8 + p % 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int q = 0; q + 15 < inch; q += 16)
            {
                for (int i = 0; i < 16; i++)
                {
                    const float* k00 = k0.row(q + i);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
}

static void conv3x3s1_winograd64_pack16to1_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd64_transform_input_pack16_avx512(bottom_blob_bordered, bottom_blob_tm, opt);
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
        if (tiles >= 16)
            bottom_blob_tm2.create(16 * inch, tiles / 16 + (tiles % 16) / 8 + tiles % 8, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + tiles % 8, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 15 < tiles; i += 16)
            {
                float* tmpptr = tm2.row(i / 16);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 16;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 16x16
                    __m512 _r0 = _mm512_loadu_ps(r0);
                    __m512 _r1 = _mm512_loadu_ps(r0 + 16);
                    __m512 _r2 = _mm512_loadu_ps(r0 + 16 * 2);
                    __m512 _r3 = _mm512_loadu_ps(r0 + 16 * 3);
                    __m512 _r4 = _mm512_loadu_ps(r0 + 16 * 4);
                    __m512 _r5 = _mm512_loadu_ps(r0 + 16 * 5);
                    __m512 _r6 = _mm512_loadu_ps(r0 + 16 * 6);
                    __m512 _r7 = _mm512_loadu_ps(r0 + 16 * 7);
                    __m512 _r8 = _mm512_loadu_ps(r0 + 16 * 8);
                    __m512 _r9 = _mm512_loadu_ps(r0 + 16 * 9);
                    __m512 _ra = _mm512_loadu_ps(r0 + 16 * 10);
                    __m512 _rb = _mm512_loadu_ps(r0 + 16 * 11);
                    __m512 _rc = _mm512_loadu_ps(r0 + 16 * 12);
                    __m512 _rd = _mm512_loadu_ps(r0 + 16 * 13);
                    __m512 _re = _mm512_loadu_ps(r0 + 16 * 14);
                    __m512 _rf = _mm512_loadu_ps(r0 + 16 * 15);

                    transpose16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                    _mm512_storeu_ps(tmpptr, _r0);
                    _mm512_storeu_ps(tmpptr + 16, _r1);
                    _mm512_storeu_ps(tmpptr + 16 * 2, _r2);
                    _mm512_storeu_ps(tmpptr + 16 * 3, _r3);
                    _mm512_storeu_ps(tmpptr + 16 * 4, _r4);
                    _mm512_storeu_ps(tmpptr + 16 * 5, _r5);
                    _mm512_storeu_ps(tmpptr + 16 * 6, _r6);
                    _mm512_storeu_ps(tmpptr + 16 * 7, _r7);
                    _mm512_storeu_ps(tmpptr + 16 * 8, _r8);
                    _mm512_storeu_ps(tmpptr + 16 * 9, _r9);
                    _mm512_storeu_ps(tmpptr + 16 * 10, _ra);
                    _mm512_storeu_ps(tmpptr + 16 * 11, _rb);
                    _mm512_storeu_ps(tmpptr + 16 * 12, _rc);
                    _mm512_storeu_ps(tmpptr + 16 * 13, _rd);
                    _mm512_storeu_ps(tmpptr + 16 * 14, _re);
                    _mm512_storeu_ps(tmpptr + 16 * 15, _rf);

                    tmpptr += 256;
                    r0 += bottom_blob_tm.cstep * 16;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 16 + (i % 16) / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 16;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 16x8
                    __m512 _r0 = _mm512_load_ps(r0);
                    __m512 _r1 = _mm512_load_ps(r0 + 16);
                    __m512 _r2 = _mm512_load_ps(r0 + 16 * 2);
                    __m512 _r3 = _mm512_load_ps(r0 + 16 * 3);
                    __m512 _r4 = _mm512_load_ps(r0 + 16 * 4);
                    __m512 _r5 = _mm512_load_ps(r0 + 16 * 5);
                    __m512 _r6 = _mm512_load_ps(r0 + 16 * 6);
                    __m512 _r7 = _mm512_load_ps(r0 + 16 * 7);

                    __m512 _tmp0 = _mm512_unpacklo_ps(_r0, _r1);
                    __m512 _tmp1 = _mm512_unpackhi_ps(_r0, _r1);
                    __m512 _tmp2 = _mm512_unpacklo_ps(_r2, _r3);
                    __m512 _tmp3 = _mm512_unpackhi_ps(_r2, _r3);
                    __m512 _tmp4 = _mm512_unpacklo_ps(_r4, _r5);
                    __m512 _tmp5 = _mm512_unpackhi_ps(_r4, _r5);
                    __m512 _tmp6 = _mm512_unpacklo_ps(_r6, _r7);
                    __m512 _tmp7 = _mm512_unpackhi_ps(_r6, _r7);

                    __m512 _tmp8 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmp9 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpa = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpb = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpc = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpd = _mm512_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512 _tmpe = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512 _tmpf = _mm512_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

                    _tmp0 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp1 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp2 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp3 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(2, 0, 2, 0));
                    _tmp4 = _mm512_shuffle_f32x4(_tmp8, _tmpc, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp5 = _mm512_shuffle_f32x4(_tmp9, _tmpd, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp6 = _mm512_shuffle_f32x4(_tmpa, _tmpe, _MM_SHUFFLE(3, 1, 3, 1));
                    _tmp7 = _mm512_shuffle_f32x4(_tmpb, _tmpf, _MM_SHUFFLE(3, 1, 3, 1));

                    _r0 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
                    _r1 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
                    _r2 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(2, 0, 2, 0));
                    _r3 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(2, 0, 2, 0));
                    _r4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
                    _r5 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
                    _r6 = _mm512_shuffle_f32x4(_tmp4, _tmp5, _MM_SHUFFLE(3, 1, 3, 1));
                    _r7 = _mm512_shuffle_f32x4(_tmp6, _tmp7, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_store_ps(tmpptr, _r0);
                    _mm512_store_ps(tmpptr + 16, _r1);
                    _mm512_store_ps(tmpptr + 16 * 2, _r2);
                    _mm512_store_ps(tmpptr + 16 * 3, _r3);
                    _mm512_store_ps(tmpptr + 16 * 4, _r4);
                    _mm512_store_ps(tmpptr + 16 * 5, _r5);
                    _mm512_store_ps(tmpptr + 16 * 6, _r6);
                    _mm512_store_ps(tmpptr + 16 * 7, _r7);

                    tmpptr += 128;
                    r0 += bottom_blob_tm.cstep * 16;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 16 + (i % 16) / 8 + i % 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 16;

                for (int q = 0; q < inch; q++)
                {
                    __m512 _val = _mm512_load_ps(r0);
                    _mm512_store_ps(tmpptr, _val);

                    tmpptr += 16;
                    r0 += bottom_blob_tm.cstep * 16;
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
                for (; i + 15 < tiles; i += 16)
                {
                    const float* r0 = bb2.row(i / 16);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 16; // inch always > 0

                    __m512 _sum0 = _mm512_setzero_ps();
                    __m512 _sum1 = _mm512_setzero_ps();
                    __m512 _sum2 = _mm512_setzero_ps();
                    __m512 _sum3 = _mm512_setzero_ps();
                    __m512 _sum4 = _mm512_setzero_ps();
                    __m512 _sum5 = _mm512_setzero_ps();
                    __m512 _sum6 = _mm512_setzero_ps();
                    __m512 _sum7 = _mm512_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m512 _val0 = _mm512_load_ps(r0);

                        __m512 _w0 = _mm512_set1_ps(kptr[0]);
                        __m512 _w1 = _mm512_set1_ps(kptr[1]);
                        _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm512_fmadd_ps(_val0, _w1, _sum1);
                        __m512 _w2 = _mm512_set1_ps(kptr[2]);
                        __m512 _w3 = _mm512_set1_ps(kptr[3]);
                        _sum2 = _mm512_fmadd_ps(_val0, _w2, _sum2);
                        _sum3 = _mm512_fmadd_ps(_val0, _w3, _sum3);
                        __m512 _w4 = _mm512_set1_ps(kptr[4]);
                        __m512 _w5 = _mm512_set1_ps(kptr[5]);
                        _sum4 = _mm512_fmadd_ps(_val0, _w4, _sum4);
                        _sum5 = _mm512_fmadd_ps(_val0, _w5, _sum5);
                        __m512 _w6 = _mm512_set1_ps(kptr[6]);
                        __m512 _w7 = _mm512_set1_ps(kptr[7]);
                        _sum6 = _mm512_fmadd_ps(_val0, _w6, _sum6);
                        _sum7 = _mm512_fmadd_ps(_val0, _w7, _sum7);

                        r0 += 16;
                        kptr += 8;
                    }

                    _mm512_storeu_ps(outptr0_tm, _sum0);
                    _mm512_storeu_ps(outptr1_tm, _sum1);
                    _mm512_storeu_ps(outptr2_tm, _sum2);
                    _mm512_storeu_ps(outptr3_tm, _sum3);
                    _mm512_storeu_ps(outptr4_tm, _sum4);
                    _mm512_storeu_ps(outptr5_tm, _sum5);
                    _mm512_storeu_ps(outptr6_tm, _sum6);
                    _mm512_storeu_ps(outptr7_tm, _sum7);

                    outptr0_tm += 16;
                    outptr1_tm += 16;
                    outptr2_tm += 16;
                    outptr3_tm += 16;
                    outptr4_tm += 16;
                    outptr5_tm += 16;
                    outptr6_tm += 16;
                    outptr7_tm += 16;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 16 + (i % 16) / 8);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 16; // inch always > 0

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
                        _sum0 = _mm256_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_fmadd_ps(_val0, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                        __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                        _sum2 = _mm256_fmadd_ps(_val0, _w2, _sum2);
                        _sum3 = _mm256_fmadd_ps(_val0, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                        __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                        _sum4 = _mm256_fmadd_ps(_val0, _w4, _sum4);
                        _sum5 = _mm256_fmadd_ps(_val0, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                        __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                        _sum6 = _mm256_fmadd_ps(_val0, _w6, _sum6);
                        _sum7 = _mm256_fmadd_ps(_val0, _w7, _sum7);

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
                    const float* r0 = bb2.row(i / 16 + (i % 16) / 8 + i % 8);

                    const float* kptr = kernel01_tm.row(r);

                    int nn = inch * 16; // inch always > 0

                    __m256 _sum = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _w0 = _mm256_load_ps(kptr);
                        _sum = _mm256_fmadd_ps(_val0, _w0, _sum);

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
                for (; i + 15 < tiles; i += 16)
                {
                    const float* r0 = bb2.row(i / 16);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch * 16; // inch always > 0

                    __m512 _sum0 = _mm512_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m512 _val0 = _mm512_load_ps(r0);
                        __m512 _w0 = _mm512_set1_ps(kptr[0]);
                        _sum0 = _mm512_fmadd_ps(_w0, _val0, _sum0);

                        r0 += 16;
                        kptr += 1;
                    }

                    _mm512_storeu_ps(outptr0_tm, _sum0);
                    outptr0_tm += 16;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 16 + (i % 16) / 8);

                    const float* kptr = kernel0_tm.row(r);

                    int nn = inch * 16; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _val0 = _mm256_load_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(kptr);
                        _sum0 = _mm256_fmadd_ps(_w0, _val0, _sum0);

                        r0 += 8;
                        kptr += 1;
                    }

                    _mm256_storeu_ps(outptr0_tm, _sum0);
                    outptr0_tm += 8;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 16 + (i % 16) / 8 + i % 8);

                    const float* kptr = kernel0_tm.row(r);

                    __m512 _sum0 = _mm512_setzero_ps();

                    for (int q = 0; q < inch; q++)
                    {
                        __m512 _val0 = _mm512_load_ps(r0);
                        __m512 _w0 = _mm512_load_ps(kptr);
                        _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 16;
                        kptr += 16;
                    }

                    float sum0 = _mm512_comp_reduce_add_ps(_sum0);

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
        conv3x3s1_winograd64_transform_output_sse(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
