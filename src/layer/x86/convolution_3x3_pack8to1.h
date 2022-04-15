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

static void conv3x3s1_winograd64_pack8to1_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        conv3x3s1_winograd64_transform_input_pack8_avx(bottom_blob_bordered, bottom_blob_tm, opt);
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
        conv3x3s1_winograd64_transform_output_sse(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
