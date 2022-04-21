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

static void conv3x3s1_winograd64_pack4to1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd64_transform_input_pack4_sse(bottom_blob_bordered, bottom_blob_tm, opt);
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
        conv3x3s1_winograd64_transform_output_sse(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
