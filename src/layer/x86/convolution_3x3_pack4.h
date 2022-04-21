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

static void conv3x3s1_winograd64_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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
        int tiles = w_tiles * h_tiles;

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
        conv3x3s1_winograd64_transform_output_pack4_sse(top_blob_tm, top_blob_bordered, bias, opt);
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

static void conv3x3s1_winograd42_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd42_transform_input_pack4_sse(bottom_blob_bordered, bottom_blob_tm, opt);
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
        conv3x3s1_winograd42_transform_output_pack4_sse(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
