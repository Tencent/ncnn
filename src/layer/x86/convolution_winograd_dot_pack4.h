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

static void convolution_winograd_dot_pack4_sse(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 16u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 12)
        bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 8)
        bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else if (tiles >= 2)
        bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, batch, 16u, 4, opt.workspace_allocator);
    else // if (tiles >= 1)
        bottom_blob_tm2.create(1 * inch, tiles, batch, 16u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
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

    top_blob_tm.create(tiles, batch, outch, 16u, 4, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* output0_tm = top_blob_tm.channel(p);

        const Mat kernel0_tm = kernel_tm.channel(p);

        for (int r = 0; r < batch; r++)
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
