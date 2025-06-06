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

static void convolution_winograd_dot_msa(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 4u, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
    if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 4u, opt.workspace_allocator);
    else
        bottom_blob_tm2.create(1 * inch, tiles, batch, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
    {
        Mat tm2 = bottom_blob_tm2.channel(r);

        // tile
        int i = 0;
        for (; i + 3 < tiles; i += 4)
        {
            float* tmpptr = tm2.row(i / 4);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i);

            for (int q = 0; q < inch; q++)
            {
#if __mips_msa
                __msa_st_w(__msa_ld_w(r0, 0), tmpptr, 0);
#else
                tmpptr[0] = r0[0];
                tmpptr[1] = r0[1];
                tmpptr[2] = r0[2];
                tmpptr[3] = r0[3];
#endif

                r0 += bottom_blob_tm.cstep;
                tmpptr += 4;
            }
        }
        for (; i < tiles; i++)
        {
            float* tmpptr = tm2.row(i / 4 + i % 4);

            const float* r0 = bottom_blob_tm;

            r0 += (r * tiles + i);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = r0[0];

                r0 += bottom_blob_tm.cstep;
                tmpptr += 1;
            }
        }
    }

    bottom_blob_tm = Mat();
    // permute end

    top_blob_tm.create(tiles, batch, outch, 4u, opt.workspace_allocator);

#if __mips_msa
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        float* output0_tm = top_blob_tm.channel(p);
        float* output1_tm = top_blob_tm.channel(p + 1);
        float* output2_tm = top_blob_tm.channel(p + 2);
        float* output3_tm = top_blob_tm.channel(p + 3);
        float* output4_tm = top_blob_tm.channel(p + 4);
        float* output5_tm = top_blob_tm.channel(p + 5);
        float* output6_tm = top_blob_tm.channel(p + 6);
        float* output7_tm = top_blob_tm.channel(p + 7);

        const Mat kernel0_tm = kernel_tm.channel(p / 8);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);
                v4f32 _sum4 = (v4f32)__msa_fill_w(0);
                v4f32 _sum5 = (v4f32)__msa_fill_w(0);
                v4f32 _sum6 = (v4f32)__msa_fill_w(0);
                v4f32 _sum7 = (v4f32)__msa_fill_w(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 32);
                    v4f32 _val = (v4f32)__msa_ld_w(r0, 0);
                    v4i32 _w0123 = __msa_ld_w(k0, 0);
                    v4i32 _w4567 = __msa_ld_w(k0 + 4, 0);
                    _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w0123, 0));
                    _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w0123, 1));
                    _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w0123, 2));
                    _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w0123, 3));
                    _sum4 = __msa_fmadd_w(_sum4, _val, (v4f32)__msa_splati_w(_w4567, 0));
                    _sum5 = __msa_fmadd_w(_sum5, _val, (v4f32)__msa_splati_w(_w4567, 1));
                    _sum6 = __msa_fmadd_w(_sum6, _val, (v4f32)__msa_splati_w(_w4567, 2));
                    _sum7 = __msa_fmadd_w(_sum7, _val, (v4f32)__msa_splati_w(_w4567, 3));

                    r0 += 4;
                    k0 += 8;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output1_tm, 0);
                __msa_st_w((v4i32)_sum2, output2_tm, 0);
                __msa_st_w((v4i32)_sum3, output3_tm, 0);
                __msa_st_w((v4i32)_sum4, output4_tm, 0);
                __msa_st_w((v4i32)_sum5, output5_tm, 0);
                __msa_st_w((v4i32)_sum6, output6_tm, 0);
                __msa_st_w((v4i32)_sum7, output7_tm, 0);

                output0_tm += 4;
                output1_tm += 4;
                output2_tm += 4;
                output3_tm += 4;
                output4_tm += 4;
                output5_tm += 4;
                output6_tm += 4;
                output7_tm += 4;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 4 + i % 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;
                float sum4 = 0.f;
                float sum5 = 0.f;
                float sum6 = 0.f;
                float sum7 = 0.f;

                int j = 0;
                for (; j < nn; j++)
                {
                    sum0 += r0[0] * k0[0];
                    sum1 += r0[0] * k0[1];
                    sum2 += r0[0] * k0[2];
                    sum3 += r0[0] * k0[3];
                    sum4 += r0[0] * k0[4];
                    sum5 += r0[0] * k0[5];
                    sum6 += r0[0] * k0[6];
                    sum7 += r0[0] * k0[7];

                    r0 += 1;
                    k0 += 8;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output2_tm[0] = sum2;
                output3_tm[0] = sum3;
                output4_tm[0] = sum4;
                output5_tm[0] = sum5;
                output6_tm[0] = sum6;
                output7_tm[0] = sum7;

                output0_tm++;
                output1_tm++;
                output2_tm++;
                output3_tm++;
                output4_tm++;
                output5_tm++;
                output6_tm++;
                output7_tm++;
            }
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        float* output0_tm = top_blob_tm.channel(p);
        float* output1_tm = top_blob_tm.channel(p + 1);
        float* output2_tm = top_blob_tm.channel(p + 2);
        float* output3_tm = top_blob_tm.channel(p + 3);

        const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 16);
                    v4f32 _val = (v4f32)__msa_ld_w(r0, 0);
                    v4i32 _w0123 = __msa_ld_w(k0, 0);
                    _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w0123, 0));
                    _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w0123, 1));
                    _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w0123, 2));
                    _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w0123, 3));

                    r0 += 4;
                    k0 += 4;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                __msa_st_w((v4i32)_sum1, output1_tm, 0);
                __msa_st_w((v4i32)_sum2, output2_tm, 0);
                __msa_st_w((v4i32)_sum3, output3_tm, 0);

                output0_tm += 4;
                output1_tm += 4;
                output2_tm += 4;
                output3_tm += 4;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 4 + i % 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;

                int j = 0;
                for (; j < nn; j++)
                {
                    sum0 += r0[0] * k0[0];
                    sum1 += r0[0] * k0[1];
                    sum2 += r0[0] * k0[2];
                    sum3 += r0[0] * k0[3];

                    r0 += 1;
                    k0 += 4;
                }

                output0_tm[0] = sum0;
                output1_tm[0] = sum1;
                output2_tm[0] = sum2;
                output3_tm[0] = sum3;

                output0_tm++;
                output1_tm++;
                output2_tm++;
                output3_tm++;
            }
        }
    }

    remain_outch_start += nn_outch << 2;
#else
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        float* output0_tm = top_blob_tm.channel(p);
        float* output1_tm = top_blob_tm.channel(p + 1);

        const Mat kernel0_tm = kernel_tm.channel(p / 2);

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                float sum00 = 0.f;
                float sum01 = 0.f;
                float sum02 = 0.f;
                float sum03 = 0.f;
                float sum10 = 0.f;
                float sum11 = 0.f;
                float sum12 = 0.f;
                float sum13 = 0.f;

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 8);
                    float w0 = k0[0];
                    float w1 = k0[1];
                    sum00 += r0[0] * w0;
                    sum01 += r0[1] * w0;
                    sum02 += r0[2] * w0;
                    sum03 += r0[3] * w0;
                    sum10 += r0[0] * w1;
                    sum11 += r0[1] * w1;
                    sum12 += r0[2] * w1;
                    sum13 += r0[3] * w1;

                    r0 += 4;
                    k0 += 2;
                }

                output0_tm[0] = sum00;
                output0_tm[1] = sum01;
                output0_tm[2] = sum02;
                output0_tm[3] = sum03;
                output1_tm[0] = sum10;
                output1_tm[1] = sum11;
                output1_tm[2] = sum12;
                output1_tm[3] = sum13;

                output0_tm += 4;
                output1_tm += 4;
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 4 + i % 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                float sum00 = 0.f;
                float sum10 = 0.f;

                for (int j = 0; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 4);
                    __builtin_prefetch(k0 + 8);
                    float val0 = r0[0];
                    sum00 += val0 * k0[0];
                    sum10 += val0 * k0[1];

                    r0 += 1;
                    k0 += 2;
                }

                output0_tm[0] = sum00;
                output1_tm[0] = sum10;
                output0_tm++;
                output1_tm++;
            }
        }
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* output0_tm = top_blob_tm.channel(p);

#if __mips_msa
        const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
        const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

        for (int r = 0; r < batch; r++)
        {
            const Mat bb2 = bottom_blob_tm2.channel(r);

            int i = 0;
            for (; i + 3 < tiles; i += 4)
            {
                const float* r0 = bb2.row(i / 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                int j = 0;
#if __mips_msa
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);

                for (; j < nn; j++)
                {
                    _sum0 = __msa_fmadd_w(_sum0, __msa_fill_w_f32(k0[0]), (v4f32)__msa_ld_w(r0, 0));
                    r0 += 4;
                    k0++;
                }

                __msa_st_w((v4i32)_sum0, output0_tm, 0);
                output0_tm += 4;
#else  // __mips_msa
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;

                for (; j < nn; j++)
                {
                    __builtin_prefetch(r0 + 16);
                    __builtin_prefetch(k0 + 4);
                    float w0 = k0[0];
                    sum0 += r0[0] * w0;
                    sum1 += r0[1] * w0;
                    sum2 += r0[2] * w0;
                    sum3 += r0[3] * w0;

                    r0 += 4;
                    k0++;
                }

                output0_tm[0] = sum0;
                output0_tm[1] = sum1;
                output0_tm[2] = sum2;
                output0_tm[3] = sum3;
                output0_tm += 4;
#endif // __mips_msa
            }
            for (; i < tiles; i++)
            {
                const float* r0 = bb2.row(i / 4 + i % 4);
                const float* k0 = kernel0_tm.row(r);

                int nn = inch; // inch always > 0

                float sum = 0.f;

                for (int j = 0; j < nn; j++)
                {
                    float w0 = k0[0];
                    float val0 = r0[0];
                    sum += val0 * w0;

                    r0 += 1;
                    k0 += 1;
                }

                output0_tm[0] = sum;
                output0_tm += 1;
            }
        }
    }
}
