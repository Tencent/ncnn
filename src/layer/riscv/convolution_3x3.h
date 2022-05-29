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

static void conv3x3s1_winograd23_transform_kernel_rvv(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
{
    Mat kernel_tm(4 * 4, inch, outch);

    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
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
            float tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 16-inch-outch
    // dst = inch-16-outch
#if __riscv_vector
    kernel_tm2.create(8 * inch, 16, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(2 * inch, 16, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __riscv_vector
    for (; q + 1 < outch; q += 2)
    {
        Mat g0 = kernel_tm2.channel(q / 2);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
#if __riscv_vector
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q / 2 + q % 2);
#endif

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd23_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        const int tiles = h_tm / 4 * w_tm / 4;

        // permute
        Mat bottom_blob_tm2;
#if __riscv_vector
        if (tiles >= packn)
            bottom_blob_tm2.create(packn * inch, tiles / packn + tiles % packn, 16, 4u, opt.workspace_allocator);
#else
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 16, 4u, opt.workspace_allocator);
#endif
        else
            bottom_blob_tm2.create(1 * inch, tiles, 16, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 16; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __riscv_vector
            for (; i + (packn - 1) < tiles; i += packn)
            {
                float* tmpptr = tm2.row(i / packn);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    vse32_v_f32m1(tmpptr, vle32_v_f32m1(r0, vl), vl);
                    r0 += bottom_blob_tm.cstep;
                    tmpptr += packn;
                }
            }
#else  // __riscv_vector
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row(i / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    tmpptr[0] = r0[0];
                    tmpptr[1] = r0[1];
                    tmpptr[2] = r0[2];
                    tmpptr[3] = r0[3];

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 4;
                }
            }
#endif // __riscv_vector
            for (; i < tiles; i++)
            {
#if __riscv_vector
                float* tmpptr = tm2.row(i / packn + i % packn);
#else
                float* tmpptr = tm2.row(i / 4 + i % 4);
#endif

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

        top_blob_tm.create(tiles, 16, outch, 4u, opt.workspace_allocator);

#if __riscv_vector
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

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

                    int j = 0;
                    for (; j < nn; j++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], _val, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, k0[1], _val, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, k0[2], _val, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, k0[3], _val, vl);
                        _sum4 = vfmacc_vf_f32m1(_sum4, k0[4], _val, vl);
                        _sum5 = vfmacc_vf_f32m1(_sum5, k0[5], _val, vl);
                        _sum6 = vfmacc_vf_f32m1(_sum6, k0[6], _val, vl);
                        _sum7 = vfmacc_vf_f32m1(_sum7, k0[7], _val, vl);
                        r0 += packn;
                        k0 += 8;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output1_tm, _sum1, vl);
                    vse32_v_f32m1(output2_tm, _sum2, vl);
                    vse32_v_f32m1(output3_tm, _sum3, vl);
                    vse32_v_f32m1(output4_tm, _sum4, vl);
                    vse32_v_f32m1(output5_tm, _sum5, vl);
                    vse32_v_f32m1(output6_tm, _sum6, vl);
                    vse32_v_f32m1(output7_tm, _sum7, vl);

                    output0_tm += packn;
                    output1_tm += packn;
                    output2_tm += packn;
                    output3_tm += packn;
                    output4_tm += packn;
                    output5_tm += packn;
                    output6_tm += packn;
                    output7_tm += packn;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / packn + i % packn);
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

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

                    int j = 0;
                    for (; j < nn; j++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], _val, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, k0[1], _val, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, k0[2], _val, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, k0[3], _val, vl);
                        r0 += packn;
                        k0 += 4;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output1_tm, _sum1, vl);
                    vse32_v_f32m1(output2_tm, _sum2, vl);
                    vse32_v_f32m1(output3_tm, _sum3, vl);

                    output0_tm += packn;
                    output1_tm += packn;
                    output2_tm += packn;
                    output3_tm += packn;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / packn + i % packn);
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

            for (int r = 0; r < 16; r++)
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

#if __riscv_vector
            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

            for (int r = 0; r < 16; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __riscv_vector
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], vle32_v_f32m1(r0, vl), vl);
                        r0 += packn;
                        k0++;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    output0_tm += packn;
                }
#else  // __riscv_vector
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int j = 0; j < nn; j++)
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
                }
#endif // __riscv_vector
                for (; i < tiles; i++)
                {
#if __riscv_vector
                    const float* r0 = bb2.row(i / packn + i % packn);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum = 0.f;

                    for (int j = 0; j < nn; j++)
                    {
                        sum += r0[0] * k0[0];
                        r0 += 1;
                        k0 += 1;
                    }

                    output0_tm[0] = sum;
                    output0_tm += 1;
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
        top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd23_transform_output_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_rvv(const Mat& kernel, Mat& kernel_tm2, int inch, int outch, const Option& opt)
{
    Mat kernel_tm(6 * 6, inch, outch);

    // G
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
    // dst = inch-36-outch
#if __riscv_vector
    kernel_tm2.create(8 * inch, 36, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm2.create(2 * inch, 36, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm2.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#else  // __riscv_vector
    for (; q + 1 < outch; q += 2)
    {
        Mat g0 = kernel_tm2.channel(q / 2);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const float* k00 = kernel_tm.channel(q + i).row(p);
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
#if __riscv_vector
        Mat g0 = kernel_tm2.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        Mat g0 = kernel_tm2.channel(q / 2 + q % 2);
#endif

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm.channel(q).row(p);
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd43_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2, winograd F(4,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;

    Option opt_b = opt;
    opt_b.blob_allocator = opt.workspace_allocator;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt_b);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, 4u, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_rvv(bottom_blob_bordered, bottom_blob_tm, opt);
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
        Mat bottom_blob_tm2;
#if __riscv_vector
        if (tiles >= packn)
            bottom_blob_tm2.create(packn * inch, tiles / packn + tiles % packn, 36, 4u, opt.workspace_allocator);
#else
        if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, 36, 4u, opt.workspace_allocator);
#endif
        else
            bottom_blob_tm2.create(1 * inch, tiles, 36, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
#if __riscv_vector
            for (; i + (packn - 1) < tiles; i += packn)
            {
                float* tmpptr = tm2.row(i / packn);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    vse32_v_f32m1(tmpptr, vle32_v_f32m1(r0, vl), vl);
                    r0 += bottom_blob_tm.cstep;
                    tmpptr += packn;
                }
            }
#else  // __riscv_vector
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row(i / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    tmpptr[0] = r0[0];
                    tmpptr[1] = r0[1];
                    tmpptr[2] = r0[2];
                    tmpptr[3] = r0[3];

                    r0 += bottom_blob_tm.cstep;
                    tmpptr += 4;
                }
            }
#endif // __riscv_vector
            for (; i < tiles; i++)
            {
#if __riscv_vector
                float* tmpptr = tm2.row(i / packn + i % packn);
#else
                float* tmpptr = tm2.row(i / 4 + i % 4);
#endif

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

        top_blob_tm.create(tiles, 36, outch, 4u, opt.workspace_allocator);

#if __riscv_vector
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

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum4 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum5 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum6 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum7 = vfmv_v_f_f32m1(0.f, vl);

                    int j = 0;
                    for (; j < nn; j++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], _val, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, k0[1], _val, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, k0[2], _val, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, k0[3], _val, vl);
                        _sum4 = vfmacc_vf_f32m1(_sum4, k0[4], _val, vl);
                        _sum5 = vfmacc_vf_f32m1(_sum5, k0[5], _val, vl);
                        _sum6 = vfmacc_vf_f32m1(_sum6, k0[6], _val, vl);
                        _sum7 = vfmacc_vf_f32m1(_sum7, k0[7], _val, vl);
                        r0 += packn;
                        k0 += 8;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output1_tm, _sum1, vl);
                    vse32_v_f32m1(output2_tm, _sum2, vl);
                    vse32_v_f32m1(output3_tm, _sum3, vl);
                    vse32_v_f32m1(output4_tm, _sum4, vl);
                    vse32_v_f32m1(output5_tm, _sum5, vl);
                    vse32_v_f32m1(output6_tm, _sum6, vl);
                    vse32_v_f32m1(output7_tm, _sum7, vl);

                    output0_tm += packn;
                    output1_tm += packn;
                    output2_tm += packn;
                    output3_tm += packn;
                    output4_tm += packn;
                    output5_tm += packn;
                    output6_tm += packn;
                    output7_tm += packn;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / packn + i % packn);
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

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum1 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum2 = vfmv_v_f_f32m1(0.f, vl);
                    vfloat32m1_t _sum3 = vfmv_v_f_f32m1(0.f, vl);

                    int j = 0;
                    for (; j < nn; j++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(r0, vl);
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], _val, vl);
                        _sum1 = vfmacc_vf_f32m1(_sum1, k0[1], _val, vl);
                        _sum2 = vfmacc_vf_f32m1(_sum2, k0[2], _val, vl);
                        _sum3 = vfmacc_vf_f32m1(_sum3, k0[3], _val, vl);
                        r0 += packn;
                        k0 += 4;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    vse32_v_f32m1(output1_tm, _sum1, vl);
                    vse32_v_f32m1(output2_tm, _sum2, vl);
                    vse32_v_f32m1(output3_tm, _sum3, vl);

                    output0_tm += packn;
                    output1_tm += packn;
                    output2_tm += packn;
                    output3_tm += packn;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / packn + i % packn);
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

            for (int r = 0; r < 36; r++)
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

#if __riscv_vector
            const Mat kernel0_tm = kernel_tm.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);
#endif

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
#if __riscv_vector
                for (; i + (packn - 1) < tiles; i += packn)
                {
                    const float* r0 = bb2.row(i / packn);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    vfloat32m1_t _sum0 = vfmv_v_f_f32m1(0.f, vl);

                    for (int j = 0; j < nn; j++)
                    {
                        _sum0 = vfmacc_vf_f32m1(_sum0, k0[0], vle32_v_f32m1(r0, vl), vl);
                        r0 += packn;
                        k0++;
                    }

                    vse32_v_f32m1(output0_tm, _sum0, vl);
                    output0_tm += packn;
                }
#else  // __riscv_vector
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum2 = 0.f;
                    float sum3 = 0.f;

                    for (int j = 0; j < nn; j++)
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
                }
#endif // __riscv_vector
                for (; i < tiles; i++)
                {
#if __riscv_vector
                    const float* r0 = bb2.row(i / packn + i % packn);
#else
                    const float* r0 = bb2.row(i / 4 + i % 4);
#endif
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0

                    float sum = 0.f;

                    for (int j = 0; j < nn; j++)
                    {
                        sum += r0[0] * k0[0];
                        r0 += 1;
                        k0 += 1;
                    }

                    output0_tm[0] = sum;
                    output0_tm += 1;
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
        top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    }
    {
        conv3x3s1_winograd43_transform_output_rvv(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
