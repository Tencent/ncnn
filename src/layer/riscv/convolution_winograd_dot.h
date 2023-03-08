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

static void convolution_winograd_dot_rvv(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);
#endif

    // Mat bottom_blob_tm(tiles, 16/36/64, inch, 4u, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.w;
    const int batch = bottom_blob_tm.h;
    const int inch = bottom_blob_tm.c;

    // permute
    Mat bottom_blob_tm2;
#if __riscv_vector
    if (tiles >= packn)
        bottom_blob_tm2.create(packn * inch, tiles / packn + tiles % packn, batch, 4u, opt.workspace_allocator);
#else
    if (tiles >= 4)
        bottom_blob_tm2.create(4 * inch, tiles / 4 + tiles % 4, batch, 4u, opt.workspace_allocator);
#endif
    else
        bottom_blob_tm2.create(1 * inch, tiles, batch, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r = 0; r < batch; r++)
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

    top_blob_tm.create(tiles, batch, outch, 4u, opt.workspace_allocator);

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

        for (int r = 0; r < batch; r++)
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

        for (int r = 0; r < batch; r++)
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

        for (int r = 0; r < batch; r++)
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
