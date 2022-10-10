// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void im2col_sgemm_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);
#endif

    // Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
#if __riscv_vector
    if (size >= packn)
        tmp.create(packn * maxk, inch, size / packn + size % packn, 4u, 1, opt.workspace_allocator);
#else
    if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
#endif
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
#if __riscv_vector
        int nn_size = size / packn;
        int remain_size_start = nn_size * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * packn;

            float* tmpptr = tmp.channel(i / packn);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    vse32_v_f32m1(tmpptr, vle32_v_f32m1(img0, vl), vl);
                    img0 += size;
                    tmpptr += packn;
                }
            }
        }
#else // __riscv_vector
        int nn_size = size / 4;
        int remain_size_start = nn_size * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            float* tmpptr = tmp.channel(i / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }
#endif // __riscv_vector

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
#if __riscv_vector
            float* tmpptr = tmp.channel(i / packn + i % packn);
#else
            float* tmpptr = tmp.channel(i / 4 + i % 4);
#endif

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }

#if __riscv_vector
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);
        float* outptr4 = top_blob.channel(p + 4);
        float* outptr5 = top_blob.channel(p + 5);
        float* outptr6 = top_blob.channel(p + 6);
        float* outptr7 = top_blob.channel(p + 7);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + (packn - 1) < size; i += packn)
        {
            const float* tmpptr = tmp.channel(i / packn);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(biasptr[0], vl);
            vfloat32m1_t _sum1 = vfmv_v_f_f32m1(biasptr[1], vl);
            vfloat32m1_t _sum2 = vfmv_v_f_f32m1(biasptr[2], vl);
            vfloat32m1_t _sum3 = vfmv_v_f_f32m1(biasptr[3], vl);
            vfloat32m1_t _sum4 = vfmv_v_f_f32m1(biasptr[4], vl);
            vfloat32m1_t _sum5 = vfmv_v_f_f32m1(biasptr[5], vl);
            vfloat32m1_t _sum6 = vfmv_v_f_f32m1(biasptr[6], vl);
            vfloat32m1_t _sum7 = vfmv_v_f_f32m1(biasptr[7], vl);

            for (int q = 0; q < nn; q++)
            {
                vfloat32m1_t _val = vle32_v_f32m1(tmpptr, vl);
                _sum0 = vfmacc_vf_f32m1(_sum0, kptr[0], _val, vl);
                _sum1 = vfmacc_vf_f32m1(_sum1, kptr[1], _val, vl);
                _sum2 = vfmacc_vf_f32m1(_sum2, kptr[2], _val, vl);
                _sum3 = vfmacc_vf_f32m1(_sum3, kptr[3], _val, vl);
                _sum4 = vfmacc_vf_f32m1(_sum4, kptr[4], _val, vl);
                _sum5 = vfmacc_vf_f32m1(_sum5, kptr[5], _val, vl);
                _sum6 = vfmacc_vf_f32m1(_sum6, kptr[6], _val, vl);
                _sum7 = vfmacc_vf_f32m1(_sum7, kptr[7], _val, vl);
                tmpptr += packn;
                kptr += 8;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);
            vse32_v_f32m1(outptr1, _sum1, vl);
            vse32_v_f32m1(outptr2, _sum2, vl);
            vse32_v_f32m1(outptr3, _sum3, vl);
            vse32_v_f32m1(outptr4, _sum4, vl);
            vse32_v_f32m1(outptr5, _sum5, vl);
            vse32_v_f32m1(outptr6, _sum6, vl);
            vse32_v_f32m1(outptr7, _sum7, vl);

            outptr0 += packn;
            outptr1 += packn;
            outptr2 += packn;
            outptr3 += packn;
            outptr4 += packn;
            outptr5 += packn;
            outptr6 += packn;
            outptr7 += packn;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / packn + i % packn);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            float sum0 = biasptr[0];
            float sum1 = biasptr[1];
            float sum2 = biasptr[2];
            float sum3 = biasptr[3];
            float sum4 = biasptr[4];
            float sum5 = biasptr[5];
            float sum6 = biasptr[6];
            float sum7 = biasptr[7];

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];
                sum4 += tmpptr[0] * kptr[4];
                sum5 += tmpptr[0] * kptr[5];
                sum6 += tmpptr[0] * kptr[6];
                sum7 += tmpptr[0] * kptr[7];
                tmpptr++;
                kptr += 8;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;
            outptr4[0] = sum4;
            outptr5[0] = sum5;
            outptr6[0] = sum6;
            outptr7[0] = sum7;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
        }
    }

    nn_outch = (outch - remain_outch_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);
        float* outptr2 = top_blob.channel(p + 2);
        float* outptr3 = top_blob.channel(p + 3);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + (packn - 1) < size; i += packn)
        {
            const float* tmpptr = tmp.channel(i / packn);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(biasptr[0], vl);
            vfloat32m1_t _sum1 = vfmv_v_f_f32m1(biasptr[1], vl);
            vfloat32m1_t _sum2 = vfmv_v_f_f32m1(biasptr[2], vl);
            vfloat32m1_t _sum3 = vfmv_v_f_f32m1(biasptr[3], vl);

            for (int q = 0; q < nn; q++)
            {
                vfloat32m1_t _val = vle32_v_f32m1(tmpptr, vl);
                _sum0 = vfmacc_vf_f32m1(_sum0, kptr[0], _val, vl);
                _sum1 = vfmacc_vf_f32m1(_sum1, kptr[1], _val, vl);
                _sum2 = vfmacc_vf_f32m1(_sum2, kptr[2], _val, vl);
                _sum3 = vfmacc_vf_f32m1(_sum3, kptr[3], _val, vl);
                tmpptr += packn;
                kptr += 4;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);
            vse32_v_f32m1(outptr1, _sum1, vl);
            vse32_v_f32m1(outptr2, _sum2, vl);
            vse32_v_f32m1(outptr3, _sum3, vl);

            outptr0 += packn;
            outptr1 += packn;
            outptr2 += packn;
            outptr3 += packn;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / packn + i % packn);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            float sum0 = biasptr[0];
            float sum1 = biasptr[1];
            float sum2 = biasptr[2];
            float sum3 = biasptr[3];

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                sum2 += tmpptr[0] * kptr[2];
                sum3 += tmpptr[0] * kptr[3];
                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
        }
    }

    remain_outch_start += nn_outch << 2;
#else // __riscv_vector
    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        const float zeros[2] = {0.f, 0.f};
        const float* biasptr = bias ? bias + p : zeros;

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 4);
            const float* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float sum00 = biasptr[0];
            float sum01 = biasptr[0];
            float sum02 = biasptr[0];
            float sum03 = biasptr[0];
            float sum10 = biasptr[1];
            float sum11 = biasptr[1];
            float sum12 = biasptr[1];
            float sum13 = biasptr[1];

            for (int q = 0; q < nn; q++)
            {
                float k0 = kptr[0];
                float k1 = kptr[1];
                sum00 += tmpptr[0] * k0;
                sum01 += tmpptr[1] * k0;
                sum02 += tmpptr[2] * k0;
                sum03 += tmpptr[3] * k0;
                sum10 += tmpptr[0] * k1;
                sum11 += tmpptr[1] * k1;
                sum12 += tmpptr[2] * k1;
                sum13 += tmpptr[3] * k1;
                tmpptr += 4;
                kptr += 2;
            }

            outptr0[0] = sum00;
            outptr0[1] = sum01;
            outptr0[2] = sum02;
            outptr0[3] = sum03;
            outptr1[0] = sum10;
            outptr1[1] = sum11;
            outptr1[2] = sum12;
            outptr1[3] = sum13;

            outptr0 += 4;
            outptr1 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
            const float* kptr = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float sum0 = biasptr[0];
            float sum1 = biasptr[1];

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[0] * kptr[1];
                tmpptr++;
                kptr += 2;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;

            outptr0++;
            outptr1++;
        }
    }
#endif // __riscv_vector

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
#if __riscv_vector
        for (; i + (packn - 1) < size; i += packn)
        {
            const float* tmpptr = tmp.channel(i / packn);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);

            int nn = inch * maxk; // inch always > 0

            vfloat32m1_t _sum0 = vfmv_v_f_f32m1(bias0, vl);

            for (int q = 0; q < nn; q++)
            {
                _sum0 = vfmacc_vf_f32m1(_sum0, kptr[0], vle32_v_f32m1(tmpptr, vl), vl);
                tmpptr += packn;
                kptr++;
            }

            vse32_v_f32m1(outptr0, _sum0, vl);

            outptr0 += packn;
        }
#else  // __riscv_vector
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 4);
            const float* kptr = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;

            for (int q = 0; q < nn; q++)
            {
                float k0 = kptr[0];
                sum0 += tmpptr[0] * k0;
                sum1 += tmpptr[1] * k0;
                sum2 += tmpptr[2] * k0;
                sum3 += tmpptr[3] * k0;
                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
#if __riscv_vector
            const float* tmpptr = tmp.channel(i / packn + i % packn);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
            const float* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int nn = inch * maxk; // inch always > 0

            float sum0 = bias0;

            for (int q = 0; q < nn; q++)
            {
                sum0 += tmpptr[0] * kptr[0];
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_rvv(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-maxk-inch-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __riscv_vector
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __riscv_vector
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);
        const Mat k4 = kernel.channel(q + 4);
        const Mat k5 = kernel.channel(q + 5);
        const Mat k6 = kernel.channel(q + 6);
        const Mat k7 = kernel.channel(q + 7);

        float* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);
            const float* k40 = k4.row(p);
            const float* k50 = k5.row(p);
            const float* k60 = k6.row(p);
            const float* k70 = k7.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00 += 8;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
#else
    for (; q + 1 < outch; q += 2)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);

        float* g00 = kernel_tm.channel(q / 2);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];

                g00 += 2;
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

#if __riscv_vector
        float* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4 + q % 4);
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];

                g00 += 1;
            }
        }
    }
}

static void convolution_im2col_sgemm_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
