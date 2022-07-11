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

static void im2col_sgemm_msa(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 4u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u, 1, opt.workspace_allocator);
    {
        int nn_size = size / 4;

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
#if __mips_msa
                    __msa_st_w(__msa_ld_w(img0, 0), tmpptr, 0);
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];
#endif
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        int remain_size_start = nn_size * 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 4 + i % 4);

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

#if __mips_msa
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
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 4);
            const float* kptr = kernel.channel(p / 8);

            int nn = inch * maxk; // inch always > 0

            v4f32 _sum0 = __msa_fill_w_f32(biasptr[0]);
            v4f32 _sum1 = __msa_fill_w_f32(biasptr[1]);
            v4f32 _sum2 = __msa_fill_w_f32(biasptr[2]);
            v4f32 _sum3 = __msa_fill_w_f32(biasptr[3]);
            v4f32 _sum4 = __msa_fill_w_f32(biasptr[4]);
            v4f32 _sum5 = __msa_fill_w_f32(biasptr[5]);
            v4f32 _sum6 = __msa_fill_w_f32(biasptr[6]);
            v4f32 _sum7 = __msa_fill_w_f32(biasptr[7]);

            for (int q = 0; q < nn; q++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr + 32);
                v4f32 _val = (v4f32)__msa_ld_w(tmpptr, 0);
                v4i32 _w0123 = __msa_ld_w(kptr, 0);
                v4i32 _w4567 = __msa_ld_w(kptr + 4, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w0123, 0));
                _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w0123, 1));
                _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w0123, 2));
                _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w0123, 3));
                _sum4 = __msa_fmadd_w(_sum4, _val, (v4f32)__msa_splati_w(_w4567, 0));
                _sum5 = __msa_fmadd_w(_sum5, _val, (v4f32)__msa_splati_w(_w4567, 1));
                _sum6 = __msa_fmadd_w(_sum6, _val, (v4f32)__msa_splati_w(_w4567, 2));
                _sum7 = __msa_fmadd_w(_sum7, _val, (v4f32)__msa_splati_w(_w4567, 3));

                tmpptr += 4;
                kptr += 8;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr1, 0);
            __msa_st_w((v4i32)_sum2, outptr2, 0);
            __msa_st_w((v4i32)_sum3, outptr3, 0);
            __msa_st_w((v4i32)_sum4, outptr4, 0);
            __msa_st_w((v4i32)_sum5, outptr5, 0);
            __msa_st_w((v4i32)_sum6, outptr6, 0);
            __msa_st_w((v4i32)_sum7, outptr7, 0);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
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
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 4);
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4);

            int nn = inch * maxk; // inch always > 0

            v4f32 _sum0 = __msa_fill_w_f32(biasptr[0]);
            v4f32 _sum1 = __msa_fill_w_f32(biasptr[1]);
            v4f32 _sum2 = __msa_fill_w_f32(biasptr[2]);
            v4f32 _sum3 = __msa_fill_w_f32(biasptr[3]);

            for (int q = 0; q < nn; q++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr + 16);
                v4f32 _val = (v4f32)__msa_ld_w(tmpptr, 0);
                v4i32 _w0123 = __msa_ld_w(kptr, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w0123, 0));
                _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w0123, 1));
                _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w0123, 2));
                _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w0123, 3));

                tmpptr += 4;
                kptr += 4;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);
            __msa_st_w((v4i32)_sum1, outptr1, 0);
            __msa_st_w((v4i32)_sum2, outptr2, 0);
            __msa_st_w((v4i32)_sum3, outptr3, 0);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
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
#else // __mips_msa
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
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr + 8);
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
                __builtin_prefetch(tmpptr + 4);
                __builtin_prefetch(kptr + 8);
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
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 4);
#if __mips_msa
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
            const float* kptr = kernel.channel(p / 2 + p % 2);
#endif

            int nn = inch * maxk; // inch always > 0

#if __mips_msa
            v4f32 _sum0 = __msa_fill_w_f32(bias0);

            for (int q = 0; q < nn; q++)
            {
                _sum0 = __msa_fmadd_w(_sum0, __msa_fill_w_f32(kptr[0]), (v4f32)__msa_ld_w(tmpptr, 0));
                tmpptr += 4;
                kptr++;
            }

            __msa_st_w((v4i32)_sum0, outptr0, 0);

            outptr0 += 4;
#else
            float sum0 = bias0;
            float sum1 = bias0;
            float sum2 = bias0;
            float sum3 = bias0;

            for (int q = 0; q < nn; q++)
            {
                __builtin_prefetch(tmpptr + 16);
                __builtin_prefetch(kptr + 4);
                sum0 += tmpptr[0] * kptr[0];
                sum1 += tmpptr[1] * kptr[0];
                sum2 += tmpptr[2] * kptr[0];
                sum3 += tmpptr[3] * kptr[0];
                tmpptr += 4;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;

            outptr0 += 4;
#endif // __mips_msa
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 4 + i % 4);
#if __mips_msa
            const float* kptr = kernel.channel(p / 8 + (p % 8) / 4 + p % 4);
#else
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

static void convolution_im2col_sgemm_transform_kernel_msa(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-maxk-inch-outch/8b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __mips_msa
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4 + outch % 4);
#else
    kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2);
#endif

    int q = 0;
#if __mips_msa
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
#endif // __mips_msa
    for (; q < outch; q++)
    {
        const Mat k0 = kernel.channel(q);

#if __mips_msa
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

static void convolution_im2col_sgemm_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    im2col_sgemm_msa(bottom_im2col, top_blob, kernel, _bias, opt);
}
