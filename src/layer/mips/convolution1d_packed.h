// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution1d_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    // src = kernel(outh, inh, kernel_w)

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
    if (outh >= 4)
    {
        if (inh >= 4)
            kernel_tm.create(4 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else if (inh >= 2)
            kernel_tm.create(4 * 2 * kernel_w, inh / 2 + inh % 2, outh / 4 + (outh % 4) / 2 + outh % 2);
        else
            kernel_tm.create(4 * kernel_w, inh, outh / 4 + (outh % 4) / 2 + outh % 2);
    }
    else
#endif // __mips_msa
    if (outh >= 2)
    {
#if __mips_msa
        if (inh >= 4)
            kernel_tm.create(2 * 4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh / 2 + outh % 2);
        else
#endif // __mips_msa
        if (inh >= 2)
            kernel_tm.create(2 * 2 * kernel_w, inh / 2 + inh % 2, outh / 2 + outh % 2);
        else
            kernel_tm.create(2 * kernel_w, inh, outh / 2 + outh % 2);
    }
    else
    {
#if __mips_msa
        if (inh >= 4)
            kernel_tm.create(4 * kernel_w, inh / 4 + (inh % 4) / 2 + inh % 2, outh);
        else
#endif // __mips_msa
        if (inh >= 2)
            kernel_tm.create(2 * kernel_w, inh / 2 + inh % 2, outh);
        else
            kernel_tm.create(kernel_w, inh, outh);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; q + 3 < outh; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;
        const float* kptr2 = (const float*)kernel + (q + 2) * inh * kernel_w;
        const float* kptr3 = (const float*)kernel + (q + 3) * inh * kernel_w;

        float* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
            kptr2 += kernel_w * 4;
            kptr3 += kernel_w * 4;
        }
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    g00[2] = k2[0];
                    g00[3] = k3[0];
                    k0 += kernel_w;
                    k1 += kernel_w;
                    k2 += kernel_w;
                    k3 += kernel_w;
                    g00 += 4;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
            kptr2 += kernel_w * 2;
            kptr3 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;
                const float* k2 = kptr2 + k;
                const float* k3 = kptr3 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00[2] = k2[0];
                g00[3] = k3[0];
                g00 += 4;
            }
        }
    }
#endif // __mips_msa
    for (; q + 1 < outh; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;
        const float* kptr1 = (const float*)kernel + (q + 1) * inh * kernel_w;

#if __mips_msa
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k0[kernel_w];
                g00[2] = k0[kernel_w * 2];
                g00[3] = k0[kernel_w * 3];
                g00[4] = k1[0];
                g00[5] = k1[kernel_w];
                g00[6] = k1[kernel_w * 2];
                g00[7] = k1[kernel_w * 3];
                g00 += 8;
            }

            kptr0 += kernel_w * 4;
            kptr1 += kernel_w * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    k0 += kernel_w;
                    k1 += kernel_w;
                    g00 += 2;
                }
            }

            kptr0 += kernel_w * 2;
            kptr1 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outh; q++)
    {
        const float* kptr0 = (const float*)kernel + q * inh * kernel_w;

#if __mips_msa
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 3 < inh; p += 4)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr0 += kernel_w * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inh; p += 2)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    k0 += kernel_w;
                    g00 += 1;
                }
            }

            kptr0 += kernel_w * 2;
        }
        for (; p < inh; p++)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                const float* k0 = kptr0 + k;

                g00[0] = k0[0];
                g00 += 1;
            }
        }
    }
}

static void convolution1d_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int elempack = bottom_blob.elempack;
    const int inh = bottom_blob.h * elempack;
    const int N = bottom_blob.w * elempack;

    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;
    const int outh = top_blob.h * out_elempack;
    const int M = top_blob.w * out_elempack;

    const float* bias_data_ptr = bias_data;

    int nn_outh = 0;
    int remain_outh_start = 0;

#if __mips_msa
    nn_outh = outh / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = pp * 4;

        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.row(p / out_elempack);

        for (int j = 0; j < outw; j++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (v4f32)__msa_ld_w((const float*)bias_data_ptr + p, 0);
            }

            const float* kptr = weight_data_tm.channel(p / 4);

            int q = 0;
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                        v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);

                        _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0[0]));
                        _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0[1]));
                        _sum2 = __msa_fmadd_w(_sum2, _w2, __msa_fill_w_f32(r0[2]));
                        _sum3 = __msa_fmadd_w(_sum3, _w3, __msa_fill_w_f32(r0[3]));

                        r0 += dilation_w * 4;
                        kptr += 16;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                        v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);

                        _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0[0]));
                        _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0[N]));
                        _sum2 = __msa_fmadd_w(_sum2, _w2, __msa_fill_w_f32(r0[N * 2]));
                        _sum3 = __msa_fmadd_w(_sum3, _w3, __msa_fill_w_f32(r0[N * 3]));

                        r0 += dilation_w;
                        kptr += 16;
                    }
                }
            }
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);

                        _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0[0]));
                        _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0[N]));

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _val = __msa_fill_w_f32(r0[0]);
                        v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                        _sum0 = __msa_fmadd_w(_sum0, _val, _w);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum0 = __msa_fadd_w(_sum0, _sum2);

            _sum0 = activation_msa(_sum0, activation_type, activation_params);

            if (out_elempack == 4)
            {
                __msa_st_w((v4i32)_sum0, outptr, 0);
                outptr += 4;
            }
            if (out_elempack == 1)
            {
                float sum[4];
                __msa_st_w((v4i32)_sum0, sum, 0);

                outptr[0] = sum[0];
                outptr[M] = sum[1];
                outptr[M * 2] = sum[2];
                outptr[M * 3] = sum[3];
                outptr += 1;
            }
        }
    }
    remain_outh_start += nn_outh * 4;
    nn_outh = (outh - remain_outh_start) / 2;
#else  // __mips_msa
    nn_outh = (outh - remain_outh_start) / 2;
#endif // __mips_msa
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outh; pp++)
    {
        const int p = remain_outh_start + pp * 2;

        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        float* outptr0 = top_blob.row(p);
        float* outptr1 = top_blob.row(p + 1);

        for (int j = 0; j < outw; j++)
        {
            float sum0 = 0.f;
            float sum1 = 0.f;

            if (bias_data_ptr)
            {
                sum0 = bias_data_ptr[p];
                sum1 = bias_data_ptr[p + 1];
            }

#if __mips_msa
            const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
            const float* kptr = weight_data_tm.channel(p / 2);
#endif

            int q = 0;
#if __mips_msa
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        _sum0 = __msa_fmadd_w(_sum0, _r0, _w0);
                        _sum1 = __msa_fmadd_w(_sum1, _r0, _w1);

                        r0 += dilation_w * 4;
                        kptr += 8;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _r0 = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        _sum0 = __msa_fmadd_w(_sum0, _r0, _w0);
                        _sum1 = __msa_fmadd_w(_sum1, _r0, _w1);

                        r0 += dilation_w;
                        kptr += 8;
                    }
                }
            }
            sum0 += __msa_reduce_fadd_w(_sum0);
            sum1 += __msa_reduce_fadd_w(_sum1);
#endif // __mips_msa
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        sum0 += r0[0] * kptr[0];
                        sum1 += r0[0] * kptr[1];
                        sum0 += r0[N] * kptr[2];
                        sum1 += r0[N] * kptr[3];

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        float val = r0[0];
                        sum0 += val * kptr[0];
                        sum1 += val * kptr[1];

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }

            sum0 = activation_ss(sum0, activation_type, activation_params);
            sum1 = activation_ss(sum1, activation_type, activation_params);

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr0 += 1;
            outptr1 += 1;
        }
    }
    remain_outh_start += nn_outh * 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outh_start; p < outh; p++)
    {
        // shadowed variables for thread safety
        const int elempack = bottom_blob.elempack;
        const int inh = bottom_blob.h * elempack;
        const int outw = top_blob.w;

        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
            {
                sum = bias_data_ptr[p];
            }

#if __mips_msa
            const float* kptr = weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
            const float* kptr = weight_data_tm.channel(p / 2 + p % 2);
#endif

            int q = 0;
#if __mips_msa
            v4f32 _sum = (v4f32)__msa_fill_w(0);
            for (; q + 3 < inh; q += 4)
            {
                const float* r0 = bottom_blob.row(q / elempack) + j * stride_w * elempack;

                if (elempack == 4)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(r0 + 16);
                        __builtin_prefetch(kptr + 16);

                        v4f32 _r0 = (v4f32)__msa_ld_w(r0, 0);
                        v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                        _sum = __msa_fmadd_w(_sum, _r0, _w);

                        r0 += dilation_w * 4;
                        kptr += 4;
                    }
                }
                if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        v4f32 _r0 = {r0[0], r0[N], r0[N * 2], r0[N * 3]};
                        v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                        _sum = __msa_fmadd_w(_sum, _r0, _w);

                        r0 += dilation_w;
                        kptr += 4;
                    }
                }
            }
            sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
            for (; q + 1 < inh; q += 2)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        sum += r0[0] * kptr[0];
                        sum += r0[N] * kptr[1];

                        r0 += dilation_w;
                        kptr += 2;
                    }
                }
            }
            for (; q < inh; q++)
            {
                const float* r0 = bottom_blob.row(q) + j * stride_w;

                // if (elempack == 1)
                {
                    for (int k = 0; k < kernel_w; k++)
                    {
                        __builtin_prefetch(kptr + 16);

                        float val = r0[0];
                        sum += val * kptr[0];

                        r0 += dilation_w;
                        kptr += 1;
                    }
                }
            }

            sum = activation_ss(sum, activation_type, activation_params);

            outptr[0] = sum;
            outptr += 1;
        }
    }
}
