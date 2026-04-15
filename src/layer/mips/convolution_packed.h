// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_transform_kernel_packed(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
    if (outch >= 4)
    {
        if (inch >= 4)
            kernel_tm.create(4 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else if (inch >= 2)
            kernel_tm.create(4 * 2 * maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2);
        else
            kernel_tm.create(4 * maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2);
    }
    else
#endif // __mips_msa
    if (outch >= 2)
    {
#if __mips_msa
        if (inch >= 4)
            kernel_tm.create(2 * 4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch / 2 + outch % 2);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(2 * 2 * maxk, inch / 2 + inch % 2, outch / 2 + outch % 2);
        else
            kernel_tm.create(2 * maxk, inch, outch / 2 + outch % 2);
    }
    else
    {
#if __mips_msa
        if (inch >= 4)
            kernel_tm.create(4 * maxk, inch / 4 + (inch % 4) / 2 + inch % 2, outch);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(2 * maxk, inch / 2 + inch % 2, outch);
        else
            kernel_tm.create(maxk, inch, outch);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
    for (; q + 3 < outch; q += 4)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;
        const float* kptr2 = (const float*)kernel + (q + 2) * inch * maxk;
        const float* kptr3 = (const float*)kernel + (q + 3) * inch * maxk;

        float* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
            kptr2 += maxk * 4;
            kptr3 += maxk * 4;
        }

        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
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
                    k0 += maxk;
                    k1 += maxk;
                    k2 += maxk;
                    k3 += maxk;
                    g00 += 4;
                }
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
            kptr2 += maxk * 2;
            kptr3 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
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
    for (; q + 1 < outch; q += 2)
    {
        const float* kptr0 = (const float*)kernel + q * inch * maxk;
        const float* kptr1 = (const float*)kernel + (q + 1) * inch * maxk;

#if __mips_msa
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        float* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k0[maxk];
                g00[2] = k0[maxk * 2];
                g00[3] = k0[maxk * 3];
                g00[4] = k1[0];
                g00[5] = k1[maxk];
                g00[6] = k1[maxk * 2];
                g00[7] = k1[maxk * 3];
                g00 += 8;
            }

            kptr0 += maxk * 4;
            kptr1 += maxk * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    g00[1] = k1[0];
                    k0 += maxk;
                    k1 += maxk;
                    g00 += 2;
                }
            }

            kptr0 += maxk * 2;
            kptr1 += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr0 + k;
                const float* k1 = kptr1 + k;

                g00[0] = k0[0];
                g00[1] = k1[0];
                g00 += 2;
            }
        }
    }
    for (; q < outch; q++)
    {
        const float* kptr = (const float*)kernel + q * inch * maxk;

#if __mips_msa
        float* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        float* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __mips_msa
        for (; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 4; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 4;
        }
#endif // __mips_msa
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                for (int i = 0; i < 2; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }

            kptr += maxk * 2;
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const float* k0 = kptr + k;

                g00[0] = k0[0];
                g00++;
            }
        }
    }
}

static void convolution_packed(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const size_t N = bottom_blob.cstep * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;

    const size_t M = top_blob.cstep * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __mips_msa
    nn_outch = outch / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 4;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;
        const int out_elempack = top_blob.elempack;

        float* outptr = top_blob.channel(p / out_elempack);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p, 0);
                }

                const float* kptr = weight_data_tm.channel(p / 4);

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const float* r0s = r0 + space_ofs[k];

                            v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                            v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                            v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                            v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);

                            _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0s[0]));
                            _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0s[1]));
                            _sum2 = __msa_fmadd_w(_sum2, _w2, __msa_fill_w_f32(r0s[2]));
                            _sum3 = __msa_fmadd_w(_sum3, _w3, __msa_fill_w_f32(r0s[3]));

                            kptr += 16;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                            v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                            v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                            v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);

                            _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0[sok]));
                            _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0[sok + N]));
                            _sum2 = __msa_fmadd_w(_sum2, _w2, __msa_fill_w_f32(r0[sok + N * 2]));
                            _sum3 = __msa_fmadd_w(_sum3, _w3, __msa_fill_w_f32(r0[sok + N * 3]));

                            kptr += 16;
                        }
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                            v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);

                            _sum0 = __msa_fmadd_w(_sum0, _w0, __msa_fill_w_f32(r0[sok]));
                            _sum1 = __msa_fmadd_w(_sum1, _w1, __msa_fill_w_f32(r0[sok + N]));

                            kptr += 8;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            v4f32 _val = __msa_fill_w_f32(r0[space_ofs[k]]);
                            v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                            _sum0 = __msa_fmadd_w(_sum0, _val, _w);

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
    }
    remain_outch_start += nn_outch * 4;
    nn_outch = (outch - remain_outch_start) / 2;
#else // __mips_msa
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
#endif // __mips_msa
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int elempack = bottom_blob.elempack;
        const int inch = bottom_blob.c * elempack;
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        float* outptr0 = top_blob.channel(p);
        float* outptr1 = top_blob.channel(p + 1);

        for (int i = 0; i < outh; i++)
        {
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
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            v4f32 _r0 = (v4f32)__msa_ld_w(r0 + sok, 0);
                            v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                            v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                            _sum0 = __msa_fmadd_w(_sum0, _r0, _w0);
                            _sum1 = __msa_fmadd_w(_sum1, _r0, _w1);

                            kptr += 8;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            float tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            v4f32 _r0 = (v4f32)__msa_ld_w(tmpbuf, 0);
                            v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                            v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                            _sum0 = __msa_fmadd_w(_sum0, _r0, _w0);
                            _sum1 = __msa_fmadd_w(_sum1, _r0, _w1);

                            kptr += 8;
                        }
                    }
                }
                sum0 += __msa_reduce_fadd_w(_sum0);
                sum1 += __msa_reduce_fadd_w(_sum1);
#endif // __mips_msa
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            sum0 += r0[sok] * kptr[0];
                            sum1 += r0[sok] * kptr[1];
                            sum0 += r0[sok + N] * kptr[2];
                            sum1 += r0[sok + N] * kptr[3];

                            kptr += 4;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            float val = r0[space_ofs[k]];
                            sum0 += val * kptr[0];
                            sum1 += val * kptr[1];

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
    }
    remain_outch_start += nn_outch * 2;
    for (int p = remain_outch_start; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
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
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob.channel(q / elempack).row(i * stride_h) + j * stride_w * elempack;

                    if (elempack == 4)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            v4f32 _r0 = (v4f32)__msa_ld_w(r0 + sok, 0);
                            v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                            _sum = __msa_fmadd_w(_sum, _r0, _w);

                            kptr += 4;
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];
                            float tmpbuf[4] = {r0[sok], r0[sok + N], r0[sok + N * 2], r0[sok + N * 3]};
                            v4f32 _r0 = (v4f32)__msa_ld_w(tmpbuf, 0);
                            v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                            _sum = __msa_fmadd_w(_sum, _r0, _w);

                            kptr += 4;
                        }
                    }
                }
                sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
                for (; q + 1 < inch; q += 2)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const int sok = space_ofs[k];

                            sum += r0[sok] * kptr[0];
                            sum += r0[sok + N] * kptr[1];

                            kptr += 2;
                        }
                    }
                }
                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob.channel(q).row(i * stride_h) + j * stride_w;

                    // if (elempack == 1)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            float val = r0[space_ofs[k]];
                            sum += val * kptr[0];

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
}
