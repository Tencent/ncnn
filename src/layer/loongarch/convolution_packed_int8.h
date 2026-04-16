// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __loongarch_sx
    if (outch >= 4)
    {
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)32u, 32);
        else if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)8u, 8);
        else
            kernel_tm.create(maxk, inch, outch / 4 + (outch % 4) / 2 + outch % 2, (size_t)4u, 4);
    }
    else
#endif // __loongarch_sx
    if (outch >= 2)
    {
#if __loongarch_sx
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2, (size_t)16u, 16);
        else
#endif // __loongarch_sx
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)4u, 4);
        else
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
#if __loongarch_sx
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch, (size_t)8u, 8);
        else
#endif // __loongarch_sx
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch, (size_t)2u, 2);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __loongarch_sx
    for (; q + 3 < outch; q += 4)
    {
        signed char* g00 = kernel_tm.channel(q / 4);

        int p = 0;
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    for (int j = 0; j < 8; j++)
                    {
                        g00[0] = k0[(p + j) * maxk + k];
                        g00++;
                    }
                }
            }
        }
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    for (int j = 0; j < 2; j++)
                    {
                        g00[0] = k0[(p + j) * maxk + k];
                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    g00[0] = k0[p * maxk + k];
                    g00++;
                }
            }
        }
    }
#endif // __loongarch_sx
    for (; q + 1 < outch; q += 2)
    {
#if __loongarch_sx
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2);
#endif

        int p = 0;
#if __loongarch_sx
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    for (int j = 0; j < 8; j++)
                    {
                        g00[0] = k0[(p + j) * maxk + k];
                        g00++;
                    }
                }
            }
        }
#endif // __loongarch_sx
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    for (int j = 0; j < 2; j++)
                    {
                        g00[0] = k0[(p + j) * maxk + k];
                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 2; i++)
                {
                    const signed char* k0 = (const signed char*)kernel + (q + i) * inch * maxk;
                    g00[0] = k0[p * maxk + k];
                    g00++;
                }
            }
        }
    }
    for (; q < outch; q++)
    {
#if __loongarch_sx
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);
#else
        signed char* g00 = kernel_tm.channel(q / 2 + q % 2);
#endif

        int p = 0;
#if __loongarch_sx
        for (; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = (const signed char*)kernel + q * inch * maxk;
                for (int j = 0; j < 8; j++)
                {
                    g00[0] = k0[(p + j) * maxk + k];
                    g00++;
                }
            }
        }
#endif // __loongarch_sx
        for (; p + 1 < inch; p += 2)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = (const signed char*)kernel + q * inch * maxk;
                for (int j = 0; j < 2; j++)
                {
                    g00[0] = k0[(p + j) * maxk + k];
                    g00++;
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = (const signed char*)kernel + q * inch * maxk;
                g00[0] = k0[p * maxk + k];
                g00++;
            }
        }
    }
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * bottom_blob.elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

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
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const size_t N = bottom_blob.cstep * elempack;

    int remain_outch_start = 0;
    int nn_outch = 0;
#if __loongarch_sx
    nn_outch = outch / 4;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = pp * 4;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        int* outptr = top_blob.channel(out_elempack == 4 ? p / 4 : p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 4);

                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* sptr = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* sloc = sptr + space_ofs[k] * elempack;

                        signed char val[8];
                        if (elempack == 8)
                        {
                            for (int n = 0; n < 8; n++)
                                val[n] = sloc[n];
                        }
                        else // elempack == 1
                        {
                            for (int n = 0; n < 8; n++)
                                val[n] = sloc[N * n];
                        }

                        for (int n = 0; n < 8; n++)
                        {
                            sum0 += val[n] * kptr[n];
                            sum1 += val[n] * kptr[8 + n];
                            sum2 += val[n] * kptr[16 + n];
                            sum3 += val[n] * kptr[24 + n];
                        }

                        kptr += 32;
                    }
                }
                for (; q + 1 < inch; q += 2)
                {
                    const signed char* sptr0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val0, val1;
                        if (elempack == 1)
                        {
                            val0 = sptr0[space_ofs[k]];
                            const signed char* sptr1 = bottom_blob.channel(q + 1).row<const signed char>(i * stride_h) + j * stride_w;
                            val1 = sptr1[space_ofs[k]];
                        }
                        else
                        {
                            val0 = sptr0[space_ofs[k] * elempack + q % elempack];
                            val1 = sptr0[space_ofs[k] * elempack + q % elempack + 1];
                        }

                        sum0 += val0 * kptr[0];
                        sum0 += val1 * kptr[1];
                        sum1 += val0 * kptr[2];
                        sum1 += val1 * kptr[3];
                        sum2 += val0 * kptr[4];
                        sum2 += val1 * kptr[5];
                        sum3 += val0 * kptr[6];
                        sum3 += val1 * kptr[7];

                        kptr += 8;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* sptr = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val;
                        if (elempack == 1)
                        {
                            val = sptr[space_ofs[k]];
                        }
                        else
                        {
                            val = sptr[space_ofs[k] * elempack + q % elempack];
                        }

                        sum0 += val * kptr[0];
                        sum1 += val * kptr[1];
                        sum2 += val * kptr[2];
                        sum3 += val * kptr[3];

                        kptr += 4;
                    }
                }

                if (out_elempack == 4)
                {
                    outptr[j * 4] = sum0;
                    outptr[j * 4 + 1] = sum1;
                    outptr[j * 4 + 2] = sum2;
                    outptr[j * 4 + 3] = sum3;
                }
                if (out_elempack == 1)
                {
                    outptr[j] = sum0;
                    outptr[out_hstep + j] = sum1;
                    outptr[out_hstep * 2 + j] = sum2;
                    outptr[out_hstep * 3 + j] = sum3;
                }
            }

            outptr += outw * out_elempack;
        }
    }
    remain_outch_start += nn_outch * 4;
#endif // __loongarch_sx
    nn_outch = (outch - remain_outch_start) / 2;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * 2;

        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        int* outptr = (int*)top_blob + p * out_hstep;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
#if __loongarch_sx
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 4 + (p % 4) / 2);
#else
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 2);
#endif

                int sum0 = 0;
                int sum1 = 0;

                int q = 0;
#if __loongarch_sx
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* sptr = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* sloc = sptr + space_ofs[k] * elempack;

                        signed char val[8];
                        if (elempack == 8)
                        {
                            for (int n = 0; n < 8; n++)
                                val[n] = sloc[n];
                        }
                        else // elempack == 1
                        {
                            for (int n = 0; n < 8; n++)
                                val[n] = sloc[N * n];
                        }

                        for (int n = 0; n < 8; n++)
                        {
                            sum0 += val[n] * kptr[n];
                            sum1 += val[n] * kptr[8 + n];
                        }

                        kptr += 16;
                    }
                }
#endif // __loongarch_sx
                for (; q + 1 < inch; q += 2)
                {
                    const signed char* sptr0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                    const signed char* sptr1 = bottom_blob.channel(q + 1).row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val0 = sptr0[space_ofs[k]];
                        signed char val1 = sptr1[space_ofs[k]];

                        sum0 += val0 * kptr[0];
                        sum0 += val1 * kptr[1];
                        sum1 += val0 * kptr[2];
                        sum1 += val1 * kptr[3];

                        kptr += 4;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val = sptr[space_ofs[k]];

                        sum0 += val * kptr[0];
                        sum1 += val * kptr[1];

                        kptr += 2;
                    }
                }

                outptr[j] = sum0;
                outptr[out_hstep + j] = sum1;
            }

            outptr += outw;
        }
    }
    remain_outch_start += nn_outch * 2;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        // shadowed variable for less openmp task args
        const int outw = top_blob.w;
        const int outh = top_blob.h;

        int* outptr = (int*)top_blob + p * out_hstep;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
#if __loongarch_sx
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 4 + (p % 4) / 2 + p % 2);
#else
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p / 2 + p % 2);
#endif

                int sum = 0;

                int q = 0;
#if __loongarch_sx
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* sptr = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* sloc = sptr + space_ofs[k] * elempack;

                        if (elempack == 8)
                        {
                            for (int n = 0; n < 8; n++)
                            {
                                sum += (int)sloc[n] * (int)kptr[n];
                            }
                        }
                        else // elempack == 1
                        {
                            for (int n = 0; n < 8; n++)
                            {
                                sum += (int)sloc[N * n] * (int)kptr[n];
                            }
                        }

                        kptr += 8;
                    }
                }
#endif // __loongarch_sx
                for (; q + 1 < inch; q += 2)
                {
                    const signed char* sptr0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                    const signed char* sptr1 = bottom_blob.channel(q + 1).row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val0 = sptr0[space_ofs[k]];
                        signed char val1 = sptr1[space_ofs[k]];

                        sum += val0 * kptr[0];
                        sum += val1 * kptr[1];

                        kptr += 2;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val = sptr[space_ofs[k]];

                        sum += val * kptr[0];

                        kptr += 1;
                    }
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
