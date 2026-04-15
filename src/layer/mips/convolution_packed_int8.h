// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pa-pb-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __mips_msa
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
#endif // __mips_msa
    if (outch >= 2)
    {
#if __mips_msa
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch / 2 + outch % 2, (size_t)16u, 16);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch / 2 + outch % 2, (size_t)4u, 4);
        else
            kernel_tm.create(maxk, inch, outch / 2 + outch % 2, (size_t)2u, 2);
    }
    else
    {
#if __mips_msa
        if (inch >= 8)
            kernel_tm.create(maxk, inch / 8 + (inch % 8) / 2 + inch % 2, outch, (size_t)8u, 8);
        else
#endif // __mips_msa
        if (inch >= 2)
            kernel_tm.create(maxk, inch / 2 + inch % 2, outch, (size_t)2u, 2);
        else
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __mips_msa
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
#endif // __mips_msa
    for (; q + 1 < outch; q += 2)
    {
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2);

        int p = 0;
#if __mips_msa
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
#endif // __mips_msa
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
        signed char* g00 = kernel_tm.channel(q / 4 + (q % 4) / 2 + q % 2);

        int p = 0;
#if __mips_msa
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
#endif // __mips_msa
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
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c;

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                const signed char* kptr = (const signed char*)weight_data_tm.channel(p);

#if __mips_msa
                if (out_elempack == 4)
                {
                    v4i32 _sum0 = __msa_fill_w(0);
                    v4i32 _sum1 = __msa_fill_w(0);
                    v4i32 _sum2 = __msa_fill_w(0);
                    v4i32 _sum3 = __msa_fill_w(0);

                    if (elempack == 8)
                    {
                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                v16i8 _val = __msa_ld_b(sptr + space_ofs[k] * 8, 0);
                                v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                                v16i8 _w01 = __msa_ld_b(kptr, 0);
                                v16i8 _w23 = __msa_ld_b(kptr + 16, 0);
                                v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                                v16i8 _extw23 = __msa_clti_s_b(_w23, 0);
                                v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                                v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);
                                v8i16 _w2 = (v8i16)__msa_ilvr_b(_extw23, _w23);
                                v8i16 _w3 = (v8i16)__msa_ilvl_b(_extw23, _w23);

                                v8i16 _s0 = __msa_mulv_h(_val16, _w0);
                                v8i16 _s1 = __msa_mulv_h(_val16, _w1);
                                v8i16 _s2 = __msa_mulv_h(_val16, _w2);
                                v8i16 _s3 = __msa_mulv_h(_val16, _w3);

                                _sum0 = __msa_addv_w(_sum0, __msa_hadd_s_w(_s0, _s0));
                                _sum1 = __msa_addv_w(_sum1, __msa_hadd_s_w(_s1, _s1));
                                _sum2 = __msa_addv_w(_sum2, __msa_hadd_s_w(_s2, _s2));
                                _sum3 = __msa_addv_w(_sum3, __msa_hadd_s_w(_s3, _s3));

                                kptr += 32;
                            }
                        }
                    }
                    if (elempack == 1)
                    {
                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                v8i16 _val = __msa_fill_h((short)sptr[space_ofs[k]]);

                                v16i8 _w = __msa_ld_b(kptr, 0);
                                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                                v8i16 _s0 = __msa_mulv_h(_val, _w16);
                                v4i32 _s032 = (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0);

                                _sum0 = __msa_addv_w(_sum0, _s032);

                                kptr += 4;
                            }
                        }

                        _sum1 = __msa_fill_w(0);
                        _sum2 = __msa_fill_w(0);
                        _sum3 = __msa_fill_w(0);
                    }

                    // transpose 4x4 and reduce (for elempack==8 path)
                    if (elempack == 8)
                    {
                        v4i32 _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = __msa_ilvr_w(_sum1, _sum0);
                        _tmp1 = __msa_ilvr_w(_sum3, _sum2);
                        _tmp2 = __msa_ilvl_w(_sum1, _sum0);
                        _tmp3 = __msa_ilvl_w(_sum3, _sum2);
                        _sum0 = (v4i32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp0);
                        _sum1 = (v4i32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp0);
                        _sum2 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp2);
                        _sum3 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp2);

                        _sum0 = __msa_addv_w(_sum0, _sum1);
                        _sum2 = __msa_addv_w(_sum2, _sum3);
                        _sum0 = __msa_addv_w(_sum0, _sum2);
                    }

                    __msa_st_w(_sum0, outptr + j * 4, 0);
                }
                if (out_elempack == 1)
                {
                    int sum = 0;

                    if (elempack == 8)
                    {
                        v4i32 _sum = __msa_fill_w(0);

                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                v16i8 _val = __msa_ld_b(sptr + space_ofs[k] * 8, 0);
                                v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                                v16i8 _w = __msa_ld_b(kptr, 0);
                                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                                v8i16 _s0 = __msa_mulv_h(_val16, _w16);

                                _sum = __msa_addv_w(_sum, __msa_hadd_s_w(_s0, _s0));

                                kptr += 8;
                            }
                        }

                        sum = __msa_reduce_add_w(_sum);
                    }
                    if (elempack == 1)
                    {
                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                signed char val = sptr[space_ofs[k]];
                                signed char wt = kptr[0];
                                sum += val * wt;

                                kptr += 1;
                            }
                        }
                    }

                    outptr[j * out_elempack] = sum;
                }
#else  // __mips_msa
                {
                    int sum = 0;

                    for (int q = 0; q < inch; q++)
                    {
                        const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            signed char val = sptr[space_ofs[k]];
                            signed char wt = kptr[0];
                            sum += val * wt;

                            kptr += 1;
                        }
                    }

                    outptr[j] = sum;
                }
#endif // __mips_msa
            }

            outptr += outw * out_elempack;
        }
    }
}
