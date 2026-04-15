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

#if __loongarch_sx
                if (out_elempack == 4)
                {
                    __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum3 = __lsx_vreplgr2vr_w(0);

                    if (elempack == 8)
                    {
                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128i _val = __lsx_vld(sptr + space_ofs[k] * 8, 0);
                                __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                                __m128i _w01 = __lsx_vld(kptr, 0);
                                __m128i _w23 = __lsx_vld(kptr + 16, 0);
                                __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                                __m128i _extw23 = __lsx_vslti_b(_w23, 0);
                                __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                                __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);
                                __m128i _w2 = __lsx_vilvl_b(_extw23, _w23);
                                __m128i _w3 = __lsx_vilvh_b(_extw23, _w23);

                                __m128i _s0 = __lsx_vmul_h(_val16, _w0);
                                __m128i _s1 = __lsx_vmul_h(_val16, _w1);
                                __m128i _s2 = __lsx_vmul_h(_val16, _w2);
                                __m128i _s3 = __lsx_vmul_h(_val16, _w3);

                                _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                                _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                                _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                                _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));

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
                                __m128i _val = __lsx_vreplgr2vr_h((short)sptr[space_ofs[k]]);

                                __m128i _w = __lsx_vld(kptr, 0);
                                __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                                __m128i _s0 = __lsx_vmul_h(_val, _w16);
                                __m128i _s032 = __lsx_vilvl_h(__lsx_vslti_h(_s0, 0), _s0);

                                _sum0 = __lsx_vadd_w(_sum0, _s032);

                                kptr += 4;
                            }
                        }

                        _sum1 = __lsx_vreplgr2vr_w(0);
                        _sum2 = __lsx_vreplgr2vr_w(0);
                        _sum3 = __lsx_vreplgr2vr_w(0);
                    }

                    // transpose 4x4 and reduce (for elempack==8 path)
                    if (elempack == 8)
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = __lsx_vilvl_w(_sum1, _sum0);
                        _tmp1 = __lsx_vilvl_w(_sum3, _sum2);
                        _tmp2 = __lsx_vilvh_w(_sum1, _sum0);
                        _tmp3 = __lsx_vilvh_w(_sum3, _sum2);
                        _sum0 = __lsx_vilvl_d(_tmp1, _tmp0);
                        _sum1 = __lsx_vilvh_d(_tmp1, _tmp0);
                        _sum2 = __lsx_vilvl_d(_tmp3, _tmp2);
                        _sum3 = __lsx_vilvh_d(_tmp3, _tmp2);

                        _sum0 = __lsx_vadd_w(_sum0, _sum1);
                        _sum2 = __lsx_vadd_w(_sum2, _sum3);
                        _sum0 = __lsx_vadd_w(_sum0, _sum2);
                    }

                    __lsx_vst(_sum0, outptr + j * 4, 0);
                }
                if (out_elempack == 1)
                {
                    int sum = 0;

                    if (elempack == 8)
                    {
                        __m128i _sum = __lsx_vreplgr2vr_w(0);

                        for (int q = 0; q < inch; q++)
                        {
                            const signed char* sptr = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128i _val = __lsx_vld(sptr + space_ofs[k] * 8, 0);
                                __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                                __m128i _w = __lsx_vld(kptr, 0);
                                __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                                __m128i _s0 = __lsx_vmul_h(_val16, _w16);

                                _sum = __lsx_vadd_w(_sum, __lsx_vhaddw_w_h(_s0, _s0));

                                kptr += 8;
                            }
                        }

                        sum = __lsx_reduce_add_w(_sum);
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
#else  // __loongarch_sx
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
#endif // __loongarch_sx
            }

            outptr += outw * out_elempack;
        }
    }
}
