// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convolution_transform_kernel_packed_int8_rvv(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/vlm1-outch/vlm1

    // clang-format off
    // *INDENT-OFF*
#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();

    const int packn = (int)vlm1;
    const int pack4n = (int)vlm4;

    if (outch >= pack4n)
    {
        if (inch >= pack4n)
            kernel_tm.create(maxk, inch / pack4n + inch % pack4n, outch / pack4n + (outch % pack4n) / packn + outch % packn, (size_t)(pack4n * pack4n), pack4n * pack4n);
        else
            kernel_tm.create(maxk, inch, outch / pack4n + (outch % pack4n) / packn + outch % packn, (size_t)pack4n, pack4n);
    }
    else if (outch >= packn)
    {
        if (inch >= pack4n)
            kernel_tm.create(maxk, inch / pack4n + inch % pack4n, outch / packn + outch % packn, (size_t)(pack4n * packn), pack4n * packn);
        else
            kernel_tm.create(maxk, inch, outch / packn + outch % packn, (size_t)packn, packn);
    }
    else
#endif // __riscv_vector
    {
#if __riscv_vector
        if (inch >= pack4n)
            kernel_tm.create(maxk, inch / pack4n + inch % pack4n, outch, (size_t)pack4n, pack4n);
        else
#endif // __riscv_vector
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __riscv_vector
    for (; q + pack4n - 1 < outch; q += pack4n)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / pack4n);

        int p = 0;
        for (; p + pack4n - 1 < inch; p += pack4n)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (size_t i = 0; i < pack4n; i++)
                {
                    const signed char* src = kptr + (p + i) * maxk + k;
                    vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, vlm4);
                    __riscv_vse8_v_i8m1(g00, row, vlm4);
                    g00 += pack4n;
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* src = kptr + p * maxk + k;
                vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, vlm4);
                __riscv_vse8_v_i8m1(g00, row, vlm4);
                g00 += pack4n;
            }
        }
    }
    for (; q + packn - 1 < outch; q += packn)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / pack4n + (q % pack4n) / packn);
        int p = 0;
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* src = kptr + p * maxk + k;
                vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, vlm1);
                __riscv_vse8_v_i8m1(g00, row, vlm1);
                g00 += packn;
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
#if __riscv_vector
        signed char* g00 = kernel_tm.channel(q / pack4n + (q % pack4n) / packn + q % packn);
#else
        signed char* g00 = kernel_tm.channel(q);
#endif

        int p = 0;
#if __riscv_vector
        for (; p + pack4n - 1 < inch; p += pack4n)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (size_t i = 0; i < pack4n; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }
            kptr += maxk * pack4n;
        }
#endif // __riscv_vector
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                g00[0] = kptr[0];
                g00++;
                kptr++;
            }
        }
    }
    return;
}

static void convolution_packed_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;
#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlm4 = __riscv_vsetvlmax_e32m4();

    const int pack4n = (int)vlm4;
    const int packn = (int)vlm1;
    const size_t N = (elempack == pack4n) ? 1 : bottom_blob.cstep * elempack;
    const size_t M = top_blob.cstep * out_elempack;
#endif
    // kernel offsets
    const int maxk = kernel_w * kernel_h;
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

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __riscv_vector
    nn_outch = (outch - remain_outch_start) / pack4n;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * pack4n;
        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int j0 = ij % outw;
            const int i1 = (ij + 1) / outw;
            const int j1 = (ij + 1) % outw;

            vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vlm4);
            vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vlm4);
            const signed char* kptr = weight_data_tm.channel(p / pack4n);

            int q = 0;
            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    for (int l = 0; l < pack4n; l++)
                    {
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                        vint16m2_t _s0 = __riscv_vwmul_vx_i16m2(_w, r0s[l * N], vlm4);
                        vint16m2_t _s1 = __riscv_vwmul_vx_i16m2(_w, r1s[l * N], vlm4);
                        _sum0 = __riscv_vwadd_wv_i32m4(_sum0, _s0, vlm4);
                        _sum1 = __riscv_vwadd_wv_i32m4(_sum1, _s1, vlm4);

                        kptr += pack4n;
                    }
                }
            }

            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                    vint16m2_t _s0 = __riscv_vwmul_vx_i16m2(_w, r0s[0], vlm4);
                    vint16m2_t _s1 = __riscv_vwmul_vx_i16m2(_w, r1s[0], vlm4);
                    _sum0 = __riscv_vwadd_wv_i32m4(_sum0, _s0, vlm4);
                    _sum1 = __riscv_vwadd_wv_i32m4(_sum1, _s1, vlm4);

                    kptr += pack4n;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vlm1);
                __riscv_vse32_v_i32m1(outptr + M, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vlm1);
                outptr += packn;
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vlm1);
                __riscv_vse32_v_i32m1(outptr + M, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 2, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 3, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vlm1);
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum0, vlm4);
                outptr += 1;
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum1, vlm4);
                outptr += 1;
            }
        }

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vlm4);
            const signed char* kptr = weight_data_tm.channel(p / pack4n);

            int q = 0;
            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    for (int l = 0; l < pack4n; l++)
                    {
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                        vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[l * N], vlm4);
                        _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm4);

                        kptr += pack4n;
                    }
                }
            }

            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                    vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[0], vlm4);
                    _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm4);

                    kptr += pack4n;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum, 0), vlm1);
                __riscv_vse32_v_i32m1(outptr + M, __riscv_vget_v_i32m4_i32m1(_sum, 1), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 2, __riscv_vget_v_i32m4_i32m1(_sum, 2), vlm1);
                __riscv_vse32_v_i32m1(outptr + M * 3, __riscv_vget_v_i32m4_i32m1(_sum, 3), vlm1);
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum, vlm4);
                outptr += 1;
            }
        }
    }
    remain_outch_start += nn_outch * pack4n;

    nn_outch = (outch - remain_outch_start) / packn;
    const size_t vl = __riscv_vsetvl_e8m1(packn);
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const int p = remain_outch_start + pp * packn;
        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int j0 = ij % outw;
            const int i1 = (ij + 1) / outw;
            const int j1 = (ij + 1) % outw;

            vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
            const signed char* kptr = weight_data_tm.channel(p / pack4n + (p % pack4n) / packn);

            int q = 0;
            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                for (int l = 0; l < pack4n; l++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        const signed char* r1s = r1 + space_ofs[k];
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                        vint16m2_t _s0 = __riscv_vwmul_vx_i16m2(_w, r0s[l * N], vl);
                        vint16m2_t _s1 = __riscv_vwmul_vx_i16m2(_w, r1s[l * N], vl);
                        _sum0 = __riscv_vwadd_wv_i32m4(_sum0, _s0, vl);
                        _sum1 = __riscv_vwadd_wv_i32m4(_sum1, _s1, vl);

                        kptr += vl;
                    }
                }
            }

            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                    vint16m2_t _s0 = __riscv_vwmul_vx_i16m2(_w, r0s[0], vl);
                    vint16m2_t _s1 = __riscv_vwmul_vx_i16m2(_w, r1s[0], vl);
                    _sum0 = __riscv_vwadd_wv_i32m4(_sum0, _s0, vl);
                    _sum1 = __riscv_vwadd_wv_i32m4(_sum1, _s1, vl);

                    kptr += vl;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl);
                outptr += packn;
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl);
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum0, vl);
                outptr += 1;
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum1, vl);
                outptr += 1;
            }
        }

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);
            const signed char* kptr = weight_data_tm.channel(p / pack4n + (p % pack4n) / packn);

            int q = 0;
            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;
                for (int l = 0; l < pack4n; l++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                        vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[l * N], vl);
                        _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vl);

                        kptr += vl;
                    }
                }
            }

            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                    vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[0], vl);
                    _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vl);

                    kptr += vl;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum, 0), vl);
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum, vl);
                outptr += 1;
            }
        }
    }

    remain_outch_start += nn_outch * packn;
#endif // __riscv_vector
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;
        for (; ij + 1 < outw * outh; ij += 2)
        {
            const int i0 = ij / outw;
            const int j0 = ij % outw;
            const int i1 = (ij + 1) / outw;
            const int j1 = (ij + 1) % outw;

            int sum0 = 0;
            int sum1 = 0;
#if __riscv_vector
            const signed char* kptr = weight_data_tm.channel(p / pack4n + (p % pack4n) / packn + p % packn);
#else
            const signed char* kptr = weight_data_tm.channel(p);
#endif
            int q = 0;
#if __riscv_vector
            vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vlm4);
            vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vlm4);

            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q / elempack).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    vint8m1_t _r0 = __riscv_vlse8_v_i8m1(r0s, N, vlm4);
                    vint8m1_t _r1 = __riscv_vlse8_v_i8m1(r1s, N, vlm4);
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                    vint16m2_t _s0 = __riscv_vwmul_vv_i16m2(_w, _r0, vlm4);
                    vint16m2_t _s1 = __riscv_vwmul_vv_i16m2(_w, _r1, vlm4);
                    _sum0 = __riscv_vwadd_wv_i32m4(_sum0, _s0, vlm4);
                    _sum1 = __riscv_vwadd_wv_i32m4(_sum1, _s1, vlm4);

                    kptr += pack4n;
                }
            }

            vint32m1_t _sum00 = __riscv_vmv_v_x_i32m1(0, vlm1);
            _sum00 = __riscv_vredsum_vs_i32m4_i32m1(_sum0, _sum00, vlm4);
            sum0 += __riscv_vmv_x_s_i32m1_i32(_sum00);

            vint32m1_t _sum11 = __riscv_vmv_v_x_i32m1(0, vlm1);
            _sum11 = __riscv_vredsum_vs_i32m4_i32m1(_sum1, _sum11, vlm4);
            sum1 += __riscv_vmv_x_s_i32m1_i32(_sum11);
#endif
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i0 * stride_h) + j0 * stride_w * elempack;
                const signed char* r1 = bottom_blob.channel(q).row<const signed char>(i1 * stride_h) + j1 * stride_w * elempack;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    const signed char* r1s = r1 + space_ofs[k];
                    sum0 += r0s[0] * kptr[0];
                    sum1 += r1s[0] * kptr[0];
                    kptr++;
                }
            }
            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr += 2;
        }

        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;
#if __riscv_vector
            const signed char* kptr = weight_data_tm.channel(p / pack4n + (p % pack4n) / packn + p % packn);
#else
            const signed char* kptr = weight_data_tm.channel(p);
#endif
            int q = 0;
#if __riscv_vector
            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vlm4);

            for (; q + pack4n - 1 < inch; q += pack4n)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    vint8m1_t _r = __riscv_vlse8_v_i8m1(r0s, N, vlm4);
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm4);
                    vint16m2_t _s = __riscv_vwmul_vv_i16m2(_w, _r, vlm4);
                    _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm4);
                    kptr += pack4n;
                }
            }

            vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, vlm1);
            _sum0 = __riscv_vredsum_vs_i32m4_i32m1(_sum, _sum0, vlm4);
            sum += __riscv_vmv_x_s_i32m1_i32(_sum0);
#endif
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    sum += r0s[0] * kptr[0];
                    kptr++;
                }
            }
            outptr[0] = sum;
            outptr += 1;
        }
    }

    return;
}