// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void pooling_global_max_bf16s_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void pooling_global_avg_bf16s_zvfbfmin(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void pooling_max_bf16s_zvfbfmin(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt);
void pooling_avg_bf16s_zvfbfmin(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt);
#endif

#if __riscv_vector
#if __riscv_xtheadvector
// FIXME inline causes illegal instruction :(
__attribute__((noinline)) static vfloat32m8_t
pooling_bf16s_reset_tails(vfloat32m8_t x, size_t vl, float v)
{
    const size_t vlm8 = __riscv_vsetvlmax_e32m8();
    vbool4_t _vl_mask = __riscv_vmsgeu_vx_u32m8_b4(__riscv_vid_v_u32m8(vlm8), vl, vlm8);
    x = __riscv_vfmerge_vfm_f32m8(x, v, _vl_mask, vlm8);
    return x;
}
#endif // __riscv_xtheadvector
#endif // __riscv_vector

static void pooling_global_max_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        pooling_global_max_bf16s_zvfbfmin(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int size = bottom_blob.w * bottom_blob.h * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = (unsigned short*)top_blob + q * elempack;

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

        // full vl chunks keep every chunk aligned with the packn lanes
        const size_t vl = __riscv_vsetvlmax_e16m4();

        vfloat32m8_t _max = __riscv_vfmv_v_f_f32m8(-FLT_MAX, vl);
        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m4(remain);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlr), vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = pooling_bf16s_reset_tails(_p, vlr, -FLT_MAX);
            _max = __riscv_vfmax_vv_f32m8(_max, _p, vl);
#else  // __riscv_xtheadvector
            _max = __riscv_vfmax_vv_f32m8_tu(_max, _max, _p, vlr);
#endif // __riscv_xtheadvector
        }

        if (elempack == packn)
        {
            // reduce max nn,nn,nn,nn to nn
            const size_t vlm2 = __riscv_vsetvlmax_e32m2();

            vfloat32m4_t _max0 = __riscv_vfmax_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_max, 0), __riscv_vget_v_f32m8_f32m4(_max, 1), __riscv_vsetvlmax_e32m4());
            vfloat32m2_t _max2 = __riscv_vfmax_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_max0, 0), __riscv_vget_v_f32m4_f32m2(_max0, 1), vlm2);
            __riscv_vse16_v_u16m1(outptr, float2bfloat_rvv(_max2, vlm2), vlm2);
        }
        else // if (elempack == 1)
        {
            // reduce max n,n,n,n,n,n,n,n to 1
            vfloat32m1_t _max0 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
            _max0 = __riscv_vfredmax_vs_f32m8_f32m1(_max, _max0, vl);
            outptr[0] = float32_to_bfloat16(__riscv_vfmv_f_s_f32m1_f32(_max0));
        }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        float max = bfloat16_to_float32(ptr[0]);
        for (int i = 1; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(ptr[i]));
        }

        outptr[0] = float32_to_bfloat16(max);
#endif // __riscv_vector
    }
}

static void pooling_global_avg_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        pooling_global_avg_bf16s_zvfbfmin(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int elemcount = bottom_blob.w * bottom_blob.h;
    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        unsigned short* outptr = (unsigned short*)top_blob + q * elempack;

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        unsigned long old_fxcr = csrr_fxcr();
        // enable BF16 with round-to-nearest-even
        old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

        // full vl chunks keep every chunk aligned with the packn lanes
        const size_t vl = __riscv_vsetvlmax_e16m4();

        vfloat32m8_t _sum = __riscv_vfmv_v_f_f32m8(0.f, vl);
        int n = size / vl * vl;
        while (n > 0)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
            ptr += vl;
            n -= vl;
        }
        int remain = size % vl;
        if (remain > 0)
        {
            size_t vlr = __riscv_vsetvl_e16m4(remain);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vlr), vlr);
#if __riscv_xtheadvector
            // xtheadvector does not support tail undisturbed policy
            _p = pooling_bf16s_reset_tails(_p, vlr, 0.f);
            _sum = __riscv_vfadd_vv_f32m8(_sum, _p, vl);
#else  // __riscv_xtheadvector
            _sum = __riscv_vfadd_vv_f32m8_tu(_sum, _sum, _p, vlr);
#endif // __riscv_xtheadvector
        }

        if (elempack == packn)
        {
            // reduce sum nn,nn,nn,nn to nn
            const size_t vlm2 = __riscv_vsetvlmax_e32m2();

            vfloat32m4_t _sum0 = __riscv_vfadd_vv_f32m4(__riscv_vget_v_f32m8_f32m4(_sum, 0), __riscv_vget_v_f32m8_f32m4(_sum, 1), __riscv_vsetvlmax_e32m4());
            vfloat32m2_t _sum2 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(_sum0, 0), __riscv_vget_v_f32m4_f32m2(_sum0, 1), vlm2);
            vfloat32m2_t _avg2 = __riscv_vfmul_vf_f32m2(_sum2, 1.f / elemcount, vlm2);
            __riscv_vse16_v_u16m1(outptr, float2bfloat_rvv(_avg2, vlm2), vlm2);
        }
        else // if (elempack == 1)
        {
            // reduce sum n,n,n,n,n,n,n,n to 1
            vfloat32m1_t _sum0 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
            _sum0 = __riscv_vfredusum_vs_f32m8_f32m1(_sum, _sum0, vl);
            outptr[0] = float32_to_bfloat16(__riscv_vfmv_f_s_f32m1_f32(_sum0) / elemcount);
        }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
        // restore FXCR while retaining accrued exception flags, including FE
        csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
        float sum = 0.f;
        for (int i = 0; i < size; i++)
        {
            sum += bfloat16_to_float32(ptr[i]);
        }

        outptr[0] = float32_to_bfloat16(sum / elemcount);
#endif // __riscv_vector
    }
}

static void pooling_max_bf16s(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        pooling_max_bf16s_zvfbfmin(bottom_blob_bordered, top_blob, kernel_w, kernel_h, stride_w, stride_h, opt);
        return;
    }
#endif

    const int w = bottom_blob_bordered.w;
    const int channels = bottom_blob_bordered.c;
    const int elempack = bottom_blob_bordered.elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    if (elempack == packn)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            unsigned long old_fxcr = csrr_fxcr();
            // enable BF16 with round-to-nearest-even
            old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * packn;

                    vfloat32m2_t _max = bfloat2float_rvv(__riscv_vle16_v_u16m1(sptr, vl), vl);

                    for (int k = 1; k < maxk; k++)
                    {
                        vfloat32m2_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m1(sptr + space_ofs[k] * packn, vl), vl);
                        _max = __riscv_vfmax_vv_f32m2(_max, _val, vl);
                    }

                    __riscv_vse16_v_u16m1(outptr + j * packn, float2bfloat_rvv(_max, vl), vl);
                }

                outptr += outw * packn;
            }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
            // restore FXCR while retaining accrued exception flags, including FE
            csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
        }

        return;
    }
#endif // __riscv_vector

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                    float max = bfloat16_to_float32(sptr[0]);

                    for (int k = 1; k < maxk; k++)
                    {
                        float val = bfloat16_to_float32(sptr[space_ofs[k]]);
                        max = std::max(max, val);
                    }

                    outptr[j] = float32_to_bfloat16(max);
                }

                outptr += outw;
            }
        }
    }
}

static void pooling_avg_bf16s(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        pooling_avg_bf16s_zvfbfmin(bottom_blob_bordered, bottom_blob, top_blob, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, avgpool_count_include_pad, opt);
        return;
    }
#endif

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int channels = bottom_blob_bordered.c;
    const int elempack = bottom_blob_bordered.elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif

    if (avgpool_count_include_pad == 0)
    {
        int wtailpad = 0;
        int htailpad = 0;

        if (pad_mode == 0) // full padding
        {
            wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
            htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
        }

#if __riscv_vector
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                // enable BF16 with round-to-nearest-even
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                vfloat32m2_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m1(m.row<const unsigned short>(sy) + sx * packn, vl), vl);
                                _sum = __riscv_vfadd_vv_f32m2(_sum, _val, vl);
                                area += 1;
                            }
                        }

                        vfloat32m2_t _avg = __riscv_vfmul_vf_f32m2(_sum, 1.f / area, vl);
                        __riscv_vse16_v_u16m1(outptr + j * packn, float2bfloat_rvv(_avg, vl), vl);
                    }

                    outptr += outw * packn;
                }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                // restore FXCR while retaining accrued exception flags, including FE
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
            }

            return;
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        float sum = 0.f;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                sum += bfloat16_to_float32(m.row<const unsigned short>(sy)[sx]);
                                area += 1;
                            }
                        }

                        outptr[j] = float32_to_bfloat16(sum / area);
                    }

                    outptr += outw;
                }
            }
        }
    }
    else // if (avgpool_count_include_pad == 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const float inv_maxk = 1.f / maxk;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                // enable BF16 with round-to-nearest-even
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * packn;

                        vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat32m2_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m1(sptr + space_ofs[k] * packn, vl), vl);
                            _sum = __riscv_vfadd_vv_f32m2(_sum, _val, vl);
                        }

                        vfloat32m2_t _avg = __riscv_vfmul_vf_f32m2(_sum, inv_maxk, vl);
                        __riscv_vse16_v_u16m1(outptr + j * packn, float2bfloat_rvv(_avg, vl), vl);
                    }

                    outptr += outw * packn;
                }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                // restore FXCR while retaining accrued exception flags, including FE
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
            }

            return;
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                        float sum = 0.f;

                        for (int k = 0; k < maxk; k++)
                        {
                            sum += bfloat16_to_float32(sptr[space_ofs[k]]);
                        }

                        outptr[j] = float32_to_bfloat16(sum / maxk);
                    }

                    outptr += outw;
                }
            }
        }
    }
}
