// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

static void pooling_global_max_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __mips_msa
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            v4f32 _max = bfloat2float_msa(ptr);
            ptr += 4;
            for (int i = 1; i < size; i++)
            {
                v4f32 _val = bfloat2float_msa(ptr);
                _max = __msa_fmax_w(_max, _val);
                ptr += 4;
            }

            unsigned short* outptr = top_blob;
            float2bfloat_msa_store(_max, outptr + q * 4);
        }

        return;
    }
#endif // __mips_msa

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            float max_val = bfloat16_to_float32(ptr[0]);
            for (int i = 1; i < size; i++)
            {
                float val = bfloat16_to_float32(ptr[i]);
                max_val = std::max(max_val, val);
            }

            unsigned short* outptr = top_blob;
            outptr[q] = float32_to_bfloat16(max_val);
        }
    }
}

static void pooling_global_avg_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __mips_msa
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            v4f32 _sum = (v4f32)__msa_fill_w(0);
            for (int i = 0; i < size; i++)
            {
                v4f32 _val = bfloat2float_msa(ptr);
                _sum = __msa_fadd_w(_sum, _val);
                ptr += 4;
            }

            v4f32 _inv_size = __msa_fill_w_f32(1.f / size);
            v4f32 _avg = __msa_fmul_w(_sum, _inv_size);

            unsigned short* outptr = top_blob;
            float2bfloat_msa_store(_avg, outptr + q * 4);
        }

        return;
    }
#endif // __mips_msa

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);

            float sum = 0.f;
            for (int i = 0; i < size; i++)
            {
                sum += bfloat16_to_float32(ptr[i]);
            }

            unsigned short* outptr = top_blob;
            outptr[q] = float32_to_bfloat16(sum / size);
        }
    }
}

static void pooling_max_bf16s_msa(const Mat& bottom_blob_bordered, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob_bordered.w;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

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

#if __mips_msa
    if (elempack == 4)
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
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                    v4f32 _max = bfloat2float_msa(sptr);

                    for (int k = 0; k < maxk; k++)
                    {
                        v4f32 _val = bfloat2float_msa(sptr + space_ofs[k] * 4);
                        _max = __msa_fmax_w(_max, _val);
                    }

                    float2bfloat_msa_store(_max, outptr + j * 4);
                }

                outptr += outw * 4;
            }
        }

        return;
    }
#endif // __mips_msa

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

                    float max_val = bfloat16_to_float32(sptr[0]);

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = bfloat16_to_float32(sptr[space_ofs[k]]);
                        max_val = std::max(max_val, val);
                    }

                    outptr[j] = float32_to_bfloat16(max_val);
                }

                outptr += outw;
            }
        }
    }
}

static void pooling_avg_bf16s_msa(const Mat& bottom_blob_bordered, const Mat& bottom_blob, Mat& top_blob, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_mode, int avgpool_count_include_pad, const Option& opt)
{
    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

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

    if (avgpool_count_include_pad == 0)
    {
        int wtailpad = 0;
        int htailpad = 0;

        if (pad_mode == 0) // full padding
        {
            wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
            htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
        }

#if __mips_msa
        if (elempack == 4)
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

                        v4f32 _sum = (v4f32)__msa_fill_w(0);
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

                                v4f32 _val = bfloat2float_msa(m.row<const unsigned short>(sy) + sx * 4);
                                _sum = __msa_fadd_w(_sum, _val);
                                area += 1;
                            }
                        }

                        v4f32 _inv_area = __msa_fill_w_f32(1.f / area);
                        v4f32 _avg = __msa_fmul_w(_sum, _inv_area);
                        float2bfloat_msa_store(_avg, outptr + j * 4);
                    }

                    outptr += outw * 4;
                }
            }

            return;
        }
#endif // __mips_msa

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
#if __mips_msa
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                v4f32 _inv_maxk = __msa_fill_w_f32(1.f / maxk);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                        v4f32 _sum = (v4f32)__msa_fill_w(0);

                        for (int k = 0; k < maxk; k++)
                        {
                            v4f32 _val = bfloat2float_msa(sptr + space_ofs[k] * 4);
                            _sum = __msa_fadd_w(_sum, _val);
                        }

                        v4f32 _avg = __msa_fmul_w(_sum, _inv_maxk);
                        float2bfloat_msa_store(_avg, outptr + j * 4);
                    }

                    outptr += outw * 4;
                }
            }

            return;
        }
#endif // __mips_msa

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
