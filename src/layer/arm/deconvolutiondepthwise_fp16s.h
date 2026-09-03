// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

static void deconvolutiondepthwise_fp16s(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob_bordered.w;
    const int outh = top_blob_bordered.h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int maxk = kernel_w * kernel_h;

    if (elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob_bordered.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32((const float*)bias_data + g * 4);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                int k = y * kernel_w + x;

                                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                _sum = vfmaq_f32(_sum, _val, _w);
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob_bordered.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[g];
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float val = (float)sptr[sx];

                                int k = y * kernel_w + x;

                                float w = (float)kptr[k];

                                sum += val * w;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
}

static void deconvolutiondepthwise_fp16sa(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob_bordered.w;
    const int outh = top_blob_bordered.h;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int maxk = kernel_w * kernel_h;

    if (elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob_bordered.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 8;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16((const __fp16*)bias_data_fp16 + g * 8);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                float16x8_t _val = vld1q_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x8_t _w = vld1q_f16(kptr + k * 8);

                                _sum = vfmaq_f16(_sum, _val, _w);
                            }
                        }

                        _sum = activation_ps_f16(_sum, activation_type, activation_params);

                        vst1q_f16(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob_bordered.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16((const __fp16*)bias_data_fp16 + g * 4);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                float16x4_t _val = vld1_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x4_t _w = vld1_f16(kptr + k * 4);

                                _sum = vfma_f16(_sum, _val, _w);
                            }
                        }

                        _sum = activation_ps_f16(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob_bordered.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[g];
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                __fp16 val = sptr[sx];

                                int k = y * kernel_w + x;

                                __fp16 w = kptr[k];

                                sum += val * w;
                            }
                        }

                        sum = activation_ss_f16(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
