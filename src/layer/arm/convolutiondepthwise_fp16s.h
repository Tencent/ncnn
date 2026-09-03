// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if NCNN_GNU_INLINE_ASM
#include "convolutiondepthwise_3x3_fp16s.h"
#include "convolutiondepthwise_3x3_pack8_fp16s.h"
#include "convolutiondepthwise_5x5_pack8_fp16s.h"
#endif // NCNN_GNU_INLINE_ASM

static int convolutiondepthwise_fp16s(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int group, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob_bordered.w;
    const int channels = bottom_blob_bordered.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int elempack = bottom_blob_bordered.elempack;

    if (elempack == 4)
    {
        {
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
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                        }

                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                        for (int k = 0; k < maxk; k++)
                        {
                            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));
                            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));
                            _sum = vfmaq_f32(_sum, _val, _w);
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
            for (int g = 0; g < group; g++)
            {
                __fp16* outptr = top_blob.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[g];

                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = (float)sptr[space_ofs[k]];
                            float w = (float)kptr[k];
                            sum += val * w;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

static int convolutiondepthwise_fp16sa(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int group, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob_bordered.w;
    const int channels = bottom_blob_bordered.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int elempack = bottom_blob_bordered.elempack;

    int activation_unfused = 0;

    if (elempack == 8)
    {
#if NCNN_GNU_INLINE_ASM
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convdw3x3s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convdw3x3s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convdw5x5s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convdw5x5s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else
#endif // NCNN_GNU_INLINE_ASM
        {
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
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 8;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f16(((const __fp16*)bias_data_fp16) + g * 8);
                        }

                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                        for (int k = 0; k < maxk; k++)
                        {
                            float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);
                            float16x8_t _w = vld1q_f16(kptr + k * 8);
                            _sum = vfmaq_f16(_sum, _val, _w);
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
            for (int g = 0; g < channels; g++)
            {
                __fp16* outptr = top_blob.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                        if (bias_term)
                        {
                            _sum = vld1_f16(((const __fp16*)bias_data_fp16) + g * 4);
                        }

                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                        for (int k = 0; k < maxk; k++)
                        {
                            float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);
                            float16x4_t _w = vld1_f16(kptr + k * 4);
                            _sum = vfma_f16(_sum, _val, _w);
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
#if NCNN_GNU_INLINE_ASM
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convdw3x3s1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convdw3x3s2_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            activation_unfused = 1;
        }
        else
#endif // NCNN_GNU_INLINE_ASM
        {
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
            for (int g = 0; g < group; g++)
            {
                __fp16* outptr = top_blob.channel(g);
                const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[g];

                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            __fp16 val = sptr[space_ofs[k]];
                            __fp16 w = kptr[k];
                            sum += val * w;
                        }

                        sum = activation_ss_f16(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return activation_unfused;
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
