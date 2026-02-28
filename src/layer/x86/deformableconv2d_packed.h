// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deformableconv2d_packed(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int activation_type, const Mat& activation_params, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    const bool offset_not_pack = offset.elempack == 1;
    const bool mask_not_pack = has_mask ? bottom_blobs[2].elempack == 1 : true;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;
    const int out_elempack = top_blob.elempack;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int h_col = 0; h_col < outh; h_col++)
    {
        for (int w_col = 0; w_col < outw; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            for (int oc = 0; oc < outch; oc++)
            {
                const float* kptr = weight_data_packed.channel(oc);
                float* outptr = top_blob.channel(oc);

#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _sum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
                __m256 _sum_avx = _mm256_setzero_ps();
#endif // __AVX__
                __m128 _sum_sse = _mm_setzero_ps();
#endif // __SSE2__
                float _sum_scalar = 0.f;

                if (bias_data_ptr)
                {
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    if (out_elempack == 16)
                        _sum_avx512 = _mm512_load_ps(bias_data_ptr + oc * out_elempack);
#endif // __AVX512F__
                    if (out_elempack == 8)
                        _sum_avx = _mm256_load_ps(bias_data_ptr + oc * out_elempack);
#endif // __AVX__
                    if (out_elempack == 4)
                        _sum_sse = _mm_load_ps(bias_data_ptr + oc * out_elempack);
#endif // __SSE2__
                    if (out_elempack == 1)
                        _sum_scalar = *(bias_data_ptr + oc);
                }

                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        float offset_h = 0.f;
                        float offset_w = 0.f;
                        float mask_ = 1.f;
                        if (offset_not_pack)
                        {
                            offset_h = offset.channel((i * kernel_w + j) * 2).row(h_col)[w_col];
                            offset_w = offset.channel((i * kernel_w + j) * 2 + 1).row(h_col)[w_col];
                        }
                        else
                        {
                            const int y_c = (i * kernel_w + j) * 2;
                            const int x_c = (i * kernel_w + j) * 2 + 1;
                            offset_h = offset.channel(y_c / offset.elempack).row(h_col)[w_col * offset.elempack + y_c % offset.elempack];
                            offset_w = offset.channel(x_c / offset.elempack).row(h_col)[w_col * offset.elempack + x_c % offset.elempack];
                        }
                        if (has_mask)
                        {
                            const Mat& mask = bottom_blobs[2];
                            if (mask_not_pack)
                            {
                                mask_ = mask.channel(i * kernel_w + j).row(h_col)[w_col];
                            }
                            else
                            {
                                const int m_c = i * kernel_w + j;
                                mask_ = mask.channel(m_c / mask.elempack).row(h_col)[w_col * mask.elempack + m_c % mask.elempack];
                            }
                        }
                        const float h_im = h_in + i * dilation_h + offset_h;
                        const float w_im = w_in + j * dilation_w + offset_w;

                        // Bilinear
                        const bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                        float bw1 = 0.f;
                        float bw2 = 0.f;
                        float bw3 = 0.f;
                        float bw4 = 0.f;
                        bool v1_cond = false;
                        bool v2_cond = false;
                        bool v3_cond = false;
                        bool v4_cond = false;
                        int v1_pos = 0;
                        int v2_pos = 0;
                        int v3_pos = 0;
                        int v4_pos = 0;
                        if (cond)
                        {
                            int h_low = (int)floorf(h_im);
                            int w_low = (int)floorf(w_im);
                            int h_high = h_low + 1;
                            int w_high = w_low + 1;

                            float lh = h_im - h_low;
                            float lw = w_im - w_low;
                            float hh = 1 - lh;
                            float hw = 1 - lw;

                            v1_cond = (h_low >= 0 && w_low >= 0);
                            v2_cond = (h_low >= 0 && w_high <= w - 1);
                            v3_cond = (h_high <= h - 1 && w_low >= 0);
                            v4_cond = (h_high <= h - 1 && w_high <= w - 1);
                            if (v1_cond)
                                v1_pos = h_low * w + w_low;
                            if (v2_cond)
                                v2_pos = h_low * w + w_high;
                            if (v3_cond)
                                v3_pos = h_high * w + w_low;
                            if (v4_cond)
                                v4_pos = h_high * w + w_high;

                            bw1 = hh * hw;
                            bw2 = hh * lw;
                            bw3 = lh * hw;
                            bw4 = lh * lw;
                        }

                        for (int ic = 0; ic < inch; ic++)
                        {
                            const float* data_im_ptr = bottom_blob.channel(ic);

#if __SSE2__
#if __AVX__
#if __AVX512F__
                            if (out_elempack == 16)
                            {
                                __m512 _bw1 = _mm512_set1_ps(bw1);
                                __m512 _bw2 = _mm512_set1_ps(bw2);
                                __m512 _bw3 = _mm512_set1_ps(bw3);
                                __m512 _bw4 = _mm512_set1_ps(bw4);
                                __m512 _mask = _mm512_set1_ps(mask_);

                                for (int ez = 0; ez < elempack; ez++)
                                {
                                    __m512 _val = _mm512_setzero_ps();
                                    if (cond)
                                    {
                                        __m512 _v1 = v1_cond ? _mm512_set1_ps(*(data_im_ptr + v1_pos * elempack + ez)) : _mm512_setzero_ps();
                                        __m512 _v2 = v2_cond ? _mm512_set1_ps(*(data_im_ptr + v2_pos * elempack + ez)) : _mm512_setzero_ps();
                                        __m512 _v3 = v3_cond ? _mm512_set1_ps(*(data_im_ptr + v3_pos * elempack + ez)) : _mm512_setzero_ps();
                                        __m512 _v4 = v4_cond ? _mm512_set1_ps(*(data_im_ptr + v4_pos * elempack + ez)) : _mm512_setzero_ps();

                                        _val = _mm512_fmadd_ps(_v1, _bw1, _val);
                                        _val = _mm512_fmadd_ps(_v2, _bw2, _val);
                                        _val = _mm512_fmadd_ps(_v3, _bw3, _val);
                                        _val = _mm512_fmadd_ps(_v4, _bw4, _val);
                                    }
                                    if (has_mask)
                                    {
                                        _val = _mm512_mul_ps(_val, _mask);
                                    }
                                    __m512 _conv_w = _mm512_load_ps(kptr);
                                    _sum_avx512 = _mm512_fmadd_ps(_val, _conv_w, _sum_avx512);
                                    kptr += 16;
                                }
                            }
#endif // __AVX512F__
                            if (out_elempack == 8)
                            {
                                __m256 _bw1 = _mm256_set1_ps(bw1);
                                __m256 _bw2 = _mm256_set1_ps(bw2);
                                __m256 _bw3 = _mm256_set1_ps(bw3);
                                __m256 _bw4 = _mm256_set1_ps(bw4);
                                __m256 _mask = _mm256_set1_ps(mask_);

                                for (int ez = 0; ez < elempack; ez++)
                                {
                                    __m256 _val = _mm256_setzero_ps();
                                    if (cond)
                                    {
                                        __m256 _v1 = v1_cond ? _mm256_broadcast_ss(data_im_ptr + v1_pos * elempack + ez) : _mm256_setzero_ps();
                                        __m256 _v2 = v2_cond ? _mm256_broadcast_ss(data_im_ptr + v2_pos * elempack + ez) : _mm256_setzero_ps();
                                        __m256 _v3 = v3_cond ? _mm256_broadcast_ss(data_im_ptr + v3_pos * elempack + ez) : _mm256_setzero_ps();
                                        __m256 _v4 = v4_cond ? _mm256_broadcast_ss(data_im_ptr + v4_pos * elempack + ez) : _mm256_setzero_ps();

                                        _val = _mm256_comp_fmadd_ps(_v1, _bw1, _val);
                                        _val = _mm256_comp_fmadd_ps(_v2, _bw2, _val);
                                        _val = _mm256_comp_fmadd_ps(_v3, _bw3, _val);
                                        _val = _mm256_comp_fmadd_ps(_v4, _bw4, _val);
                                    }
                                    if (has_mask)
                                    {
                                        _val = _mm256_mul_ps(_val, _mask);
                                    }
                                    __m256 _conv_w = _mm256_load_ps(kptr);
                                    _sum_avx = _mm256_comp_fmadd_ps(_val, _conv_w, _sum_avx);
                                    kptr += 8;
                                }
                            }
#endif // __AVX__
                            if (out_elempack == 4)
                            {
                                __m128 _bw1 = _mm_set1_ps(bw1);
                                __m128 _bw2 = _mm_set1_ps(bw2);
                                __m128 _bw3 = _mm_set1_ps(bw3);
                                __m128 _bw4 = _mm_set1_ps(bw4);
                                __m128 _mask = _mm_set1_ps(mask_);

                                for (int ez = 0; ez < elempack; ez++)
                                {
                                    __m128 _val = _mm_setzero_ps();
                                    if (cond)
                                    {
                                        __m128 _v1 = v1_cond ? _mm_load1_ps(data_im_ptr + v1_pos * elempack + ez) : _mm_setzero_ps();
                                        __m128 _v2 = v2_cond ? _mm_load1_ps(data_im_ptr + v2_pos * elempack + ez) : _mm_setzero_ps();
                                        __m128 _v3 = v3_cond ? _mm_load1_ps(data_im_ptr + v3_pos * elempack + ez) : _mm_setzero_ps();
                                        __m128 _v4 = v4_cond ? _mm_load1_ps(data_im_ptr + v4_pos * elempack + ez) : _mm_setzero_ps();

                                        _val = _mm_comp_fmadd_ps(_v1, _bw1, _val);
                                        _val = _mm_comp_fmadd_ps(_v2, _bw2, _val);
                                        _val = _mm_comp_fmadd_ps(_v3, _bw3, _val);
                                        _val = _mm_comp_fmadd_ps(_v4, _bw4, _val);
                                    }
                                    if (has_mask)
                                    {
                                        _val = _mm_mul_ps(_val, _mask);
                                    }
                                    __m128 _conv_w = _mm_load_ps(kptr);
                                    _sum_sse = _mm_comp_fmadd_ps(_val, _conv_w, _sum_sse);
                                    kptr += 4;
                                }
                            }
#endif // __SSE2__

                            if (out_elempack == 1)
                            {
                                for (int ez = 0; ez < elempack; ez++)
                                {
                                    float val = 0.f;
                                    if (cond)
                                    {
                                        float v1 = v1_cond ? *(data_im_ptr + v1_pos * elempack + ez) : 0.f;
                                        float v2 = v2_cond ? *(data_im_ptr + v2_pos * elempack + ez) : 0.f;
                                        float v3 = v3_cond ? *(data_im_ptr + v3_pos * elempack + ez) : 0.f;
                                        float v4 = v4_cond ? *(data_im_ptr + v4_pos * elempack + ez) : 0.f;
                                        val = bw1 * v1 + bw2 * v2 + bw3 * v3 + bw4 * v4;
                                    }
                                    if (has_mask)
                                    {
                                        val *= mask_;
                                    }
                                    _sum_scalar += val * *(kptr);
                                    kptr += 1;
                                }
                            }
                        }
                    }
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (out_elempack == 16)
                {
                    _sum_avx512 = activation_avx512(_sum_avx512, activation_type, activation_params);
                    _mm512_store_ps(outptr + (h_col * outw + w_col) * out_elempack, _sum_avx512);
                }
#endif // __AVX512F__
                if (out_elempack == 8)
                {
                    _sum_avx = activation_avx(_sum_avx, activation_type, activation_params);
                    _mm256_store_ps(outptr + (h_col * outw + w_col) * out_elempack, _sum_avx);
                }
#endif // __AVX__
                if (out_elempack == 4)
                {
                    _sum_sse = activation_sse(_sum_sse, activation_type, activation_params);
                    _mm_store_ps(outptr + (h_col * outw + w_col) * out_elempack, _sum_sse);
                }
#endif // __SSE2__
                if (out_elempack == 1)
                {
                    _sum_scalar = activation_ss(_sum_scalar, activation_type, activation_params);
                    *(outptr + h_col * outw + w_col) = _sum_scalar;
                }
            }
        }
    }
}
