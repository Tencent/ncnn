// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void deformableconv2d_im2col_fma(const Mat& bottom_blob, const Mat& offset_unpacked, const Mat& mask_unpacked, Mat& bottom_im2col, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int outw, int outh, int has_mask, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void deformableconv2d_im2col_fma4(const Mat& bottom_blob, const Mat& offset_unpacked, const Mat& mask_unpacked, Mat& bottom_im2col, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int outw, int outh, int has_mask, const Option& opt);
#endif

static void deformableconv2d_im2col(const Mat& bottom_blob, const Mat& offset_unpacked, const Mat& mask_unpacked, Mat& bottom_im2col, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int outw, int outh, int has_mask, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        deformableconv2d_im2col_fma(bottom_blob, offset_unpacked, mask_unpacked, bottom_im2col, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, outw, outh, has_mask, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        deformableconv2d_im2col_fma4(bottom_blob, offset_unpacked, mask_unpacked, bottom_im2col, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, outw, outh, has_mask, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int maxk = kernel_w * kernel_h;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.row(p * maxk);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                    const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                    const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float offset_h = offset_h_k.row(i)[j];
                            float offset_w = offset_w_k.row(i)[j];

                            int h_in = i * stride_h - pad_top;
                            int w_in = j * stride_w - pad_left;

                            const float h_im = h_in + u * dilation_h + offset_h;
                            const float w_im = w_in + v * dilation_w + offset_w;

                            // Bilinear
                            __m512 _val = _mm512_setzero_ps();
                            bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                            if (cond)
                            {
                                int h_low = floor(h_im);
                                int w_low = floor(w_im);
                                int h_high = h_low + 1;
                                int w_high = w_low + 1;

                                float lh = h_im - h_low;
                                float lw = w_im - w_low;
                                float hh = 1 - lh;
                                float hw = 1 - lw;

                                bool v1_cond = (h_low >= 0 && w_low >= 0);
                                bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                float w1 = hh * hw;
                                float w2 = hh * lw;
                                float w3 = lh * hw;
                                float w4 = lh * lw;

                                __m512 _v1 = v1_cond ? _mm512_load_ps(img.row(h_low) + w_low * 16) : _mm512_setzero_ps();
                                __m512 _v2 = v2_cond ? _mm512_load_ps(img.row(h_low) + w_high * 16) : _mm512_setzero_ps();
                                __m512 _v3 = v3_cond ? _mm512_load_ps(img.row(h_high) + w_low * 16) : _mm512_setzero_ps();
                                __m512 _v4 = v4_cond ? _mm512_load_ps(img.row(h_high) + w_high * 16) : _mm512_setzero_ps();

                                _val = _mm512_fmadd_ps(_v1, _mm512_set1_ps(w1), _val);
                                _val = _mm512_fmadd_ps(_v2, _mm512_set1_ps(w2), _val);
                                _val = _mm512_fmadd_ps(_v3, _mm512_set1_ps(w3), _val);
                                _val = _mm512_fmadd_ps(_v4, _mm512_set1_ps(w4), _val);

                                if (has_mask)
                                    _val = _mm512_mul_ps(_val, _mm512_set1_ps(mask_k.row(i)[j]));
                            }

                            _mm512_store_ps(ptr, _val);

                            ptr += 16;
                        }
                    }
                }
            }
        }
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.row(p * maxk);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                    const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                    const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float offset_h = offset_h_k.row(i)[j];
                            float offset_w = offset_w_k.row(i)[j];

                            int h_in = i * stride_h - pad_top;
                            int w_in = j * stride_w - pad_left;

                            const float h_im = h_in + u * dilation_h + offset_h;
                            const float w_im = w_in + v * dilation_w + offset_w;

                            // Bilinear
                            __m256 _val = _mm256_setzero_ps();
                            bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                            if (cond)
                            {
                                int h_low = floor(h_im);
                                int w_low = floor(w_im);
                                int h_high = h_low + 1;
                                int w_high = w_low + 1;

                                float lh = h_im - h_low;
                                float lw = w_im - w_low;
                                float hh = 1 - lh;
                                float hw = 1 - lw;

                                bool v1_cond = (h_low >= 0 && w_low >= 0);
                                bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                float w1 = hh * hw;
                                float w2 = hh * lw;
                                float w3 = lh * hw;
                                float w4 = lh * lw;

                                __m256 _v1 = v1_cond ? _mm256_load_ps(img.row(h_low) + w_low * 8) : _mm256_setzero_ps();
                                __m256 _v2 = v2_cond ? _mm256_load_ps(img.row(h_low) + w_high * 8) : _mm256_setzero_ps();
                                __m256 _v3 = v3_cond ? _mm256_load_ps(img.row(h_high) + w_low * 8) : _mm256_setzero_ps();
                                __m256 _v4 = v4_cond ? _mm256_load_ps(img.row(h_high) + w_high * 8) : _mm256_setzero_ps();

                                _val = _mm256_comp_fmadd_ps(_v1, _mm256_set1_ps(w1), _val);
                                _val = _mm256_comp_fmadd_ps(_v2, _mm256_set1_ps(w2), _val);
                                _val = _mm256_comp_fmadd_ps(_v3, _mm256_set1_ps(w3), _val);
                                _val = _mm256_comp_fmadd_ps(_v4, _mm256_set1_ps(w4), _val);

                                if (has_mask)
                                    _val = _mm256_mul_ps(_val, _mm256_set1_ps(mask_k.row(i)[j]));
                            }

                            _mm256_store_ps(ptr, _val);

                            ptr += 8;
                        }
                    }
                }
            }
        }
    }
#endif // __AVX__

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.row(p * maxk);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                    const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                    const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float offset_h = offset_h_k.row(i)[j];
                            float offset_w = offset_w_k.row(i)[j];

                            int h_in = i * stride_h - pad_top;
                            int w_in = j * stride_w - pad_left;

                            const float h_im = h_in + u * dilation_h + offset_h;
                            const float w_im = w_in + v * dilation_w + offset_w;

                            // Bilinear
                            __m128 _val = _mm_setzero_ps();
                            bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
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

                                bool v1_cond = (h_low >= 0 && w_low >= 0);
                                bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                float w1 = hh * hw;
                                float w2 = hh * lw;
                                float w3 = lh * hw;
                                float w4 = lh * lw;

                                __m128 _v1 = v1_cond ? _mm_load_ps(img.row(h_low) + w_low * 4) : _mm_setzero_ps();
                                __m128 _v2 = v2_cond ? _mm_load_ps(img.row(h_low) + w_high * 4) : _mm_setzero_ps();
                                __m128 _v3 = v3_cond ? _mm_load_ps(img.row(h_high) + w_low * 4) : _mm_setzero_ps();
                                __m128 _v4 = v4_cond ? _mm_load_ps(img.row(h_high) + w_high * 4) : _mm_setzero_ps();

                                _val = _mm_comp_fmadd_ps(_v1, _mm_set1_ps(w1), _val);
                                _val = _mm_comp_fmadd_ps(_v2, _mm_set1_ps(w2), _val);
                                _val = _mm_comp_fmadd_ps(_v3, _mm_set1_ps(w3), _val);
                                _val = _mm_comp_fmadd_ps(_v4, _mm_set1_ps(w4), _val);

                                if (has_mask)
                                    _val = _mm_mul_ps(_val, _mm_set1_ps(mask_k.row(i)[j]));
                            }

                            _mm_store_ps(ptr, _val);

                            ptr += 4;
                        }
                    }
                }
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.row(p * maxk);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                    const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                    const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float offset_h = offset_h_k.row(i)[j];
                            float offset_w = offset_w_k.row(i)[j];

                            int h_in = i * stride_h - pad_top;
                            int w_in = j * stride_w - pad_left;

                            const float h_im = h_in + u * dilation_h + offset_h;
                            const float w_im = w_in + v * dilation_w + offset_w;

                            // Bilinear
                            float val = 0.f;
                            bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
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

                                bool v1_cond = (h_low >= 0 && w_low >= 0);
                                bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                float w1 = hh * hw;
                                float w2 = hh * lw;
                                float w3 = lh * hw;
                                float w4 = lh * lw;

                                float v1 = v1_cond ? img.row(h_low)[w_low] : 0.f;
                                float v2 = v2_cond ? img.row(h_low)[w_high] : 0.f;
                                float v3 = v3_cond ? img.row(h_high)[w_low] : 0.f;
                                float v4 = v4_cond ? img.row(h_high)[w_high] : 0.f;
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;

                                if (has_mask)
                                    val *= mask_k.row(i)[j];
                            }

                            ptr[0] = val;

                            ptr += 1;
                        }
                    }
                }
            }
        }
    }
}
