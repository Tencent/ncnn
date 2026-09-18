// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int normalize_fp32_fma(Mat& bottom_top_blob, const Mat& scale_data, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
int normalize_fp32_fma4(Mat& bottom_top_blob, const Mat& scale_data, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode, const Option& opt);
#endif

static int normalize_fp32(Mat& bottom_top_blob, const Mat& scale_data, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return normalize_fp32_fma(bottom_top_blob, scale_data, across_spatial, across_channel, channel_shared, eps, eps_mode, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
        return normalize_fp32_fma4(bottom_top_blob, scale_data, across_spatial, across_channel, channel_shared, eps, eps_mode, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d;

    if (dims == 1 || dims == 2)
    {
        // packing is along the spatial dimensions, not channels
        size *= elempack;
        elempack = 1;
    }

    if (across_spatial && across_channel)
    {
        Mat square_sum_blob;
        square_sum_blob.create(channels, 4u, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            const int n = size * elempack;
            float ssum = 0.f;
            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _ssum_avx512 = _mm512_setzero_ps();
            for (; i + 15 < n; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                _ssum_avx512 = _mm512_fmadd_ps(_p, _p, _ssum_avx512);
                ptr += 16;
            }
            ssum += _mm512_comp_reduce_add_ps(_ssum_avx512);
#endif // __AVX512F__
            __m256 _ssum_avx = _mm256_setzero_ps();
            for (; i + 7 < n; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _ssum_avx = _mm256_comp_fmadd_ps(_p, _p, _ssum_avx);
                ptr += 8;
            }
            ssum += _mm256_reduce_add_ps(_ssum_avx);
#endif // __AVX__
            __m128 _ssum = _mm_setzero_ps();
            for (; i + 3 < n; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                _ssum = _mm_comp_fmadd_ps(_p, _p, _ssum);
                ptr += 4;
            }
            ssum += _mm_reduce_add_ps(_ssum);
#endif // __SSE2__
            for (; i < n; i++)
            {
                ssum += ptr[0] * ptr[0];
                ptr++;
            }
            square_sum_blob[q] = ssum;
        }

        float a = 0.f;
        for (int q = 0; q < channels; q++)
        {
            a += square_sum_blob[q];
        }

        if (eps_mode == 0) // caffe/mxnet
            a = 1.f / sqrtf(a + eps);
        else if (eps_mode == 1) // pytorch
            a = 1.f / std::max(sqrtf(a), eps);
        else // if (eps_mode == 2) // tensorflow
            a = 1.f / sqrtf(std::max(a, eps));

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * elempack);
            const float scale = a * scale_ptr[0];
            const int n = size * elempack;

#if __SSE2__
            __m128 _scale = _mm_set1_ps(scale);
            if (!channel_shared && elempack == 4)
                _scale = _mm_mul_ps(_mm_set1_ps(a), _mm_loadu_ps(scale_ptr));
#if __AVX__
            __m256 _scale_avx = combine4x2_ps(_scale, _scale);
            if (!channel_shared && elempack == 8)
                _scale_avx = _mm256_mul_ps(_mm256_set1_ps(a), _mm256_loadu_ps(scale_ptr));
#if __AVX512F__
            __m512 _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
            if (!channel_shared && elempack == 16)
                _scale_avx512 = _mm512_mul_ps(_mm512_set1_ps(a), _mm512_loadu_ps(scale_ptr));
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < n; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                _p = _mm512_mul_ps(_p, _scale_avx512);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < n; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_mul_ps(_p, _scale_avx);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < n; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                _p = _mm_mul_ps(_p, _scale);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < n; i++)
            {
                ptr[0] *= scale;
                ptr++;
            }
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * elempack);
            const int n = size * elempack;

            float ssum = 0.f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _ssum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
            __m256 _ssum_avx = _mm256_setzero_ps();
#endif // __AVX__
            __m128 _ssum = _mm_setzero_ps();
#endif // __SSE2__

            {
                const float* ptr0 = ptr;
                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < n; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr0);
                    _ssum_avx512 = _mm512_fmadd_ps(_p, _p, _ssum_avx512);
                    ptr0 += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < n; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr0);
                    _ssum_avx = _mm256_comp_fmadd_ps(_p, _p, _ssum_avx);
                    ptr0 += 8;
                }
#endif // __AVX__
                for (; i + 3 < n; i += 4)
                {
                    __m128 _p = _mm_loadu_ps(ptr0);
                    _ssum = _mm_comp_fmadd_ps(_p, _p, _ssum);
                    ptr0 += 4;
                }
#endif // __SSE2__
                for (; i < n; i++)
                {
                    ssum += ptr0[0] * ptr0[0];
                    ptr0++;
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack < 16)
                {
                    __m256 _ssum0 = _mm512_castps512_ps256(_ssum_avx512);
                    __m256 _ssum1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_ssum_avx512), 1));
                    _ssum_avx = _mm256_add_ps(_ssum_avx, _ssum0);
                    _ssum_avx = _mm256_add_ps(_ssum_avx, _ssum1);
                }
#endif // __AVX512F__
                if (elempack < 8)
                {
                    __m128 _ssum0 = _mm256_castps256_ps128(_ssum_avx);
                    __m128 _ssum1 = _mm256_extractf128_ps(_ssum_avx, 1);
                    _ssum = _mm_add_ps(_ssum, _ssum0);
                    _ssum = _mm_add_ps(_ssum, _ssum1);
                }
#endif // __AVX__
#endif // __SSE2__

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    __m512 _eps = _mm512_set1_ps(eps);
                    __m512 _one = _mm512_set1_ps(1.f);

                    if (eps_mode == 0) // caffe/mxnet
                        _ssum_avx512 = _mm512_div_ps(_one, _mm512_sqrt_ps(_mm512_add_ps(_ssum_avx512, _eps)));
                    else if (eps_mode == 1) // pytorch
                        _ssum_avx512 = _mm512_div_ps(_one, _mm512_max_ps(_eps, _mm512_sqrt_ps(_ssum_avx512)));
                    else // if (eps_mode == 2) // tensorflow
                        _ssum_avx512 = _mm512_div_ps(_one, _mm512_sqrt_ps(_mm512_max_ps(_eps, _ssum_avx512)));

                    __m512 _scale = channel_shared ? _mm512_set1_ps(scale_ptr[0]) : _mm512_loadu_ps(scale_ptr);
                    _ssum_avx512 = _mm512_mul_ps(_ssum_avx512, _scale);
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    __m256 _eps = _mm256_set1_ps(eps);
                    __m256 _one = _mm256_set1_ps(1.f);

                    if (eps_mode == 0) // caffe/mxnet
                        _ssum_avx = _mm256_div_ps(_one, _mm256_sqrt_ps(_mm256_add_ps(_ssum_avx, _eps)));
                    else if (eps_mode == 1) // pytorch
                        _ssum_avx = _mm256_div_ps(_one, _mm256_max_ps(_eps, _mm256_sqrt_ps(_ssum_avx)));
                    else // if (eps_mode == 2) // tensorflow
                        _ssum_avx = _mm256_div_ps(_one, _mm256_sqrt_ps(_mm256_max_ps(_eps, _ssum_avx)));

                    __m256 _scale = channel_shared ? _mm256_set1_ps(scale_ptr[0]) : _mm256_loadu_ps(scale_ptr);
                    _ssum_avx = _mm256_mul_ps(_ssum_avx, _scale);
#if __AVX512F__
                    _ssum_avx512 = combine8x2_ps(_ssum_avx, _ssum_avx);
#endif // __AVX512F__
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _eps = _mm_set1_ps(eps);
                    __m128 _one = _mm_set1_ps(1.f);

                    if (eps_mode == 0) // caffe/mxnet
                        _ssum = _mm_div_ps(_one, _mm_sqrt_ps(_mm_add_ps(_ssum, _eps)));
                    else if (eps_mode == 1) // pytorch
                        _ssum = _mm_div_ps(_one, _mm_max_ps(_eps, _mm_sqrt_ps(_ssum)));
                    else // if (eps_mode == 2) // tensorflow
                        _ssum = _mm_div_ps(_one, _mm_sqrt_ps(_mm_max_ps(_eps, _ssum)));

                    __m128 _scale = channel_shared ? _mm_set1_ps(scale_ptr[0]) : _mm_loadu_ps(scale_ptr);
                    _ssum = _mm_mul_ps(_ssum, _scale);
#if __AVX__
                    _ssum_avx = combine4x2_ps(_ssum, _ssum);
#if __AVX512F__
                    _ssum_avx512 = combine8x2_ps(_ssum_avx, _ssum_avx);
#endif // __AVX512F__
#endif // __AVX__
                }
#endif // __SSE2__
                if (elempack == 1)
                {
#if __SSE2__
                    ssum += _mm_reduce_add_ps(_ssum);
#endif // __SSE2__

                    if (eps_mode == 0) // caffe/mxnet
                        ssum = 1.f / sqrtf(ssum + eps);
                    else if (eps_mode == 1) // pytorch
                        ssum = 1.f / std::max(sqrtf(ssum), eps);
                    else // if (eps_mode == 2) // tensorflow
                        ssum = 1.f / sqrtf(std::max(ssum, eps));
                    ssum *= scale_ptr[0];
#if __SSE2__
                    _ssum = _mm_set1_ps(ssum);
#if __AVX__
                    _ssum_avx = _mm256_set1_ps(ssum);
#if __AVX512F__
                    _ssum_avx512 = _mm512_set1_ps(ssum);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
                }
            }

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < n; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                _p = _mm512_mul_ps(_p, _ssum_avx512);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < n; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_mul_ps(_p, _ssum_avx);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < n; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                _p = _mm_mul_ps(_p, _ssum);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < n; i++)
            {
                ptr[0] *= ssum;
                ptr++;
            }
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        Mat square_sum_blob;
        square_sum_blob.create(size, 4u, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        float* square_sum = square_sum_blob;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                __m512 _ssum = _mm512_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i * 16;
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _ssum = _mm512_fmadd_ps(_p, _p, _ssum);
                }
                float ssum = _mm512_comp_reduce_add_ps(_ssum);
                if (eps_mode == 0) // caffe/mxnet
                    ssum = 1.f / sqrtf(ssum + eps);
                else if (eps_mode == 1) // pytorch
                    ssum = 1.f / std::max(sqrtf(ssum), eps);
                else // if (eps_mode == 2) // tensorflow
                    ssum = 1.f / sqrtf(std::max(ssum, eps));
                square_sum[i] = ssum;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                __m256 _ssum = _mm256_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i * 8;
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _ssum = _mm256_comp_fmadd_ps(_p, _p, _ssum);
                }
                float ssum = _mm256_reduce_add_ps(_ssum);
                if (eps_mode == 0) // caffe/mxnet
                    ssum = 1.f / sqrtf(ssum + eps);
                else if (eps_mode == 1) // pytorch
                    ssum = 1.f / std::max(sqrtf(ssum), eps);
                else // if (eps_mode == 2) // tensorflow
                    ssum = 1.f / sqrtf(std::max(ssum, eps));
                square_sum[i] = ssum;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                __m128 _ssum = _mm_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i * 4;
                    __m128 _p = _mm_loadu_ps(ptr);
                    _ssum = _mm_comp_fmadd_ps(_p, _p, _ssum);
                }
                float ssum = _mm_reduce_add_ps(_ssum);
                if (eps_mode == 0) // caffe/mxnet
                    ssum = 1.f / sqrtf(ssum + eps);
                else if (eps_mode == 1) // pytorch
                    ssum = 1.f / std::max(sqrtf(ssum), eps);
                else // if (eps_mode == 2) // tensorflow
                    ssum = 1.f / sqrtf(std::max(ssum, eps));
                square_sum[i] = ssum;
            }
        }
#endif // __SSE2__
        if (elempack == 1)
        {
            int nn_size = 0;
            int remain_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            nn_size = (size - remain_size_start) / 16;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 16;
                __m512 _ssum = _mm512_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i;
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _ssum = _mm512_fmadd_ps(_p, _p, _ssum);
                }
                __m512 _eps = _mm512_set1_ps(eps);
                __m512 _one = _mm512_set1_ps(1.f);
                if (eps_mode == 0) // caffe/mxnet
                    _ssum = _mm512_div_ps(_one, _mm512_sqrt_ps(_mm512_add_ps(_ssum, _eps)));
                else if (eps_mode == 1) // pytorch
                    _ssum = _mm512_div_ps(_one, _mm512_max_ps(_eps, _mm512_sqrt_ps(_ssum)));
                else // if (eps_mode == 2) // tensorflow
                    _ssum = _mm512_div_ps(_one, _mm512_sqrt_ps(_mm512_max_ps(_eps, _ssum)));
                _mm512_storeu_ps(square_sum + i, _ssum);
            }
            remain_size_start += nn_size * 16;
#endif // __AVX512F__
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _ssum = _mm256_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i;
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _ssum = _mm256_comp_fmadd_ps(_p, _p, _ssum);
                }
                __m256 _eps = _mm256_set1_ps(eps);
                __m256 _one = _mm256_set1_ps(1.f);
                if (eps_mode == 0) // caffe/mxnet
                    _ssum = _mm256_div_ps(_one, _mm256_sqrt_ps(_mm256_add_ps(_ssum, _eps)));
                else if (eps_mode == 1) // pytorch
                    _ssum = _mm256_div_ps(_one, _mm256_max_ps(_eps, _mm256_sqrt_ps(_ssum)));
                else // if (eps_mode == 2) // tensorflow
                    _ssum = _mm256_div_ps(_one, _mm256_sqrt_ps(_mm256_max_ps(_eps, _ssum)));
                _mm256_storeu_ps(square_sum + i, _ssum);
            }
            remain_size_start += nn_size * 8;
#endif // __AVX__
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _ssum = _mm_setzero_ps();
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = (const float*)bottom_top_blob.channel(q) + i;
                    __m128 _p = _mm_loadu_ps(ptr);
                    _ssum = _mm_comp_fmadd_ps(_p, _p, _ssum);
                }
                __m128 _eps = _mm_set1_ps(eps);
                __m128 _one = _mm_set1_ps(1.f);
                if (eps_mode == 0) // caffe/mxnet
                    _ssum = _mm_div_ps(_one, _mm_sqrt_ps(_mm_add_ps(_ssum, _eps)));
                else if (eps_mode == 1) // pytorch
                    _ssum = _mm_div_ps(_one, _mm_max_ps(_eps, _mm_sqrt_ps(_ssum)));
                else // if (eps_mode == 2) // tensorflow
                    _ssum = _mm_div_ps(_one, _mm_sqrt_ps(_mm_max_ps(_eps, _ssum)));
                _mm_storeu_ps(square_sum + i, _ssum);
            }
            remain_size_start += nn_size * 4;
#endif // __SSE2__
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                float ssum = 0.f;
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_top_blob.channel(q);
                    ssum += ptr[i] * ptr[i];
                }
                if (eps_mode == 0) // caffe/mxnet
                    ssum = 1.f / sqrtf(ssum + eps);
                else if (eps_mode == 1) // pytorch
                    ssum = 1.f / std::max(sqrtf(ssum), eps);
                else // if (eps_mode == 2) // tensorflow
                    ssum = 1.f / sqrtf(std::max(ssum, eps));
                square_sum[i] = ssum;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float* scale_ptr = (const float*)scale_data + (channel_shared ? 0 : q * elempack);
            const float* square_sum_ptr = square_sum;
            const int n = size * elempack;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                __m512 _scale = channel_shared ? _mm512_set1_ps(scale_ptr[0]) : _mm512_loadu_ps(scale_ptr);
                for (; i + 15 < n; i += 16)
                {
                    __m512 _ssum = _mm512_set1_ps(square_sum_ptr[0]);
                    _ssum = _mm512_mul_ps(_ssum, _scale);
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_mul_ps(_p, _ssum);
                    _mm512_storeu_ps(ptr, _p);
                    ptr += 16;
                    square_sum_ptr++;
                }
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                __m256 _scale = channel_shared ? _mm256_set1_ps(scale_ptr[0]) : _mm256_loadu_ps(scale_ptr);
                for (; i + 7 < n; i += 8)
                {
                    __m256 _ssum = _mm256_set1_ps(square_sum_ptr[0]);
                    _ssum = _mm256_mul_ps(_ssum, _scale);
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_mul_ps(_p, _ssum);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                    square_sum_ptr++;
                }
            }
#endif // __AVX__
            if (elempack == 4)
            {
                __m128 _scale = channel_shared ? _mm_set1_ps(scale_ptr[0]) : _mm_loadu_ps(scale_ptr);
                for (; i + 3 < n; i += 4)
                {
                    __m128 _ssum = _mm_set1_ps(square_sum_ptr[0]);
                    _ssum = _mm_mul_ps(_ssum, _scale);
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_mul_ps(_p, _ssum);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                    square_sum_ptr++;
                }
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                const float scale = scale_ptr[0];
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _scale_avx512 = _mm512_set1_ps(scale);
                for (; i + 15 < n; i += 16)
                {
                    __m512 _ssum = _mm512_loadu_ps(square_sum_ptr);
                    _ssum = _mm512_mul_ps(_ssum, _scale_avx512);
                    __m512 _p = _mm512_loadu_ps(ptr);
                    _p = _mm512_mul_ps(_p, _ssum);
                    _mm512_storeu_ps(ptr, _p);
                    ptr += 16;
                    square_sum_ptr += 16;
                }
#endif // __AVX512F__
                __m256 _scale_avx = _mm256_set1_ps(scale);
                for (; i + 7 < n; i += 8)
                {
                    __m256 _ssum = _mm256_loadu_ps(square_sum_ptr);
                    _ssum = _mm256_mul_ps(_ssum, _scale_avx);
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_mul_ps(_p, _ssum);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                    square_sum_ptr += 8;
                }
#endif // __AVX__
                __m128 _scale = _mm_set1_ps(scale);
                for (; i + 3 < n; i += 4)
                {
                    __m128 _ssum = _mm_loadu_ps(square_sum_ptr);
                    _ssum = _mm_mul_ps(_ssum, _scale);
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_mul_ps(_p, _ssum);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                    square_sum_ptr += 4;
                }
#endif // __SSE2__
                for (; i < n; i++)
                {
                    ptr[0] *= square_sum_ptr[0] * scale;
                    ptr++;
                    square_sum_ptr++;
                }
            }
        }

        return 0;
    }

    return 0;
}
