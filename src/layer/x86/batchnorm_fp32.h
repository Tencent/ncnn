// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void batchnorm_fp32_fma(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void batchnorm_fp32_fma4(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt);
#endif

static void batchnorm_fp32(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        batchnorm_fp32_fma(bottom_top_blob, a_data, b_data, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        batchnorm_fp32_fma4(bottom_top_blob, a_data, b_data, opt);
        return;
    }
#endif

    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;

        const int size = w * elempack;

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
            __m512 _p512 = _mm512_loadu_ps(ptr + i);
            __m512 _a512 = _mm512_loadu_ps(aptr + i);
            __m512 _b512 = _mm512_loadu_ps(bptr + i);
            _mm512_storeu_ps(ptr + i, _mm512_fmadd_ps(_p512, _b512, _a512));
        }
        remain_size_start += nn_size * 16;
#endif // __AVX512F__
        nn_size = (size - remain_size_start) / 8;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;
            __m256 _p256 = _mm256_loadu_ps(ptr + i);
            __m256 _a256 = _mm256_loadu_ps(aptr + i);
            __m256 _b256 = _mm256_loadu_ps(bptr + i);
            _mm256_storeu_ps(ptr + i, _mm256_comp_fmadd_ps(_p256, _b256, _a256));
        }
        remain_size_start += nn_size * 8;
#endif // __AVX__
        nn_size = (size - remain_size_start) / 4;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;
            __m128 _p128 = _mm_loadu_ps(ptr + i);
            __m128 _a128 = _mm_loadu_ps(aptr + i);
            __m128 _b128 = _mm_loadu_ps(bptr + i);
            _mm_storeu_ps(ptr + i, _mm_comp_fmadd_ps(_p128, _b128, _a128));
        }
        remain_size_start += nn_size * 4;
#endif // __SSE2__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            ptr[i] = bptr[i] * ptr[i] + aptr[i];
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

#if __SSE2__
            __m128 _a128 = (elempack == 4) ? _mm_loadu_ps((const float*)a_data + i * 4) : _mm_set1_ps(a);
            __m128 _b128 = (elempack == 4) ? _mm_loadu_ps((const float*)b_data + i * 4) : _mm_set1_ps(b);
#if __AVX__
            __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps((const float*)a_data + i * 8) : combine4x2_ps(_a128, _a128);
            __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps((const float*)b_data + i * 8) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
            __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps((const float*)a_data + i * 16) : combine8x2_ps(_a256, _a256);
            __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps((const float*)b_data + i * 16) : combine8x2_ps(_b256, _b256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; j + 15 < size; j += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
                _mm512_storeu_ps(ptr, _p512);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < size; j += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
                _mm256_storeu_ps(ptr, _p256);
                ptr += 8;
            }
#endif // __AVX__
            for (; j + 3 < size; j += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
                _mm_storeu_ps(ptr, _p128);
                ptr += 4;
            }
#endif // __SSE__
            for (; j < size; j++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

#if __SSE2__
            __m128 _a128 = (elempack == 4) ? _mm_loadu_ps((const float*)a_data + q * 4) : _mm_set1_ps(a);
            __m128 _b128 = (elempack == 4) ? _mm_loadu_ps((const float*)b_data + q * 4) : _mm_set1_ps(b);
#if __AVX__
            __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps((const float*)a_data + q * 8) : combine4x2_ps(_a128, _a128);
            __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps((const float*)b_data + q * 8) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
            __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps((const float*)a_data + q * 16) : combine8x2_ps(_a256, _a256);
            __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps((const float*)b_data + q * 16) : combine8x2_ps(_b256, _b256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
                _mm512_storeu_ps(ptr, _p512);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
                _mm256_storeu_ps(ptr, _p256);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
                _mm_storeu_ps(ptr, _p128);
                ptr += 4;
            }
#endif // __SSE__
            for (; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }
}
