#include "layernorm_x86.h"

#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

static NCNN_FORCEINLINE float fast_sum(float* ptr, int size)
{
    float sum = 0.0f;
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _sum = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _sum = _mm512_add_ps(_sum, _cur);
        }
        sum += _mm512_reduce_add_ps(_sum);
    }
#endif // __AVX512F__
    {
        __m256 _sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _sum = _mm256_add_ps(_sum, _cur);
        }
        sum += _sum[0] + _sum[1] + _sum[2] + _sum[3] + _sum[4] + _sum[5] + _sum[6] + _sum[7];
    }
#endif // __AVX__
    {
        __m128 _sum = _mm_setzero_ps();
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _sum = _mm_add_ps(_sum, _cur);
        }
        sum += _sum[0] + _sum[1] + _sum[2] + _sum[3];
    }
#endif // __SSE2__
    for (; i < size; ++i, ++ptr)
    {
        sum += *ptr;
    }
    return sum;
}

static NCNN_FORCEINLINE float fast_var(float* ptr, int size, float mean)
{
    float sq_sum = 0.0f;
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _mean = _mm512_set1_ps(mean);
        __m512 _sq_sum = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_sub_ps(_cur, _mean);
            _cur = _mm512_mul_ps(_cur, _cur);
            _sq_sum = _mm512_add_ps(_sq_sum, _cur);
        }
        sq_sum += _mm512_reduce_add_ps(_sq_sum);
    }
#endif // __AVX512F__
    {
        __m256 _mean = _mm256_set1_ps(mean);
        __m256 _sq_sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _cur = _mm256_sub_ps(_cur, _mean);
            _cur = _mm256_mul_ps(_cur, _cur);
            _sq_sum = _mm256_add_ps(_sq_sum, _cur);
        }
        sq_sum += _sq_sum[0] + _sq_sum[1] + _sq_sum[2] + _sq_sum[3] + _sq_sum[4] + _sq_sum[5] + _sq_sum[6] + _sq_sum[7];
    }
#endif // __AVX__
    {
        __m128 _mean = _mm_set1_ps(mean);
        __m128 _sq_sum = _mm_setzero_ps();
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_sub_ps(_cur, _mean);
            _cur = _mm_mul_ps(_cur, _cur);
            _sq_sum = _mm_add_ps(_sq_sum, _cur);
        }
        sq_sum += _sq_sum[0] + _sq_sum[1] + _sq_sum[2] + _sq_sum[3];
    }
#endif // __SSE2__
    for (; i < size; ++i, ++ptr)
    {
        float tmp = *ptr - mean;
        sq_sum += tmp * tmp;
    }
    return sq_sum / size;
}

static NCNN_FORCEINLINE void fast_fmadd(float* ptr, float a, float b, int size)
{
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        // 512 bit FMA instructions are included in AVX512F.
        __m512 _a = _mm512_set1_ps(a);
        __m512 _b = _mm512_set1_ps(b);
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _mm512_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX512F__
    {
        // 256 bit FMA instructions are not included in AVX1
        __m256 _a = _mm256_set1_ps(a);
        __m256 _b = _mm256_set1_ps(b);
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
#if __FMA__
            _cur = _mm256_fmadd_ps(_cur, _a, _b);
#else
            _cur = _mm256_mul_ps(_cur, _a);
            _cur = _mm256_add_ps(_cur, _b);
#endif
            _mm256_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX__
    {
        __m128 _a = _mm_set1_ps(a);
        __m128 _b = _mm_set1_ps(b);
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_mul_ps(_cur, _a);
            _cur = _mm_add_ps(_cur, _b);
            _mm_storeu_ps(ptr, _cur);
        }
    }
#endif // __SSE2__
    for (; i < size; ++i, ++ptr)
    {
        *ptr = (*ptr) * a + b;
    }
}

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_fmadd_fmadd(float* ptr, float a, float b, int size) const
{
    int i = 0;
    auto gamma = static_cast<const float*>(gamma_data);
    auto beta = static_cast<const float*>(beta_data);
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _a = _mm512_set1_ps(a);
        __m512 _b = _mm512_set1_ps(b);
        for (; i + 16 <= size; i += 16, ptr += 16, gamma += 16, beta += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            __m512 _gamma = _mm512_loadu_ps(gamma);
            __m512 _beta = _mm512_loadu_ps(beta);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _cur = _mm512_fmadd_ps(_cur, _gamma, _beta);
            _mm512_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX512F__
    {
        __m256 _a = _mm256_set1_ps(a);
        __m256 _b = _mm256_set1_ps(b);

        for (; i + 8 <= size; i += 8, ptr += 8, gamma += 8, beta += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            __m256 _gamma = _mm256_loadu_ps(gamma);
            __m256 _beta = _mm256_loadu_ps(beta);
#if __FMA__
            _cur = _mm256_fmadd_ps(_cur, _a, _b);
            _cur = _mm256_fmadd_ps(_cur, _gamma, _beta);
#else
            _cur = _mm256_mul_ps(_cur, _a);
            _cur = _mm256_add_ps(_cur, _b);
            _cur = _mm256_mul_ps(_cur, _gamma);
            _cur = _mm256_add_ps(_cur, _beta);
#endif
            _mm256_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX__
    {
        __m128 _a = _mm_set1_ps(a);
        __m128 _b = _mm_set1_ps(b);
        for (; i + 4 <= size; i += 4, ptr += 4, gamma += 4, beta += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            __m128 _gamma = _mm_loadu_ps(gamma);
            __m128 _beta = _mm_loadu_ps(beta);
            _cur = _mm_mul_ps(_cur, _a);
            _cur = _mm_add_ps(_cur, _b);
            _cur = _mm_mul_ps(_cur, _gamma);
            _cur = _mm_add_ps(_cur, _beta);
            _mm_storeu_ps(ptr, _cur);
        }
    }
#endif // __SSE2__
    for (; i < size; ++i, ++ptr, ++gamma, ++beta)
    {
        *ptr = ((*ptr) * a + b) * (*gamma) + (*beta);
    }
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_1d_layer_norm(float* ptr, int size) const
{
    // mean and var
    float sum = fast_sum(ptr, size);
    float mean = sum / size;
    float var = fast_var(ptr, size, mean);

    float a = static_cast<float>(1.0f / sqrt(var + eps));
    float b = -mean * a;

    if (affine)
    {
        fast_fmadd_fmadd(ptr, a, b, size);
    }
    else
    {
        fast_fmadd(ptr, a, b, size);
    }
}

int NCNN_FORCEINLINE LayerNorm_x86::forward_inplace_unpacked(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int size = bottom_top_blob.w;
        float* ptr = bottom_top_blob;
        fast_1d_layer_norm(ptr, size);
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int size = w;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);

            fast_1d_layer_norm(ptr, size);
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (affine_size == w)
        {
            size = w;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    fast_1d_layer_norm(ptr, size);
                }
            }
        }
        else // if (affine_size == size)
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                fast_1d_layer_norm(ptr, size);
            }
        }
    }

    return 0;
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elempack == 1)
    {
        return forward_inplace_unpacked(bottom_top_blob, opt);
    }
    else
    {
        fprintf(stderr, "Packed forward not implemented!\n");
        return -1;
    }
}

} // namespace ncnn
