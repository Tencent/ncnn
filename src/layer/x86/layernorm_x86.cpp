#include "layernorm_x86.h"
#include "x86_usability.h"
#include <math.h>
#include <cpu.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

static NCNN_FORCEINLINE void fast_mean(float* ptr, float* mean, int elempack, int elemcount, int size)
{
    int i = 0;
    float sum = 0.0f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16 || elempack == 1)
    {
        __m512 _sum = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _sum = _mm512_add_ps(_sum, _cur);
        }
        if (elempack == 16)
        {
            __m512 _elemcount = _mm512_set1_ps(float(elemcount));
            __m512 _mean = _mm512_div_ps(_sum, _elemcount);
            _mm512_storeu_ps(mean, _mean);
        }
        else
        {
            sum += _mm512_reduce_add_ps(_sum);
        }
    }
#endif // __AVX512F__
    if (elempack == 8 || elempack == 1)
    {
        __m256 _sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _sum = _mm256_add_ps(_sum, _cur);
        }
        if (elempack == 8)
        {
            __m256 _elemcount = _mm256_set1_ps(float(elemcount));
            __m256 _mean = _mm256_div_ps(_sum, _elemcount);
            _mm256_storeu_ps(mean, _mean);
        }
        else
        {
            sum += _mm256_reduce_add_ps(_sum);
        }
    }
#endif // __AVX__
    if (elempack == 4 || elempack == 1)
    {
        __m128 _sum = _mm_setzero_ps();
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _sum = _mm_add_ps(_sum, _cur);
        }
        if (elempack == 4)
        {
            __m128 _elemcount = _mm_set1_ps(float(elemcount));
            __m128 _mean = _mm_div_ps(_sum, _elemcount);
            _mm_storeu_ps(mean, _mean);
        }
        else
        {
            sum += _mm_reduce_add_ps(_sum);
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        for (; i < size; ++i, ++ptr)
        {
            sum += *ptr;
        }
        *mean = sum / elemcount;
    }
}

static NCNN_FORCEINLINE void fast_var(float* ptr, float* var, float* mean, int elempack, int elemcount, int size)
{
    int i = 0;
    float sq_sum = 0.0f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16 || elempack == 1)
    {
        __m512 _mean = elempack == 1 ? _mm512_set1_ps(*mean) : _mm512_loadu_ps(mean);
        __m512 _sq_sum = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_sub_ps(_cur, _mean);
            _sq_sum = _mm512_fmadd_ps(_cur, _cur, _sq_sum);
        }
        if (elempack == 16)
        {
            __m512 _elemcount = _mm512_set1_ps(float(elemcount));
            __m512 _var = _mm512_div_ps(_sq_sum, _elemcount);
            _mm512_storeu_ps(var, _var);
        }
        else
        {
            sq_sum += _mm512_reduce_add_ps(_sq_sum);
        }
    }
#endif // __AVX512F__
    if (elempack == 8 || elempack == 1)
    {
        __m256 _mean = elempack == 1 ? _mm256_set1_ps(*mean) : _mm256_loadu_ps(mean);
        __m256 _sq_sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _cur = _mm256_sub_ps(_cur, _mean);
            _sq_sum = _mm256_comp_fmadd_ps(_cur, _cur, _sq_sum);
        }
        if (elempack == 8)
        {
            __m256 _elemcount = _mm256_set1_ps(float(elemcount));
            __m256 _var = _mm256_div_ps(_sq_sum, _elemcount);
            _mm256_storeu_ps(var, _var);
        }
        else
        {
            sq_sum += _mm256_reduce_add_ps(_sq_sum);
        }
    }
#endif // __AVX__
    if (elempack == 4 || elempack == 1)
    {
        __m128 _mean = elempack == 1 ? _mm_set1_ps(*mean) : _mm_loadu_ps(mean);
        __m128 _sq_sum = _mm_setzero_ps();
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_sub_ps(_cur, _mean);
            _sq_sum = _mm_comp_fmadd_ps(_cur, _cur, _sq_sum);
        }
        if (elempack == 4)
        {
            __m128 _elemcount = _mm_set1_ps(float(elemcount));
            __m128 _var = _mm_div_ps(_sq_sum, _elemcount);
            _mm_storeu_ps(var, _var);
        }
        else
        {
            sq_sum += _mm_reduce_add_ps(_sq_sum);
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        float _mean = *mean;
        for (; i < size; ++i, ++ptr)
        {
            float tmp = *ptr - _mean;
            sq_sum += tmp * tmp;
        }
        *var = sq_sum / elemcount;
    }
}

static NCNN_FORCEINLINE void fast_fmadd(float* ptr, float* a, float* b, int elempack, int elemcount, int size)
{
    int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16 || elempack == 1)
    {
        __m512 _a = elempack == 1 ? _mm512_set1_ps(*a) : _mm512_loadu_ps(a);
        __m512 _b = elempack == 1 ? _mm512_set1_ps(*b) : _mm512_loadu_ps(b);
        for (; i + 16 <= size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _mm512_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX512F__
    if (elempack == 8 || elempack == 1)
    {
        __m256 _a = elempack == 1 ? _mm256_set1_ps(*a) : _mm256_loadu_ps(a);
        __m256 _b = elempack == 1 ? _mm256_set1_ps(*b) : _mm256_loadu_ps(b);
        for (; i + 8 <= size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _cur = _mm256_comp_fmadd_ps(_cur, _a, _b);
            _mm256_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX__
    if (elempack == 4 || elempack == 1)
    {
        __m128 _a = elempack == 1 ? _mm_set1_ps(*a) : _mm_loadu_ps(a);
        __m128 _b = elempack == 1 ? _mm_set1_ps(*b) : _mm_loadu_ps(b);
        for (; i + 4 <= size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_comp_fmadd_ps(_cur, _a, _b);
            _mm_storeu_ps(ptr, _cur);
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        for (; i < elemcount; ++i, ++ptr)
        {
            *ptr = (*ptr) * (*a) + (*b);
        }
    }
}

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_fmadd_fmadd(float* ptr, float* a, float* b, int elempack, int elemcount, int size) const
{
    int i = 0;
    const float* gamma = static_cast<const float*>(gamma_data);
    const float* beta = static_cast<const float*>(beta_data);

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _a = _mm512_loadu_ps(a);
        __m512 _b = _mm512_loadu_ps(b);
        for (; i + 16 <= size; i += 16, ptr += 16, ++gamma, ++beta)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            __m512 _gamma = _mm512_set1_ps(*gamma);
            __m512 _beta = _mm512_set1_ps(*beta);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _cur = _mm512_fmadd_ps(_cur, _gamma, _beta);
            _mm512_storeu_ps(ptr, _cur);
        }
    }
    else if (elempack == 1)
    {
        __m512 _a = _mm512_set1_ps(*a);
        __m512 _b = _mm512_set1_ps(*b);
        for (; i + 16 <= elemcount; i += 16, ptr += 16, gamma += 16, beta += 16)
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
    if (elempack == 8)
    {
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);
        for (; i + 8 <= size; i += 8, ptr += 8, ++gamma, ++beta)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            __m256 _gamma = _mm256_set1_ps(*gamma);
            __m256 _beta = _mm256_set1_ps(*beta);
            _cur = _mm256_comp_fmadd_ps(_cur, _a, _b);
            _cur = _mm256_comp_fmadd_ps(_cur, _gamma, _beta);
            _mm256_storeu_ps(ptr, _cur);
        }
    }
    else if (elempack == 1)
    {
        __m256 _a = _mm256_set1_ps(*a);
        __m256 _b = _mm256_set1_ps(*b);
        for (; i + 8 <= elemcount; i += 8, ptr += 8, gamma += 8, beta += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            __m256 _gamma = _mm256_loadu_ps(gamma);
            __m256 _beta = _mm256_loadu_ps(beta);
            _cur = _mm256_comp_fmadd_ps(_cur, _a, _b);
            _cur = _mm256_comp_fmadd_ps(_cur, _gamma, _beta);
            _mm256_storeu_ps(ptr, _cur);
        }
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _a = _mm_loadu_ps(a);
        __m128 _b = _mm_loadu_ps(b);
        for (; i + 4 <= size; i += 4, ptr += 4, ++gamma, ++beta)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            __m128 _gamma = _mm_set1_ps(*gamma);
            __m128 _beta = _mm_set1_ps(*beta);
            _cur = _mm_comp_fmadd_ps(_cur, _a, _b);
            _cur = _mm_comp_fmadd_ps(_cur, _gamma, _beta);
            _mm_storeu_ps(ptr, _cur);
        }
    }
    else if (elempack == 1)
    {
        __m128 _a = _mm_set1_ps(*a);
        __m128 _b = _mm_set1_ps(*b);
        for (; i + 4 <= elemcount; i += 4, ptr += 4, gamma += 4, beta += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            __m128 _gamma = _mm_loadu_ps(gamma);
            __m128 _beta = _mm_loadu_ps(beta);
            _cur = _mm_comp_fmadd_ps(_cur, _a, _b);
            _cur = _mm_comp_fmadd_ps(_cur, _gamma, _beta);
            _mm_storeu_ps(ptr, _cur);
        }
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        for (; i < elemcount; ++i, ++ptr, ++gamma, ++beta)
        {
            *ptr = ((*ptr) * (*a) + (*b)) * (*gamma) + (*beta);
        }
    }
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_1d_layer_norm(float* ptr, int elempack, int elemcount, int size) const
{
    float mean[16], var[16];
    fast_mean(ptr, mean, elempack, elemcount, size);
    fast_var(ptr, var, mean, elempack, elemcount, size);
    float *a = var, *b = mean;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _a = _mm512_set1_ps(1.0f);
        __m512 _eps = _mm512_set1_ps(eps);
        __m512 _b = _mm512_setzero_ps();
        __m512 _var = _mm512_loadu_ps(var);
        _var = _mm512_add_ps(_var, _eps);
        __m512 _sqrt_var = _mm512_sqrt_ps(_var);
        _a = _mm512_div_ps(_a, _sqrt_var);
        __m512 _mean = _mm512_loadu_ps(mean);
        _b = _mm512_fnmadd_ps(_mean, _a, _b);

        _mm512_storeu_ps(a, _a);
        _mm512_storeu_ps(b, _b);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        __m256 _a = _mm256_set1_ps(1.0f);
        __m256 _eps = _mm256_set1_ps(eps);
        __m256 _b = _mm256_setzero_ps();
        __m256 _var = _mm256_loadu_ps(var);
        _var = _mm256_add_ps(_var, _eps);
        __m256 _sqrt_var = _mm256_sqrt_ps(_var);
        _a = _mm256_div_ps(_a, _sqrt_var);
        __m256 _mean = _mm256_loadu_ps(mean);
        _b = _mm256_comp_fnmadd_ps(_mean, _a, _b);
        _mm256_storeu_ps(a, _a);
        _mm256_storeu_ps(b, _b);
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _a = _mm_set1_ps(1.0f);
        __m128 _eps = _mm_set1_ps(eps);
        __m128 _b = _mm_setzero_ps();
        __m128 _var = _mm_loadu_ps(var);
        _var = _mm_add_ps(_var, _eps);
        __m128 _sqrt_var = _mm_sqrt_ps(_var);
        _a = _mm_div_ps(_a, _sqrt_var);
        __m128 _mean = _mm_loadu_ps(mean);
        _b = _mm_comp_fnmadd_ps(_mean, _a, _b);

        _mm_storeu_ps(a, _a);
        _mm_storeu_ps(b, _b);
    }
#endif // __SSE2__
    if (elempack == 1)
    {
        *a = static_cast<float>(1.0f / sqrt(*var + eps));
        *b = -*mean * (*a);
    }

    if (affine)
    {
        fast_fmadd_fmadd(ptr, a, b, elempack, elemcount, size);
    }
    else
    {
        fast_fmadd(ptr, a, b, elempack, elemcount, size);
    }
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        int elemcount = w * elempack;
        float* ptr = bottom_top_blob;
        // 1D layer norm is special. Treat them as unpacked.
        fast_1d_layer_norm(ptr, 1, elemcount, elemcount);
    }
    else if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);
            fast_1d_layer_norm(ptr, elempack, w, w * elempack);
        }
    }
    else if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; ++q)
            {
                for (int i = 0; i < h; ++i)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    fast_1d_layer_norm(ptr, elempack, w, w * elempack);
                }
            }
        }
        else // if(affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; ++q)
            {
                float* ptr = bottom_top_blob.channel(q);
                fast_1d_layer_norm(ptr, elempack, w * h, w * h * elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
