#include "layernorm_x86.h"

#include <math.h>
#include <cpu.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

static NCNN_FORCEINLINE float fast_mean(float* ptr, int elemcount)
{
    float sum = 0.0f;
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _sum = _mm512_setzero_ps();
        for (; i + 16 <= elemcount; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _sum = _mm512_add_ps(_sum, _cur);
        }
        sum += _mm512_reduce_add_ps(_sum);
    }
#endif // __AVX512F__
    {
        __m256 _sum = _mm256_setzero_ps();
        for (; i + 8 <= elemcount; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _sum = _mm256_add_ps(_sum, _cur);
        }
        sum += _sum[0] + _sum[1] + _sum[2] + _sum[3] + _sum[4] + _sum[5] + _sum[6] + _sum[7];
    }
#endif // __AVX__
    {
        __m128 _sum = _mm_setzero_ps();
        for (; i + 4 <= elemcount; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _sum = _mm_add_ps(_sum, _cur);
        }
        sum += _sum[0] + _sum[1] + _sum[2] + _sum[3];
    }
#endif // __SSE2__
    for (; i < elemcount; ++i, ++ptr)
    {
        sum += *ptr;
    }
    return sum / elemcount;
}

static NCNN_FORCEINLINE float fast_var(float* ptr, int elemcount, float mean)
{
    float sq_sum = 0.0f;
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        __m512 _mean = _mm512_set1_ps(mean);
        __m512 _sq_sum = _mm512_setzero_ps();
        for (; i + 16 <= elemcount; i += 16, ptr += 16)
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
        for (; i + 8 <= elemcount; i += 8, ptr += 8)
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
        for (; i + 4 <= elemcount; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_sub_ps(_cur, _mean);
            _cur = _mm_mul_ps(_cur, _cur);
            _sq_sum = _mm_add_ps(_sq_sum, _cur);
        }
        sq_sum += _sq_sum[0] + _sq_sum[1] + _sq_sum[2] + _sq_sum[3];
    }
#endif // __SSE2__
    for (; i < elemcount; ++i, ++ptr)
    {
        float tmp = *ptr - mean;
        sq_sum += tmp * tmp;
    }
    return sq_sum / elemcount;
}

static NCNN_FORCEINLINE void fast_fmadd(float* ptr, float a, float b, int elemcount)
{
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    {
        // 512 bit FMA instructions are included in AVX512F.
        __m512 _a = _mm512_set1_ps(a);
        __m512 _b = _mm512_set1_ps(b);
        for (; i + 16 <= elemcount; i += 16, ptr += 16)
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
        for (; i + 8 <= elemcount; i += 8, ptr += 8)
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
        for (; i + 4 <= elemcount; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_mul_ps(_cur, _a);
            _cur = _mm_add_ps(_cur, _b);
            _mm_storeu_ps(ptr, _cur);
        }
    }
#endif // __SSE2__
    for (; i < elemcount; ++i, ++ptr)
    {
        *ptr = (*ptr) * a + b;
    }
}

static NCNN_FORCEINLINE void fast_mean_packed(float* ptr, float* mean, int elempack, int elemcount, int size)
{
    int i = 0;
    if (elempack == 4)
    {
        __m128 _sum = _mm_setzero_ps();
        __m128 _elemcount = _mm_set1_ps(float(elemcount));
        for (; i < size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _sum = _mm_add_ps(_sum, _cur);
        }
        __m128 _mean = _mm_div_ps(_sum, _elemcount);
        _mm_storeu_ps(mean, _mean);
    }
    else if (elempack == 8)
    {
#if __AVX__
        __m256 _sum = _mm256_setzero_ps();
        __m256 _elemcount = _mm256_set1_ps(float(elemcount));
        for (; i < size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _sum = _mm256_add_ps(_sum, _cur);
        }
        __m256 _mean = _mm256_div_ps(_sum, _elemcount);
        _mm256_storeu_ps(mean, _mean);
#endif
    }
    else if (elempack == 16)
    {
#if __AVX512F__
        __m512 _sum = _mm512_setzero_ps();
        __m512 _elemcount = _mm512_set1_ps(float(elemcount));
        for (; i < size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _sum = _mm512_add_ps(_sum, _cur);
        }
        __m512 _mean = _mm512_div_ps(_sum, _elemcount);
        _mm512_storeu_ps(mean, _mean);
#endif
    }
}

static NCNN_FORCEINLINE void fast_var_packed(float* ptr, float* var, float* mean, int elempack, int elemcount, int size)
{
    int i = 0;
    if (elempack == 4)
    {
        __m128 _mean = _mm_loadu_ps(mean);
        __m128 _sq_sum = _mm_setzero_ps();
        __m128 _elemcount = _mm_set1_ps(float(elemcount));
        for (; i < size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_sub_ps(_cur, _mean);
            _cur = _mm_mul_ps(_cur, _cur);
            _sq_sum = _mm_add_ps(_sq_sum, _cur);
        }
        __m128 _var = _mm_div_ps(_sq_sum, _elemcount);
        _mm_storeu_ps(var, _var);
    }
    else if (elempack == 8)
    {
#if __AVX__
        __m256 _mean = _mm256_loadu_ps(mean);
        __m256 _sq_sum = _mm256_setzero_ps();
        __m256 _elemcount = _mm256_set1_ps(float(elemcount));
        for (; i < size; i += 8, ptr += 8)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            _cur = _mm256_sub_ps(_cur, _mean);
            _cur = _mm256_mul_ps(_cur, _cur);
            _sq_sum = _mm256_add_ps(_sq_sum, _cur);
        }
        __m256 _var = _mm256_div_ps(_sq_sum, _elemcount);
        _mm256_storeu_ps(var, _var);
#endif
    }
    else if (elempack == 16)
    {
#if __AVX512F__
        __m512 _mean = _mm512_loadu_ps(mean);
        __m512 _sq_sum = _mm512_setzero_ps();
        __m512 _elemcount = _mm512_set1_ps(float(elemcount));
        for (; i < size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_sub_ps(_cur, _mean);
            _cur = _mm512_mul_ps(_cur, _cur);
            _sq_sum = _mm512_add_ps(_sq_sum, _cur);
        }
        __m512 _var = _mm512_div_ps(_sq_sum, _elemcount);
        _mm512_storeu_ps(var, _var);
#endif
    }
}

static NCNN_FORCEINLINE void fast_fmadd_packed(float* ptr, float* a, float* b, int elempack, int size)
{
    int i = 0;
    if (elempack == 4)
    {
        __m128 _a = _mm_loadu_ps(a);
        __m128 _b = _mm_loadu_ps(b);
        for (; i < size; i += 4, ptr += 4)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            _cur = _mm_mul_ps(_cur, _a);
            _cur = _mm_add_ps(_cur, _b);
            _mm_storeu_ps(ptr, _cur);
        }
    }
    else if (elempack == 8)
    {
#if __AVX__
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);
        for (; i < size; i += 8, ptr += 8)
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
#endif
    }
    else if (elempack == 16)
    {
#if __AVX512F__
        __m512 _a = _mm512_loadu_ps(a);
        __m512 _b = _mm512_loadu_ps(b);
        for (; i < size; i += 16, ptr += 16)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _mm512_storeu_ps(ptr, _cur);
        }
#endif
    }
}

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_fmadd_fmadd(float* ptr, float a, float b, int elemcount) const
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
    {
        __m256 _a = _mm256_set1_ps(a);
        __m256 _b = _mm256_set1_ps(b);

        for (; i + 8 <= elemcount; i += 8, ptr += 8, gamma += 8, beta += 8)
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
        for (; i + 4 <= elemcount; i += 4, ptr += 4, gamma += 4, beta += 4)
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
    for (; i < elemcount; ++i, ++ptr, ++gamma, ++beta)
    {
        *ptr = ((*ptr) * a + b) * (*gamma) + (*beta);
    }
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_fmadd_fmadd_packed(float* ptr, float* a, float* b, int elempack, int size) const
{
    int i = 0;
    auto gamma = static_cast<const float*>(gamma_data);
    auto beta = static_cast<const float*>(beta_data);
    if (elempack == 4)
    {
        __m128 _a = _mm_loadu_ps(a);
        __m128 _b = _mm_loadu_ps(b);
        for (; i < size; i += 4, ptr += 4, ++gamma, ++beta)
        {
            __m128 _cur = _mm_loadu_ps(ptr);
            __m128 _gamma = _mm_set1_ps(*gamma);
            __m128 _beta = _mm_set1_ps(*beta);
            _cur = _mm_mul_ps(_cur, _a);
            _cur = _mm_add_ps(_cur, _b);
            _cur = _mm_mul_ps(_cur, _gamma);
            _cur = _mm_add_ps(_cur, _beta);
            _mm_storeu_ps(ptr, _cur);
        }
    }
    else if (elempack == 8)
    {
#if __AVX__
        __m256 _a = _mm256_loadu_ps(a);
        __m256 _b = _mm256_loadu_ps(b);
        for (; i < size; i += 8, ptr += 8, ++gamma, ++beta)
        {
            __m256 _cur = _mm256_loadu_ps(ptr);
            __m256 _gamma = _mm256_set1_ps(*gamma);
            __m256 _beta = _mm256_set1_ps(*beta);
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
#endif
    }
    else if (elempack == 16)
    {
#if __AVX512F__
        __m512 _a = _mm512_loadu_ps(a);
        __m512 _b = _mm512_loadu_ps(b);
        for (; i < size; i += 16, ptr += 16, ++gamma, ++beta)
        {
            __m512 _cur = _mm512_loadu_ps(ptr);
            __m512 _gamma = _mm512_set1_ps(*gamma);
            __m512 _beta = _mm512_set1_ps(*beta);
            _cur = _mm512_fmadd_ps(_cur, _a, _b);
            _cur = _mm512_fmadd_ps(_cur, _gamma, _beta);
            _mm512_storeu_ps(ptr, _cur);
        }
#endif
    }
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_1d_layer_norm(float* ptr, int elemcount) const
{
    // mean and var
    float mean = fast_mean(ptr, elemcount);
    float var = fast_var(ptr, elemcount, mean);

    float a = static_cast<float>(1.0f / sqrt(var + eps));
    float b = -mean * a;

    if (affine)
    {
        fast_fmadd_fmadd(ptr, a, b, elemcount);
    }
    else
    {
        fast_fmadd(ptr, a, b, elemcount);
    }
}

void NCNN_FORCEINLINE LayerNorm_x86::fast_1d_layer_norm_packed(float* ptr, int elempack, int elemcount, int size) const
{
    float mean[16], var[16];
    fast_mean_packed(ptr, mean, elempack, elemcount, size);
    fast_var_packed(ptr, var, mean, elempack, elemcount, size);
    float *a = var, *b = mean;

    if (elempack == 4)
    {
        __m128 _a = _mm_set1_ps(1.0f);
        __m128 _eps = _mm_set1_ps(eps);
        __m128 _b = _mm_setzero_ps();
        __m128 _var = _mm_loadu_ps(var);
        _var = _mm_add_ps(_var, _eps);
        __m128 _sqrt_var = _mm_sqrt_ps(_var);
        _a = _mm_div_ps(_a, _sqrt_var);
        __m128 _mean_a = _mm_loadu_ps(mean);
        _mean_a = _mm_mul_ps(_mean_a, _a);
        _b = _mm_sub_ps(_b, _mean_a);

        _mm_storeu_ps(a, _a);
        _mm_storeu_ps(b, _b);
    }
    else if (elempack == 8)
    {
#if __AVX__
        __m256 _a = _mm256_set1_ps(1.0f);
        __m256 _eps = _mm256_set1_ps(eps);
        __m256 _b = _mm256_setzero_ps();
        __m256 _var = _mm256_loadu_ps(var);
        _var = _mm256_add_ps(_var, _eps);
        __m256 _sqrt_var = _mm256_sqrt_ps(_var);
        _a = _mm256_div_ps(_a, _sqrt_var);
#if __FMA__
        __m256 _mean = _mm256_loadu_ps(mean);
        _b = _mm256_fnmadd_ps(_mean, _a, _b);
#else
        __m256 _mean_a = _mm256_loadu_ps(mean);
        _mean_a = _mm256_mul_ps(_mean_a, _a);
        _b = _mm256_sub_ps(_b, _mean_a);
#endif
        _mm256_storeu_ps(a, _a);
        _mm256_storeu_ps(b, _b);
#endif
    }
    else if (elempack == 16)
    {
#if __AVX512F__
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
#endif
    }

    if (affine)
    {
        fast_fmadd_fmadd_packed(ptr, a, b, elempack, size);
    }
    else
    {
        fast_fmadd_packed(ptr, a, b, elempack, size);
    }
}

int NCNN_FORCEINLINE LayerNorm_x86::forward_inplace_unpacked(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int elemcount = bottom_top_blob.w * bottom_top_blob.elempack;
        float* ptr = bottom_top_blob;
        fast_1d_layer_norm(ptr, elemcount);
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int elemcount = w * bottom_top_blob.elempack;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);

            fast_1d_layer_norm(ptr, elemcount);
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int elemcount = w * h * bottom_top_blob.elempack;

        if (affine_size == w)
        {
            elemcount = w * bottom_top_blob.elempack;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    fast_1d_layer_norm(ptr, elemcount);
                }
            }
        }
        else // if (affine_elemcount == elemcount)
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                fast_1d_layer_norm(ptr, elemcount);
            }
        }
    }

    return 0;
}

int NCNN_FORCEINLINE LayerNorm_x86::forward_inplace_packed(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;

    // Now, bottoms_top_blob.dims >= 2
    if (bottom_top_blob.dims == 2)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);
            fast_1d_layer_norm_packed(ptr, elempack, w, w * elempack);
        }
    }
    else if (bottom_top_blob.dims == 3)
    {
        int channels = bottom_top_blob.c;
        if (affine_size == w)
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; ++q)
            {
                for (int i = 0; i < h; ++i)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    fast_1d_layer_norm_packed(ptr, elempack, w, w * elempack);
                }
            }
        }
        else // if(affine_size == w * h)
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; ++q)
            {
                float* ptr = bottom_top_blob.channel(q);
                fast_1d_layer_norm_packed(ptr, elempack, w * h, w * h * elempack);
            }
        }
    }

    return 0;
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elempack == 1 || bottom_top_blob.dims == 1)
    {
        return forward_inplace_unpacked(bottom_top_blob, opt);
    }
    else
    {
        return forward_inplace_packed(bottom_top_blob, opt);
    }
}

} // namespace ncnn
