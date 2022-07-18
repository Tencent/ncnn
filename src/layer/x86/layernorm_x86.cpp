#include "layernorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            float* ptr = (float*)bottom_top_blob;

            __m256 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m256 _fsum = _mm256_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                _fsum = _mm256_add_ps(_fsum, _fLoad);
            }

            const float* q = (const float*)&_fsum;

            sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

            // var
            float mean = sum / (w << 3);
            __m256 _mean = _mm256_set1_ps(mean);
            __m256 _fsqsum = _mm256_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                _fLoad = _mm256_sub_ps(_fLoad, _mean);
                _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
            }

            q = (const float*)&_fsqsum;
            sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

            float var = sqsum / (w << 3);

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m256 _a = _mm256_set1_ps(a);
            __m256 _b = _mm256_set1_ps(b);
            __m256 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fLoad = _mm256_mul_ps(_fLoad, _a);
                    _fLoad = _mm256_add_ps(_fLoad, _b);

                    _gamma = _mm256_loadu_ps((const float*)gamma_data + (i << 3));
                    _beta = _mm256_loadu_ps((const float*)beta_data + (i << 3));
                    _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm256_add_ps(_fLoad, _beta);

                    _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fLoad = _mm256_mul_ps(_fLoad, _a);
                    _fLoad = _mm256_add_ps(_fLoad, _b);
                    _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m256 _fLoad;

                // mean

                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                // var
                __m256 _size = _mm256_set1_ps((float)w);
                __m256 _mean = _mm256_div_ps(_fsum, _size);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                __m256 _var = _mm256_div_ps(_fsqsum, _size);

                __m256 _eps = _mm256_set1_ps(eps);
                __m256 _a = _mm256_add_ps(_var, _eps);
                _a = _mm256_rsqrt_ps(_a);
                __m256 _b = _mm256_mul_ps(-_mean, _a);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);

                        _gamma = _mm256_set1_ps(((const float*)gamma_data)[i]);
                        _beta = _mm256_set1_ps(((const float*)beta_data)[i]);
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }
                }
                else
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }
                }
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
#pragma omp parallel for num_threads(opt.num_threads)
                for (int qq = 0; qq < channels; qq++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(qq).row(i);

                        __m256 _fLoad;

                        // mean
                        __m256 _fsum = _mm256_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                            _fsum = _mm256_add_ps(_fsum, _fLoad);
                        }

                        // var
                        __m256 _size = _mm256_set1_ps((float)w);
                        __m256 _mean = _mm256_div_ps(_fsum, _size);
                        __m256 _fsqsum = _mm256_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                            _fLoad = _mm256_sub_ps(_fLoad, _mean);
                            _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                            _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                        }

                        __m256 _var = _mm256_div_ps(_fsqsum, _size);

                        __m256 _eps = _mm256_set1_ps(eps);
                        __m256 _a = _mm256_add_ps(_var, _eps);
                        _a = _mm256_rsqrt_ps(_a);
                        __m256 _b = _mm256_mul_ps(_mean, _a);
                        __m256 _gamma, _beta;

                        if (affine)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm256_loadu_ps(ptr + (j << 3));
                                _fLoad = _mm256_mul_ps(_fLoad, _a);
                                _fLoad = _mm256_sub_ps(_fLoad, _b);

                                _gamma = _mm256_set1_ps(((const float*)gamma_data)[j]);
                                _beta = _mm256_set1_ps(((const float*)beta_data)[j]);
                                _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                                _fLoad = _mm256_add_ps(_fLoad, _beta);

                                _mm256_storeu_ps(ptr + (j << 3), _fLoad);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm256_loadu_ps(ptr + (j << 3));
                                _fLoad = _mm256_mul_ps(_fLoad, _a);
                                _fLoad = _mm256_sub_ps(_fLoad, _b);
                                _mm256_storeu_ps(ptr + (j << 3), _fLoad);
                            }
                        }
                    }
                }
            }

            else // if (affine_size == size)
            {
#pragma omp parallel for num_threads(opt.num_threads)
                for (int qq = 0; qq < channels; qq++)
                {
                    float* ptr = bottom_top_blob.channel(qq);
                    // int ssize = size;

                    __m256 _fLoad;

                    // mean
                    __m256 _fsum = _mm256_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fsum = _mm256_add_ps(_fsum, _fLoad);
                    }

                    // const float* sum = (const float*)&_fsum;

                    // var
                    __m256 _size = _mm256_set1_ps((float)size);
                    __m256 _mean = _mm256_div_ps(_fsum, _size);
                    __m256 _fsqsum = _mm256_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_sub_ps(_fLoad, _mean);
                        _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                        _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                    }

                    // const float* sqsum = (const float*)&_fsqsum;

                    __m256 _var = _mm256_div_ps(_fsqsum, _size);

                    // float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    // float b = -mean * a;
                    __m256 _eps = _mm256_set1_ps(eps);
                    __m256 _a = _mm256_add_ps(_var, _eps);
                    _a = _mm256_rsqrt_ps(_a);
                    __m256 _b = _mm256_mul_ps(-_mean, _a);
                    __m256 _gamma, _beta;

                    if (affine)
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                            _fLoad = _mm256_mul_ps(_fLoad, _a);
                            _fLoad = _mm256_add_ps(_fLoad, _b);

                            // _gamma = _mm256_loadu_ps((const float*)gamma_data + (i << 3));
                            // _beta = _mm256_loadu_ps((const float*)beta_data + (i << 3));
                            _gamma = _mm256_set1_ps(((const float*)gamma_data)[i]);
                            _beta = _mm256_set1_ps(((const float*)beta_data)[i]);
                            _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                            _fLoad = _mm256_add_ps(_fLoad, _beta);

                            _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                            _fLoad = _mm256_mul_ps(_fLoad, _a);
                            _fLoad = _mm256_add_ps(_fLoad, _b);
                            _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif // __AVX__
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            __m128 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m128 _fsum = _mm_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i << 2));
                _fsum = _mm_add_ps(_fsum, _fLoad);
            }

            const float* q = (const float*)&_fsum;

            sum = q[0] + q[1] + q[2] + q[3];

            // var
            float mean = sum / (w << 2);
            __m128 _mean = _mm_set1_ps(mean);
            __m128 _fsqsum = _mm_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i << 2));
                _fLoad = _mm_sub_ps(_fLoad, _mean);
                _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
            }

            q = (const float*)&_fsqsum;
            sqsum = q[0] + q[1] + q[2] + q[3];

            float var = sqsum / (w << 2);

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            __m128 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i << 2));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);

                    _gamma = _mm_load_ps((const float*)gamma_data + (i << 2));
                    _beta = _mm_load_ps((const float*)beta_data + (i << 2));
                    _fLoad = _mm_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm_add_ps(_fLoad, _beta);

                    _mm_store_ps(ptr + (i << 2), _fLoad);
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i << 2));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);
                    _mm_store_ps(ptr + (i << 2), _fLoad);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m128 _fLoad;

                // mean
                __m128 _fsum = _mm_setzero_ps();

                for (int j = 0; j < w; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j << 2));
                    _fsum = _mm_add_ps(_fsum, _fLoad);
                }

                // var
                __m128 _size = _mm_set1_ps((float)w);
                __m128 _mean = _mm_div_ps(_fsum, _size);
                __m128 _fsqsum = _mm_setzero_ps();

                for (int j = 0; j < w; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j << 2));
                    _fLoad = _mm_sub_ps(_fLoad, _mean);
                    _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                }
                __m128 _var = _mm_div_ps(_fsqsum, _size);

                __m128 _eps = _mm_set1_ps(eps);
                __m128 _a = _mm_add_ps(_var, _eps);
                _a = _mm_rsqrt_ps(_a);
                __m128 _b = _mm_mul_ps(-_mean, _a);
                __m128 _gamma, _beta;

                if (affine)
                {
                    for (int j = 0; j < w; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);

                        _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                        _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                        _fLoad = _mm_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm_add_ps(_fLoad, _beta);

                        _mm_store_ps(ptr + (j << 2), _fLoad);
                    }
                }
                else
                {
                    for (int j = 0; j < w; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);
                        _mm_store_ps(ptr + (j << 2), _fLoad);
                    }
                }
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
#pragma omp parallel for num_threads(opt.num_threads)
                for (int qq = 0; qq < channels; qq++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(qq).row(i);

                        __m128 _fLoad;

                        // mean
                        __m128 _fsum = _mm_setzero_ps();

                        for (int j = 0; j < w; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j << 2));
                            _fsum = _mm_add_ps(_fsum, _fLoad);
                        }

                        // var
                        __m128 _size = _mm_set1_ps((float)w);
                        __m128 _mean = _mm_div_ps(_fsum, _size);
                        __m128 _fsqsum = _mm_setzero_ps();

                        for (int j = 0; j < w; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j << 2));
                            _fLoad = _mm_sub_ps(_fLoad, _mean);
                            _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                            _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                        }

                        __m128 _var = _mm_div_ps(_fsqsum, _size);
                        __m128 _eps = _mm_set1_ps(eps);
                        __m128 _a = _mm_add_ps(_var, _eps);
                        _a = _mm_rsqrt_ps(_a);
                        __m128 _b = _mm_mul_ps(-_mean, _a);
                        __m128 _gamma, _beta;

                        if (affine)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm_load_ps(ptr + (j << 2));
                                _fLoad = _mm_mul_ps(_fLoad, _a);
                                _fLoad = _mm_add_ps(_fLoad, _b);

                                _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                                _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                                _fLoad = _mm_mul_ps(_fLoad, _gamma);
                                _fLoad = _mm_add_ps(_fLoad, _beta);

                                _mm_store_ps(ptr + (j << 2), _fLoad);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm_load_ps(ptr + (j << 2));
                                _fLoad = _mm_mul_ps(_fLoad, _a);
                                _fLoad = _mm_add_ps(_fLoad, _b);
                                _mm_store_ps(ptr + (j << 2), _fLoad);
                            }
                        }
                    }
                }
            }

            else // if (affine_size == size)
            {
#pragma omp parallel for num_threads(opt.num_threads)
                for (int qq = 0; qq < channels; qq++)
                {
                    float* ptr = bottom_top_blob.channel(qq);

                    __m128 _fLoad;

                    // mean

                    __m128 _fsum = _mm_setzero_ps();

                    for (int j = 0; j < size; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fsum = _mm_add_ps(_fsum, _fLoad);
                    }

                    // var
                    __m128 _size = _mm_set1_ps((float)size);
                    __m128 _mean = _mm_div_ps(_fsum, _size);
                    __m128 _fsqsum = _mm_setzero_ps();

                    for (int j = 0; j < size; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fLoad = _mm_sub_ps(_fLoad, _mean);

                        _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                        _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                    }

                    __m128 _var = _mm_div_ps(_fsqsum, _size);
                    __m128 _eps = _mm_set1_ps(eps);
                    __m128 _a = _mm_add_ps(_var, _eps);
                    _a = _mm_rsqrt_ps(_a);
                    __m128 _b = _mm_mul_ps(-_mean, _a);
                    __m128 _gamma, _beta;

                    if (affine)
                    {
                        for (int j = 0; j < size; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j << 2));
                            _fLoad = _mm_mul_ps(_fLoad, _a);
                            _fLoad = _mm_add_ps(_fLoad, _b);

                            _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                            _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                            _fLoad = _mm_mul_ps(_fLoad, _gamma);
                            _fLoad = _mm_add_ps(_fLoad, _beta);

                            _mm_store_ps(ptr + (j << 2), _fLoad);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < size; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j << 2));
                            _fLoad = _mm_mul_ps(_fLoad, _a);
                            _fLoad = _mm_add_ps(_fLoad, _b);
                            _mm_store_ps(ptr + (j << 2), _fLoad);
                        }
                    }
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (dims != 4)
        return LayerNorm::forward_inplace(bottom_top_blob, opt);

#if __SSE2__
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
#if __AVX__

    if (affine_size == w)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < channels; qq++)
        {
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.channel(qq).row(i);
                int ww = w >> 3;
                int remainw = ww << 3;

                __m256 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < ww; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                const float* q = (const float*)&_fsum;

                sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

                for (int i = remainw; i < w; i++)
                    sum += ptr[i];

                // var
                float mean = sum / w;
                __m256 _mean = _mm256_set1_ps(mean);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < ww; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                q = (const float*)&_fsqsum;
                sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

                for (int i = remainw; i < w; i++)
                {
                    sqsum += (ptr[i] - mean) * (ptr[i] - mean);
                }

                float var = sqsum / w;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < ww; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);

                        _gamma = _mm256_loadu_ps((const float*)gamma_data + (i << 3));
                        _beta = _mm256_loadu_ps((const float*)beta_data + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }

                    for (int i = remainw; i < w; i++)
                    {
                        ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                    }
                }
                else
                {
                    for (int i = 0; i < ww; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }
                    for (int i = remainw; i < w; i++)
                    {
                        ptr[i] = ptr[i] * a + b;
                    }
                }
            }
        }
    }
    else // if (affine_size == size)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < channels; qq++)
        {
            {
                float* ptr = bottom_top_blob.channel(qq);
                int ssize = size >> 3;
                int remain_size = ssize << 3;

                __m256 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                const float* q = (const float*)&_fsum;

                sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

                for (int i = remain_size; i < size; i++)
                    sum += ptr[i];

                // var
                float mean = sum / size;
                __m256 _mean = _mm256_set1_ps(mean);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                q = (const float*)&_fsqsum;
                sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

                for (int i = remain_size; i < size; i++)
                {
                    sqsum += (ptr[i] - mean) * (ptr[i] - mean);
                }

                float var = sqsum / size;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < ssize; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);

                        _gamma = _mm256_loadu_ps((const float*)gamma_data + (i << 3));
                        _beta = _mm256_loadu_ps((const float*)beta_data + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }

                    for (int i = remain_size; i < size; i++)
                    {
                        ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                    }
                }
                else
                {
                    for (int i = 0; i < ssize; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i << 3));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i << 3), _fLoad);
                    }
                    for (int i = remain_size; i < size; i++)
                    {
                        ptr[i] = ptr[i] * a + b;
                    }
                }
            }
        }
    }

    return 0;
#endif // __AVX__
    if (affine_size == w)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < channels; qq++)
        {
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.channel(qq).row(i);

                int ww = w >> 2;
                int remainw = ww << 2;

                __m128 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m128 _fsum = _mm_setzero_ps();

                for (int j = 0; j < ww; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j << 2));
                    _fsum = _mm_add_ps(_fsum, _fLoad);
                }

                const float* q = (const float*)&_fsum;
                sum = q[0] + q[1] + q[2] + q[3];

                for (int j = remainw; j < w; j++)
                    sum += ptr[j];

                // var
                float mean = sum / w;
                __m128 _mean = _mm_set1_ps(mean);
                __m128 _fsqsum = _mm_setzero_ps();

                for (int j = 0; j < ww; j++)
                {
                    _fLoad = _mm_sub_ps(_fLoad, _mean);
                    _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                }

                q = (const float*)&_fsqsum;
                sqsum = q[0] + q[1] + q[2] + q[3];
                for (int j = remainw; j < w; j++)
                {
                    sqsum += (ptr[j] - mean) * (ptr[j] - mean);
                }
                float var = sqsum / w;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m128 _a = _mm_set1_ps(a);
                __m128 _b = _mm_set1_ps(b);
                __m128 _gamma, _beta;

                if (affine)
                {
                    for (int j = 0; j < ww; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);

                        _gamma = _mm_load_ps((const float*)gamma_data + (j << 2));
                        _beta = _mm_load_ps((const float*)beta_data + (j << 2));
                        _fLoad = _mm_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm_add_ps(_fLoad, _beta);

                        _mm_store_ps(ptr + (j << 2), _fLoad);
                    }

                    for (int j = remainw; j < w; j++)
                    {
                        ptr[j] = (ptr[j] * a + b) * gamma_data[j] + beta_data[j];
                    }
                }
                else
                {
                    for (int j = 0; j < ww; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j << 2));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);
                        _mm_store_ps(ptr + (j << 2), _fLoad);
                    }
                    for (int j = remainw; j < w; j++)
                    {
                        ptr[j] = ptr[j] * a + b;
                    }
                }
            }
        }
    }

    else // if (affine_size == size)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < channels; qq++)
        {
            float* ptr = bottom_top_blob.channel(qq);
            int ssize = size >> 2;
            int remainsize = ssize << 2;

            __m128 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m128 _fsum = _mm_setzero_ps();

            for (int i = 0; i < ssize; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i << 2));
                _fsum = _mm_add_ps(_fsum, _fLoad);
            }
            const float* q = (const float*)&_fsum;

            sum = q[0] + q[1] + q[2] + q[3];

            for (int i = remainsize; i < size; i++)
                sum += ptr[i];

            float mean = sum / size;
            __m128 _mean = _mm_set1_ps(mean);
            __m128 _fsqsum = _mm_setzero_ps();

            for (int i = 0; i < ssize; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i << 2));
                _fLoad = _mm_sub_ps(_fLoad, _mean);
                _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
            }

            q = (const float*)&_fsqsum;
            sqsum = q[0] + q[1] + q[2] + q[3];

            for (int i = remainsize; i < size; i++)
            {
                sqsum += (ptr[i] - mean) * (ptr[i] - mean);
            }

            // var
            float var = sqsum / size;
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            __m128 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm_loadu_ps(ptr + (i << 2));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);

                    _gamma = _mm_load_ps((const float*)gamma_data + (i << 3));
                    _beta = _mm_load_ps((const float*)beta_data + (i << 3));
                    _fLoad = _mm_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm_add_ps(_fLoad, _beta);

                    _mm_store_ps(ptr + (i << 2), _fLoad);
                }
                for (int i = remainsize; i < size; i++)
                {
                    ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    _fLoad = _mm_loadu_ps(ptr + (i << 2));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);
                    _mm_store_ps(ptr + (i << 2), _fLoad);
                }
                for (int i = remainsize; i < size; i++)
                {
                    ptr[i] = (ptr[i] * a + b);
                }
            }
        }
    }

#endif // __SSE2__
    return 0;
}


} // namespace ncnn
