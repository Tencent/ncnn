// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_powerpc.h"

#include <float.h>
#include <math.h>

#include "cpu.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

Softmax_powerpc::Softmax_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

// softmax over a contiguous run of size = elemcount * elempack elements
static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __ALTIVEC__
    const bool vectorized = elempack == 1 || elempack == 4;
#endif // __ALTIVEC__

    // reduce max
#if __ALTIVEC__
    v4f32 _max = vec_splats(-FLT_MAX);
#endif // __ALTIVEC__
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        int i = 0;
#if __ALTIVEC__
        if (vectorized)
        {
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _max = vec_max(_max, _p);
                ptr += 4;
            }
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

#if __ALTIVEC__
    if (elempack == 1)
    {
        max = std::max(max, __ncnn_altivec_reduce_max(_max));
        _max = vec_splats(max);
    }
#endif // __ALTIVEC__

    // reduce exp(x - max)
#if __ALTIVEC__
    v4f32 _sum = vec_splats(0.f);
#endif // __ALTIVEC__
    float sum = 0.f;
    {
        float* ptr = _ptr;

        int i = 0;
#if __ALTIVEC__
        if (vectorized)
        {
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = vec_sub(_p, _max);
                _p = exp_ps(_p);
                __ncnn_altivec_store(ptr, _p);
                _sum = vec_add(_sum, _p);
                ptr += 4;
            }
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

#if __ALTIVEC__
    if (elempack == 4)
    {
        _sum = __ncnn_altivec_recip2(_sum);
    }
#endif // __ALTIVEC__
    if (elempack == 1)
    {
#if __ALTIVEC__
        sum += __ncnn_altivec_reduce_add(_sum);
#endif // __ALTIVEC__

        sum = 1.f / sum;

#if __ALTIVEC__
        _sum = vec_splats(sum);
#endif // __ALTIVEC__
    }

    // div sum
    {
        float* ptr = _ptr;

        int i = 0;
#if __ALTIVEC__
        if (vectorized)
        {
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = __ncnn_altivec_mul(_p, _sum);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
            }
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

#if __ALTIVEC__
static void softmax_pack4(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p0 = __ncnn_altivec_load(ptr);
            v4f32 _p1 = __ncnn_altivec_load(ptr + 4);
            v4f32 _p2 = __ncnn_altivec_load(ptr + 8);
            v4f32 _p3 = __ncnn_altivec_load(ptr + 12);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            v4f32 _max01 = vec_max(_p0, _p1);
            v4f32 _max23 = vec_max(_p2, _p3);
            v4f32 _max0123 = vec_max(_max01, _max23);
            v4f32 _max = __ncnn_altivec_load(maxptr);
            _max = vec_max(_max, _max0123);
            __ncnn_altivec_store(maxptr, _max);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            *maxptr = std::max(*maxptr, __ncnn_altivec_reduce_max(_p));
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p0 = __ncnn_altivec_load(ptr);
            v4f32 _p1 = __ncnn_altivec_load(ptr + 4);
            v4f32 _p2 = __ncnn_altivec_load(ptr + 8);
            v4f32 _p3 = __ncnn_altivec_load(ptr + 12);
            _p0 = exp_ps(vec_sub(_p0, vec_splats(maxptr[0])));
            _p1 = exp_ps(vec_sub(_p1, vec_splats(maxptr[1])));
            _p2 = exp_ps(vec_sub(_p2, vec_splats(maxptr[2])));
            _p3 = exp_ps(vec_sub(_p3, vec_splats(maxptr[3])));
            __ncnn_altivec_store(ptr, _p0);
            __ncnn_altivec_store(ptr + 4, _p1);
            __ncnn_altivec_store(ptr + 8, _p2);
            __ncnn_altivec_store(ptr + 12, _p3);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            v4f32 _sum01 = vec_add(_p0, _p1);
            v4f32 _sum23 = vec_add(_p2, _p3);
            v4f32 _sum0123 = vec_add(_sum01, _sum23);
            v4f32 _sum = __ncnn_altivec_load(sumptr);
            _sum = vec_add(_sum, _sum0123);
            __ncnn_altivec_store(sumptr, _sum);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _max = vec_splats(*maxptr);
            _p = exp_ps(vec_sub(_p, _max));
            __ncnn_altivec_store(ptr, _p);
            *sumptr += __ncnn_altivec_reduce_add(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _sum = __ncnn_altivec_load(sumptr);
            _sum = __ncnn_altivec_recip2(_sum);
            __ncnn_altivec_store(sumptr, _sum);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p0 = __ncnn_altivec_load(ptr);
            v4f32 _p1 = __ncnn_altivec_load(ptr + 4);
            v4f32 _p2 = __ncnn_altivec_load(ptr + 8);
            v4f32 _p3 = __ncnn_altivec_load(ptr + 12);
            _p0 = __ncnn_altivec_mul(_p0, vec_splats(sumptr[0]));
            _p1 = __ncnn_altivec_mul(_p1, vec_splats(sumptr[1]));
            _p2 = __ncnn_altivec_mul(_p2, vec_splats(sumptr[2]));
            _p3 = __ncnn_altivec_mul(_p3, vec_splats(sumptr[3]));
            __ncnn_altivec_store(ptr, _p0);
            __ncnn_altivec_store(ptr + 4, _p1);
            __ncnn_altivec_store(ptr + 8, _p2);
            __ncnn_altivec_store(ptr + 12, _p3);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            _p = __ncnn_altivec_mul(_p, vec_splats(*sumptr));
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __ALTIVEC__

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __ALTIVEC__
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _max = __ncnn_altivec_load(maxptr);
            _max = vec_max(_max, _p);
            __ncnn_altivec_store(maxptr, _max);
            ptr += 4;
            maxptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, *ptr);
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __ALTIVEC__
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _max = __ncnn_altivec_load(maxptr);
            v4f32 _sum = __ncnn_altivec_load(sumptr);
            _p = vec_sub(_p, _max);
            _p = exp_ps(_p);
            __ncnn_altivec_store(ptr, _p);
            _sum = vec_add(_sum, _p);
            __ncnn_altivec_store(sumptr, _sum);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            float v = expf(*ptr - *maxptr);
            *ptr = v;
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __ALTIVEC__
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _sum = __ncnn_altivec_load(sumptr);
            _sum = __ncnn_altivec_recip2(_sum);
            __ncnn_altivec_store(sumptr, _sum);
            sumptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        float* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __ALTIVEC__
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4f32 _sum = __ncnn_altivec_load(sumptr);
            _p = __ncnn_altivec_mul(_p, _sum);
            __ncnn_altivec_store(ptr, _p);
            ptr += 4;
            sumptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            *ptr *= *sumptr;
            ptr++;
            sumptr++;
        }
    }
}

static void softmax(float* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // init max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __ALTIVEC__
        v4f32 _negmax = vec_splats(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            __ncnn_altivec_store(maxptr, _negmax);
            maxptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __ALTIVEC__
        v4f32 _zero = vec_splats(0.f);
        for (; j + 3 < size1; j += 4)
        {
            __ncnn_altivec_store(sumptr, _zero);
            sumptr += 4;
        }
#endif // __ALTIVEC__
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __ALTIVEC__
    if (elempack == 4)
    {
        softmax_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }
#endif // __ALTIVEC__
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
        return;
    }

    // any other elempack, the pack lanes are independent so the scalar pack1 path is valid
    softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
}

int Softmax_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        softmax(ptr, w * elempack, 1);
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w, elempack);
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + sizen;

            float* ptr = (float*)bottom_top_blob + i * elempack;

            softmax(ptr, channels, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w, elempack);
                ptr += w * elempack;
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
