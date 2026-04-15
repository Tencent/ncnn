// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_loongarch.h"

#include <float.h>
#include <math.h>

#include "cpu.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#include "loongarch_usability.h"
#if __loongarch_asx
#include <lasxintrin.h>
#include "lasx_mathfun.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

#if NCNN_BF16
#include "softmax_bf16s.h"
#endif

int Softmax_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    if (dims != 3 || positive_axis != 0)
        return Softmax::forward_inplace(bottom_top_blob, opt);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    Mat max;
    max.create(w, h, elemsize, opt.workspace_allocator);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i = 0; i < size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _max = (__m128)__lsx_vld(maxptr, 0);

            _p = exp_ps(__lsx_vfsub_s(_p, _max));

            __lsx_vst(_p, ptr, 0);

            ptr += 4;
            maxptr += 4;
        }
#endif // __loongarch_sx

        for (; i < size; i++)
        {
            *ptr = expf(*ptr - *maxptr);

            ptr++;
            maxptr++;
        }
    }

    Mat sum;
    sum.create(w, h, elemsize, opt.workspace_allocator);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _sum = __lsx_vfadd_s(_sum, _p);
            __lsx_vst(_sum, sumptr, 0);

            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx

        for (; i < size; i++)
        {
            *sumptr += *ptr;

            ptr++;
            sumptr++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _sum = (__m128)__lsx_vld(sumptr, 0);
            _p = __lsx_vfdiv_s(_p, _sum);
            __lsx_vst(_p, ptr, 0);

            ptr += 4;
            sumptr += 4;
        }
#endif // __loongarch_sx

        for (; i < size; i++)
        {
            *ptr /= *sumptr;

            ptr++;
            sumptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Softmax_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_bf16s_lsx(ptr, size, 1);
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

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s_lsx_dispatch(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            softmax_bf16s_lsx(ptr, w, elempack);
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

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

            softmax_bf16s_lsx_dispatch(ptr, channels, elempack, stride, size1, maxptr, sumptr);
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
                unsigned short* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax_bf16s_lsx_dispatch(ptr, h, 1, size, size, maxptr, sumptr);
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax_bf16s_lsx(ptr, w, elempack);
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
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax_bf16s_lsx_dispatch(ptr, d, 1, size, size, maxptr, sumptr);
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_bf16s_lsx(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
