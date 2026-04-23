// Copyright 2020 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_mips.h"

#include <float.h>
#include <math.h>

#include "cpu.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

#if NCNN_BF16
#include "softmax_bf16s.h"
#endif

Softmax_mips::Softmax_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void softmax(float* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __mips_msa
    v4f32 _max = (v4f32)__msa_fill_w_f32(-FLT_MAX);
#endif // __mips_msa
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _max = __msa_fmax_w(_max, _p);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

#if __mips_msa
    if (elempack == 1)
    {
        max = std::max(max, __msa_reduce_fmax_w(_max));
        _max = (v4f32)__msa_fill_w_f32(max);
    }
#endif // __mips_msa

    // reduce exp(x - max)
#if __mips_msa
    v4f32 _sum = (v4f32)__msa_fill_w_f32(0.f);
#endif // __mips_msa
    float sum = 0.f;
    {
        float* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fsub_w(_p, _max);
            _p = exp_ps(_p);
            __msa_st_w((v4i32)_p, ptr, 0);
            _sum = __msa_fadd_w(_sum, _p);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = expf(*ptr - max);
            *ptr = v;
            sum += v;
            ptr++;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        _sum = __msa_fdiv_w(_one, _sum);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa

        sum = 1.f / sum;

#if __mips_msa
        _sum = (v4f32)__msa_fill_w_f32(sum);
#endif // __mips_msa
    }

    // div sum
    {
        float* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = __msa_fmul_w(_p, _sum);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr++ *= sum;
        }
    }
}

#if __mips_msa
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
            v4f32 _p0 = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(ptr + 4, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(ptr + 8, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(ptr + 12, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            v4f32 _max01 = __msa_fmax_w(_p0, _p1);
            v4f32 _max23 = __msa_fmax_w(_p2, _p3);
            v4f32 _max0123 = __msa_fmax_w(_max01, _max23);
            v4f32 _max = (v4f32)__msa_ld_w(maxptr, 0);
            _max = __msa_fmax_w(_max, _max0123);
            __msa_st_w((v4i32)_max, maxptr, 0);
            ptr += 16;
            maxptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            *maxptr = std::max(*maxptr, __msa_reduce_fmax_w(_p));
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
            v4f32 _p0 = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(ptr + 4, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(ptr + 8, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(ptr + 12, 0);
            _p0 = exp_ps(__msa_fsub_w(_p0, (v4f32)__msa_fill_w_f32(maxptr[0])));
            _p1 = exp_ps(__msa_fsub_w(_p1, (v4f32)__msa_fill_w_f32(maxptr[1])));
            _p2 = exp_ps(__msa_fsub_w(_p2, (v4f32)__msa_fill_w_f32(maxptr[2])));
            _p3 = exp_ps(__msa_fsub_w(_p3, (v4f32)__msa_fill_w_f32(maxptr[3])));
            __msa_st_w((v4i32)_p0, ptr, 0);
            __msa_st_w((v4i32)_p1, ptr + 4, 0);
            __msa_st_w((v4i32)_p2, ptr + 8, 0);
            __msa_st_w((v4i32)_p3, ptr + 12, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            v4f32 _sum01 = __msa_fadd_w(_p0, _p1);
            v4f32 _sum23 = __msa_fadd_w(_p2, _p3);
            v4f32 _sum0123 = __msa_fadd_w(_sum01, _sum23);
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            _sum = __msa_fadd_w(_sum, _sum0123);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            ptr += 16;
            maxptr += 4;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _max = (v4f32)__msa_fill_w_f32(*maxptr);
            _p = exp_ps(__msa_fsub_w(_p, _max));
            __msa_st_w((v4i32)_p, ptr, 0);
            *sumptr += __msa_reduce_fadd_w(_p);
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
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            _sum = __msa_fdiv_w(_one, _sum);
            __msa_st_w((v4i32)_sum, sumptr, 0);
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
            v4f32 _p0 = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(ptr + 4, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(ptr + 8, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(ptr + 12, 0);
            _p0 = __msa_fmul_w(_p0, (v4f32)__msa_fill_w_f32(sumptr[0]));
            _p1 = __msa_fmul_w(_p1, (v4f32)__msa_fill_w_f32(sumptr[1]));
            _p2 = __msa_fmul_w(_p2, (v4f32)__msa_fill_w_f32(sumptr[2]));
            _p3 = __msa_fmul_w(_p3, (v4f32)__msa_fill_w_f32(sumptr[3]));
            __msa_st_w((v4i32)_p0, ptr, 0);
            __msa_st_w((v4i32)_p1, ptr + 4, 0);
            __msa_st_w((v4i32)_p2, ptr + 8, 0);
            __msa_st_w((v4i32)_p3, ptr + 12, 0);
            ptr += 16;
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _sum = (v4f32)__msa_fill_w_f32(*sumptr);
            _p = __msa_fmul_w(_p, _sum);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __mips_msa

static void softmax_pack1(float* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const float* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _max = (v4f32)__msa_ld_w(maxptr, 0);
            _max = __msa_fmax_w(_max, _p);
            __msa_st_w((v4i32)_max, maxptr, 0);
            ptr += 4;
            maxptr += 4;
        }
#endif // __mips_msa
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
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _max = (v4f32)__msa_ld_w(maxptr, 0);
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            _p = __msa_fsub_w(_p, _max);
            _p = exp_ps(_p);
            __msa_st_w((v4i32)_p, ptr, 0);
            _sum = __msa_fadd_w(_sum, _p);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __mips_msa
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
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            _sum = __msa_fdiv_w(_one, _sum);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            sumptr += 4;
        }
#endif // __mips_msa
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
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            _p = __msa_fmul_w(_p, _sum);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
            sumptr += 4;
        }
#endif // __mips_msa
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
#if __mips_msa
        v4f32 _negmax = (v4f32)__msa_fill_w_f32(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            __msa_st_w((v4i32)_negmax, maxptr, 0);
            maxptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __mips_msa
        v4f32 _zero = (v4f32)__msa_fill_w_f32(0.f);
        for (; j + 3 < size1; j += 4)
        {
            __msa_st_w((v4i32)_zero, sumptr, 0);
            sumptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        softmax_pack4(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
        softmax_pack1(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}

int Softmax_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

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

        const int size = w * elempack;

        softmax(ptr, size, 1);
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

#if NCNN_BF16
int Softmax_mips::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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

        softmax_bf16s_msa(ptr, size, 1);
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

            softmax_bf16s_msa_dispatch(ptr, h, elempack, stride, size1, maxptr, sumptr);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            softmax_bf16s_msa(ptr, w, elempack);
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

            softmax_bf16s_msa_dispatch(ptr, channels, elempack, stride, size1, maxptr, sumptr);
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

                softmax_bf16s_msa_dispatch(ptr, h, 1, size, size, maxptr, sumptr);
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
                softmax_bf16s_msa(ptr, w, elempack);
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

            softmax_bf16s_msa_dispatch(ptr, d, 1, size, size, maxptr, sumptr);
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
                    softmax_bf16s_msa(ptr, w, elempack);
                    ptr += w * elempack;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
