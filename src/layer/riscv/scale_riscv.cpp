// Copyright 2026 darkavatar23 <matteo.forzan95@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "scale_bf16s.h"
#endif

Scale_riscv::Scale_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_riscv::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

#if NCNN_ZFH || NCNN_BF16
    int elembits = bottom_top_blob.elembits();
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const float* ptr_s = scale_blob;
#if __riscv_vector
        if (bias_term)
        {
            const float* ptr_b = bias_data;
            int n = bottom_top_blob.w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(ptr_s, vl);
                vfloat32m8_t _bias = __riscv_vle32_v_f32m8(ptr_b, vl);

                _p = __riscv_vfmadd_vv_f32m8(_p, _s, _bias, vl);

                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                ptr_s += vl;
                ptr_b += vl;
                n -= vl;
            }
        }
        else
        {
            int n = bottom_top_blob.w * elempack;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(ptr_s, vl);

                _p = __riscv_vfmul_vv_f32m8(_p, _s, vl);

                __riscv_vse32_v_f32m8(ptr, _p, vl);

                ptr += vl;
                ptr_s += vl;
                n -= vl;
            }
        }
#else // __riscv_vector
        int w = bottom_top_blob.w;
        if (bias_term)
        {
            const float* ptr_b = bias_data;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] = ptr[i] * ptr_s[i] + ptr_b[i];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                ptr[i] *= ptr_s[i];
            }
        }
#endif // __riscv_vector
        return 0;
    }

#if __riscv_vector
    if (elempack == 1)
#endif
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (dims == 2)
        {
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float s = scale_blob[i];
                    float bias = bias_data[i];

#if __riscv_vector
                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
                        _p = __riscv_vfadd_vf_f32m8(_p, bias, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
#else  // __riscv_vector
                    for (int j = 0; j < w; j++)
                    {
                        ptr[j] = ptr[j] * s + bias;
                    }
#endif // __riscv_vector
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float s = scale_blob[i];

#if __riscv_vector
                    int n = w;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
#else  // __riscv_vector
                    for (int j = 0; j < w; j++)
                    {
                        ptr[j] *= s;
                    }
#endif // __riscv_vector
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int d = bottom_top_blob.d;
            int channels = bottom_top_blob.c;
            int size = w * h * d;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float s = scale_blob[q];
                    float bias = bias_data[q];

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
                        _p = __riscv_vfadd_vf_f32m8(_p, bias, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
#else  // __riscv_vector
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] = ptr[i] * s + bias;
                    }
#endif // __riscv_vector
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float s = scale_blob[q];

#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        _p = __riscv_vfmul_vf_f32m8(_p, s, vl);
                        __riscv_vse32_v_f32m8(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
#else  // __riscv_vector
                    for (int i = 0; i < size; i++)
                    {
                        ptr[i] *= s;
                    }
#endif // __riscv_vector
                }
            }
        }

        return 0;
    }

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e32m1(packn);

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (dims == 2)
        {
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_blob + i * elempack, vl);
                    vfloat32m1_t _bias = __riscv_vle32_v_f32m1((const float*)bias_data + i * elempack, vl);

                    int n = w * elempack;
                    while (n > 0)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        _p = __riscv_vfmadd_vv_f32m1(_p, _s, _bias, vl);
                        __riscv_vse32_v_f32m1(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_blob + i * elempack, vl);

                    int n = w * elempack;
                    while (n > 0)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        _p = __riscv_vfmul_vv_f32m1(_p, _s, vl);
                        __riscv_vse32_v_f32m1(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int d = bottom_top_blob.d;
            int channels = bottom_top_blob.c;
            int size = w * h * d * elempack;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_blob + q * elempack, vl);
                    vfloat32m1_t _bias = __riscv_vle32_v_f32m1((const float*)bias_data + q * elempack, vl);

                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        _p = __riscv_vfmadd_vv_f32m1(_p, _s, _bias, vl);
                        __riscv_vse32_v_f32m1(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    vfloat32m1_t _s = __riscv_vle32_v_f32m1((const float*)scale_blob + q * elempack, vl);

                    int n = size;
                    while (n > 0)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        _p = __riscv_vfmul_vv_f32m1(_p, _s, vl);
                        __riscv_vse32_v_f32m1(ptr, _p, vl);

                        ptr += vl;
                        n -= vl;
                    }
                }
            }
        }
    }
#endif // __riscv_vector

    return 0;
}

#if NCNN_BF16
int Scale_riscv::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    // scale_blob may be bf16 (from the second input) or fp32 (from scale_data weight)
    const float* scale = 0;
    Mat scale_fp32;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;

        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }

        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }

    const float* bias = bias_term ? (const float*)bias_data : 0;

    scale_bf16s(bottom_top_blob, scale, bias, bias_term, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
