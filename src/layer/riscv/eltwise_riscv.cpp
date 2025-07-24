// Copyright 2025 xiaofan <xiaofan@iscas.ac.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Eltwise_riscv::Eltwise_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

int Eltwise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_top_blob = bottom_blobs[0];
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_top_blob, opt.blob_allocator);

    if (op_type == Operation_PROD)
    {
        // top_blob = bottom_top_blob * bottom_blobs[1]
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                _p = __riscv_vfmul_vv_f32m8(_p, _p1, vl);
                __riscv_vse32_v_f32m8(outptr, _p, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
#endif
        }

        // top_blob *= bottom_blobs[i] for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);

                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(outptr, vl);
                    _p1 = __riscv_vfmul_vv_f32m8(_p1, _p, vl);
                    __riscv_vse32_v_f32m8(outptr, _p1, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
#endif
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.empty())
        {
            // top_blob = bottom_top_blob + bottom_blobs[1]
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);

                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                    _p = __riscv_vfadd_vv_f32m8(_p, _p1, vl);
                    __riscv_vse32_v_f32m8(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
#endif
            }

            // top_blob += bottom_blobs[i] for i = 2, 3, ...
            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(outptr, vl);
                        _p1 = __riscv_vfadd_vv_f32m8(_p1, _p, vl);
                        __riscv_vse32_v_f32m8(outptr, _p1, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
#endif
                }
            }
        }
        else
        {
            // top_blob = bottom_top_blob * coeffs[0] + bottom_blobs[1] * coeffs[1]
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);

                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                    _p = __riscv_vfmul_vf_f32m8(_p, coeff0, vl);
                    _p1 = __riscv_vfmul_vf_f32m8(_p1, coeff1, vl);
                    _p = __riscv_vfadd_vv_f32m8(_p, _p1, vl);
                    __riscv_vse32_v_f32m8(outptr, _p, vl);

                    ptr += vl;
                    ptr1 += vl;
                    outptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
#endif
            }

            // top_blob += bottom_blobs[i] * coeffs[i] for i = 2, 3, ...
            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(outptr, vl);
                        _p = __riscv_vfmul_vf_f32m8(_p, coeff, vl);
                        _p1 = __riscv_vfadd_vv_f32m8(_p1, _p, vl);
                        __riscv_vse32_v_f32m8(outptr, _p1, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                    }
#else
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
#endif
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // top_blob = max(bottom_top_blob, bottom_blobs[1])
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);

                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(ptr1, vl);
                _p = __riscv_vfmax_vv_f32m8(_p, _p1, vl);
                __riscv_vse32_v_f32m8(outptr, _p, vl);

                ptr += vl;
                ptr1 += vl;
                outptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
#endif
        }

        // top_blob = max(top_blob, bottom_blobs[i]) for i = 2, 3, ...
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);

                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vfloat32m8_t _p1 = __riscv_vle32_v_f32m8(outptr, vl);
                    _p1 = __riscv_vfmax_vv_f32m8(_p1, _p, vl);
                    __riscv_vse32_v_f32m8(outptr, _p1, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
#endif
            }
        }
    }

    return 0;
}

} // namespace ncnn
