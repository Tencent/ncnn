// Copyright 2025 xiaofan <xiaofan@iscas.ac.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int Eltwise_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_top_blob = bottom_blobs[0];

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
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_top_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);

                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(ptr1, vl);
                _p = __riscv_vfmul_vv_f16m8(_p, _p1, vl);
                __riscv_vse16_v_f16m8(outptr, _p, vl);

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

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);

                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(outptr, vl);
                    _p1 = __riscv_vfmul_vv_f16m8(_p1, _p, vl);
                    __riscv_vse16_v_f16m8(outptr, _p1, vl);

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
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_top_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);

                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(ptr1, vl);
                    _p = __riscv_vfadd_vv_f16m8(_p, _p1, vl);
                    __riscv_vse16_v_f16m8(outptr, _p, vl);

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

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m8(n);

                        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(outptr, vl);
                        _p1 = __riscv_vfadd_vv_f16m8(_p1, _p, vl);
                        __riscv_vse16_v_f16m8(outptr, _p1, vl);

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
            const Mat& bottom_blob1 = bottom_blobs[1];
            __fp16 coeff0 = (__fp16)coeffs[0];
            __fp16 coeff1 = (__fp16)coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_top_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);

                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(ptr1, vl);
                    _p = __riscv_vfmul_vf_f16m8(_p, coeff0, vl);
                    _p1 = __riscv_vfmul_vf_f16m8(_p1, coeff1, vl);
                    _p = __riscv_vfadd_vv_f16m8(_p, _p1, vl);
                    __riscv_vse16_v_f16m8(outptr, _p, vl);

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

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                __fp16 coeff = (__fp16)coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e16m8(n);

                        vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                        vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(outptr, vl);
                        _p = __riscv_vfmul_vf_f16m8(_p, coeff, vl);
                        _p1 = __riscv_vfadd_vv_f16m8(_p1, _p, vl);
                        __riscv_vse16_v_f16m8(outptr, _p1, vl);

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
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_top_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);

                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(ptr1, vl);
                _p = __riscv_vfmax_vv_f16m8(_p, _p1, vl);
                __riscv_vse16_v_f16m8(outptr, _p, vl);

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

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);
#if __riscv_zvfh
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);

                    vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                    vfloat16m8_t _p1 = __riscv_vle16_v_f16m8(outptr, vl);
                    _p1 = __riscv_vfmax_vv_f16m8(_p1, _p, vl);
                    __riscv_vse16_v_f16m8(outptr, _p1, vl);

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
#endif // NCNN_ZFH

} // namespace ncnn
