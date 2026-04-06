// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Dequantize_riscv::Dequantize_riscv()
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
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    float scale = scale_data[0];

#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();
    const size_t vlm2 = __riscv_vsetvlmax_e32m2();
    vfloat32m8_t _scale;
    if (scale_data.w == 1)
    {
        _scale = __riscv_vfmv_v_f_f32m8(scale, __riscv_vsetvlmax_e32m8());
    }
    else if (elempack == vlm1)
    {
        vfloat32m1_t _s = __riscv_vle32_v_f32m1(scale_data, vlm1);
        _scale = __riscv_vcreate_v_f32m1_f32m8(_s, _s, _s, _s, _s, _s, _s, _s);
    }
#endif // __riscv_vector

    if (bias_data.w == 0)
    {
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(__riscv_vle32_v_i32m8(intptr, vl), vl);
            _v = __riscv_vfmul_vv_f32m8(_v, _scale, vl);
            __riscv_vse32_v_f32m8(ptr, _v, vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
    else
    {
        float bias = bias_data[0];
#if __riscv_vector
        vfloat32m8_t _bias;
        if (bias_data.w == 1)
        {
            _bias = __riscv_vfmv_v_f_f32m8(bias, __riscv_vsetvlmax_e32m8());
        }
        else if (elempack == vlm1)
        {
            vfloat32m1_t _b = __riscv_vle32_v_f32m1(bias_data, vlm1);
            _bias = __riscv_vcreate_v_f32m1_f32m8(_b, _b, _b, _b, _b, _b, _b, _b);
        }

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(__riscv_vle32_v_i32m8(intptr, vl), vl);
            _v = __riscv_vfmacc_vv_f32m8(_bias, _v, _scale, vl);
            __riscv_vse32_v_f32m8(ptr, _v, vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
}

int Dequantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            float* ptr = (float*)top_blob + i * elempack;
            const int size = std::min(w - i, wp) * elempack;

            dequantize(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);
            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}
} // namespace ncnn
