// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cast_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Cast_riscv::Cast_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Cast_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 4)
    {
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int size = w * h * d * elempack;

#if NCNN_ZFH
    if (type_from == 1 && type_to == 2)
    {
#if __riscv_vector
        if (cpu_support_riscv_zvfh())
#else
        if (cpu_support_riscv_zfh())
#endif
        {
            cast_fp32_to_fp16(bottom_blob, top_blob, opt);
            return 0;
        }
    }

    if (type_from == 2 && type_to == 1)
    {
#if __riscv_vector
        if (cpu_support_riscv_zvfh())
#else
        if (cpu_support_riscv_zfh())
#endif
        {
            cast_fp16_to_fp32(bottom_blob, top_blob, opt);
            return 0;
        }
    }
#endif // NCNN_ZFH

    if (type_from == 3 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const signed char* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = (float)ptr[i];
            }
        }

        return 0;
    }

    // TODO more cast type
    return Cast::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
