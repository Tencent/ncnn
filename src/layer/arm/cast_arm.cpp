// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cast_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#if __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

#include "cast_bf16.h"
#include "cast_fp16.h"

Cast_arm::Cast_arm()
{
    support_packing = true;
#if __ARM_FEATURE_SVE
    support_any_packing = true;
#endif // __ARM_FEATURE_SVE
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif

    support_bf16_storage = true;
}

int Cast_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
    int batch = bottom_blob.n;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
#if !__ARM_FEATURE_SVE
        if (type_from == 3)
        {
            return Cast::forward(bottom_blob, top_blob, opt);
        }
#endif // !__ARM_FEATURE_SVE

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
        top_blob.create(w, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(w, h, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, elempack, batch, opt.blob_allocator);
    else if (dims == 4)
        top_blob.create(w, h, d, channels, out_elemsize, elempack, batch, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int size = w * h * d * elempack;

    if (type_from == 1 && type_to == 2)
    {
        cast_fp32_to_fp16_neon(bottom_blob, top_blob, opt);
    }

    if (type_from == 2 && type_to == 1)
    {
        cast_fp16_to_fp32_neon(bottom_blob, top_blob, opt);
    }

    if (type_from == 3 && type_to == 1)
    {
        const int total_bc = batch * channels;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int bc = 0; bc < total_bc; bc++)
        {
            int b = bc / channels;
            int q = bc % channels;
            const signed char* ptr = bottom_blob.batch(b).channel(q);
            float* outptr = top_blob.batch(b).channel(q);

#if __ARM_FEATURE_SVE
            const int packn = svcntw();
            for (int i = 0; i < size; i += packn)
            {
                const svbool_t _pg = svwhilelt_b32((unsigned int)i, (unsigned int)size);
                svint32_t _p = svld1sb_s32(_pg, ptr + i);
                svst1_f32(_pg, outptr + i, svcvt_f32_s32_x(_pg, _p));
            }
#else
            for (int i = 0; i < size; i++)
            {
                outptr[i] = (float)ptr[i];
            }
#endif // __ARM_FEATURE_SVE
        }
    }

    if (type_from == 1 && type_to == 4)
    {
        cast_fp32_to_bf16_neon(bottom_blob, top_blob, opt);
    }

    if (type_from == 4 && type_to == 1)
    {
        cast_bf16_to_fp32_neon(bottom_blob, top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
