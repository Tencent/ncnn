// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flatten_powerpc.h"

#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Flatten_powerpc::Flatten_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

int Flatten_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits != 32)
    {
        // int8 / bf16 storage are not vectorized in this round
        // the generic implementation is not packing aware, so unpack first
        if (bottom_blob.elempack != 1)
        {
            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
            if (bottom_blob_unpacked.empty())
                return -100;

            return Flatten::forward(bottom_blob_unpacked, top_blob, opt);
        }

        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h * d;

    int total = size * channels * elempack;

    int out_elempack = 1;
#if __ALTIVEC__
    if (opt.use_packing_layout)
    {
        out_elempack = total % 4 == 0 ? 4 : 1;
    }
#endif // __ALTIVEC__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 4
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = bottom_blob.cstep / out_elempack;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
#if __ALTIVEC__
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 4;
                float* outptr1 = (float*)top_blob + w * (i * 4 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 4 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 4 + 3);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(ptr);
                    v4f32 _r1 = __ncnn_altivec_load(ptr + 4);
                    v4f32 _r2 = __ncnn_altivec_load(ptr + 4 * 2);
                    v4f32 _r3 = __ncnn_altivec_load(ptr + 4 * 3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr0, _r0);
                    __ncnn_altivec_store(outptr1, _r1);
                    __ncnn_altivec_store(outptr2, _r2);
                    __ncnn_altivec_store(outptr3, _r3);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __ALTIVEC__
    }

    if (dims == 3 || dims == 4)
    {
#if __ALTIVEC__
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 4;
                float* outptr1 = (float*)top_blob + size * (q * 4 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 4 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 4 + 3);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(ptr);
                    v4f32 _r1 = __ncnn_altivec_load(ptr + 4);
                    v4f32 _r2 = __ncnn_altivec_load(ptr + 4 * 2);
                    v4f32 _r3 = __ncnn_altivec_load(ptr + 4 * 3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr0, _r0);
                    __ncnn_altivec_store(outptr1, _r1);
                    __ncnn_altivec_store(outptr2, _r2);
                    __ncnn_altivec_store(outptr3, _r3);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __ALTIVEC__

        if (elempack == 1) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    v4f32 _p = __ncnn_altivec_load(ptr);
                    __ncnn_altivec_store(outptr, _p);
                    ptr += 4;
                    outptr += 4;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
