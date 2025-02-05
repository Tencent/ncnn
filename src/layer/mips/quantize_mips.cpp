// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "quantize_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Quantize_mips::Quantize_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __mips_msa
    v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = (v4f32)__msa_ld_w((const float*)scale_data, 0);
        }
    }
#endif // __mips_msa

    int i = 0;
#if __mips_msa
    for (; i + 7 < size; i += 8)
    {
        __builtin_prefetch(ptr + 32);
        v4f32 _v0 = (v4f32)__msa_ld_w(ptr, 0);
        v4f32 _v1 = (v4f32)__msa_ld_w(ptr + 4, 0);
        _v0 = __msa_fmul_w(_v0, _scale);
        _v1 = __msa_fmul_w(_v1, _scale);
        *((int64_t*)s8ptr) = float2int8(_v0, _v1);
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        v4f32 _v = (v4f32)__msa_ld_w(ptr, 0);
        _v = __msa_fmul_w(_v, _scale);
        v16i8 v = float2int8(_v, _v);
        s8ptr[0] = v[0];
        s8ptr[1] = v[1];
        s8ptr[2] = v[2];
        s8ptr[3] = v[3];
        ptr += 4;
        s8ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 1u;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_blob.row(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

            quantize(ptr, s8ptr, scale_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

            quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
