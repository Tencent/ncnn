// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "dequantize_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

Dequantize_loongarch::Dequantize_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __loongarch_sx
    __m128 _scale = (__m128)__lsx_vreplfr2vr_s(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = (__m128)__lsx_vld((const float*)scale_data, 0);
        }
    }
#endif // __loongarch_sx

    if (bias_data_size == 0)
    {
        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(intptr + 16);
            __m128 _v = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
            _v = __lsx_vfmul_s(_v, _scale);
            __lsx_vst(_v, ptr, 0);
            intptr += 4;
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __loongarch_sx
        __m128 _bias = (__m128)__lsx_vreplfr2vr_s(bias);
        if (bias_data_size > 1)
        {
            if (elempack == 4)
            {
                _bias = (__m128)__lsx_vld((const float*)bias_data, 0);
            }
        }
#endif // __loongarch_sx

        int i = 0;
#if __loongarch_sx
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(intptr + 16);
            __m128 _v = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
            _v = __lsx_vfmadd_s(_scale, _v, _bias);
            __lsx_vst(_v, ptr, 0);
            intptr += 4;
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
}

int Dequantize_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

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

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

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
