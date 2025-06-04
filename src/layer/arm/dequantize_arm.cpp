// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dequantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

Dequantize_arm::Dequantize_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }
#endif // __ARM_NEON

    if (bias_data_size == 0)
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
            _v = vmulq_f32(_v, _scale);
            vst1q_f32(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        if (bias_data_size > 1)
        {
            if (elempack == 4)
            {
                _bias = vld1q_f32((const float*)bias_data);
            }
        }
#endif // __ARM_NEON

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
#if __aarch64__
            _v = vfmaq_f32(_bias, _v, _scale);
#else
            _v = vmlaq_f32(_bias, _v, _scale);
#endif
            vst1q_f32(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
}

int Dequantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blob, top_blob, opt);
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

#if NCNN_BF16
static void dequantize_bf16s(const int* intptr, unsigned short* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("dequantize_bf16s %d %d   %d %d", scale_data_size, bias_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __ARM_NEON
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }
#endif // __ARM_NEON

    if (bias_data_size == 0)
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
            _v = vmulq_f32(_v, _scale);
            vst1_u16(ptr, float2bfloat(_v));
            intptr += 4;
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(*intptr * scale);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        if (bias_data_size > 1)
        {
            if (elempack == 4)
            {
                _bias = vld1q_f32((const float*)bias_data);
            }
        }
#endif // __ARM_NEON

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
#if __aarch64__
            _v = vfmaq_f32(_bias, _v, _scale);
#else
            _v = vmlaq_f32(_bias, _v, _scale);
#endif
            vst1_u16(ptr, float2bfloat(_v));
            intptr += 4;
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(*intptr * scale + bias);
            intptr++;
            ptr++;
        }
    }
}

int Dequantize_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 2u;

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

            const int* intptr = (const int*)bottom_blob + i * elempack;
            unsigned short* ptr = (unsigned short*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize_bf16s(intptr, ptr, scale_data, bias_data, size, 1);
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
            const int* intptr = bottom_blob.row<const int>(i);
            unsigned short* ptr = top_blob.row<unsigned short>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize_bf16s(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
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
            const int* intptr = bottom_blob.channel(q);
            unsigned short* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize_bf16s(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
