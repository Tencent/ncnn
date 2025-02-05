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

#include "quantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

Quantize_arm::Quantize_arm()
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

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

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

    int i = 0;
#if __ARM_NEON
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _v0 = vld1q_f32(ptr);
        float32x4_t _v1 = vld1q_f32(ptr + 4);
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = vld1q_f32(ptr);
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

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

#if NCNN_BF16
static void quantize_bf16s(const unsigned short* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_bf16s %d   %d %d", scale_data_size, elemcount, elempack);

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

    int i = 0;
#if __ARM_NEON
    for (; i + 7 < size; i += 8)
    {
        uint16x8_t _v01 = vld1q_u16(ptr);
        float32x4_t _v0 = bfloat2float(vget_low_u16(_v01));
        float32x4_t _v1 = bfloat2float(vget_high_u16(_v01));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = bfloat2float(vld1_u16(ptr));
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

            const unsigned short* ptr = (const unsigned short*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize_bf16s(ptr, s8ptr, scale_data, size, 1);
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
            const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

            quantize_bf16s(ptr, s8ptr, scale_data_i, w, elempack);
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
            const unsigned short* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

            quantize_bf16s(ptr, s8ptr, scale_data_q, w * h, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
