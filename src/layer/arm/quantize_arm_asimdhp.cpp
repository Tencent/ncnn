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

#include "quantize_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void quantize_fp16s(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_fp16s %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
    float32x4_t _scale = vdupq_n_f32(scale);
    if (scale_data_size > 1)
    {
        if (elempack == 4)
        {
            _scale = vld1q_f32((const float*)scale_data);
        }
    }

    int i = 0;
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _v01 = vld1q_f16(ptr);
        float32x4_t _v0 = vcvt_f32_f16(vget_low_f16(_v01));
        float32x4_t _v1 = vcvt_f32_f16(vget_high_f16(_v01));
        _v0 = vmulq_f32(_v0, _scale);
        _v1 = vmulq_f32(_v1, _scale);
        vst1_s8(s8ptr, float2int8(_v0, _v1));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _v = vcvt_f32_f16(vld1_f16(ptr));
        _v = vmulq_f32(_v, _scale);
        int8x8_t v = float2int8(_v, _v);
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
    for (; i < size; i++)
    {
        float v = (float)(*ptr) * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

            const __fp16* ptr = (const __fp16*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize_fp16s(ptr, s8ptr, scale_data, size, 1);
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
            const __fp16* ptr = bottom_blob.row<const __fp16>(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

            quantize_fp16s(ptr, s8ptr, scale_data_i, w, elempack);
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
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

            quantize_fp16s(ptr, s8ptr, scale_data_q, w * h, elempack);
        }
    }

    return 0;
}

static void quantize_fp16sa(const __fp16* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize_fp16sa %d   %d %d", scale_data_size, elemcount, elempack);

    __fp16 scale = (__fp16)scale_data[0];
    float16x4_t _scale0 = vdup_n_f16(scale);
    float16x4_t _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        if (elempack == 8)
        {
            _scale0 = vcvt_f16_f32(vld1q_f32((const float*)scale_data));
            _scale1 = vcvt_f16_f32(vld1q_f32((const float*)scale_data + 4));
        }
        if (elempack == 4)
        {
            _scale0 = vcvt_f16_f32(vld1q_f32((const float*)scale_data));
            _scale1 = _scale0;
        }
    }
    float16x8_t _scale = vcombine_f16(_scale0, _scale1);

    int i = 0;
    for (; i + 7 < size; i += 8)
    {
        float16x8_t _v = vld1q_f16(ptr);
        _v = vmulq_f16(_v, _scale);
        vst1_s8(s8ptr, float2int8(_v));
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float16x4_t _v = vld1_f16(ptr);
        _v = vmul_f16(_v, _scale0);
        int8x8_t v = float2int8(vcombine_f16(_v, _v));
        s8ptr[0] = vget_lane_s8(v, 0);
        s8ptr[1] = vget_lane_s8(v, 1);
        s8ptr[2] = vget_lane_s8(v, 2);
        s8ptr[3] = vget_lane_s8(v, 3);
        ptr += 4;
        s8ptr += 4;
    }
    for (; i < size; i++)
    {
        __fp16 v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

int Quantize_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

            const __fp16* ptr = (const __fp16*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize_fp16sa(ptr, s8ptr, scale_data, size, 1);
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
            const __fp16* ptr = bottom_blob.row<const __fp16>(i);
            signed char* s8ptr = top_blob.row<signed char>(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

            quantize_fp16sa(ptr, s8ptr, scale_data_i, w, elempack);
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
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* s8ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

            quantize_fp16sa(ptr, s8ptr, scale_data_q, w * h, elempack);
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
