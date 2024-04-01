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
int Quantize_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8((float)ptr0[0] * scale);
                    outptr[1] = float2int8((float)ptr0[1] * scale);
                    outptr[2] = float2int8((float)ptr0[2] * scale);
                    outptr[3] = float2int8((float)ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8((float)ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8((float)ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8((float)ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8((float)ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                        const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _vlow = vcvt_f32_f16(vld1_f16(ptr0));
                            float32x4_t _vhigh = vcvt_f32_f16(vld1_f16(ptr1));
                            _vlow = vmulq_f32(_vlow, _scale);
                            _vhigh = vmulq_f32(_vhigh, _scale);
                            int8x8_t _v = float2int8(_vlow, _vhigh);
                            vst1_s8(outptr, _v);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i * 2);
                        const __fp16* ptr1 = bottom_blob.row<const __fp16>(i * 2 + 1);
                        signed char* outptr = top_blob.row<signed char>(i);

                        float32x4_t _scale0 = vld1q_f32((const float*)scale_data + i * 8);
                        float32x4_t _scale1 = vld1q_f32((const float*)scale_data + i * 8 + 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _vlow = vcvt_f32_f16(vld1_f16(ptr0));
                            float32x4_t _vhigh = vcvt_f32_f16(vld1_f16(ptr1));
                            _vlow = vmulq_f32(_vlow, _scale0);
                            _vhigh = vmulq_f32(_vhigh, _scale1);
                            int8x8_t _v = float2int8(_vlow, _vhigh);
                            vst1_s8(outptr, _v);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * scale);
                            outptr1[0] = float2int8((float)ptr0[1] * scale);
                            outptr2[0] = float2int8((float)ptr0[2] * scale);
                            outptr3[0] = float2int8((float)ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * 4);
                        signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                        signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                        signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                        const float s0 = scale_data[i * 4];
                        const float s1 = scale_data[i * 4 + 1];
                        const float s2 = scale_data[i * 4 + 2];
                        const float s3 = scale_data[i * 4 + 3];

                        for (int j = 0; j < w; j++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * s0);
                            outptr1[0] = float2int8((float)ptr0[1] * s1);
                            outptr2[0] = float2int8((float)ptr0[2] * s2);
                            outptr3[0] = float2int8((float)ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int out_elempack = opt.use_packing_layout && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == 8)
            {
                if (scale_data_size == 1)
                {
                    float32x4_t _scale = vdupq_n_f32(scale_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q * 2);
                        const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _vlow = vcvt_f32_f16(vld1_f16(ptr0));
                            float32x4_t _vhigh = vcvt_f32_f16(vld1_f16(ptr1));
                            _vlow = vmulq_f32(_vlow, _scale);
                            _vhigh = vmulq_f32(_vhigh, _scale);
                            int8x8_t _v = float2int8(_vlow, _vhigh);
                            vst1_s8(outptr, _v);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q * 2);
                        const __fp16* ptr1 = bottom_blob.channel(q * 2 + 1);
                        signed char* outptr = top_blob.channel(q);

                        float32x4_t _scale0 = vld1q_f32((const float*)scale_data + q * 8);
                        float32x4_t _scale1 = vld1q_f32((const float*)scale_data + q * 8 + 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _vlow = vcvt_f32_f16(vld1_f16(ptr0));
                            float32x4_t _vhigh = vcvt_f32_f16(vld1_f16(ptr1));
                            _vlow = vmulq_f32(_vlow, _scale0);
                            _vhigh = vmulq_f32(_vhigh, _scale1);
                            int8x8_t _v = float2int8(_vlow, _vhigh);
                            vst1_s8(outptr, _v);

                            ptr0 += 4;
                            ptr1 += 4;
                            outptr += 8;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * scale);
                            outptr1[0] = float2int8((float)ptr0[1] * scale);
                            outptr2[0] = float2int8((float)ptr0[2] * scale);
                            outptr3[0] = float2int8((float)ptr0[3] * scale);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * 4);
                        signed char* outptr1 = top_blob.channel(q * 4 + 1);
                        signed char* outptr2 = top_blob.channel(q * 4 + 2);
                        signed char* outptr3 = top_blob.channel(q * 4 + 3);

                        const float s0 = scale_data[q * 4];
                        const float s1 = scale_data[q * 4 + 1];
                        const float s2 = scale_data[q * 4 + 2];
                        const float s3 = scale_data[q * 4 + 3];

                        for (int i = 0; i < size; i++)
                        {
                            outptr0[0] = float2int8((float)ptr0[0] * s0);
                            outptr1[0] = float2int8((float)ptr0[1] * s1);
                            outptr2[0] = float2int8((float)ptr0[2] * s2);
                            outptr3[0] = float2int8((float)ptr0[3] * s3);

                            ptr0 += 4;
                            outptr0 += 1;
                            outptr1 += 1;
                            outptr2 += 1;
                            outptr3 += 1;
                        }
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const __fp16* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8((float)ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8((float)ptr[i] * scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8((float)*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                *outptr++ = float2int8((float)*ptr++ * scale);
            }
        }
    }

    return 0;
}

int Quantize_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
#if _MSC_VER
                float16x4_t _scale0 = vcvt_f16_f32(vdupq_n_f32(scale_data[0]));
                float16x8_t _scale = vcombine_f16(_scale0, _scale0);
#else
                float16x8_t _scale = vdupq_n_f16((__fp16)scale_data[0]);
#endif

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    float16x8_t _v = vld1q_f16(ptr0);
                    _v = vmulq_f16(_v, _scale);
                    vst1_s8(outptr, float2int8(_v));
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 8;
                    signed char* outptr = (signed char*)top_blob + i * 8;

                    float16x8_t _v = vld1q_f16(ptr0);
                    float16x8_t _scale = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));
                    _v = vmulq_f16(_v, _scale);
                    vst1_s8(outptr, float2int8(_v));
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
#if _MSC_VER
                float16x4_t _scale0 = vcvt_f16_f32(vdupq_n_f32(scale_data[0]));
                float16x8_t _scale = vcombine_f16(_scale0, _scale0);
#else
                float16x8_t _scale = vdupq_n_f16((__fp16)scale_data[0]);
#endif

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _v = vld1q_f16(ptr0);
                        _v = vmulq_f16(_v, _scale);
                        vst1_s8(outptr0, float2int8(_v));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i);

                    float16x8_t _scale = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + i * 8 + 4)));

                    for (int j = 0; j < w; j++)
                    {
                        float16x8_t _v = vld1q_f16(ptr0);
                        _v = vmulq_f16(_v, _scale);
                        vst1_s8(outptr0, float2int8(_v));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)8u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
#if _MSC_VER
                float16x4_t _scale0 = vcvt_f16_f32(vdupq_n_f32(scale_data[0]));
                float16x8_t _scale = vcombine_f16(_scale0, _scale0);
#else
                float16x8_t _scale = vdupq_n_f16((__fp16)scale_data[0]);
#endif

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _v = vld1q_f16(ptr0);
                        _v = vmulq_f16(_v, _scale);
                        vst1_s8(outptr0, float2int8(_v));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q);

                    float16x8_t _scale = vcombine_f16(vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8)), vcvt_f16_f32(vld1q_f32((const float*)scale_data + q * 8 + 4)));

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _v = vld1q_f16(ptr0);
                        _v = vmulq_f16(_v, _scale);
                        vst1_s8(outptr0, float2int8(_v));

                        ptr0 += 8;
                        outptr0 += 8;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * elempack;

            top_blob.create(outw, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const __fp16* ptr0 = (const __fp16*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * (__fp16)scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * (__fp16)scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * (__fp16)scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * (__fp16)scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * elempack;

            top_blob.create(w, outh, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i * 4);
                    signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                    signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                    signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * scale);
                        outptr1[0] = float2int8(ptr0[1] * scale);
                        outptr2[0] = float2int8(ptr0[2] * scale);
                        outptr3[0] = float2int8(ptr0[3] * scale);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
                    signed char* outptr0 = top_blob.row<signed char>(i * 4);
                    signed char* outptr1 = top_blob.row<signed char>(i * 4 + 1);
                    signed char* outptr2 = top_blob.row<signed char>(i * 4 + 2);
                    signed char* outptr3 = top_blob.row<signed char>(i * 4 + 3);

                    const __fp16 s0 = scale_data[i * 4];
                    const __fp16 s1 = scale_data[i * 4 + 1];
                    const __fp16 s2 = scale_data[i * 4 + 2];
                    const __fp16 s3 = scale_data[i * 4 + 3];

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * s0);
                        outptr1[0] = float2int8(ptr0[1] * s1);
                        outptr2[0] = float2int8(ptr0[2] * s2);
                        outptr3[0] = float2int8(ptr0[3] * s3);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int outc = channels * elempack;

            top_blob.create(w, h, outc, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const __fp16 scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q * 4);
                    signed char* outptr1 = top_blob.channel(q * 4 + 1);
                    signed char* outptr2 = top_blob.channel(q * 4 + 2);
                    signed char* outptr3 = top_blob.channel(q * 4 + 3);

                    for (int i = 0; i < size; i++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * scale);
                        outptr1[0] = float2int8(ptr0[1] * scale);
                        outptr2[0] = float2int8(ptr0[2] * scale);
                        outptr3[0] = float2int8(ptr0[3] * scale);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr0 = bottom_blob.channel(q);
                    signed char* outptr0 = top_blob.channel(q * 4);
                    signed char* outptr1 = top_blob.channel(q * 4 + 1);
                    signed char* outptr2 = top_blob.channel(q * 4 + 2);
                    signed char* outptr3 = top_blob.channel(q * 4 + 3);

                    const __fp16 s0 = scale_data[q * 4];
                    const __fp16 s1 = scale_data[q * 4 + 1];
                    const __fp16 s2 = scale_data[q * 4 + 2];
                    const __fp16 s3 = scale_data[q * 4 + 3];

                    for (int i = 0; i < size; i++)
                    {
                        outptr0[0] = float2int8(ptr0[0] * s0);
                        outptr1[0] = float2int8(ptr0[1] * s1);
                        outptr2[0] = float2int8(ptr0[2] * s2);
                        outptr3[0] = float2int8(ptr0[3] * s3);

                        ptr0 += 4;
                        outptr0 += 1;
                        outptr1 += 1;
                        outptr2 += 1;
                        outptr3 += 1;
                    }
                }
            }
        }

        return 0;
    }

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const __fp16* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const __fp16 scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                outptr[i] = float2int8(ptr[i] * (__fp16)scale_data[i]);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const __fp16* ptr0 = bottom_blob.row<const __fp16>(i);
            signed char* outptr0 = top_blob.row<signed char>(i);

            const __fp16 scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

            for (int j = 0; j < w; j++)
            {
                *outptr0++ = float2int8(*ptr0++ * scale);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const __fp16 scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

            for (int i = 0; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
