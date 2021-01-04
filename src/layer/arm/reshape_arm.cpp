// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "reshape_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

Reshape_arm::Reshape_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Reshape_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    int elempack = bottom_blob.elempack;

    if (permute == 1)
    {
        // TODO implement permute on-the-fly
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        Mat top_blob_unpacked;
        int ret = Reshape::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
        if (ret != 0)
            return ret;

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            // resolve dst_elempack
            int dims = top_blob_unpacked.dims;
            if (dims == 1) out_elempack = top_blob_unpacked.w % 4 == 0 ? 4 : 1;
            if (dims == 2) out_elempack = top_blob_unpacked.h % 4 == 0 ? 4 : 1;
            if (dims == 3) out_elempack = top_blob_unpacked.c % 4 == 0 ? 4 : 1;
        }
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

        return 0;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        int out_elempack = opt.use_packing_layout && _h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = _w;
            top_blob.h = _h;
            top_blob.cstep = _w * _h;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int outw = top_blob.w;
        int outh = top_blob.h;

        // assert out_elempack == 4

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < outh; i++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
            const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
            float* outptr = (float*)top_blob.row(i);

            int j = 0;
#if __ARM_NEON
            for (; j + 3 < outw; j += 4)
            {
                float32x4x4_t _v4;
                _v4.val[0] = vld1q_f32(ptr0);
                _v4.val[1] = vld1q_f32(ptr1);
                _v4.val[2] = vld1q_f32(ptr2);
                _v4.val[3] = vld1q_f32(ptr3);

                vst4q_f32(outptr, _v4);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
#endif
            for (; j < outw; j++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }
    }

    if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        int out_elempack = opt.use_packing_layout && _c % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3 && bottom_blob.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h;

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vld1q_f32(ptr);
                    vst1q_f32(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Reshape_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

    if (permute == 1)
    {
        // TODO implement permute on-the-fly
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        Mat bottom_blob_unpacked_fp32;
        cast_bfloat16_to_float32(bottom_blob_unpacked, bottom_blob_unpacked_fp32, opt_pack);

        Mat top_blob_unpacked_fp32;
        int ret = Reshape::forward(bottom_blob_unpacked_fp32, top_blob_unpacked_fp32, opt_pack);
        if (ret != 0)
            return ret;

        Mat top_blob_unpacked;
        cast_float32_to_bfloat16(top_blob_unpacked_fp32, top_blob_unpacked, opt_pack);

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            // resolve dst_elempack
            int dims = top_blob_unpacked.dims;
            if (dims == 1) out_elempack = opt.use_fp16_arithmetic && top_blob_unpacked.w % 8 == 0 ? 8 : top_blob_unpacked.w % 4 == 0 ? 4 : 1;
            if (dims == 2) out_elempack = opt.use_fp16_arithmetic && top_blob_unpacked.h % 8 == 0 ? 8 : top_blob_unpacked.h % 4 == 0 ? 4 : 1;
            if (dims == 3) out_elempack = opt.use_fp16_arithmetic && top_blob_unpacked.c % 8 == 0 ? 8 : top_blob_unpacked.c % 4 == 0 ? 4 : 1;
        }
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);

        return 0;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            out_elempack = opt.use_fp16_arithmetic && _h % 8 == 0 ? 8 : _h % 4 == 0 ? 4 : 1;
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h == _h && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = _w;
            top_blob.h = _h;
            top_blob.cstep = _w * _h;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(_w, _h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int outw = top_blob.w;
        int outh = top_blob.h;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const __fp16* ptr0 = (const __fp16*)bottom_blob_flattened + outw * i * 8;
                const __fp16* ptr1 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 1);
                const __fp16* ptr2 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 2);
                const __fp16* ptr3 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 3);
                const __fp16* ptr4 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 4);
                const __fp16* ptr5 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 5);
                const __fp16* ptr6 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 6);
                const __fp16* ptr7 = (const __fp16*)bottom_blob_flattened + outw * (i * 8 + 7);
                __fp16* outptr = top_blob.row<__fp16>(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    float16x8_t _p01 = vcombine_f16(vld1_f16(ptr0), vld1_f16(ptr1));
                    float16x8_t _p23 = vcombine_f16(vld1_f16(ptr2), vld1_f16(ptr3));
                    float16x8_t _p45 = vcombine_f16(vld1_f16(ptr4), vld1_f16(ptr5));
                    float16x8_t _p67 = vcombine_f16(vld1_f16(ptr6), vld1_f16(ptr7));

                    float16x8x2_t _p0415 = vzipq_f16(_p01, _p45);
                    float16x8x2_t _p2637 = vzipq_f16(_p23, _p67);

                    float16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_f16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 3);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < outw; j += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }
    }

    if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (_h == 0)
            _h = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (_c == 0)
            _c = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            out_elempack = opt.use_fp16_arithmetic && _c % 8 == 0 ? 8 : _c % 4 == 0 ? 4 : 1;
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3 && bottom_blob.c == _c && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = _w;
            top_blob.h = _h;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(_w, _h, _c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const __fp16* ptr0 = (const __fp16*)bottom_blob_flattened + size * q * 8;
                const __fp16* ptr1 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 1);
                const __fp16* ptr2 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 2);
                const __fp16* ptr3 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 3);
                const __fp16* ptr4 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 4);
                const __fp16* ptr5 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 5);
                const __fp16* ptr6 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 6);
                const __fp16* ptr7 = (const __fp16*)bottom_blob_flattened + size * (q * 8 + 7);
                __fp16* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p01 = vcombine_f16(vld1_f16(ptr0), vld1_f16(ptr1));
                    float16x8_t _p23 = vcombine_f16(vld1_f16(ptr2), vld1_f16(ptr3));
                    float16x8_t _p45 = vcombine_f16(vld1_f16(ptr4), vld1_f16(ptr5));
                    float16x8_t _p67 = vcombine_f16(vld1_f16(ptr6), vld1_f16(ptr7));

                    float16x8x2_t _p0415 = vzipq_f16(_p01, _p45);
                    float16x8x2_t _p2637 = vzipq_f16(_p23, _p67);

                    float16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_f16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 3);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4_t _v = vld1_u16(ptr);
                    vst1_u16(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
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
