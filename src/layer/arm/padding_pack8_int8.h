// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void padding_constant_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int8x8_t _v)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            vst1_s8(outptr, _v);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _v);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr);
            vst1_s8(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _v);
            outptr += 8;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            vst1_s8(outptr, _v);
            outptr += 8;
        }
    }
}

static void padding_replicate_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const signed char* ptr0 = ptr;
        int8x8_t _p = vld1_s8(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        int8x8_t _p = vld1_s8(ptr);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr);
            vst1_s8(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const signed char* ptr0 = ptr;
        int8x8_t _p = vld1_s8(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const signed char* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr);
            vst1_s8(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const signed char* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}
