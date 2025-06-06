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

static void padding_constant_pack4_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, __m128 v)
{
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        __lsx_vst(v, outptr, 0);
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(v, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            __builtin_prefetch(ptr + 32);
            __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(v, outptr, 0);
            outptr += 4;
        }
    }
    // fill top
    for (int y = 0; y < bottom_size; y++)
    {
        __lsx_vst(v, outptr, 0);
        outptr += 4;
    }
}

static void padding_replicate_pack4_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        __m128 _p = (__m128)__lsx_vld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m128)__lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m128 _p = (__m128)__lsx_vld(ptr, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m128)__lsx_vld(ptr, 0);
            __lsx_vst(_p, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        __m128 _p = (__m128)__lsx_vld(ptr0, 0);
        for (int x = 0; x < left; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = (__m128)__lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
    }
}

static void padding_reflect_pack4_lsx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    ptr += top * src.w * 4;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0 + (left - x) * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0 - 8 - x * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr + (left - x) * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __lsx_vst(_p, outptr, 0);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr - 8 - x * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0 + (left - x) * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            __lsx_vst(_p, outptr, 0);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0 - 8 - x * 4, 0);
            __lsx_vst(_p, outptr, 0);
            outptr += 4;
        }
        ptr -= src.w * 4;
    }
}
