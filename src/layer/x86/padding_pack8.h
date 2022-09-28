// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void padding_constant_pack8_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right, __m256 v)
{
    const float* ptr = src;
    float* outptr = dst;
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
        _mm256_store_ps(outptr, v);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            _mm256_store_ps(outptr, v);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _mm256_store_ps(outptr, _mm256_load_ps(ptr));
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_store_ps(outptr, v);
            outptr += 8;
        }
    }
    // fill top
    for (int y = 0; y < bottom_size; y++)
    {
        _mm256_store_ps(outptr, v);
        outptr += 8;
    }
}

static void padding_replicate_pack8_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        __m256 _p = _mm256_load_ps(ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_load_ps(ptr0);
            _mm256_store_ps(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        __m256 _p = _mm256_load_ps(ptr);
        for (int x = 0; x < left; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_load_ps(ptr);
            _mm256_store_ps(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        __m256 _p = _mm256_load_ps(ptr0);
        for (int x = 0; x < left; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = _mm256_load_ps(ptr0);
            _mm256_store_ps(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_avx(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0 + (left - x) * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0);
            _mm256_store_ps(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0 - 16 - x * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __m256 _p = _mm256_load_ps(ptr + (left - x) * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = _mm256_load_ps(ptr);
            _mm256_store_ps(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = _mm256_load_ps(ptr - 16 - x * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0 + (left - x) * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0);
            _mm256_store_ps(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            __m256 _p = _mm256_load_ps(ptr0 - 16 - x * 8);
            _mm256_store_ps(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}
