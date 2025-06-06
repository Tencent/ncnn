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

static void padding_constant_pack8_int8_sse(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int64_t _v)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            *outptr++ = _v;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = _v;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = _v;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom; y++)
    {
        for (int x = 0; x < dst.w; x++)
        {
            *outptr++ = _v;
        }
    }
}

static void padding_replicate_pack8_int8_sse(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr0;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-1];
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr[-1];
        }
    }
    // fill bottom
    ptr -= src.w;
    for (int y = 0; y < bottom; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = *ptr0;
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-1];
        }
    }
}

static void padding_reflect_pack8_int8_sse(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const int64_t* ptr = src;
    int64_t* outptr = dst;

    // fill top
    ptr += top * src.w;
    for (int y = 0; y < top; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr0[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-2 - x];
        }
        ptr -= src.w;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr[-2 - x];
        }
    }
    // fill bottom
    ptr -= 2 * src.w;
    for (int y = 0; y < bottom; y++)
    {
        const int64_t* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            *outptr++ = ptr0[left - x];
        }
        for (int x = 0; x < src.w; x++)
        {
            *outptr++ = *ptr0++;
        }
        for (int x = 0; x < right; x++)
        {
            *outptr++ = ptr0[-2 - x];
        }
        ptr -= src.w;
    }
}
