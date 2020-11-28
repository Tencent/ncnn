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

#include "concat_arm.h"

namespace ncnn {

Concat_arm::Concat_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Concat_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);

    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_w % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const float* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_h % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        float* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const float* r0 = bottom_blob.row(i);

                    float* outptr0 = outptr;
                    float* outptr1 = outptr + w;
                    float* outptr2 = outptr + w * 2;
                    float* outptr3 = outptr + w * 3;

                    for (int j = 0; j < w; j++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    outptr += w * 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = w * bottom_blob.h;

                const float* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* outptr = top_blob.row(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const float* ptr = bottom_blob.row(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = opt.use_packing_layout && top_channels % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);
                    float* outptr1 = top_blob_unpacked.channel(p + 1);
                    float* outptr2 = top_blob_unpacked.channel(p + 2);
                    float* outptr3 = top_blob_unpacked.channel(p + 3);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    p += 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = bottom_blob.total();

                const float* ptr = bottom_blob;
                float* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const float* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    const float* ptr = bottom_blob.channel(q).row(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w * elempack;
                }
            }
        }
    }

    return 0;
}

int Concat_arm::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            out_elempack = opt.use_fp16_arithmetic && top_w % 8 == 0 ? 8 : top_w % 4 == 0 ? 4 : 1;
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned short* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const unsigned short* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            out_elempack = opt.use_fp16_arithmetic && top_h % 8 == 0 ? 8 : top_h % 4 == 0 ? 4 : 1;
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        unsigned short* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w * 4;

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr1[0] = r0[4];
                        outptr1[1] = r0[5];
                        outptr1[2] = r0[6];
                        outptr1[3] = r0[7];

                        outptr0 += 4;
                        outptr1 += 4;
                        r0 += 8;
                    }

                    outptr += w * 8;
                }
            }
            if (bottom_blob.elempack == 8 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w;
                    unsigned short* outptr2 = outptr + w * 2;
                    unsigned short* outptr3 = outptr + w * 3;
                    unsigned short* outptr4 = outptr + w * 4;
                    unsigned short* outptr5 = outptr + w * 5;
                    unsigned short* outptr6 = outptr + w * 6;
                    unsigned short* outptr7 = outptr + w * 7;

                    for (int j = 0; j < w; j++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];
                        *outptr4++ = r0[4];
                        *outptr5++ = r0[5];
                        *outptr6++ = r0[6];
                        *outptr7++ = r0[7];

                        r0 += 8;
                    }

                    outptr += w * 8;
                }
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;
                    unsigned short* outptr1 = outptr + w;
                    unsigned short* outptr2 = outptr + w * 2;
                    unsigned short* outptr3 = outptr + w * 3;

                    for (int j = 0; j < w; j++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    outptr += w * 4;
                }
            }
            if (bottom_blob.elempack == elempack) // 1-1 4-4 8-8
            {
                int size = w * bottom_blob.h;

                const unsigned short* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned short* ptr = bottom_blob.row<unsigned short>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
            out_elempack = opt.use_fp16_arithmetic && top_channels % 8 == 0 ? 8 : top_channels % 4 == 0 ? 4 : 1;
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (bottom_blob.elempack == 8 && elempack == 4)
            {
                int size = bottom_blob.w * bottom_blob.h;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);

                    for (int i = 0; i < size; i++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr1[0] = r0[4];
                        outptr1[1] = r0[5];
                        outptr1[2] = r0[6];
                        outptr1[3] = r0[7];

                        outptr0 += 4;
                        outptr1 += 4;
                        r0 += 8;
                    }

                    p += 2;
                }
            }
            if (bottom_blob.elempack == 8 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);
                    unsigned short* outptr2 = top_blob_unpacked.channel(p + 2);
                    unsigned short* outptr3 = top_blob_unpacked.channel(p + 3);
                    unsigned short* outptr4 = top_blob_unpacked.channel(p + 4);
                    unsigned short* outptr5 = top_blob_unpacked.channel(p + 5);
                    unsigned short* outptr6 = top_blob_unpacked.channel(p + 6);
                    unsigned short* outptr7 = top_blob_unpacked.channel(p + 7);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];
                        *outptr4++ = r0[4];
                        *outptr5++ = r0[5];
                        *outptr6++ = r0[6];
                        *outptr7++ = r0[7];

                        r0 += 8;
                    }

                    p += 8;
                }
            }
#endif
            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);
                    unsigned short* outptr1 = top_blob_unpacked.channel(p + 1);
                    unsigned short* outptr2 = top_blob_unpacked.channel(p + 2);
                    unsigned short* outptr3 = top_blob_unpacked.channel(p + 3);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }

                    p += 4;
                }
            }
            if (bottom_blob.elempack == elempack) // 1-1 4-4 8-8
            {
                int size = bottom_blob.total();

                const unsigned short* ptr = bottom_blob;
                unsigned short* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const unsigned short* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    const unsigned short* ptr = bottom_blob.channel(q).row<const unsigned short>(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
