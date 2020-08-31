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

#include "slice_arm.h"

namespace ncnn {

Slice_arm::Slice_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Slice_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;

    if (dims == 1) // axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = (const float*)bottom_blob + q;
            float* outptr = top_blob;
            memcpy(outptr, ptr, top_blob.w * top_blob.elemsize);

            q += slice;
        }
    }

    if (dims == 2 && axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
        }

        const float* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const float* r0 = ptr;
                    const float* r1 = ptr + w;
                    const float* r2 = ptr + w * 2;
                    const float* r3 = ptr + w * 3;

                    float* outptr0 = top_blob.row(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    ptr += w * 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = w * top_blob.h;

                float* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const float* ptr = bottom_blob.row(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                float* outptr = top_blob.row(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if (dims == 3 && axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (channels - q) / (top_blobs.size() - i);
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);
                    const float* r2 = bottom_blob_unpacked.channel(p + 2);
                    const float* r3 = bottom_blob_unpacked.channel(p + 3);

                    float* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    p += 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = top_blob.total();

                const float* ptr = bottom_blob_unpacked.channel(p);
                float* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                p += top_blob.c;
            }
        }
    }

    if (dims == 3 && axis == 1)
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h;

                float* outptr = top_blob.channel(p);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack;
            }
        }
    }

    if (dims == 3 && axis == 2)
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (int j = 0; j < h; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    float* outptr = top_blob.channel(p).row(j);
                    memcpy(outptr, ptr, top_blob.w * elemsize);

                    ptr += top_blob.w * elempack;
                }
            }
        }
    }

    return 0;
}

int Slice_arm::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;

    if (dims == 1) // axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const unsigned short* ptr = (const unsigned short*)bottom_blob + q;
            unsigned short* outptr = top_blob;
            memcpy(outptr, ptr, top_blob.w * top_blob.elemsize);

            q += slice;
        }
    }

    if (dims == 2 && axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
                out_elempack = opt.use_fp16_arithmetic && slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
            }
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
        }

        const unsigned short* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (out_elempack == 4 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w * 4;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr0[4] = r1[0];
                        outptr0[5] = r1[1];
                        outptr0[6] = r1[2];
                        outptr0[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr0 += 8;
                    }

                    ptr += w * 8;
                }
            }
            if (out_elempack == 1 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w;
                    const unsigned short* r2 = ptr + w * 2;
                    const unsigned short* r3 = ptr + w * 3;
                    const unsigned short* r4 = ptr + w * 4;
                    const unsigned short* r5 = ptr + w * 5;
                    const unsigned short* r6 = ptr + w * 6;
                    const unsigned short* r7 = ptr + w * 7;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;
                        outptr0[4] = *r4++;
                        outptr0[5] = *r5++;
                        outptr0[6] = *r6++;
                        outptr0[7] = *r7++;

                        outptr0 += 8;
                    }

                    ptr += w * 8;
                }
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w;
                    const unsigned short* r2 = ptr + w * 2;
                    const unsigned short* r3 = ptr + w * 3;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    ptr += w * 4;
                }
            }
            if (out_elempack == top_blob.elempack) // 1-1 4-4 8-8
            {
                int size = w * top_blob.h;

                unsigned short* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const unsigned short* ptr = bottom_blob.row<const unsigned short>(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                unsigned short* outptr = top_blob.row<unsigned short>(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if (dims == 3 && axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (channels - q) / (top_blobs.size() - i);
            }

            int out_elempack = 1;
            if (opt.use_packing_layout)
            {
                out_elempack = opt.use_fp16_arithmetic && slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
            }
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (out_elempack == 4 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr0[4] = r1[0];
                        outptr0[5] = r1[1];
                        outptr0[6] = r1[2];
                        outptr0[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr0 += 8;
                    }

                    p += 2;
                }
            }
            if (out_elempack == 1 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);
                    const unsigned short* r2 = bottom_blob_unpacked.channel(p + 2);
                    const unsigned short* r3 = bottom_blob_unpacked.channel(p + 3);
                    const unsigned short* r4 = bottom_blob_unpacked.channel(p + 4);
                    const unsigned short* r5 = bottom_blob_unpacked.channel(p + 5);
                    const unsigned short* r6 = bottom_blob_unpacked.channel(p + 6);
                    const unsigned short* r7 = bottom_blob_unpacked.channel(p + 7);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;
                        outptr0[4] = *r4++;
                        outptr0[5] = *r5++;
                        outptr0[6] = *r6++;
                        outptr0[7] = *r7++;

                        outptr0 += 8;
                    }

                    p += 8;
                }
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);
                    const unsigned short* r2 = bottom_blob_unpacked.channel(p + 2);
                    const unsigned short* r3 = bottom_blob_unpacked.channel(p + 3);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    p += 4;
                }
            }
            if (out_elempack == top_blob.elempack) // 1-1 4-4 8-8
            {
                int size = top_blob.total();

                const unsigned short* ptr = bottom_blob_unpacked.channel(p);
                unsigned short* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                p += top_blob.c;
            }
        }
    }

    if (dims == 3 && axis == 1)
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (h - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const unsigned short* ptr = bottom_blob.channel(p);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h;

                unsigned short* outptr = top_blob.channel(p);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack;
            }
        }
    }

    if (dims == 3 && axis == 2)
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice = slices_ptr[i];
            if (slice == -233)
            {
                slice = (w - q) / (top_blobs.size() - i);
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const unsigned short* ptr = bottom_blob.channel(p);

            for (int j = 0; j < h; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    unsigned short* outptr = top_blob.channel(p).row<unsigned short>(j);
                    memcpy(outptr, ptr, top_blob.w * elemsize);

                    ptr += top_blob.w * elempack;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
