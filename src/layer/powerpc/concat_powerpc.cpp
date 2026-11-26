// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat_powerpc.h"

#include <algorithm>
#include <string.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Concat_powerpc::Concat_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized contiguous copy of n floats
static void copy_vec(float* dst, const float* src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _v = __ncnn_altivec_load(src + i);
        __ncnn_altivec_store(dst + i, _v);
    }
    for (; i < n; i++)
    {
        dst[i] = src[i];
    }
}

// interleave 4 pack1 channels into one pack4 channel, vectorized 4x4 transpose
static void pack4_to_pack1_vec(const float* r0, float* outptr0, float* outptr1, float* outptr2, float* outptr3, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _r0 = __ncnn_altivec_load(r0);
        v4f32 _r1 = __ncnn_altivec_load(r0 + 4);
        v4f32 _r2 = __ncnn_altivec_load(r0 + 8);
        v4f32 _r3 = __ncnn_altivec_load(r0 + 12);

        transpose4x4_ps(_r0, _r1, _r2, _r3);

        __ncnn_altivec_store(outptr0, _r0);
        __ncnn_altivec_store(outptr1, _r1);
        __ncnn_altivec_store(outptr2, _r2);
        __ncnn_altivec_store(outptr3, _r3);

        r0 += 16;
        outptr0 += 4;
        outptr1 += 4;
        outptr2 += 4;
        outptr3 += 4;
    }
    for (; i < n; i++)
    {
        *outptr0++ = r0[0];
        *outptr1++ = r0[1];
        *outptr2++ = r0[2];
        *outptr3++ = r0[3];

        r0 += 4;
    }
}
#endif // __ALTIVEC__

int Concat_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();
    if (elembits != 32)
    {
        // int8 / fp16 / bf16 -> base implementation, unpacked
        std::vector<Mat> bottom_blobs_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            if (bottom_blobs[b].elempack != 1)
            {
                Mat bottom_blob_unpacked;
                convert_packing(bottom_blobs[b], bottom_blob_unpacked, 1, opt);
                if (bottom_blob_unpacked.empty())
                    return -100;
                bottom_blobs_unpacked.push_back(bottom_blob_unpacked);
            }
            else
            {
                bottom_blobs_unpacked.push_back(bottom_blobs[b]);
            }
        }

        return Concat::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

#if __ALTIVEC__
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
            int n = bottom_blob.w * bottom_blob.elempack;
            copy_vec(outptr, ptr, n);

            outptr += n;
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

                    pack4_to_pack1_vec(r0, outptr0, outptr1, outptr2, outptr3, w);

                    outptr += w * 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = w * bottom_blob.h;

                const float* ptr = bottom_blob;
                copy_vec(outptr, ptr, size * bottom_blob.elempack);

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
                copy_vec(outptr, ptr, bottom_blob.w * elempack);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

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
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;

            top_blob_unpacked.dims = dims;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            if (bottom_blob.elempack == 4 && elempack == 1)
            {
                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);
                    float* outptr1 = top_blob_unpacked.channel(p + 1);
                    float* outptr2 = top_blob_unpacked.channel(p + 2);
                    float* outptr3 = top_blob_unpacked.channel(p + 3);

                    pack4_to_pack1_vec(r0, outptr0, outptr1, outptr2, outptr3, size);

                    p += 4;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
            {
                int size = bottom_blob.total();

                const float* ptr = bottom_blob;
                float* outptr = top_blob_unpacked.channel(p);
                copy_vec(outptr, ptr, size * bottom_blob.elempack);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
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
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const float* ptr = bottom_blob.channel(q).depth(i);
                    copy_vec(outptr, ptr, size * elempack);

                    outptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
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
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const float* ptr = bottom_blob.channel(q).depth(i).row(j);
                        copy_vec(outptr, ptr, bottom_blob.w * elempack);

                        outptr += bottom_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const float* ptr = bottom_blob.channel(q);
                copy_vec(outptr, ptr, size * elempack);

                outptr += size * elempack;
            }
        }
    }

    return 0;
#else
    // no altivec -> base implementation
    return Concat::forward(bottom_blobs, top_blobs, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
