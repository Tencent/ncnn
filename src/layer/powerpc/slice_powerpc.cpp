// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "slice_powerpc.h"

#include <algorithm>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Slice_powerpc::Slice_powerpc()
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

// interleave 4 pack1 rows into one pack4 row, vectorized 4x4 transpose
static void pack1_to_pack4_vec(const float* r0, const float* r1, const float* r2, const float* r3, float* outptr0, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _r0 = __ncnn_altivec_load(r0);
        v4f32 _r1 = __ncnn_altivec_load(r1);
        v4f32 _r2 = __ncnn_altivec_load(r2);
        v4f32 _r3 = __ncnn_altivec_load(r3);

        transpose4x4_ps(_r0, _r1, _r2, _r3);

        __ncnn_altivec_store(outptr0, _r0);
        __ncnn_altivec_store(outptr0 + 4, _r1);
        __ncnn_altivec_store(outptr0 + 8, _r2);
        __ncnn_altivec_store(outptr0 + 12, _r3);

        r0 += 4;
        r1 += 4;
        r2 += 4;
        r3 += 4;
        outptr0 += 16;
    }
    for (; i < n; i++)
    {
        outptr0[0] = *r0++;
        outptr0[1] = *r1++;
        outptr0[2] = *r2++;
        outptr0[3] = *r3++;

        outptr0 += 4;
    }
}
#endif // __ALTIVEC__

int Slice_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    int elembits = bottom_blob.elembits();
    if (elembits != 32)
    {
        // fp16 / bf16 -> base implementation, unpacked
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
            if (bottom_blob_unpacked.empty())
                return -100;

            std::vector<Mat> bottom_blobs_unpacked;
            bottom_blobs_unpacked.push_back(bottom_blob_unpacked);
            return Slice::forward(bottom_blobs_unpacked, top_blobs, opt);
        }

        return Slice::forward(bottom_blobs, top_blobs, opt);
    }

#if __ALTIVEC__
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;
    const int* indices_ptr = indices;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = (const float*)bottom_blob + q;
            float* outptr = top_blob;
            copy_vec(outptr, ptr, slice);

            q += slice;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
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
            if (bottom_blob_unpacked.empty())
                return -100;
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

                    pack1_to_pack4_vec(r0, r1, r2, r3, outptr0, w);

                    ptr += w * 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = w * top_blob.h;

                float* outptr = top_blob;
                copy_vec(outptr, ptr, size * top_blob.elempack);

                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
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
                copy_vec(outptr, ptr, top_blob.w * elempack);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = channels - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? channels + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = opt.use_packing_layout && slice % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

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
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);
                    const float* r2 = bottom_blob_unpacked.channel(p + 2);
                    const float* r3 = bottom_blob_unpacked.channel(p + 3);

                    float* outptr0 = top_blob.channel(q);

                    pack1_to_pack4_vec(r0, r1, r2, r3, outptr0, size);

                    p += 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = top_blob.total();

                const float* ptr = bottom_blob_unpacked.channel(p);
                float* outptr = top_blob;
                copy_vec(outptr, ptr, size * top_blob.elempack);

                p += top_blob.c;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    int size = top_blob.w * top_blob.h;

                    float* outptr = top_blob.channel(p).depth(j);
                    copy_vec(outptr, ptr, size * elempack);

                    ptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];

                        float* outptr = top_blob.channel(p).depth(j).row(k);
                        copy_vec(outptr, ptr, top_blob.w * elempack);

                        ptr += top_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = d - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? d + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((d - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice, channels, elemsize, elempack, opt.blob_allocator);
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

                int size = top_blob.w * top_blob.h * top_blob.d;

                float* outptr = top_blob.channel(p);
                copy_vec(outptr, ptr, size * elempack);

                ptr += size * elempack;
            }
        }
    }

    return 0;
#else
    // no altivec -> base implementation
    return Slice::forward(bottom_blobs, top_blobs, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
