// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "slice.h"

namespace ncnn {

Slice::Slice()
{
}

int Slice::load_param(const ParamDict& pd)
{
    slices = pd.get(0, Mat());
    axis = pd.get(1, 0);
    indices = pd.get(2, Mat());

    return 0;
}

int Slice::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    const int* slices_ptr = slices;
    const int* indices_ptr = indices;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        int w = bottom_blob.w;

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
            top_blob.create(slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const unsigned char* ptr = (const unsigned char*)bottom_blob + q * elemsize;
            unsigned char* outptr = top_blob;
            memcpy(outptr, ptr, slice * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
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
            top_blob.create(w, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            size_t size = (size_t)w * slice;

            const unsigned char* ptr = bottom_blob.row<const unsigned char>(q);
            unsigned char* outptr = top_blob;
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
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
            top_blob.create(slice, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < h; j++)
            {
                unsigned char* outptr = top_blob.row<unsigned char>(j);
                const unsigned char* ptr = bottom_blob.row<const unsigned char>(j) + q * elemsize;
                memcpy(outptr, ptr, slice * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
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

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            size_t size = bottom_blob.cstep * slice;

            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob;
            memcpy(outptr, ptr, size * elemsize);

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
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
            top_blob.create(w, slice, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                for (int j = 0; j < d; j++)
                {
                    size_t size = (size_t)w * slice;

                    unsigned char* outptr = top_blob.channel(p).depth(j);
                    const unsigned char* ptr = bottom_blob.channel(p).depth(j).row<const unsigned char>(q);
                    memcpy(outptr, ptr, size * elemsize);
                }
            }

            q += slice;
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
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
            top_blob.create(slice, h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                unsigned char* outptr = top_blob.channel(p);
                const Mat m = bottom_blob.channel(p);

                for (int j = 0; j < d; j++)
                {
                    for (int k = 0; k < h; k++)
                    {
                        const unsigned char* ptr = m.depth(j).row<const unsigned char>(k) + q * elemsize;
                        memcpy(outptr, ptr, slice * elemsize);

                        outptr += slice * elemsize;
                    }
                }
            }

            q += slice;
        }

        return 0;
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
            top_blob.create(w, h, slice, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                size_t size = (size_t)w * h * slice;

                unsigned char* outptr = top_blob.channel(p);
                const unsigned char* ptr = bottom_blob.channel(p).depth(q);
                memcpy(outptr, ptr, size * elemsize);
            }

            q += slice;
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
