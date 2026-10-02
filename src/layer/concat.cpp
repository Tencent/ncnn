// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat.h"

namespace ncnn {

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

template<typename MatType>
static int concat_check_shape(const std::vector<MatType>& bottom_blobs, int axis)
{
    if (bottom_blobs.empty())
        return -1;

    const MatType& ref = bottom_blobs[0];
    const int dims = ref.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims < 1 || dims > 4 || positive_axis < 0 || positive_axis >= dims)
    {
        NCNN_LOGE("Concat invalid dims=%d axis=%d", dims, axis);
        return -1;
    }

    for (size_t b = 1; b < bottom_blobs.size(); b++)
    {
        const MatType& bottom_blob = bottom_blobs[b];

        if (bottom_blob.dims != dims)
        {
            NCNN_LOGE("Concat shape mismatch: dims %d vs %d", dims, bottom_blob.dims);
            return -1;
        }

        if (dims == 2 && positive_axis == 0 && bottom_blob.w != ref.w)
        {
            NCNN_LOGE("Concat shape mismatch: w %d vs %d (axis=%d)", ref.w, bottom_blob.w, axis);
            return -1;
        }
        if (dims == 2 && positive_axis == 1 && bottom_blob.h != ref.h)
        {
            NCNN_LOGE("Concat shape mismatch: h %d vs %d (axis=%d)", ref.h, bottom_blob.h, axis);
            return -1;
        }

        if ((dims == 3 || dims == 4) && positive_axis == 0)
        {
            if (bottom_blob.w != ref.w || bottom_blob.h != ref.h || bottom_blob.d != ref.d)
            {
                NCNN_LOGE("Concat shape mismatch: (%d,%d,%d) vs (%d,%d,%d) (axis=%d)",
                          ref.w, ref.h, ref.d, bottom_blob.w, bottom_blob.h, bottom_blob.d, axis);
                return -1;
            }
        }
        if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
        {
            if (bottom_blob.w != ref.w || bottom_blob.d != ref.d || bottom_blob.c != ref.c)
            {
                NCNN_LOGE("Concat shape mismatch: non-concat dims disagree (axis=%d)", axis);
                return -1;
            }
        }
        if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
        {
            if (bottom_blob.h != ref.h || bottom_blob.d != ref.d || bottom_blob.c != ref.c)
            {
                NCNN_LOGE("Concat shape mismatch: non-concat dims disagree (axis=%d)", axis);
                return -1;
            }
        }
        if (dims == 4 && positive_axis == 1)
        {
            if (bottom_blob.w != ref.w || bottom_blob.h != ref.h || bottom_blob.c != ref.c)
            {
                NCNN_LOGE("Concat shape mismatch: non-concat dims disagree (axis=%d)", axis);
                return -1;
            }
        }
    }

    return 0;
}

int Concat::check_shape(const std::vector<Mat>& bottom_blobs) const
{
    return concat_check_shape(bottom_blobs, axis);
}

#if NCNN_VULKAN
int Concat::check_shape(const std::vector<VkMat>& bottom_blobs) const
{
    return concat_check_shape(bottom_blobs, axis);
}
#endif // NCNN_VULKAN

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int ret_shape = check_shape(bottom_blobs);
    if (ret_shape != 0)
        return ret_shape;

    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w * elemsize;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int size = w * bottom_blob.h;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, size * elemsize);

            outptr += size * elemsize;
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned char* outptr = top_blob.row<unsigned char>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned char* ptr = bottom_blob.row<const unsigned char>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elemsize;
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
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.c;
            size_t size = bottom_blob.cstep * channels;

            const unsigned char* ptr = bottom_blob;
            unsigned char* outptr = top_blob.channel(q);
            memcpy(outptr, ptr, size * elemsize);

            q += channels;
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const unsigned char* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elemsize;
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

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const unsigned char* ptr = bottom_blob.channel(q).depth(i).row<const unsigned char>(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elemsize;
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

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const unsigned char* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elemsize;
            }
        }
    }

    return 0;
}

} // namespace ncnn
