// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat.h"

namespace ncnn {

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_BATCH
    support_batch = true;
#endif
}

int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

#if NCNN_BATCH
    support_batch = true;
#else
    if (axis == -233)
    {
        NCNN_LOGE("please build ncnn with NCNN_BATCH enabled for batch concat");
        return -1;
    }
#endif

    return 0;
}

#if NCNN_BATCH
bool Concat::same_shape_except_axis(const Mat& a, const Mat& b, int axis)
{
    if (a.dims != b.dims)
        return false;

    const Mat a_shape = a.shape();
    const Mat b_shape = b.shape();
    const int dims = a_shape.dims;

    if ((axis < 0 || axis != dims - 1) && a_shape.w != b_shape.w)
        return false;
    if (dims >= 2 && (axis < 0 || axis != dims - 2) && a_shape.h != b_shape.h)
        return false;
    if (dims == 4 && (axis < 0 || axis != 1) && a_shape.d != b_shape.d)
        return false;
    if (dims >= 3 && (axis < 0 || axis != 0) && a_shape.c != b_shape.c)
        return false;

    return true;
}

static void copy_mat_data(const Mat& src, Mat& dst)
{
    if (src.cstep == dst.cstep)
    {
        memcpy(dst.data, src.data, src.total() * src.elemsize);
        return;
    }

    const size_t plane_size = (size_t)src.w * src.h * src.d * src.elemsize;
    for (int q = 0; q < src.c; q++)
    {
        memcpy(dst.channel(q).data, src.channel(q).data, plane_size);
    }
}

int Concat::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    const bool axis_is_batch = axis == -233;
    int positive_axis = axis_is_batch ? -1 : (axis < 0 ? dims + axis : axis);

    if (axis_is_batch)
    {
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int c = bottom_blobs[0].c;

        int top_n = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            if (bottom_blob.dims != dims || bottom_blob.w != w || bottom_blob.h != h || bottom_blob.d != d || bottom_blob.c != c || bottom_blob.elemsize != elemsize || bottom_blob.elempack != elempack)
            {
                NCNN_LOGE("concat along batch axis with mismatched shape");
                return -100;
            }
            top_n += bottom_blob.n;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create_like(bottom_blobs[0], top_n, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int out_b = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            for (int j = 0; j < bottom_blob.n; j++)
            {
                const Mat bottom_slice = bottom_blob.batch(j);
                Mat top_slice = top_blob.batch(out_b + j);
                copy_mat_data(bottom_slice, top_slice);
            }
            out_b += bottom_blob.n;
        }

        return 0;
    }

    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    for (size_t b = 1; b < bottom_blobs.size(); b++)
    {
        if (!same_shape_except_axis(bottom_blobs[0], bottom_blobs[b], positive_axis))
        {
            NCNN_LOGE("concat with mismatched shape on non-concat axis");
            return -100;
        }
    }

    int B = 1;
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        B = std::max(B, bottom_blobs[b].n);
    }
    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        if (bottom_blobs[b].n != 1 && bottom_blobs[b].n != B)
        {
            NCNN_LOGE("concat batch size mismatch: expected batch 1 or %d but bottom %d has batch %d", B, (int)b, bottom_blobs[b].n);
            return -1;
        }
    }

    std::vector<Mat> bottom_b(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        bottom_b[i] = bottom_blobs[i].n > 1 ? bottom_blobs[i].batch(0) : bottom_blobs[i];
    }
    std::vector<Mat> top_b(1);
    int ret = forward(bottom_b, top_b, opt);
    if (ret != 0)
        return ret;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(top_b[0], B, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    copy_mat_data(top_b[0], top_blob.batch(0));

    for (int b = 1; b < B; b++)
    {
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_b[i] = bottom_blobs[i].n > 1 ? bottom_blobs[i].batch(b) : bottom_blobs[i];
        }
        top_b[0].release();
        ret = forward(bottom_b, top_b, opt);
        if (ret != 0)
            return ret;

        copy_mat_data(top_b[0], top_blob.batch(b));
    }

    return 0;
}
#endif // NCNN_BATCH

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    int positive_axis = axis == -233 ? -1 : (axis < 0 ? dims + axis : axis);

#if NCNN_BATCH
    if (support_batch)
    {
        int B = 1;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            if (bottom_blobs[b].n > 1)
            {
                B = bottom_blobs[b].n;
                break;
            }
        }
        if (axis == -233 || B > 1)
            return forward_batch(bottom_blobs, top_blobs, opt);
    }
#else
    if (axis == -233)
    {
        NCNN_LOGE("please build ncnn with NCNN_BATCH enabled for batch concat");
        return -1;
    }
#endif

    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    size_t elemsize = bottom_blobs[0].elemsize;

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
