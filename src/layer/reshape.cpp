// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape.h"

#include "expression.h"

#include <string.h>

namespace ncnn {

Reshape::Reshape()
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_BATCH
    batch_mode = 0;
    batch_axis = 0;
#endif
}

int Reshape::load_param(const ParamDict& pd)
{
    w = pd.get(0, -233);
    h = pd.get(1, -233);
    d = pd.get(11, -233);
    c = pd.get(2, -233);

    ndim = 4;
    if (d == -233)
        ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

#if NCNN_BATCH
    batch_mode = pd.get(12, 0);
    batch_axis = pd.get(13, 0);
    if (batch_mode != 0)
    {
        support_batch = true;
    }
#else
    if (pd.get(12, 0) != 0)
    {
        NCNN_LOGE("please build ncnn with NCNN_BATCH enabled for batch inference");
        return -1;
    }
#endif

    shape_expr = pd.get(6, "");

    // count reference blobs
    if (!shape_expr.empty())
    {
        const int blob_count = count_expression_blobs(shape_expr);
        if (blob_count > 1)
            one_blob_only = false;

        // resolve ndim from expression
        std::vector<Mat> blobs(blob_count);
        std::vector<int> outshape;
        int er = eval_list_expression(shape_expr, blobs, outshape);
        if (er != 0)
            return -1;

        ndim = (int)outshape.size();
    }

    return 0;
}

int Reshape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1);
    bottom_blobs[0] = bottom_blob;
    std::vector<Mat> top_blobs(1);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Reshape::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
    }

#if NCNN_BATCH
    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;
#else
    const int batch_mode = 0;
#endif

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
#if NCNN_BATCH
    if (batch_mode == 1)
        total *= bottom_blob.n;
#endif

    int dims = bottom_blob.dims;

#if NCNN_BATCH
    if (batch_mode != 0 && ndim == 0)
        return -1;
#endif

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;

        if (outw == -1)
            outw = total;

        if (batch_mode == 0 && dims == 1 && bottom_blob.w == outw)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        if (batch_mode == 0 && dims == 2 && bottom_blob.h == outh)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        if (batch_mode == 0 && dims == 3 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;

        if (batch_mode == 0 && dims == 4 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }
    }

#if NCNN_BATCH
    if (batch_mode == 1)
    {
        if (bottom_blob.elempack != 1)
            return -1;

        int shape[4] = {0, 0, 0, 0};
        if (ndim == 1)
            shape[0] = outw;
        if (ndim == 2)
        {
            shape[0] = outh;
            shape[1] = outw;
        }
        if (ndim == 3)
        {
            shape[0] = outc;
            shape[1] = outh;
            shape[2] = outw;
        }
        if (ndim == 4)
        {
            shape[0] = outc;
            shape[1] = outd;
            shape[2] = outh;
            shape[3] = outw;
        }

        if (batch_axis != 0)
        {
            size_t prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            size_t suffix = 1;
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];

            if (ndim == 1)
                top_blob.create(outw, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 2)
                top_blob.create(outw, outh, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 3)
                top_blob.create(outw, outh, outc, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 4)
                top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, 1, opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
            const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;
            const size_t suffix_bytes = suffix * bottom_blob.elemsize;
            for (size_t p = 0; p < prefix; p++)
            {
                const size_t srci = p * suffix;
                const int sq = srci / bottom_channel_size;
                const size_t sr = srci - (size_t)sq * bottom_channel_size;

                for (int b = 0; b < bottom_blob.n; b++)
                {
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.cstep + sr) * bottom_blob.elemsize;

                    const size_t dsti = (p * bottom_blob.n + b) * suffix;
                    const int dq = dsti / top_channel_size;
                    const size_t dr = dsti - (size_t)dq * top_channel_size;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)dq * top_blob.cstep + dr) * bottom_blob.elemsize;

                    memcpy(outptr, ptr, suffix_bytes);
                }
            }

            return 0;
        }

        Mat bottom_blob_flattened(total, bottom_blob.elemsize, opt.blob_allocator);
        if (bottom_blob_flattened.empty())
            return -100;

        unsigned char* outptr = bottom_blob_flattened;
        const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elemsize;
        for (int b = 0; b < bottom_blob.n; b++)
        {
            const Mat bottom_blob_b = bottom_blob.batch(b);
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob_b + bottom_blob.cstep * q * bottom_blob.elemsize;
                memcpy(outptr, ptr, size);
                outptr += size;
            }
        }

        if (ndim == 1)
            top_blob = bottom_blob_flattened.reshape(outw, opt.blob_allocator);
        if (ndim == 2)
            top_blob = bottom_blob_flattened.reshape(outw, outh, opt.blob_allocator);
        if (ndim == 3)
            top_blob = bottom_blob_flattened.reshape(outw, outh, outc, opt.blob_allocator);
        if (ndim == 4)
            top_blob = bottom_blob_flattened.reshape(outw, outh, outd, outc, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        return 0;
    }

    if (batch_mode == 2)
    {
        if (bottom_blob.n != 1 || bottom_blob.elempack != 1)
            return -1;

        size_t out_total = outw;
        if (ndim == 2)
            out_total *= outh;
        if (ndim == 3)
            out_total *= (size_t)outh * outc;
        if (ndim == 4)
            out_total *= (size_t)outh * outd * outc;

        if (out_total == 0)
            return -1;

        const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
        const int batch = bottom_total / out_total;
        if ((size_t)batch * out_total != bottom_total)
            return -1;

        if (batch_axis != 0)
        {
            int shape[4] = {0, 0, 0, 0};
            if (ndim == 1)
                shape[0] = outw;
            if (ndim == 2)
            {
                shape[0] = outh;
                shape[1] = outw;
            }
            if (ndim == 3)
            {
                shape[0] = outc;
                shape[1] = outh;
                shape[2] = outw;
            }
            if (ndim == 4)
            {
                shape[0] = outc;
                shape[1] = outd;
                shape[2] = outh;
                shape[3] = outw;
            }

            size_t prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            size_t suffix = 1;
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];

            if (ndim == 1)
                top_blob.create(outw, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
            if (ndim == 2)
                top_blob.create(outw, outh, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
            if (ndim == 3)
                top_blob.create(outw, outh, outc, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
            if (ndim == 4)
                top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, 1, batch, opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
            const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;
            const size_t suffix_bytes = suffix * bottom_blob.elemsize;
            for (size_t p = 0; p < prefix; p++)
            {
                for (int b = 0; b < batch; b++)
                {
                    const size_t srci = (p * batch + b) * suffix;
                    const int sq = srci / bottom_channel_size;
                    const size_t sr = srci - (size_t)sq * bottom_channel_size;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)sq * bottom_blob.cstep + sr) * bottom_blob.elemsize;

                    const size_t dsti = p * suffix;
                    const int dq = dsti / top_channel_size;
                    const size_t dr = dsti - (size_t)dq * top_channel_size;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)dq * top_blob.cstep + dr) * bottom_blob.elemsize;

                    memcpy(outptr, ptr, suffix_bytes);
                }
            }

            return 0;
        }

        Mat bottom_blob_flattened(bottom_total, bottom_blob.elemsize, opt.workspace_allocator);
        if (bottom_blob_flattened.empty())
            return -100;

        unsigned char* outptr = bottom_blob_flattened;
        const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elemsize;
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + bottom_blob.cstep * q * bottom_blob.elemsize;
            memcpy(outptr, ptr, size);
            outptr += size;
        }

        if (ndim == 1)
            top_blob.create(outw, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create(outw, outh, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc, bottom_blob.elemsize, 1, batch, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, 1, batch, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        const unsigned char* ptr = bottom_blob_flattened;
        const size_t out_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d * bottom_blob.elemsize;
        for (int b = 0; b < batch; b++)
        {
            Mat top_blob_b = top_blob.batch(b);
            for (int q = 0; q < top_blob.c; q++)
            {
                memcpy(top_blob_b.channel(q), ptr, out_channel_size);
                ptr += out_channel_size;
            }
        }

        return 0;
    }
#endif // NCNN_BATCH

    if (ndim == 1)
    {
        top_blob = bottom_blob.reshape(outw, opt.blob_allocator);
    }
    if (ndim == 2)
    {
        top_blob = bottom_blob.reshape(outw, outh, opt.blob_allocator);
    }
    if (ndim == 3)
    {
        top_blob = bottom_blob.reshape(outw, outh, outc, opt.blob_allocator);
    }
    if (ndim == 4)
    {
        top_blob = bottom_blob.reshape(outw, outh, outd, outc, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    return 0;
}

int Reshape::eval_shape_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh, int& outd, int& outc) const
{
    // [size(@0,0),size(@0,1),12,64]
    std::vector<int> shape;
    int er = eval_list_expression(shape_expr, bottom_blobs, shape);
    if (er != 0)
        return -1;

    outw = 1;
    outh = 1;
    outd = 1;
    outc = 1;
    if (shape.size() == 1)
    {
        outw = shape[0];
    }
    if (shape.size() == 2)
    {
        outw = shape[0];
        outh = shape[1];
    }
    if (shape.size() == 3)
    {
        outw = shape[0];
        outh = shape[1];
        outc = shape[2];
    }
    if (shape.size() == 4)
    {
        outw = shape[0];
        outh = shape[1];
        outd = shape[2];
        outc = shape[3];
    }

    return 0;
}

} // namespace ncnn
