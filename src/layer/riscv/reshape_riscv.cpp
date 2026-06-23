// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_riscv.h"

#include <string.h>

namespace ncnn {

Reshape_riscv::Reshape_riscv()
{
}

int Reshape_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (batch_mode == 0 || batch_axis == 0)
        return Reshape::forward(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
    if (batch_mode == 1)
        total *= bottom_blob.n;

    if (ndim == 0)
        return -1;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outw == -1)
            outw = total;
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
    }

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

    if (batch_mode == 1)
    {
        if (bottom_blob.elempack != 1)
            return -1;

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

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        for (int i = batch_axis; i < ndim; i++)
            suffix *= shape[i];

        if (ndim == 1)
            top_blob.create_batch(outw, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create_batch(outw, outh, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create_batch(outw, outh, outc, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create_batch(outw, outh, outd, outc, batch, bottom_blob.elemsize, 1, opt.blob_allocator);

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

    return -1;
}

} // namespace ncnn
