// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_loongarch.h"

#include <string.h>

namespace ncnn {

Reshape_loongarch::Reshape_loongarch()
{
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static int unpack_to_pack1(const Mat& src, Mat& dst, const Option& opt)
{
    if (src.elempack == 1)
    {
        dst = src;
        return 0;
    }

    const size_t itemsize = src.elemsize / src.elempack;

    if (src.dims == 1)
    {
        dst.create(src.w * src.elempack, itemsize, opt.workspace_allocator);
        if (dst.empty())
            return -100;

        memcpy(dst, src, (size_t)src.w * src.elemsize);
        return 0;
    }

    if (src.dims == 2)
    {
        dst.create(src.w, src.h * src.elempack, itemsize, opt.workspace_allocator);
        if (dst.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < src.h; i++)
        {
            const unsigned char* ptr = src.row<const unsigned char>(i);

            for (int k = 0; k < src.elempack; k++)
            {
                unsigned char* outptr = dst.row<unsigned char>(i * src.elempack + k);

                for (int j = 0; j < src.w; j++)
                {
                    memcpy(outptr + j * itemsize, ptr + j * src.elemsize + k * itemsize, itemsize);
                }
            }
        }

        return 0;
    }

    if (src.dims == 3)
    {
        dst.create(src.w, src.h, src.c * src.elempack, itemsize, opt.workspace_allocator);
        if (dst.empty())
            return -100;
    }
    else // if (src.dims == 4)
    {
        dst.create(src.w, src.h, src.d, src.c * src.elempack, itemsize, opt.workspace_allocator);
        if (dst.empty())
            return -100;
    }

    const int size = src.w * src.h * src.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < src.c; q++)
    {
        const unsigned char* ptr = src.channel(q);

        for (int k = 0; k < src.elempack; k++)
        {
            unsigned char* outptr = dst.channel(q * src.elempack + k);

            for (int i = 0; i < size; i++)
            {
                memcpy(outptr + i * itemsize, ptr + i * src.elemsize + k * itemsize, itemsize);
            }
        }
    }

    return 0;
}

static int pack_from_pack1(const Mat& src, Mat& dst, int out_elempack, const Option& opt)
{
    if (out_elempack == 1)
    {
        dst = src;
        return 0;
    }

    const size_t itemsize = src.elemsize;
    const size_t out_elemsize = itemsize * out_elempack;

    if (src.dims == 2)
    {
        dst.create(src.w, src.h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < dst.h; i++)
        {
            unsigned char* outptr = dst.row<unsigned char>(i);

            for (int j = 0; j < src.w; j++)
            {
                for (int k = 0; k < out_elempack; k++)
                {
                    const unsigned char* ptr = src.row<const unsigned char>(i * out_elempack + k);
                    memcpy(outptr + (j * out_elempack + k) * itemsize, ptr + j * itemsize, itemsize);
                }
            }
        }

        return 0;
    }

    if (src.dims == 3)
    {
        dst.create(src.w, src.h, src.c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return -100;
    }
    else // if (src.dims == 4)
    {
        dst.create(src.w, src.h, src.d, src.c / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (dst.empty())
            return -100;
    }

    const int size = src.w * src.h * src.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < dst.c; q++)
    {
        unsigned char* outptr = dst.channel(q);

        for (int i = 0; i < size; i++)
        {
            for (int k = 0; k < out_elempack; k++)
            {
                const unsigned char* ptr = src.channel(q * out_elempack + k);
                memcpy(outptr + (i * out_elempack + k) * itemsize, ptr + i * itemsize, itemsize);
            }
        }
    }

    if (dst.empty())
        return -100;

    return 0;
}

static int reshape_pack1_blob(const Mat& bottom_blob, Mat& top_blob, int ndim, int outw, int outh, int outd, int outc, const Option& opt)
{
    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    if (ndim == 2)
    {
        top_blob = bottom_blob.reshape(outw, outh, opt.blob_allocator);
    }
    else if (ndim == 3)
    {
        top_blob = bottom_blob.reshape(outw, outh, outc, opt.blob_allocator);
    }
    else if (ndim == 4)
    {
        top_blob = bottom_blob.reshape(outw, outh, outd, outc, opt.blob_allocator);
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

int Reshape_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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

    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        Mat bottom_blob_unpacked;
        int ret = unpack_to_pack1(bottom_blob, bottom_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        Option opt_pack1 = opt;
        opt_pack1.use_packing_layout = false;

        Mat top_blob_unpacked;
        ret = reshape_pack1_blob(bottom_blob_unpacked, top_blob_unpacked, ndim, outw, outh, outd, outc, opt_pack1);
        if (ret != 0)
            return ret;

        if (out_elempack == 1)
        {
            top_blob = top_blob_unpacked;
            return 0;
        }

        return pack_from_pack1(top_blob_unpacked, top_blob, out_elempack, opt);
    }

    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        outd = 1;
    }

    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outd == 0)
            outd = bottom_blob.d;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;
    }

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
        out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
    }

    if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
    {
        top_blob = bottom_blob;
        top_blob.dims = ndim;
        top_blob.w = outw;
        top_blob.h = outh;
        top_blob.d = outd;
        return 0;
    }

    Mat bottom_blob_unpacked;
    int ret = unpack_to_pack1(bottom_blob, bottom_blob_unpacked, opt);
    if (ret != 0)
        return ret;

    Option opt_pack1 = opt;
    opt_pack1.use_packing_layout = false;

    Mat top_blob_unpacked;
    ret = reshape_pack1_blob(bottom_blob_unpacked, top_blob_unpacked, ndim, outw, outh, outd, outc, opt_pack1);
    if (ret != 0)
        return ret;

    if (out_elempack == 1)
    {
        top_blob = top_blob_unpacked;
        return 0;
    }

    return pack_from_pack1(top_blob_unpacked, top_blob, out_elempack, opt);
}

} // namespace ncnn
