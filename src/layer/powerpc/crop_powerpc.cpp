// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "crop_powerpc.h"

#include <stdint.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Crop_powerpc::Crop_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
static void crop_pack4_altivec(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const float* ptr = src.row(top) + left * 4;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            __ncnn_altivec_store(outptr, _p);

            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}
#endif // __ALTIVEC__

int Crop_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ALTIVEC__
    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (!starts_expr.empty() && !ends_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(1);
        bottom_blob_shapes[0] = bottom_blob.shape();
        eval_crop_expr(bottom_blob_shapes, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    if (elempack == 4 && elemsize == 4u)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                crop_pack4_altivec(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                crop_pack4_altivec(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;

                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    crop_pack4_altivec(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;

                    return 0;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);

                        crop_pack4_altivec(m, borderm, _hoffset, _woffset);
                    }
                }

                return 0;
            }
        }
    }
#endif // __ALTIVEC__

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Crop::forward(bottom_blob_unpacked, top_blob, opt);
}

int Crop_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __ALTIVEC__
    int _woffset, _hoffset, _doffset, _coffset;
    int _outw, _outh, _outd, _outc;
    if (!starts_expr.empty() && !ends_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_crop_expr(bottom_blob_shapes, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
    }

    if (elempack == 4 && elemsize == 4u)
    {
        if (dims == 1)
        {
            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                crop_pack4_altivec(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                crop_pack4_altivec(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;

                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    crop_pack4_altivec(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == 4)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone(opt.blob_allocator);
                    if (top_blob.empty())
                        return -100;

                    return 0;
                }

                top_blob.create(_outw, _outh, _outd, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    for (int z = 0; z < _outd; z++)
                    {
                        const Mat m = bottom_blob_sliced.channel(q).depth(z + _doffset);
                        Mat borderm = top_blob.channel(q).depth(z);

                        crop_pack4_altivec(m, borderm, _hoffset, _woffset);
                    }
                }

                return 0;
            }
        }
    }
#endif // __ALTIVEC__

    // the generic crop kernel is not packing aware, unpack every input blob
    std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack == 1)
        {
            bottom_blobs_unpacked[i] = bottom_blobs[i];
            continue;
        }

        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blobs[i], bottom_blobs_unpacked[i], 1, opt_pack1);
        if (bottom_blobs_unpacked[i].empty())
            return -100;
    }

    return Crop::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
