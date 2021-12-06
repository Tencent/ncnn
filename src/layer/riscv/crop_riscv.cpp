// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "crop_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Crop_riscv::Crop_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

#if __riscv_vector
static void crop_packn_rvv(const Mat& src, Mat& dst, int top, int left, int packn)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const word_type vl = vsetvl_e32m1(packn);

    const float* ptr = src.row(top) + left * packn;
    float* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            vfloat32m1_t _p = vle32_v_f32m1(ptr, vl);
            vse32_v_f32m1(outptr, _p, vl);

            ptr += packn;
            outptr += packn;
        }

        ptr += (left + right) * packn;
    }
}

static void crop_packn_bf16_fp16s_rvv(const Mat& src, Mat& dst, int top, int left, int packn)
{
    int w = dst.w;
    int h = dst.h;
    int right = src.w - dst.w - left;

    const word_type vl = vsetvl_e16m1(packn);

    const unsigned short* ptr = src.row<unsigned short>(top) + left * packn;
    unsigned short* outptr = dst;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            vuint16m1_t _p = vle16_v_u16m1(ptr, vl);
            vse16_v_u16m1(outptr, _p, vl);

            ptr += packn;
            outptr += packn;
        }

        ptr += (left + right) * packn;
    }
}
#endif // __riscv_vector

int Crop_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __riscv_vector
    const int packn = csrr_vlenb() / (elembits / 8);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == packn)
    {
        int _woffset, _hoffset, _doffset, _coffset;
        int _outw, _outh, _outd, _outc;
        resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);

        if (dims == 1)
        {
            int out_elempack = _outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % packn == 0 && out_elempack == packn)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elembits == 16)
                    crop_packn_bf16_fp16s_rvv(bottom_blob, top_blob, 0, _woffset / elempack, packn);
                else
                    crop_packn_rvv(bottom_blob, top_blob, 0, _woffset / elempack, packn);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % packn == 0 && out_elempack == packn)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elembits == 16)
                    crop_packn_bf16_fp16s_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset, packn);
                else
                    crop_packn_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset, packn);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % packn == 0 && out_elempack == packn)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elembits == 16)
                        crop_packn_bf16_fp16s_rvv(m, borderm, _hoffset, _woffset, packn);
                    else
                        crop_packn_rvv(m, borderm, _hoffset, _woffset, packn);
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % packn == 0 && out_elempack == packn)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
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

                        if (elembits == 16)
                            crop_packn_bf16_fp16s_rvv(m, borderm, _hoffset, _woffset, packn);
                        else
                            crop_packn_rvv(m, borderm, _hoffset, _woffset, packn);
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    return Crop::forward(bottom_blob_unpacked, top_blob, opt);
}

int Crop_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int elembits = bottom_blob.elembits();

#if __riscv_vector
    const int packn = csrr_vlenb() / (elembits / 8);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int ref_elempack = reference_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __riscv_vector
    if (elempack == packn)
    {
        int _woffset, _hoffset, _doffset, _coffset;
        int _outw, _outh, _outd, _outc;
        if (woffset == -233)
        {
            resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
        }
        else
        {
            resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);
        }

        if (dims == 1)
        {
            int out_elempack = _outw % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_woffset % packn == 0 && out_elempack == packn)
            {
                top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elembits == 16)
                    crop_packn_bf16_fp16s_rvv(bottom_blob, top_blob, 0, _woffset / elempack, packn);
                else
                    crop_packn_rvv(bottom_blob, top_blob, 0, _woffset / elempack, packn);

                return 0;
            }
        }

        if (dims == 2)
        {
            int out_elempack = _outh % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_hoffset % packn == 0 && out_elempack == packn)
            {
                top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                if (elembits == 16)
                    crop_packn_bf16_fp16s_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset, packn);
                else
                    crop_packn_rvv(bottom_blob, top_blob, _hoffset / elempack, _woffset, packn);

                return 0;
            }
        }

        if (dims == 3)
        {
            int out_elempack = _outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outc / out_elempack == channels && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % packn == 0 && out_elempack == packn)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    if (elembits == 16)
                        crop_packn_bf16_fp16s_rvv(m, borderm, _hoffset, _woffset, packn);
                    else
                        crop_packn_rvv(m, borderm, _hoffset, _woffset, packn);
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int out_elempack = _outc % packn == 0 ? packn : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh == h && _outd == d && _outc / out_elempack == channels && out_elempack == packn)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (_coffset % packn == 0 && out_elempack == packn)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h && _outd == d)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
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

                        if (elembits == 16)
                            crop_packn_bf16_fp16s_rvv(m, borderm, _hoffset, _woffset, packn);
                        else
                            crop_packn_rvv(m, borderm, _hoffset, _woffset, packn);
                    }
                }

                return 0;
            }
        }
    }
#endif // __riscv_vector

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
    }

    Mat reference_blob_unpacked = reference_blob;
    if (ref_elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(reference_blob, reference_blob_unpacked, 1, opt_pack1);
    }

    std::vector<Mat> bottom_blobs_unpacked(2);
    bottom_blobs_unpacked[0] = bottom_blob_unpacked;
    bottom_blobs_unpacked[1] = reference_blob_unpacked;

    return Crop::forward(bottom_blobs_unpacked, top_blobs, opt);
}

} // namespace ncnn
