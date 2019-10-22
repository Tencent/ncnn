// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "crop_arm.h"
#include <algorithm>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop_arm)

Crop_arm::Crop_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#if __ARM_NEON
static void crop_pack4_neon(const Mat& src, Mat& dst, int top, int left)
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
            float32x4_t _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}
#endif // __ARM_NEON

int Crop_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int _woffset = woffset;
    int _hoffset = hoffset;
    int _coffset = coffset;
    int _woffset2 = woffset2;
    int _hoffset2 = hoffset2;
    int _coffset2 = coffset2;
    int _outw;
    int _outh;
    int _outc;

    if (elempack == 4)
    {
        if (dims == 1)
        {
            if (outw == -233)
                _outw = w * elempack - _woffset - _woffset2;
            else
                _outw = std::min(outw, w * elempack - _woffset - _woffset2);

            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                crop_pack4_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            if (_hoffset == -233)
            {
                _woffset = 0;
                _woffset2 = 0;
                _outw = w;

                _hoffset = woffset;
                _hoffset2 = woffset2;

                if (outw == -233)
                    _outh = h * elempack - _hoffset - _hoffset2;
                else
                    _outh = std::min(outw, h * elempack - _hoffset - _hoffset2);
            }
            else
            {
                if (outw == -233)
                    _outw = w - _woffset - _woffset2;
                else
                    _outw = std::min(outw, w - _woffset - _woffset2);

                if (outh == -233)
                    _outh = h * elempack - _hoffset - _hoffset2;
                else
                    _outh = std::min(outh, h * elempack - _hoffset - _hoffset2);
            }

            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                crop_pack4_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            if (_hoffset == -233 && _coffset == -233)
            {
                _woffset = 0;
                _woffset2 = 0;
                _outw = w;
                _hoffset = 0;
                _hoffset2 = 0;
                _outh = h;

                _coffset = woffset;
                _coffset2 = woffset2;

                if (outw == -233)
                    _outc = channels * elempack - _coffset - _coffset2;
                else
                    _outc = std::min(outw, channels * elempack - _coffset - _coffset2);
            }
            else if (_hoffset == -233)
            {
                _woffset = 0;
                _woffset2 = 0;
                _outw = w;

                _hoffset = woffset;
                _hoffset2 = woffset2;

                if (outw == -233)
                    _outh = h - _hoffset - _hoffset2;
                else
                    _outh = std::min(outw, h - _hoffset - _hoffset2);

                _coffset = hoffset;
                _coffset2 = hoffset2;

                if (outh == -233)
                    _outc = channels * elempack - _coffset - _coffset2;
                else
                    _outc = std::min(outh, channels * elempack - _coffset - _coffset2);
            }
            else
            {
                if (outw == -233)
                    _outw = w - _woffset - _woffset2;
                else
                    _outw = std::min(outw, w - _woffset - _woffset2);

                if (outh == -233)
                    _outh = h - _hoffset - _hoffset2;
                else
                    _outh = std::min(outh, h - _hoffset - _hoffset2);

                if (outc == -233)
                    _outc = channels * elempack - _coffset - _coffset2;
                else
                    _outc = std::min(outc, channels * elempack - _coffset - _coffset2);
            }

            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    crop_pack4_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Crop::forward(bottom_blob, top_blob, opt);
}

int Crop_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    Mat& top_blob = top_blobs[0];

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    if (elempack == 4)
    {
        int _woffset = woffset;
        int _hoffset = hoffset;
        int _coffset = coffset;
        int _outw;
        int _outh;
        int _outc;

        if (dims == 1)
        {
            if (_woffset == -233)
            {
                const int* param_data = reference_blob;

                _woffset = param_data[0];
                _outw = param_data[3];
            }
            else
            {
                int ref_elempack = reference_blob.elempack;

                if (reference_blob.dims == 1)
                {
                    _outw = reference_blob.w * ref_elempack;
                }
                else if (reference_blob.dims == 2)
                {
                    _outw = reference_blob.w;
                }
                else // if (reference_blob.dims == 3)
                {
                    _outw = reference_blob.w;
                }
            }

            int out_elempack = _outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw / out_elempack == w)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_woffset % 4 == 0 && out_elempack == 4)
            {
                crop_pack4_neon(bottom_blob, top_blob, 0, _woffset / elempack);

                return 0;
            }
        }

        if (dims == 2)
        {
            if (_woffset == -233 && _hoffset == -233)
            {
                const int* param_data = reference_blob;

                _woffset = param_data[0];
                _hoffset = param_data[1];
                _outw = param_data[3];
                _outh = param_data[4];
            }
            else
            {
                int ref_elempack = reference_blob.elempack;

                if (reference_blob.dims == 1)
                {
                    _outw = reference_blob.w * ref_elempack;
                    _outh = h * elempack;
                }
                else if (reference_blob.dims == 2)
                {
                    _outw = reference_blob.w;
                    _outh = reference_blob.h * ref_elempack;
                }
                else // if (reference_blob.dims == 3)
                {
                    _outw = reference_blob.w;
                    _outh = reference_blob.h;
                }
            }

            int out_elempack = _outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_outw == w && _outh / out_elempack == h)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_outw, _outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_hoffset % 4 == 0 && out_elempack == 4)
            {
                crop_pack4_neon(bottom_blob, top_blob, _hoffset / elempack, _woffset);

                return 0;
            }
        }

        if (dims == 3)
        {
            if (_woffset == -233 && _hoffset == -233 && _coffset == -233)
            {
                const int* param_data = reference_blob;

                _woffset = param_data[0];
                _hoffset = param_data[1];
                _coffset = param_data[2];
                _outw = param_data[3];
                _outh = param_data[4];
                _outc = param_data[5];
            }
            else
            {
                int ref_elempack = reference_blob.elempack;

                if (reference_blob.dims == 1)
                {
                    _outw = reference_blob.w * ref_elempack;
                    _outh = h;
                    _outc = channels * elempack;
                }
                else if (reference_blob.dims == 2)
                {
                    _outw = reference_blob.w;
                    _outh = reference_blob.h * ref_elempack;
                    _outc = channels * elempack;
                }
                else // if (reference_blob.dims == 3)
                {
                    _outw = reference_blob.w;
                    _outh = reference_blob.h;
                    _outc = reference_blob.c * ref_elempack;
                }
            }

            int out_elempack = _outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (_coffset % 4 == 0 && out_elempack == 4)
            {
                const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset / out_elempack, _outc / out_elempack);

                if (_outw == w && _outh == h)
                {
                    top_blob = bottom_blob_sliced.clone();
                    if (top_blob.empty())
                        return -100;
                }

                if (_outw == w && _outh == h && _outc / out_elempack == channels)
                {
                    top_blob = bottom_blob;
                    return 0;
                }

                top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<top_blob.c; q++)
                {
                    const Mat m = bottom_blob_sliced.channel(q);
                    Mat borderm = top_blob.channel(q);

                    crop_pack4_neon(m, borderm, _hoffset, _woffset);
                }

                return 0;
            }
        }

    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Crop::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
