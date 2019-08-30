// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "crop.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Crop)

Crop::Crop()
{
    one_blob_only = false;
    support_inplace = false;
}

int Crop::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);
    coffset = pd.get(2, 0);
    outw = pd.get(3, 0);
    outh = pd.get(4, 0);
    outc = pd.get(5, 0);
    woffset2 = pd.get(6, 0);
    hoffset2 = pd.get(7, 0);
    coffset2 = pd.get(8, 0);

    if (outw != 0 || outh != 0 || outc != 0)
    {
        one_blob_only = true;
    }

    return 0;
}

template<typename T>
static void copy_cut_border_image(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src.row<T>(top) + left;
    T* outptr = dst;//.data;

    for (int y = 0; y < h; y++)
    {
        if(w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w*sizeof(T));
        }
        outptr += w;
        ptr += src.w;
    }
}

int Crop::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int _woffset = woffset;
    int _hoffset = hoffset;
    int _coffset = coffset;
    int _woffset2 = woffset2;
    int _hoffset2 = hoffset2;
    int _coffset2 = coffset2;
    int _outw;
    int _outh;
    int _outc;

    if (dims == 1)
    {
        if (outw == -233)
            _outw = w - _woffset - _woffset2;
        else
            _outw = std::min(outw, w - _woffset - _woffset2);

        if (_outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_cut_border_image<signed char>(bottom_blob, top_blob, 0, _woffset);
        else if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, 0, _woffset);

        return 0;
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
                _outh = h - _hoffset - _hoffset2;
            else
                _outh = std::min(outw, h - _hoffset - _hoffset2);
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
        }

        if (_outw == w && _outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_outw, _outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_cut_border_image<signed char>(bottom_blob, top_blob, _hoffset, _woffset);
        else if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, _hoffset, _woffset);

        return 0;
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
                _outc = channels - _coffset - _coffset2;
            else
                _outc = std::min(outw, channels - _coffset - _coffset2);
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
                _outc = channels - _coffset - _coffset2;
            else
                _outc = std::min(outh, channels - _coffset - _coffset2);
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
                _outc = channels - _coffset - _coffset2;
            else
                _outc = std::min(outc, channels - _coffset - _coffset2);
        }

        if (_outw == w && _outh == h && _outc == channels)
        {
            top_blob = bottom_blob;
            return 0;
        }

        const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);

        if (_outw == w && _outh == h)
        {
            top_blob = bottom_blob_sliced.clone();
            if (top_blob.empty())
                return -100;

            return 0;
        }

        top_blob.create(_outw, _outh, _outc, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<_outc; q++)
        {
            const Mat m = bottom_blob_sliced.channel(q);
            Mat borderm = top_blob.channel(q);

            if (elemsize == 1)
                copy_cut_border_image<signed char>(m, borderm, _hoffset, _woffset);
            else if (elemsize == 4)
                copy_cut_border_image<float>(m, borderm, _hoffset, _woffset);
        }

        return 0;
    }

    return 0;
}

int Crop::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    Mat& top_blob = top_blobs[0];

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
            _outw = reference_blob.w;
        }

        if (_outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_cut_border_image<signed char>(bottom_blob, top_blob, 0, _woffset);
        else if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, 0, _woffset);

        return 0;
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
            _outw = reference_blob.w;
            _outh = reference_blob.h;
        }

        if (_outw == w && _outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_outw, _outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_cut_border_image<signed char>(bottom_blob, top_blob, _hoffset, _woffset);
        else if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, _hoffset, _woffset);

        return 0;
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
            _outw = reference_blob.w;
            _outh = reference_blob.h;
            _outc = reference_blob.dims == 3 ? reference_blob.c : channels;
        }

        if (_outw == w && _outh == h && _outc == channels)
        {
            top_blob = bottom_blob;
            return 0;
        }

        const Mat bottom_blob_sliced = bottom_blob.channel_range(_coffset, _outc);

        if (_outw == w && _outh == h)
        {
            top_blob = bottom_blob_sliced.clone();
            if (top_blob.empty())
                return -100;

            return 0;
        }

        top_blob.create(_outw, _outh, _outc, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<_outc; q++)
        {
            const Mat m = bottom_blob_sliced.channel(q);
            Mat borderm = top_blob.channel(q);

            if (elemsize == 1)
                copy_cut_border_image<signed char>(m, borderm, _hoffset, _woffset);
            else if (elemsize == 4)
                copy_cut_border_image<float>(m, borderm, _hoffset, _woffset);
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
