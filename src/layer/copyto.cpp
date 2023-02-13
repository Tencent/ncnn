// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "copyto.h"

namespace ncnn {

CopyTo::CopyTo()
{
    one_blob_only = false;
    support_inplace = false;
}

int CopyTo::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);
    doffset = pd.get(13, 0);
    coffset = pd.get(2, 0);
    outw = pd.get(3, 0);
    outh = pd.get(4, 0);
    outd = pd.get(14, 0);
    outc = pd.get(5, 0);
    woffset2 = pd.get(6, 0);
    hoffset2 = pd.get(7, 0);
    doffset2 = pd.get(15, 0);
    coffset2 = pd.get(8, 0);

    starts = pd.get(9, Mat());
    ends = pd.get(10, Mat());
    axes = pd.get(11, Mat());

    return 0;
}

template<typename T>
static void copy_to_image(const Mat& src, Mat& self, int x, int y, int w, int h)
{
    const T* ptr = src;
    T* outptr = self.row<T>(y) + x;

    for (int i = 0; i < h; i++)
    {
        memcpy(outptr, ptr, w * sizeof(T));
        ptr += w;
        outptr += self.w;
    }
}

int CopyTo::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& self_blob = bottom_blobs[0];
    const Mat& src_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int w = self_blob.w;
    int h = self_blob.h;
    int d = self_blob.d;
    int channels = self_blob.c;
    int dims = self_blob.dims;
    size_t elemsize = self_blob.elemsize;

    int _woffset, _hoffset, _doffset, _coffset;
    int _outw = -1, _outh = -1, _outd = -1, _outc;
    resolve_copyto_roi(self_blob.shape(), _woffset, _hoffset, _doffset, _coffset, _outw, _outh, _outd, _outc);

    if (dims == 1)
    {
        if (_outw == w)
        {
            top_blob = src_blob;
            return 0;
        }

        top_blob = self_blob.clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_to_image<signed char>(src_blob, top_blob, _woffset, 0, _outw, 0);
        if (elemsize == 2)
            copy_to_image<unsigned short>(src_blob, top_blob, _woffset, 0, _outw, 0);
        if (elemsize == 4)
            copy_to_image<float>(src_blob, top_blob, _woffset, 0, _outw, 0);

        return 0;
    }

    if (dims == 2)
    {
        if (_outw == w && _outh == h)
        {
            top_blob = src_blob;
            return 0;
        }

        top_blob = self_blob.clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_to_image<signed char>(src_blob, top_blob, _woffset, _hoffset, _outw, _outh);
        if (elemsize == 2)
            copy_to_image<unsigned short>(src_blob, top_blob, _woffset, _hoffset, _outw, _outh);
        if (elemsize == 4)
            copy_to_image<float>(src_blob, top_blob, _woffset, _hoffset, _outw, _outh);

        return 0;
    }

    if (dims == 3)
    {
        if (_outw == w && _outh == h && _outc == channels)
        {
            top_blob = src_blob;
            return 0;
        }

        top_blob = self_blob.clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < _outc; q++)
        {
            const Mat roim = src_blob.channel(q);
            Mat m = top_blob.channel(q + _coffset);

            if (elemsize == 1)
                copy_to_image<signed char>(roim, m, _woffset, _hoffset, _outw, _outh);
            if (elemsize == 2)
                copy_to_image<unsigned short>(roim, m, _woffset, _hoffset, _outw, _outh);
            if (elemsize == 4)
                copy_to_image<float>(roim, m, _woffset, _hoffset, _outw, _outh);
        }

        return 0;
    }

    if (dims == 4)
    {
        if (_outw == w && _outh == h && _outd == d && _outc == channels)
        {
            top_blob = src_blob;
            return 0;
        }

        top_blob = self_blob.clone(opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < _outc; q++)
        {
            for (int z = 0; z < _outd; z++)
            {
                const Mat roim = src_blob.channel(q).depth(z);
                Mat m = top_blob.channel(q + _coffset).depth(z + _doffset);

                if (elemsize == 1)
                    copy_to_image<signed char>(roim, m, _woffset, _hoffset, _outw, _outh);
                if (elemsize == 2)
                    copy_to_image<unsigned short>(roim, m, _woffset, _hoffset, _outw, _outh);
                if (elemsize == 4)
                    copy_to_image<float>(roim, m, _woffset, _hoffset, _outw, _outh);
            }
        }

        return 0;
    }

    return 0;
}

void CopyTo::resolve_copyto_roi(const Mat& bottom_blob, int& _woffset, int& _hoffset, int& _doffset, int& _coffset, int& _outw, int& _outh, int& _outd, int& _outc) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    bool numpy_style_slice = !starts.empty() && !ends.empty();
    if (numpy_style_slice)
    {
        _woffset = 0;
        _hoffset = 0;
        _doffset = 0;
        _coffset = 0;
        _outw = w;
        _outh = h;
        _outd = d;
        _outc = channels;

        const int* starts_ptr = starts;
        const int* ends_ptr = ends;
        const int* axes_ptr = axes;

        int _axes[4] = {0, 1, 2, 3};
        int num_axis = axes.w;
        if (num_axis == 0)
        {
            num_axis = dims;
        }
        else
        {
            for (int i = 0; i < num_axis; i++)
            {
                int axis = axes_ptr[i];
                if (axis < 0)
                    axis = dims + axis;
                _axes[i] = axis;
            }
        }

        for (int i = 0; i < num_axis; i++)
        {
            int axis = _axes[i];
            int start = starts_ptr[i];
            int end = ends_ptr[i];

            if (dims == 1) // axis == 0
            {
                if (start == -233) start = 0;
                if (end == -233) end = w;
                _woffset = start >= 0 ? start : w + start;
                _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
            }
            if (dims == 2)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = h;
                    _hoffset = start >= 0 ? start : h + start;
                    _outh = std::min(h, end > 0 ? end : h + end) - _hoffset;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = w;
                    _woffset = start >= 0 ? start : w + start;
                    _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
                }
            }
            if (dims == 3)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = channels;
                    _coffset = start >= 0 ? start : channels + start;
                    _outc = std::min(channels, end > 0 ? end : channels + end) - _coffset;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = h;
                    _hoffset = start >= 0 ? start : h + start;
                    _outh = std::min(h, end > 0 ? end : h + end) - _hoffset;
                }
                if (axis == 2)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = w;
                    _woffset = start >= 0 ? start : w + start;
                    _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
                }
            }
            if (dims == 4)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = channels;
                    _coffset = start >= 0 ? start : channels + start;
                    _outc = std::min(channels, end > 0 ? end : channels + end) - _coffset;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = d;
                    _doffset = start >= 0 ? start : d + start;
                    _outd = std::min(d, end > 0 ? end : d + end) - _doffset;
                }
                if (axis == 2)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = h;
                    _hoffset = start >= 0 ? start : h + start;
                    _outh = std::min(h, end > 0 ? end : h + end) - _hoffset;
                }
                if (axis == 3)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = w;
                    _woffset = start >= 0 ? start : w + start;
                    _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
                }
            }
        }
    }
    else
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _doffset = doffset;
        _coffset = coffset;
        _outw = w;
        _outh = h;
        _outd = d;
        _outc = channels;

        if (dims == 1)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);
        }
        if (dims == 2)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);

            _outh = h - hoffset - hoffset2;
            if (outh != -233)
                _outh = std::min(outh, _outh);
        }
        if (dims == 3)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);

            _outh = h - hoffset - hoffset2;
            if (outh != -233)
                _outh = std::min(outh, _outh);

            _outc = channels - coffset - coffset2;
            if (outc != -233)
                _outc = std::min(outc, _outc);
        }
        if (dims == 4)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);

            _outh = h - hoffset - hoffset2;
            if (outh != -233)
                _outh = std::min(outh, _outh);

            _outd = d - doffset - doffset2;
            if (outd != -233)
                _outd = std::min(outd, _outd);

            _outc = channels - coffset - coffset2;
            if (outc != -233)
                _outc = std::min(outc, _outc);
        }
    }
}

} // namespace ncnn
