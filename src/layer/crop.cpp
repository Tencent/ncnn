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

namespace ncnn {

Crop::Crop()
{
    one_blob_only = true;
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

    starts = pd.get(9, Mat());
    ends = pd.get(10, Mat());
    axes = pd.get(11, Mat());

    bool numpy_style_slice = !starts.empty() && !ends.empty();

    if (outw == 0 && outh == 0 && outc == 0 && !numpy_style_slice)
    {
        one_blob_only = false;
    }

    return 0;
}

template<typename T>
static void copy_cut_border_image(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src.row<T>(top) + left;
    T* outptr = dst; //.data;

    for (int y = 0; y < h; y++)
    {
        if (w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w * sizeof(T));
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

    int _woffset, _hoffset, _coffset;
    int _outw = -1, _outh = -1, _outc;
    resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);

    if (dims == 1)
    {
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
        if (elemsize == 2)
            copy_cut_border_image<unsigned short>(bottom_blob, top_blob, 0, _woffset);
        if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, 0, _woffset);

        return 0;
    }

    if (dims == 2)
    {
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
        if (elemsize == 2)
            copy_cut_border_image<unsigned short>(bottom_blob, top_blob, _hoffset, _woffset);
        if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, _hoffset, _woffset);

        return 0;
    }

    if (dims == 3)
    {
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
        for (int q = 0; q < _outc; q++)
        {
            const Mat m = bottom_blob_sliced.channel(q);
            Mat borderm = top_blob.channel(q);

            if (elemsize == 1)
                copy_cut_border_image<signed char>(m, borderm, _hoffset, _woffset);
            if (elemsize == 2)
                copy_cut_border_image<unsigned short>(m, borderm, _hoffset, _woffset);
            if (elemsize == 4)
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

    int _woffset, _hoffset, _coffset = -1;
    int _outw = -1, _outh = -1, _outc;
    if (woffset == -233)
    {
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob, _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }

    if (dims == 1)
    {
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
        if (elemsize == 2)
            copy_cut_border_image<unsigned short>(bottom_blob, top_blob, 0, _woffset);
        if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, 0, _woffset);

        return 0;
    }

    if (dims == 2)
    {
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
        if (elemsize == 2)
            copy_cut_border_image<unsigned short>(bottom_blob, top_blob, _hoffset, _woffset);
        if (elemsize == 4)
            copy_cut_border_image<float>(bottom_blob, top_blob, _hoffset, _woffset);

        return 0;
    }

    if (dims == 3)
    {
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
        for (int q = 0; q < _outc; q++)
        {
            const Mat m = bottom_blob_sliced.channel(q);
            Mat borderm = top_blob.channel(q);

            if (elemsize == 1)
                copy_cut_border_image<signed char>(m, borderm, _hoffset, _woffset);
            if (elemsize == 2)
                copy_cut_border_image<unsigned short>(m, borderm, _hoffset, _woffset);
            if (elemsize == 4)
                copy_cut_border_image<float>(m, borderm, _hoffset, _woffset);
        }

        return 0;
    }

    return 0;
}

void Crop::resolve_crop_roi(const Mat& bottom_blob, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    bool numpy_style_slice = !starts.empty() && !ends.empty();
    if (numpy_style_slice)
    {
        _woffset = 0;
        _hoffset = 0;
        _coffset = 0;
        _outw = w;
        _outh = h;
        _outc = channels;

        const int* starts_ptr = starts;
        const int* ends_ptr = ends;
        const int* axes_ptr = axes;

        int _axes[3] = {0, 1, 2};
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
        }
    }
    else
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _coffset = coffset;
        _outw = w;
        _outh = h;
        _outc = channels;

        if (dims == 1)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);
        }
        if (dims == 2)
        {
            if (hoffset == -233)
            {
                _woffset = 0;
                _hoffset = woffset;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);
            }
        }
        if (dims == 3)
        {
            if (hoffset == -233 && coffset == -233)
            {
                _woffset = 0;
                _hoffset = 0;

                _outw = w;
                _outh = h;
                _outc = channels - woffset - woffset2;
                if (outw != -233)
                    _outc = std::min(outw, _outc);
            }
            else if (coffset == -233)
            {
                _woffset = 0;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);

                _outc = channels - hoffset - hoffset2;
                if (outh != -233)
                    _outc = std::min(outh, _outc);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);

                _outc = channels - coffset2 - coffset2;
                if (outc != -233)
                    _outc = std::min(outc, _outc);
            }
        }
    }
}

void Crop::resolve_crop_roi(const Mat& bottom_blob, const Mat& reference_blob, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    int ref_w = reference_blob.w;
    int ref_h = reference_blob.h;
    int ref_channels = reference_blob.c;
    int ref_dims = reference_blob.dims;

    if (dims == 1)
    {
        _woffset = woffset;
        _outw = ref_w;
    }
    if (dims == 2)
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _outw = ref_w;
        _outh = ref_h;
    }
    if (dims == 3)
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _coffset = coffset;
        _outw = ref_w;
        _outh = ref_h;
        _outc = ref_dims == 3 ? ref_channels : channels;
    }
}

void Crop::resolve_crop_roi(const Mat& bottom_blob, const int* param_data, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        _woffset = param_data[0];
        _outw = param_data[3];
    }
    if (dims == 2)
    {
        _woffset = param_data[0];
        _hoffset = param_data[1];
        _outw = param_data[3];
        _outh = param_data[4];
    }
    if (dims == 3)
    {
        _woffset = param_data[0];
        _hoffset = param_data[1];
        _coffset = param_data[2];
        _outw = param_data[3];
        _outh = param_data[4];
        _outc = param_data[5];
    }
}

} // namespace ncnn
