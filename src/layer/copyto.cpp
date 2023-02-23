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

    starts = pd.get(9, Mat());
    axes = pd.get(11, Mat());

    return 0;
}

template<typename T>
static void copy_to_image(const Mat& src, Mat& self, int top, int left)
{
    int w = src.w;
    int h = src.h;

    const T* ptr = src;
    T* outptr = self.row<T>(top) + left;

    for (int y = 0; y < h; y++)
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

    if (src_blob.dims == dims && src_blob.w == w && src_blob.h == h && src_blob.d == d && src_blob.c == channels)
    {
        top_blob = src_blob;
        return 0;
    }

    top_blob = self_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int _woffset, _hoffset, _doffset, _coffset;
    resolve_copyto_offset(self_blob.shape(), _woffset, _hoffset, _doffset, _coffset);

    if (dims == 1)
    {
        if (elemsize == 1)
            copy_to_image<signed char>(src_blob, top_blob, 0, _woffset);
        if (elemsize == 2)
            copy_to_image<unsigned short>(src_blob, top_blob, 0, _woffset);
        if (elemsize == 4)
            copy_to_image<float>(src_blob, top_blob, 0, _woffset);
    }

    if (dims == 2)
    {
        if (elemsize == 1)
            copy_to_image<signed char>(src_blob, top_blob, _hoffset, _woffset);
        if (elemsize == 2)
            copy_to_image<unsigned short>(src_blob, top_blob, _hoffset, _woffset);
        if (elemsize == 4)
            copy_to_image<float>(src_blob, top_blob, _hoffset, _woffset);
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < src_blob.c; q++)
        {
            const Mat roim = src_blob.channel(q);
            Mat m = top_blob.channel(q + _coffset);

            if (elemsize == 1)
                copy_to_image<signed char>(roim, m, _hoffset, _woffset);
            if (elemsize == 2)
                copy_to_image<unsigned short>(roim, m, _hoffset, _woffset);
            if (elemsize == 4)
                copy_to_image<float>(roim, m, _hoffset, _woffset);
        }
    }

    if (dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < src_blob.c; q++)
        {
            for (int z = 0; z < src_blob.d; z++)
            {
                const Mat roim = src_blob.channel(q).depth(z);
                Mat m = top_blob.channel(q + _coffset).depth(z + _doffset);

                if (elemsize == 1)
                    copy_to_image<signed char>(roim, m, _hoffset, _woffset);
                if (elemsize == 2)
                    copy_to_image<unsigned short>(roim, m, _hoffset, _woffset);
                if (elemsize == 4)
                    copy_to_image<float>(roim, m, _hoffset, _woffset);
            }
        }
    }

    return 0;
}

void CopyTo::resolve_copyto_offset(const Mat& self_blob, int& _woffset, int& _hoffset, int& _doffset, int& _coffset) const
{
    int w = self_blob.w;
    int h = self_blob.h;
    int d = self_blob.d;
    int channels = self_blob.c;
    int dims = self_blob.dims;

    bool numpy_style_slice = !starts.empty();
    if (numpy_style_slice)
    {
        _woffset = 0;
        _hoffset = 0;
        _doffset = 0;
        _coffset = 0;

        const int* starts_ptr = starts;
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

            if (dims == 1) // axis == 0
            {
                if (start == -233) start = 0;
                _woffset = start >= 0 ? start : w + start;
            }
            if (dims == 2)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    _hoffset = start >= 0 ? start : h + start;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    _woffset = start >= 0 ? start : w + start;
                }
            }
            if (dims == 3)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    _coffset = start >= 0 ? start : channels + start;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    _hoffset = start >= 0 ? start : h + start;
                }
                if (axis == 2)
                {
                    if (start == -233) start = 0;
                    _woffset = start >= 0 ? start : w + start;
                }
            }
            if (dims == 4)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    _coffset = start >= 0 ? start : channels + start;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    _doffset = start >= 0 ? start : d + start;
                }
                if (axis == 2)
                {
                    if (start == -233) start = 0;
                    _hoffset = start >= 0 ? start : h + start;
                }
                if (axis == 3)
                {
                    if (start == -233) start = 0;
                    _woffset = start >= 0 ? start : w + start;
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
    }
}

} // namespace ncnn
