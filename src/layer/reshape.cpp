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

#include "reshape.h"

namespace ncnn {

Reshape::Reshape()
{
    one_blob_only = true;
    support_inplace = false;
}

int Reshape::load_param(const ParamDict& pd)
{
    w = pd.get(0, -233);
    h = pd.get(1, -233);
    c = pd.get(2, -233);
    permute = pd.get(3, 0);

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    return 0;
}

int Reshape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

    int dims = bottom_blob.dims;

    // resolve out shape
    int outw = w;
    int outh = h;
    int outc = c;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;

        if (outw == -1)
            outw = total;

        if (dims == 1 && bottom_blob.w == outw)
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

        if (dims == 2 && bottom_blob.h == outh)
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

        if (dims == 3 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }

    bool need_permute = permute == 1;
    if (dims == 2 && ndim == 2 && bottom_blob.h == outh)
        need_permute = false;
    if (dims == 3 && ndim == 3 && bottom_blob.c == outc)
        need_permute = false;

    if (need_permute)
    {
        Mat bottom_blob_permuted = bottom_blob;

        if (dims == 2)
        {
            // hw -> wh
            int _w = bottom_blob.w;
            int _h = bottom_blob.h;

            bottom_blob_permuted.create(_h, _w, elemsize, opt.workspace_allocator);
            if (bottom_blob_permuted.empty())
                return -100;

            const float* ptr = bottom_blob;
            float* outptr = bottom_blob_permuted;

            for (int i = 0; i < _w; i++)
            {
                for (int j = 0; j < _h; j++)
                {
                    outptr[i * _h + j] = ptr[j * _w + i];
                }
            }
        }
        if (dims == 3)
        {
            // chw -> hwc
            int _w = bottom_blob.w;
            int _h = bottom_blob.h;
            int channels = bottom_blob.c;

            bottom_blob_permuted.create(channels, _w, _h, elemsize, opt.workspace_allocator);
            if (bottom_blob_permuted.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < _h; q++)
            {
                float* outptr = bottom_blob_permuted.channel(q);

                for (int i = 0; i < _w; i++)
                {
                    for (int j = 0; j < channels; j++)
                    {
                        const float* ptr = bottom_blob.channel(j).row(q);
                        outptr[i * channels + j] = ptr[i];
                    }
                }
            }
        }

        if (ndim == 1)
        {
            top_blob = bottom_blob_permuted.reshape(outw, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            return 0;
        }

        // permute on nhwc/nhc
        Mat top_blob_permuted;
        if (ndim == 2)
        {
            top_blob_permuted = bottom_blob_permuted.reshape(outh, outw, opt.workspace_allocator);
        }
        if (ndim == 3)
        {
            top_blob_permuted = bottom_blob_permuted.reshape(outc, outw, outh, opt.workspace_allocator);
        }

        if (top_blob_permuted.empty())
            return -100;

        if (ndim == 2)
        {
            // wh -> hw
            top_blob.create(outw, outh, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = top_blob_permuted;
            float* outptr = top_blob;

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    outptr[i * outw + j] = ptr[j * outh + i];
                }
            }
        }
        if (ndim == 3)
        {
            // chw -> hwc
            top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    const float* ptr = top_blob_permuted.channel(i);

                    for (int j = 0; j < outw; j++)
                    {
                        outptr[i * outw + j] = ptr[j * outc + q];
                    }
                }
            }
        }

        return 0;
    }

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

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
