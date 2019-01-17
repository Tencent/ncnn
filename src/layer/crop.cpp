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

DEFINE_LAYER_CREATOR(Crop)

Crop::Crop()
{
}

int Crop::load_param(const ParamDict& pd)
{
    woffset = pd.get(0, 0);
    hoffset = pd.get(1, 0);
    coffset = pd.get(2, 0);
    outw = pd.get(3, 0);
    outh = pd.get(4, 0);
    outc = pd.get(5, 0);

    if (outw != 0 || outh != 0 || outc != 0)
    {
        one_blob_only = true;
    }

    return 0;
}

int Crop::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int _outw = outw == -233 ? w - woffset : outw;
    int _outh = outh == -233 ? h - hoffset : outh;
    int _outc = outc == -233 ? channels - coffset : outc;

    const Mat bottom_blob_sliced(w, h, _outc, (void*)(const float*)bottom_blob.channel(coffset));

    int top = hoffset;
    int bottom = h - _outh - hoffset;
    int left = woffset;
    int right = w - _outw - woffset;

    copy_cut_border(bottom_blob_sliced, top_blob, top, bottom, left, right, opt.blob_allocator, opt.num_threads);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Crop::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int _outw = reference_blob.w;
    int _outh = reference_blob.h;
    int _outc = reference_blob.dims == 3 ? reference_blob.c : channels;

    const Mat bottom_blob_sliced(w, h, _outc, (void*)(const float*)bottom_blob.channel(coffset));

    int top = hoffset;
    int bottom = h - _outh - hoffset;
    int left = woffset;
    int right = w - _outw - woffset;

    Mat& top_blob = top_blobs[0];

    copy_cut_border(bottom_blob_sliced, top_blob, top, bottom, left, right, opt.blob_allocator, opt.num_threads);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
