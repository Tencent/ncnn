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

#include "concat.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Concat)

Concat::Concat()
{
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    // total channels
    int top_channels = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];
        top_channels += bottom_blob.c;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, top_channels);
    if (top_blob.empty())
        return -100;

    int q = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob = bottom_blobs[b];

        int channels = bottom_blob.c;
        int size = bottom_blob.cstep * channels;

        const float* ptr = bottom_blob;
        float* outptr = top_blob.channel(q);
        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i];
        }

        q += channels;
    }

    return 0;
}

} // namespace ncnn
