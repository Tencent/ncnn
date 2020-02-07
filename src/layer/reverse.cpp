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
#include "reverse.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Reverse)

Reverse::Reverse()
{
    one_blob_only = true;
    support_inplace = false;
}


int Reverse::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels= bottom_blob.c;


    top_blob.create(w,h,channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q=0; q<channels; q++)//8
    {
        const float* ptr = bottom_blob.channel(q);
        float* target = top_blob.channel(channels-q-1);
        for (int i = 0; i < h; i++)//1
        {
            for (int j = 0; i < w; i++)//256
            {
               target[j+h*i] = ptr[j+h*i];
            }
        }
    }

    
    return 0;
}

} // namespace ncnn
