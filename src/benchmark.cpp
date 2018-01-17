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

#include "benchmark.h"

namespace ncnn {

void benchmark(const ncnn::Layer* layer, clock_t begin, clock_t end)
{
    fprintf(stderr, "%-24s %-24s %8.2lfms", layer->type.c_str(), layer->name.c_str(), (end - begin) * 1.0 / CLOCKS_PER_SEC * 1000);
    fprintf(stderr, "    |");
    fprintf(stderr, "\n");
}

void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, clock_t begin, clock_t end)
{
    fprintf(stderr, "%-24s %-24s %8.2lfms", layer->type.c_str(), layer->name.c_str(), (end - begin) * 1.0 / CLOCKS_PER_SEC * 1000);
    fprintf(stderr, "    |    feature_map: %4d x %-4d    inch: %4d    outch: %4d", bottom_blob.w, bottom_blob.h, bottom_blob.c, top_blob.c);
    if (layer->type == "Convolution")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d", 
        ((Convolution*)layer)->kernel_h, 
        ((Convolution*)layer)->kernel_w,
        ((Convolution*)layer)->stride_h,
        ((Convolution*)layer)->stride_w
        );
    }
    fprintf(stderr, "\n");
}

}