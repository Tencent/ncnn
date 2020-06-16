// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
#include <algorithm>
#if __AVX__
#include <immintrin.h>
#endif
#include "pooling_x86.h"
#include <float.h>

namespace ncnn {

#if __AVX__
#include "pooling_2x2.h"
#endif

DEFINE_LAYER_CREATOR(Pooling_x86)

Pooling_x86::Pooling_x86() {}

int Pooling_x86::forward(const Mat &bottom_blob, Mat &top_blob,
                         const Option &opt) const {

    // max value in NxN window
    // avg value in NxN window
#if __AVX__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d
    //     stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom,
    //     kernel_w, kernel_h, stride_w, stride_h);

    if (kernel_w != kernel_h || stride_w != stride_h) {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (pooling_type != PoolMethod_MAX || stride != 2 || global_pooling == 1) {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    if (kernel_size != 2) {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;
    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (kernel_size == 2)
        pooling2x2s2_max_avx(bottom_blob_bordered, top_blob, opt);

    return 0;
#else
    return Pooling::forward(bottom_blob, top_blob, opt);
#endif
}

} // namespace ncnn
