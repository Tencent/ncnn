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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#include "benchmark.h"

#if NCNN_BENCHMARK
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/deconvolution.h"
#include "layer/deconvolutiondepthwise.h"

#include <stdio.h>
#endif // NCNN_BENCHMARK

namespace ncnn {

double get_current_time()
{
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
}

#if NCNN_BENCHMARK

void benchmark(const Layer* layer, double start, double end)
{
    fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);
    fprintf(stderr, "    |");
    fprintf(stderr, "\n");
}

void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, double start, double end)
{
    fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type.c_str(), layer->name.c_str(), end - start);

    char in_shape_str[64] = {'\0'};
    char out_shape_str[64] = {'\0'};

    if (bottom_blob.dims == 1)
    {
        sprintf(in_shape_str, "[%3d *%d]", bottom_blob.w, bottom_blob.elempack);
    }
    if (bottom_blob.dims == 2)
    {
        sprintf(in_shape_str, "[%3d, %3d *%d]", bottom_blob.w, bottom_blob.h, bottom_blob.elempack);
    }
    if (bottom_blob.dims == 3)
    {
        sprintf(in_shape_str, "[%3d, %3d, %3d *%d]", bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack);
    }

    if (top_blob.dims == 1)
    {
        sprintf(out_shape_str, "[%3d *%d]", top_blob.w, top_blob.elempack);
    }
    if (top_blob.dims == 2)
    {
        sprintf(out_shape_str, "[%3d, %3d *%d]", top_blob.w, top_blob.h, top_blob.elempack);
    }
    if (top_blob.dims == 3)
    {
        sprintf(out_shape_str, "[%3d, %3d, %3d *%d]", top_blob.w, top_blob.h, top_blob.c, top_blob.elempack);
    }

    fprintf(stderr, "    | %22s -> %-22s", in_shape_str, out_shape_str);

    if (layer->type == "Convolution")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                ((Convolution*)layer)->kernel_w,
                ((Convolution*)layer)->kernel_h,
                ((Convolution*)layer)->stride_w,
                ((Convolution*)layer)->stride_h);
    }
    else if (layer->type == "ConvolutionDepthWise")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                ((ConvolutionDepthWise*)layer)->kernel_w,
                ((ConvolutionDepthWise*)layer)->kernel_h,
                ((ConvolutionDepthWise*)layer)->stride_w,
                ((ConvolutionDepthWise*)layer)->stride_h);
    }
    else if (layer->type == "Deconvolution")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                ((Deconvolution*)layer)->kernel_w,
                ((Deconvolution*)layer)->kernel_h,
                ((Deconvolution*)layer)->stride_w,
                ((Deconvolution*)layer)->stride_h);
    }
    else if (layer->type == "DeconvolutionDepthWise")
    {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d",
                ((DeconvolutionDepthWise*)layer)->kernel_w,
                ((DeconvolutionDepthWise*)layer)->kernel_h,
                ((DeconvolutionDepthWise*)layer)->stride_w,
                ((DeconvolutionDepthWise*)layer)->stride_h);
    }
    fprintf(stderr, "\n");
}

#endif // NCNN_BENCHMARK

} // namespace ncnn
