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

#include "softmax.h"
#include <float.h>
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Softmax)

Softmax::Softmax()
{
    one_blob_only = true;
    support_inplace = true;
}

int Softmax::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    Mat max;
    max.create(w, h);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            outptr[i] = exp(ptr[i] - maxptr[i]);
        }
    }

    Mat sum;
    sum.create(w, h);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q=0; q<channels; q++)
    {
        const float* outptr = top_blob.channel(q);
        float* sumptr = sum;

        for (int i=0; i<size; i++)
        {
            sumptr[i] += outptr[i];
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* outptr = top_blob.channel(q);
        float* sumptr = sum;

        for (int i=0; i<size; i++)
        {
            outptr[i] /= sumptr[i];
        }
    }

    return 0;
}

int Softmax::forward_inplace(Mat& bottom_top_blob) const
{
    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    Mat max;
    max.create(w, h);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            ptr[i] = exp(ptr[i] - maxptr[i]);
        }
    }

    Mat sum;
    sum.create(w, h);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

        for (int i=0; i<size; i++)
        {
            sumptr[i] += ptr[i];
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

        for (int i=0; i<size; i++)
        {
            ptr[i] /= sumptr[i];
        }
    }

    return 0;
}

} // namespace ncnn
