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
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Softmax)

Softmax::Softmax()
{
    one_blob_only = true;
    support_inplace = false;
}

int Softmax::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Softmax::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    if (axis == 0)
    {
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

    }
    else if (axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels);
        if (top_blob.empty())
            return -100;

        Mat max;
        max.create(h, channels);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i=0; i<h; i++)
            {
                float max = -FLT_MAX;
                for (int j=0; j<w; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                maxptr[i] = max;
                ptr += w;
            }
        }

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i=0; i<h; i++)
            {
                float max = maxptr[i];
                for (int j=0; j<w; j++)
                {
                    outptr[j] = exp(ptr[j] - max);
                }

                ptr += w;
                outptr += w;
            }
        }

        Mat sum;
        sum.create(h, channels);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* outptr = top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i=0; i<h; i++)
            {
                float sum = 0.f;
                for (int j=0; j<w; j++)
                {
                    sum += outptr[j];
                }

                sumptr[i] = sum;
                outptr += w;
            }
        }

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* outptr = top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i=0; i<h; i++)
            {
                float sum = sumptr[i];
                for (int j=0; j<w; j++)
                {
                    outptr[j] /= sum;
                }

                outptr += w;
            }
        }

    }
    else if (axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels);
        if (top_blob.empty())
            return -100;

        Mat max;
        max.create(w, channels);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i=0; i<h; i++)
            {
                for (int j=0; j<w; j++)
                {
                    maxptr[j] = std::max(maxptr[j], ptr[j]);
                }

                ptr += w;
            }
        }

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i=0; i<h; i++)
            {
                for (int j=0; j<w; j++)
                {
                    outptr[j] = exp(ptr[j] - maxptr[j]);
                }

                ptr += w;
                outptr += w;
            }
        }

        Mat sum;
        sum.create(w, channels);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* outptr = top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i=0; i<h; i++)
            {
                for (int j=0; j<w; j++)
                {
                    sumptr[j] += outptr[j];
                }

                outptr += w;
            }
        }

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* outptr = top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i=0; i<h; i++)
            {
                for (int j=0; j<w; j++)
                {
                    outptr[j] /= sumptr[j];
                }

                outptr += w;
            }
        }

    }

    return 0;
}

} // namespace ncnn
