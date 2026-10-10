// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax.h"

#include <float.h>

namespace ncnn {

Softmax::Softmax()
{
    one_blob_only = true;
    support_inplace = true;
}

int Softmax::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    // the original softmax handle axis on 3-dim blob incorrectly
    // ask user to regenerate param instead of producing wrong result
    int fixbug0 = pd.get(1, 0);
    if (fixbug0 == 0 && axis != 0)
    {
        NCNN_LOGE("param is too old, please regenerate!");
        return -1;
    }

    return 0;
}

static void softmax(float* _ptr, int size)
{
    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            max = std::max(max, *ptr++);
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            *ptr = expf(*ptr - max);
            sum += *ptr;
            ptr++;
        }
    }

    // div sum
    {
        float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            *ptr++ /= sum;
        }
    }
}

static void softmax(float* _ptr, int size, int stride)
{
    // reduce max
    float max = -FLT_MAX;
    {
        const float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            max = std::max(max, *ptr);
            ptr += stride;
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            *ptr = expf(*ptr - max);
            sum += *ptr;
            ptr += stride;
        }
    }

    // div sum
    {
        float* ptr = _ptr;
        for (int i = 0; i < size; i++)
        {
            *ptr /= sum;
            ptr += stride;
        }
    }
}

int Softmax::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // value = expf( value - global max value )
    // sum all value
    // value = value / sum

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        float* ptr = bottom_top_blob;

        softmax(ptr, w);
    }

    if (dims == 2 && positive_axis == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, h, w);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            softmax(ptr, w);
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, channels, bottom_top_blob.cstep);
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < w; i++)
            {
                softmax(ptr, h, w);
                ptr += 1;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                softmax(ptr, w);
                ptr += w;
            }
        }
    }

    if (dims == 4 && positive_axis == 0)
    {
        const int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            float* ptr = (float*)bottom_top_blob + i;

            softmax(ptr, channels, bottom_top_blob.cstep);
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < w * h; i++)
            {
                softmax(ptr, d, w * h);
                ptr += 1;
            }
        }
    }

    if (dims == 4 && positive_axis == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                float* ptr = bottom_top_blob.channel(q).depth(i);

                for (int j = 0; j < w; j++)
                {
                    softmax(ptr, h, w);
                    ptr += 1;
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax(ptr, w);
                    ptr += w;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
