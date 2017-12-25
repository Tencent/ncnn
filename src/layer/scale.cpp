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

#include "scale.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Scale)

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

    return 0;
}

#if NCNN_STDIO
int Scale::load_model(FILE* binfp)
{
    int nread;

    if (scale_data_size != -233)
    {
        scale_data.create(scale_data_size);
        if (scale_data.empty())
            return -100;
        nread = fread(scale_data, scale_data_size * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Scale read scale_data failed %d\n", nread);
            return -1;
        }
    }

    if (bias_term)
    {
        bias_data.create(scale_data_size);
        if (bias_data.empty())
            return -100;
        nread = fread(bias_data, scale_data_size * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            fprintf(stderr, "Scale read bias_data failed %d\n", nread);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STDIO

int Scale::load_model(const unsigned char*& mem)
{
    if (scale_data_size != -233)
    {
        scale_data = Mat(scale_data_size, (float*)mem);
        mem += scale_data_size * sizeof(float);
    }

    if (bias_term)
    {
        bias_data = Mat(scale_data_size, (float*)mem);
        mem += scale_data_size * sizeof(float);
    }

    return 0;
}

int Scale::forward_inplace(std::vector<Mat>& bottom_top_blobs) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* bias_ptr = bias_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_blob.channel(q)[0];
            float bias = bias_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * s + bias;
            }
        }
    }
    else
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_blob.channel(q)[0];

            for (int i=0; i<size; i++)
            {
                ptr[i] *= s;
            }
        }
    }

    return 0;
}

int Scale::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* scale_ptr = scale_data;
        const float* bias_ptr = bias_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];
            float bias = bias_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * s + bias;
            }
        }
    }
    else
    {
        const float* scale_ptr = scale_data;
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] *= s;
            }
        }
    }

    return 0;
}

} // namespace ncnn
