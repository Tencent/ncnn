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

#include "batchnorm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm)

BatchNorm::BatchNorm()
{
    one_blob_only = true;
    support_inplace = true;
}

BatchNorm::~BatchNorm()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int BatchNorm::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d", &channels);
    if (nscan != 1)
    {
        fprintf(stderr, "BatchNorm load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int BatchNorm::load_param_bin(FILE* paramfp)
{
    fread(&channels, sizeof(int), 1, paramfp);

    return 0;
}

int BatchNorm::load_model(FILE* binfp)
{
    int nread;

    slope_data.create(channels);
    if (slope_data.empty())
        return -100;
    nread = fread(slope_data, channels * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "BatchNorm read slope_data failed %d\n", nread);
        return -1;
    }

    mean_data.create(channels);
    if (mean_data.empty())
        return -100;
    nread = fread(mean_data, channels * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "BatchNorm read mean_data failed %d\n", nread);
        return -1;
    }

    var_data.create(channels);
    if (var_data.empty())
        return -100;
    nread = fread(var_data, channels * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "BatchNorm read var_data failed %d\n", nread);
        return -1;
    }

    bias_data.create(channels);
    if (bias_data.empty())
        return -100;
    nread = fread(bias_data, channels * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "BatchNorm read bias_data failed %d\n", nread);
        return -1;
    }

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;
    const float* slope_data_ptr = slope_data;
    const float* mean_data_ptr = mean_data;
    const float* var_data_ptr = var_data;
    const float* bias_data_ptr = bias_data;
    float* a_data_ptr = a_data;
    float* b_data_ptr = b_data;
    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data_ptr[i]);
        a_data_ptr[i] = bias_data_ptr[i] - slope_data_ptr[i] * mean_data_ptr[i] / sqrt_var;
        b_data_ptr[i] = slope_data_ptr[i] / sqrt_var;
    }

    return 0;
}
#endif // NCNN_STDIO

int BatchNorm::load_param(const unsigned char*& mem)
{
    channels = *(int*)(mem);
    mem += 4;

    return 0;
}

int BatchNorm::load_model(const unsigned char*& mem)
{
    slope_data = Mat(channels, (float*)mem);
    mem += channels * sizeof(float);

    mean_data = Mat(channels, (float*)mem);
    mem += channels * sizeof(float);

    var_data = Mat(channels, (float*)mem);
    mem += channels * sizeof(float);

    bias_data = Mat(channels, (float*)mem);
    mem += channels * sizeof(float);

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;
    const float* slope_data_ptr = slope_data;
    const float* mean_data_ptr = mean_data;
    const float* var_data_ptr = var_data;
    const float* bias_data_ptr = bias_data;
    float* a_data_ptr = a_data;
    float* b_data_ptr = b_data;
    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data_ptr[i]);
        a_data_ptr[i] = bias_data_ptr[i] - slope_data_ptr[i] * mean_data_ptr[i] / sqrt_var;
        b_data_ptr[i] = slope_data_ptr[i] / sqrt_var;
    }

    return 0;
}

int BatchNorm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    const float* a_data_ptr = a_data;
    const float* b_data_ptr = b_data;
    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

        for (int i=0; i<size; i++)
        {
            outptr[i] = b * ptr[i] + a;
        }
    }

    return 0;
}

int BatchNorm::forward_inplace(Mat& bottom_top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    const float* a_data_ptr = a_data;
    const float* b_data_ptr = b_data;
    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

        for (int i=0; i<size; i++)
        {
            ptr[i] = b * ptr[i] + a;
        }
    }

    return 0;
}

} // namespace ncnn
