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

#include "prelu.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(PReLU)

PReLU::PReLU()
{
    one_blob_only = true;
    support_inplace = true;
}

#if NCNN_STDIO
#if NCNN_STRING
int PReLU::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d", &num_slope);
    if (nscan != 1)
    {
        fprintf(stderr, "PReLU load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int PReLU::load_param_bin(FILE* paramfp)
{
    fread(&num_slope, sizeof(int), 1, paramfp);

    return 0;
}

int PReLU::load_model(FILE* binfp)
{
    int nread;

    slope_data.create(num_slope);
    if (slope_data.empty())
        return -100;
    nread = fread(slope_data, num_slope * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "PReLU read slope_data failed %d\n", nread);
        return -1;
    }

    return 0;
}
#endif // NCNN_STDIO

int PReLU::load_param(const unsigned char*& mem)
{
    num_slope = *(int*)(mem);
    mem += 4;

    return 0;
}

int PReLU::load_model(const unsigned char*& mem)
{
    slope_data = Mat(num_slope, (float*)mem);
    mem += num_slope * sizeof(float);

    return 0;
}

int PReLU::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

        for (int i=0; i<size; i++)
        {
            if (ptr[i] < 0)
                outptr[i] = ptr[i] * slope;
            else
                outptr[i] = ptr[i];
        }
    }

    return 0;
}

int PReLU::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

        for (int i=0; i<size; i++)
        {
            if (ptr[i] < 0)
                ptr[i] *= slope;
        }
    }

    return 0;
}

} // namespace ncnn
