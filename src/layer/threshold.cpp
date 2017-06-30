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

#include "threshold.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Threshold)

Threshold::Threshold()
{
    one_blob_only = true;
    support_inplace = true;
}

#if NCNN_STDIO
#if NCNN_STRING
int Threshold::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%f", &threshold);
    if (nscan != 1)
    {
        fprintf(stderr, "Threshold load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Threshold::load_param_bin(FILE* paramfp)
{
    fread(&threshold, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Threshold::load_param(const unsigned char*& mem)
{
    threshold = *(float*)(mem);
    mem += 4;

    return 0;
}

int Threshold::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i] > threshold ? 1.f : 0.f;
        }
    }

    return 0;
}

int Threshold::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            ptr[i] = ptr[i] > threshold ? 1.f : 0.f;
        }
    }

    return 0;
}

} // namespace ncnn
