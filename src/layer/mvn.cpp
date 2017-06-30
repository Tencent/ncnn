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

#include "mvn.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(MVN)

MVN::MVN()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int MVN::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %f",
                       &normalize_variance, &across_channels, &eps);
    if (nscan != 3)
    {
        fprintf(stderr, "MVN load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int MVN::load_param_bin(FILE* paramfp)
{
    fread(&normalize_variance, sizeof(int), 1, paramfp);

    fread(&across_channels, sizeof(int), 1, paramfp);

    fread(&eps, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int MVN::load_param(const unsigned char*& mem)
{
    normalize_variance = *(int*)(mem);
    mem += 4;

    across_channels = *(int*)(mem);
    mem += 4;

    eps = *(float*)(mem);
    mem += 4;

    return 0;
}

int MVN::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    // prepare sum per channel
    Mat sum(channels);
    if (sum.empty())
        return -100;
    float* sum_ptr = sum;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

        float sum = 0.f;
        for (int i=0; i<size; i++)
        {
            sum += ptr[i];
        }

        sum_ptr[q] = sum;
    }

    if (across_channels)
    {
        // compute mean across channels
        float mean = 0.f;
        for (int q=0; q<channels; q++)
        {
            mean += sum_ptr[q];
        }
        mean = mean / (channels * size);

        // substract mean
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }
    else
    {
        // substract mean
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            float mean = sum_ptr[q] / size;

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] - mean;
            }
        }
    }

    if (normalize_variance)
    {
        // prepare squared sum per channel
        Mat sqsum(channels);
        if (sqsum.empty())
            return -100;
        float* sqsum_ptr = sqsum;

        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = top_blob.channel(q);

            float sum = 0.f;
            for (int i=0; i<size; i++)
            {
                sum += ptr[i] * ptr[i];
            }

            sqsum_ptr[q] = sum;
        }

        if (across_channels)
        {
            // compute squared mean across channels
            float sqmean = 0.f;
            for (int q=0; q<channels; q++)
            {
                sqmean += sqsum_ptr[q];
            }
            sqmean = sqmean / (channels * size);

            // normalize variance
            float norm_var = sqrt(sqmean) + eps;
            float norm_var_inv = 1.f / norm_var;

            // apply normalize_variance
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * norm_var_inv;
                }
            }
        }
        else
        {
            // apply normalize_variance
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);
                float sqmean = sqsum_ptr[q] / size;
                float norm_var = sqrt(sqmean) + eps;
                float norm_var_inv = 1.f / norm_var;

                for (int i=0; i<size; i++)
                {
                    outptr[i] = ptr[i] * norm_var_inv;
                }
            }
        }

    }

    return 0;
}

} // namespace ncnn
