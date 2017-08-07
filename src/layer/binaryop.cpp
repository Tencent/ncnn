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

#include "binaryop.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BinaryOp)

BinaryOp::BinaryOp()
{
    one_blob_only = false;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int BinaryOp::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d", &op_type);
    if (nscan != 1)
    {
        fprintf(stderr, "BinaryOp load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int BinaryOp::load_param_bin(FILE* paramfp)
{
    fread(&op_type, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int BinaryOp::load_param(const unsigned char*& mem)
{
    op_type = *(int*)(mem);
    mem += 4;

    return 0;
}

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_ADD)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] + ptr1[i];
            }
        }
    }
    else if (op_type == Operation_SUB)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] - ptr1[i];
            }
        }
    }
    else if (op_type == Operation_MUL)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
        }
    }
    else if (op_type == Operation_DIV)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] / ptr1[i];
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
        }
    }
    else if (op_type == Operation_MIN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = std::min(ptr[i], ptr1[i]);
            }
        }
    }
    else if (op_type == Operation_POW)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = pow(ptr[i], ptr1[i]);
            }
        }
    }

    return 0;
}

} // namespace ncnn
