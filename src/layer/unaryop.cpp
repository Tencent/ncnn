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

#include "unaryop.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(UnaryOp)

UnaryOp::UnaryOp()
{
    one_blob_only = true;
    support_inplace = true;
}

#if NCNN_STDIO
#if NCNN_STRING
int UnaryOp::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d", &op_type);
    if (nscan != 1)
    {
        fprintf(stderr, "UnaryOp load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int UnaryOp::load_param_bin(FILE* paramfp)
{
    fread(&op_type, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int UnaryOp::load_param(const unsigned char*& mem)
{
    op_type = *(int*)(mem);
    mem += 4;

    return 0;
}

int UnaryOp::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_ABS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = fabs(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_NEG)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = -ptr[i];
            }
        }
    }
    else if (op_type == Operation_FLOOR)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = floor(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_CEIL)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ceil(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_SQUARE)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = ptr[i] * ptr[i];
            }
        }
    }
    else if (op_type == Operation_SQRT)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = sqrt(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_RSQRT)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = 1.f / sqrt(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_EXP)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = exp(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_LOG)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = log(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_SIN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = sin(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_COS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = cos(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_TAN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = tan(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ASIN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = asin(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ACOS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = acos(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ATAN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                outptr[i] = atan(ptr[i]);
            }
        }
    }

    return 0;
}

int UnaryOp::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (op_type == Operation_ABS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = fabs(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_NEG)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = -ptr[i];
            }
        }
    }
    else if (op_type == Operation_FLOOR)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = floor(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_CEIL)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = ceil(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_SQUARE)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = ptr[i] * ptr[i];
            }
        }
    }
    else if (op_type == Operation_SQRT)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = sqrt(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_RSQRT)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = 1.f / sqrt(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_EXP)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = exp(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_LOG)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = log(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_SIN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = sin(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_COS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = cos(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_TAN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = tan(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ASIN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = asin(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ACOS)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = acos(ptr[i]);
            }
        }
    }
    else if (op_type == Operation_ATAN)
    {
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                ptr[i] = atan(ptr[i]);
            }
        }
    }

    return 0;
}

} // namespace ncnn
