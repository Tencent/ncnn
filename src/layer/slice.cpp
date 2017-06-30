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

#include "slice.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Slice)

Slice::Slice()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int Slice::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d", &num_slice);
    if (nscan != 1)
    {
        fprintf(stderr, "Slice load_param failed %d\n", nscan);
        return -1;
    }

    slices.create(num_slice);
    if (slices.empty())
        return -100;
    int* slices_ptr = (int*)slices.data;
    for (int i=0; i<num_slice; i++)
    {
        int nscan = fscanf(paramfp, "%d", &slices_ptr[i]);
        if (nscan != 1)
        {
            fprintf(stderr, "Slice load_param failed %d\n", nscan);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STRING
int Slice::load_param_bin(FILE* paramfp)
{
    fread(&num_slice, sizeof(int), 1, paramfp);

    slices.create(num_slice);
    if (slices.empty())
        return -100;
    int* slices_ptr = (int*)slices.data;
    fread(slices_ptr, sizeof(int), num_slice, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Slice::load_param(const unsigned char*& mem)
{
    num_slice = *(int*)(mem);
    mem += 4;

    slices = Mat(num_slice, (float*)mem);
    mem += num_slice * sizeof(int);

    return 0;
}

int Slice::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int q = 0;
    const int* slices_ptr = (const int*)slices.data;
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        int slice = slices_ptr[i];
        if (slice == -233)
        {
            slice = (channels - q) / (top_blobs.size() - i);
        }

        Mat& top_blob = top_blobs[i];
        top_blob.create(w, h, slice);
        if (top_blob.empty())
            return -100;

        int size = bottom_blob.cstep * slice;

        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.data;
        for (int j=0; j<size; j++)
        {
            outptr[j] = ptr[j];
        }

        q += slice;
    }

    return 0;
}

} // namespace ncnn
