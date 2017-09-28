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

#include "expanddims.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ExpandDims)

ExpandDims::ExpandDims()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int ExpandDims::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d", &expand_w, &expand_h, &expand_c);
    if (nscan != 3)
    {
        fprintf(stderr, "ExpandDims load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int ExpandDims::load_param_bin(FILE* paramfp)
{
    fread(&expand_w, sizeof(int), 1, paramfp);

    fread(&expand_h, sizeof(int), 1, paramfp);

    fread(&expand_c, sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int ExpandDims::load_param(const unsigned char*& mem)
{
    expand_w = *(int*)(mem);
    mem += 4;

    expand_h = *(int*)(mem);
    mem += 4;

    expand_c = *(int*)(mem);
    mem += 4;

    return 0;
}

int ExpandDims::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int dims = bottom_blob.dims;

    top_blob = bottom_blob;

    if (dims == 1)
    {
        if (expand_w)
        {
            if (expand_h)
                top_blob = bottom_blob.reshape(1, 1, w);
            else if (expand_c)
                top_blob = bottom_blob.reshape(1, w, 1);
            else
                top_blob = bottom_blob.reshape(1, w);
        }
        else if (expand_h)
        {
            if (expand_c)
                top_blob = bottom_blob.reshape(w, 1, 1);
            else
                top_blob = bottom_blob.reshape(w, 1);
        }
    }
    else if (dims == 2)
    {
        if (expand_w)
            top_blob = bottom_blob.reshape(1, w, h);
        else if (expand_h)
            top_blob = bottom_blob.reshape(w, 1, h);
        else if (expand_c)
            top_blob = bottom_blob.reshape(w, h, 1);
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
