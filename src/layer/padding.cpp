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

#include "padding.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding)

Padding::Padding()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int Padding::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %d %d %f", &top, &bottom, &left, &right, &type, &value);
    if (nscan != 6)
    {
        fprintf(stderr, "Padding load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Padding::load_param_bin(FILE* paramfp)
{
    fread(&top, sizeof(int), 1, paramfp);

    fread(&bottom, sizeof(int), 1, paramfp);

    fread(&left, sizeof(int), 1, paramfp);

    fread(&right, sizeof(int), 1, paramfp);

    fread(&type, sizeof(int), 1, paramfp);

    fread(&value, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Padding::load_param(const unsigned char*& mem)
{
    top = *(int*)(mem);
    mem += 4;

    bottom = *(int*)(mem);
    mem += 4;

    left = *(int*)(mem);
    mem += 4;

    right = *(int*)(mem);
    mem += 4;

    type = *(int*)(mem);
    mem += 4;

    value = *(float*)(mem);
    mem += 4;

    return 0;
}

int Padding::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    copy_make_border(bottom_blob, top_blob, top, bottom, left, right, type, value);

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
