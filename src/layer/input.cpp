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

#include "input.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Input)

Input::Input()
{
    one_blob_only = true;
    support_inplace = true;
}

#if NCNN_STDIO
#if NCNN_STRING
int Input::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d",
                       &size[0], &size[1], &size[2]);
    if (nscan != 3)
    {
        fprintf(stderr, "Input load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int Input::load_param_bin(FILE* paramfp)
{
    fread(&size[0], sizeof(int), 1, paramfp);

    fread(&size[1], sizeof(int), 1, paramfp);

    fread(&size[2], sizeof(int), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int Input::load_param(const unsigned char*& mem)
{
    size[0] = *(int*)(mem);
    mem += 4;

    size[1] = *(int*)(mem);
    mem += 4;

    size[2] = *(int*)(mem);
    mem += 4;

    return 0;
}

int Input::forward(const Mat& /*bottom_blob*/, Mat& /*top_blob*/) const
{
    return 0;
}

int Input::forward_inplace(Mat& /*bottom_top_blob*/) const
{
    return 0;
}

} // namespace ncnn
