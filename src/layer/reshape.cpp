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

#include "reshape.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Reshape)

Reshape::Reshape()
{
    one_blob_only = true;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int Reshape::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d",
                       &w, &h, &c);
    if (nscan != 3)
    {
        fprintf(stderr, "Reshape load_param failed %d\n", nscan);
        return -1;
    }

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    return 0;
}
#endif // NCNN_STRING
int Reshape::load_param_bin(FILE* paramfp)
{
    fread(&w, sizeof(int), 1, paramfp);

    fread(&h, sizeof(int), 1, paramfp);

    fread(&c, sizeof(int), 1, paramfp);

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    return 0;
}
#endif // NCNN_STDIO

int Reshape::load_param(const unsigned char*& mem)
{
    w = *(int*)(mem);
    mem += 4;

    h = *(int*)(mem);
    mem += 4;

    c = *(int*)(mem);
    mem += 4;

    ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    return 0;
}

int Reshape::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int total = bottom_blob.total();

    if (ndim == 1)
    {
        int _w = w;

        if (_w == 0)
            _w = bottom_blob.w;

        if (_w == -1)
            _w = total;

        top_blob = bottom_blob.reshape(_w);
    }
    else if (ndim == 2)
    {
        int _w = w;
        int _h = h;

        if (_w == 0)
            _w = bottom_blob.w;
        if (_h == 0)
            _h = bottom_blob.h;

        if (_w == -1)
            _w = total / _h;
        if (_h == -1)
            _h = total / _w;

        top_blob = bottom_blob.reshape(_w, _h);
    }
    else if (ndim == 3)
    {
        int _w = w;
        int _h = h;
        int _c = c;

        if (_w == 0)
            _w = bottom_blob.w;
        if (_h == 0)
            _h = bottom_blob.h;
        if (_c == 0)
            _c = bottom_blob.c;

        if (_w == -1)
            _w = total / _c / _h;
        if (_h == -1)
            _h = total / _c / _w;
        if (_c == -1)
            _c = total / _h / _w;

        top_blob = bottom_blob.reshape(_w, _h, _c);
    }

    return 0;
}

} // namespace ncnn
