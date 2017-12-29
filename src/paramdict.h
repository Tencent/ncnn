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

#ifndef NCNN_PARAMDICT_H
#define NCNN_PARAMDICT_H

#include <stdio.h>
#include "mat.h"
#include "platform.h"

// at most 20 parameters
#define NCNN_MAX_PARAM_COUNT 20

namespace ncnn {

class Net;
class ParamDict
{
public:
    // get int
    int get(int id, int def) const
    {
        return params[id].loaded ? params[id].i : def;
    }

    // get float
    float get(int id, float def) const
    {
        return params[id].loaded ? params[id].f : def;
    }

    // get array
    Mat get(int id, const Mat& def) const
    {
        return params[id].loaded ? params[id].v : def;
    }

protected:
    friend class Net;

    ParamDict();

    void clear();

#if NCNN_STDIO
#if NCNN_STRING
    int load_param(FILE* fp);
#endif // NCNN_STRING
    int load_param_bin(FILE* fp);
#endif // NCNN_STDIO
    int load_param(const unsigned char*& mem);

protected:
    struct
    {
        int loaded;
        union { int i; float f; };
        Mat v;
    } params[NCNN_MAX_PARAM_COUNT];
};

} // namespace ncnn

#endif // NCNN_PARAMDICT_H
