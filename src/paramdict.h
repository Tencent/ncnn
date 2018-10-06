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
    // empty
    ParamDict();

    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    Mat get(int id, const Mat& def) const;

    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, const Mat& v);

public:
    int use_winograd_convolution;
    int use_sgemm_convolution;
    int use_int8_inference;

protected:
    friend class Net;

    void clear();

#if NCNN_STDIO
#if NCNN_STRING
    int load_param(FILE* fp);
#endif // NCNN_STRING
    int load_param_bin(FILE* fp);
#endif // NCNN_STDIO
    int load_param(const unsigned char*& mem);

protected:
    const int PARAM_TYPE_NULL = 0;
    const int PARAM_TYPE_INT = 1;
    const int PARAM_TYPE_FLOAT = 2;
    const int PARAM_TYPE_INT_MAT = 3;
    const int PARAM_TYPE_FLOAT_MAT = 4;
    struct
    {
        int type;
        union { int i; float f; };
        Mat v;
    } params[NCNN_MAX_PARAM_COUNT];
};

#if NCNN_SAVER
class ParamDictSaver : public ParamDict
{
public:
    int save_param(FILE* pp);
};
#endif
} // namespace ncnn

#endif // NCNN_PARAMDICT_H
