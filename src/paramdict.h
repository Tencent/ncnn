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

#include "mat.h"

// at most 32 parameters
#define NCNN_MAX_PARAM_COUNT 32

namespace ncnn {

class DataReader;
class Net;
class ParamDictPrivate;
class NCNN_EXPORT ParamDict
{
public:
    // empty
    ParamDict();

    virtual ~ParamDict();

    // copy
    ParamDict(const ParamDict&);

    // assign
    ParamDict& operator=(const ParamDict&);

    // get type
    int type(int id) const;

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

protected:
    friend class Net;

    void clear();

    int load_param(const DataReader& dr);
    int load_param_bin(const DataReader& dr);

private:
    ParamDictPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_PARAMDICT_H
