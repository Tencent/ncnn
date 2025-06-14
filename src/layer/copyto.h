// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_COPYTO_H
#define LAYER_COPYTO_H

#include "layer.h"

namespace ncnn {

class CopyTo : public Layer
{
public:
    CopyTo();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void resolve_copyto_offset(const Mat& self_blob, int& woffset, int& hoffset, int& doffset, int& coffset) const;

public:
    int woffset;
    int hoffset;
    int doffset;
    int coffset;

    // numpy-style slice
    // if provided, all the above attributes will be ignored
    Mat starts;
    Mat axes;
};

} // namespace ncnn

#endif // LAYER_COPYTO_H
