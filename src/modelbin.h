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

#ifndef NCNN_MODELBIN_H
#define NCNN_MODELBIN_H

#include <stdio.h>
#include "mat.h"
#include "platform.h"

namespace ncnn {

class Net;
class ModelBin
{
public:
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = uint8
    // load vec
    Mat load(int w, int type) const;
    // load image
    Mat load(int w, int h, int type) const;
    // load dim
    Mat load(int w, int h, int c, int type) const;

    // construct from weight blob array
    ModelBin(const Mat* weights);

protected:
    mutable const Mat* weights;

    friend class Net;

#if NCNN_STDIO
    ModelBin(FILE* binfp);
    FILE* binfp;
#endif // NCNN_STDIO

    ModelBin(const unsigned char*& mem);
    const unsigned char*& mem;
};

} // namespace ncnn

#endif // NCNN_MODELBIN_H
