// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_TRANSFORM_H
#define NCNN_TRANSFORM_H

#include "mat.h"

namespace ncnn {

namespace transform {

Mat combine(const Mat& m1, const Mat& m2);

template<typename... Args>
Mat combine(const Mat& m1, const Mat& m2, const Args&... mats)
{
    return combine(combine(m1, m2), mats...);
}

Mat rotation(const float degree, const float* center = NULL);

Mat shear(const float degree, float* center = NULL);

Mat shear(const float* degrees, const float* center = NULL);

Mat scale(const float* scale, const float* center = NULL);

Mat translation(const float* offsets);

} // namespace transform

} // namespace ncnn

#endif // NCNN_TRANSFORM_H
