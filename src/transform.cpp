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

#include <math.h>

#include "transform.h"

namespace ncnn {

namespace transform {

Mat combine(const Mat& m1, const Mat& m2)
{
    Mat M(3, 2, sizeof(float));
    float row3[] = {0, 0, 1};
    float* p = M;
    const float* p1 = m1;
    const float* p2 = m2;
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            p[row * 3 + col] = p1[row * 3 + 0] * p2[0 * 3 + col] + p1[row * 3 + 1] * p2[1 * 3 + col] + p1[row * 3 + 2] * row3[col];
        }
    }
    return M;
}

Mat rotation(const float degree, const float* center)
{
    int ndim = 2;
    Mat M(3, 2, sizeof(float));
    float* mat = M;
    float rad = degree * 3.14159265358979323846 / 180.0;
    float c = cos(rad);
    float s = sin(rad);

    M.fill(0);
    mat[0] = c;
    mat[1] = -s;
    mat[3] = s;
    mat[4] = c;

    if (center)
    {
        for (int d = 0; d < ndim; d++)
        {
            mat[d * 3 + ndim] = center[d] - center[0] * mat[d * 3 + 0] - center[1] * mat[d * 3 + 1];
        }
    }

    return M;
}

Mat shear(const float degree, float* center)
{
    int ndim = 2;
    Mat M(3, 2, sizeof(float));
    float* mat = M;
    float rad = degree * 3.14159265358979323846 / 180.0;
    float s = tan(rad);

    M.fill(0);
    mat[0] = 1;
    mat[1] = s;
    mat[3] = s;
    mat[4] = 1;

    if (center)
    {
        for (int d = 0; d < ndim; d++)
        {
            mat[d * 3 + ndim] = center[d] - center[0] * mat[d * 3 + 0] - center[1] * mat[d * 3 + 1];
        }
    }

    return M;
}

Mat shear(const float* degrees, const float* center)
{
    int ndim = 2;
    Mat M(3, 2, sizeof(float));
    float* mat = M;
    float rad1 = degrees[0] * 3.14159265358979323846 / 180.0;
    float s1 = tan(rad1);
    float rad2 = degrees[1] * 3.14159265358979323846 / 180.0;
    float s2 = tan(rad2);

    M.fill(0);
    mat[0] = 1;
    mat[1] = s1;
    mat[3] = s2;
    mat[4] = 1;

    if (center)
    {
        for (int d = 0; d < ndim; d++)
        {
            mat[d * 3 + ndim] = center[d] - center[0] * mat[d * 3 + 0] - center[1] * mat[d * 3 + 1];
        }
    }

    return M;
}

Mat scale(const float* scale, const float* center)
{
    int ndim = 2;
    Mat M(3, 2, sizeof(float));
    M.fill(0);
    float* mat = M;

    for (int d = 0; d < ndim; d++)
    {
        mat[d * 3 + d] = scale[d];
    }
    if (center)
    {
        for (int d = 0; d < ndim; d++)
        {
            mat[d * 3 + ndim] = center[d] * (1 - scale[d]);
        }
    }

    return M;
}

Mat translation(const float* offsets)
{
    int ndim = 2;
    Mat M(3, 2, sizeof(float));
    float* mat = M;

    M.fill(0);
    mat[0] = 1;
    mat[4] = 1;
    for (int d = 0; d < ndim; d++)
    {
        mat[d * 3 + ndim] = offsets[d];
    }

    return M;
}

} // namespace transform

} // namespace ncnn
