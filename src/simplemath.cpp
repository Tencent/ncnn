// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#include "simplemath.h"
#if NCNN_SIMPLEMATH
float abs(float x) {
    return x > 0 ? x : -x;
}

float fmod(float numer, float denom) {
    if (denom == 0.0) {
        return numer;
    }

    int quotient = static_cast<int>(numer / denom);
    return numer - quotient * denom;
}

float sin(float x){
    if(x < 0){
        return -sin(-x);
    }
    x = fmod(static_cast<double>(x), static_cast<double>(2 * PI));

    float taylor = x;
    float fact = static_cast<float>(1);
    float index = x * x * x;
    float term = x;
    float sign = static_cast<float>(-1);
    int i = 1;

    while(abs(term) > 1e-15){
        fact = fact * (i + 1) * (i + 2);

        term = index / fact * sign;
        taylor += term;

        sign = -sign;
        index *= x * x;
        i += 2;
    }
    return taylor;
}
#endif // NCNN_SIMPLEMATH