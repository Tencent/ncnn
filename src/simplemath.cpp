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
float fabs(float x) {
    return x > 0 ? x : -x;
}

float fmod(float numer, float denom) {
    if (denom == 0.0) {
        return numer;
    }

    int quotient = static_cast<int>(numer / denom);
    return numer - quotient * denom;
}

float sinf(float x){
    if(x < 0){
        return -sinf(-x);
    }
    x = fmod(static_cast<double>(x), static_cast<double>(2 * PI));

    float taylor = x;
    float fact = static_cast<float>(1);
    float index = x * x * x;
    float term = x;
    float sign = static_cast<float>(-1);
    int i = 1;

    while(fabs(term) > 1e-15){
        fact = fact * (i + 1) * (i + 2);

        term = index / fact * sign;
        taylor += term;

        sign = -sign;
        index *= x * x;
        i += 2;
    }
    return taylor;
}
float cosf(float x){
    x = fabs(x);
    return sinf(PI_2 + x); // sin(PI_2 + x) = cos(x)
}
float floor(float x){
    int intValue = static_cast<int>(x);
    if (x < 0 && x != intValue){
        intValue -= 1;
    }
    return intValue;
}
float round(float x){
    int intValue = static_cast<int>(x); // 获取 x 的整数部分
    double decimalPart = x - intValue; // 获取 x 的小数部分

    if (decimalPart < 0.5) {
        if (x < 0 && decimalPart != 0) {
            return intValue - 1; // 向下取整
        } else {
            return intValue; // 向上取整
        }
    } else {
        if (x < 0) {
            return intValue - 1; // 向下取整
        } else {
            return intValue + 1; // 向上取整
        }
    }
}
float sqrtf(float x){
    return 0.1;
}
float sqrt(float x){
    return sqrtf(x);
}
float logf(float x){
    return 0.1;
}
float expf(float x){
    return 0.1;
}
float fmaxf(float x, float y){
    return 0.1;
}
float tanhf(float x){
    return 0.1;
}
float powf(float x, float y){
    return 0.1;
}
float fabsf(float x){
    return 0.1;
}
float floorf(float x){
    return floor(x);
}
float ceilf(float x){
    return 0.1;
}
float ceil(float x){
    return 0.1;
}
float atan2f(float x, float y){
    return 0.1;
}
float tanf(float x){
    return 0.1;
}
float asinf(float x){
    return 0.1;
}
float acosf(float x){
    return 0.1;
}
float atanf(float x){
    return 0.1;
}
float log10f(float x){
    return 0.1;
}
float truncf(float x){
    return 0.1;
}
float roundf(float x){
    return 0.1;
}
float erfcf(float x){
    return 0.1;
}
#endif // NCNN_SIMPLEMATH