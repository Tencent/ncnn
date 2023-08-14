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
// 非连续函数
float fabs(float x) {
    return x > 0 ? x : -x;
}
float fabsf(float x){
    return fabs(x);
}
float fmod(float numer, float denom) {
    return numer - truncf(numer / denom) * denom;
}
float floor(float x){
    int intValue = static_cast<int>(x);
    if (x < 0 && x != intValue){
        intValue -= 1;
    }
    return intValue;
}
float floorf(float x){
    return floor(x);
}
float round(float x){
    int intValue = static_cast<int>(x); // 获取 x 的整数部分
    float decimalPart = fabs(x - intValue); // 获取 x 的小数部分
    if(decimalPart == 0){
        return x;
    }
    if(decimalPart < 0.5){
        return intValue;
    }else{
        if(x < 0){
            return intValue - 1;
        }else{
            return intValue + 1;
        }
    }
}
float roundf(float x){
    return round(x);
}
float ceilf(float x){
    return ceil(x);
}
float ceil(float x){
    int intValue = static_cast<int>(x);
    if(x == intValue){
        return x;
    }
    return floor(x + 1);
}
float fmaxf(float x, float y){
    return x > y ? x : y;
}
float truncf(float x){
    int intValue = static_cast<int>(x);
    return static_cast<float>(intValue);
}
// 三角函数
float sinf(float x){
    if(x < 0){
        return -sinf(-x);
    }
    x = fmod(static_cast<float>(x), static_cast<float>(2 * PI));

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
float tanf(float x){
    return sinf(x) / cosf(x);
}
float asinf(float x){
    if(x > 1 || x < -1){
        // error
        return -1;
    }
    float taylor = 0.0;
    float deno = 1.0 / x;  // 分母
    float numer = 1.0; // 分子
    int i = 0;  // 第i项
    float term = x; // 第i项的值
    float xpower = 1 / x; // x的幂

    while(fabs(term) > 1e-15){
        // 第i项
        i++;
        // 计算分子
        numer = numer * (2 * i - 1) * x * x;
        // 计算分母
        deno = deno * i;
        // 计算第i项的值
        term = numer / deno;
        // 加上第i项的值
        taylor += term;
    }
    return taylor;
}
float acosf(float x){
    if(x > 1 || x < -1){
        // error
        return -1;
    }
    return PI_2 - asinf(x);
}
float atanf(float x){
    return asinf(x / sqrtf(1 + x * x));
}
float atan2f(float y, float x){
    if(x == 0 && y == 0){
        // error
        return -1;
    }
    if(y == 0){
        return 0;
    }
    if(x == 0){
        return PI_2;
    }
    if(x > 0 && y > 0){
        return atanf(y / x);
    }
    if(x < 0 && y > 0){
        return PI - atanf(y / -x);
    }
    if(x > 0 && y < 0){
        return -atanf(-y / x);
    }
    if(x < 0 && y < 0){
        return -PI + atanf(-y / -x);
    }
}
float tanhf(float x){
    return  (logf(x) - logf(-x)) / (logf(x) + logf(-x));
}

// 幂函数
float sqrtf(float x){
    if(x < 0){
        // error
        return -1;
    }
    float right = 3.402823466e+38F / 2;
    float left = 0;
    while(left < right){
        float mid = (left + right) / 2;
        if(mid * mid < x){
            left = mid;
        }else{
            right = mid;
        }
        if(fabs(mid * mid - x) < 1e-5){
            return mid;
        }
    }
}
float sqrt(float x){
    return sqrtf(x);
}
float powf(float x, float y){
    int intValue = static_cast<int>(y);
    float res = 1;
    for(int i = 0; i < fabs(intValue); i ++){
        res *= x;
    }
    float decimalPart = fabs(y) - fabs(intValue);
    if(decimalPart == 0){
        return res;
    }
    float left = 0, right = x;
    while(left < right){
        float mid = (left + right) / 2;
        if(powf(mid, intValue) < x){
            left = mid;
        }else{
            right = mid;
        }
        if(fabs(powf(mid, intValue) - x) < 1e-5){
            return mid * res;
        }
    }
}
// 指数对数函数
float logf(float x){
    x = x - 1;

    float taylor = x;
    float index = x;
    float term = x;
    float sign = static_cast<float>(-1);
    int i = 2;

    while(fabs(term) > 1e-15){
        term = index / i * sign;
        taylor += term;

        sign = -sign;
        index *= x;
        i += 1;
    }
    return taylor;
}
float expf(float x){
    float taylor = 1;
    float fact = static_cast<float>(1);
    float index = x;
    float term = x;
    int i = 1;

    while(fabs(term) > 1e-15){
        term = index / fact;
        taylor += term;

        index *= x;
        index = index * (i + 1);
        i += 1;
    }
    return taylor;
}

float log10f(float x){
    return logf(x) / logf(10.0);
}

// 概率函数
float erf(float x){
    float taylor = 0.0;
    float deno = 1.0 / x;  // 分母
    float numer = 1.0; // 分子
    int i = -1;  // 第i项
    float term = 1.0; // 第i项的值
    float xpower = 1.0; // x的幂
    float fact = 1.0;  // 从1到i的阶乘
    int sign = -1; // 符号

    while(fabs(term) > 1e-15){
        // 第i项
        i++;
        // 阶乘
        fact = fact * (i + 1);
        // 计算分子
        numer = numer * x * x;
        // 计算分母
        deno = fact * (2 * i + 1);
        // 计算第i项的值
        term = numer / deno;
        // 改变符号
        sign = -sign;
        // 加上第i项的值
        taylor += sign * term;
    }
    return 2 / sqrtf(PI) * taylor;
}
float erfcf(float x){
    return 1.0 - erf(x);
}
#endif // NCNN_SIMPLEMATH