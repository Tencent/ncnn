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
#ifndef NCNN_SIMPLEMATH_H
#define NCNN_SIMPLEMATH_H
#define PI 3.14159265358979323846

namespace sm{
template<typename T>
T abs(T x) {
    return x > 0 ? x : -x;
}

template<typename T>
T fmod(T numer, T denom) {
    if (denom == 0.0) {
        return numer;
    }

    T quotient = static_cast<int>(numer / denom);
    return numer - quotient * denom;
}

template<typename T>
T sin(T x){
    if(x < 0){
        return -sin(-x);
    }
    x = fmod(static_cast<double>(x), static_cast<double>(2 * PI));

    T taylor = x;
    T fact = static_cast<T>(1);
    T index = x * x * x;
    T term = x;
    T sign = static_cast<T>(-1);
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
} // namespace std
#endif // NCNN_SIMPLEMATH_H