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

#if NCNN_SIMPLEMATH
const float PI = 3.14159265358979323846;
const float PI_2 = 1.57079632679489661923; // PI/2
const float E = 2.71828182845904523536;
float fabs(float x);
float fmod(float numer, float denom);
float sinf(float x);
float cosf(float x);
float sqrtf(float x);
float sqrt(float x);
float expf(float x);
float logf(float x);
float tanhf(float x);
float powf(float x, float y);
float atan2f(float, float);
float atanf(float);
float log10f(float);
float nearbyintf(float);
float asinf(float);
float acosf(float);
float tanf(float);
float erfcf(float);

// 非连续函数
float floor(float x);
float round(float x);
float ceil(float);
float fmaxf(float x, float y);
float fabsf(float x);
float floorf(float x);
float ceilf(float x);
float truncf(float);
float roundf(float);
#endif // NCNN_SIMPLEMATH