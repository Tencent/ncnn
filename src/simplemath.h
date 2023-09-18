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

#ifndef NCNN_SIMPLEMATH_H
#define NCNN_SIMPLEMATH_H
#include <stdint.h>

/*
* ====================================================
* discrete functions
* ====================================================
*/
float fabs(float);
float fabsf(float);
float fmod(float, float);
float floor(float);
float floorf(float);
float round(float);
float roundf(float);
float ceil(float);
float ceilf(float);
float fmaxf(float, float);
float truncf(float);

/*
* ====================================================
* trigonometric functions
* ====================================================
*/
float sinf(float);
float cosf(float);
float tanf(float);
float asinf(float);
float acosf(float);
float atanf(float);
float atan2f(float, float);
float tanhf(float);

/*
* ====================================================
* power functions
* ====================================================
*/
float sqrtf(float);
float sqrt(float);
float powf(float, float);

/*
* ====================================================
* exponential and logarithm functions
* ====================================================
*/
float expf(float);
float frexp(float, int*);
float logf(float);
float log(float);
float log10f(float);

/*
* ====================================================
* probability functions
* ====================================================
*/
float erf(float);
float erfcf(float);

/*
* ====================================================
* other functions
* ====================================================
*/
int msb(unsigned int);
float fmaf(float, float, float);
float copysignf(float, float);
void fesetround(int);
int fegetround();
float nearbyintf(float);
#endif // NCNN_SIMPLEMATH