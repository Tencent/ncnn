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

#include "platform.h"

#if NCNN_SIMPLEMATH

#ifdef __cplusplus
extern "C" {
#endif
/*
* ====================================================
* discrete functions
* ====================================================
*/
NCNN_EXPORT float fabs(float);
NCNN_EXPORT float fabsf(float);
NCNN_EXPORT float fmod(float, float);
NCNN_EXPORT float floor(float);
NCNN_EXPORT float floorf(float);
NCNN_EXPORT float round(float);
NCNN_EXPORT float roundf(float);
NCNN_EXPORT float ceil(float);
NCNN_EXPORT float ceilf(float);
NCNN_EXPORT float fmaxf(float, float);
NCNN_EXPORT float truncf(float);
NCNN_EXPORT float frac(float);
/*
* ====================================================
* trigonometric functions
* ====================================================
*/
NCNN_EXPORT float sinf(float);
NCNN_EXPORT float cosf(float);
NCNN_EXPORT float tanf(float);
NCNN_EXPORT float asinf(float);
NCNN_EXPORT float acosf(float);
NCNN_EXPORT float atanf(float);
NCNN_EXPORT float atan2f(float, float);
NCNN_EXPORT float tanhf(float);

/*
* ====================================================
* power functions
* ====================================================
*/
NCNN_EXPORT float sqrtf(float);
NCNN_EXPORT float sqrt(float);
NCNN_EXPORT float powf(float, float);

/*
* ====================================================
* exponential and logarithm functions
* ====================================================
*/
NCNN_EXPORT float expf(float);
NCNN_EXPORT float frexp(float, int*);
NCNN_EXPORT float logf(float);
NCNN_EXPORT float log(float);
NCNN_EXPORT float log10f(float);

/*
* ====================================================
* probability functions
* ====================================================
*/
NCNN_EXPORT float erf(float);
NCNN_EXPORT float erff(float);
NCNN_EXPORT float erfcf(float);

/*
* ====================================================
* other functions
* ====================================================
*/
NCNN_EXPORT int msb(unsigned int);
NCNN_EXPORT float fmaf(float, float, float);
NCNN_EXPORT float copysignf(float, float);
NCNN_EXPORT void fesetround(int);
NCNN_EXPORT int fegetround();
NCNN_EXPORT float nearbyintf(float);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEMATH

#endif // NCNN_SIMPLEMATH_H
