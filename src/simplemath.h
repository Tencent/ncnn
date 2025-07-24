// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
NCNN_EXPORT float fmodf(float, float);
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
NCNN_EXPORT float sinhf(float);
NCNN_EXPORT float coshf(float);
NCNN_EXPORT float tanhf(float);
NCNN_EXPORT float asinhf(float);
NCNN_EXPORT float acoshf(float);
NCNN_EXPORT float atanhf(float);

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
