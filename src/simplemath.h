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
#include <stdint.h>
#define SIN_RED_SWITCHOVER (201.15625f)
#define COS_RED_SWITCHOVER (142.90625f)
#define __HI(X)            *(1 + (short*)&x)
#define __LO(X)            *(short*)&x
#define INFINITY           (1.0 / 0)
#define FE_TONEAREST       0
#define FE_DOWNWARD        1024
#define FE_UPWARD          2048
#define FE_TOWARDZERO      3072

/*
* ====================================================
* useful constants
* ====================================================
*/
const float PI = 3.14159265358979323846;
const float PI_2 = 1.57079632679489661923; /* PI/2 */
const float E = 2.71828182845904523536;
const uint32_t two_over_pi_f[] = {
    0x28be60db,
    0x9391054a,
    0x7f09d5f4,
    0x7d4d3770,
    0x36d8a566,
    0x4f10e410
}; /* 2/ PI*/

/*
* ====================================================
* util functions
* ====================================================
*/
uint32_t float_as_uint32(float a);
float uint32_as_float(uint32_t a);
uint32_t umul32_hi(uint32_t a, uint32_t b);
float uint32_as_float(uint32_t a);
uint32_t umul32_hi(uint32_t a, uint32_t b);
float trig_red_slowpath_f(float a, int* quadrant);
float trig_red_f(float a, float switch_over, int* q);
float sinf_poly(float a, float s);
float cosf_poly(float s);
float sinf_cosf_core(float a, int i);

/*
* ====================================================
* trigonometric functions
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
* power functions
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