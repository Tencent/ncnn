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

#include "platform.h"

#if NCNN_SIMPLEMATH

#include "simplemath.h"
#define __HI(X)       *(1 + (short*)&x)
#define __LO(X)       *(short*)&x
#define INFINITY      (1.0 / 0)
#define FE_TONEAREST  0
#define FE_DOWNWARD   1024
#define FE_UPWARD     2048
#define FE_TOWARDZERO 3072

/*
* ====================================================
* some useful constants
* ====================================================
*/
static const float PI = 3.14159265358979323846;
static const float PI_2 = 1.57079632679489661923; /* PI/2 */
static const float E = 2.71828182845904523536;

/* re-interpret the bit pattern of a uint32 as an IEEE-754 float */
static float uint32_as_float(uint32_t a)
{
    float r;
    float* rp = &r;
    uint32_t* ap = &a;

    *rp = *(float*)ap;

    return r;
}

#ifdef __cplusplus
extern "C" {
#endif
/*
* ====================================================
* Discontinuous function
* ====================================================
*/
float fabs(float x)
{
    return x > 0 ? x : -x;
}

float fabsf(float x)
{
    return fabs(x);
}

float fmod(float numer, float denom)
{
    if (denom == 0.0)
    {
        return numer;
    }
    if (numer <= denom)
    {
        return numer;
    }

    int quotient = static_cast<int>(numer / denom);
    return numer - quotient * denom;
}

float floor(float x)
{
    int intValue = static_cast<int>(x);
    if (x < 0 && x != intValue)
    {
        intValue -= 1;
    }
    return intValue;
}

float floorf(float x)
{
    return floor(x);
}

float round(float x)
{
    float ret = x > 0 ? floor(x + 0.5) : ceil(x - 0.5);
    return ret;
}

float roundf(float x)
{
    return round(x);
}

float ceilf(float x)
{
    return ceil(x);
}

float ceil(float x)
{
    int intValue = static_cast<int>(x);
    if (x == intValue)
    {
        return x;
    }
    return floor(x + 1);
}

float fmaxf(float x, float y)
{
    return x > y ? x : y;
}

float truncf(float x)
{
    int intValue = static_cast<int>(x);
    return static_cast<float>(intValue);
}

float frac(float x)
{
    return x - floor(x);
}

/*
* ====================================================
* trigonometric functions
* ====================================================
*/

/*
    modify from https://developer.download.nvidia.cn/cg/sin.html
*/
float sinf(float a)
{
    const int x = 0;
    const int y = 1;
    const int z = 2;
    const int w = 3;

    float c0[4] = {0.0, 0.5, 1.0, 0.0};
    float c1[4] = {0.25, -9.0, 0.75, 0.159154943091};
    float c2[4] = {24.9808039603, -24.9808039603, -60.1458091736, 60.1458091736};
    float c3[4] = {85.4537887573, -85.4537887573, -64.9393539429, 64.9393539429};
    float c4[4] = {19.7392082214, -19.7392082214, -1.0, 1.0};
    float r0[3], r1[3], r2[3];

    // r1.x = c1.w * a - c1.x
    r1[x] = c1[w] * a - c1[x];
    // r1.y  = frac( r1.x );
    r1[y] = frac(r1[x]);
    // r2.x  = (float) ( r1.y < c1.x );
    r2[x] = (float)(r1[y] < c1[x]);
    // r2.yz = (float2) ( r1.yy >= c1.yz );
    r2[y] = (float)(r1[y] >= c1[y]);
    r2[z] = (float)(r1[y] >= c1[z]);
    // r2.y  = dot( r2, c4.zwz );
    r2[y] = r2[x] * c4[z] + r2[y] * c4[w] + r2[z] * c4[z];

    // r0 = c0.xyz - r1.yyy
    r0[x] = c0[x] - r1[y];
    r0[y] = c0[y] - r1[y];
    r0[z] = c0[z] - r1[y];

    // r0 = r0 * r0
    r0[x] = r0[x] * r0[x];
    r0[y] = r0[y] * r0[y];
    r0[z] = r0[z] * r0[z];

    // r1 = c2.xyx * r0 + c2.zwz
    r1[x] = c2[x] * r0[x] + c2[z];
    r1[y] = c2[y] * r0[y] + c2[w];
    r1[z] = c2[x] * r0[z] + c2[z];

    // r1 = r1 * r0 + c3.xyx
    r1[x] = r1[x] * r0[x] + c3[x];
    r1[y] = r1[y] * r0[y] + c3[y];
    r1[z] = r1[z] * r0[z] + c3[x];

    // r1 = r1 * r0 + c3.zwz
    r1[x] = r1[x] * r0[x] + c3[z];
    r1[y] = r1[y] * r0[y] + c3[w];
    r1[z] = r1[z] * r0[z] + c3[z];

    // r1 = r1 * r0 + c4.xyx
    r1[x] = r1[x] * r0[x] + c4[x];
    r1[y] = r1[y] * r0[y] + c4[y];
    r1[z] = r1[z] * r0[z] + c4[x];

    // r1 = r1 * r0 + c4.zwz
    r1[x] = r1[x] * r0[x] + c4[z];
    r1[y] = r1[y] * r0[y] + c4[w];
    r1[z] = r1[z] * r0[z] + c4[z];

    //r0.x = dot(r1, -r2)
    r0[x] = -(r1[x] * r2[x] + r1[y] * r2[y] + r1[z] * r2[z]);

    return r0[x];
}

float cosf(float x)
{
    return sinf(PI_2 + x);
}

float tanf(float x)
{
    return sinf(x) / cosf(x);
}

/* copy from https://developer.download.nvidia.cn/cg/asin.html */
float asinf(float x)
{
    float negate = float(x < 0);
    x = fabs(x);
    float ret = -0.0187293;
    ret *= x;
    ret += 0.0742610;
    ret *= x;
    ret -= 0.2121144;
    ret *= x;
    ret += 1.5707288;
    ret = PI * 0.5 - sqrt(1.0 - x) * ret;
    return ret - 2 * negate * ret;
}

/* copy from https://developer.download.nvidia.cn/cg/acos.html */
float acosf(float x)
{
    float negate = float(x < 0);
    x = fabs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0 - x);
    ret = ret - 2 * negate * ret;
    return negate * PI + ret;
}

/* copy from https://developer.download.nvidia.cn/cg/atan.html */
float atanf(float a)
{
    if (a < 0)
    {
        return -atanf(-a);
    }
    if (a > 1)
    {
        return PI_2 - atanf(1 / a);
    }
    float s = a * a;
    float r = 0.0027856871020048857;

    r = r * s - 0.015866000205278397;
    r = r * s + 0.042472220957279205;
    r = r * s - 0.07497530430555344f;
    r = r * s + 0.10644879937171936;
    r = r * s - 0.14207030832767487;
    r = r * s + 0.19993454217910767f;
    r = r * s - 0.33333146572113037f;
    r = r * s;
    return r * a + a;
}

float atan2f(float y, float x)
{
    if (x == 0 && y == 0)
    {
        // error
        return 0;
    }
    if (y == 0)
    {
        return x > 0 ? 0 : PI;
    }
    if (x == 0)
    {
        return copysignf(PI_2, y);
    }

    if (x > 0 && y > 0)
    {
        return atanf(y / x);
    }
    else if (x < 0 && y > 0)
    {
        return PI - atanf(y / -x);
    }
    else if (x > 0 && y < 0)
    {
        return -atanf(-y / x);
    }
    else
    {
        return -PI + atanf(-y / -x);
    }
}

float tanhf(float v)
{
    if (v >= 8 || v <= -8)
    {
        return copysignf(1, v);
    }
    float exp2v = expf(2 * v);
    return (exp2v - 1) / (exp2v + 1);
}

/*
* ====================================================
* power functions
* ====================================================
*/

float sqrtf(float x)
{
    return powf(x, 0.5);
}

float sqrt(float x)
{
    return sqrtf(x);
}

float powf(float x, float y)
{
    return expf(y * logf(x));
}

/*
* ====================================================
* exponential and logarithm functions
* ====================================================
*/

/* copy and modify from https://zhuanlan.zhihu.com/p/541466411 */
float logf(float x)
{
    static const float
    ln2_hi
    = 6.93147180369123816490e-01,        /* 3fe62e42 fee00000 */
    ln2_lo = 1.90821492927058770002e-10, /* 3dea39ef 35793c76 */
    two25 = 3.3554432e+07,
    Lg1 = 6.666666666666735130e-01, /* 3FE55555 55555593 */
    Lg2 = 3.999999999940941908e-01, /* 3FD99999 9997FA04 */
    Lg3 = 2.857142874366239149e-01, /* 3FD24924 94229359 */
    Lg4 = 2.222219843214978396e-01, /* 3FCC71C5 1D8E78AF */
    Lg5 = 1.818357216161805012e-01, /* 3FC74664 96CB03DE */
    Lg6 = 1.531383769920937332e-01, /* 3FC39A09 D078C69F */
    Lg7 = 1.479819860511658591e-01; /* 3FC2F112 DF3E5244 */

    static float zero = 0.0;
    float f, s, z, R, w, t1, t2, dk;
    short k, hx, i;
    unsigned short lx;

    hx = __HI(x); /* high word of x */
    lx = __LO(x); /* low  word of x */

    k = 0;
    if (hx < 0x0080)
    {   /* x < 2**-126 */
        if (((hx & 0x7fff) | lx) == 0)
            return -two25 / zero;          /* log(+-0)=-inf */
        if (hx < 0) return (x - x) / zero; /* log(-#) = NaN */
        k -= 25;
        x *= two25;   /* subnormal number, scale up x */
        hx = __HI(x); /* high word of x */
    }

    if (hx >= 0x7f80) return x + x;
    k += (hx >> 7) - 127;
    hx &= 0x007f;
    i = (hx + 0x4b) & 0x0080;
    __HI(x) = hx | (i ^ 0x3f80); /* normalize x or x/2 */
    k += (i >> 7);
    f = x - 1.0f;

    s = f / (2.0f + f);
    dk = (float)k;
    z = s * s;
    w = z * z;
    t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    R = t2 + t1;
    if (k == 0)
        return f - s * (f - R);
    else
        return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

/* copy from https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff */
float expf(float a)
{
    if (a < 0)
    {
        float tmp = expf(-a);

        float ret = 1 / tmp;

        return ret;
    }
    float f, r, j;
    int i;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = 1.442695f * a;
    j = round(j) + 12582912.f; // There is a bug, and the program lives on it.
    j = j - 12582912.f;
    // j = fmaf(1.442695f, a, 12582912.f) - 12582912.f; // 0x1.715476p0, 0x1.8p23
    f = fmaf(j, -6.93145752e-1f, a); // -0x1.62e400p-1  // log_2_hi
    f = fmaf(j, -1.42860677e-6f, f); // -0x1.7f7d1cp-20 // log_2_lo
    i = (int)j;
    // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
    r = 1.37805939e-3f;             // 0x1.694000p-10
    r = fmaf(r, f, 8.37312452e-3f); // 0x1.125edcp-7
    r = fmaf(r, f, 4.16695364e-2f); // 0x1.555b5ap-5
    r = fmaf(r, f, 1.66664720e-1f); // 0x1.555450p-3
    r = fmaf(r, f, 4.99999851e-1f); // 0x1.fffff6p-2
    r = fmaf(r, f, 1.00000000e+0f); // 0x1.000000p+0
    r = fmaf(r, f, 1.00000000e+0f); // 0x1.000000p+0

    float s, t;
    uint32_t ia;
    // exp(a) = 2**i * r
    ia = (i > 0) ? 0 : 0x83000000u;
    s = uint32_as_float(0x7f000000u + ia);
    t = uint32_as_float(((uint32_t)i << 23) - ia);
    r = r * s;
    r = r * t;

    // handle special cases: severe overflow / underflow
    if (fabsf(a) >= 104.0f) r = (a > 0) ? INFINITY : 0.0f;

    return r;
}

float frexp(float x, int* y)
{
    int hx, k;
    hx = __HI(x);
    k = (hx >> 7) & 0x00ff;
    k = k - 127;
    __HI(x) = hx & 0x807f;
    __HI(x) = __HI(x) | 0x3f80;

    *y = k + 1; // y in [1/2, 1)
    return x / 2;
}

float log(float x)
{
    return logf(x);
}

float log10f(float x)
{
    static const float ln10 = 2.3025850929940456840179914546844;
    return logf(x) / ln10;
}

/*
* ====================================================
* probability functions
* ====================================================
*/

/* copy from https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff */
float erf(float a)
{
    float r, s, t, u;

    t = fabsf(a);
    s = a * a;
    if (t > 0.927734375f)
    {   // 475/512
        // maximum error 0.99527 ulp
        r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
        u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
        r = fmaf(r, s, u);
        r = fmaf(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
        r = fmaf(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
        r = fmaf(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
        r = fmaf(r, t, -t);
        r = 1.0f - expf(r);
        r = copysignf(r, a);
    }
    else
    {
        // maximum error 0.98929 ulp
        r = -5.96761703e-4f;             // -0x1.38e000p-11
        r = fmaf(r, s, 4.99119423e-3f);  //  0x1.471a58p-8
        r = fmaf(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
        r = fmaf(r, s, 1.12819925e-1f);  //  0x1.ce1c44p-4
        r = fmaf(r, s, -3.76125336e-1f); // -0x1.812700p-2
        r = fmaf(r, s, 1.28379166e-1f);  //  0x1.06eba8p-3
        r = fmaf(r, a, a);
    }
    return r;
}

float erff(float x)
{
    return erf(x);
}

float erfcf(float x)
{
    return 1.0 - erf(x);
}

/*
* ====================================================
* other functions
* ====================================================
*/

int msb(unsigned int v)
{
    static const int pos[32] = {0, 1, 28, 2, 29, 14, 24, 3,
                                30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19,
                                16, 7, 26, 12, 18, 6, 11, 5, 10, 9
                               };
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = (v >> 1) + 1;
    return pos[(v * 0x077CB531UL) >> 27];
}

float fmaf(float x, float y, float z)
{
    float tmp = x * y;
    float ret = tmp + z;
    return ret;
}

float copysignf(float x, float y)
{
    return fabsf(x) * (y > 0 ? 1 : -1);
}

int round_mode = 0;
void fesetround(int mode)
{
    round_mode = mode;
}

int fegetround()
{
    return round_mode;
}

float nearbyintf(float x)
{
    int intPart = static_cast<int>(x);
    float floatPart = fabs(x - intPart);
    if (floatPart == 0)
    {
        return x;
    }

    if (x > 0)
    {
        if (round_mode == FE_DOWNWARD || round_mode == FE_TOWARDZERO)
        {
            return static_cast<float>(intPart);
        }
        if (round_mode == FE_UPWARD)
        {
            return static_cast<float>(intPart) + 1.0;
        }
        if (round_mode == FE_TONEAREST)
        {
            if (floatPart == 0.5)
            {
                return intPart % 2 == 0 ? static_cast<float>(intPart) : static_cast<float>(intPart) + 1;
            }
            return round(x);
        }
    }
    if (x < 0)
    {
        if (round_mode == FE_UPWARD || round_mode == FE_TOWARDZERO)
        {
            return static_cast<float>(intPart);
        }
        if (round_mode == FE_DOWNWARD)
        {
            return static_cast<float>(intPart) - 1.0;
        }
        if (round_mode == FE_TONEAREST)
        {
            if (floatPart == 0.5)
            {
                return intPart % 2 == 0 ? static_cast<float>(intPart) : static_cast<float>(intPart) - 1;
            }
            return round(x);
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEMATH
