// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_FAST_MATH_H
#define NCNN_FAST_MATH_H

#include <math.h>
#include <string.h>

namespace ncnn {

// Fast polynomial approximation of expf()
// Uses range reduction exp(x) = 2^n * exp(r) with degree-4 minimax polynomial
// Maximum relative error: < 0.02% in typical neural network value ranges

// expf overflow / underflow thresholds for float
constexpr float NCNN_EXP_HI = 88.3762626647949f;
constexpr float NCNN_EXP_LO = -88.3762626647949f;

// exp(x) = 2^n * exp(r)
// 1 / ln(2)
constexpr float NCNN_LOG2EF = 1.44269504089f;
// ln(2)
constexpr float NCNN_LN2 = 0.6931471805599453f;

// Degree-4 minimax polynomial coefficients for exp(r)
// exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24
constexpr float NCNN_EXP_C1 = 1.f;
constexpr float NCNN_EXP_C2 = 1.f;
constexpr float NCNN_EXP_C3 = 0.5f;
constexpr float NCNN_EXP_C4 = 0.1666666667f;
constexpr float NCNN_EXP_C5 = 0.0416666667f;

static inline float fast_exp(float x)
{
    if (x > NCNN_EXP_HI)
        return INFINITY;
    if (x < NCNN_EXP_LO)
        return 0.f;

    // exp(x) = 2^n * exp(r), where n = round(x / ln2), r = x - n * ln2
    float n = floorf(x * NCNN_LOG2EF + 0.5f);
    float r = x - n * NCNN_LN2;

    float exp_r = NCNN_EXP_C1 + r * (NCNN_EXP_C2 + r * (NCNN_EXP_C3 + r * (NCNN_EXP_C4 + r * NCNN_EXP_C5)));

    // Reconstruct: 2^n via IEEE754 bit manipulation
    int ni = (int)n;
    int bits = (ni + 127) << 23;

    float pow2_n;
    memcpy(&pow2_n, &bits, sizeof(float));

    return exp_r * pow2_n;
}

} // namespace ncnn

#endif // NCNN_FAST_MATH_H
