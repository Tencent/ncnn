// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_CONSTANTS_H
#define NCNN_CONSTANTS_H

namespace ncnn {
namespace constants {

// basic mathematical constants, double precision
static const double PI = 3.14159265358979323846;
static const double PI_2 = 1.57079632679489661923;
static const double E = 2.71828182845904523536;

// basic mathematical constants, float precision
static const float PI_F = 3.14159265358979323846f;
static const float PI_2_F = 1.57079632679489661923f;
static const float E_F = 2.71828182845904523536f;

// gelu tanh approximation: y = 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
// values kept as-is for numerical compatibility with existing ncnn output
static const float SQRT_2_OVER_PI = 0.79788452f;
static const float GELU_TANH_CUBIC_COEFF = 0.044715f;
static const float INV_SQRT2 = 0.70710678f;

// sqrt(2), shared by 3x3 winograd convolution transforms
static const double SQRT2 = 1.41421356237;

} // namespace constants
} // namespace ncnn

#endif // NCNN_CONSTANTS_H
