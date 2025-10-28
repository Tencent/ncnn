// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "utils.h"

#include <math.h>

namespace pnnx {

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

std::string float_to_string(float f)
{
    if (f == 0.f)
        return "0.0";

    const float abs_f = std::abs(f);
    char buffer[64];

    if (abs_f < 0.0001f || abs_f >= 1000000.0f)
    {
        snprintf(buffer, sizeof(buffer), "%e", f);
        return std::string(buffer);
    }

    const int len = snprintf(buffer, sizeof(buffer), "%g", f);

    bool is_integer = true;
    for (int i = 0; i < len; i++)
    {
        if (buffer[i] == '.' || buffer[i] == 'e' || buffer[i] == 'E')
        {
            is_integer = false;
            break;
        }
    }

    // maintain point-zero
    if (is_integer)
    {
        buffer[len] = '.';
        buffer[len + 1] = '0';
        buffer[len + 2] = '\0';
    }

    return std::string(buffer);
}

std::string double_to_string(double d)
{
    if (d == 0.0)
        return "0.0";

    const double abs_d = std::abs(d);
    char buffer[128];

    if (abs_d < 0.0001 || abs_d >= 1000000.0)
    {
        snprintf(buffer, sizeof(buffer), "%e", d);
        return std::string(buffer);
    }

    const int len = snprintf(buffer, sizeof(buffer), "%g", d);

    bool is_integer = true;
    for (int i = 0; i < len; i++)
    {
        if (buffer[i] == '.' || buffer[i] == 'e' || buffer[i] == 'E')
        {
            is_integer = false;
            break;
        }
    }

    // maintain point-zero
    if (is_integer)
    {
        buffer[len] = '.';
        buffer[len + 1] = '0';
        buffer[len + 2] = '\0';
    }

    return std::string(buffer);
}

void apply_weight_norm(std::vector<float>& weight, const std::vector<float>& weight_g, int dim0, int size)
{
    for (int i = 0; i < dim0; i++)
    {
        float* pw = weight.data() + i * size;

        double norm = 0.f;
        for (int j = 0; j < size; j++)
        {
            float w = pw[j];
            norm += w * w;
        }
        norm = sqrt(norm);

        for (int j = 0; j < size; j++)
        {
            pw[j] = pw[j] * (weight_g[i] / norm);
        }
    }
}

} // namespace pnnx
