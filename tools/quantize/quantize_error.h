// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_QUANTIZE_ERROR_H
#define NCNN_QUANTIZE_ERROR_H

#include <cstdio>

#define NCNN_QUANTIZE_CHECK(operation, ...) \
    do                                      \
    {                                       \
        if ((operation) != 0)               \
        {                                   \
            fprintf(stderr, __VA_ARGS__);   \
            return -1;                      \
        }                                   \
    } while (0)

#endif // NCNN_QUANTIZE_ERROR_H
