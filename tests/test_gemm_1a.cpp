// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_1.h"

int main()
{
    SRAND(7767517);

    // transpose and output layouts cover short, aligned and remainder dimensions
    static const int layouts[][3] = {
        {1, 1, 1},
        {31, 31, 31},
        {31, 32, 31},
        {32, 31, 32},
        {32, 32, 32},
        {20, 32, 20},
    };
    for (int i = 0; i < (int)(sizeof(layouts) / sizeof(layouts[0])); i++)
    {
        const int* t = layouts[i];
        if (test_gemm_0(t[0], t[1], t[2], 100, 100, 100) != 0)
            return -1;
    }

    // independent dimensions retain small and intermediate matrix sizes
    static const int dimensions[][3] = {
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {6, 6, 6},
        {7, 7, 7},
        {8, 8, 8},
        {15, 15, 15},
        {16, 16, 16},
        {24, 24, 24},
    };
    for (int i = 0; i < (int)(sizeof(dimensions) / sizeof(dimensions[0])); i++)
    {
        const int* t = dimensions[i];
        if (test_gemm(t[0], t[1], t[2], 100, 100, 100, 2.1f, 0, 0, 0) != 0)
            return -1;
    }

    // aligned and remainder shapes exercise every tile and transpose layout
    static const int boundary[][3] = {
        {31, 31, 31},
        {32, 32, 32},
    };
    static const int tiles[] = {1, 2, 4, 8, 12, 16, 20, 24, 28};
    const int boundary_count = sizeof(boundary) / sizeof(boundary[0]);
    for (int i = 0; i < boundary_count; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            // tile overrides affect cpu kernels
            if (test_gemm_0(boundary[i][0], boundary[i][1], boundary[i][2], tiles[j], tiles[j], tiles[j], TEST_LAYER_DISABLE_GPU_TESTING) != 0)
                return -1;
        }
    }

    // short matrices and asymmetric dimensions cover partial tiles independently
    static const int partial[][6] = {
        {2, 2, 2, 1, 1, 1}, {3, 3, 3, 2, 2, 2},
        {5, 5, 5, 4, 4, 4}, {9, 9, 9, 8, 8, 8},
        {13, 13, 13, 12, 12, 12}, {17, 17, 17, 16, 16, 16},
        {21, 21, 21, 20, 20, 20}, {25, 25, 25, 24, 24, 24},
        {29, 29, 29, 28, 28, 28},
        {20, 32, 20, 12, 12, 12},
        {31, 32, 31, 1, 1, 1}, {32, 31, 32, 4, 8, 4}
    };
    const int partial_count = sizeof(partial) / sizeof(partial[0]);
    for (int i = 0; i < partial_count; i++)
    {
        const int* t = partial[i];
        if (test_gemm_0(t[0], t[1], t[2], t[3], t[4], t[5], TEST_LAYER_DISABLE_GPU_TESTING) != 0)
            return -1;
    }

    return 0;
}
