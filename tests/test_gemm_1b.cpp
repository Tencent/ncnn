// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_1.h"

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {40, 40, 40},
        {47, 47, 47},
        {48, 48, 48},
        {52, 52, 52},
        {63, 64, 63},
        {64, 63, 64},
        {64, 64, 64}
    };

    int tile_mnk[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {4, 4, 4},
        {8, 8, 8},
        {12, 12, 12},
        {16, 16, 16},
        {20, 20, 20},
        {24, 24, 24},
        {28, 28, 28}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;
    int tile_mnk_count = sizeof(tile_mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        for (int j = 0; j < tile_mnk_count; j++)
        {
            int TILE_M = tile_mnk[j][0];
            int TILE_N = tile_mnk[j][1];
            int TILE_K = tile_mnk[j][2];

            if (TILE_M >= M && TILE_N >= N && TILE_K >= K)
                continue;

            int ret = test_gemm_0(M, N, K, TILE_M, TILE_N, TILE_K);
            if (ret != 0)
                return ret;
        }

        // test no tiling
        int ret = test_gemm_0(M, N, K, 100, 100, 100);
        if (ret != 0)
            return ret;
    }

    return 0;
}
