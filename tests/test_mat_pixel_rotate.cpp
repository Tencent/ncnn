// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mat.h"
#include "prng.h"

#include <string.h>

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

static ncnn::Mat RandomMat(int w, int h, int elempack)
{
    ncnn::Mat m(w, h, (size_t)elempack, elempack);

    unsigned char* p = m;
    for (int i = 0; i < w * h * elempack; i++)
    {
        p[i] = RAND() % 256;
    }

    return m;
}

static int test_mat_pixel_rotate_c1(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 1);

    ncnn::Mat a1(w, h, 1u, 1);
    ncnn::Mat a2(w, h, 1u, 1);
    ncnn::Mat a3(w, h, 1u, 1);
    ncnn::Mat a4(w, h, 1u, 1);
    ncnn::Mat a5(h, w, 1u, 1);
    ncnn::Mat a6(w, h, 1u, 1);
    ncnn::Mat a7(h, w, 1u, 1);
    ncnn::Mat a8(w, h, 1u, 1);

    ncnn::kanna_rotate_c1(a0, w, h, a1, w, h, 1);
    ncnn::kanna_rotate_c1(a1, w, h, a2, w, h, 2);
    ncnn::kanna_rotate_c1(a2, w, h, a3, w, h, 3);
    ncnn::kanna_rotate_c1(a3, w, h, a4, w, h, 4);
    ncnn::kanna_rotate_c1(a4, w, h, a5, h, w, 5);
    ncnn::kanna_rotate_c1(a5, h, w, a6, w, h, 6);
    ncnn::kanna_rotate_c1(a6, w, h, a7, h, w, 7);
    ncnn::kanna_rotate_c1(a7, h, w, a8, w, h, 8);

    if (memcmp(a0, a8, w * h * 1) != 0)
    {
        fprintf(stderr, "test_mat_pixel_rotate_c1 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_rotate_c2(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 2);

    ncnn::Mat a1(w, h, 2u, 2);
    ncnn::Mat a2(w, h, 2u, 2);
    ncnn::Mat a3(w, h, 2u, 2);
    ncnn::Mat a4(w, h, 2u, 2);
    ncnn::Mat a5(h, w, 2u, 2);
    ncnn::Mat a6(w, h, 2u, 2);
    ncnn::Mat a7(h, w, 2u, 2);
    ncnn::Mat a8(w, h, 2u, 2);

    ncnn::kanna_rotate_c2(a0, w, h, a1, w, h, 1);
    ncnn::kanna_rotate_c2(a1, w, h, a2, w, h, 2);
    ncnn::kanna_rotate_c2(a2, w, h, a3, w, h, 3);
    ncnn::kanna_rotate_c2(a3, w, h, a4, w, h, 4);
    ncnn::kanna_rotate_c2(a4, w, h, a5, h, w, 5);
    ncnn::kanna_rotate_c2(a5, h, w, a6, w, h, 6);
    ncnn::kanna_rotate_c2(a6, w, h, a7, h, w, 7);
    ncnn::kanna_rotate_c2(a7, h, w, a8, w, h, 8);

    if (memcmp(a0, a8, w * h * 2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_rotate_c2 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_rotate_c3(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 3);

    ncnn::Mat a1(w, h, 3u, 3);
    ncnn::Mat a2(w, h, 3u, 3);
    ncnn::Mat a3(w, h, 3u, 3);
    ncnn::Mat a4(w, h, 3u, 3);
    ncnn::Mat a5(h, w, 3u, 3);
    ncnn::Mat a6(w, h, 3u, 3);
    ncnn::Mat a7(h, w, 3u, 3);
    ncnn::Mat a8(w, h, 3u, 3);

    ncnn::kanna_rotate_c3(a0, w, h, a1, w, h, 1);
    ncnn::kanna_rotate_c3(a1, w, h, a2, w, h, 2);
    ncnn::kanna_rotate_c3(a2, w, h, a3, w, h, 3);
    ncnn::kanna_rotate_c3(a3, w, h, a4, w, h, 4);
    ncnn::kanna_rotate_c3(a4, w, h, a5, h, w, 5);
    ncnn::kanna_rotate_c3(a5, h, w, a6, w, h, 6);
    ncnn::kanna_rotate_c3(a6, w, h, a7, h, w, 7);
    ncnn::kanna_rotate_c3(a7, h, w, a8, w, h, 8);

    if (memcmp(a0, a8, w * h * 3) != 0)
    {
        fprintf(stderr, "test_mat_pixel_rotate_c3 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_rotate_c4(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 4);

    ncnn::Mat a1(w, h, 4u, 4);
    ncnn::Mat a2(w, h, 4u, 4);
    ncnn::Mat a3(w, h, 4u, 4);
    ncnn::Mat a4(w, h, 4u, 4);
    ncnn::Mat a5(h, w, 4u, 4);
    ncnn::Mat a6(w, h, 4u, 4);
    ncnn::Mat a7(h, w, 4u, 4);
    ncnn::Mat a8(w, h, 4u, 4);

    ncnn::kanna_rotate_c4(a0, w, h, a1, w, h, 1);
    ncnn::kanna_rotate_c4(a1, w, h, a2, w, h, 2);
    ncnn::kanna_rotate_c4(a2, w, h, a3, w, h, 3);
    ncnn::kanna_rotate_c4(a3, w, h, a4, w, h, 4);
    ncnn::kanna_rotate_c4(a4, w, h, a5, h, w, 5);
    ncnn::kanna_rotate_c4(a5, h, w, a6, w, h, 6);
    ncnn::kanna_rotate_c4(a6, w, h, a7, h, w, 7);
    ncnn::kanna_rotate_c4(a7, h, w, a8, w, h, 8);

    if (memcmp(a0, a8, w * h * 4) != 0)
    {
        fprintf(stderr, "test_mat_pixel_rotate_c4 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_rotate_0()
{
    return 0
           || test_mat_pixel_rotate_c1(6, 7)
           || test_mat_pixel_rotate_c2(6, 7)
           || test_mat_pixel_rotate_c3(6, 7)
           || test_mat_pixel_rotate_c4(6, 7)
           || test_mat_pixel_rotate_c1(12, 16)
           || test_mat_pixel_rotate_c2(12, 16)
           || test_mat_pixel_rotate_c3(12, 16)
           || test_mat_pixel_rotate_c4(12, 16)
           || test_mat_pixel_rotate_c1(22, 33)
           || test_mat_pixel_rotate_c2(22, 33)
           || test_mat_pixel_rotate_c3(22, 33)
           || test_mat_pixel_rotate_c4(22, 33);
}

static int test_mat_pixel_rotate_yuv420sp(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h * 3 / 2, 1);

    ncnn::Mat a1(w, h * 3 / 2, 1u, 1);
    ncnn::Mat a2(w, h * 3 / 2, 1u, 1);
    ncnn::Mat a3(w, h * 3 / 2, 1u, 1);
    ncnn::Mat a4(w, h * 3 / 2, 1u, 1);
    ncnn::Mat a5(h, w * 3 / 2, 1u, 1);
    ncnn::Mat a6(w, h * 3 / 2, 1u, 1);
    ncnn::Mat a7(h, w * 3 / 2, 1u, 1);
    ncnn::Mat a8(w, h * 3 / 2, 1u, 1);

    ncnn::kanna_rotate_yuv420sp(a0, w, h, a1, w, h, 1);
    ncnn::kanna_rotate_yuv420sp(a1, w, h, a2, w, h, 2);
    ncnn::kanna_rotate_yuv420sp(a2, w, h, a3, w, h, 3);
    ncnn::kanna_rotate_yuv420sp(a3, w, h, a4, w, h, 4);
    ncnn::kanna_rotate_yuv420sp(a4, w, h, a5, h, w, 5);
    ncnn::kanna_rotate_yuv420sp(a5, h, w, a6, w, h, 6);
    ncnn::kanna_rotate_yuv420sp(a6, w, h, a7, h, w, 7);
    ncnn::kanna_rotate_yuv420sp(a7, h, w, a8, w, h, 8);

    if (memcmp(a0, a8, w * h * 3 / 2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_rotate_yuv420sp failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_rotate_1()
{
    return 0
           || test_mat_pixel_rotate_yuv420sp(6, 4)
           || test_mat_pixel_rotate_yuv420sp(12, 16)
           || test_mat_pixel_rotate_yuv420sp(22, 34);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mat_pixel_rotate_0()
           || test_mat_pixel_rotate_1();
}
