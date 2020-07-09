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

static int test_mat_pixel_gray(int w, int h)
{
    int pixel_type_from[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_GRAY2RGB, ncnn::Mat::PIXEL_GRAY2BGR, ncnn::Mat::PIXEL_GRAY2RGBA, ncnn::Mat::PIXEL_GRAY2BGRA};
    int pixel_type_to[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_RGB2GRAY, ncnn::Mat::PIXEL_BGR2GRAY, ncnn::Mat::PIXEL_RGBA2GRAY, ncnn::Mat::PIXEL_BGRA2GRAY};

    ncnn::Mat a = RandomMat(w, h, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 1; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels(a, pixel_type_from[i], w, h);
        ncnn::Mat b(w, h, 1u, 1);
        m.to_pixels(b, pixel_type_to[i]);

        if (memcmp(a, b, w * h * 1) != 0)
        {
            fprintf(stderr, "test_mat_pixel_gray failed w=%d h=%d pixel_type=%d\n", w, h, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_rgb(int w, int h)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGB2RGBA, ncnn::Mat::PIXEL_RGB2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_RGBA2RGB, ncnn::Mat::PIXEL_BGRA2RGB};

    ncnn::Mat a = RandomMat(w, h, 3);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels(a, pixel_type_from[i], w, h);
        ncnn::Mat b(w, h, 3u, 3);
        m.to_pixels(b, pixel_type_to[i]);

        if (memcmp(a, b, w * h * 3) != 0)
        {
            fprintf(stderr, "test_mat_pixel_rgb failed w=%d h=%d pixel_type=%d\n", w, h, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_bgr(int w, int h)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_BGR2RGBA, ncnn::Mat::PIXEL_BGR2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGBA2BGR, ncnn::Mat::PIXEL_BGRA2BGR};

    ncnn::Mat a = RandomMat(w, h, 3);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels(a, pixel_type_from[i], w, h);
        ncnn::Mat b(w, h, 3u, 3);
        m.to_pixels(b, pixel_type_to[i]);

        if (memcmp(a, b, w * h * 3) != 0)
        {
            fprintf(stderr, "test_mat_pixel_bgr failed w=%d h=%d pixel_type=%d\n", w, h, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_rgba(int w, int h)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_RGBA2BGRA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_BGRA2RGBA};

    ncnn::Mat a = RandomMat(w, h, 4);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels(a, pixel_type_from[i], w, h);
        ncnn::Mat b(w, h, 4u, 4);
        m.to_pixels(b, pixel_type_to[i]);

        if (memcmp(a, b, w * h * 4) != 0)
        {
            fprintf(stderr, "test_mat_pixel_rgba failed w=%d h=%d pixel_type=%d\n", w, h, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_bgra(int w, int h)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_BGRA2RGBA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_RGBA2BGRA};

    ncnn::Mat a = RandomMat(w, h, 4);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels(a, pixel_type_from[i], w, h);
        ncnn::Mat b(w, h, 4u, 4);
        m.to_pixels(b, pixel_type_to[i]);

        if (memcmp(a, b, w * h * 4) != 0)
        {
            fprintf(stderr, "test_mat_pixel_bgra failed w=%d h=%d pixel_type=%d\n", w, h, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_0()
{
    return 0
           || test_mat_pixel_gray(16, 16)
           || test_mat_pixel_rgb(16, 16)
           || test_mat_pixel_bgr(16, 16)
           || test_mat_pixel_rgba(16, 16)
           || test_mat_pixel_bgra(16, 16);
}

static int test_mat_pixel_1()
{
    return 0
           || test_mat_pixel_gray(15, 15)
           || test_mat_pixel_rgb(15, 15)
           || test_mat_pixel_bgr(15, 15)
           || test_mat_pixel_rgba(15, 15)
           || test_mat_pixel_bgra(15, 15);
}

static int test_mat_pixel_2()
{
    return 0
           || test_mat_pixel_gray(1, 1)
           || test_mat_pixel_rgb(1, 1)
           || test_mat_pixel_bgr(1, 1)
           || test_mat_pixel_rgba(1, 1)
           || test_mat_pixel_bgra(1, 1);
}

static int test_mat_pixel_3()
{
    return 0
           || test_mat_pixel_gray(3, 3)
           || test_mat_pixel_rgb(3, 3)
           || test_mat_pixel_bgr(3, 3)
           || test_mat_pixel_rgba(3, 3)
           || test_mat_pixel_bgra(3, 3);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mat_pixel_0()
           || test_mat_pixel_1()
           || test_mat_pixel_2()
           || test_mat_pixel_3();
}
