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

#include <math.h>
#include <string.h>

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

static ncnn::Mat RandomMat(int w, int h, int elempack)
{
    ncnn::Mat m(w, h, 1, (size_t)elempack, elempack);

    unsigned char* p = m;
    for (int i = 0; i < w * h * elempack; i++)
    {
        p[i] = RAND() % 256;
    }

    return m;
}

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
#define CHECK_MEMBER(m)                                                                 \
    if (a.m != b.m)                                                                     \
    {                                                                                   \
        fprintf(stderr, #m " not match    expect %d but got %d\n", (int)a.m, (int)b.m); \
        return -1;                                                                      \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int i = 0; i < a.h; i++)
        {
            const float* pa = ma.row(i);
            const float* pb = mb.row(i);
            for (int j = 0; j < a.w; j++)
            {
                if (!NearlyEqual(pa[j], pb[j], epsilon))
                {
                    fprintf(stderr, "value not match  at c:%d h:%d w:%d    expect %f but got %f\n", q, i, j, pa[j], pb[j]);
                    return -1;
                }
            }
        }
    }

    return 0;
}

static int test_mat_pixel_resize(int w, int h, int ch, int target_width, int target_height)
{
    ncnn::Mat a = RandomMat(w, h, ch);

    ncnn::Mat b(target_width, target_height, 1, (size_t)ch, ch);

    if (ch == 1) resize_bilinear_c1(a, w, h, b, target_width, target_height);
    if (ch == 2) resize_bilinear_c2(a, w, h, b, target_width, target_height);
    if (ch == 3) resize_bilinear_c3(a, w, h, b, target_width, target_height);
    if (ch == 4) resize_bilinear_c4(a, w, h, b, target_width, target_height);

    ncnn::Mat a2;
    ncnn::convert_packing(a, a2, 1);

    ncnn::Mat b2;
    ncnn::convert_packing(b, b2, 1);

    for (int i = 0; i < ch; i++)
    {
        ncnn::Mat c = ncnn::Mat::from_pixels(a2.channel(i), ncnn::Mat::PIXEL_GRAY, w, h);
        ncnn::Mat d = ncnn::Mat::from_pixels(b2.channel(i), ncnn::Mat::PIXEL_GRAY, target_width, target_height);

        ncnn::Mat e;
        ncnn::resize_bilinear(c, e, target_width, target_height);

        if (Compare(e, d, 0.5) != 0)
        {
            fprintf(stderr, "test_mat_pixel_resize failed w=%d h=%d ch=%d target_width=%d target_height=%d\n", w, h, ch, target_width, target_height);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_resize_gray(int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height)
{
    int pixel_type_from[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_GRAY2RGB, ncnn::Mat::PIXEL_GRAY2BGR, ncnn::Mat::PIXEL_GRAY2RGBA, ncnn::Mat::PIXEL_GRAY2BGRA};
    int pixel_type_to[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_RGB2GRAY, ncnn::Mat::PIXEL_BGR2GRAY, ncnn::Mat::PIXEL_RGBA2GRAY, ncnn::Mat::PIXEL_BGRA2GRAY};

    ncnn::Mat a = RandomMat(w, h, 1);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 1; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi_resize(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih, target_width, target_height);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 1);
        ncnn::Mat d2 = ncnn::Mat::from_pixels_resize(c2, pixel_type_from[i], c2.w, c2.h, target_width, target_height);

        if (memcmp(m, d2, target_width * target_height * d2.c) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_resize_gray failed w=%d h=%d roi=[%d %d %d %d] target_width=%d target_height=%d pixel_type=%d\n", w, h, roix, roiy, roiw, roih, target_width, target_height, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_resize_rgb(int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGB2RGBA, ncnn::Mat::PIXEL_RGB2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_RGBA2RGB, ncnn::Mat::PIXEL_BGRA2RGB};

    ncnn::Mat a = RandomMat(w, h, 3);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi_resize(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih, target_width, target_height);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 3);
        ncnn::Mat d2 = ncnn::Mat::from_pixels_resize(c2, pixel_type_from[i], c2.w, c2.h, target_width, target_height);

        if (memcmp(m, d2, target_width * target_height * d2.c) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_resize_rgb failed w=%d h=%d roi=[%d %d %d %d] target_width=%d target_height=%d pixel_type=%d\n", w, h, roix, roiy, roiw, roih, target_width, target_height, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_resize_bgr(int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_BGR2RGBA, ncnn::Mat::PIXEL_BGR2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGBA2BGR, ncnn::Mat::PIXEL_BGRA2BGR};

    ncnn::Mat a = RandomMat(w, h, 3);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi_resize(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih, target_width, target_height);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 3);
        ncnn::Mat d2 = ncnn::Mat::from_pixels_resize(c2, pixel_type_from[i], c2.w, c2.h, target_width, target_height);

        if (memcmp(m, d2, target_width * target_height * d2.c) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_resize_bgr failed w=%d h=%d roi=[%d %d %d %d] target_width=%d target_height=%d pixel_type=%d\n", w, h, roix, roiy, roiw, roih, target_width, target_height, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_resize_rgba(int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_RGBA2BGRA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_BGRA2RGBA};

    ncnn::Mat a = RandomMat(w, h, 4);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi_resize(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih, target_width, target_height);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 4);
        ncnn::Mat d2 = ncnn::Mat::from_pixels_resize(c2, pixel_type_from[i], c2.w, c2.h, target_width, target_height);

        if (memcmp(m, d2, target_width * target_height * d2.c) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_resize_rgba failed w=%d h=%d roi=[%d %d %d %d] target_width=%d target_height=%d pixel_type=%d\n", w, h, roix, roiy, roiw, roih, target_width, target_height, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_resize_bgra(int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_BGRA2RGBA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_RGBA2BGRA};

    ncnn::Mat a = RandomMat(w, h, 4);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi_resize(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih, target_width, target_height);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 4);
        ncnn::Mat d2 = ncnn::Mat::from_pixels_resize(c2, pixel_type_from[i], c2.w, c2.h, target_width, target_height);

        if (memcmp(m, d2, target_width * target_height * d2.c) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_resize_bgra failed w=%d h=%d roi=[%d %d %d %d] target_width=%d target_height=%d pixel_type=%d\n", w, h, roix, roiy, roiw, roih, target_width, target_height, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_0()
{
    for (int c = 1; c <= 4; c++)
    {
        int ret = 0
                  || test_mat_pixel_resize(24, 48, c, 24, 48)
                  || test_mat_pixel_resize(13, 17, c, 11, 14)
                  || test_mat_pixel_resize(33, 23, c, 5, 6)
                  || test_mat_pixel_resize(5, 4, c, 11, 16)
                  || test_mat_pixel_resize(23, 11, c, 15, 21);

        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_mat_pixel_1()
{
    return 0
           || test_mat_pixel_roi_resize_gray(16, 16, 1, 1, 13, 13, 10, 11)
           || test_mat_pixel_roi_resize_rgb(16, 16, 2, 1, 11, 11, 2, 3)
           || test_mat_pixel_roi_resize_bgr(16, 16, 1, 2, 11, 9, 22, 13)
           || test_mat_pixel_roi_resize_rgba(16, 16, 3, 2, 9, 11, 12, 4)
           || test_mat_pixel_roi_resize_bgra(16, 16, 2, 3, 9, 7, 7, 7);
}

static int test_mat_pixel_2()
{
    return 0
           || test_mat_pixel_roi_resize_gray(15, 15, 2, 3, 2, 3, 2, 2)
           || test_mat_pixel_roi_resize_rgb(15, 15, 3, 4, 5, 4, 5, 4)
           || test_mat_pixel_roi_resize_bgr(15, 15, 4, 5, 6, 7, 4, 1)
           || test_mat_pixel_roi_resize_rgba(15, 15, 6, 6, 3, 4, 1, 3)
           || test_mat_pixel_roi_resize_bgra(15, 15, 7, 3, 1, 1, 1, 1);
}

int main()
{
    SRAND(7767517);

    return test_mat_pixel_0() || test_mat_pixel_1() || test_mat_pixel_2();
}
