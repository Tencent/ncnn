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

static ncnn::Mat generate_ncnn_logo(int w, int h)
{
    // clang-format off
    // *INDENT-OFF*
    static const unsigned char ncnn_logo_data[16][16] =
    {
        {245, 245,  33, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245,  33, 245, 245},
        {245,  33,  33,  33, 245, 245, 245, 245, 245, 245, 245, 245,  33,  33,  33, 245},
        {245,  33, 158, 158,  33, 245, 245, 245, 245, 245, 245,  33, 158, 158,  33, 245},
        { 33, 117, 158, 224, 158,  33, 245, 245, 245, 245,  33, 158, 224, 158, 117,  33},
        { 33, 117, 224, 224, 224,  66,  33,  33,  33,  33,  66, 224, 224, 224, 117,  33},
        { 33, 189, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 189,  33},
        { 33, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97,  97,  97,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97, 189, 189,  97, 224, 224,  33},
        { 33,  66,  66,  66, 224, 224, 224, 224, 224, 224, 224, 224,  66,  66,  66,  33},
        { 66, 158, 158,  66,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158,  66,  66},
        { 66, 158, 158, 208,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158, 208,  66},
        { 66, 224, 202, 158,  66, 224, 224, 224, 224, 224, 224,  66, 224, 202, 158,  66},
        { 66, 158, 224, 158,  66, 224, 224, 224, 224, 224, 224,  66, 158, 224, 158,  66}
    };
    // *INDENT-ON*
    // clang-format on

    ncnn::Mat m(w, h, (size_t)1, 1);
    resize_bilinear_c1((const unsigned char*)ncnn_logo_data, 16, 16, m, w, h);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int elempack)
{
    ncnn::Mat image = generate_ncnn_logo(w, h);

    ncnn::Mat m(w, h, 1, (size_t)elempack, elempack);

    for (int i = 0; i < h; i++)
    {
        unsigned char* p = m.row<unsigned char>(i);
        const unsigned char* pb = image.row<const unsigned char>(i);
        for (int j = 0; j < w; j++)
        {
            for (int k = 0; k < elempack; k++)
            {
                p[k] = pb[0];
            }

            p += elempack;
            pb += 1;
        }
    }

    return m;
}

static bool NearlyEqual(unsigned char a, unsigned char b)
{
    return abs(a - b) <= 10;
}

static int CompareNearlyEqual(const ncnn::Mat& a, const ncnn::Mat& b)
{
    for (int i = 0; i < a.h; i++)
    {
        const unsigned char* pa = a.row<const unsigned char>(i);
        const unsigned char* pb = b.row<const unsigned char>(i);
        for (int j = 0; j < a.w; j++)
        {
            for (int k = 0; k < a.elempack; k++)
            {
                if (!NearlyEqual(pa[k], pb[k]))
                {
                    fprintf(stderr, "value not match  at  h:%d w:%d [%d]   expect %d but got %d\n", i, j, k, pa[k], pb[k]);
                    return -1;
                }
            }

            pa += a.elempack;
            pb += a.elempack;
        }
    }

    return 0;
}

static int test_mat_pixel_affine_c1(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 1);

    float tm[6];
    float tm_inv[6];
    ncnn::get_rotation_matrix(10.f, 0.15f, w / 2, h / 2, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

    ncnn::Mat a1(w / 2, h / 2, 1u, 1);
    ncnn::Mat a2 = a0.clone();

    ncnn::warpaffine_bilinear_c1(a0, w, h, a1, w / 2, h / 2, tm, 0);
    ncnn::warpaffine_bilinear_c1(a1, w / 2, h / 2, a2, w, h, tm_inv, -233);

    if (CompareNearlyEqual(a0, a2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_c1 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_affine_c2(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 2);

    float tm[6];
    float tm_inv[6];
    ncnn::get_rotation_matrix(20.f, 0.25f, w / 4, h / 4, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

    ncnn::Mat a1(w / 4, h / 4, 2u, 2);
    ncnn::Mat a2 = a0.clone();

    ncnn::warpaffine_bilinear_c2(a0, w, h, a1, w / 4, h / 4, tm, 0);
    ncnn::warpaffine_bilinear_c2(a1, w / 4, h / 4, a2, w, h, tm_inv, -233);

    if (CompareNearlyEqual(a0, a2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_c2 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_affine_c3(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 3);

    float tm[6];
    float tm_inv[6];
    ncnn::get_rotation_matrix(-30.f, 0.6f, w / 2, h / 2, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

    ncnn::Mat a1(w / 2, h / 2, 3u, 3);
    ncnn::Mat a2 = a0.clone();

    ncnn::warpaffine_bilinear_c3(a0, w, h, a1, w / 2, h / 2, tm, 0);
    ncnn::warpaffine_bilinear_c3(a1, w / 2, h / 2, a2, w, h, tm_inv, -233);

    if (CompareNearlyEqual(a0, a2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_c3 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_affine_c4(int w, int h)
{
    ncnn::Mat a0 = RandomMat(w, h, 4);

    const float points_from[4] = {w / 8.f, h / 8.f, w / 8.f + 1.f, h / 8.f + 3.f};
    const float points_to[4] = {w / 2.f, h / 2.f, w / 2.f + 2.f, h / 2.f};

    float tm[6];
    float tm_inv[6];
    ncnn::get_affine_transform(points_from, points_to, 2, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

    ncnn::Mat a1(w / 4, h / 4, 4u, 4);
    ncnn::Mat a2 = a0.clone();

    ncnn::warpaffine_bilinear_c4(a0, w, h, a1, w / 4, h / 4, tm, 0);
    ncnn::warpaffine_bilinear_c4(a1, w / 4, h / 4, a2, w, h, tm_inv, -233);

    if (CompareNearlyEqual(a0, a2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_c4 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_affine_0()
{
    return 0
           || test_mat_pixel_affine_c1(60, 70)
           || test_mat_pixel_affine_c2(60, 70)
           || test_mat_pixel_affine_c3(60, 70)
           || test_mat_pixel_affine_c4(60, 70)
           || test_mat_pixel_affine_c1(120, 160)
           || test_mat_pixel_affine_c2(120, 160)
           || test_mat_pixel_affine_c3(120, 160)
           || test_mat_pixel_affine_c4(120, 160)
           || test_mat_pixel_affine_c1(220, 330)
           || test_mat_pixel_affine_c2(220, 330)
           || test_mat_pixel_affine_c3(220, 330)
           || test_mat_pixel_affine_c4(220, 330);
}

static int test_mat_pixel_affine_yuv420sp(int w, int h)
{
    ncnn::Mat a0(w, h * 3 / 2, 1u, 1);

    ncnn::Mat a0_y = RandomMat(w, h, 1);
    ncnn::Mat a0_uv = RandomMat(w / 2, h / 2, 2);
    memcpy(a0, a0_y, w * h);
    memcpy((unsigned char*)a0 + w * h, a0_uv, w * h / 2);

    float tm[6];
    float tm_inv[6];
    ncnn::get_rotation_matrix(-70.f, 0.3f, w / 2, h / 2, tm);
    ncnn::invert_affine_transform(tm, tm_inv);

    ncnn::Mat a1(w / 2, (h / 2) * 3 / 2, 1u, 1);
    ncnn::Mat a2 = a0.clone();

    ncnn::warpaffine_bilinear_yuv420sp(a0, w, h, a1, w / 2, h / 2, tm, 0);
    ncnn::warpaffine_bilinear_yuv420sp(a1, w / 2, h / 2, a2, w, h, tm_inv, -233);

    // Y
    if (CompareNearlyEqual(ncnn::Mat(w, h, (unsigned char*)a0, 1u, 1), ncnn::Mat(w, h, (unsigned char*)a2, 1u, 1)) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_yuv420sp Y failed w=%d h=%d\n", w, h);
        return -1;
    }

    // UV
    if (CompareNearlyEqual(ncnn::Mat(w / 2, h / 2, (unsigned char*)a0 + w * h, 2u, 2), ncnn::Mat(w / 2, h / 2, (unsigned char*)a2 + w * h, 2u, 2)) != 0)
    {
        fprintf(stderr, "test_mat_pixel_affine_yuv420sp UV failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_affine_1()
{
    return 0
           || test_mat_pixel_affine_yuv420sp(60, 40)
           || test_mat_pixel_affine_yuv420sp(120, 160)
           || test_mat_pixel_affine_yuv420sp(220, 340);
}

int main()
{
    SRAND(7767517);

    return test_mat_pixel_affine_0() || test_mat_pixel_affine_1();
}
