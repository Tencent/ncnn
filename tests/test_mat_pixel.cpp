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

static ncnn::Mat FilledMat(int w, int h, int elempack, unsigned char v)
{
    ncnn::Mat m(w, h, (size_t)elempack, elempack);

    unsigned char* p = m;
    for (int i = 0; i < w * h * elempack; i++)
    {
        p[i] = v;
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

static int test_mat_pixel_roi_gray(int w, int h, int roix, int roiy, int roiw, int roih)
{
    int pixel_type_from[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_GRAY2RGB, ncnn::Mat::PIXEL_GRAY2BGR, ncnn::Mat::PIXEL_GRAY2RGBA, ncnn::Mat::PIXEL_GRAY2BGRA};
    int pixel_type_to[5] = {ncnn::Mat::PIXEL_GRAY, ncnn::Mat::PIXEL_RGB2GRAY, ncnn::Mat::PIXEL_BGR2GRAY, ncnn::Mat::PIXEL_RGBA2GRAY, ncnn::Mat::PIXEL_BGRA2GRAY};

    ncnn::Mat a = RandomMat(w, h, 1);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 1; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih);
        ncnn::Mat b(roiw, roih, 1u, 1);
        m.to_pixels(b, pixel_type_to[i]);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 1);

        if (memcmp(b, c2, roiw * roih * 1) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_gray failed w=%d h=%d roi=[%d %d %d %d] pixel_type=%d\n", w, h, roix, roiy, roiw, roih, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_rgb(int w, int h, int roix, int roiy, int roiw, int roih)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGB2RGBA, ncnn::Mat::PIXEL_RGB2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_RGB, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_RGBA2RGB, ncnn::Mat::PIXEL_BGRA2RGB};

    ncnn::Mat a = RandomMat(w, h, 3);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih);
        ncnn::Mat b(roiw, roih, 3u, 3);
        m.to_pixels(b, pixel_type_to[i]);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 3);

        if (memcmp(b, c2, roiw * roih * 3) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_rgb failed w=%d h=%d roi=[%d %d %d %d] pixel_type=%d\n", w, h, roix, roiy, roiw, roih, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_bgr(int w, int h, int roix, int roiy, int roiw, int roih)
{
    int pixel_type_from[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_BGR2RGB, ncnn::Mat::PIXEL_BGR2RGBA, ncnn::Mat::PIXEL_BGR2BGRA};
    int pixel_type_to[4] = {ncnn::Mat::PIXEL_BGR, ncnn::Mat::PIXEL_RGB2BGR, ncnn::Mat::PIXEL_RGBA2BGR, ncnn::Mat::PIXEL_BGRA2BGR};

    ncnn::Mat a = RandomMat(w, h, 3);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    // FIXME enable more convert types
    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih);
        ncnn::Mat b(roiw, roih, 3u, 3);
        m.to_pixels(b, pixel_type_to[i]);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 3);

        if (memcmp(b, c2, roiw * roih * 3) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_bgr failed w=%d h=%d roi=[%d %d %d %d] pixel_type=%d\n", w, h, roix, roiy, roiw, roih, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_rgba(int w, int h, int roix, int roiy, int roiw, int roih)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_RGBA2BGRA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_RGBA, ncnn::Mat::PIXEL_BGRA2RGBA};

    ncnn::Mat a = RandomMat(w, h, 4);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih);
        ncnn::Mat b(roiw, roih, 4u, 4);
        m.to_pixels(b, pixel_type_to[i]);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 4);

        if (memcmp(b, c2, roiw * roih * 4) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_rgba failed w=%d h=%d roi=[%d %d %d %d] pixel_type=%d\n", w, h, roix, roiy, roiw, roih, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_roi_bgra(int w, int h, int roix, int roiy, int roiw, int roih)
{
    int pixel_type_from[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_BGRA2RGBA};
    int pixel_type_to[2] = {ncnn::Mat::PIXEL_BGRA, ncnn::Mat::PIXEL_RGBA2BGRA};

    ncnn::Mat a = RandomMat(w, h, 4);

    ncnn::Mat a2;
    ncnn::convert_packing(a.reshape(w, h, 1), a2, 1);

    for (int i = 0; i < 2; i++)
    {
        ncnn::Mat m = ncnn::Mat::from_pixels_roi(a, pixel_type_from[i], w, h, roix, roiy, roiw, roih);
        ncnn::Mat b(roiw, roih, 4u, 4);
        m.to_pixels(b, pixel_type_to[i]);

        ncnn::Mat b2;
        ncnn::Mat c2;
        ncnn::copy_cut_border(a2, b2, roiy, h - (roiy + roih), roix, w - (roix + roiw));
        ncnn::convert_packing(b2, c2, 4);

        if (memcmp(b, c2, roiw * roih * 4) != 0)
        {
            fprintf(stderr, "test_mat_pixel_roi_bgra failed w=%d h=%d roi=[%d %d %d %d] pixel_type=%d\n", w, h, roix, roiy, roiw, roih, i);
            return -1;
        }
    }

    return 0;
}

static int test_mat_pixel_yuv420sp2rgb(int w, int h)
{
    ncnn::Mat nv21 = RandomMat(w, h / 2 * 3, 1);

    ncnn::Mat nv12 = nv21.clone();

    // swap VU to UV
    unsigned char* p = (unsigned char*)nv12 + w * h;
    for (int i = 0; i < w * h / 4; i++)
    {
        unsigned char v = p[0];
        unsigned char u = p[1];
        p[0] = u;
        p[1] = v;
        p += 2;
    }

    ncnn::Mat rgb(w, h, 3u, 3);
    yuv420sp2rgb(nv21, w, h, rgb);

    ncnn::Mat rgb2(w, h, 3u, 3);
    yuv420sp2rgb_nv12(nv12, w, h, rgb2);

    if (memcmp(rgb, rgb2, w * h * 3) != 0)
    {
        fprintf(stderr, "test_mat_pixel_yuv420sp2rgb failed w=%d h=%d\n", w, h);
        return -1;
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

static int test_mat_pixel_4()
{
    return 0
           || test_mat_pixel_roi_gray(16, 16, 1, 1, 13, 13)
           || test_mat_pixel_roi_rgb(16, 16, 2, 1, 11, 11)
           || test_mat_pixel_roi_bgr(16, 16, 1, 2, 11, 9)
           || test_mat_pixel_roi_rgba(16, 16, 3, 2, 9, 11)
           || test_mat_pixel_roi_bgra(16, 16, 2, 3, 9, 7);
}

static int test_mat_pixel_5()
{
    return 0
           || test_mat_pixel_roi_gray(15, 15, 2, 3, 2, 3)
           || test_mat_pixel_roi_rgb(15, 15, 3, 4, 5, 4)
           || test_mat_pixel_roi_bgr(15, 15, 4, 5, 6, 7)
           || test_mat_pixel_roi_rgba(15, 15, 6, 6, 3, 1)
           || test_mat_pixel_roi_bgra(15, 15, 7, 3, 1, 1);
}

static int test_mat_pixel_6()
{
    return 0
           || test_mat_pixel_yuv420sp2rgb(16, 16)
           || test_mat_pixel_yuv420sp2rgb(12, 12)
           || test_mat_pixel_yuv420sp2rgb(2, 2)
           || test_mat_pixel_yuv420sp2rgb(6, 6);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mat_pixel_0()
           || test_mat_pixel_1()
           || test_mat_pixel_2()
           || test_mat_pixel_3()
           || test_mat_pixel_4()
           || test_mat_pixel_5()
           || test_mat_pixel_6();
}
