// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static int RandomInt(int a, int b)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    int diff = b - a;
    float r = random * diff;
    return a + (int)r;
}

static int RandomInt2(int a, int b)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    int diff = b - a;
    float r = random * diff;
    return (a + (int)r + 1) / 2 * 2;
}

static int test_mat_pixel_drawing_c1(int w, int h)
{
    ncnn::Mat a(w, h, (size_t)1u, 1);
    ncnn::Mat b(h, w, (size_t)1u, 1);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    ncnn::draw_rectangle_c1(a, w, h, 0, 0, w, h, _color, -1);
    ncnn::draw_rectangle_c1(b, h, w, 0, 0, h, w, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, w);
    int ry = RandomInt(0, h);
    int rw = RandomInt(0, w - rx);
    int rh = RandomInt(0, h - ry);
    color[0] = 100;
    ncnn::draw_rectangle_c1(a, w, h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c1(b, h, w, ry, rx, rh, rw, _color, 3);

    // draw filled rectangle out of image
    color[0] = 144;
    ncnn::draw_rectangle_c1(a, w, h, w - 10, -10, 20, 30, _color, -1);
    ncnn::draw_rectangle_c1(b, h, w, -10, w - 10, 30, 20, _color, -1);
    color[0] = 166;
    ncnn::draw_rectangle_c1(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c1(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw rectangle out of image
    color[0] = 44;
    ncnn::draw_rectangle_c1(a, w, h, rx + w / 2, ry + h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c1(b, h, w, ry + h / 2, rx + w / 2, rh, rw, _color, 1);
    color[0] = 66;
    ncnn::draw_rectangle_c1(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c1(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw filled circle
    int cx = RandomInt(0, w);
    int cy = RandomInt(0, h);
    int radius = RandomInt(0, std::min(w, h));
    color[0] = 20;
    ncnn::draw_circle_c1(a, w, h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c1(b, h, w, cy, cx, radius, _color, -1);

    // draw filled circle out of image
    color[0] = 230;
    ncnn::draw_circle_c1(a, w, h, 10, -4, 6, _color, -1);
    ncnn::draw_circle_c1(b, h, w, -4, 10, 6, _color, -1);

    // draw circle out of image
    color[0] = 130;
    ncnn::draw_circle_c1(a, w, h, cx, cy, radius + std::min(w, h) / 2, _color, 5);
    ncnn::draw_circle_c1(b, h, w, cy, cx, radius + std::min(w, h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, w);
    int y0 = RandomInt(0, h);
    int x1 = RandomInt(0, w);
    int y1 = RandomInt(0, h);
    color[0] = 233;
    ncnn::draw_line_c1(a, w, h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c1(b, h, w, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    ncnn::draw_line_c1(a, w, h, x0 - w, y0 - h, x1 + w, y1 + h, _color, 1);
    ncnn::draw_line_c1(b, h, w, y0 - h, x0 - w, y1 + h, x1 + w, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, (size_t)1u, 1);
    ncnn::kanna_rotate_c1(b, h, w, c, w, h, 5);

    // draw text
    const char text[] = "saJIEWdl\nj43@o";
    int tx = RandomInt(0, w / 2);
    int ty = RandomInt(0, h / 2);
    int fontpixelsize = 10;
    color[0] = 128;
    ncnn::draw_text_c1(a, w, h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 8; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c1(c, w, h, ch, tx + tw / 8 * i, ty, fontpixelsize, _color);
    }
    for (int i = 9; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c1(c, w, h, ch, tx + tw / 8 * (i - 9), ty + th / 2, fontpixelsize, _color);
    }

    // draw text out of image
    fontpixelsize = std::max(w, h) / 2;
    color[0] = 228;
    ncnn::draw_text_c1(a, w, h, "QAQ", -3, -5, fontpixelsize, _color);
    ncnn::get_text_drawing_size("QAQ", fontpixelsize, &tw, &th);
    ncnn::draw_text_c1(c, w, h, "Q", -3, -5, fontpixelsize, _color);
    ncnn::draw_text_c1(c, w, h, "A", -3 + tw / 3, -5, fontpixelsize, _color);
    ncnn::draw_text_c1(c, w, h, "Q", -3 + tw / 3 * 2, -5, fontpixelsize, _color);

    if (memcmp(a, c, w * h) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c1 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_c2(int w, int h)
{
    ncnn::Mat a(w, h, (size_t)2u, 2);
    ncnn::Mat b(h, w, (size_t)2u, 2);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 251;
    ncnn::draw_rectangle_c2(a, w, h, 0, 0, w, h, _color, -1);
    ncnn::draw_rectangle_c2(b, h, w, 0, 0, h, w, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, w);
    int ry = RandomInt(0, h);
    int rw = RandomInt(0, w - rx);
    int rh = RandomInt(0, h - ry);
    color[0] = 100;
    color[1] = 130;
    ncnn::draw_rectangle_c2(a, w, h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c2(b, h, w, ry, rx, rh, rw, _color, 3);

    // draw filled rectangle out of image
    color[0] = 144;
    color[1] = 133;
    ncnn::draw_rectangle_c2(a, w, h, w - 10, -10, 20, 30, _color, -1);
    ncnn::draw_rectangle_c2(b, h, w, -10, w - 10, 30, 20, _color, -1);
    color[0] = 166;
    color[1] = 133;
    ncnn::draw_rectangle_c2(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c2(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 33;
    ncnn::draw_rectangle_c2(a, w, h, rx + w / 2, ry + h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c2(b, h, w, ry + h / 2, rx + w / 2, rh, rw, _color, 1);
    color[0] = 66;
    color[1] = 44;
    ncnn::draw_rectangle_c2(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c2(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw filled circle
    int cx = RandomInt(0, w);
    int cy = RandomInt(0, h);
    int radius = RandomInt(0, std::min(w, h));
    color[0] = 20;
    color[1] = 120;
    ncnn::draw_circle_c2(a, w, h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c2(b, h, w, cy, cx, radius, _color, -1);

    // draw filled circle out of image
    color[0] = 230;
    color[1] = 130;
    ncnn::draw_circle_c2(a, w, h, 10, -4, 6, _color, -1);
    ncnn::draw_circle_c2(b, h, w, -4, 10, 6, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 30;
    ncnn::draw_circle_c2(a, w, h, cx, cy, radius + std::min(w, h) / 2, _color, 5);
    ncnn::draw_circle_c2(b, h, w, cy, cx, radius + std::min(w, h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, w);
    int y0 = RandomInt(0, h);
    int x1 = RandomInt(0, w);
    int y1 = RandomInt(0, h);
    color[0] = 233;
    color[1] = 233;
    ncnn::draw_line_c2(a, w, h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c2(b, h, w, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    color[1] = 192;
    ncnn::draw_line_c2(a, w, h, x0 - w, y0 - h, x1 + w, y1 + h, _color, 1);
    ncnn::draw_line_c2(b, h, w, y0 - h, x0 - w, y1 + h, x1 + w, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, (size_t)2u, 2);
    ncnn::kanna_rotate_c2(b, h, w, c, w, h, 5);

    // draw text
    const char text[] = "Q`~\\=f\nPN\'/<DSA";
    int tx = RandomInt(0, w / 2);
    int ty = RandomInt(0, h / 2);
    int fontpixelsize = 12;
    color[0] = 0;
    color[1] = 128;
    ncnn::draw_text_c2(a, w, h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 6; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c2(c, w, h, ch, tx + tw / 8 * i, ty, fontpixelsize, _color);
    }
    for (int i = 7; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c2(c, w, h, ch, tx + tw / 8 * (i - 7), ty + th / 2, fontpixelsize, _color);
    }

    // draw text out of image
    fontpixelsize = std::max(w, h) / 3;
    color[0] = 228;
    color[1] = 0;
    ncnn::draw_text_c2(a, w, h, "!@#$%^&", -1, -2, fontpixelsize, _color);
    ncnn::get_text_drawing_size("!@#$%^&", fontpixelsize, &tw, &th);
    ncnn::draw_text_c2(c, w, h, "!@#", -1, -2, fontpixelsize, _color);
    ncnn::draw_text_c2(c, w, h, "$", -1 + tw / 7 * 3, -2, fontpixelsize, _color);
    ncnn::draw_text_c2(c, w, h, "%^&", -1 + tw / 7 * 4, -2, fontpixelsize, _color);

    if (memcmp(a, c, w * h * 2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c2 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_c3(int w, int h)
{
    ncnn::Mat a(w, h, (size_t)3u, 3);
    ncnn::Mat b(h, w, (size_t)3u, 3);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 251;
    color[2] = 244;
    ncnn::draw_rectangle_c3(a, w, h, 0, 0, w, h, _color, -1);
    ncnn::draw_rectangle_c3(b, h, w, 0, 0, h, w, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, w);
    int ry = RandomInt(0, h);
    int rw = RandomInt(0, w - rx);
    int rh = RandomInt(0, h - ry);
    color[0] = 100;
    color[1] = 130;
    color[2] = 150;
    ncnn::draw_rectangle_c3(a, w, h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c3(b, h, w, ry, rx, rh, rw, _color, 3);

    // draw filled rectangle out of image
    color[0] = 144;
    color[1] = 133;
    color[2] = 122;
    ncnn::draw_rectangle_c3(a, w, h, w - 10, -10, 20, 30, _color, -1);
    ncnn::draw_rectangle_c3(b, h, w, -10, w - 10, 30, 20, _color, -1);
    color[0] = 166;
    color[1] = 133;
    color[2] = 122;
    ncnn::draw_rectangle_c3(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c3(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 33;
    color[2] = 22;
    ncnn::draw_rectangle_c3(a, w, h, rx + w / 2, ry + h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c3(b, h, w, ry + h / 2, rx + w / 2, rh, rw, _color, 1);
    color[0] = 66;
    color[1] = 44;
    color[2] = 33;
    ncnn::draw_rectangle_c3(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c3(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw filled circle
    int cx = RandomInt(0, w);
    int cy = RandomInt(0, h);
    int radius = RandomInt(0, std::min(w, h));
    color[0] = 20;
    color[1] = 120;
    color[2] = 220;
    ncnn::draw_circle_c3(a, w, h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c3(b, h, w, cy, cx, radius, _color, -1);

    // draw filled circle out of image
    color[0] = 230;
    color[1] = 130;
    color[2] = 110;
    ncnn::draw_circle_c3(a, w, h, 10, -4, 6, _color, -1);
    ncnn::draw_circle_c3(b, h, w, -4, 10, 6, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 30;
    color[2] = 230;
    ncnn::draw_circle_c3(a, w, h, cx, cy, radius + std::min(w, h) / 2, _color, 5);
    ncnn::draw_circle_c3(b, h, w, cy, cx, radius + std::min(w, h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, w);
    int y0 = RandomInt(0, h);
    int x1 = RandomInt(0, w);
    int y1 = RandomInt(0, h);
    color[0] = 233;
    color[1] = 233;
    color[2] = 233;
    ncnn::draw_line_c3(a, w, h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c3(b, h, w, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    color[1] = 192;
    color[2] = 0;
    ncnn::draw_line_c3(a, w, h, x0 - w, y0 - h, x1 + w, y1 + h, _color, 1);
    ncnn::draw_line_c3(b, h, w, y0 - h, x0 - w, y1 + h, x1 + w, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, (size_t)3u, 3);
    ncnn::kanna_rotate_c3(b, h, w, c, w, h, 5);

    // draw text
    const char text[] = "Q`~\\=f\nPN\'/<DSA";
    int tx = RandomInt(0, w / 2);
    int ty = RandomInt(0, h / 2);
    int fontpixelsize = 12;
    color[0] = 0;
    color[1] = 128;
    color[2] = 128;
    ncnn::draw_text_c3(a, w, h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 6; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c3(c, w, h, ch, tx + tw / 8 * i, ty, fontpixelsize, _color);
    }
    for (int i = 7; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c3(c, w, h, ch, tx + tw / 8 * (i - 7), ty + th / 2, fontpixelsize, _color);
    }

    // draw text out of image
    fontpixelsize = std::max(w, h) / 2;
    color[0] = 228;
    color[1] = 0;
    color[2] = 128;
    ncnn::draw_text_c3(a, w, h, "qwqwqwq", -13, -15, fontpixelsize, _color);
    ncnn::get_text_drawing_size("qwqwqwq", fontpixelsize, &tw, &th);
    ncnn::draw_text_c3(c, w, h, "qwq", -13, -15, fontpixelsize, _color);
    ncnn::draw_text_c3(c, w, h, "w", -13 + tw / 7 * 3, -15, fontpixelsize, _color);
    ncnn::draw_text_c3(c, w, h, "qwq", -13 + tw / 7 * 4, -15, fontpixelsize, _color);

    if (memcmp(a, c, w * h * 3) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c3 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_c4(int w, int h)
{
    ncnn::Mat a(w, h, (size_t)4u, 4);
    ncnn::Mat b(h, w, (size_t)4u, 4);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 255;
    color[2] = 255;
    color[3] = 0;
    ncnn::draw_rectangle_c4(a, w, h, 0, 0, w, h, _color, -1);
    ncnn::draw_rectangle_c4(b, h, w, 0, 0, h, w, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, w);
    int ry = RandomInt(0, h);
    int rw = RandomInt(0, w - rx);
    int rh = RandomInt(0, h - ry);
    color[0] = 100;
    color[1] = 20;
    color[2] = 200;
    color[3] = 100;
    ncnn::draw_rectangle_c4(a, w, h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c4(b, h, w, ry, rx, rh, rw, _color, 3);

    // draw filled rectangle out of image
    color[0] = 144;
    color[1] = 133;
    color[2] = 122;
    color[3] = 30;
    ncnn::draw_rectangle_c4(a, w, h, w - 10, -10, 20, 30, _color, -1);
    ncnn::draw_rectangle_c4(b, h, w, -10, w - 10, 30, 20, _color, -1);
    color[0] = 166;
    color[1] = 133;
    color[2] = 122;
    color[3] = 20;
    ncnn::draw_rectangle_c4(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c4(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 144;
    color[2] = 44;
    color[3] = 144;
    ncnn::draw_rectangle_c4(a, w, h, rx + w / 2, ry + h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c4(b, h, w, ry + h / 2, rx + w / 2, rh, rw, _color, 1);
    color[0] = 66;
    color[1] = 44;
    color[2] = 33;
    color[3] = 112;
    ncnn::draw_rectangle_c4(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 7);
    ncnn::draw_rectangle_c4(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 7);

    // draw filled circle
    int cx = RandomInt(0, w);
    int cy = RandomInt(0, h);
    int radius = RandomInt(0, std::min(w, h));
    color[0] = 10;
    color[1] = 2;
    color[2] = 200;
    color[3] = 20;
    ncnn::draw_circle_c4(a, w, h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c4(b, h, w, cy, cx, radius, _color, -1);

    // draw filled circle out of image
    color[0] = 230;
    color[1] = 130;
    color[2] = 110;
    color[3] = 5;
    ncnn::draw_circle_c4(a, w, h, 10, -4, 6, _color, -1);
    ncnn::draw_circle_c4(b, h, w, -4, 10, 6, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 255;
    color[2] = 130;
    color[3] = 255;
    ncnn::draw_circle_c4(a, w, h, cx, cy, radius + std::min(w, h) / 2, _color, 5);
    ncnn::draw_circle_c4(b, h, w, cy, cx, radius + std::min(w, h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, w);
    int y0 = RandomInt(0, h);
    int x1 = RandomInt(0, w);
    int y1 = RandomInt(0, h);
    color[0] = 233;
    color[1] = 233;
    color[2] = 233;
    color[3] = 233;
    ncnn::draw_line_c4(a, w, h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c4(b, h, w, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    color[1] = 22;
    color[2] = 1;
    color[3] = 0;
    ncnn::draw_line_c4(a, w, h, x0 - w, y0 - h, x1 + w, y1 + h, _color, 1);
    ncnn::draw_line_c4(b, h, w, y0 - h, x0 - w, y1 + h, x1 + w, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, (size_t)4u, 4);
    ncnn::kanna_rotate_c4(b, h, w, c, w, h, 5);

    // draw text
    const char text[] = "!@)\n($ 34\n2]\"M,";
    int tx = RandomInt(0, w / 2);
    int ty = RandomInt(0, h / 2);
    int fontpixelsize = 23;
    color[0] = 11;
    color[1] = 128;
    color[2] = 12;
    color[3] = 128;
    ncnn::draw_text_c4(a, w, h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 3; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, w, h, ch, tx + tw / 5 * i, ty, fontpixelsize, _color);
    }
    for (int i = 4; i < 9; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, w, h, ch, tx + tw / 5 * (i - 4), ty + th / 3, fontpixelsize, _color);
    }
    for (int i = 10; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, w, h, ch, tx + tw / 5 * (i - 10), ty + th / 3 * 2, fontpixelsize, _color);
    }

    // draw text out of image
    fontpixelsize = std::max(w, h) / 3;
    color[0] = 228;
    color[1] = 0;
    color[2] = 128;
    color[3] = 200;
    ncnn::draw_text_c4(a, w, h, "=_+!//zzzz", -13, -15, fontpixelsize, _color);
    ncnn::get_text_drawing_size("=_+!//zzzz", fontpixelsize, &tw, &th);
    ncnn::draw_text_c4(c, w, h, "=_+", -13, -15, fontpixelsize, _color);
    ncnn::draw_text_c4(c, w, h, "!", -13 + tw / 10 * 3, -15, fontpixelsize, _color);
    ncnn::draw_text_c4(c, w, h, "//zzzz", -13 + tw / 10 * 4, -15, fontpixelsize, _color);

    if (memcmp(a, c, w * h * 4) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c4 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_0()
{
    return 0
           || test_mat_pixel_drawing_c1(22, 33)
           || test_mat_pixel_drawing_c2(22, 23)
           || test_mat_pixel_drawing_c3(32, 23)
           || test_mat_pixel_drawing_c4(42, 13)

           || test_mat_pixel_drawing_c1(202, 303)
           || test_mat_pixel_drawing_c2(202, 203)
           || test_mat_pixel_drawing_c3(302, 203)
           || test_mat_pixel_drawing_c4(402, 103);
}

static int test_mat_pixel_drawing_yuv420sp(int w, int h)
{
    ncnn::Mat a(w, h * 3 / 2, (size_t)1u, 1);
    ncnn::Mat b(h, w * 3 / 2, (size_t)1u, 1);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 255;
    color[2] = 255;
    ncnn::draw_rectangle_yuv420sp(a, w, h, 0, 0, w, h, _color, -1);
    ncnn::draw_rectangle_yuv420sp(b, h, w, 0, 0, h, w, _color, -1);

    // draw rectangle
    int rx = RandomInt2(0, w);
    int ry = RandomInt2(0, h);
    int rw = RandomInt2(0, w - rx);
    int rh = RandomInt2(0, h - ry);
    color[0] = 100;
    color[1] = 20;
    color[2] = 200;
    ncnn::draw_rectangle_yuv420sp(a, w, h, rx, ry, rw, rh, _color, 4);
    ncnn::draw_rectangle_yuv420sp(b, h, w, ry, rx, rh, rw, _color, 4);

    // draw filled rectangle out of image
    color[0] = 144;
    color[1] = 133;
    color[2] = 122;
    ncnn::draw_rectangle_yuv420sp(a, w, h, w - 10, -10, 20, 30, _color, -1);
    ncnn::draw_rectangle_yuv420sp(b, h, w, -10, w - 10, 30, 20, _color, -1);
    color[0] = 166;
    color[1] = 133;
    color[2] = 122;
    ncnn::draw_rectangle_yuv420sp(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 8);
    ncnn::draw_rectangle_yuv420sp(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 8);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 144;
    color[2] = 44;
    ncnn::draw_rectangle_yuv420sp(a, w, h, rx + w / 2, ry + h / 2, rw, rh, _color, 2);
    ncnn::draw_rectangle_yuv420sp(b, h, w, ry + h / 2, rx + w / 2, rh, rw, _color, 2);
    color[0] = 66;
    color[1] = 44;
    color[2] = 33;
    ncnn::draw_rectangle_yuv420sp(a, w, h, -rw / 2, -rh / 3, rw, rh, _color, 8);
    ncnn::draw_rectangle_yuv420sp(b, h, w, -rh / 3, -rw / 2, rh, rw, _color, 8);

    // draw filled circle
    int cx = RandomInt2(0, w);
    int cy = RandomInt2(0, h);
    int radius = RandomInt2(0, std::min(w, h));
    color[0] = 10;
    color[1] = 2;
    color[2] = 200;
    ncnn::draw_circle_yuv420sp(a, w, h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_yuv420sp(b, h, w, cy, cx, radius, _color, -1);

    // draw filled circle out of image
    color[0] = 230;
    color[1] = 130;
    color[2] = 110;
    ncnn::draw_circle_yuv420sp(a, w, h, 10, -4, 6, _color, -1);
    ncnn::draw_circle_yuv420sp(b, h, w, -4, 10, 6, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 255;
    color[2] = 130;
    ncnn::draw_circle_yuv420sp(a, w, h, cx, cy, radius + std::min(w, h) / 2, _color, 6);
    ncnn::draw_circle_yuv420sp(b, h, w, cy, cx, radius + std::min(w, h) / 2, _color, 6);

    // draw line
    int x0 = RandomInt2(0, w);
    int y0 = RandomInt2(0, h);
    int x1 = RandomInt2(0, w);
    int y1 = RandomInt2(0, h);
    color[0] = 233;
    color[1] = 233;
    color[2] = 233;
    ncnn::draw_line_yuv420sp(a, w, h, x0, y0, x1, y1, _color, 8);
    ncnn::draw_line_yuv420sp(b, h, w, y0, x0, y1, x1, _color, 8);

    // draw line out of image
    color[0] = 192;
    color[1] = 22;
    color[2] = 1;
    ncnn::draw_line_yuv420sp(a, w, h, x0 - w, y0 - h, x1 + w, y1 + h, _color, 2);
    ncnn::draw_line_yuv420sp(b, h, w, y0 - h, x0 - w, y1 + h, x1 + w, _color, 2);

    // transpose b
    ncnn::Mat c(w, h * 3 / 2, (size_t)1u, 1);
    ncnn::kanna_rotate_yuv420sp(b, h, w, c, w, h, 5);

    // draw text
    const char text[] = "!@)\n($ 34\n2]\"M,";
    int tx = RandomInt2(0, w / 2);
    int ty = RandomInt2(0, h / 2);
    int fontpixelsize = 24;
    color[0] = 11;
    color[1] = 128;
    color[2] = 12;
    ncnn::draw_text_yuv420sp(a, w, h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 3; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_yuv420sp(c, w, h, ch, tx + tw / 5 * i, ty, fontpixelsize, _color);
    }
    for (int i = 4; i < 9; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_yuv420sp(c, w, h, ch, tx + tw / 5 * (i - 4), ty + th / 3, fontpixelsize, _color);
    }
    for (int i = 10; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_yuv420sp(c, w, h, ch, tx + tw / 5 * (i - 10), ty + th / 3 * 2, fontpixelsize, _color);
    }

    // draw text out of image
    fontpixelsize = (std::max(w, h) / 3 + 1) / 2 * 2;
    color[0] = 228;
    color[1] = 0;
    color[2] = 128;
    ncnn::draw_text_yuv420sp(a, w, h, "=_+!//zzzz", -14, -12, fontpixelsize, _color);
    ncnn::get_text_drawing_size("=_+!//zzzz", fontpixelsize, &tw, &th);
    ncnn::draw_text_yuv420sp(c, w, h, "=_+", -14, -12, fontpixelsize, _color);
    ncnn::draw_text_yuv420sp(c, w, h, "!", -14 + tw / 10 * 3, -12, fontpixelsize, _color);
    ncnn::draw_text_yuv420sp(c, w, h, "//zzzz", -14 + tw / 10 * 4, -12, fontpixelsize, _color);

    if (memcmp(a, c, w * h * 3 / 2) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_yuv420sp failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_1()
{
    return 0
           || test_mat_pixel_drawing_yuv420sp(10, 10)
           || test_mat_pixel_drawing_yuv420sp(120, 160)
           || test_mat_pixel_drawing_yuv420sp(220, 340);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mat_pixel_drawing_0()
           || test_mat_pixel_drawing_1();
}
