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

static int test_mat_pixel_drawing_c1(int w, int h)
{
    ncnn::Mat a(w, h, 1u, 1);
    ncnn::Mat b(h, w, 1u, 1);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    ncnn::draw_rectangle_c1(a, a.w, a.h, 0, 0, a.w, a.h, _color, -1);
    ncnn::draw_rectangle_c1(b, b.w, b.h, 0, 0, b.w, b.h, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, a.w);
    int ry = RandomInt(0, a.h);
    int rw = RandomInt(0, a.w - rx);
    int rh = RandomInt(0, a.h - ry);
    color[0] = 100;
    ncnn::draw_rectangle_c1(a, a.w, a.h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c1(b, b.w, b.h, ry, rx, rh, rw, _color, 3);

    // draw rectangle out of image
    color[0] = 44;
    ncnn::draw_rectangle_c1(a, a.w, a.h, rx + a.w / 2, ry + a.h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c1(b, b.w, b.h, ry + b.w / 2, rx + b.h / 2, rh, rw, _color, 1);

    // draw filled circle
    int cx = RandomInt(0, a.w);
    int cy = RandomInt(0, a.h);
    int radius = RandomInt(0, std::min(a.w, a.h));
    color[0] = 20;
    ncnn::draw_circle_c1(a, a.w, a.h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c1(b, b.w, b.h, cy, cx, radius, _color, -1);

    // draw circle out of image
    color[0] = 130;
    ncnn::draw_circle_c1(a, a.w, a.h, cx, cy, radius + std::min(a.w, a.h) / 2, _color, 5);
    ncnn::draw_circle_c1(b, b.w, b.h, cy, cx, radius + std::min(a.w, a.h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, a.w);
    int y0 = RandomInt(0, a.h);
    int x1 = RandomInt(0, a.w);
    int y1 = RandomInt(0, a.h);
    color[0] = 233;
    ncnn::draw_line_c1(a, a.w, a.h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c1(b, b.w, b.h, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    ncnn::draw_line_c1(a, a.w, a.h, x0 - a.w, y0 - a.h, x1 + a.w, y1 + a.h, _color, 1);
    ncnn::draw_line_c1(b, b.w, b.h, y0 - b.w, x0 - b.h, y1 + b.w, x1 + b.h, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, 1u, 1);
    ncnn::kanna_rotate_c1(b, b.w, b.h, c, c.w, c.h, 5);

    // draw text
    const char text[] = "saJIEWdl\nj43@o";
    int tx = RandomInt(0, a.w / 2);
    int ty = RandomInt(0, a.h / 2);
    int fontpixelsize = 10;
    color[0] = 128;
    ncnn::draw_text_c1(a, a.w, a.h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 8; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c1(c, c.w, c.h, ch, tx + tw / 8 * i, ty, fontpixelsize, _color);
    }
    for (int i = 9; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c1(c, c.w, c.h, ch, tx + tw / 8 * (i - 9), ty + th / 2, fontpixelsize, _color);
    }

    if (memcmp(a, c, w * h) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c1 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_c3(int w, int h)
{
    ncnn::Mat a(w, h, 3u, 3);
    ncnn::Mat b(h, w, 3u, 3);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 251;
    color[2] = 244;
    ncnn::draw_rectangle_c3(a, a.w, a.h, 0, 0, a.w, a.h, _color, -1);
    ncnn::draw_rectangle_c3(b, b.w, b.h, 0, 0, b.w, b.h, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, a.w);
    int ry = RandomInt(0, a.h);
    int rw = RandomInt(0, a.w - rx);
    int rh = RandomInt(0, a.h - ry);
    color[0] = 100;
    color[1] = 130;
    color[2] = 150;
    ncnn::draw_rectangle_c3(a, a.w, a.h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c3(b, b.w, b.h, ry, rx, rh, rw, _color, 3);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 33;
    color[2] = 22;
    ncnn::draw_rectangle_c3(a, a.w, a.h, rx + a.w / 2, ry + a.h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c3(b, b.w, b.h, ry + b.w / 2, rx + b.h / 2, rh, rw, _color, 1);

    // draw filled circle
    int cx = RandomInt(0, a.w);
    int cy = RandomInt(0, a.h);
    int radius = RandomInt(0, std::min(a.w, a.h));
    color[0] = 20;
    color[1] = 120;
    color[2] = 220;
    ncnn::draw_circle_c3(a, a.w, a.h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c3(b, b.w, b.h, cy, cx, radius, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 30;
    color[2] = 230;
    ncnn::draw_circle_c3(a, a.w, a.h, cx, cy, radius + std::min(a.w, a.h) / 2, _color, 5);
    ncnn::draw_circle_c3(b, b.w, b.h, cy, cx, radius + std::min(a.w, a.h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, a.w);
    int y0 = RandomInt(0, a.h);
    int x1 = RandomInt(0, a.w);
    int y1 = RandomInt(0, a.h);
    color[0] = 233;
    color[1] = 233;
    color[2] = 233;
    ncnn::draw_line_c3(a, a.w, a.h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c3(b, b.w, b.h, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    color[1] = 192;
    color[2] = 0;
    ncnn::draw_line_c3(a, a.w, a.h, x0 - a.w, y0 - a.h, x1 + a.w, y1 + a.h, _color, 1);
    ncnn::draw_line_c3(b, b.w, b.h, y0 - b.w, x0 - b.h, y1 + b.w, x1 + b.h, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, 3u, 3);
    ncnn::kanna_rotate_c3(b, b.w, b.h, c, c.w, c.h, 5);

    // draw text
    const char text[] = "Q`~\\=f\nPN\'/<DSA";
    int tx = RandomInt(0, a.w / 2);
    int ty = RandomInt(0, a.h / 2);
    int fontpixelsize = 12;
    color[0] = 0;
    color[1] = 128;
    color[2] = 128;
    ncnn::draw_text_c3(a, a.w, a.h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 6; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c3(c, c.w, c.h, ch, tx + tw / 8 * i, ty, fontpixelsize, _color);
    }
    for (int i = 7; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c3(c, c.w, c.h, ch, tx + tw / 8 * (i - 7), ty + th / 2, fontpixelsize, _color);
    }

    if (memcmp(a, c, w * h * 3) != 0)
    {
        fprintf(stderr, "test_mat_pixel_drawing_c3 failed w=%d h=%d\n", w, h);
        return -1;
    }

    return 0;
}

static int test_mat_pixel_drawing_c4(int w, int h)
{
    ncnn::Mat a(w, h, 4u, 4);
    ncnn::Mat b(h, w, 4u, 4);

    int _color = 0;
    unsigned char* color = (unsigned char*)&_color;

    // fill with color
    color[0] = 255;
    color[1] = 255;
    color[2] = 255;
    color[3] = 0;
    ncnn::draw_rectangle_c4(a, a.w, a.h, 0, 0, a.w, a.h, _color, -1);
    ncnn::draw_rectangle_c4(b, b.w, b.h, 0, 0, b.w, b.h, _color, -1);

    // draw rectangle
    int rx = RandomInt(0, a.w);
    int ry = RandomInt(0, a.h);
    int rw = RandomInt(0, a.w - rx);
    int rh = RandomInt(0, a.h - ry);
    color[0] = 100;
    color[1] = 20;
    color[2] = 200;
    color[3] = 100;
    ncnn::draw_rectangle_c4(a, a.w, a.h, rx, ry, rw, rh, _color, 3);
    ncnn::draw_rectangle_c4(b, b.w, b.h, ry, rx, rh, rw, _color, 3);

    // draw rectangle out of image
    color[0] = 44;
    color[1] = 144;
    color[2] = 44;
    color[3] = 144;
    ncnn::draw_rectangle_c4(a, a.w, a.h, rx + a.w / 2, ry + a.h / 2, rw, rh, _color, 1);
    ncnn::draw_rectangle_c4(b, b.w, b.h, ry + b.w / 2, rx + b.h / 2, rh, rw, _color, 1);

    // draw filled circle
    int cx = RandomInt(0, a.w);
    int cy = RandomInt(0, a.h);
    int radius = RandomInt(0, std::min(a.w, a.h));
    color[0] = 10;
    color[1] = 2;
    color[2] = 200;
    color[3] = 20;
    ncnn::draw_circle_c4(a, a.w, a.h, cx, cy, radius, _color, -1);
    ncnn::draw_circle_c4(b, b.w, b.h, cy, cx, radius, _color, -1);

    // draw circle out of image
    color[0] = 130;
    color[1] = 255;
    color[2] = 130;
    color[3] = 255;
    ncnn::draw_circle_c4(a, a.w, a.h, cx, cy, radius + std::min(a.w, a.h) / 2, _color, 5);
    ncnn::draw_circle_c4(b, b.w, b.h, cy, cx, radius + std::min(a.w, a.h) / 2, _color, 5);

    // draw line
    int x0 = RandomInt(0, a.w);
    int y0 = RandomInt(0, a.h);
    int x1 = RandomInt(0, a.w);
    int y1 = RandomInt(0, a.h);
    color[0] = 233;
    color[1] = 233;
    color[2] = 233;
    color[3] = 233;
    ncnn::draw_line_c4(a, a.w, a.h, x0, y0, x1, y1, _color, 7);
    ncnn::draw_line_c4(b, b.w, b.h, y0, x0, y1, x1, _color, 7);

    // draw line out of image
    color[0] = 192;
    color[1] = 22;
    color[2] = 1;
    color[3] = 0;
    ncnn::draw_line_c4(a, a.w, a.h, x0 - a.w, y0 - a.h, x1 + a.w, y1 + a.h, _color, 1);
    ncnn::draw_line_c4(b, b.w, b.h, y0 - b.w, x0 - b.h, y1 + b.w, x1 + b.h, _color, 1);

    // transpose b
    ncnn::Mat c(w, h, 4u, 4);
    ncnn::kanna_rotate_c4(b, b.w, b.h, c, c.w, c.h, 5);

    // draw text
    const char text[] = "!@)\n($ 34\n2]\"M,";
    int tx = RandomInt(0, a.w / 2);
    int ty = RandomInt(0, a.h / 2);
    int fontpixelsize = 23;
    color[0] = 11;
    color[1] = 128;
    color[2] = 12;
    color[3] = 128;
    ncnn::draw_text_c4(a, a.w, a.h, text, tx, ty, fontpixelsize, _color);
    int tw;
    int th;
    ncnn::get_text_drawing_size(text, fontpixelsize, &tw, &th);
    const int len = strlen(text);
    for (int i = 0; i < 3; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, c.w, c.h, ch, tx + tw / 5 * i, ty, fontpixelsize, _color);
    }
    for (int i = 4; i < 9; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, c.w, c.h, ch, tx + tw / 5 * (i - 4), ty + th / 3, fontpixelsize, _color);
    }
    for (int i = 10; i < len; i++)
    {
        const char ch[2] = {text[i], '\0'};
        ncnn::draw_text_c4(c, c.w, c.h, ch, tx + tw / 5 * (i - 10), ty + th / 3 * 2, fontpixelsize, _color);
    }

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
           || test_mat_pixel_drawing_c1(202, 303)
           || test_mat_pixel_drawing_c3(302, 203)
           || test_mat_pixel_drawing_c4(402, 103);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mat_pixel_drawing_0();
}
