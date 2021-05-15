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

#include <ctype.h>
#include <math.h>

#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_DRAWING

#include "mat_pixel_drawing_font.h"

void draw_rectangle_c1(unsigned char* src, int srcw, int srch, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c1(src, srcw, srch, srcw, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c3(unsigned char* src, int srcw, int srch, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c3(src, srcw, srch, srcw * 3, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c4(unsigned char* src, int srcw, int srch, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c4(src, srcw, srch, srcw * 4, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c1(unsigned char* src, int srcw, int srch, int srcstride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x] = border_color[0];
            }
        }

        return;
    }

    const int t0 = thickness / 2;
    const int t1 = thickness - t0;

    // draw top
    {
        for (int y = ry - t0; y < ry + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x] = border_color[0];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x] = border_color[0];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x] = border_color[0];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x] = border_color[0];
        }
    }
}

void draw_rectangle_c3(unsigned char* src, int srcw, int srch, int srcstride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 3 + 0] = border_color[0];
                p[x * 3 + 1] = border_color[1];
                p[x * 3 + 2] = border_color[2];
            }
        }

        return;
    }

    const int t0 = thickness / 2;
    const int t1 = thickness - t0;

    // draw top
    {
        for (int y = ry - t0; y < ry + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 3 + 0] = border_color[0];
                p[x * 3 + 1] = border_color[1];
                p[x * 3 + 2] = border_color[2];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 3 + 0] = border_color[0];
                p[x * 3 + 1] = border_color[1];
                p[x * 3 + 2] = border_color[2];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x * 3 + 0] = border_color[0];
            p[x * 3 + 1] = border_color[1];
            p[x * 3 + 2] = border_color[2];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x * 3 + 0] = border_color[0];
            p[x * 3 + 1] = border_color[1];
            p[x * 3 + 2] = border_color[2];
        }
    }
}

void draw_rectangle_c4(unsigned char* src, int srcw, int srch, int srcstride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 4 + 0] = border_color[0];
                p[x * 4 + 1] = border_color[1];
                p[x * 4 + 2] = border_color[2];
                p[x * 4 + 3] = border_color[3];
            }
        }

        return;
    }

    const int t0 = thickness / 2;
    const int t1 = thickness - t0;

    // draw top
    {
        for (int y = ry - t0; y < ry + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 4 + 0] = border_color[0];
                p[x * 4 + 1] = border_color[1];
                p[x * 4 + 2] = border_color[2];
                p[x * 4 + 3] = border_color[3];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                p[x * 4 + 0] = border_color[0];
                p[x * 4 + 1] = border_color[1];
                p[x * 4 + 2] = border_color[2];
                p[x * 4 + 3] = border_color[3];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x * 4 + 0] = border_color[0];
            p[x * 4 + 1] = border_color[1];
            p[x * 4 + 2] = border_color[2];
            p[x * 4 + 3] = border_color[3];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= srcw)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            p[x * 4 + 0] = border_color[0];
            p[x * 4 + 1] = border_color[1];
            p[x * 4 + 2] = border_color[2];
            p[x * 4 + 3] = border_color[3];
        }
    }
}

static inline float distance(int x0, int y0, int x1, int y1)
{
    int dx = x0 - x1;
    int dy = y0 - y1;
    return (float)sqrt(dx * dx + dy * dy);
}

void draw_circle_c1(unsigned char* src, int srcw, int srch, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c1(src, srcw, srch, srcw, cx, cy, radius, color, thickness);
}

void draw_circle_c3(unsigned char* src, int srcw, int srch, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c3(src, srcw, srch, srcw * 3, cx, cy, radius, color, thickness);
}

void draw_circle_c4(unsigned char* src, int srcw, int srch, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c4(src, srcw, srch, srcw * 4, cx, cy, radius, color, thickness);
}

void draw_circle_c1(unsigned char* src, int srcw, int srch, int srcstride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                // distance from cx cy
                float dis = distance(x, y, cx, cy);
                if (dis <= radius)
                {
                    p[x] = border_color[0];
                }
            }
        }

        return;
    }

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    for (int y = cy - (radius - 1) - t0; y < cy + radius + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= srch)
            break;

        unsigned char* p = src + srcstride * y;

        for (int x = cx - (radius - 1) - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= srcw)
                break;

            // distance from cx cy
            float dis = distance(x, y, cx, cy);
            if (dis >= radius - t0 && dis < radius + t1)
            {
                p[x] = border_color[0];
            }
        }
    }
}

void draw_circle_c3(unsigned char* src, int srcw, int srch, int srcstride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                // distance from cx cy
                float dis = distance(x, y, cx, cy);
                if (dis <= radius)
                {
                    p[x * 3 + 0] = border_color[0];
                    p[x * 3 + 1] = border_color[1];
                    p[x * 3 + 2] = border_color[2];
                }
            }
        }

        return;
    }

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    for (int y = cy - radius - t0; y < cy + radius + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= srch)
            break;

        unsigned char* p = src + srcstride * y;

        for (int x = cx - radius - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= srcw)
                break;

            // distance from cx cy
            float dis = distance(x, y, cx, cy);
            if (dis >= radius - t0 && dis < radius + t1)
            {
                p[x * 3 + 0] = border_color[0];
                p[x * 3 + 1] = border_color[1];
                p[x * 3 + 2] = border_color[2];
            }
        }
    }
}

void draw_circle_c4(unsigned char* src, int srcw, int srch, int srcstride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= srch)
                break;

            unsigned char* p = src + srcstride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= srcw)
                    break;

                // distance from cx cy
                float dis = distance(x, y, cx, cy);
                if (dis <= radius)
                {
                    p[x * 4 + 0] = border_color[0];
                    p[x * 4 + 1] = border_color[1];
                    p[x * 4 + 2] = border_color[2];
                    p[x * 4 + 3] = border_color[3];
                }
            }
        }

        return;
    }

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    for (int y = cy - (radius - 1) - t0; y < cy + radius + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= srch)
            break;

        unsigned char* p = src + srcstride * y;

        for (int x = cx - (radius - 1) - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= srcw)
                break;

            // distance from cx cy
            float dis = distance(x, y, cx, cy);
            if (dis >= radius - t0 && dis < radius + t1)
            {
                p[x * 4 + 0] = border_color[0];
                p[x * 4 + 1] = border_color[1];
                p[x * 4 + 2] = border_color[2];
                p[x * 4 + 3] = border_color[3];
            }
        }
    }
}

void get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h)
{
    *w = 0;
    *h = 0;

    const int n = strlen(text);

    int line_w = 0;
    for (int i = 0; i < n; i++)
    {
        char ch = text[i];

        if (ch == '\n')
        {
            // newline
            *w = std::max(*w, line_w);
            *h += fontpixelsize * 2;
            line_w = 0;
        }

        if (isprint(ch) != 0)
        {
            line_w += fontpixelsize;
        }
    }

    *w = std::max(*w, line_w);
    *h += fontpixelsize * 2;
}

void draw_text_c1(unsigned char* src, int srcw, int srch, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c1(src, srcw, srch, srcw, text, x, y, fontpixelsize, color);
}

void draw_text_c3(unsigned char* src, int srcw, int srch, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c3(src, srcw, srch, srcw * 3, text, x, y, fontpixelsize, color);
}

void draw_text_c4(unsigned char* src, int srcw, int srch, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c4(src, srcw, srch, srcw * 4, text, x, y, fontpixelsize, color);
}

void draw_text_c1(unsigned char* src, int srcw, int srch, int srcstride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    unsigned char* resized_font_bitmap = new unsigned char[fontpixelsize * fontpixelsize * 2];

    const int n = strlen(text);

    int cursor_x = x;
    int cursor_y = y;
    for (int i = 0; i < n; i++)
    {
        char ch = text[i];

        if (ch == '\n')
        {
            // newline
            cursor_x = x;
            cursor_y += fontpixelsize * 2;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - ' '];

            // draw resized character
            resize_bilinear_c1(font_bitmap, 20, 40, resized_font_bitmap, fontpixelsize, fontpixelsize * 2);

            for (int j = cursor_y; j < cursor_y + fontpixelsize * 2; j++)
            {
                if (j < 0)
                    continue;

                if (j >= srch)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = src + srcstride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= srcw)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k] = (p[k] * (255 - alpha) + border_color[0] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_c3(unsigned char* src, int srcw, int srch, int srcstride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    unsigned char* resized_font_bitmap = new unsigned char[fontpixelsize * fontpixelsize * 2];

    const int n = strlen(text);

    int cursor_x = x;
    int cursor_y = y;
    for (int i = 0; i < n; i++)
    {
        char ch = text[i];

        fprintf(stderr, "cc %c\n", ch);

        if (ch == '\n')
        {
            // newline
            cursor_x = x;
            cursor_y += fontpixelsize * 2;
        }

        if (isprint(ch) != 0)
        {
            int font_bitmap_index = ch - ' ';
            const unsigned char* font_bitmap = mono_font_data[font_bitmap_index];

            fprintf(stderr, "font_bitmap_index %d\n", font_bitmap_index);

            // draw resized character
            resize_bilinear_c1(font_bitmap, 20, 40, resized_font_bitmap, fontpixelsize, fontpixelsize * 2);

            for (int j = cursor_y; j < cursor_y + fontpixelsize * 2; j++)
            {
                if (j < 0)
                    continue;

                if (j >= srch)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = src + srcstride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= srcw)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k * 3 + 0] = (p[k * 3 + 0] * (255 - alpha) + border_color[0] * alpha) / 255;
                    p[k * 3 + 1] = (p[k * 3 + 1] * (255 - alpha) + border_color[1] * alpha) / 255;
                    p[k * 3 + 2] = (p[k * 3 + 2] * (255 - alpha) + border_color[2] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_c4(unsigned char* src, int srcw, int srch, int srcstride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* border_color = (const unsigned char*)&color;

    unsigned char* resized_font_bitmap = new unsigned char[fontpixelsize * fontpixelsize * 2];

    const int n = strlen(text);

    int cursor_x = x;
    int cursor_y = y;
    for (int i = 0; i < n; i++)
    {
        char ch = text[i];

        if (ch == '\n')
        {
            // newline
            cursor_x = x;
            cursor_y += fontpixelsize * 2;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - ' '];

            // draw resized character
            resize_bilinear_c1(font_bitmap, 20, 40, resized_font_bitmap, fontpixelsize, fontpixelsize * 2);

            for (int j = cursor_y; j < cursor_y + fontpixelsize * 2; j++)
            {
                if (j < 0)
                    continue;

                if (j >= srch)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = src + srcstride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= srcw)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k * 4 + 0] = (p[k * 4 + 0] * (255 - alpha) + border_color[0] * alpha) / 255;
                    p[k * 4 + 1] = (p[k * 4 + 1] * (255 - alpha) + border_color[1] * alpha) / 255;
                    p[k * 4 + 2] = (p[k * 4 + 2] * (255 - alpha) + border_color[2] * alpha) / 255;
                    p[k * 4 + 3] = (p[k * 4 + 3] * (255 - alpha) + border_color[3] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

#endif // NCNN_PIXEL_DRAWING

} // namespace ncnn
