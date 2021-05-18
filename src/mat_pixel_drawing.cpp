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

#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_DRAWING

#include "mat_pixel_drawing_font.h"

void draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c1(pixels, w, h, w, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c2(pixels, w, h, w * 2, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c3(pixels, w, h, w * 3, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    return draw_rectangle_c4(pixels, w, h, w * 4, rx, ry, rw, rh, color, thickness);
}

void draw_rectangle_c1(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x] = pen_color[0];
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

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x] = pen_color[0];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x] = pen_color[0];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x] = pen_color[0];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x] = pen_color[0];
        }
    }
}

void draw_rectangle_c2(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 2 + 0] = pen_color[0];
                p[x * 2 + 1] = pen_color[1];
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

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 2 + 0] = pen_color[0];
                p[x * 2 + 1] = pen_color[1];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 2 + 0] = pen_color[0];
                p[x * 2 + 1] = pen_color[1];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 2 + 0] = pen_color[0];
            p[x * 2 + 1] = pen_color[1];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 2 + 0] = pen_color[0];
            p[x * 2 + 1] = pen_color[1];
        }
    }
}

void draw_rectangle_c3(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 3 + 0] = pen_color[0];
                p[x * 3 + 1] = pen_color[1];
                p[x * 3 + 2] = pen_color[2];
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

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 3 + 0] = pen_color[0];
                p[x * 3 + 1] = pen_color[1];
                p[x * 3 + 2] = pen_color[2];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 3 + 0] = pen_color[0];
                p[x * 3 + 1] = pen_color[1];
                p[x * 3 + 2] = pen_color[2];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 3 + 0] = pen_color[0];
            p[x * 3 + 1] = pen_color[1];
            p[x * 3 + 2] = pen_color[2];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 3 + 0] = pen_color[0];
            p[x * 3 + 1] = pen_color[1];
            p[x * 3 + 2] = pen_color[2];
        }
    }
}

void draw_rectangle_c4(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = ry; y < ry + rh; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx; x < rx + rw; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 4 + 0] = pen_color[0];
                p[x * 4 + 1] = pen_color[1];
                p[x * 4 + 2] = pen_color[2];
                p[x * 4 + 3] = pen_color[3];
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

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 4 + 0] = pen_color[0];
                p[x * 4 + 1] = pen_color[1];
                p[x * 4 + 2] = pen_color[2];
                p[x * 4 + 3] = pen_color[3];
            }
        }
    }

    // draw bottom
    {
        for (int y = ry + rh - t0; y < ry + rh + t1; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = rx - t0; x < rx + rw + t1; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                p[x * 4 + 0] = pen_color[0];
                p[x * 4 + 1] = pen_color[1];
                p[x * 4 + 2] = pen_color[2];
                p[x * 4 + 3] = pen_color[3];
            }
        }
    }

    // draw left
    for (int x = rx - t0; x < rx + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 4 + 0] = pen_color[0];
            p[x * 4 + 1] = pen_color[1];
            p[x * 4 + 2] = pen_color[2];
            p[x * 4 + 3] = pen_color[3];
        }
    }

    // draw right
    for (int x = rx + rw - t0; x < rx + rw + t1; x++)
    {
        if (x < 0)
            continue;

        if (x >= w)
            break;

        for (int y = ry + t1; y < ry + rh - t0; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            p[x * 4 + 0] = pen_color[0];
            p[x * 4 + 1] = pen_color[1];
            p[x * 4 + 2] = pen_color[2];
            p[x * 4 + 3] = pen_color[3];
        }
    }
}

void draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    // assert w % 2 == 0
    // assert h % 2 == 0
    // assert rx % 2 == 0
    // assert ry % 2 == 0
    // assert rw % 2 == 0
    // assert rh % 2 == 0
    // assert thickness % 2 == 0

    const unsigned char* pen_color = (const unsigned char*)&color;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* pen_color_y = (unsigned char*)&v_y;
    unsigned char* pen_color_uv = (unsigned char*)&v_uv;
    pen_color_y[0] = pen_color[0];
    pen_color_uv[0] = pen_color[1];
    pen_color_uv[1] = pen_color[2];

    unsigned char* Y = yuv420sp;
    draw_rectangle_c1(Y, w, h, rx, ry, rw, rh, v_y, thickness);

    unsigned char* UV = yuv420sp + w * h;
    int thickness_uv = thickness == -1 ? thickness : std::max(thickness / 2, 1);
    draw_rectangle_c2(UV, w / 2, h / 2, rx / 2, ry / 2, rw / 2, rh / 2, v_uv, thickness_uv);
}

static inline bool distance_lessequal(int x0, int y0, int x1, int y1, float r)
{
    int dx = x0 - x1;
    int dy = y0 - y1;
    int q = dx * dx + dy * dy;
    return q <= r * r;
}

static inline bool distance_inrange(int x0, int y0, int x1, int y1, float r0, float r1)
{
    int dx = x0 - x1;
    int dy = y0 - y1;
    int q = dx * dx + dy * dy;
    return q >= r0 * r0 && q < r1 * r1;
}

void draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c1(pixels, w, h, w, cx, cy, radius, color, thickness);
}

void draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c2(pixels, w, h, w * 2, cx, cy, radius, color, thickness);
}

void draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c3(pixels, w, h, w * 3, cx, cy, radius, color, thickness);
}

void draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    return draw_circle_c4(pixels, w, h, w * 4, cx, cy, radius, color, thickness);
}

void draw_circle_c1(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                // distance from cx cy
                if (distance_lessequal(x, y, cx, cy, radius))
                {
                    p[x] = pen_color[0];
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

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = cx - (radius - 1) - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from cx cy
            if (distance_inrange(x, y, cx, cy, radius - t0, radius + t1))
            {
                p[x] = pen_color[0];
            }
        }
    }
}

void draw_circle_c2(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                // distance from cx cy
                if (distance_lessequal(x, y, cx, cy, radius))
                {
                    p[x * 2 + 0] = pen_color[0];
                    p[x * 2 + 1] = pen_color[1];
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

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = cx - radius - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from cx cy
            if (distance_inrange(x, y, cx, cy, radius - t0, radius + t1))
            {
                p[x * 2 + 0] = pen_color[0];
                p[x * 2 + 1] = pen_color[1];
            }
        }
    }
}

void draw_circle_c3(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                // distance from cx cy
                if (distance_lessequal(x, y, cx, cy, radius))
                {
                    p[x * 3 + 0] = pen_color[0];
                    p[x * 3 + 1] = pen_color[1];
                    p[x * 3 + 2] = pen_color[2];
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

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = cx - radius - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from cx cy
            if (distance_inrange(x, y, cx, cy, radius - t0, radius + t1))
            {
                p[x * 3 + 0] = pen_color[0];
                p[x * 3 + 1] = pen_color[1];
                p[x * 3 + 2] = pen_color[2];
            }
        }
    }
}

void draw_circle_c4(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    if (thickness == -1)
    {
        // filled
        for (int y = cy - (radius - 1); y < cy + radius; y++)
        {
            if (y < 0)
                continue;

            if (y >= h)
                break;

            unsigned char* p = pixels + stride * y;

            for (int x = cx - (radius - 1); x < cx + radius; x++)
            {
                if (x < 0)
                    continue;

                if (x >= w)
                    break;

                // distance from cx cy
                if (distance_lessequal(x, y, cx, cy, radius))
                {
                    p[x * 4 + 0] = pen_color[0];
                    p[x * 4 + 1] = pen_color[1];
                    p[x * 4 + 2] = pen_color[2];
                    p[x * 4 + 3] = pen_color[3];
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

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = cx - (radius - 1) - t0; x < cx + radius + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from cx cy
            if (distance_inrange(x, y, cx, cy, radius - t0, radius + t1))
            {
                p[x * 4 + 0] = pen_color[0];
                p[x * 4 + 1] = pen_color[1];
                p[x * 4 + 2] = pen_color[2];
                p[x * 4 + 3] = pen_color[3];
            }
        }
    }
}

void draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    // assert w % 2 == 0
    // assert h % 2 == 0
    // assert cx % 2 == 0
    // assert cy % 2 == 0
    // assert radius % 2 == 0
    // assert thickness % 2 == 0

    const unsigned char* pen_color = (const unsigned char*)&color;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* pen_color_y = (unsigned char*)&v_y;
    unsigned char* pen_color_uv = (unsigned char*)&v_uv;
    pen_color_y[0] = pen_color[0];
    pen_color_uv[0] = pen_color[1];
    pen_color_uv[1] = pen_color[2];

    unsigned char* Y = yuv420sp;
    draw_circle_c1(Y, w, h, cx, cy, radius, v_y, thickness);

    unsigned char* UV = yuv420sp + w * h;
    int thickness_uv = thickness == -1 ? thickness : std::max(thickness / 2, 1);
    draw_circle_c2(UV, w / 2, h / 2, cx / 2, cy / 2, radius / 2, v_uv, thickness_uv);
}

static inline bool distance_lessthan(int x, int y, int x0, int y0, int x1, int y1, float t)
{
    int dx01 = x1 - x0;
    int dy01 = y1 - y0;
    int dx0 = x - x0;
    int dy0 = y - y0;

    float r = (float)(dx0 * dx01 + dy0 * dy01) / (dx01 * dx01 + dy01 * dy01);

    if (r < 0 || r > 1)
        return false;

    float px = x0 + dx01 * r;
    float py = y0 + dy01 * r;
    float dx = x - px;
    float dy = y - py;
    float p = dx * dx + dy * dy;
    return p < t;
}

void draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    draw_line_c1(pixels, w, h, w, x0, y0, x1, y1, color, thickness);
}

void draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    draw_line_c2(pixels, w, h, w * 2, x0, y0, x1, y1, color, thickness);
}

void draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    draw_line_c3(pixels, w, h, w * 3, x0, y0, x1, y1, color, thickness);
}

void draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    draw_line_c4(pixels, w, h, w * 4, x0, y0, x1, y1, color, thickness);
}

void draw_line_c1(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    int x_min = std::min(x0, x1);
    int x_max = std::max(x0, x1);
    int y_min = std::min(y0, y1);
    int y_max = std::max(y0, y1);

    for (int y = y_min - t0; y < y_max + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = x_min - t0; x < x_max + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from line
            if (distance_lessthan(x, y, x0, y0, x1, y1, t1))
            {
                p[x] = pen_color[0];
            }
        }
    }
}

void draw_line_c2(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    int x_min = std::min(x0, x1);
    int x_max = std::max(x0, x1);
    int y_min = std::min(y0, y1);
    int y_max = std::max(y0, y1);

    for (int y = y_min - t0; y < y_max + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = x_min - t0; x < x_max + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from line
            if (distance_lessthan(x, y, x0, y0, x1, y1, t1))
            {
                p[x * 2 + 0] = pen_color[0];
                p[x * 2 + 1] = pen_color[1];
            }
        }
    }
}

void draw_line_c3(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    int x_min = std::min(x0, x1);
    int x_max = std::max(x0, x1);
    int y_min = std::min(y0, y1);
    int y_max = std::max(y0, y1);

    for (int y = y_min - t0; y < y_max + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = x_min - t0; x < x_max + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from line
            if (distance_lessthan(x, y, x0, y0, x1, y1, t1))
            {
                p[x * 3 + 0] = pen_color[0];
                p[x * 3 + 1] = pen_color[1];
                p[x * 3 + 2] = pen_color[2];
            }
        }
    }
}

void draw_line_c4(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

    const float t0 = thickness / 2.f;
    const float t1 = thickness - t0;

    int x_min = std::min(x0, x1);
    int x_max = std::max(x0, x1);
    int y_min = std::min(y0, y1);
    int y_max = std::max(y0, y1);

    for (int y = y_min - t0; y < y_max + t1; y++)
    {
        if (y < 0)
            continue;

        if (y >= h)
            break;

        unsigned char* p = pixels + stride * y;

        for (int x = x_min - t0; x < x_max + t1; x++)
        {
            if (x < 0)
                continue;

            if (x >= w)
                break;

            // distance from line
            if (distance_lessthan(x, y, x0, y0, x1, y1, t1))
            {
                p[x * 4 + 0] = pen_color[0];
                p[x * 4 + 1] = pen_color[1];
                p[x * 4 + 2] = pen_color[2];
                p[x * 4 + 3] = pen_color[3];
            }
        }
    }
}

void draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    // assert w % 2 == 0
    // assert h % 2 == 0
    // assert x0 % 2 == 0
    // assert y0 % 2 == 0
    // assert x1 % 2 == 0
    // assert y1 % 2 == 0
    // assert thickness % 2 == 0

    const unsigned char* pen_color = (const unsigned char*)&color;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* pen_color_y = (unsigned char*)&v_y;
    unsigned char* pen_color_uv = (unsigned char*)&v_uv;
    pen_color_y[0] = pen_color[0];
    pen_color_uv[0] = pen_color[1];
    pen_color_uv[1] = pen_color[2];

    unsigned char* Y = yuv420sp;
    draw_line_c1(Y, w, h, x0, y0, x1, y1, v_y, thickness);

    unsigned char* UV = yuv420sp + w * h;
    int thickness_uv = thickness == -1 ? thickness : std::max(thickness / 2, 1);
    draw_line_c2(UV, w / 2, h / 2, x0 / 2, y0 / 2, x1 / 2, y1 / 2, v_uv, thickness_uv);
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

void draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c1(pixels, w, h, w, text, x, y, fontpixelsize, color);
}

void draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c2(pixels, w, h, w * 2, text, x, y, fontpixelsize, color);
}

void draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c3(pixels, w, h, w * 3, text, x, y, fontpixelsize, color);
}

void draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    return draw_text_c4(pixels, w, h, w * 4, text, x, y, fontpixelsize, color);
}

void draw_text_c1(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

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

                if (j >= h)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = pixels + stride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= w)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k] = (p[k] * (255 - alpha) + pen_color[0] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_c2(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

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
            int font_bitmap_index = ch - ' ';
            const unsigned char* font_bitmap = mono_font_data[font_bitmap_index];

            // draw resized character
            resize_bilinear_c1(font_bitmap, 20, 40, resized_font_bitmap, fontpixelsize, fontpixelsize * 2);

            for (int j = cursor_y; j < cursor_y + fontpixelsize * 2; j++)
            {
                if (j < 0)
                    continue;

                if (j >= h)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = pixels + stride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= w)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k * 2 + 0] = (p[k * 2 + 0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[k * 2 + 1] = (p[k * 2 + 1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_c3(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

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
            int font_bitmap_index = ch - ' ';
            const unsigned char* font_bitmap = mono_font_data[font_bitmap_index];

            // draw resized character
            resize_bilinear_c1(font_bitmap, 20, 40, resized_font_bitmap, fontpixelsize, fontpixelsize * 2);

            for (int j = cursor_y; j < cursor_y + fontpixelsize * 2; j++)
            {
                if (j < 0)
                    continue;

                if (j >= h)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = pixels + stride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= w)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k * 3 + 0] = (p[k * 3 + 0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[k * 3 + 1] = (p[k * 3 + 1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                    p[k * 3 + 2] = (p[k * 3 + 2] * (255 - alpha) + pen_color[2] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_c4(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    const unsigned char* pen_color = (const unsigned char*)&color;

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

                if (j >= h)
                    break;

                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize;
                unsigned char* p = pixels + stride * j;

                for (int k = cursor_x; k < cursor_x + fontpixelsize; k++)
                {
                    if (k < 0)
                        continue;

                    if (k >= w)
                        break;

                    unsigned char alpha = palpha[k - cursor_x];

                    p[k * 4 + 0] = (p[k * 4 + 0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[k * 4 + 1] = (p[k * 4 + 1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                    p[k * 4 + 2] = (p[k * 4 + 2] * (255 - alpha) + pen_color[2] * alpha) / 255;
                    p[k * 4 + 3] = (p[k * 4 + 3] * (255 - alpha) + pen_color[3] * alpha) / 255;
                }
            }

            cursor_x += fontpixelsize;
        }
    }

    delete[] resized_font_bitmap;
}

void draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    // assert w % 2 == 0
    // assert h % 2 == 0
    // assert x % 2 == 0
    // assert y % 2 == 0
    // assert fontpixelsize % 2 == 0

    const unsigned char* pen_color = (const unsigned char*)&color;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* pen_color_y = (unsigned char*)&v_y;
    unsigned char* pen_color_uv = (unsigned char*)&v_uv;
    pen_color_y[0] = pen_color[0];
    pen_color_uv[0] = pen_color[1];
    pen_color_uv[1] = pen_color[2];

    unsigned char* Y = yuv420sp;
    draw_text_c1(Y, w, h, text, x, y, fontpixelsize, v_y);

    unsigned char* UV = yuv420sp + w * h;
    draw_text_c2(UV, w / 2, h / 2, text, x / 2, y / 2, std::max(fontpixelsize / 2, 1), v_uv);
}

#endif // NCNN_PIXEL_DRAWING

} // namespace ncnn
