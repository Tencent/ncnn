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
#include <limits.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#if __SSE2__
#include <emmintrin.h>
#endif

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

void resize_bilinear_font(const unsigned char* font_bitmap, unsigned char* resized_font_bitmap, int fontpixelsize)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    const int srcw = 20;
    const int srch = 40;
    const int w = fontpixelsize;
    const int h = fontpixelsize * 2;

    double scale = (double)srcw / w;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0;
    short* rows1 = (short*)rowsbuf1;

    {
        short* rows1p = rows1;
        for (int dx = 0; dx < w; dx++)
        {
            rows1p[dx] = 0;
        }
    }

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = font_bitmap + 10 * (sy + 1);

            if (sy >= srch - 1)
            {
                short* rows1p = rows1;
                for (int dx = 0; dx < w; dx++)
                {
                    rows1p[dx] = 0;
                }
            }
            else
            {
                const short* ialphap = ialpha;
                short* rows1p = rows1;
                for (int dx = 0; dx < w; dx++)
                {
                    sx = xofs[dx];
                    short a0 = ialphap[0];
                    short a1 = ialphap[1];

                    unsigned char S1p0;
                    unsigned char S1p1;

                    if (sx < 0)
                    {
                        S1p0 = 0;
                        S1p1 = S1[0] & 0x0f;
                    }
                    else if (sx >= srcw - 1)
                    {
                        S1p0 = (S1[9] & 0xf0) >> 4;
                        S1p1 = 0;
                    }
                    else
                    {
                        S1p0 = sx % 2 == 0 ? S1[sx / 2] & 0x0f : (S1[sx / 2] & 0xf0) >> 4;
                        S1p1 = sx % 2 == 0 ? (S1[sx / 2] & 0xf0) >> 4 : S1[sx / 2 + 1] & 0x0f;
                    }
                    rows1p[dx] = (S1p0 * a0 + S1p1 * a1) * 17 >> 4;

                    ialphap += 2;
                }
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = font_bitmap + 10 * (sy);
            const unsigned char* S1 = font_bitmap + 10 * (sy + 1);

            if (sy >= srch - 1)
            {
                const short* ialphap = ialpha;
                short* rows0p = rows0;
                short* rows1p = rows1;
                for (int dx = 0; dx < w; dx++)
                {
                    sx = xofs[dx];
                    short a0 = ialphap[0];
                    short a1 = ialphap[1];

                    unsigned char S0p0;
                    unsigned char S0p1;

                    if (sx < 0)
                    {
                        S0p0 = 0;
                        S0p1 = S0[0] & 0x0f;
                    }
                    else if (sx >= srcw - 1)
                    {
                        S0p0 = (S0[9] & 0xf0) >> 4;
                        S0p1 = 0;
                    }
                    else
                    {
                        S0p0 = sx % 2 == 0 ? S0[sx / 2] & 0x0f : (S0[sx / 2] & 0xf0) >> 4;
                        S0p1 = sx % 2 == 0 ? (S0[sx / 2] & 0xf0) >> 4 : S0[sx / 2 + 1] & 0x0f;
                    }
                    rows0p[dx] = (S0p0 * a0 + S0p1 * a1) * 17 >> 4;
                    rows1p[dx] = 0;

                    ialphap += 2;
                }
            }
            else
            {
                const short* ialphap = ialpha;
                short* rows0p = rows0;
                short* rows1p = rows1;
                for (int dx = 0; dx < w; dx++)
                {
                    sx = xofs[dx];
                    short a0 = ialphap[0];
                    short a1 = ialphap[1];

                    unsigned char S0p0;
                    unsigned char S0p1;
                    unsigned char S1p0;
                    unsigned char S1p1;

                    if (sx < 0)
                    {
                        S0p0 = 0;
                        S0p1 = S0[0] & 0x0f;
                        S1p0 = 0;
                        S1p1 = S1[0] & 0x0f;
                    }
                    else if (sx >= srcw - 1)
                    {
                        S0p0 = (S0[9] & 0xf0) >> 4;
                        S0p1 = 0;
                        S1p0 = (S1[9] & 0xf0) >> 4;
                        S1p1 = 0;
                    }
                    else
                    {
                        S0p0 = sx % 2 == 0 ? S0[sx / 2] & 0x0f : (S0[sx / 2] & 0xf0) >> 4;
                        S0p1 = sx % 2 == 0 ? (S0[sx / 2] & 0xf0) >> 4 : S0[sx / 2 + 1] & 0x0f;
                        S1p0 = sx % 2 == 0 ? S1[sx / 2] & 0x0f : (S1[sx / 2] & 0xf0) >> 4;
                        S1p1 = sx % 2 == 0 ? (S1[sx / 2] & 0xf0) >> 4 : S1[sx / 2 + 1] & 0x0f;
                    }
                    rows0p[dx] = (S0p0 * a0 + S0p1 * a1) * 17 >> 4;
                    rows1p[dx] = (S1p0 * a0 + S1p1 * a1) * 17 >> 4;

                    ialphap += 2;
                }
            }
        }

        prev_sy1 = sy;

        if (dy + 1 < h && yofs[dy + 1] == sy)
        {
            // vresize for two rows
            short b0 = ibeta[0];
            short b1 = ibeta[1];
            short b2 = ibeta[2];
            short b3 = ibeta[3];

            short* rows0p = rows0;
            short* rows1p = rows1;
            unsigned char* Dp0 = resized_font_bitmap + w * (dy);
            unsigned char* Dp1 = resized_font_bitmap + w * (dy + 1);

            int dx = 0;
#if __ARM_NEON
            int16x8_t _b0 = vdupq_n_s16(b0);
            int16x8_t _b1 = vdupq_n_s16(b1);
            int16x8_t _b2 = vdupq_n_s16(b2);
            int16x8_t _b3 = vdupq_n_s16(b3);
            for (; dx + 15 < w; dx += 16)
            {
                int16x8_t _r00 = vld1q_s16(rows0p);
                int16x8_t _r01 = vld1q_s16(rows0p + 8);
                int16x8_t _r10 = vld1q_s16(rows1p);
                int16x8_t _r11 = vld1q_s16(rows1p + 8);
                int16x8_t _acc00 = vaddq_s16(vqdmulhq_s16(_r00, _b0), vqdmulhq_s16(_r10, _b1));
                int16x8_t _acc01 = vaddq_s16(vqdmulhq_s16(_r01, _b0), vqdmulhq_s16(_r11, _b1));
                int16x8_t _acc10 = vaddq_s16(vqdmulhq_s16(_r00, _b2), vqdmulhq_s16(_r10, _b3));
                int16x8_t _acc11 = vaddq_s16(vqdmulhq_s16(_r01, _b2), vqdmulhq_s16(_r11, _b3));
                uint8x16_t _Dp0 = vcombine_u8(vqrshrun_n_s16(_acc00, 3), vqrshrun_n_s16(_acc01, 3));
                uint8x16_t _Dp1 = vcombine_u8(vqrshrun_n_s16(_acc10, 3), vqrshrun_n_s16(_acc11, 3));
                vst1q_u8(Dp0, _Dp0);
                vst1q_u8(Dp1, _Dp1);
                Dp0 += 16;
                Dp1 += 16;
                rows0p += 16;
                rows1p += 16;
            }
            for (; dx + 7 < w; dx += 8)
            {
                int16x8_t _r0 = vld1q_s16(rows0p);
                int16x8_t _r1 = vld1q_s16(rows1p);
                int16x8_t _acc0 = vaddq_s16(vqdmulhq_s16(_r0, _b0), vqdmulhq_s16(_r1, _b1));
                int16x8_t _acc1 = vaddq_s16(vqdmulhq_s16(_r0, _b2), vqdmulhq_s16(_r1, _b3));
                uint8x8_t _Dp0 = vqrshrun_n_s16(_acc0, 3);
                uint8x8_t _Dp1 = vqrshrun_n_s16(_acc1, 3);
                vst1_u8(Dp0, _Dp0);
                vst1_u8(Dp1, _Dp1);
                Dp0 += 8;
                Dp1 += 8;
                rows0p += 8;
                rows1p += 8;
            }
#endif // __ARM_NEON
#if __SSE2__
            __m128i _b0 = _mm_set1_epi16(b0);
            __m128i _b1 = _mm_set1_epi16(b1);
            __m128i _b2 = _mm_set1_epi16(b2);
            __m128i _b3 = _mm_set1_epi16(b3);
            __m128i _v2 = _mm_set1_epi16(2);
            for (; dx + 15 < w; dx += 16)
            {
                __m128i _r00 = _mm_loadu_si128((const __m128i*)rows0p);
                __m128i _r01 = _mm_loadu_si128((const __m128i*)(rows0p + 8));
                __m128i _r10 = _mm_loadu_si128((const __m128i*)rows1p);
                __m128i _r11 = _mm_loadu_si128((const __m128i*)(rows1p + 8));
                __m128i _acc00 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b0), _mm_mulhi_epi16(_r10, _b1));
                __m128i _acc01 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b0), _mm_mulhi_epi16(_r11, _b1));
                __m128i _acc10 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b2), _mm_mulhi_epi16(_r10, _b3));
                __m128i _acc11 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b2), _mm_mulhi_epi16(_r11, _b3));
                _acc00 = _mm_srai_epi16(_mm_add_epi16(_acc00, _v2), 2);
                _acc01 = _mm_srai_epi16(_mm_add_epi16(_acc01, _v2), 2);
                _acc10 = _mm_srai_epi16(_mm_add_epi16(_acc10, _v2), 2);
                _acc11 = _mm_srai_epi16(_mm_add_epi16(_acc11, _v2), 2);
                __m128i _Dp0 = _mm_packus_epi16(_acc00, _acc01);
                __m128i _Dp1 = _mm_packus_epi16(_acc10, _acc11);
                _mm_storeu_si128((__m128i*)Dp0, _Dp0);
                _mm_storeu_si128((__m128i*)Dp1, _Dp1);
                Dp0 += 16;
                Dp1 += 16;
                rows0p += 16;
                rows1p += 16;
            }
            for (; dx + 7 < w; dx += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)rows0p);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)rows1p);
                __m128i _acc0 = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b0), _mm_mulhi_epi16(_r1, _b1));
                __m128i _acc1 = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b2), _mm_mulhi_epi16(_r1, _b3));
                _acc0 = _mm_srai_epi16(_mm_add_epi16(_acc0, _v2), 2);
                _acc1 = _mm_srai_epi16(_mm_add_epi16(_acc1, _v2), 2);
                __m128i _Dp0 = _mm_packus_epi16(_acc0, _acc0);
                __m128i _Dp1 = _mm_packus_epi16(_acc1, _acc1);
                _mm_storel_epi64((__m128i*)Dp0, _Dp0);
                _mm_storel_epi64((__m128i*)Dp1, _Dp1);
                Dp0 += 8;
                Dp1 += 8;
                rows0p += 8;
                rows1p += 8;
            }
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                short s0 = *rows0p++;
                short s1 = *rows1p++;

                *Dp0++ = (unsigned char)(((short)((b0 * s0) >> 16) + (short)((b1 * s1) >> 16) + 2) >> 2);
                *Dp1++ = (unsigned char)(((short)((b2 * s0) >> 16) + (short)((b3 * s1) >> 16) + 2) >> 2);
            }

            ibeta += 4;
            dy += 1;
        }
        else
        {
            // vresize
            short b0 = ibeta[0];
            short b1 = ibeta[1];

            short* rows0p = rows0;
            short* rows1p = rows1;
            unsigned char* Dp = resized_font_bitmap + w * (dy);

            int dx = 0;
#if __ARM_NEON
            int16x8_t _b0 = vdupq_n_s16(b0);
            int16x8_t _b1 = vdupq_n_s16(b1);
            for (; dx + 15 < w; dx += 16)
            {
                int16x8_t _r00 = vld1q_s16(rows0p);
                int16x8_t _r01 = vld1q_s16(rows0p + 8);
                int16x8_t _r10 = vld1q_s16(rows1p);
                int16x8_t _r11 = vld1q_s16(rows1p + 8);
                int16x8_t _acc0 = vaddq_s16(vqdmulhq_s16(_r00, _b0), vqdmulhq_s16(_r10, _b1));
                int16x8_t _acc1 = vaddq_s16(vqdmulhq_s16(_r01, _b0), vqdmulhq_s16(_r11, _b1));
                uint8x16_t _Dp = vcombine_u8(vqrshrun_n_s16(_acc0, 3), vqrshrun_n_s16(_acc1, 3));
                vst1q_u8(Dp, _Dp);
                Dp += 16;
                rows0p += 16;
                rows1p += 16;
            }
            for (; dx + 7 < w; dx += 8)
            {
                int16x8_t _r0 = vld1q_s16(rows0p);
                int16x8_t _r1 = vld1q_s16(rows1p);
                int16x8_t _acc = vaddq_s16(vqdmulhq_s16(_r0, _b0), vqdmulhq_s16(_r1, _b1));
                uint8x8_t _Dp = vqrshrun_n_s16(_acc, 3);
                vst1_u8(Dp, _Dp);
                Dp += 8;
                rows0p += 8;
                rows1p += 8;
            }
#endif // __ARM_NEON
#if __SSE2__
            __m128i _b0 = _mm_set1_epi16(b0);
            __m128i _b1 = _mm_set1_epi16(b1);
            __m128i _v2 = _mm_set1_epi16(2);
            for (; dx + 15 < w; dx += 16)
            {
                __m128i _r00 = _mm_loadu_si128((const __m128i*)rows0p);
                __m128i _r01 = _mm_loadu_si128((const __m128i*)(rows0p + 8));
                __m128i _r10 = _mm_loadu_si128((const __m128i*)rows1p);
                __m128i _r11 = _mm_loadu_si128((const __m128i*)(rows1p + 8));
                __m128i _acc0 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b0), _mm_mulhi_epi16(_r10, _b1));
                __m128i _acc1 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b0), _mm_mulhi_epi16(_r11, _b1));
                _acc0 = _mm_srai_epi16(_mm_add_epi16(_acc0, _v2), 2);
                _acc1 = _mm_srai_epi16(_mm_add_epi16(_acc1, _v2), 2);
                __m128i _Dp = _mm_packus_epi16(_acc0, _acc1);
                _mm_storeu_si128((__m128i*)Dp, _Dp);
                Dp += 16;
                rows0p += 16;
                rows1p += 16;
            }
            for (; dx + 7 < w; dx += 8)
            {
                __m128i _r0 = _mm_loadu_si128((const __m128i*)rows0p);
                __m128i _r1 = _mm_loadu_si128((const __m128i*)rows1p);
                __m128i _acc = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b0), _mm_mulhi_epi16(_r1, _b1));
                _acc = _mm_srai_epi16(_mm_add_epi16(_acc, _v2), 2);
                __m128i _Dp = _mm_packus_epi16(_acc, _acc);
                _mm_storel_epi64((__m128i*)Dp, _Dp);
                Dp += 8;
                rows0p += 8;
                rows1p += 8;
            }
#endif // __SSE2__
            for (; dx < w; dx++)
            {
                short s0 = *rows0p++;
                short s1 = *rows1p++;

                *Dp++ = (unsigned char)(((short)((b0 * s0) >> 16) + (short)((b1 * s1) >> 16) + 2) >> 2);
            }

            ibeta += 2;
        }
    }

    delete[] buf;
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
            continue;
        }

        if (ch == ' ')
        {
            cursor_x += fontpixelsize;
            continue;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - '!'];

            // draw resized character
            resize_bilinear_font(font_bitmap, resized_font_bitmap, fontpixelsize);

            const int ystart = std::max(cursor_y, 0);
            const int yend = std::min(cursor_y + fontpixelsize * 2, h);
            const int xstart = std::max(cursor_x, 0);
            const int xend = std::min(cursor_x + fontpixelsize, w);

            for (int j = ystart; j < yend; j++)
            {
                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize + xstart - cursor_x;
                unsigned char* p = pixels + stride * j + xstart;

                for (int k = xstart; k < xend; k++)
                {
                    unsigned char alpha = *palpha++;

                    p[0] = (p[0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p += 1;
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
            continue;
        }

        if (ch == ' ')
        {
            cursor_x += fontpixelsize;
            continue;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - '!'];

            // draw resized character
            resize_bilinear_font(font_bitmap, resized_font_bitmap, fontpixelsize);

            const int ystart = std::max(cursor_y, 0);
            const int yend = std::min(cursor_y + fontpixelsize * 2, h);
            const int xstart = std::max(cursor_x, 0);
            const int xend = std::min(cursor_x + fontpixelsize, w);

            for (int j = ystart; j < yend; j++)
            {
                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize + xstart - cursor_x;
                unsigned char* p = pixels + stride * j + xstart * 2;

                for (int k = xstart; k < xend; k++)
                {
                    unsigned char alpha = *palpha++;

                    p[0] = (p[0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[1] = (p[1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                    p += 2;
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
            continue;
        }

        if (ch == ' ')
        {
            cursor_x += fontpixelsize;
            continue;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - '!'];

            // draw resized character
            resize_bilinear_font(font_bitmap, resized_font_bitmap, fontpixelsize);

            const int ystart = std::max(cursor_y, 0);
            const int yend = std::min(cursor_y + fontpixelsize * 2, h);
            const int xstart = std::max(cursor_x, 0);
            const int xend = std::min(cursor_x + fontpixelsize, w);

            for (int j = ystart; j < yend; j++)
            {
                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize + xstart - cursor_x;
                unsigned char* p = pixels + stride * j + xstart * 3;

                for (int k = xstart; k < xend; k++)
                {
                    unsigned char alpha = *palpha++;

                    p[0] = (p[0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[1] = (p[1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                    p[2] = (p[2] * (255 - alpha) + pen_color[2] * alpha) / 255;
                    p += 3;
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
            continue;
        }

        if (ch == ' ')
        {
            cursor_x += fontpixelsize;
            continue;
        }

        if (isprint(ch) != 0)
        {
            const unsigned char* font_bitmap = mono_font_data[ch - '!'];

            // draw resized character
            resize_bilinear_font(font_bitmap, resized_font_bitmap, fontpixelsize);

            const int ystart = std::max(cursor_y, 0);
            const int yend = std::min(cursor_y + fontpixelsize * 2, h);
            const int xstart = std::max(cursor_x, 0);
            const int xend = std::min(cursor_x + fontpixelsize, w);

            for (int j = ystart; j < yend; j++)
            {
                const unsigned char* palpha = resized_font_bitmap + (j - cursor_y) * fontpixelsize + xstart - cursor_x;
                unsigned char* p = pixels + stride * j + xstart * 4;

                for (int k = xstart; k < xend; k++)
                {
                    unsigned char alpha = *palpha++;

                    p[0] = (p[0] * (255 - alpha) + pen_color[0] * alpha) / 255;
                    p[1] = (p[1] * (255 - alpha) + pen_color[1] * alpha) / 255;
                    p[2] = (p[2] * (255 - alpha) + pen_color[2] * alpha) / 255;
                    p[3] = (p[3] * (255 - alpha) + pen_color[3] * alpha) / 255;
                    p += 4;
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
