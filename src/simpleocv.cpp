// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "simpleocv.h"

#if NCNN_SIMPLEOCV

#include <stdio.h>
#if NCNN_SIMPLEOCV_STB
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif
namespace cv {

Mat imread(const std::string& path, int flags)
{
    (void)flags;
    Mat m = Mat();
    // read pgm/ppm
    if (path.find(".pgm") != std::string::npos || path.find(".ppm") != std::string::npos)
    {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp)
            return m;

        char magic[3];
        int w, h;
        int nscan = fscanf(fp, "%2s\n%d %d\n255\n", magic, &w, &h);
        if (nscan == 3 && magic[0] == 'P' && (magic[1] == '5' || magic[1] == '6'))
        {
            if (magic[1] == '5')
            {
                m.create(h, w, CV_8UC1);
            }
            else if (magic[1] == '6')
            {
                m.create(h, w, CV_8UC3);
            }
            if (m.empty())
            {
                fclose(fp);
                return Mat();
            }

            fread(m.data, 1, m.total(), fp);
        }

        fclose(fp);
    }
    else
    {
#if NCNN_SIMPLEOCV_STB
        int w, h, c;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);
        ncnn::Mat in = ncnn::Mat::from_pixels(data, ncnn::Mat::PIXEL_RGB, w, h);
        m.create(h, w, CV_8UC3);
        in.to_pixels(m.data, ncnn::Mat::PIXEL_RGB2BGR);
        stbi_image_free(data);
#endif
    }

    return m;
}

void imshow(const std::string img, const Mat& m)
{
#if NCNN_SIMPLEOCV_STB
    std::string jpg = img + ".jpg";
    imwrite(jpg, m);
#ifdef linux
    system(("xdg-open ./" + jpg).c_str());
#endif
#ifdef _WIN32
    system("start ./" + jpg).c_str());
#endif
#endif
}
void imwrite(const std::string& path, const Mat& m)
{
    // write pgm/ppm
    if (path.find(".pgm") != std::string::npos || path.find(".ppm") != std::string::npos)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        if (!fp)
            return;

        if (m.channels() == 1)
        {
            fprintf(fp, "P5\n%d %d\n255\n", m.cols, m.rows);
        }
        else if (m.channels() == 3)
        {
            fprintf(fp, "P6\n%d %d\n255\n", m.cols, m.rows);
        }

        fwrite(m.data, 1, m.total(), fp);

        fclose(fp);
    }
    else
    {
#if NCNN_SIMPLEOCV_STB
        if (path.find(".jpg") != std::string::npos)
        {
            ncnn::Mat out = ncnn::Mat::from_pixels(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows);
            out.to_pixels(m.data, ncnn::Mat::PIXEL_RGB);
            stbi_write_jpg(path.c_str(), m.cols, m.rows, m.channels(), m.data, 100);
        }
#endif
    }
}

void waitKey(int delay)
{
    (void)delay;
    getchar();
}

void circle(const Mat& m, Point2f p, int redius, Scalar scalar, int thickness)
{
    (void)thickness;
    ncnn::Mat out = ncnn::Mat::from_pixels(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows);
    float* r = out.channel(0);
    float* g = out.channel(1);
    float* b = out.channel(2);
    for (int i = p.x; i < p.x + redius; i++)
    {
        for (int j = p.y; j < p.y + redius; j++)
        {
            *(r + i + j * m.cols) = scalar.r;
            *(g + i + j * m.cols) = scalar.g;
            *(b + i + j * m.cols) = scalar.b;
        }
    }
    out.to_pixels(m.data, ncnn::Mat::PIXEL_RGB2BGR);
}

void line(const Mat& m, Point2f p1, Point2f p2, Scalar scalar, int thickness)
{
    ncnn::Mat out = ncnn::Mat::from_pixels(m.data, ncnn::Mat::PIXEL_BGR2RGB, m.cols, m.rows);
    float* r = out.channel(0);
    float* g = out.channel(1);
    float* b = out.channel(2);

    int s1, s2, interchange;
    int X = p1.x;
    int Y = p1.y;
    int deltax, deltay, f, Temp;
    deltax = abs(p2.x - p1.x);
    deltay = abs(p2.y - p1.y);
    if (p2.x - p1.x >= 0)
        s1 = 1;
    else
        s1 = -1; //设置步进值

    if (p2.y - p1.y >= 0)
        s2 = 1;
    else
        s2 = -1;

    f = 2 * deltay - deltax; //2dy-dx

    if (deltay > deltax) //斜率大于一，进行坐标转换
    {
        Temp = deltax;
        deltax = deltay;
        deltay = Temp;
        interchange = 1;
    }
    else
        interchange = 0;

    for (int i = 1; i <= deltax + deltay; i++)
    {
        if (f >= 0)
        {
            if (interchange == 1)
                X += s1;

            else
                Y += s2;

            for (int ll = -thickness; ll < thickness; ll++)
            {
                *(r + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.r;
                *(g + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.g;
                *(b + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.b;
            }

            f = f - 2 * deltax;
        }
        else
        {
            if (interchange == 1)
                Y += s2;

            else
                X += s1;

            for (int ll = -thickness; ll < thickness; ll++)
            {
                *(r + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.r;
                *(g + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.g;
                *(b + (int)(X) + (int)(int(Y) * m.cols) + ll) = scalar.b;
            }

            f = f + 2 * deltay;
        }
    }

    out.to_pixels(m.data, ncnn::Mat::PIXEL_RGB2BGR);
}

#if NCNN_PIXEL
void resize(const Mat& src, Mat& dst, const Size& size, float sw, float sh, int flags)
{
    (void)flags;

    int srcw = src.cols;
    int srch = src.rows;

    int w = size.width;
    int h = size.height;

    if (w == 0 || h == 0)
    {
        w = srcw * sw;
        h = srch * sh;
    }

    if (w == 0 || h == 0)
        return;

    if (w == srcw && h == srch)
    {
        dst = src.clone();
        return;
    }

    cv::Mat tmp(h, w, src.c);
    if (tmp.empty())
        return;

    if (src.c == 1)
        ncnn::resize_bilinear_c1(src.data, srcw, srch, tmp.data, w, h);
    else if (src.c == 3)
        ncnn::resize_bilinear_c3(src.data, srcw, srch, tmp.data, w, h);
    else if (src.c == 4)
        ncnn::resize_bilinear_c4(src.data, srcw, srch, tmp.data, w, h);

    dst = tmp;
}
#endif // NCNN_PIXEL

} // namespace cv

#endif // NCNN_SIMPLEOCV
