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
#include "simpleocv.dat"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
namespace cv {

template<typename... Args>
std::string string_sprintf(const char* format, Args... args)
{
    int length = std::snprintf(nullptr, 0, format, args...);
    if (length <= 0)
    {
        return "";
    }

    char* buf = new char[length + 1];
    std::snprintf(buf, length + 1, format, args...);

    std::string str(buf);
    delete[] buf;
    return std::move(str);
}

std::string html = "";

void circle(const Mat& src, Point2f p, int redius, Scalar scalar, int thickness)
{
    (void)src;
    (void)redius;
    (void)scalar;
    (void)thickness;
    html += string_sprintf(drawPoint.c_str(), (int)p.x, (int)p.y);
}
Mat imread(const std::string& path, int flags)
{
    (void)flags;
    Mat m;
    html += string_sprintf(head.c_str(), path.c_str()) + bodyA;
    if (path.find(".pgm") != std::string::npos || path.find(".ppm") != std::string::npos)
    {
        // read pgm/ppm
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp)
            return Mat();

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
        int w, h, c;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);
        m.create(h, w, CV_8UC3);
        memcpy(m.data, data, m.total());
        html += string_sprintf(drawscreen.c_str(), path.c_str(), w, h) + imgs;
        stbi_image_free(data);
    }

    return m;
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
        return;
    }
    html += end;
    FILE* fp = fopen("./tmp.html", "wb");
    if (!fp)
        return;
    fprintf(fp, "%s\n", html.c_str());
    fclose(fp);
}

void imshow(const std::string img, const Mat& m)
{
    (void)img;
    (void)m;

#ifdef linux
    system("xdg-open ./tmp.html");
#endif
#ifdef _WIN32
    system("start ./tmp.html");
#endif
}

void waitKey(int i)
{
    (void)i;
    getchar();
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
