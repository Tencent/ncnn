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

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace cv {

Mat imread(const std::string& path, int flags)
{
    int desired_channels = 0;
    if (flags == IMREAD_UNCHANGED)
    {
        desired_channels = 0;
    }
    else if (flags == IMREAD_GRAYSCALE)
    {
        desired_channels = 1;
    }
    else if (flags == IMREAD_COLOR)
    {
        desired_channels = 3;
    }
    else
    {
        // unknown flags
        return Mat();
    }

    int w;
    int h;
    int c;
    unsigned char* pixeldata = stbi_load(path.c_str(), &w, &h, &c, desired_channels);
    if (!pixeldata)
    {
        // load failed
        return Mat();
    }

    if (desired_channels)
    {
        c = desired_channels;
    }

    // copy pixeldata to Mat
    Mat img;
    if (c == 1)
    {
        img.create(h, w, CV_8UC1);
    }
    else if (c == 3)
    {
        img.create(h, w, CV_8UC3);
    }
    else if (c == 4)
    {
        img.create(h, w, CV_8UC4);
    }
    else
    {
        // unexpected channels
        stbi_image_free(pixeldata);
        return Mat();
    }

    memcpy(img.data, pixeldata, w * h * c);

    stbi_image_free(pixeldata);

    //     // resolve exif orientation
    //     {
    //         std::ifstream ifs;
    //         ifs.open(filename.c_str(), std::ifstream::in);
    //
    //         if (ifs.good())
    //         {
    //             ExifReader exif_reader(ifs);
    //             if (exif_reader.parse())
    //             {
    //                 ExifEntry_t e = exif_reader.getTag(ORIENTATION);
    //                 int orientation = e.field_u16;
    //                 if (orientation >= 1 && orientation <= 8)
    //                     rotate_by_orientation(img, img, orientation);
    //             }
    //         }
    //
    //         ifs.close();
    //     }

    // rgb to bgr
    if (c == 3)
    {
        uchar* p = img.data;
        for (int i = 0; i < w * h; i++)
        {
            std::swap(p[0], p[2]);
            p += 3;
        }
    }
    if (c == 4)
    {
        uchar* p = img.data;
        for (int i = 0; i < w * h; i++)
        {
            std::swap(p[0], p[2]);
            p += 4;
        }
    }

    return img;
}

bool imwrite(const std::string& path, const Mat& m, const std::vector<int>& params)
{
    const char* _ext = strrchr(path.c_str(), '.');
    if (!_ext)
    {
        // missing extension
        return false;
    }

    std::string ext = _ext;
    Mat img = m.clone();

    // bgr to rgb
    int c = 0;
    if (img.type() == CV_8UC1)
    {
        c = 1;
    }
    else if (img.type() == CV_8UC3)
    {
        c = 3;
        uchar* p = img.data;
        for (int i = 0; i < img.cols * img.rows; i++)
        {
            std::swap(p[0], p[2]);
            p += 3;
        }
    }
    else if (img.type() == CV_8UC4)
    {
        c = 4;
        uchar* p = img.data;
        for (int i = 0; i < img.cols * img.rows; i++)
        {
            std::swap(p[0], p[2]);
            p += 4;
        }
    }
    else
    {
        // unexpected image channels
        return false;
    }

    bool success = false;

    if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG")
    {
        int quality = 95;
        for (size_t i = 0; i < params.size(); i += 2)
        {
            if (params[i] == IMWRITE_JPEG_QUALITY)
            {
                quality = params[i + 1];
                break;
            }
        }
        success = stbi_write_jpg(path.c_str(), img.cols, img.rows, c, img.data, quality);
    }
    else if (ext == ".png" || ext == ".PNG")
    {
        success = stbi_write_png(path.c_str(), img.cols, img.rows, c, img.data, 0);
    }
    else if (ext == ".bmp" || ext == ".BMP")
    {
        success = stbi_write_bmp(path.c_str(), img.cols, img.rows, c, img.data);
    }
    else
    {
        // unknown extension type
        return false;
    }

    return success;
}

void imshow(const std::string& name, const Mat& m)
{
    NCNN_LOGE("imshow save image to %s.png", name.c_str());

    imwrite(name + ".png", m);
}

int waitKey(int delay)
{
    NCNN_LOGE("waitKey stub");
    return -1;
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

#if NCNN_PIXEL_DRAWING

void rectangle(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness)
{
    Rect rec;
    rec.x = std::min(pt1.x, pt2.x);
    rec.y = std::min(pt1.y, pt2.y);
    rec.width = std::max(pt1.x, pt2.x) - rec.x;
    rec.height = std::max(pt1.y, pt2.y) - rec.y;
    rectangle(img, rec, color, thickness);
}

void rectangle(Mat& img, Rect rec, const Scalar& _color, int thickness)
{
    unsigned int color = 0;
    unsigned char* border_color = (unsigned char*)&color;

    if (img.c == 1)
    {
        border_color[0] = _color[0];
        ncnn::draw_rectangle_c1(img.data, img.cols, img.rows, rec.x, rec.y, rec.width, rec.height, color, thickness);
    }
    else if (img.c == 3)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        ncnn::draw_rectangle_c3(img.data, img.cols, img.rows, rec.x, rec.y, rec.width, rec.height, color, thickness);
    }
    else if (img.c == 4)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        border_color[3] = _color[3];
        ncnn::draw_rectangle_c4(img.data, img.cols, img.rows, rec.x, rec.y, rec.width, rec.height, color, thickness);
    }
}

void circle(Mat& img, Point center, int radius, const Scalar& _color, int thickness)
{
    unsigned int color = 0;
    unsigned char* border_color = (unsigned char*)&color;

    if (img.c == 1)
    {
        border_color[0] = _color[0];
        ncnn::draw_circle_c1(img.data, img.cols, img.rows, center.x, center.y, radius, color, thickness);
    }
    else if (img.c == 3)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        ncnn::draw_circle_c3(img.data, img.cols, img.rows, center.x, center.y, radius, color, thickness);
    }
    else if (img.c == 4)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        border_color[3] = _color[3];
        ncnn::draw_circle_c4(img.data, img.cols, img.rows, center.x, center.y, radius, color, thickness);
    }
}

void line(Mat& img, Point p0, Point p1, const Scalar& _color, int thickness)
{
    unsigned int color = 0;
    unsigned char* border_color = (unsigned char*)&color;

    if (img.c == 1)
    {
        border_color[0] = _color[0];
        ncnn::draw_line_c1(img.data, img.cols, img.rows, p0.x, p0.y, p1.x, p1.y, color, thickness);
    }
    else if (img.c == 3)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        ncnn::draw_line_c3(img.data, img.cols, img.rows, p0.x, p0.y, p1.x, p1.y, color, thickness);
    }
    else if (img.c == 4)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        border_color[3] = _color[3];
        ncnn::draw_line_c4(img.data, img.cols, img.rows, p0.x, p0.y, p1.x, p1.y, color, thickness);
    }
}

void putText(Mat& img, const std::string& text, Point org, int fontFace, double fontScale, Scalar _color, int thickness)
{
    const int fontpixelsize = 20 * fontScale;

    unsigned int color = 0;
    unsigned char* border_color = (unsigned char*)&color;

    if (img.c == 1)
    {
        border_color[0] = _color[0];
        ncnn::draw_text_c1(img.data, img.cols, img.rows, text.c_str(), org.x, org.y - fontpixelsize * 2, fontpixelsize, color);
    }
    else if (img.c == 3)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        ncnn::draw_text_c3(img.data, img.cols, img.rows, text.c_str(), org.x, org.y - fontpixelsize * 2, fontpixelsize, color);
    }
    else if (img.c == 4)
    {
        border_color[0] = _color[0];
        border_color[1] = _color[1];
        border_color[2] = _color[2];
        border_color[3] = _color[3];
        ncnn::draw_text_c4(img.data, img.cols, img.rows, text.c_str(), org.x, org.y - fontpixelsize * 2, fontpixelsize, color);
    }
}

Size getTextSize(const std::string& text, int fontFace, double fontScale, int thickness, int* baseLine)
{
    const int fontpixelsize = 20 * fontScale;

    int w;
    int h;
    ncnn::get_text_drawing_size(text.c_str(), fontpixelsize, &w, &h);

    *baseLine = 0;

    return Size(w, h);
}

#endif // NCNN_PIXEL_DRAWING

} // namespace cv

#endif // NCNN_SIMPLEOCV
