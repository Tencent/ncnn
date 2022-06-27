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

#include "imreadwrite.h"

#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "../../src/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../src/stb_image_write.h"

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

    memcpy(img.data, pixeldata, static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(c));

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

} // namespace cv
