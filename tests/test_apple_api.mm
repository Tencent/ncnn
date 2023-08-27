// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#import <Cocoa/Cocoa.h>
#import <CoreVideo/CoreVideo.h>

#include <stdio.h>
#include <string.h>
#include "testutil.h"
#include "mat.h"

static ncnn::Mat generate_ncnn_logo(int w, int h)
{
    // clang-format off
    // *INDENT-OFF*
    static const unsigned char ncnn_logo_data[16][16] =
    {
        {245, 245,  33, 245, 245, 245, 245, 245, 245, 245, 245, 245, 245,  33, 245, 245},
        {245,  33,  33,  33, 245, 245, 245, 245, 245, 245, 245, 245,  33,  33,  33, 245},
        {245,  33, 158, 158,  33, 245, 245, 245, 245, 245, 245,  33, 158, 158,  33, 245},
        { 33, 117, 158, 224, 158,  33, 245, 245, 245, 245,  33, 158, 224, 158, 117,  33},
        { 33, 117, 224, 224, 224,  66,  33,  33,  33,  33,  66, 224, 224, 224, 117,  33},
        { 33, 189, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 189,  33},
        { 33, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97,  97,  97,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  33,   0, 189, 224, 224,  97,   0,  33,  97, 224, 224,  33},
        { 33, 224, 224,  97,  97,  97,  97, 224, 224,  97, 189, 189,  97, 224, 224,  33},
        { 33,  66,  66,  66, 224, 224, 224, 224, 224, 224, 224, 224,  66,  66,  66,  33},
        { 66, 158, 158,  66,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158,  66,  66},
        { 66, 158, 158, 208,  66, 224, 224, 224, 224, 224, 224,  66, 158, 158, 208,  66},
        { 66, 224, 202, 158,  66, 224, 224, 224, 224, 224, 224,  66, 224, 202, 158,  66},
        { 66, 158, 224, 158,  66, 224, 224, 224, 224, 224, 224,  66, 158, 224, 158,  66}
    };
    // *INDENT-ON*
    // clang-format on

    ncnn::Mat m(w, h, (size_t)1, 1);
    resize_bilinear_c1((const unsigned char*)ncnn_logo_data, 16, 16, m, w, h);
    return m;
}

int main()
{
    ncnn::Mat m = generate_ncnn_logo(256, 256);
    fprintf(stderr, "m = %p\n", m.data);

    NSImage* nsimg = m.to_apple_nsimage();
    fprintf(stderr, "nsimg = %p\n", nsimg);

    ncnn::Mat m2 = ncnn::Mat::from_apple_nsimage(nsimg, ncnn::Mat::PIXEL_RGB);
    ncnn::Mat m3 = ncnn::Mat::from_apple_nsimage(nsimg, ncnn::Mat::PIXEL_GRAY);
    fprintf(stderr, "m2 = %p\n", m2.data);
    fprintf(stderr, "m3 = %p\n", m3.data);

    ncnn::Mat gray(256, 256, (size_t)1u, 1);
    m2.to_pixels((unsigned char*)gray.data, ncnn::Mat::PIXEL_RGB2GRAY);

    if (memcmp(m.data, gray.data, 256 * 256) != 0)
    {
        fprintf(stderr, "m gray data mismatch\n");
        return -1;
    }

    NSImage* nsimg2 = m.to_apple_nsimage();
    fprintf(stderr, "nsimg2 = %p\n", nsimg2);

    ncnn::Mat m4 = ncnn::Mat::from_apple_nsimage(nsimg2, ncnn::Mat::PIXEL_GRAY);
    fprintf(stderr, "m4 = %p\n", m4.data);

    if (memcmp(m3.data, m4.data, 256 * 256 * sizeof(float)) != 0)
    {
        fprintf(stderr, "m3 m4 data mismatch\n");
        return -1;
    }

    return 0;
}
