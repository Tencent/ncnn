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

#include "mat.h"

#if NCNN_PIXEL
#if NCNN_PLATFORM_API
#if __APPLE__

#import <CoreGraphics/CoreGraphics.h>

namespace ncnn {

Mat Mat::from_apple_samplebuffer(CMSampleBufferRef samplebuffer, int type_to, Allocator* allocator)
{
    CMFormatDescriptionRef des = CMSampleBufferGetFormatDescription(samplebuffer);
    if (!des)
        return Mat();

    if (CMFormatDescriptionGetMediaType(des) != kCMMediaType_Video)
        return Mat();

    CVPixelBufferRef pixel = CMSampleBufferGetImageBuffer(samplebuffer);

    CVPixelBufferRetain(pixel);

    Mat m = Mat::from_apple_pixelbuffer(pixel, type_to, allocator);

    CVPixelBufferRelease(pixel);

    return m;
}

Mat Mat::from_apple_pixelbuffer(CVPixelBufferRef pixelbuffer, int type_to, Allocator* allocator)
{
    const int w = CVPixelBufferGetWidth(pixelbuffer);
    const int h = CVPixelBufferGetHeight(pixelbuffer);
    const int stride = CVPixelBufferGetBytesPerRow(pixelbuffer);
    const OSType format = CVPixelBufferGetPixelFormatType(pixelbuffer);

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    if (format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
    {
        if (type_to == PIXEL_GRAY)
        {
            // fast path for y-channel to gray
            CVPixelBufferLockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

            const unsigned char* y_data = (const unsigned char*)CVPixelBufferGetBaseAddressOfPlane(pixelbuffer, 0);

            Mat m = from_pixels(y_data, PIXEL_GRAY, w, h, stride, allocator);

            CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

            return m;
        }
        else
        {
            // convert to rgb
            Mat rgb(w, h, (size_t)3u, 3);

            CVPixelBufferLockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

            const unsigned char* y_data = (const unsigned char*)CVPixelBufferGetBaseAddressOfPlane(pixelbuffer, 0);
            const unsigned char* uv_data = (const unsigned char*)CVPixelBufferGetBaseAddressOfPlane(pixelbuffer, 1);
            const int y_stride = CVPixelBufferGetBytesPerRowOfPlane(pixelbuffer, 0);
            const int uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pixelbuffer, 1);

            if (uv_data == y_data + w * h && y_stride == w && uv_stride == w)
            {
                // already nv12  :)
                yuv420sp2rgb_nv12(y_data, w, h, (unsigned char*)rgb.data);
            }
            else
            {
                // construct nv12
                unsigned char* nv12 = new unsigned char[w * h + w * h / 2];
                {
                    // Y
                    for (int y = 0; y < h; y++)
                    {
                        unsigned char* yptr = nv12 + w * y;
                        const unsigned char* y_data_ptr = y_data + y_stride * y;
                        memcpy(yptr, y_data_ptr, w);
                    }

                    // UV
                    for (int y = 0; y < h / 2; y++)
                    {
                        unsigned char* uvptr = nv12 + w * h + w * y;
                        const unsigned char* uv_data_ptr = uv_data + uv_stride * y;
                        memcpy(uvptr, uv_data_ptr, w);
                    }
                }

                yuv420sp2rgb_nv12(nv12, w, h, (unsigned char*)rgb.data);

                delete[] nv12;
            }

            CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

            int type_from = PIXEL_RGB;
            int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

            return from_pixels((const unsigned char*)rgb.data, type, w, h, allocator);
        }
    }

    if (format == kCVPixelFormatType_32ARGB)
    {
        CVPixelBufferLockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

        const unsigned char* rgba_data = (const unsigned char*)CVPixelBufferGetBaseAddress(pixelbuffer);

        int type_from = PIXEL_BGRA;
        int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

        Mat m = from_pixels(rgba_data, type, w, h, stride, allocator);

        CVPixelBufferUnlockBaseAddress(pixelbuffer, kCVPixelBufferLock_ReadOnly);

        return m;
    }

    // unsupported format
    return Mat();
}

int Mat::to_apple_pixelbuffer(CVPixelBufferRef pixelbuffer, int type_from) const
{
    const int target_width = CVPixelBufferGetWidth(pixelbuffer);
    const int target_height = CVPixelBufferGetHeight(pixelbuffer);
    const int target_stride = CVPixelBufferGetBytesPerRow(pixelbuffer);
    const OSType format = CVPixelBufferGetPixelFormatType(pixelbuffer);

    int type_to;

    if (format == kCVPixelFormatType_24RGB)
    {
        type_to = PIXEL_BGR;
    }
    else if (format == kCVPixelFormatType_32ARGB)
    {
        type_to = PIXEL_BGRA;
    }
    else
    {
        // unsuppored apple pixelbuffer format
        return -1;
    }

    // let PIXEL_XXX2RGBA become PIXEL_XXX
    type_from = (type_from & PIXEL_CONVERT_MASK) ? (type_from & PIXEL_FORMAT_MASK) : type_from;

    CVPixelBufferLockBaseAddress(pixelbuffer, 0);

    int type = type_from == type_to ? type_to : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    unsigned char* data = (unsigned char*)CVPixelBufferGetBaseAddress(pixelbuffer);

    to_pixels_resize(data, type, target_width, target_height, target_stride);

    CVPixelBufferUnlockBaseAddress(pixelbuffer, 0);

    return 0;
}

Mat Mat::from_apple_cgimage(CGImageRef image, int type_to, Allocator* allocator)
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image);
    int w = (int)CGImageGetWidth(image);
    int h = (int)CGImageGetHeight(image);

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    if (CGColorSpaceGetModel(colorSpace) == kCGColorSpaceModelMonochrome)
    {
        // fast path for monochrome source
        Mat gray(w, h, (size_t)1u, 1);

        CGContextRef context = CGBitmapContextCreate(gray.data, w, h, 8, w, colorSpace, kCGImageAlphaNone);

        CGContextDrawImage(context, CGRectMake(0, 0, w, h), image);
        CGContextRelease(context);

        int type_from = PIXEL_GRAY;
        int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

        return from_pixels((const unsigned char*)gray.data, type, w, h, allocator);
    }

    // always fallback to rgba
    colorSpace = CGColorSpaceCreateDeviceRGB();

    Mat rgba(w, h, (size_t)4u, 4);

    CGContextRef context = CGBitmapContextCreate(rgba.data, w, h, 8, w * 4, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);

    CGColorSpaceRelease(colorSpace);

    CGContextDrawImage(context, CGRectMake(0, 0, w, h), image);
    CGContextRelease(context);

    int type_from = PIXEL_RGBA;
    int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    return from_pixels((const unsigned char*)rgba.data, type, w, h, allocator);
}

CGImageRef Mat::to_apple_cgimage() const
{
    if (c == 1)
    {
        // gray
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();

        Mat gray(w, h, (size_t)1u, 1);
        to_pixels((unsigned char*)gray.data, PIXEL_GRAY, w);

        CGContextRef newContext = CGBitmapContextCreate(gray.data, w, h, 8, w, colorSpace, kCGImageAlphaNone | kCGBitmapByteOrderDefault);

        CGImageRef image = CGBitmapContextCreateImage(newContext);

        CGContextRelease(newContext);
        CGColorSpaceRelease(colorSpace);

        return image;
    }

    if (c == 3)
    {
        // rgb
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

        Mat rgb(w, h, (size_t)3u, 3);
        to_pixels((unsigned char*)rgb.data, PIXEL_RGB, w * 3);

        CGContextRef newContext = CGBitmapContextCreate(rgb.data, w, h, 8, w * 3, colorSpace, kCGImageAlphaNone | kCGBitmapByteOrderDefault);

        CGImageRef image = CGBitmapContextCreateImage(newContext);

        CGContextRelease(newContext);
        CGColorSpaceRelease(colorSpace);

        return image;
    }

    if (c == 4)
    {
        // rgba
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

        Mat rgba(w, h, (size_t)4u, 4);
        to_pixels((unsigned char*)rgba.data, PIXEL_RGBA, w * 4);

        CGContextRef newContext = CGBitmapContextCreate(rgba.data, w, h, 8, w * 4, colorSpace, kCGImageAlphaLast | kCGBitmapByteOrderDefault);

        CGImageRef image = CGBitmapContextCreateImage(newContext);

        CGContextRelease(newContext);
        CGColorSpaceRelease(colorSpace);

        return image;
    }

    // unsupported from format
    return nil;
}

#if TARGET_OS_IOS
Mat Mat::from_apple_uiimage(UIImage* image, int type_to, Allocator* allocator)
{
    CGImageRef imageRef = image.CGImage;
    return from_apple_cgimage(imageRef, type_to, allocator);
}

UIImage* Mat::to_apple_uiimage() const
{
    CGImageRef image = to_apple_cgimage();

    UIImage* uiImage = [UIImage imageWithCGImage:image];
    CGImageRelease(image);

    return uiImage;
}
#else
Mat Mat::from_apple_nsimage(NSImage* image, int type_to, Allocator* allocator)
{
    CGImageRef imageRef = [image CGImageForProposedRect:NULL context:NULL hints:NULL];
    return from_apple_cgimage(imageRef, type_to, allocator);
}

NSImage* Mat::to_apple_nsimage() const
{
    CGImageRef image = to_apple_cgimage();

    NSImage* nsImage = [[NSImage alloc] initWithCGImage:image size:NSMakeSize(w, h)];
    CGImageRelease(image);

    return nsImage;
}
#endif

} // namespace ncnn

#endif // __APPLE__

#endif // NCNN_PLATFORM_API
#endif // NCNN_PIXEL
