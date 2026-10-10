// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mat.h"

#if NCNN_PIXEL

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
#include <android/bitmap.h>
#include <jni.h>
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

namespace ncnn {

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 9
Mat Mat::from_android_bitmap(JNIEnv* env, jobject bitmap, int type_to, Allocator* allocator)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    int type_from;
    int elempack;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_from = PIXEL_GRAY;
        elempack = 1;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_from = PIXEL_RGBA;
        elempack = 4;
    }
    else
    {
        // unsuppored android bitmap format
        return Mat();
    }

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    void* data;
    AndroidBitmap_lockPixels(env, bitmap, &data);

    int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    Mat m = Mat::from_pixels((const unsigned char*)data, type, info.width, info.height, info.stride, allocator);

    AndroidBitmap_unlockPixels(env, bitmap);

    return m;
}

Mat Mat::from_android_bitmap_resize(JNIEnv* env, jobject bitmap, int type_to, int target_width, int target_height, Allocator* allocator)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    int type_from;
    int elempack;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_from = PIXEL_GRAY;
        elempack = 1;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_from = PIXEL_RGBA;
        elempack = 4;
    }
    else
    {
        // unsuppored android bitmap format
        return Mat();
    }

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    void* data;
    AndroidBitmap_lockPixels(env, bitmap, &data);

    int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    Mat m = Mat::from_pixels_resize((const unsigned char*)data, type, info.width, info.height, info.stride, target_width, target_height, allocator);

    AndroidBitmap_unlockPixels(env, bitmap);

    return m;
}

Mat Mat::from_android_bitmap_roi(JNIEnv* env, jobject bitmap, int type_to, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    int type_from;
    int elempack;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_from = PIXEL_GRAY;
        elempack = 1;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_from = PIXEL_RGBA;
        elempack = 4;
    }
    else
    {
        // unsuppored android bitmap format
        return Mat();
    }

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    void* data;
    AndroidBitmap_lockPixels(env, bitmap, &data);

    int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    Mat m = Mat::from_pixels_roi((const unsigned char*)data, type, info.width, info.height, info.stride, roix, roiy, roiw, roih, allocator);

    AndroidBitmap_unlockPixels(env, bitmap);

    return m;
}

Mat Mat::from_android_bitmap_roi_resize(JNIEnv* env, jobject bitmap, int type_to, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    int type_from;
    int elempack;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_from = PIXEL_GRAY;
        elempack = 1;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_from = PIXEL_RGBA;
        elempack = 4;
    }
    else
    {
        // unsuppored android bitmap format
        return Mat();
    }

    // let PIXEL_RGBA2XXX become PIXEL_XXX
    type_to = (type_to & PIXEL_CONVERT_MASK) ? (type_to >> PIXEL_CONVERT_SHIFT) : (type_to & PIXEL_FORMAT_MASK);

    void* data;
    AndroidBitmap_lockPixels(env, bitmap, &data);

    int type = type_to == type_from ? type_from : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    Mat m = Mat::from_pixels_roi_resize((const unsigned char*)data, type, info.width, info.height, info.stride, roix, roiy, roiw, roih, target_width, target_height, allocator);

    AndroidBitmap_unlockPixels(env, bitmap);

    return m;
}

void Mat::to_android_bitmap(JNIEnv* env, jobject bitmap, int type_from) const
{
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    int type_to;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_to = PIXEL_GRAY;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_to = PIXEL_RGBA;
    }
    else
    {
        // unsuppored android bitmap format
        return;
    }

    // let PIXEL_XXX2RGBA become PIXEL_XXX
    type_from = (type_from & PIXEL_CONVERT_MASK) ? (type_from & PIXEL_FORMAT_MASK) : type_from;

    void* _data;
    AndroidBitmap_lockPixels(env, bitmap, &_data);

    int type = type_from == type_to ? type_to : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    to_pixels_resize((unsigned char*)_data, type, info.width, info.height, info.stride);

    AndroidBitmap_unlockPixels(env, bitmap);
}
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

} // namespace ncnn

#endif // NCNN_PIXEL
