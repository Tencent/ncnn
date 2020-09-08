// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __ANDROID_API__ >= 9
#include <android/bitmap.h>
#include <jni.h>
#endif // __ANDROID_API__ >= 9

namespace ncnn {

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
    int elempack;

    if (info.format == ANDROID_BITMAP_FORMAT_A_8)
    {
        type_to = PIXEL_GRAY;
        elempack = 1;
    }
    else if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888)
    {
        type_to = PIXEL_RGBA;
        elempack = 4;
    }
    else
    {
        // unsuppored android bitmap format
        return;
    }

    // let PIXEL_XXX2RGBA become PIXEL_XXX
    type_from = (type_from & PIXEL_CONVERT_MASK) ? (type_from & PIXEL_FORMAT_MASK) : type_from;

    void* data;
    AndroidBitmap_lockPixels(env, bitmap, &data);

    int type = type_from == type_to ? type_to : (type_from | (type_to << PIXEL_CONVERT_SHIFT));

    to_pixels_resize((unsigned char*)data, type, info.width, info.height, info.stride);

    AndroidBitmap_unlockPixels(env, bitmap);
}
#endif // __ANDROID_API__ >= 9

} // namespace ncnn

#endif // NCNN_PIXEL
