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

#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "cpu.h"
#include "layer.h"
#include "layer_type.h"

#include <math.h>

#if NCNN_VULKAN
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API
#endif // NCNN_VULKAN

namespace ncnn {

Mat Mat::clone(Allocator* _allocator) const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, elempack, _allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, _allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, _allocator);
    else if (dims == 4)
        m.create(w, h, d, c, elemsize, elempack, _allocator);

    if (total() > 0)
    {
        if (cstep == m.cstep)
            memcpy(m.data, data, total() * elemsize);
        else
        {
            // copy by channel for differnet cstep
            size_t size = (size_t)w * h * d * elemsize;
            for (int i = 0; i < c; i++)
            {
                memcpy(m.channel(i), channel(i), size);
            }
        }
    }

    return m;
}

void Mat::clone_from(const ncnn::Mat& mat, Allocator* allocator)
{
    *this = mat.clone(allocator);
}

Mat Mat::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * d * c != _w)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.d = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

Mat Mat::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h)
        return Mat();

    if (dims >= 3 && cstep != (size_t)w * h * d)
    {
        Mat m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i = 0; i < c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + (size_t)i * w * h * d * elemsize;
            memcpy(mptr, ptr, (size_t)w * h * d * elemsize);
        }

        return m;
    }

    Mat m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = 1;

    m.cstep = (size_t)_w * _h;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize((size_t)_w * _h * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.d = 1;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * elemsize, 16) / elemsize;

    return m;
}

Mat Mat::reshape(int _w, int _h, int _d, int _c, Allocator* _allocator) const
{
    if (w * h * d * c != _w * _h * _d * _c)
        return Mat();

    if (dims < 3)
    {
        if ((size_t)_w * _h * _d != alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize)
        {
            Mat m;
            m.create(_w, _h, _d, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i = 0; i < _c; i++)
            {
                const void* ptr = (unsigned char*)data + (size_t)i * _w * _h * _d * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, (size_t)_w * _h * _d * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Mat tmp = reshape(_w * _h * _d * _c, _allocator);
        return tmp.reshape(_w, _h, _d, _c, _allocator);
    }

    Mat m = *this;

    m.dims = 4;
    m.w = _w;
    m.h = _h;
    m.d = _d;
    m.c = _c;

    m.cstep = alignSize((size_t)_w * _h * _d * elemsize, 16) / elemsize;

    return m;
}

void Mat::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = (size_t)w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = (size_t)w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize((size_t)w * h * d * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

void Mat::create_like(const Mat& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

#if NCNN_VULKAN
void Mat::create_like(const VkMat& m, Allocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void Mat::create_like(const VkImageMat& im, Allocator* _allocator)
{
    int _dims = im.dims;
    if (_dims == 1)
        create(im.w, im.elemsize, im.elempack, _allocator);
    if (_dims == 2)
        create(im.w, im.h, im.elemsize, im.elempack, _allocator);
    if (_dims == 3)
        create(im.w, im.h, im.c, im.elemsize, im.elempack, _allocator);
    if (_dims == 4)
        create(im.w, im.h, im.d, im.c, im.elemsize, im.elempack, _allocator);
}
#endif // NCNN_VULKAN

#if NCNN_VULKAN
void VkMat::create(int _w, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, int _d, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize(w * h * d * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    cstep = alignSize(w * h * d * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);

        data = allocator->fastMalloc(totalsize);

        refcount = (int*)((unsigned char*)data + offsetof(VkBufferMemory, refcount));
        *refcount = 1;
    }
}

void VkMat::create_like(const Mat& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void VkMat::create_like(const VkMat& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void VkMat::create_like(const VkImageMat& im, VkAllocator* _allocator)
{
    int _dims = im.dims;
    if (_dims == 1)
        create(im.w, im.elemsize, im.elempack, _allocator);
    if (_dims == 2)
        create(im.w, im.h, im.elemsize, im.elempack, _allocator);
    if (_dims == 3)
        create(im.w, im.h, im.c, im.elemsize, im.elempack, _allocator);
    if (_dims == 4)
        create(im.w, im.h, im.d, im.c, im.elemsize, im.elempack, _allocator);
}

void VkImageMat::create(int _w, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, int _d, int _c, size_t _elemsize, VkAllocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    if (total() > 0)
    {
        // underlying image is 3d
        data = allocator->fastMalloc(w, h * d, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    d = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    d = 1;
    c = 1;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    d = 1;
    c = _c;

    if (total() > 0)
    {
        data = allocator->fastMalloc(w, h, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create(int _w, int _h, int _d, int _c, size_t _elemsize, int _elempack, VkAllocator* _allocator)
{
    if (dims == 4 && w == _w && h == _h && d == _d && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 4;
    w = _w;
    h = _h;
    d = _d;
    c = _c;

    if (total() > 0)
    {
        // underlying image is 3d
        data = allocator->fastMalloc(w, h * d, c, elemsize, elempack);
        if (!data)
            return;

        refcount = (int*)((unsigned char*)data + offsetof(VkImageMemory, refcount));
        *refcount = 1;
    }
}

void VkImageMat::create_like(const Mat& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void VkImageMat::create_like(const VkMat& m, VkAllocator* _allocator)
{
    int _dims = m.dims;
    if (_dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    if (_dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    if (_dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
    if (_dims == 4)
        create(m.w, m.h, m.d, m.c, m.elemsize, m.elempack, _allocator);
}

void VkImageMat::create_like(const VkImageMat& im, VkAllocator* _allocator)
{
    int _dims = im.dims;
    if (_dims == 1)
        create(im.w, im.elemsize, im.elempack, _allocator);
    if (_dims == 2)
        create(im.w, im.h, im.elemsize, im.elempack, _allocator);
    if (_dims == 3)
        create(im.w, im.h, im.c, im.elemsize, im.elempack, _allocator);
    if (_dims == 4)
        create(im.w, im.h, im.d, im.c, im.elemsize, im.elempack, _allocator);
}
#endif // NCNN_VULKAN

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    Layer* op;

    if (mean_vals && !norm_vals)
    {
        // substract mean only
        op = create_layer(LayerType::Bias);

        ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        Mat weights[1];
        weights[0] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = -mean_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else if (!mean_vals && norm_vals)
    {
        // normalize only
        op = create_layer(LayerType::Scale);

        ParamDict pd;
        pd.set(0, c);

        op->load_param(pd);

        Mat weights[1];
        weights[0] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = norm_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else if (mean_vals && norm_vals)
    {
        // substract mean and normalize
        op = create_layer(LayerType::Scale);

        ParamDict pd;
        pd.set(0, c);
        pd.set(1, 1);

        op->load_param(pd);

        Mat weights[2];
        weights[0] = Mat(c);
        weights[1] = Mat(c);
        for (int q = 0; q < c; q++)
        {
            weights[0][q] = norm_vals[q];
            weights[1][q] = -mean_vals[q] * norm_vals[q];
        }

        op->load_model(ModelBinFromMatArray(weights));
    }
    else // if (!mean_vals && !norm_vals)
    {
        return;
    }

    Option opt;
    opt.num_threads = 1; // TODO

    op->create_pipeline(opt);

    op->forward_inplace(*this, opt);

    op->destroy_pipeline(opt);

    delete op;
}

Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m; //.data;

#if __ARM_NEON && (__ARM_FP & 2)
    int nn = cpu_support_arm_vfpv4() ? size >> 2 : 0;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON && (__ARM_FP & 2)
#if __aarch64__
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "ld1    {v0.4h}, [%1], #8       \n"
            "fcvtl  v1.4s, v0.4h            \n"
            "subs   %w0, %w0, #1            \n"
            "st1    {v1.4s}, [%2], #16      \n"
            "bne    0b                      \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "v0", "v1");
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #64]           \n"
            "vld1.s16   {d0}, [%1 :64]!     \n"
            "vcvt.f32.f16 q1, d0            \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d2-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),   // %0
            "=r"(data), // %1
            "=r"(ptr)   // %2
            : "0"(nn),
            "1"(data),
            "2"(ptr)
            : "cc", "memory", "q0", "q1");
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain > 0; remain--)
    {
        *ptr = float16_to_float32(*data);

        data++;
        ptr++;
    }

    return m;
}

#if NCNN_VULKAN
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
VkImageMat VkImageMat::from_android_hardware_buffer(VkAndroidHardwareBufferImageAllocator* allocator)
{
    int width = allocator->width();
    int height = allocator->height();

    return VkImageMat(width, height, allocator);
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API
#endif // NCNN_VULKAN

unsigned short float32_to_float16(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        }
        else
        {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

void copy_make_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, int type, float v, const Option& opt)
{
    Layer* padding = create_layer(LayerType::Padding);

    ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);
    pd.set(7, front);
    pd.set(8, behind);

    padding->load_param(pd);

    padding->create_pipeline(opt);

    padding->forward(src, dst, opt);

    padding->destroy_pipeline(opt);

    delete padding;
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const Option& opt)
{
    if (left + right > src.w || top + bottom > src.h)
    {
        NCNN_LOGE("copy_cut_border parameter error, top: %d, bottom: %d, left: %d, right: %d, src.w: %d, src.h: %d", top, bottom, left, right, src.w, src.h);
        return;
    }
    Layer* crop = create_layer(LayerType::Crop);

    ParamDict pd;
    pd.set(0, left);
    pd.set(1, top);
    pd.set(2, 0);
    pd.set(3, src.w - left - right);
    pd.set(4, src.h - top - bottom);
    pd.set(5, -233);

    crop->load_param(pd);

    crop->create_pipeline(opt);

    crop->forward(src, dst, opt);

    crop->destroy_pipeline(opt);

    delete crop;
}

void copy_cut_border_3d(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int front, int behind, const Option& opt)
{
    if (left + right > src.w || top + bottom > src.h || front + behind > src.d)
    {
        NCNN_LOGE("copy_cut_border_3d parameter error, top: %d, bottom: %d, left: %d, right: %d, front: %d, behind: %d, src.w: %d, src.h: %d, src.d: %d", top, bottom, left, right, front, behind, src.w, src.h, src.d);
        return;
    }
    Layer* crop = create_layer(LayerType::Crop);

    ParamDict pd;
    pd.set(0, left);
    pd.set(1, top);
    pd.set(13, front);
    pd.set(2, 0);
    pd.set(3, src.w - left - right);
    pd.set(4, src.h - top - bottom);
    pd.set(14, src.d - front - behind);
    pd.set(5, -233);

    crop->load_param(pd);

    crop->create_pipeline(opt);

    crop->forward(src, dst, opt);

    crop->destroy_pipeline(opt);

    delete crop;
}

void resize_nearest(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void resize_bilinear(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 2);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void resize_bicubic(const Mat& src, Mat& dst, int w, int h, const Option& opt)
{
    Layer* interp = create_layer(LayerType::Interp);

    ParamDict pd;
    pd.set(0, 3);
    pd.set(3, h);
    pd.set(4, w);

    interp->load_param(pd);

    interp->create_pipeline(opt);

    interp->forward(src, dst, opt);

    interp->destroy_pipeline(opt);

    delete interp;
}

void convert_packing(const Mat& src, Mat& dst, int _elempack, const Option& opt)
{
    Layer* packing = create_layer(LayerType::Packing);

    ParamDict pd;
    pd.set(0, _elempack);

    packing->load_param(pd);

    packing->create_pipeline(opt);

    packing->forward(src, dst, opt);

    packing->destroy_pipeline(opt);

    delete packing;
}

void flatten(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* flatten = create_layer(LayerType::Flatten);

    ParamDict pd;

    flatten->load_param(pd);

    flatten->create_pipeline(opt);

    flatten->forward(src, dst, opt);

    flatten->destroy_pipeline(opt);

    delete flatten;
}

void cast_float32_to_float16(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 2);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_float16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 2);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_int8_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 3);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_float32_to_bfloat16(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 4);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void cast_bfloat16_to_float32(const Mat& src, Mat& dst, const Option& opt)
{
    Layer* cast = create_layer(LayerType::Cast);

    ParamDict pd;
    pd.set(0, 4);
    pd.set(1, 1);

    cast->load_param(pd);

    cast->create_pipeline(opt);

    cast->forward(src, dst, opt);

    cast->destroy_pipeline(opt);

    delete cast;
}

void quantize_to_int8(const Mat& src, Mat& dst, const Mat& scale_data, const Option& opt)
{
    Layer* quantize = create_layer(LayerType::Quantize);

    ParamDict pd;
    pd.set(0, scale_data.w);

    quantize->load_param(pd);

    Mat weights[1];
    weights[0] = scale_data;

    quantize->load_model(ModelBinFromMatArray(weights));

    quantize->create_pipeline(opt);

    quantize->forward(src, dst, opt);

    quantize->destroy_pipeline(opt);

    delete quantize;
}

void dequantize_from_int32(const Mat& src, Mat& dst, const Mat& scale_data, const Mat& bias_data, const Option& opt)
{
    Layer* dequantize = create_layer(LayerType::Dequantize);

    ParamDict pd;
    pd.set(0, scale_data.w);
    pd.set(1, bias_data.w);

    dequantize->load_param(pd);

    Mat weights[2];
    weights[0] = scale_data;
    weights[1] = bias_data;

    dequantize->load_model(ModelBinFromMatArray(weights));

    dequantize->create_pipeline(opt);

    dequantize->forward(src, dst, opt);

    dequantize->destroy_pipeline(opt);

    delete dequantize;
}

void requantize_from_int32_to_int8(const Mat& src, Mat& dst, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    Layer* requantize = create_layer(LayerType::Requantize);

    ParamDict pd;
    pd.set(0, scale_in_data.w);
    pd.set(1, scale_out_data.w);
    pd.set(2, bias_data.w);
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    requantize->load_param(pd);

    Mat weights[3];
    weights[0] = scale_in_data;
    weights[1] = scale_out_data;
    weights[2] = bias_data;

    requantize->load_model(ModelBinFromMatArray(weights));

    requantize->create_pipeline(opt);

    requantize->forward(src, dst, opt);

    requantize->destroy_pipeline(opt);

    delete requantize;
}

} // namespace ncnn
