// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PYBIND11_NCNN_ALLOCATOR_H
#define PYBIND11_NCNN_ALLOCATOR_H

#include <allocator.h>

template<class Base = ncnn::Allocator>
class PyAllocator : public Base
{
public:
    using Base::Base; // Inherit constructors
    void* fastMalloc(size_t size) override
    {
        PYBIND11_OVERRIDE_PURE(void*, Base, fastMalloc, size);
    }
    void fastFree(void* ptr) override
    {
        PYBIND11_OVERRIDE_PURE(void, Base, fastFree, ptr);
    }
};

template<class Other>
class PyAllocatorOther : public PyAllocator<Other>
{
public:
    using PyAllocator<Other>::PyAllocator;
    void* fastMalloc(size_t size) override
    {
        PYBIND11_OVERRIDE(void*, Other, fastMalloc, size);
    }
    void fastFree(void* ptr) override
    {
        PYBIND11_OVERRIDE(void, Other, fastFree, ptr);
    }
};

#if NCNN_VULKAN
template<class Base = ncnn::VkAllocator>
class PyVkAllocator : public Base
{
public:
    using Base::Base; // Inherit constructors
    void clear() override
    {
        PYBIND11_OVERRIDE(void, Base, clear, );
    }
    ncnn::VkBufferMemory* fastMalloc(size_t size) override
    {
        PYBIND11_OVERRIDE_PURE(ncnn::VkBufferMemory*, Base, fastMalloc, size);
    }
    void fastFree(ncnn::VkBufferMemory* ptr) override
    {
        PYBIND11_OVERRIDE_PURE(void, Base, fastFree, ptr);
    }
    int flush(ncnn::VkBufferMemory* ptr) override
    {
        PYBIND11_OVERRIDE(int, Base, flush, ptr);
    }
    int invalidate(ncnn::VkBufferMemory* ptr) override
    {
        PYBIND11_OVERRIDE(int, Base, invalidate, ptr);
    }
};

template<class Other>
class PyVkAllocatorOther : public PyVkAllocator<Other>
{
public:
    using PyVkAllocator<Other>::PyVkAllocator;
    void clear() override
    {
        PYBIND11_OVERRIDE(void, Other, clear, );
    }
    ncnn::VkBufferMemory* fastMalloc(size_t size) override
    {
        PYBIND11_OVERRIDE(ncnn::VkBufferMemory*, Other, fastMalloc, size);
    }
    void fastFree(ncnn::VkBufferMemory* ptr) override
    {
        PYBIND11_OVERRIDE(void, Other, fastFree, ptr);
    }
};

template<class Base = ncnn::VkBlobAllocator>
class PyVkBlobAllocator : public Base
{
public:
    using Base::Base; // Inherit constructors
    void clear() override
    {
        PYBIND11_OVERRIDE(void, Base, clear, );
    }
    ncnn::VkImageMemory* fastMalloc(int width, int height, VkFormat format) override
    {
        PYBIND11_OVERRIDE_PURE(ncnn::VkImageMemory*, Base, fastMalloc, width, height, format);
    }
    void fastFree(ncnn::VkImageMemory* ptr) override
    {
        PYBIND11_OVERRIDE_PURE(void, Base, fastFree, ptr);
    }
};

//template<class Other>
//class PyVkImageAllocatorOther : public PyVkImageAllocator<Other>
//{
//public:
//    using PyVkImageAllocator<Other>::PyVkImageAllocator;
//    ncnn::VkImageMemory* fastMalloc(int width, int height,
//                                    VkFormat format) override
//    {
//        PYBIND11_OVERRIDE(ncnn::VkImageMemory*, Other, fastMalloc, width, height, format);
//    }
//    void fastFree(ncnn::VkImageMemory* ptr) override
//    {
//        PYBIND11_OVERRIDE(void, Other, fastFree, ptr);
//    }
//};
#endif // NCNN_VULKAN

#endif
