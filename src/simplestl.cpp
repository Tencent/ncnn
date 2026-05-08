// Copyright 2018 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "platform.h"

#if NCNN_SIMPLESTL

#include <stdlib.h>

// allocation functions
void* operator new(size_t size)
{
    return malloc(size);
}

void* operator new[](size_t size)
{
    return malloc(size);
}

// placement allocation functions
void* operator new(size_t /*size*/, void* ptr)
{
    return ptr;
}

void* operator new[](size_t /*size*/, void* ptr)
{
    return ptr;
}

// deallocation functions
void operator delete(void* ptr)
{
    free(ptr);
}

void operator delete[](void* ptr)
{
    free(ptr);
}

// deallocation functions since c++14
#if __cplusplus >= 201402L

void operator delete(void* ptr, size_t sz)
{
    free(ptr);
}

void operator delete[](void* ptr, size_t sz)
{
    free(ptr);
}

#endif

// placement deallocation functions
void operator delete(void* /*ptr*/, void* /*voidptr2*/)
{
}

void operator delete[](void* /*ptr*/, void* /*voidptr2*/)
{
}

extern "C" void __cxa_pure_virtual()
{
    NCNN_LOGE("[Fatal] Pure virtual func called, now exit.");
    // do not abort here to avoid more unpredictable behaviour
}

#endif // NCNN_SIMPLESTL
