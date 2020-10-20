// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#include "platform.h"

#if NCNN_SIMPLESTL

#include <stdlib.h>

void* operator new(size_t sz) noexcept
{
    void* ptr = malloc(sz);
    return ptr;
}

void* operator new(size_t sz, void* ptr) noexcept
{
    return ptr;
}

void* operator new[](size_t sz) noexcept
{
    void* ptr = malloc(sz);
    return ptr;
}

void* operator new[](size_t sz, void* ptr) noexcept
{
    return ptr;
}

void operator delete(void* ptr)noexcept
{
    free(ptr);
}

void operator delete(void* ptr, size_t sz)noexcept
{
    free(ptr);
}

void operator delete[](void* ptr) noexcept
{
    free(ptr);
}

void operator delete[](void* ptr, size_t sz) noexcept
{
    free(ptr);
}

extern "C" void __cxa_pure_virtual()
{
    NCNN_LOGE("[Fatal] Pure virtual func called, now exit.");
    // do not abort here to avoid more unpredictable behaviour
}

#endif // NCNN_SIMPLESTL
