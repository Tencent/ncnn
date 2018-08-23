// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_ALLOCATOR_H
#define NCNN_ALLOCATOR_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdlib.h>
#include <list>

namespace ncnn {

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

// exchange-add operation for atomic operations on reference counters
#if defined __INTEL_COMPILER && !(defined WIN32 || defined _WIN32)
// atomic increment on the linux version of the Intel(tm) compiler
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd(const_cast<void*>(reinterpret_cast<volatile void*>(addr)), delta)
#elif defined __GNUC__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define NCNN_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define NCNN_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define NCNN_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define NCNN_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
// thread-unsafe branch
static inline int NCNN_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif

#ifdef _WIN32
class Mutex
{
public:
    Mutex() { InitializeSRWLock(&srwlock); }
    ~Mutex() {}
    void lock() { AcquireSRWLockExclusive(&srwlock); }
    void unlock() { ReleaseSRWLockExclusive(&srwlock); }
private:
    // NOTE SRWLock is available from windows vista
    SRWLOCK srwlock;
};
#else // _WIN32
class Mutex
{
public:
    Mutex() { pthread_mutex_init(&mutex, 0); }
    ~Mutex() { pthread_mutex_destroy(&mutex); }
    void lock() { pthread_mutex_lock(&mutex); }
    void unlock() { pthread_mutex_unlock(&mutex); }
private:
    pthread_mutex_t mutex;
};
#endif // _WIN32

class Allocator
{
public:
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio;// 0~256
    std::list< std::pair<size_t, void*> > budgets;
    std::list< std::pair<size_t, void*> > payouts;
};

class UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    unsigned int size_compare_ratio;// 0~256
    std::list< std::pair<size_t, void*> > budgets;
    std::list< std::pair<size_t, void*> > payouts;
};

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
