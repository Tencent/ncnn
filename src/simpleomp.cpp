// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#if NCNN_SIMPLEOMP

#include "simpleomp.h"
#include "cpu.h" // ncnn::get_cpu_count()

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

extern "C" typedef void (*kmpc_micro)(int32_t* gtid, int32_t* tid, ...);

#ifdef __EMSCRIPTEN__
// workaround for emscipten function signature mismatch
extern "C" typedef void (*kmpc_micro_0)(int32_t* gtid, int32_t* tid);
extern "C" typedef void (*kmpc_micro_1)(int32_t* gtid, int32_t* tid, void*);
extern "C" typedef void (*kmpc_micro_2)(int32_t* gtid, int32_t* tid, void*, void*);
extern "C" typedef void (*kmpc_micro_3)(int32_t* gtid, int32_t* tid, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_4)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_5)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_6)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_7)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_8)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_9)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_10)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_11)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_12)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_13)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_14)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_15)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
#endif // __EMSCRIPTEN__

static void init_g_kmp_global();
static void* kmp_threadfunc(void* args);

namespace ncnn {

class KMPTask
{
public:
    // per-team
    kmpc_micro fn;
    int argc;
    void** argv;
    int num_threads;

    // per-task
    int thread_num;

    // finish status
    int* num_threads_to_wait;
    Mutex* finish_lock;
    ConditionVariable* finish_condition;
};

class KMPTaskQueue
{
public:
    KMPTaskQueue(int _max_size)
    {
        max_size = _max_size;
        tasks = new KMPTask*[max_size];
        size = 0;
        front = 0;
        back = 0;
    }

    ~KMPTaskQueue()
    {
        delete[] tasks;
    }

    void dispatch(KMPTask* v, int n)
    {
        lock.lock();

        if (size + n > max_size)
        {
            lock.unlock();

            for (int i = 0; i < n; i++)
            {
                put(&v[i]);
            }
            return;
        }

        for (int i = 0; i < n; i++)
        {
            tasks[back] = &v[i];
            back++;
            if (back == max_size)
                back = 0;
        }

        size += n;

        lock.unlock();

        condition.signal();
    }

    void put(KMPTask* v)
    {
        lock.lock();
        while (size >= max_size)
        {
            condition.wait(lock);
        }
        tasks[back] = v;
        back++;
        if (back == max_size)
            back = 0;
        size++;
        lock.unlock();

        condition.signal();
    }

    void get(KMPTask*& v)
    {
        lock.lock();
        while (size == 0)
        {
            condition.wait(lock);
        }
        v = tasks[front];
        front++;
        if (front == max_size)
            front = 0;
        size--;
        lock.unlock();

        condition.signal();
    }

private:
    Mutex lock;
    ConditionVariable condition;

    // ring buffer queue
    int max_size;
    KMPTask** tasks;
    int size;
    int front;
    int back;
};

class KMPGlobal
{
public:
    KMPGlobal()
    {
        is_initialized = PTHREAD_ONCE_INIT;

        kmp_max_threads = 0;
        kmp_threads = 0;
        kmp_threads_tid = 0;
        kmp_task_queue = 0;
    }

    ~KMPGlobal()
    {
        deinit();
    }

    void try_init()
    {
        pthread_once(&is_initialized, init_g_kmp_global);
    }

public:
    pthread_once_t is_initialized;

    void init()
    {
        // NCNN_LOGE("KMPGlobal init");
        kmp_max_threads = ncnn::get_cpu_count();

        kmp_task_queue = new ncnn::KMPTaskQueue(std::max(kmp_max_threads * 4, 16));

        if (kmp_max_threads > 1)
        {
            kmp_threads = new ncnn::Thread*[kmp_max_threads - 1];
            kmp_threads_tid = new int[kmp_max_threads - 1];
            for (int i = 0; i < kmp_max_threads - 1; i++)
            {
                kmp_threads_tid[i] = i + 1;
                kmp_threads[i] = new ncnn::Thread(kmp_threadfunc, (void*)&kmp_threads_tid[i]);
            }
        }
    }

    void deinit()
    {
        // NCNN_LOGE("KMPGlobal deinit");
        if (kmp_max_threads > 1)
        {
            // TODO portable stack allocation
            ncnn::KMPTask* tasks = (ncnn::KMPTask*)alloca((kmp_max_threads - 1) * sizeof(ncnn::KMPTask));
            for (int i = 0; i < kmp_max_threads - 1; i++)
            {
                tasks[i].fn = 0;
                tasks[i].argc = 0;
                tasks[i].argv = (void**)0;
                tasks[i].num_threads = kmp_max_threads;
                tasks[i].thread_num = i + 1;
                tasks[i].num_threads_to_wait = 0;
                tasks[i].finish_lock = 0;
                tasks[i].finish_condition = 0;
            }

            // dispatch 1 ~ kmp_max_threads
            kmp_task_queue->dispatch(tasks, kmp_max_threads - 1);

            for (int i = 0; i < kmp_max_threads - 1; i++)
            {
#ifndef __EMSCRIPTEN__
                // FIXME emscripten complains
                // pthread_join attempted on thread 12345678,
                // which does not point to a valid thread, or does not exist anymore!
                kmp_threads[i]->join();
#endif
                delete kmp_threads[i];
            }
            delete[] kmp_threads;
            delete[] kmp_threads_tid;
        }

        delete kmp_task_queue;
    }

public:
    int kmp_max_threads;
    ncnn::Thread** kmp_threads;
    int* kmp_threads_tid;
    ncnn::KMPTaskQueue* kmp_task_queue;
};

} // namespace ncnn

static ncnn::KMPGlobal g_kmp_global;

static ncnn::ThreadLocalStorage tls_num_threads;
static ncnn::ThreadLocalStorage tls_thread_num;

static void init_g_kmp_global()
{
    g_kmp_global.init();
}

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_max_threads()
{
    return ncnn::get_cpu_count();
}

int omp_get_dynamic()
{
    return 1;
}

void omp_set_dynamic(int /*dynamic*/)
{
    // always dynamic, ignore
}

void omp_set_num_threads(int num_threads)
{
    tls_num_threads.set(reinterpret_cast<void*>((size_t)std::max(num_threads, 1)));
}

int omp_get_num_threads()
{
    return std::max((int)reinterpret_cast<size_t>(tls_num_threads.get()), 1);
}

int omp_get_thread_num()
{
    return (int)reinterpret_cast<size_t>(tls_thread_num.get());
}

int kmp_get_blocktime()
{
    return 0;
}

void kmp_set_blocktime(int /*blocktime*/)
{
    // always passive, ignore
}

static int kmp_invoke_microtask(kmpc_micro fn, int gtid, int tid, int argc, void** argv)
{
    // fprintf(stderr, "__kmp_invoke_microtask #%lu %d %d %d\n", gettid(), gtid, tid, argc);
#ifdef __EMSCRIPTEN__
    // workaround for emscipten function signature mismatch
    switch (argc)
    {
    case 0:
        (*(kmpc_micro_0)fn)(&gtid, &tid);
        break;
    case 1:
        (*(kmpc_micro_1)fn)(&gtid, &tid, argv[0]);
        break;
    case 2:
        (*(kmpc_micro_2)fn)(&gtid, &tid, argv[0], argv[1]);
        break;
    case 3:
        (*(kmpc_micro_3)fn)(&gtid, &tid, argv[0], argv[1], argv[2]);
        break;
    case 4:
        (*(kmpc_micro_4)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3]);
        break;
    case 5:
        (*(kmpc_micro_5)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4]);
        break;
    case 6:
        (*(kmpc_micro_6)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
        break;
    case 7:
        (*(kmpc_micro_7)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
        break;
    case 8:
        (*(kmpc_micro_8)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
        break;
    case 9:
        (*(kmpc_micro_9)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
        break;
    case 10:
        (*(kmpc_micro_10)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);
        break;
    case 11:
        (*(kmpc_micro_11)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10]);
        break;
    case 12:
        (*(kmpc_micro_12)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11]);
        break;
    case 13:
        (*(kmpc_micro_13)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12]);
        break;
    case 14:
        (*(kmpc_micro_14)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13]);
        break;
    case 15:
        (*(kmpc_micro_15)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14]);
        break;
    default:
        // assert never reach here
        break;
    }
#else  // __EMSCRIPTEN__
    switch (argc)
    {
    case 0:
        (*fn)(&gtid, &tid);
        break;
    case 1:
        (*fn)(&gtid, &tid, argv[0]);
        break;
    case 2:
        (*fn)(&gtid, &tid, argv[0], argv[1]);
        break;
    case 3:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2]);
        break;
    case 4:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3]);
        break;
    case 5:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4]);
        break;
    case 6:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
        break;
    case 7:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
        break;
    case 8:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
        break;
    case 9:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
        break;
    case 10:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);
        break;
    case 11:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10]);
        break;
    case 12:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11]);
        break;
    case 13:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12]);
        break;
    case 14:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13]);
        break;
    case 15:
        (*fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14]);
        break;
    default:
        // assert never reach here
        break;
    }
#endif // __EMSCRIPTEN__

    return 0;
}

static void* kmp_threadfunc(void* args)
{
    int tid = *(int*)args;

    for (;;)
    {
        ncnn::KMPTask* task;
        g_kmp_global.kmp_task_queue->get(task);

        // fprintf(stderr, "get %d\n", tid);

        if (!task->fn)
            break;

        tls_num_threads.set(reinterpret_cast<void*>((size_t)task->num_threads));
        tls_thread_num.set(reinterpret_cast<void*>((size_t)task->thread_num));

        kmp_invoke_microtask(task->fn, task->thread_num, tid, task->argc, task->argv);

        // update finished
        {
            task->finish_lock->lock();
            *task->num_threads_to_wait = *task->num_threads_to_wait - 1;
            if (*task->num_threads_to_wait == 0)
            {
                task->finish_condition->signal();
            }
            task->finish_lock->unlock();
        }
    }

    // fprintf(stderr, "exit\n");
    return 0;
}

int32_t __kmpc_global_thread_num(void* /*loc*/)
{
    // NCNN_LOGE("__kmpc_global_thread_num");
    return 0;
}

void __kmpc_push_num_threads(void* /*loc*/, int32_t /*gtid*/, int32_t num_threads)
{
    // NCNN_LOGE("__kmpc_push_num_threads %d", num_threads);
    omp_set_num_threads(num_threads);
}

void __kmpc_fork_call(void* /*loc*/, int32_t argc, kmpc_micro fn, ...)
{
    g_kmp_global.try_init();

    // NCNN_LOGE("__kmpc_fork_call %d", argc);
    int num_threads = omp_get_num_threads();

    // build argv
    void* argv[16];
    {
        va_list ap;
        va_start(ap, fn);
        for (int i = 0; i < argc; i++)
            argv[i] = va_arg(ap, void*);
        va_end(ap);
    }

    if (g_kmp_global.kmp_max_threads == 1 || num_threads == 1)
    {
        for (int i = 0; i < num_threads; i++)
        {
            tls_thread_num.set(reinterpret_cast<void*>((size_t)i));

            kmp_invoke_microtask(fn, 0, 0, argc, argv);
        }

        return;
    }

    int num_threads_to_wait = num_threads - 1;
    ncnn::Mutex finish_lock;
    ncnn::ConditionVariable finish_condition;

    // TODO portable stack allocation
    ncnn::KMPTask* tasks = (ncnn::KMPTask*)alloca((num_threads - 1) * sizeof(ncnn::KMPTask));
    for (int i = 0; i < num_threads - 1; i++)
    {
        tasks[i].fn = fn;
        tasks[i].argc = argc;
        tasks[i].argv = (void**)argv;
        tasks[i].num_threads = num_threads;
        tasks[i].thread_num = i + 1;
        tasks[i].num_threads_to_wait = &num_threads_to_wait;
        tasks[i].finish_lock = &finish_lock;
        tasks[i].finish_condition = &finish_condition;
    }

    // dispatch 1 ~ num_threads
    g_kmp_global.kmp_task_queue->dispatch(tasks, num_threads - 1);

    // dispatch 0
    {
        tls_thread_num.set(reinterpret_cast<void*>((size_t)0));

        kmp_invoke_microtask(fn, 0, 0, argc, argv);
    }

    // wait for finished
    {
        finish_lock.lock();
        if (num_threads_to_wait != 0)
        {
            finish_condition.wait(finish_lock);
        }
        finish_lock.unlock();
    }
}

void __kmpc_for_static_init_4(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, int32_t* lower, int32_t* upper, int32_t* /*stride*/, int32_t /*incr*/, int32_t /*chunk*/)
{
    // NCNN_LOGE("__kmpc_for_static_init_4");
    int num_threads = omp_get_num_threads();

    // TODO only support i++
    int32_t count = *upper - *lower + 1;
    int32_t threads = std::min(count, (int32_t)num_threads);
    int32_t count_per_thread = count / threads;
    int32_t remain = count % threads;

    *last = gtid == (int32_t)(threads - 1);
    *lower = gtid * count_per_thread + std::min(remain, gtid);
    *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, gtid + 1) - 1, *upper);
}

void __kmpc_for_static_init_4u(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, uint32_t* lower, uint32_t* upper, int32_t* /*stride*/, int32_t /*incr*/, int32_t /*chunk*/)
{
    // NCNN_LOGE("__kmpc_for_static_init_4u");
    int num_threads = omp_get_num_threads();

    // TODO only support i++
    uint32_t count = *upper - *lower + 1;
    uint32_t threads = std::min(count, (uint32_t)num_threads);
    uint32_t count_per_thread = count / threads;
    uint32_t remain = count % threads;

    *last = gtid == (int32_t)(threads - 1);
    *lower = gtid * count_per_thread + std::min(remain, (uint32_t)gtid);
    *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (uint32_t)gtid + 1) - 1, *upper);
}

void __kmpc_for_static_init_8(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, int64_t* lower, int64_t* upper, int64_t* /*stride*/, int64_t /*incr*/, int64_t /*chunk*/)
{
    // NCNN_LOGE("__kmpc_for_static_init_8");
    int num_threads = omp_get_num_threads();

    // TODO only support i++
    int64_t count = *upper - *lower + 1;
    int64_t threads = std::min(count, (int64_t)num_threads);
    int64_t count_per_thread = count / threads;
    int64_t remain = count % threads;

    *last = gtid == (int64_t)(threads - 1);
    *lower = gtid * count_per_thread + std::min(remain, (int64_t)gtid);
    *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (int64_t)gtid + 1) - 1, *upper);
}

void __kmpc_for_static_init_8u(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, uint64_t* lower, uint64_t* upper, int64_t* /*stride*/, int64_t /*incr*/, int64_t /*chunk*/)
{
    // NCNN_LOGE("__kmpc_for_static_init_8u");
    int num_threads = omp_get_num_threads();

    // TODO only support i++
    uint64_t count = *upper - *lower + 1;
    uint64_t threads = std::min(count, (uint64_t)num_threads);
    uint64_t count_per_thread = count / threads;
    uint64_t remain = count % threads;

    *last = gtid == (int64_t)(threads - 1);
    *lower = gtid * count_per_thread + std::min(remain, (uint64_t)gtid);
    *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (uint64_t)gtid + 1) - 1, *upper);
}

void __kmpc_for_static_fini(void* /*loc*/, int32_t gtid)
{
    // NCNN_LOGE("__kmpc_for_static_fini");
    (void)gtid;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEOMP
