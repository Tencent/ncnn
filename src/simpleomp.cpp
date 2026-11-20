// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "platform.h"

#if NCNN_SIMPLEOMP

#include "simpleomp.h"
#include "cpu.h" // ncnn::get_cpu_count()

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#if !defined(_WIN32)
#include <sched.h>
#endif


static int kmp_sched_prefix(const char* s, const char* p)
{
    while (*p)
    {
        char a = *s;
        char b = *p;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b)
            return 0;
        s++;
        p++;
    }
    return 1;
}

static const char* kmp_sched_skip(const char* s)
{
    while (*s == ' ' || *s == '\t')
        s++;
    if (kmp_sched_prefix(s, "monotonic:"))
        s += 10;
    else if (kmp_sched_prefix(s, "nonmonotonic:"))
        s += 13;
    return s;
}

// clang and clang-cl emit the llvm openmp abi
#if __clang__
extern "C" typedef void (*kmpc_micro)(int32_t* gtid, int32_t* tid, ...);
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
extern "C" typedef void (*kmpc_micro_16)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_17)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_18)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_19)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_20)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_21)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_22)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_23)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_24)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_25)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_26)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_27)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_28)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_29)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_30)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
extern "C" typedef void (*kmpc_micro_31)(int32_t* gtid, int32_t* tid, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
#endif // __clang__

#ifdef __cplusplus
extern "C" {
#endif

static void* kmp_threadfunc(void* args);

#ifdef __cplusplus
} // extern "C"
#endif

namespace ncnn {

class KMPTask
{
public:
    // per-team
#if __clang__
    // libomp abi
    kmpc_micro fn;
    int argc;
    void** argv;
#else
    // libgomp abi
    void (*fn)(void*);
    void* data;
#endif
    int num_threads;

    // per-task
    int thread_num;

    // finish status
    int* num_threads_to_wait;
    Mutex* finish_lock;
    ConditionVariable* finish_condition;

    // per-team shared state
    void* team;
};


class GompTeam
{
public:
    explicit GompTeam(int n)
        : num_threads(n), barrier_count(0), barrier_gen(0), loop_lb(0), loop_ub(0), loop_incr(1), loop_chunk(1), loop_next(0), loop_sched(0), loop_active(0), loop_done(0), loop_generation(0), loop_ull_lb(0), loop_ull_ub(0), loop_ull_incr(1), loop_ull_next(0), loop_ull_chunk(1), loop_ull_sched(0), loop_ull_up(1), sections_count(0), sections_next(0), sections_done(0), sections_active(0), single_copy_data(0), single_copy_active(0), ordered_turn(0)
    {
    }

    int num_threads;

    Mutex lock;
    int barrier_count;
    int barrier_gen;

    long long loop_lb;
    long long loop_ub;
    long long loop_incr;
    long long loop_chunk;
    long long loop_next;
    int loop_sched;
    int loop_active;
    int loop_done;
    int loop_generation;

    unsigned long long loop_ull_lb;
    unsigned long long loop_ull_ub;
    unsigned long long loop_ull_incr;
    unsigned long long loop_ull_next;
    long long loop_ull_chunk;
    int loop_ull_sched;
    int loop_ull_up;

    int sections_count;
    int sections_next;
    int sections_done;
    int sections_active;

    void* single_copy_data;
    int single_copy_active;

    long long ordered_turn;
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


class KMPTeam
{
public:
    explicit KMPTeam(int _num_threads)
        : num_threads(_num_threads), barrier_count(0), barrier_gen(0), ordered_turn(0),
          saved_team(0), saved_num_threads(0), saved_thread_num(0),
          saved_dispatch_lower(0), saved_ordered_count(0), saved_cur_iter(0), saved_in_dispatch(0), saved_dispatch_iter(0), saved_dispatch_ordered_done(0), saved_requested_num_threads(0),
          serialized(0), copyprivate_data(0)
    {
        int n = num_threads > 0 ? num_threads : 1;
        reduce_slots = (void**)malloc(sizeof(void*) * n);
        if (reduce_slots)
            memset(reduce_slots, 0, sizeof(void*) * n);
        for (int i = 0; i < MAX_DISPATCH_STATES; i++)
            dispatch[i].loc = 0;
    }

    ~KMPTeam()
    {
        free(reduce_slots);
    }

public:
    int num_threads;

    // barrier
    ncnn::Mutex barrier_lock;
    int barrier_count;
    int barrier_gen;

    // ordered
    ncnn::Mutex ordered_lock;
    long long ordered_turn;

    // dynamic / guided / runtime dispatch
    enum { MAX_DISPATCH_STATES = 8 };
    struct DispatchState
    {
        const void* loc;
        int sched;
        long long lb;
        long long ub;
        long long incr;
        long long chunk;
        long long next;
        ncnn::Mutex lock;
    };
    DispatchState dispatch[MAX_DISPATCH_STATES];
    ncnn::Mutex dispatch_slot_lock;

    // reduction
    void** reduce_slots;

    // serialized parallel save / restore
    void* saved_team;
    void* saved_num_threads;
    void* saved_thread_num;
    void* saved_dispatch_lower;
    void* saved_ordered_count;
    void* saved_cur_iter;
    void* saved_in_dispatch;
    void* saved_dispatch_iter;
    void* saved_dispatch_ordered_done;
    void* saved_requested_num_threads;
    int serialized;

    // copyprivate
    void* copyprivate_data;
};

class KMPGlobal
{
public:
    KMPGlobal()
    {
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
        // portable one-time initialization (pthread_once is not available on
        // all supported platforms, e.g. windows)
        static ncnn::Mutex init_lock;
        static int init_done = 0;

        init_lock.lock();
        if (!init_done)
        {
            init();
            init_done = 1;
        }
        init_lock.unlock();
    }

public:

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
#if __clang__
                tasks[i].fn = 0;
                tasks[i].argc = 0;
                tasks[i].argv = (void**)0;
#else
                tasks[i].fn = 0;
                tasks[i].data = 0;
#endif
                tasks[i].num_threads = kmp_max_threads;
                tasks[i].thread_num = i + 1;
                tasks[i].num_threads_to_wait = 0;
                tasks[i].finish_lock = 0;
                tasks[i].finish_condition = 0;
                tasks[i].team = 0;
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
static ncnn::ThreadLocalStorage tls_gomp_team;
static ncnn::ThreadLocalStorage tls_gomp_loop_seen;
static ncnn::ThreadLocalStorage tls_serial_sections_count;
static ncnn::ThreadLocalStorage tls_serial_sections_next;
#if __clang__
static ncnn::ThreadLocalStorage tls_requested_num_threads;
static ncnn::ThreadLocalStorage tls_team;
static ncnn::ThreadLocalStorage tls_serial_dispatch_team;
static ncnn::ThreadLocalStorage tls_dispatch_lower;
static ncnn::ThreadLocalStorage tls_ordered_count;
static ncnn::ThreadLocalStorage tls_cur_iter;
static ncnn::ThreadLocalStorage tls_uid;
static ncnn::ThreadLocalStorage tls_in_dispatch;
static ncnn::ThreadLocalStorage tls_dispatch_static_first;
static ncnn::ThreadLocalStorage tls_dispatch_iter;
static ncnn::ThreadLocalStorage tls_dispatch_ordered;
static ncnn::ThreadLocalStorage tls_dispatch_ordered_done;
static ncnn::ThreadLocalStorage tls_dispatch_static_cursor;
#endif

static ncnn::ThreadLocalStorage tls_parallel_depth;
static ncnn::ThreadLocalStorage tls_active_depth;
static ncnn::ThreadLocalStorage tls_num_threads_set;

static void kmp_active_enter()
{
    size_t depth = (size_t)tls_active_depth.get();
    tls_active_depth.set(reinterpret_cast<void*>(depth + 1));
}

static void kmp_active_exit()
{
    size_t depth = (size_t)tls_active_depth.get();
    tls_active_depth.set(reinterpret_cast<void*>(depth > 0 ? depth - 1 : 0));
}

static int kmp_default_num_threads()
{
    if (tls_num_threads_set.get())
    {
        int n = (int)(size_t)tls_num_threads.get();
        return n > 0 ? n : 1;
    }
    return ncnn::get_cpu_count();
}

static void kmp_parallel_enter()
{
    size_t depth = (size_t)tls_parallel_depth.get();
    tls_parallel_depth.set(reinterpret_cast<void*>(depth + 1));
}

static void kmp_parallel_exit()
{
    size_t depth = (size_t)tls_parallel_depth.get();
    tls_parallel_depth.set(reinterpret_cast<void*>(depth > 0 ? depth - 1 : 0));
}

static int kmp_parallel_level()
{
    return (int)(size_t)tls_parallel_depth.get();
}


#if __clang__
static ncnn::KMPTeam* kmp_current_team()
{
    return (ncnn::KMPTeam*)tls_team.get();
}

static ncnn::KMPTeam* kmp_dispatch_team()
{
    ncnn::KMPTeam* team = kmp_current_team();
    if (team)
        return team;

    // orphaned worksharing construct: run it in a per-thread serial team
    team = (ncnn::KMPTeam*)tls_serial_dispatch_team.get();
    if (!team)
    {
        team = new ncnn::KMPTeam(1);
        tls_serial_dispatch_team.set(team);
    }
    return team;
}

static void kmp_yield()
{
#if defined(_WIN32)
    Sleep(0);
#else
    sched_yield();
#endif
}

static int kmp_unique_thread_id()
{
    void* v = tls_uid.get();
    if (v)
        return (int)(size_t)v;

    static ncnn::Mutex g_uid_lock;
    static int g_uid = 0;
    g_uid_lock.lock();
    int id = ++g_uid;
    g_uid_lock.unlock();

    tls_uid.set(reinterpret_cast<void*>((size_t)id));
    return id;
}

static void kmp_barrier()
{
    ncnn::KMPTeam* team = kmp_current_team();
    if (!team || team->num_threads <= 1)
        return;

    ncnn::Mutex& lock = team->barrier_lock;
    lock.lock();

    int gen = team->barrier_gen;
    team->barrier_count++;
    if (team->barrier_count == team->num_threads)
    {
        team->barrier_count = 0;
        team->barrier_gen++;
        lock.unlock();
        return;
    }

    lock.unlock();

    // portable wait: no condition variable, so no lost wakeup on any platform
    for (;;)
    {
        lock.lock();
        int cur = team->barrier_gen;
        lock.unlock();
        if (cur != gen)
            break;
        kmp_yield();
    }
}

struct KMPCriticalLock
{
    ncnn::Mutex lock;
    int owner;
    int depth;
};

static KMPCriticalLock g_kmp_critical_locks[64];

static KMPCriticalLock* kmp_critical_lock(void* crit)
{
    size_t h = (size_t)crit;
    h = (h >> 4) ^ (h >> 13) ^ (h >> 21);
    return &g_kmp_critical_locks[h % 64];
}

static ncnn::KMPTeam::DispatchState* kmp_dispatch_state(void* loc)
{
    ncnn::KMPTeam* team = kmp_dispatch_team();
    if (!team)
        return 0;

    ncnn::KMPTeam::DispatchState* found = 0;

    team->dispatch_slot_lock.lock();
    for (int i = 0; i < ncnn::KMPTeam::MAX_DISPATCH_STATES; i++)
    {
        if (team->dispatch[i].loc == loc)
        {
            found = &team->dispatch[i];
            break;
        }
    }
    if (!found)
    {
        for (int i = 0; i < ncnn::KMPTeam::MAX_DISPATCH_STATES; i++)
        {
            if (team->dispatch[i].loc == 0)
            {
                found = &team->dispatch[i];
                found->loc = loc;
                break;
            }
        }
    }
    if (!found)
        found = &team->dispatch[0];
    team->dispatch_slot_lock.unlock();

    return found;
}

static void kmp_dispatch_init_common(void* loc, int32_t gtid, int32_t sched, long long lb, long long ub, long long incr, long long chunk)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_dispatch_team();
    if (!team)
        return;

    ncnn::KMPTeam::DispatchState* st = kmp_dispatch_state(loc);

    // all team threads must arrive here before the shared counter is reset,
    // otherwise a fast thread may finish the construct and reset it while
    // other threads are still starting up
    kmp_barrier();

    if ((int)(size_t)tls_thread_num.get() == 0)
    {
        st->sched = sched;
        st->lb = lb;
        st->ub = ub;
        st->incr = incr;
        st->chunk = chunk;
        st->next = lb;
        team->ordered_turn = 0;
    }

    tls_dispatch_static_first.set(reinterpret_cast<void*>((size_t)1));
    tls_dispatch_static_cursor.set(tls_thread_num.get());
    tls_dispatch_ordered.set(reinterpret_cast<void*>((size_t)((sched & 0xff) >= 64 ? 1 : 0)));

    kmp_barrier();
}

static long long kmp_runtime_schedule_chunk(long long* out_sched)
{
    static int inited = 0;
    static long long sched = 3; // dynamic
    static long long chunk = 1;
    if (!inited)
    {
        const char* env = getenv("OMP_SCHEDULE");
        if (env)
        {
            const char* kind = kmp_sched_skip(env);
            if (kmp_sched_prefix(kind, "static")) sched = 1;
            else if (kmp_sched_prefix(kind, "dynamic")) sched = 3;
            else if (kmp_sched_prefix(kind, "guided")) sched = 4;
            const char* comma = strchr(kind, ',');
            if (comma)
            {
                chunk = atoll(comma + 1);
                if (chunk < 1) chunk = 1;
            }
            else
            {
                chunk = sched == 1 ? 0 : 1;
            }
        }
        inited = 1;
    }
    *out_sched = sched;
    return chunk;
}

static int32_t kmp_dispatch_static_runtime(ncnn::KMPTeam* team, ncnn::KMPTeam::DispatchState* st, long long* last, long long* lower, long long* upper, long long* stride, long long chunk)
{
    long long lb = st->lb;
    long long ub = st->ub;
    long long incr = st->incr != 0 ? st->incr : 1;
    unsigned long long trip = 0;
    if (incr > 0)
    {
        if (ub >= lb)
            trip = ((unsigned long long)ub - (unsigned long long)lb) / (unsigned long long)incr + 1;
    }
    else
    {
        if (lb >= ub)
            trip = ((unsigned long long)lb - (unsigned long long)ub) / (unsigned long long)(-incr) + 1;
    }

    int nt = team->num_threads > 0 ? team->num_threads : 1;
    int tid = (int)(size_t)tls_thread_num.get();
    int first = (int)(size_t)tls_dispatch_static_first.get();
    unsigned long long idx = 0;
    unsigned long long cnt = 0;

    if (chunk == 0)
    {
        if (!first)
            return 0;
        unsigned long long per = trip / (unsigned long long)nt;
        unsigned long long rem = trip % (unsigned long long)nt;
        idx = (unsigned long long)tid * per + ((unsigned long long)tid < rem ? (unsigned long long)tid : rem);
        cnt = per + ((unsigned long long)tid < rem ? 1 : 0);
    }
    else
    {
        size_t cursor = first ? (size_t)tid : (size_t)tls_dispatch_static_cursor.get();
        if (!first)
            cursor += (size_t)nt;
        tls_dispatch_static_cursor.set(reinterpret_cast<void*>(cursor));
        idx = (unsigned long long)cursor * (unsigned long long)chunk;
        if (idx >= trip)
            return 0;
        cnt = (unsigned long long)chunk;
        if (cnt > trip - idx)
            cnt = trip - idx;
    }

    tls_dispatch_static_first.set(0);
    if (cnt == 0)
        return 0;

    long long lo = lb + (long long)idx * incr;
    long long hi = lo + (long long)(cnt - 1) * incr;
    *lower = lo;
    *upper = hi;
    *stride = incr;
    *last = (long long)(idx + cnt) >= (long long)trip ? 1 : 0;
    tls_dispatch_lower.set(reinterpret_cast<void*>((size_t)idx));
    tls_dispatch_iter.set(reinterpret_cast<void*>((size_t)idx));
    tls_dispatch_ordered_done.set(0);
    tls_ordered_count.set(0);
    tls_in_dispatch.set(reinterpret_cast<void*>((size_t)1));
    return 1;
}

static int32_t kmp_dispatch_next_common(void* loc, int32_t gtid, long long* last, long long* lower, long long* upper, long long* stride)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_dispatch_team();
    if (!team)
        return 0;

    ncnn::KMPTeam::DispatchState* st = kmp_dispatch_state(loc);

    st->lock.lock();
    if (st->incr > 0 ? (st->next > st->ub) : (st->next < st->ub))
    {
        st->lock.unlock();
        return 0;
    }

    long long lo = st->next;
    long long incr = st->incr != 0 ? st->incr : 1;
    long long remaining = incr > 0 ? (st->ub - lo) / incr + 1 : (lo - st->ub) / (-incr) + 1;
    long long csize = st->chunk > 0 ? st->chunk : 1;

    int sched_low = st->sched & 0xff;
    if (sched_low == 36 || sched_low == 68)
    {
        long long g = remaining / (team->num_threads > 0 ? team->num_threads : 1);
        if (g > csize) csize = g;
    }
    else if (sched_low == 37 || sched_low == 69)
    {
        long long rt_sched = 0;
        long long rt_chunk = kmp_runtime_schedule_chunk(&rt_sched);
        if (rt_sched == 1)
        {
            int32_t ret = kmp_dispatch_static_runtime(team, st, last, lower, upper, stride, rt_chunk);
            st->lock.unlock();
            return ret;
        }
        csize = rt_chunk;
        if (rt_sched == 4)
        {
            long long g = remaining / (team->num_threads > 0 ? team->num_threads : 1);
            if (g > csize) csize = g;
        }
    }

    if (csize > remaining)
        csize = remaining;

    long long hi = lo + (csize - 1) * incr;
    st->next = hi + incr;
    *lower = lo;
    *upper = hi;
    *stride = incr;
    *last = incr > 0 ? (st->next > st->ub) : (st->next < st->ub);
    st->lock.unlock();

    long long chunk_index = incr > 0 ? (lo - st->lb) / incr : (st->lb - lo) / (-incr);
    tls_dispatch_lower.set(reinterpret_cast<void*>((size_t)chunk_index));
    tls_dispatch_iter.set(reinterpret_cast<void*>((size_t)chunk_index));
    tls_dispatch_ordered_done.set(0);
    tls_ordered_count.set(0);
    tls_in_dispatch.set(reinterpret_cast<void*>((size_t)1));

    return 1;
}


static void kmp_dispatch_deinit_common(void* loc, int32_t gtid)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_dispatch_team();
    if (!team)
        return;

    // like libomp, deinit is a no-op: the shared dispatch state is reset by
    // the next construct's init (which is team-barriered)
    (void)loc;
    (void)team;
}

static int32_t kmp_reduce_common(void* reduce_data, void (*reduce_func)(void*, void*))
{
    (void)reduce_data;
    (void)reduce_func;
    // Always ask codegen to use its self-contained atomic merge path.
    // For non-atomic types it is wrapped in __kmpc_critical by the compiler,
    // for scalar types it is a plain atomicrmw; both need no runtime work.
    return 2;
}
#endif // __clang__

template<typename T, typename ST>
static void kmp_for_static_init_t(int32_t gtid, int32_t sched, int32_t* last, T* lower, T* upper, ST* stride, T incr_in, T chunk_in)
{
    (void)sched;
    long long incr = (long long)incr_in;
    long long chunk = (long long)chunk_in;
    if (incr == 0)
        incr = 1;

    long long lb = (long long)*lower;
    long long ub = (long long)*upper;
    int nt = omp_get_num_threads();
    if (nt < 1)
        nt = 1;

    unsigned long long count_ull = 0;
    if (incr > 0)
    {
        if (ub >= lb)
        {
            unsigned long long diff = (unsigned long long)ub - (unsigned long long)lb;
            count_ull = diff / (unsigned long long)incr + 1;
        }
    }
    else
    {
        if (lb >= ub)
        {
            unsigned long long diff = (unsigned long long)lb - (unsigned long long)ub;
            count_ull = diff / (unsigned long long)(-incr) + 1;
        }
    }
    long long count = count_ull > (unsigned long long)0x7fffffffffffffffLL ? 0x7fffffffffffffffLL : (long long)count_ull;

    long long lo_v, hi_v, st_v;
    int last_v = 0;

    if (count == 0)
    {
        lo_v = ub + incr;
        hi_v = ub;
        st_v = incr;
        last_v = (gtid == 0) ? 1 : 0;
    }
    else if (chunk > 1 || (sched & 0xff) == 33)
    {
        st_v = (long long)nt * chunk * incr;
        long long start = (long long)gtid * chunk;
        if (start >= count)
        {
            lo_v = ub + incr;
            hi_v = ub;
            last_v = 0;
        }
        else
        {
            long long end = start + chunk - 1;
            if (end > count - 1)
                end = count - 1;
            lo_v = lb + start * incr;
            hi_v = lb + end * incr;
            long long last_chunk = (count - 1) / chunk;
            long long owner = last_chunk % (nt > 0 ? nt : 1);
            last_v = ((long long)gtid == owner) ? 1 : 0;
        }
    }
    else
    {
        long long threads = count < (long long)nt ? count : (long long)nt;
        if ((long long)gtid >= threads)
        {
            lo_v = ub + incr;
            hi_v = ub;
            st_v = incr;
            last_v = 0;
        }
        else
        {
            long long per = count / threads;
            long long rem = count % threads;
            long long first = (long long)gtid * per + ((long long)gtid < rem ? (long long)gtid : rem);
            long long my = per + ((long long)gtid < rem ? 1 : 0);
            lo_v = lb + first * incr;
            hi_v = lb + (first + my - 1) * incr;
            st_v = incr;
            last_v = ((long long)gtid == threads - 1) ? 1 : 0;
        }
    }

    *lower = (T)lo_v;
    *upper = (T)hi_v;
    *stride = (ST)st_v;
    *last = last_v;
}


#ifdef __cplusplus
extern "C" {
#endif

int omp_get_max_threads()
{
    return ncnn::get_cpu_count();
}

int omp_get_num_procs()
{
    return ncnn::get_cpu_count();
}

int omp_in_parallel()
{
    return (int)(size_t)tls_active_depth.get() > 0 ? 1 : 0;
}

int omp_get_level()
{
    return kmp_parallel_level();
}

int omp_get_active_level()
{
    return (int)(size_t)tls_active_depth.get();
}

int omp_get_thread_limit()
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
    tls_num_threads_set.set(reinterpret_cast<void*>((size_t)1));
}

int omp_get_num_threads()
{
    return std::max((int)reinterpret_cast<size_t>(tls_num_threads.get()), 1);
}

int omp_get_thread_num()
{
    return (int)reinterpret_cast<size_t>(tls_thread_num.get());
}

#if __clang__
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
    // fprintf(stderr, "__kmp_invoke_microtask %d %d %d\n", gtid, tid, argc);

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
    case 16:
        (*(kmpc_micro_16)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]);
        break;
    case 17:
        (*(kmpc_micro_17)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16]);
        break;
    case 18:
        (*(kmpc_micro_18)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17]);
        break;
    case 19:
        (*(kmpc_micro_19)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18]);
        break;
    case 20:
        (*(kmpc_micro_20)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19]);
        break;
    case 21:
        (*(kmpc_micro_21)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20]);
        break;
    case 22:
        (*(kmpc_micro_22)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21]);
        break;
    case 23:
        (*(kmpc_micro_23)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22]);
        break;
    case 24:
        (*(kmpc_micro_24)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23]);
        break;
    case 25:
        (*(kmpc_micro_25)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24]);
        break;
    case 26:
        (*(kmpc_micro_26)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25]);
        break;
    case 27:
        (*(kmpc_micro_27)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25], argv[26]);
        break;
    case 28:
        (*(kmpc_micro_28)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25], argv[26], argv[27]);
        break;
    case 29:
        (*(kmpc_micro_29)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25], argv[26], argv[27], argv[28]);
        break;
    case 30:
        (*(kmpc_micro_30)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25], argv[26], argv[27], argv[28], argv[29]);
        break;
    case 31:
        (*(kmpc_micro_31)fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22], argv[23], argv[24], argv[25], argv[26], argv[27], argv[28], argv[29], argv[30]);
        break;
    default:
        // assert never reach here
        break;
    }

    return 0;
}
#endif // __clang__

static void* kmp_threadfunc(void* args)
{
#if __clang__
    int tid = *(int*)args;
#else
    (void)args;
#endif

    for (;;)
    {
        ncnn::KMPTask* task;
        g_kmp_global.kmp_task_queue->get(task);

        // fprintf(stderr, "get %d\n", tid);

        if (!task->fn)
            break;

        tls_num_threads.set(reinterpret_cast<void*>((size_t)task->num_threads));
        tls_thread_num.set(reinterpret_cast<void*>((size_t)task->thread_num));
#if __clang__
        tls_team.set(task->team);
#else
        tls_gomp_team.set(task->team);
        tls_gomp_loop_seen.set(0);
#endif

        kmp_parallel_enter();
        kmp_active_enter();
#if __clang__
        kmp_invoke_microtask(task->fn, task->thread_num, tid, task->argc, task->argv);
#else
        task->fn(task->data);
#endif
        kmp_active_exit();
        kmp_parallel_exit();
#if __clang__
        tls_team.set(0);
#else
        tls_gomp_team.set(0);
#endif

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

#if __clang__
int32_t __kmpc_global_thread_num(void* /*loc*/)
{
    // NCNN_LOGE("__kmpc_global_thread_num");
    return 0;
}

void __kmpc_push_num_threads(void* /*loc*/, int32_t /*gtid*/, int32_t num_threads)
{
    // NCNN_LOGE("__kmpc_push_num_threads %d", num_threads);
    tls_requested_num_threads.set(reinterpret_cast<void*>((size_t)std::max(num_threads, 1)));
}

void __kmpc_fork_call(void* /*loc*/, int32_t argc, kmpc_micro fn, ...)
{
    g_kmp_global.try_init();

    // NCNN_LOGE("__kmpc_fork_call %d", argc);
    void* old_num_threads = tls_num_threads.get();
    void* old_thread_num = tls_thread_num.get();
    void* old_team = tls_team.get();
    int num_threads = (int)reinterpret_cast<size_t>(tls_requested_num_threads.get());
    tls_requested_num_threads.set(0);
    if (num_threads == 0)
        num_threads = kmp_default_num_threads();
    if (num_threads > g_kmp_global.kmp_max_threads)
        num_threads = g_kmp_global.kmp_max_threads;
    if (num_threads < 1)
        num_threads = 1;

    // build argv
    void* argv[32];
    {
        va_list ap;
        va_start(ap, fn);
        for (int i = 0; i < argc; i++)
            argv[i] = va_arg(ap, void*);
        va_end(ap);
    }

    ncnn::KMPTeam team(num_threads);
    tls_team.set(&team);

    if (num_threads == 1)
    {
        tls_num_threads.set(reinterpret_cast<void*>((size_t)1));
        tls_thread_num.set(reinterpret_cast<void*>((size_t)0));

        kmp_parallel_enter();
        if (num_threads > 1)
            kmp_active_enter();
        kmp_invoke_microtask(fn, 0, 0, argc, argv);
        if (num_threads > 1)
            kmp_active_exit();
        kmp_parallel_exit();

        tls_num_threads.set(old_num_threads);
        tls_thread_num.set(old_thread_num);
        tls_team.set(old_team);
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
        tasks[i].team = &team;
    }

    // dispatch 1 ~ num_threads
    g_kmp_global.kmp_task_queue->dispatch(tasks, num_threads - 1);

    // dispatch 0
    {
        tls_num_threads.set(reinterpret_cast<void*>((size_t)num_threads));
        tls_thread_num.set(reinterpret_cast<void*>((size_t)0));
        tls_team.set(&team);

        kmp_parallel_enter();
        if (num_threads > 1)
            kmp_active_enter();
        kmp_invoke_microtask(fn, 0, 0, argc, argv);
        if (num_threads > 1)
            kmp_active_exit();
        kmp_parallel_exit();
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

    tls_num_threads.set(old_num_threads);
    tls_thread_num.set(old_thread_num);
    tls_team.set(old_team);
}

void __kmpc_for_static_init_4(void* /*loc*/, int32_t gtid, int32_t sched, int32_t* last, int32_t* lower, int32_t* upper, int32_t* stride, int32_t incr, int32_t chunk)
{
    kmp_for_static_init_t<int32_t, int32_t>(gtid, sched, last, lower, upper, stride, incr, chunk);
}

void __kmpc_for_static_init_4u(void* /*loc*/, int32_t gtid, int32_t sched, int32_t* last, uint32_t* lower, uint32_t* upper, int32_t* stride, int32_t incr, int32_t chunk)
{
    kmp_for_static_init_t<uint32_t, int32_t>(gtid, sched, last, lower, upper, stride, (uint32_t)incr, (uint32_t)chunk);
}

void __kmpc_for_static_init_8(void* /*loc*/, int32_t gtid, int32_t sched, int32_t* last, int64_t* lower, int64_t* upper, int64_t* stride, int64_t incr, int64_t chunk)
{
    kmp_for_static_init_t<int64_t, int64_t>(gtid, sched, last, lower, upper, stride, incr, chunk);
}

void __kmpc_for_static_init_8u(void* /*loc*/, int32_t gtid, int32_t sched, int32_t* last, uint64_t* lower, uint64_t* upper, int64_t* stride, int64_t incr, int64_t chunk)
{
    kmp_for_static_init_t<uint64_t, int64_t>(gtid, sched, last, lower, upper, stride, (uint64_t)incr, (uint64_t)chunk);
}

void __kmpc_for_static_fini(void* /*loc*/, int32_t gtid)
{
    // NCNN_LOGE("__kmpc_for_static_fini");
    (void)gtid;
}

void __kmpc_barrier(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    kmp_barrier();
}

void __kmpc_critical(void* /*loc*/, int32_t gtid, void* crit)
{
    (void)gtid;

    KMPCriticalLock* l = kmp_critical_lock(crit);
    int me = kmp_unique_thread_id();

    for (;;)
    {
        l->lock.lock();
        if (l->owner == 0)
        {
            l->owner = me;
            l->depth = 1;
            l->lock.unlock();
            return;
        }
        if (l->owner == me)
        {
            l->depth++;
            l->lock.unlock();
            return;
        }
        l->lock.unlock();
        kmp_yield();
    }
}

void __kmpc_critical_with_hint(void* loc, int32_t gtid, void* crit, int32_t /*hint*/)
{
    __kmpc_critical(loc, gtid, crit);
}

void __kmpc_end_critical(void* /*loc*/, int32_t gtid, void* crit)
{
    (void)gtid;

    KMPCriticalLock* l = kmp_critical_lock(crit);
    int me = kmp_unique_thread_id();

    l->lock.lock();
    if (l->owner == me)
    {
        l->depth--;
        if (l->depth <= 0)
            l->owner = 0;
    }
    l->lock.unlock();
}

int32_t __kmpc_master(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    return ((int)(size_t)tls_thread_num.get() == 0) ? 1 : 0;
}

void __kmpc_end_master(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
}

int32_t __kmpc_single(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    if (!kmp_current_team())
        return 1;
    return ((int)(size_t)tls_thread_num.get() == 0) ? 1 : 0;
}

void __kmpc_end_single(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
}

void __kmpc_copyprivate(void* /*loc*/, int32_t gtid, size_t /*cpy_size*/, void* cpy_data, void (*cpy_func)(void*, void*), int32_t didit)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_current_team();
    if (!team)
        return;

    if (didit)
        team->copyprivate_data = cpy_data;

    kmp_barrier();

    if (!didit)
        (*cpy_func)(cpy_data, team->copyprivate_data);

    kmp_barrier();
}

void __kmpc_ordered(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_current_team();
    if (!team || team->num_threads <= 1)
        return;

    void* lp = tls_dispatch_lower.get();
    if (!tls_in_dispatch.get())
    {
        // not inside a dispatch loop, fall back to serializing ordered regions
        team->ordered_lock.lock();
        tls_cur_iter.set(reinterpret_cast<void*>((size_t)-1));
        return;
    }

    long long iter = (long long)(size_t)tls_dispatch_iter.get();
    (void)lp;

    for (;;)
    {
        team->ordered_lock.lock();
        if ((long long)team->ordered_turn == iter)
        {
            tls_cur_iter.set(reinterpret_cast<void*>((size_t)iter));
            tls_dispatch_ordered_done.set(0);
            team->ordered_lock.unlock();
            return;
        }
        team->ordered_lock.unlock();
        kmp_yield();
    }
}

void __kmpc_end_ordered(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    ncnn::KMPTeam* team = kmp_current_team();
    if (!team || team->num_threads <= 1)
        return;

    if (!tls_in_dispatch.get())
    {
        tls_cur_iter.set(0);
        team->ordered_lock.unlock();
        return;
    }

    long long iter = (long long)(size_t)tls_cur_iter.get();

    team->ordered_lock.lock();
    team->ordered_turn = iter + 1;
    team->ordered_lock.unlock();

    tls_dispatch_ordered_done.set(reinterpret_cast<void*>((size_t)1));
}

void __kmpc_flush(void* /*loc*/)
{
#if defined(_WIN32)
    MemoryBarrier();
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

void __kmpc_serialized_parallel(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    ncnn::KMPTeam* team = new ncnn::KMPTeam(1);
    team->serialized = 1;
    team->saved_team = tls_team.get();
    team->saved_num_threads = tls_num_threads.get();
    team->saved_thread_num = tls_thread_num.get();
    team->saved_dispatch_lower = tls_dispatch_lower.get();
    team->saved_ordered_count = tls_ordered_count.get();
    team->saved_cur_iter = tls_cur_iter.get();
    team->saved_in_dispatch = tls_in_dispatch.get();
    team->saved_dispatch_iter = tls_dispatch_iter.get();
    team->saved_dispatch_ordered_done = tls_dispatch_ordered_done.get();

    tls_team.set(team);
    tls_num_threads.set(reinterpret_cast<void*>((size_t)1));
    tls_thread_num.set(reinterpret_cast<void*>((size_t)0));
    tls_requested_num_threads.set(0);
    kmp_parallel_enter();
}

void __kmpc_end_serialized_parallel(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    ncnn::KMPTeam* team = (ncnn::KMPTeam*)tls_team.get();
    if (!team || !team->serialized)
        return;

    tls_team.set(team->saved_team);
    tls_num_threads.set(team->saved_num_threads);
    tls_thread_num.set(team->saved_thread_num);
    tls_dispatch_lower.set(team->saved_dispatch_lower);
    tls_ordered_count.set(team->saved_ordered_count);
    tls_cur_iter.set(team->saved_cur_iter);
    tls_in_dispatch.set(team->saved_in_dispatch);
    tls_dispatch_iter.set(team->saved_dispatch_iter);
    tls_dispatch_ordered_done.set(team->saved_dispatch_ordered_done);
    kmp_parallel_exit();
    delete team;
}

int32_t __kmpc_reduce_nowait(void* /*loc*/, int32_t gtid, int32_t /*num_vars*/, size_t /*reduce_size*/, void* reduce_data, void (*reduce_func)(void*, void*), void* /*lck*/)
{
    (void)gtid;
    return kmp_reduce_common(reduce_data, reduce_func);
}

void __kmpc_end_reduce_nowait(void* /*loc*/, int32_t gtid, void* /*lck*/)
{
    (void)gtid;
}

int32_t __kmpc_reduce(void* /*loc*/, int32_t gtid, int32_t /*num_vars*/, size_t /*reduce_size*/, void* reduce_data, void (*reduce_func)(void*, void*), void* /*lck*/)
{
    (void)gtid;
    return kmp_reduce_common(reduce_data, reduce_func);
}

void __kmpc_end_reduce(void* /*loc*/, int32_t gtid, void* /*lck*/)
{
    (void)gtid;
}

void __kmpc_dispatch_init_4(void* loc, int32_t gtid, int32_t sched, int32_t lb, int32_t ub, int32_t st, int32_t chunk)
{
    kmp_dispatch_init_common(loc, gtid, sched, lb, ub, st, chunk);
}

void __kmpc_dispatch_init_4u(void* loc, int32_t gtid, int32_t sched, uint32_t lb, uint32_t ub, int32_t st, int32_t chunk)
{
    kmp_dispatch_init_common(loc, gtid, sched, lb, ub, st, chunk);
}

void __kmpc_dispatch_init_8(void* loc, int32_t gtid, int32_t sched, int64_t lb, int64_t ub, int64_t st, int64_t chunk)
{
    kmp_dispatch_init_common(loc, gtid, sched, lb, ub, st, chunk);
}

void __kmpc_dispatch_init_8u(void* loc, int32_t gtid, int32_t sched, uint64_t lb, uint64_t ub, int64_t st, int64_t chunk)
{
    kmp_dispatch_init_common(loc, gtid, sched, lb, ub, st, chunk);
}

int32_t __kmpc_dispatch_next_4(void* loc, int32_t gtid, int32_t* last, int32_t* lower, int32_t* upper, int32_t* stride)
{
    long long l = 0, lo = 0, up = 0, st = 0;
    int32_t r = kmp_dispatch_next_common(loc, gtid, &l, &lo, &up, &st);
    if (r)
    {
        *last = (int32_t)l;
        *lower = (int32_t)lo;
        *upper = (int32_t)up;
        *stride = (int32_t)st;
    }
    return r;
}

int32_t __kmpc_dispatch_next_4u(void* loc, int32_t gtid, int32_t* last, uint32_t* lower, uint32_t* upper, int32_t* stride)
{
    long long l = 0, lo = 0, up = 0, st = 0;
    int32_t r = kmp_dispatch_next_common(loc, gtid, &l, &lo, &up, &st);
    if (r)
    {
        *last = (int32_t)l;
        *lower = (uint32_t)lo;
        *upper = (uint32_t)up;
        *stride = (int32_t)st;
    }
    return r;
}

int32_t __kmpc_dispatch_next_8(void* loc, int32_t gtid, int32_t* last, int64_t* lower, int64_t* upper, int64_t* stride)
{
    long long l = 0, lo = 0, up = 0, st = 0;
    int32_t r = kmp_dispatch_next_common(loc, gtid, &l, &lo, &up, &st);
    if (r)
    {
        *last = (int32_t)l;
        *lower = (int64_t)lo;
        *upper = (int64_t)up;
        *stride = (int64_t)st;
    }
    return r;
}

int32_t __kmpc_dispatch_next_8u(void* loc, int32_t gtid, int32_t* last, uint64_t* lower, uint64_t* upper, int64_t* stride)
{
    long long l = 0, lo = 0, up = 0, st = 0;
    int32_t r = kmp_dispatch_next_common(loc, gtid, &l, &lo, &up, &st);
    if (r)
    {
        *last = (int32_t)l;
        *lower = (uint64_t)lo;
        *upper = (uint64_t)up;
        *stride = (int64_t)st;
    }
    return r;
}

static void kmp_dispatch_fini_common()
{
    ncnn::KMPTeam* team = kmp_current_team();
    if (!team)
        return;

    size_t iter = (size_t)tls_dispatch_iter.get();
    int ordered_done = (int)(size_t)tls_dispatch_ordered_done.get();
    if (!ordered_done && tls_dispatch_ordered.get() && tls_in_dispatch.get())
    {
        // a skipped ordered iteration must consume its turn in iteration order,
        // otherwise a later chunk can advance the turn past an earlier chunk
        for (;;)
        {
            team->ordered_lock.lock();
            if ((long long)team->ordered_turn == (long long)iter)
            {
                team->ordered_turn = (long long)(iter + 1);
                team->ordered_lock.unlock();
                break;
            }
            team->ordered_lock.unlock();
            kmp_yield();
        }
    }

    tls_dispatch_iter.set(reinterpret_cast<void*>(iter + 1));
    tls_dispatch_ordered_done.set(0);
}

void __kmpc_dispatch_fini_4(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    kmp_dispatch_fini_common();
}

void __kmpc_dispatch_fini_4u(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    kmp_dispatch_fini_common();
}

void __kmpc_dispatch_fini_8(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    kmp_dispatch_fini_common();
}

void __kmpc_dispatch_fini_8u(void* /*loc*/, int32_t gtid)
{
    (void)gtid;
    kmp_dispatch_fini_common();
}

void __kmpc_dispatch_deinit(void* loc, int32_t gtid)
{
    kmp_dispatch_deinit_common(loc, gtid);
}

#else  // __clang__

// ---- gcc libgomp abi helpers ----
struct gomp_loop_info
{
    long lb;
    long ub;
    long incr;
    long chunk;
    int sched;
};

struct gomp_sections_info
{
    unsigned count;
};

static ncnn::ThreadLocalStorage tls_gomp_team_init;
static ncnn::ThreadLocalStorage tls_gomp_loop_info;
static ncnn::ThreadLocalStorage tls_gomp_sections_info;
static ncnn::ThreadLocalStorage tls_gomp_chunk_lower;
static ncnn::ThreadLocalStorage tls_gomp_ordered_count;
static ncnn::ThreadLocalStorage tls_gomp_static_chunk;
static ncnn::ThreadLocalStorage tls_gomp_uid;

static int gomp_unique_id()
{
    void* v = tls_gomp_uid.get();
    if (v)
        return (int)(size_t)v;
    static ncnn::Mutex lock;
    static int next_id = 0;
    lock.lock();
    int id = ++next_id;
    lock.unlock();
    tls_gomp_uid.set(reinterpret_cast<void*>((size_t)id));
    return id;
}

static ncnn::GompTeam* gomp_team()
{
    return (ncnn::GompTeam*)tls_gomp_team.get();
}

static void gomp_yield()
{
#if defined(_WIN32)
    Sleep(0);
#else
    sched_yield();
#endif
}

static void gomp_barrier()
{
    ncnn::GompTeam* team = gomp_team();
    if (!team || team->num_threads <= 1)
        return;

    ncnn::Mutex& lock = team->lock;
    lock.lock();
    int gen = team->barrier_gen;
    team->barrier_count++;
    if (team->barrier_count == team->num_threads)
    {
        team->barrier_count = 0;
        team->barrier_gen++;
        lock.unlock();
        return;
    }
    lock.unlock();
    for (;;)
    {
        lock.lock();
        int cur = team->barrier_gen;
        lock.unlock();
        if (cur != gen)
            break;
        gomp_yield();
    }
}

static long gomp_trip_count(long lb, long ub, long incr)
{
    if (incr == 0)
        return 0;
    if (incr > 0)
    {
        if (ub < lb)
            return 0;
        unsigned long long diff = (unsigned long long)ub - (unsigned long long)lb;
        unsigned long long step = (unsigned long long)incr;
        return (long)(diff / step + (diff % step != 0));
    }
    if (lb < ub)
        return 0;
    unsigned long long diff = (unsigned long long)lb - (unsigned long long)ub;
    unsigned long long step = (unsigned long long)(-incr);
    return (long)(diff / step + (diff % step != 0));
}

static void gomp_set_ordered(long idx)
{
    tls_gomp_chunk_lower.set(reinterpret_cast<void*>((size_t)idx));
    tls_gomp_ordered_count.set(0);
}

static void gomp_resolve_runtime(int* sched, long* chunk)
{
    static int inited = 0;
    static int env_sched = 1;
    static long env_chunk = 1;
    if (!inited)
    {
        const char* env = getenv("OMP_SCHEDULE");
        if (env)
        {
            const char* sched = kmp_sched_skip(env);
            if (kmp_sched_prefix(sched, "static")) env_sched = 0;
            else if (kmp_sched_prefix(sched, "dynamic")) env_sched = 1;
            else if (kmp_sched_prefix(sched, "guided")) env_sched = 2;
            const char* comma = strchr(sched, ',');
            if (comma)
            {
                env_chunk = atol(comma + 1);
                if (env_chunk < 1) env_chunk = 1;
            }
            else
            {
                env_chunk = env_sched == 0 ? 0 : 1;
            }
        }
        inited = 1;
    }
    *sched = env_sched;
    *chunk = env_chunk;
}

static void gomp_init_loop_team(ncnn::GompTeam* team)
{
    gomp_loop_info* info = (gomp_loop_info*)tls_gomp_loop_info.get();
    int sched = info->sched;
    long chunk = info->chunk;
    if (sched == 3)
        gomp_resolve_runtime(&sched, &chunk);
    team->lock.lock();
    team->loop_lb = info->lb;
    team->loop_ub = info->ub;
    team->loop_incr = info->incr;
    team->loop_chunk = chunk;
    team->loop_next = 0;
    team->loop_sched = sched;
    team->loop_active = 1;
    team->loop_done = 0;
    team->loop_generation++;
    team->ordered_turn = 0;
    team->lock.unlock();
}

static int gomp_static_claim(ncnn::GompTeam* team)
{
    void* seen = tls_gomp_loop_seen.get();
    if ((size_t)seen == (size_t)team->loop_generation)
        return 0;
    tls_gomp_loop_seen.set(reinterpret_cast<void*>((size_t)team->loop_generation));
    return 1;
}

static void gomp_init_sections_team(ncnn::GompTeam* team)
{
    gomp_sections_info* info = (gomp_sections_info*)tls_gomp_sections_info.get();
    team->lock.lock();
    team->sections_count = (int)info->count;
    team->sections_next = 0;
    team->sections_done = 0;
    team->sections_active = 1;
    team->lock.unlock();
}

static int gomp_loop_begin(long lb, long ub, long incr, long chunk, int sched)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return 0;

    team->lock.lock();
    if (!team->loop_active)
    {
        int rsched = sched;
        long rchunk = chunk;
        if (sched == 3)
            gomp_resolve_runtime(&rsched, &rchunk);
        team->loop_lb = lb;
        team->loop_ub = ub;
        team->loop_incr = incr;
        team->loop_chunk = rchunk;
        team->loop_next = 0;
        team->loop_sched = rsched;
        team->loop_active = 1;
        team->loop_done = 0;
        team->loop_generation++;
        team->ordered_turn = 0;
    }
    team->lock.unlock();
    return 1;
}

static int gomp_take_static(ncnn::GompTeam* team, int first, long* p_lb, long* p_ub)
{
    int nt = team->num_threads;
    int tid = (int)(size_t)tls_thread_num.get();
    long lb = (long)team->loop_lb;
    long incr = (long)team->loop_incr;
    long chunk = (long)team->loop_chunk;
    long trip = gomp_trip_count(lb, (long)team->loop_ub, incr);

    if (chunk == 0)
    {
        if (!first)
            return 0;
        long per = trip / nt;
        long rem = trip % nt;
        long begin = (long)tid * per + (tid < rem ? tid : rem);
        long cnt = per + (tid < rem ? 1 : 0);
        if (cnt <= 0)
            return 0;
        *p_lb = lb + begin * incr;
        *p_ub = *p_lb + cnt * incr;
        gomp_set_ordered(begin);
        return 1;
    }

    size_t cursor = first ? (size_t)tid : (size_t)tls_gomp_static_chunk.get();
    if (!first)
        cursor += (size_t)nt;
    tls_gomp_static_chunk.set(reinterpret_cast<void*>(cursor));
    long idx = (long)cursor * chunk;
    if (idx >= trip)
        return 0;
    long cnt = chunk;
    if (cnt > trip - idx)
        cnt = trip - idx;
    *p_lb = lb + idx * incr;
    *p_ub = *p_lb + cnt * incr;
    gomp_set_ordered(idx);
    return 1;
}

static int gomp_take_dynamic(ncnn::GompTeam* team, long* p_lb, long* p_ub)
{
    if (team->loop_sched == 0)
        return gomp_take_static(team, gomp_static_claim(team), p_lb, p_ub);

    long lb = (long)team->loop_lb;
    long incr = (long)team->loop_incr;
    long trip = gomp_trip_count(lb, (long)team->loop_ub, incr);
    int sched = team->loop_sched;
    long chunk = (long)team->loop_chunk;

    team->lock.lock();
    long idx = (long)team->loop_next;
    if (idx >= trip)
    {
        team->lock.unlock();
        return 0;
    }
    long remaining = trip - idx;
    long c = chunk > 0 ? chunk : 1;
    if (sched == 2)
    {
        long g = remaining / (team->num_threads > 0 ? team->num_threads : 1);
        if (g > c) c = g;
    }
    if (c > remaining) c = remaining;
    team->loop_next = idx + c;
    team->lock.unlock();

    *p_lb = lb + idx * incr;
    *p_ub = *p_lb + c * incr;
    gomp_set_ordered(idx);
    return 1;
}

static void gomp_loop_end_common(int barrier)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return;

    // retire the loop state before the implicit barrier releases the team,
    // otherwise the last arriver can start the next loop with stale state
    team->lock.lock();
    team->loop_done++;
    if (team->loop_done >= team->num_threads)
    {
        team->loop_active = 0;
        team->loop_done = 0;
    }
    team->lock.unlock();

    if (barrier && team->num_threads > 1)
        gomp_barrier();
}

static unsigned gomp_section_take()
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
    {
        int count = (int)(size_t)tls_serial_sections_count.get();
        int idx = (int)(size_t)tls_serial_sections_next.get();
        if (idx >= count)
            return 0;
        tls_serial_sections_next.set(reinterpret_cast<void*>((size_t)(idx + 1)));
        return (unsigned)(idx + 1);
    }

    team->lock.lock();
    if (!team->sections_active)
    {
        team->lock.unlock();
        return 0;
    }
    int idx = team->sections_next++;
    if (idx >= team->sections_count)
    {
        team->sections_done++;
        if (team->sections_done >= team->num_threads)
        {
            team->sections_active = 0;
            team->sections_next = 0;
            team->sections_done = 0;
        }
        team->lock.unlock();
        return 0;
    }
    team->lock.unlock();
    return (unsigned)(idx + 1);
}

static void gomp_ordered_wait()
{
    ncnn::GompTeam* team = gomp_team();
    if (!team || team->num_threads <= 1)
        return;
    long iter = (long)(size_t)tls_gomp_chunk_lower.get() + (long)(size_t)tls_gomp_ordered_count.get();
    for (;;)
    {
        team->lock.lock();
        if ((long)team->ordered_turn == iter)
        {
            team->lock.unlock();
            return;
        }
        team->lock.unlock();
        gomp_yield();
    }
}

static void gomp_ordered_release()
{
    ncnn::GompTeam* team = gomp_team();
    if (!team || team->num_threads <= 1)
        return;
    long iter = (long)(size_t)tls_gomp_chunk_lower.get() + (long)(size_t)tls_gomp_ordered_count.get();
    team->lock.lock();
    team->ordered_turn = iter + 1;
    team->lock.unlock();
    size_t count = (size_t)tls_gomp_ordered_count.get();
    tls_gomp_ordered_count.set(reinterpret_cast<void*>((size_t)(count + 1)));
}


static ncnn::ThreadLocalStorage tls_parallel_context;

struct parallel_context
{
    int num_threads_to_wait;
    ncnn::Mutex finish_lock;
    ncnn::ConditionVariable finish_condition;
    ncnn::KMPTask* tasks;
    ncnn::GompTeam* team;
    void* old_gomp_team;
    void* old_num_threads;
    void* old_thread_num;
    void* old_loop_seen;
    void* old_static_chunk;
    void* old_ordered_lower;
    void* old_ordered_count;
    struct parallel_context* parent;
    int active;
};

// split ABI: GOMP_parallel_start only prepares the team; the caller runs the
// outlined function and then calls GOMP_parallel_end
static void gomp_start_split(void (*fn)(void*), void* data, unsigned num_threads, ncnn::GompTeam* team)
{
    void* old_gomp_team = tls_gomp_team.get();
    void* old_loop_seen = tls_gomp_loop_seen.get();
    void* old_static_chunk = tls_gomp_static_chunk.get();
    void* old_ordered_lower = tls_gomp_chunk_lower.get();
    void* old_ordered_count = tls_gomp_ordered_count.get();
    tls_gomp_team.set(team);
    tls_gomp_loop_seen.set(0);

    parallel_context* pc = new parallel_context;
    pc->parent = (parallel_context*)tls_parallel_context.get();
    pc->team = team;
    pc->old_gomp_team = old_gomp_team;
    pc->old_num_threads = tls_num_threads.get();
    pc->old_thread_num = tls_thread_num.get();
    pc->old_loop_seen = old_loop_seen;
    pc->old_static_chunk = old_static_chunk;
    pc->old_ordered_lower = old_ordered_lower;
    pc->old_ordered_count = old_ordered_count;
    pc->num_threads_to_wait = (int)num_threads - 1;
    pc->tasks = 0;
    pc->active = num_threads > 1 ? 1 : 0;

    if (num_threads > 1)
    {
        pc->tasks = new ncnn::KMPTask[num_threads - 1];
        for (unsigned i = 0; i < num_threads - 1; i++)
        {
            pc->tasks[i].fn = fn;
            pc->tasks[i].data = data;
            pc->tasks[i].num_threads = num_threads;
            pc->tasks[i].thread_num = i + 1;
            pc->tasks[i].num_threads_to_wait = &pc->num_threads_to_wait;
            pc->tasks[i].finish_lock = &pc->finish_lock;
            pc->tasks[i].finish_condition = &pc->finish_condition;
            pc->tasks[i].team = team;
        }
        g_kmp_global.kmp_task_queue->dispatch(pc->tasks, num_threads - 1);
    }

    tls_parallel_context.set(pc);
    tls_num_threads.set(reinterpret_cast<void*>((size_t)num_threads));
    tls_thread_num.set(reinterpret_cast<void*>((size_t)0));
    kmp_parallel_enter();
    if (pc->active)
        kmp_active_enter();
}

void GOMP_parallel_start(void (*fn)(void*), void* data, unsigned num_threads)
{
    g_kmp_global.try_init();

    if (num_threads == 0)
        num_threads = kmp_default_num_threads();
    if (num_threads > (unsigned)g_kmp_global.kmp_max_threads)
        num_threads = g_kmp_global.kmp_max_threads;
    if (num_threads < 1)
        num_threads = 1;

    ncnn::GompTeam* team = new ncnn::GompTeam((int)num_threads);
    gomp_start_split(fn, data, num_threads, team);
}


void GOMP_parallel_end(void)
{
    parallel_context* pc = (parallel_context*)tls_parallel_context.get();
    tls_parallel_context.set(pc ? pc->parent : 0);

    if (pc)
    {
        pc->finish_lock.lock();
        if (pc->num_threads_to_wait != 0)
            pc->finish_condition.wait(pc->finish_lock);
        pc->finish_lock.unlock();

        delete[] pc->tasks;
        tls_gomp_team.set(pc->old_gomp_team);
        tls_num_threads.set(pc->old_num_threads);
        tls_thread_num.set(pc->old_thread_num);
        tls_gomp_loop_seen.set(pc->old_loop_seen);
        tls_gomp_static_chunk.set(pc->old_static_chunk);
        tls_gomp_chunk_lower.set(pc->old_ordered_lower);
        tls_gomp_ordered_count.set(pc->old_ordered_count);
        if (pc->active)
            kmp_active_exit();
        delete pc->team;
        delete pc;
    }
    else
    {
        tls_gomp_team.set(0);
    }

    kmp_parallel_exit();
}


void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned int /*flags*/)
{
    void* old_num_threads = tls_num_threads.get();
    void* old_thread_num = tls_thread_num.get();
    void* old_gomp_team = tls_gomp_team.get();
    void* old_loop_seen = tls_gomp_loop_seen.get();
    void* old_static_chunk = tls_gomp_static_chunk.get();
    void* old_ordered_lower = tls_gomp_chunk_lower.get();
    void* old_ordered_count = tls_gomp_ordered_count.get();

    g_kmp_global.try_init();

    if (num_threads == 0)
        num_threads = kmp_default_num_threads();
    if (num_threads > (unsigned)g_kmp_global.kmp_max_threads)
        num_threads = g_kmp_global.kmp_max_threads;
    if (num_threads < 1)
        num_threads = 1;

    ncnn::GompTeam team((int)num_threads);
    tls_gomp_team.set(&team);
    tls_gomp_loop_seen.set(0);

    void (*team_init)(ncnn::GompTeam*) = (void (*)(ncnn::GompTeam*))tls_gomp_team_init.get();
    if (team_init)
    {
        tls_gomp_team_init.set(0);
        team_init(&team);
    }

    if (g_kmp_global.kmp_max_threads == 1 || num_threads == 1)
    {
        tls_num_threads.set(reinterpret_cast<void*>((size_t)1));
        tls_thread_num.set(reinterpret_cast<void*>((size_t)0));
        tls_gomp_loop_seen.set(0);
        kmp_parallel_enter();
        fn(data);
        kmp_parallel_exit();

        tls_num_threads.set(old_num_threads);
        tls_thread_num.set(old_thread_num);
        tls_gomp_team.set(old_gomp_team);
        tls_gomp_loop_seen.set(old_loop_seen);
        tls_gomp_static_chunk.set(old_static_chunk);
        tls_gomp_chunk_lower.set(old_ordered_lower);
        tls_gomp_ordered_count.set(old_ordered_count);
        return;
    }

    int num_threads_to_wait = num_threads - 1;
    ncnn::Mutex finish_lock;
    ncnn::ConditionVariable finish_condition;

    ncnn::KMPTask* tasks = (ncnn::KMPTask*)alloca((num_threads - 1) * sizeof(ncnn::KMPTask));
    for (unsigned i = 0; i < num_threads - 1; i++)
    {
        tasks[i].fn = fn;
        tasks[i].data = data;
        tasks[i].num_threads = num_threads;
        tasks[i].thread_num = i + 1;
        tasks[i].num_threads_to_wait = &num_threads_to_wait;
        tasks[i].finish_lock = &finish_lock;
        tasks[i].finish_condition = &finish_condition;
        tasks[i].team = &team;
    }

    g_kmp_global.kmp_task_queue->dispatch(tasks, num_threads - 1);

    tls_num_threads.set(reinterpret_cast<void*>((size_t)num_threads));
    tls_thread_num.set(reinterpret_cast<void*>((size_t)0));
    kmp_parallel_enter();
    kmp_active_enter();
    fn(data);

    finish_lock.lock();
    if (num_threads_to_wait != 0)
        finish_condition.wait(finish_lock);
    finish_lock.unlock();

    kmp_active_exit();
    kmp_parallel_exit();

    tls_num_threads.set(old_num_threads);
    tls_thread_num.set(old_thread_num);
    tls_gomp_team.set(old_gomp_team);
    tls_gomp_loop_seen.set(old_loop_seen);
    tls_gomp_static_chunk.set(old_static_chunk);
    tls_gomp_chunk_lower.set(old_ordered_lower);
    tls_gomp_ordered_count.set(old_ordered_count);
}


static int g_gomp_critical_anonymous;
static ncnn::Mutex g_gomp_atomic_mutex;

int GOMP_loop_static_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    if (!gomp_loop_begin(lb, ub, incr, chunk, 0))
    {
        *p_lb = lb;
        *p_ub = ub;
        return 1;
    }
    ncnn::GompTeam* team = gomp_team();
    return gomp_take_static(team, gomp_static_claim(team), p_lb, p_ub);
}

int GOMP_loop_dynamic_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    if (!gomp_loop_begin(lb, ub, incr, chunk, 1))
    {
        *p_lb = lb;
        *p_ub = ub;
        return 1;
    }
    return gomp_take_dynamic(gomp_team(), p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_dynamic_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_start(lb, ub, incr, chunk, p_lb, p_ub);
}

int GOMP_loop_guided_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    if (!gomp_loop_begin(lb, ub, incr, chunk, 2))
    {
        *p_lb = lb;
        *p_ub = ub;
        return 1;
    }
    return gomp_take_dynamic(gomp_team(), p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_guided_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    return GOMP_loop_guided_start(lb, ub, incr, chunk, p_lb, p_ub);
}

int GOMP_loop_runtime_start(long lb, long ub, long incr, long* p_lb, long* p_ub)
{
    if (!gomp_loop_begin(lb, ub, incr, 1, 3))
    {
        *p_lb = lb;
        *p_ub = ub;
        return 1;
    }
    return gomp_take_dynamic(gomp_team(), p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_runtime_start(long lb, long ub, long incr, long* p_lb, long* p_ub)
{
    return GOMP_loop_runtime_start(lb, ub, incr, p_lb, p_ub);
}

int GOMP_loop_maybe_nonmonotonic_runtime_start(long lb, long ub, long incr, long* p_lb, long* p_ub)
{
    return GOMP_loop_runtime_start(lb, ub, incr, p_lb, p_ub);
}

int GOMP_loop_ordered_static_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    return GOMP_loop_static_start(lb, ub, incr, chunk, p_lb, p_ub);
}

int GOMP_loop_ordered_dynamic_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_start(lb, ub, incr, chunk, p_lb, p_ub);
}

int GOMP_loop_ordered_guided_start(long lb, long ub, long incr, long chunk, long* p_lb, long* p_ub)
{
    return GOMP_loop_guided_start(lb, ub, incr, chunk, p_lb, p_ub);
}

int GOMP_loop_ordered_guided_next(long* p_lb, long* p_ub)
{
    return gomp_take_dynamic(gomp_team(), p_lb, p_ub);
}

int GOMP_loop_ordered_runtime_start(long lb, long ub, long incr, long* p_lb, long* p_ub)
{
    return GOMP_loop_runtime_start(lb, ub, incr, p_lb, p_ub);
}

int GOMP_loop_ordered_runtime_next(long* p_lb, long* p_ub)
{
    return gomp_take_dynamic(gomp_team(), p_lb, p_ub);
}

int GOMP_loop_static_next(long* p_lb, long* p_ub)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return 0;
    return gomp_take_static(team, gomp_static_claim(team), p_lb, p_ub);
}

int GOMP_loop_dynamic_next(long* p_lb, long* p_ub)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return 0;
    return gomp_take_dynamic(team, p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_dynamic_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_guided_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_guided_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_runtime_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_nonmonotonic_runtime_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_maybe_nonmonotonic_runtime_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

int GOMP_loop_ordered_static_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_static_next(p_lb, p_ub);
}

int GOMP_loop_ordered_dynamic_next(long* p_lb, long* p_ub)
{
    return GOMP_loop_dynamic_next(p_lb, p_ub);
}

void GOMP_loop_end(void)
{
    gomp_loop_end_common(1);
}

void GOMP_loop_end_nowait(void)
{
    gomp_loop_end_common(0);
}

void GOMP_barrier(void)
{
    gomp_barrier();
}

int GOMP_single_start(void)
{
    return (int)(size_t)tls_thread_num.get() == 0;
}

void* GOMP_single_copy_start(void)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return 0;

    team->lock.lock();
    if (!team->single_copy_active)
    {
        team->single_copy_active = 1;
        team->single_copy_data = 0;
        team->lock.unlock();
        return 0;
    }
    team->lock.unlock();

    gomp_barrier();
    void* data = team->single_copy_data;
    gomp_barrier();
    return data;
}

void GOMP_single_copy_end(void* data)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return;

    team->lock.lock();
    team->single_copy_data = data;
    team->lock.unlock();

    gomp_barrier();
    gomp_barrier();

    team->lock.lock();
    team->single_copy_active = 0;
    team->single_copy_data = 0;
    team->lock.unlock();
}

struct GompCriticalLock
{
    ncnn::Mutex lock;
    int owner;
    int depth;
};

static GompCriticalLock g_gomp_critical_locks[64];

static GompCriticalLock* gomp_critical_lock(void* p)
{
    size_t h = (size_t)p;
    h = (h >> 4) ^ (h >> 13) ^ (h >> 21);
    return &g_gomp_critical_locks[h % 64];
}

static void gomp_critical_enter(void* p)
{
    GompCriticalLock* l = gomp_critical_lock(p);
    int me = gomp_unique_id();
    for (;;)
    {
        l->lock.lock();
        if (l->owner == 0)
        {
            l->owner = me;
            l->depth = 1;
            l->lock.unlock();
            return;
        }
        if (l->owner == me)
        {
            l->depth++;
            l->lock.unlock();
            return;
        }
        l->lock.unlock();
        gomp_yield();
    }
}

static void gomp_critical_leave(void* p)
{
    GompCriticalLock* l = gomp_critical_lock(p);
    int me = gomp_unique_id();
    l->lock.lock();
    if (l->owner == me)
    {
        l->depth--;
        if (l->depth <= 0)
            l->owner = 0;
    }
    l->lock.unlock();
}

void GOMP_critical_start(void)
{
    gomp_critical_enter(&g_gomp_critical_anonymous);
}

void GOMP_critical_end(void)
{
    gomp_critical_leave(&g_gomp_critical_anonymous);
}

void GOMP_critical_name_start(void** pptr)
{
    gomp_critical_enter(*pptr);
}

void GOMP_critical_name_end(void** pptr)
{
    gomp_critical_leave(*pptr);
}


void GOMP_atomic_start(void)
{
    g_gomp_atomic_mutex.lock();
}

void GOMP_atomic_end(void)
{
    g_gomp_atomic_mutex.unlock();
}

void GOMP_ordered_start(void)
{
    gomp_ordered_wait();
}

void GOMP_ordered_end(void)
{
    gomp_ordered_release();
}

unsigned GOMP_sections_start(unsigned count)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
    {
        tls_serial_sections_count.set(reinterpret_cast<void*>((size_t)count));
        tls_serial_sections_next.set(0);
        return gomp_section_take();
    }
    team->lock.lock();
    if (!team->sections_active)
    {
        team->sections_count = (int)count;
        team->sections_next = 0;
        team->sections_done = 0;
        team->sections_active = 1;
    }
    team->lock.unlock();
    return gomp_section_take();
}

unsigned GOMP_sections_next(void)
{
    return gomp_section_take();
}

void GOMP_sections_end_nowait(void)
{
}

void GOMP_sections_end(void)
{
    gomp_barrier();
}

static void gomp_parallel_loop_common(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, int sched, unsigned flags)
{
    gomp_loop_info info;
    info.lb = lb;
    info.ub = ub;
    info.incr = incr;
    info.chunk = chunk;
    info.sched = sched;
    tls_gomp_loop_info.set(&info);
    tls_gomp_team_init.set((void*)gomp_init_loop_team);
    GOMP_parallel(fn, data, num_threads, flags);
    tls_gomp_team_init.set(0);
    tls_gomp_loop_info.set(0);
}

static void gomp_parallel_loop_start_common(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, int sched)
{
    g_kmp_global.try_init();

    if (num_threads == 0)
        num_threads = kmp_default_num_threads();
    if (num_threads > (unsigned)g_kmp_global.kmp_max_threads)
        num_threads = g_kmp_global.kmp_max_threads;
    if (num_threads < 1)
        num_threads = 1;

    int rsched = sched;
    long rchunk = chunk;
    if (sched == 3)
        gomp_resolve_runtime(&rsched, &rchunk);

    ncnn::GompTeam* team = new ncnn::GompTeam((int)num_threads);
    team->loop_lb = lb;
    team->loop_ub = ub;
    team->loop_incr = incr;
    team->loop_chunk = rchunk;
    team->loop_next = 0;
    team->loop_sched = rsched;
    team->loop_active = 1;
    team->loop_done = 0;
    team->loop_generation++;
    team->ordered_turn = 0;

    gomp_start_split(fn, data, num_threads, team);
}

void GOMP_parallel_loop_static(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, chunk, 0, flags);
}

void GOMP_parallel_loop_static_start(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk)
{
    gomp_parallel_loop_start_common(fn, data, num_threads, lb, ub, incr, chunk, 0);
}

void GOMP_parallel_loop_dynamic(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, chunk, 1, flags);
}

void GOMP_parallel_loop_dynamic_start(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk)
{
    gomp_parallel_loop_start_common(fn, data, num_threads, lb, ub, incr, chunk, 1);
}

void GOMP_parallel_loop_nonmonotonic_dynamic(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, chunk, 1, flags);
}

void GOMP_parallel_loop_guided(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, chunk, 2, flags);
}

void GOMP_parallel_loop_guided_start(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk)
{
    gomp_parallel_loop_start_common(fn, data, num_threads, lb, ub, incr, chunk, 2);
}

void GOMP_parallel_loop_nonmonotonic_guided(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, long chunk, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, chunk, 2, flags);
}

void GOMP_parallel_loop_runtime(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, 1, 3, flags);
}

void GOMP_parallel_loop_runtime_start(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr)
{
    gomp_parallel_loop_start_common(fn, data, num_threads, lb, ub, incr, 0, 3);
}

void GOMP_parallel_loop_nonmonotonic_runtime(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, 1, 3, flags);
}

void GOMP_parallel_loop_maybe_nonmonotonic_runtime(void (*fn)(void*), void* data, unsigned num_threads, long lb, long ub, long incr, unsigned flags)
{
    gomp_parallel_loop_common(fn, data, num_threads, lb, ub, incr, 1, 3, flags);
}

void GOMP_parallel_sections_start(void (*fn)(void*), void* data, unsigned num_threads, unsigned count)
{
    g_kmp_global.try_init();

    if (num_threads == 0)
        num_threads = kmp_default_num_threads();
    if (num_threads > (unsigned)g_kmp_global.kmp_max_threads)
        num_threads = g_kmp_global.kmp_max_threads;
    if (num_threads < 1)
        num_threads = 1;

    ncnn::GompTeam* team = new ncnn::GompTeam((int)num_threads);
    team->sections_count = (int)count;
    team->sections_next = 0;
    team->sections_done = 0;
    team->sections_active = 1;

    gomp_start_split(fn, data, num_threads, team);
}

void GOMP_parallel_sections(void (*fn)(void*), void* data, unsigned num_threads, unsigned count, unsigned flags)
{
    gomp_sections_info info;
    info.count = count;
    tls_gomp_sections_info.set(&info);
    tls_gomp_team_init.set((void*)gomp_init_sections_team);
    GOMP_parallel(fn, data, num_threads, flags);
    tls_gomp_team_init.set(0);
    tls_gomp_sections_info.set(0);
}


// ---- gcc libgomp unsigned long long loop ABI ----
static unsigned long long gomp_ull_trip(bool up, unsigned long long lb, unsigned long long ub, unsigned long long incr)
{
    if (incr == 0)
        return 0;
    if (up)
    {
        if (ub < lb)
            return 0;
        unsigned long long diff = ub - lb;
        return diff / incr + (diff % incr != 0);
    }
    if (lb < ub)
        return 0;
    unsigned long long diff = lb - ub;
    return diff / incr + (diff % incr != 0);
}

static int gomp_loop_ull_begin(bool up, unsigned long long lb, unsigned long long ub, unsigned long long incr, unsigned long long chunk, int sched)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return 0;

    // the ULL ABI passes a descending step as its two's-complement value
    if (!up)
        incr = (unsigned long long)0 - incr;

    team->lock.lock();
    if (!team->loop_active)
    {
        int rsched = sched;
        long rchunk = (long)chunk;
        if (sched == 3)
            gomp_resolve_runtime(&rsched, &rchunk);
        team->loop_ull_lb = lb;
        team->loop_ull_ub = ub;
        team->loop_ull_incr = incr;
        team->loop_ull_chunk = rchunk;
        team->loop_ull_next = 0;
        team->loop_ull_sched = rsched;
        team->loop_ull_up = up ? 1 : 0;
        team->loop_active = 1;
        team->loop_done = 0;
        team->loop_generation++;
        team->ordered_turn = 0;
    }
    team->lock.unlock();
    return 1;
}

static int gomp_take_ull_static(ncnn::GompTeam* team, int first, unsigned long long* p_lb, unsigned long long* p_ub)
{
    int nt = team->num_threads;
    int tid = (int)(size_t)tls_thread_num.get();
    bool up = team->loop_ull_up != 0;
    unsigned long long lb = team->loop_ull_lb;
    unsigned long long incr = team->loop_ull_incr;
    unsigned long long chunk = (unsigned long long)team->loop_ull_chunk;
    unsigned long long trip = gomp_ull_trip(up, lb, team->loop_ull_ub, incr);
    unsigned long long idx = 0;
    unsigned long long cnt = 0;

    if (chunk == 0)
    {
        if (!first)
            return 0;
        unsigned long long per = trip / nt;
        unsigned long long rem = trip % nt;
        idx = (unsigned long long)tid * per + ((unsigned long long)tid < rem ? tid : rem);
        cnt = per + ((unsigned long long)tid < rem ? 1 : 0);
        if (cnt == 0)
            return 0;
    }
    else
    {
        size_t cursor = first ? (size_t)tid : (size_t)tls_gomp_static_chunk.get();
        if (!first)
            cursor += (size_t)nt;
        tls_gomp_static_chunk.set(reinterpret_cast<void*>(cursor));
        idx = (unsigned long long)cursor * chunk;
        if (idx >= trip)
            return 0;
        cnt = chunk;
        if (cnt > trip - idx)
            cnt = trip - idx;
    }

    if (up)
    {
        *p_lb = lb + idx * incr;
        *p_ub = *p_lb + cnt * incr;
    }
    else
    {
        *p_lb = lb - idx * incr;
        *p_ub = *p_lb - cnt * incr;
    }
    gomp_set_ordered((long)idx);
    return 1;
}

static int gomp_take_ull_dynamic(ncnn::GompTeam* team, unsigned long long* p_lb, unsigned long long* p_ub)
{
    bool up = team->loop_ull_up != 0;
    unsigned long long lb = team->loop_ull_lb;
    unsigned long long incr = team->loop_ull_incr;
    unsigned long long trip = gomp_ull_trip(up, lb, team->loop_ull_ub, incr);
    int sched = team->loop_ull_sched;
    unsigned long long chunk = (unsigned long long)team->loop_ull_chunk;

    team->lock.lock();
    unsigned long long idx = team->loop_ull_next;
    if (idx >= trip)
    {
        team->lock.unlock();
        return 0;
    }
    unsigned long long remaining = trip - idx;
    unsigned long long c = chunk > 0 ? chunk : 1;
    if (sched == 2)
    {
        unsigned long long g = remaining / (unsigned long long)(team->num_threads > 0 ? team->num_threads : 1);
        if (g > c) c = g;
    }
    if (c > remaining) c = remaining;
    team->loop_ull_next = idx + c;
    team->lock.unlock();

    if (up)
    {
        *p_lb = lb + idx * incr;
        *p_ub = *p_lb + c * incr;
    }
    else
    {
        *p_lb = lb - idx * incr;
        *p_ub = *p_lb - c * incr;
    }
    gomp_set_ordered((long)idx);
    return 1;
}

static bool gomp_ull_start_common(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, int sched, unsigned long long* istart, unsigned long long* iend)
{
    if (!gomp_loop_ull_begin(up, start, end, incr, chunk, sched))
    {
        *istart = start;
        *iend = end;
        return true;
    }
    ncnn::GompTeam* team = gomp_team();
    if (team->loop_ull_sched == 0)
        return gomp_take_ull_static(team, gomp_static_claim(team), istart, iend) != 0;
    return gomp_take_ull_dynamic(team, istart, iend) != 0;
}

static bool gomp_ull_next_common(unsigned long long* istart, unsigned long long* iend)
{
    ncnn::GompTeam* team = gomp_team();
    if (!team)
        return false;
    if (team->loop_ull_sched == 0)
        return gomp_take_ull_static(team, gomp_static_claim(team), istart, iend) != 0;
    return gomp_take_ull_dynamic(team, istart, iend) != 0;
}

bool GOMP_loop_ull_static_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 0, istart, iend);
}

bool GOMP_loop_ull_static_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_dynamic_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 1, istart, iend);
}

bool GOMP_loop_ull_dynamic_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_dynamic_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 1, istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_dynamic_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_guided_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 2, istart, iend);
}

bool GOMP_loop_ull_guided_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_guided_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 2, istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_guided_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_runtime_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, 0, 3, istart, iend);
}

bool GOMP_loop_ull_runtime_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_runtime_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, 0, 3, istart, iend);
}

bool GOMP_loop_ull_nonmonotonic_runtime_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_maybe_nonmonotonic_runtime_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, 0, 3, istart, iend);
}

bool GOMP_loop_ull_maybe_nonmonotonic_runtime_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_ordered_static_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 0, istart, iend);
}

bool GOMP_loop_ull_ordered_static_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_ordered_dynamic_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 1, istart, iend);
}

bool GOMP_loop_ull_ordered_dynamic_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_ordered_guided_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long chunk, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, chunk, 2, istart, iend);
}

bool GOMP_loop_ull_ordered_guided_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}

bool GOMP_loop_ull_ordered_runtime_start(bool up, unsigned long long start, unsigned long long end, unsigned long long incr, unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_start_common(up, start, end, incr, 0, 3, istart, iend);
}

bool GOMP_loop_ull_ordered_runtime_next(unsigned long long* istart, unsigned long long* iend)
{
    return gomp_ull_next_common(istart, iend);
}


#endif // __clang__

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEOMP
