// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_CPU_H
#define NCNN_CPU_H

#include <stddef.h>

#if defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined __ANDROID__ || defined __linux__
#include <sched.h> // cpu_set_t
#endif

#include "platform.h"

namespace ncnn {

// Windows processor groups each contain up to 64 logical processors.
// Support up to NCNN_MAX_PROCESSOR_GROUPS groups (512 CPUs max at 8 groups).
#define NCNN_MAX_PROCESSOR_GROUPS 8

class NCNN_EXPORT CpuSet
{
public:
    CpuSet();
    void enable(int cpu);
    void disable(int cpu);
    void disable_all();
    bool is_enabled(int cpu) const;
    int num_enabled() const;

public:
#if defined _WIN32
    KAFFINITY mask[NCNN_MAX_PROCESSOR_GROUPS];
#endif
#if defined __ANDROID__ || defined __linux__
    cpu_set_t cpu_set;
#endif
#if __APPLE__
    unsigned int policy;
#endif
};
