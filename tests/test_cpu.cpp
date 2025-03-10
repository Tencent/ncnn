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

#include <stdio.h>

#include "cpu.h"

#if defined __ANDROID__ || defined __linux__ || defined __APPLE__

static int test_cpu_set()
{
    ncnn::CpuSet set;

    if (set.num_enabled() != 0)
    {
        fprintf(stderr, "By default all cpus should be disabled\n");
        return 1;
    }

    set.enable(0);
    if (!set.is_enabled(0))
    {
        fprintf(stderr, "CpuSet enable doesn't work\n");
        return 1;
    }

    if (set.num_enabled() != 1)
    {
        fprintf(stderr, "Only one cpu should be enabled\n");
        return 1;
    }

    set.disable(0);
    if (set.is_enabled(0))
    {
        fprintf(stderr, "CpuSet disable doesn't work\n");
        return 1;
    }

    return 0;
}

#else

static int test_cpu_set()
{
    return 0;
}

#endif

#if defined __ANDROID__ || defined __linux__

static int test_cpu_info()
{
    if (ncnn::get_cpu_count() >= 0 && ncnn::get_little_cpu_count() >= 0 && ncnn::get_big_cpu_count() >= 0)
    {
        return 0;
    }
    else
    {
        fprintf(stderr, "The system cannot have a negative number of processors\n");
        return 1;
    }
}

static int test_cpu_omp()
{
    if (ncnn::get_omp_num_threads() >= 0 && ncnn::get_omp_thread_num() >= 0 && ncnn::get_omp_dynamic() >= 0)
    {
        return 0;
    }
    else
    {
        fprintf(stderr, "The OMP cannot have a negative number of processors\n");
        return 1;
    }

    ncnn::set_omp_num_threads(1);

    ncnn::set_omp_dynamic(1);
}

static int test_cpu_powersave()
{
    if (ncnn::get_cpu_powersave() >= 0)
    {
        return 0;
    }
    else
    {
        fprintf(stderr, "By default powersave must be zero\n");
        return 1;
    }

    if (ncnn::set_cpu_powersave(-1) == -1 && ncnn::set_cpu_powersave(3) == -1)
    {
        return 0;
    }
    else
    {
        fprintf(stderr, "Set cpu powersave for `-1 < argument < 2` works incorrectly.\n");
        return 1;
    }
}

#else

#if defined _WIN32
// Check SDK >= Win7
#if _WIN32_WINNT >= _WIN32_WINNT_WIN7 // win7

static int test_cpu_info()
{
    int cpucount = ncnn::get_cpu_count();
    int bigcpucount = ncnn::get_big_cpu_count();
    int littlecpucount = ncnn::get_little_cpu_count();

    fprintf(stderr, "cpucount = %d\n", cpucount);
    fprintf(stderr, "bigcpucount = %d\n", bigcpucount);
    fprintf(stderr, "littlecpucount = %d\n", littlecpucount);

    if ((cpucount != bigcpucount + littlecpucount) || (bigcpucount > cpucount) || (littlecpucount > cpucount))
    {
        fprintf(stderr, "The number of big and little cpus must be less than or equal to the total number of cpus\n");
        return -1;
    }

    return 0;
}

#endif
#else

static int test_cpu_info()
{
    return 0;
}

#endif

static int test_cpu_omp()
{
    return 0;
}

static int test_cpu_powersave()
{
    return 0;
}

#endif

int main()
{
    return 0
           || test_cpu_set()
           || test_cpu_info()
           || test_cpu_omp()
           || test_cpu_powersave();
}
