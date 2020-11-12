#include <stdio.h>

#include "cpu.h"

#if defined __ANDROID__ || defined __linux__

static int test_cpu_info()
{
    if (ncnn::get_cpu_count() >= 0 && ncnn::get_little_cpu_count() >= 0 && ncnn::get_big_cpu_count() >= 0)
    {
        fprintf(stderr, "The system cannot have a negative number of processors\n");
        return 0;
    }
    else
    {
        return 1;
    }
}

static int test_powersave()
{
    const int state = ncnn::get_cpu_powersave();
    if (state != 0)
    {
        fprintf(stderr, "ncnn::get_cpu_powersave() returned: '%d' instead of '0'\n", state);
        return 1;
    }

    if (ncnn::set_cpu_powersave(-1) != -1 || ncnn::set_cpu_powersave(3) != -1)
    {
        fprintf(stderr, "ncnn::set_cpu_powersave is avaliabe only for cpus: 0, 1, 2\n");
        return 1;
    }

    if (ncnn::set_cpu_powersave(1) != 0 || ncnn::get_cpu_powersave() != 1)
    {
        fprintf(stderr, "ncnn::set_cpu_powersave works incorrectly for zero cpu\n");
        return 1;
    }

    return 0;
}

static int test_cpu_thread_affinity()
{
    const ncnn::CpuSet& mask = ncnn::get_cpu_thread_affinity_mask(0);
    if (ncnn::set_cpu_thread_affinity(mask) == 0)
    {
        fprintf(stderr, "ncnn::set_cpu_thread_affinity doesn't work\n");
        return 0;
    }
    else
    {
        return 1;
    }
}

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

static int test_cpu_info()
{
    return 0;
}

static int test_powersave()
{
    return 0;
}

static int test_cpu_thread_affinity()
{
    return 0;
}

#endif

int main()
{
    return 0
           || test_cpu_set()
           || test_cpu_info()
           || test_powersave()
           || test_cpu_thread_affinity();
}
