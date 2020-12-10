#include <stdio.h>

#include "cpu.h"

#if defined __ANDROID__ || defined __linux__

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

static int test_cpu_omp()
{
    return 0;
}

#endif

int main()
{
    return 0
           || test_cpu_set()
           || test_cpu_info()
           || test_cpu_omp();
}
