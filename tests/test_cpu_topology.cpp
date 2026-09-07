// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Linker wrapping affects only this test executable, not the ncnn library.
// Each CTest case starts a fresh process because CPU information is cached.
extern "C" FILE* __real_fopen(const char* path, const char* mode);
extern "C" FILE* __real_fopen64(const char* path, const char* mode);

static int cpuinfo_reads = 0;
static int sysfs_reads = 0;

static FILE* missing_file()
{
    errno = ENOENT;
    return 0;
}

static FILE* fixture_file(const char* text)
{
    FILE* fp = tmpfile();
    if (!fp)
        return 0;

    if (fputs(text, fp) == EOF || fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        return 0;
    }

    return fp;
}

static FILE* open_fixture(const char* path, const char* mode, FILE* (*real_fopen)(const char*, const char*))
{
    const char* scenario = getenv("NCNN_TEST_CPU_TOPOLOGY");
    if (!scenario)
        return real_fopen(path, mode);

    if (strcmp(path, "/proc/cpuinfo") == 0)
    {
        cpuinfo_reads++;
        return fixture_file("processor : 0\nprocessor : 1\nprocessor : 2\nprocessor : 3\n"
                            "processor : 4\nprocessor : 5\nprocessor : 6\nprocessor : 7\n");
    }

    const char prefix[] = "/sys/devices/system/cpu/";
    if (strncmp(path, prefix, sizeof(prefix) - 1) != 0)
        return real_fopen(path, mode);

    sysfs_reads++;
    if (strcmp(scenario, "no_sysfs") == 0)
        return missing_file();

    int cpu = -1;
    int offset = 0;
    if (sscanf(path, "/sys/devices/system/cpu/cpu%d/%n", &cpu, &offset) != 1 || offset == 0 || cpu < 0 || cpu >= 8)
        return missing_file();

    const char* entry = path + offset;
    if (strcmp(entry, "topology/thread_siblings_list") == 0 || strcmp(entry, "topology/thread_siblings") == 0)
    {
        if (strcmp(scenario, "missing") == 0 || (strcmp(scenario, "partial") == 0 && cpu >= 4))
            return missing_file();
        if (strcmp(scenario, "empty") == 0)
            return fixture_file("");
        if (strcmp(scenario, "malformed") == 0)
            return fixture_file("not-a-cpu\n");

        // Four physical cores: {0,4}, {1,5}, {2,6}, {3,7}.
        // In the partial case every group still has a readable representative.
        const bool smt = strcmp(scenario, "non_smt") != 0;
        const int core = smt ? cpu % 4 : cpu;
        char text[32];
        if (strcmp(entry, "topology/thread_siblings_list") == 0)
        {
            if (smt)
                sprintf(text, "%d,%d\n", core, core + 4);
            else
                sprintf(text, "%d\n", core);
        }
        else
        {
            const unsigned int mask = smt ? (1u << core) | (1u << (core + 4)) : 1u << core;
            sprintf(text, "%08x\n", mask);
        }
        return fixture_file(text);
    }

    if (strcmp(entry, "cache/index0/level") == 0)
        return fixture_file("2\n");
    if (strcmp(entry, "cache/index0/type") == 0)
        return fixture_file("Unified\n");
    if (strcmp(entry, "cache/index0/size") == 0)
        return fixture_file("1024K\n");
    if (strcmp(entry, "cache/index0/shared_cpu_map") == 0)
    {
        if (strcmp(scenario, "empty_map") == 0)
            return fixture_file("00000000\n");
        if (strcmp(scenario, "outside_map") == 0)
            return fixture_file("00000300\n"); // CPUs 8 and 9 are outside the detected range.
        return fixture_file("00000033\n");     // CPUs 0, 1, 4 and 5 share this cache.
    }

    // No other CPU sysfs files: do not mix fixtures with host topology.
    return missing_file();
}

extern "C" FILE* __wrap_fopen(const char* path, const char* mode)
{
    return open_fixture(path, mode, __real_fopen);
}

extern "C" FILE* __wrap_fopen64(const char* path, const char* mode)
{
    return open_fixture(path, mode, __real_fopen64);
}

int main()
{
    const char* scenario = getenv("NCNN_TEST_CPU_TOPOLOGY");
    if (!scenario)
    {
        fprintf(stderr, "NCNN_TEST_CPU_TOPOLOGY must select a fixture\n");
        return 1;
    }

    int expected_physical = 4;
    int expected_l2 = 0;
    if (strcmp(scenario, "normal") == 0 || strcmp(scenario, "partial") == 0)
    {
        expected_l2 = 512 * 1024;
    }
    else if (strcmp(scenario, "non_smt") == 0)
    {
        expected_physical = 8;
        expected_l2 = 256 * 1024;
    }
    else if (strcmp(scenario, "no_sysfs") == 0 || strcmp(scenario, "missing") == 0 || strcmp(scenario, "empty") == 0 || strcmp(scenario, "malformed") == 0)
    {
        expected_physical = 8;
    }
    else if (strcmp(scenario, "empty_map") != 0 && strcmp(scenario, "outside_map") != 0)
    {
        fprintf(stderr, "Unknown CPU topology fixture: %s\n", scenario);
        return 1;
    }

    // Initialization must not divide by zero when the shared physical CPU count
    // cannot be resolved. The final L2 value may come from sysconf or fallback.
    const int logical = ncnn::get_cpu_count();
    const int physical = ncnn::get_physical_cpu_count();
    const int l2 = ncnn::get_cpu_level2_cache_size();

    if (cpuinfo_reads == 0 || sysfs_reads == 0)
    {
        fprintf(stderr, "CPU topology fixture was not used\n");
        return 1;
    }

    if (logical != 8 || physical != expected_physical || l2 <= 0 || (expected_l2 != 0 && l2 != expected_l2))
    {
        fprintf(stderr, "CPU topology %s: logical=%d physical=%d L2=%d, expected logical=8 physical=%d L2=%d (0 means any positive fallback)\n",
                scenario, logical, physical, l2, expected_physical, expected_l2);
        return 1;
    }

    return 0;
}
