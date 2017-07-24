// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cpu.h"

#include <stdio.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __ANDROID__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/machine.h>
#define __IOS__ 1
#endif
#endif

namespace ncnn {

#ifdef __ANDROID__

// extract the ELF HW capabilities bitmap from /proc/self/auxv
static unsigned int get_elf_hwcap_from_proc_self_auxv()
{
    FILE* fp = fopen("/proc/self/auxv", "rb");
    if (!fp)
    {
        return 0;
    }

#define AT_HWCAP 16
#define AT_HWCAP2 26

    struct { unsigned int tag; unsigned int value; } entry;

    unsigned int result = 0;
    while (!feof(fp))
    {
        int nread = fread((char*)&entry, sizeof(entry), 1, fp);
        if (nread != 1)
            break;

        if (entry.tag == 0 && entry.value == 0)
            break;

        if (entry.tag == AT_HWCAP)
        {
            result = entry.value;
            break;
        }
    }

    fclose(fp);

    return result;
}

static unsigned int g_hwcaps = get_elf_hwcap_from_proc_self_auxv();

#if __aarch64__
// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_ASIMD     (1 << 1)
#define HWCAP_ASIMDHP   (1 << 10)
#else
// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_NEON      (1 << 12)
#define HWCAP_VFPv4     (1 << 16)
#endif

#endif // __ANDROID__

#if __IOS__
static cpu_type_t get_hw_cputype()
{
    cpu_type_t value = 0;
    size_t len = sizeof(value);
    sysctlbyname("hw.cputype", &value, &len, NULL, 0);
    return value;
}

static cpu_subtype_t get_hw_cpusubtype()
{
    cpu_subtype_t value = 0;
    size_t len = sizeof(value);
    sysctlbyname("hw.cpusubtype", &value, &len, NULL, 0);
    return value;
}

static cpu_type_t g_hw_cputype = get_hw_cputype();
static cpu_subtype_t g_hw_cpusubtype = get_hw_cpusubtype();
#endif // __IOS__

int cpu_support_arm_neon()
{
#ifdef __ANDROID__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMD;
#else
    return g_hwcaps & HWCAP_NEON;
#endif
#elif __IOS__
#if __aarch64__
    return g_hw_cputype == CPU_TYPE_ARM64;
#else
    return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_vfpv4()
{
#ifdef __ANDROID__
#if __aarch64__
    // neon always enable fma and fp16
    return g_hwcaps & HWCAP_ASIMD;
#else
    return g_hwcaps & HWCAP_VFPv4;
#endif
#elif __IOS__
#if __aarch64__
    return g_hw_cputype == CPU_TYPE_ARM64;
#else
    return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7S;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_asimdhp()
{
#ifdef __ANDROID__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMDHP;
#else
    return 0;
#endif
#elif __IOS__
#if __aarch64__
    return 0;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

static int get_cpucount()
{
#ifdef __ANDROID__
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp)
        return 1;

    int count = 0;
    char line[1024];
    while (!feof(fp))
    {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (memcmp(line, "processor", 9) == 0)
        {
            count++;
        }
    }

    fclose(fp);

    if (count < 1)
        count = 1;

    return count;
#elif __IOS__
    int count = 0;
    size_t len = sizeof(count);
    sysctlbyname("hw.ncpu", &count, &len, NULL, 0);

    if (count < 1)
        count = 1;

    return count;
#else
    return 1;
#endif
}

static int g_cpucount = get_cpucount();

int get_cpu_count()
{
    return g_cpucount;
}

#ifdef __ANDROID__
static int get_max_freq_khz(int cpuid)
{
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
        return -1;

    int max_freq_khz = 0;
    while (!feof(fp))
    {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1)
            break;

        if (freq_khz > max_freq_khz)
            max_freq_khz = freq_khz;
    }

    fclose(fp);

    return max_freq_khz;
}

static int set_sched_affinity(const std::vector<int>& cpuids)
{
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
typedef struct
{
   unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
  memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
    pid_t pid = gettid();

    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i=0; i<(int)cpuids.size(); i++)
    {
        CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret)
    {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}

static int sort_cpuid_by_max_frequency(std::vector<int>& cpuids, int* little_cluster_offset)
{
    const int cpu_count = cpuids.size();

    *little_cluster_offset = 0;

    if (cpu_count == 0)
        return 0;

    std::vector<int> cpu_max_freq_khz;
    cpu_max_freq_khz.resize(cpu_count);

    for (int i=0; i<cpu_count; i++)
    {
        int max_freq_khz = get_max_freq_khz(i);

//         printf("%d max freq = %d khz\n", i, max_freq_khz);

        cpuids[i] = i;
        cpu_max_freq_khz[i] = max_freq_khz;
    }

    // sort cpuid as big core first
    // simple bubble sort
    for (int i=0; i<cpu_count; i++)
    {
        for (int j=i+1; j<cpu_count; j++)
        {
            if (cpu_max_freq_khz[i] < cpu_max_freq_khz[j])
            {
                // swap
                int tmp = cpuids[i];
                cpuids[i] = cpuids[j];
                cpuids[j] = tmp;

                tmp = cpu_max_freq_khz[i];
                cpu_max_freq_khz[i] = cpu_max_freq_khz[j];
                cpu_max_freq_khz[j] = tmp;
            }
        }
    }

    // SMP
    int mid_max_freq_khz = (cpu_max_freq_khz.front() + cpu_max_freq_khz.back()) / 2;
    if (mid_max_freq_khz == cpu_max_freq_khz.back())
        return 0;

    for (int i=0; i<cpu_count; i++)
    {
        if (cpu_max_freq_khz[i] < mid_max_freq_khz)
        {
            *little_cluster_offset = i;
            break;
        }
    }

    return 0;
}
#endif // __ANDROID__

static int g_powersave = 0;

int get_cpu_powersave()
{
    return g_powersave;
}

int set_cpu_powersave(int powersave)
{
#ifdef __ANDROID__
    static std::vector<int> sorted_cpuids;
    static int little_cluster_offset = 0;

    if (sorted_cpuids.empty())
    {
        // 0 ~ g_cpucount
        sorted_cpuids.resize(g_cpucount);
        for (int i=0; i<g_cpucount; i++)
        {
            sorted_cpuids[i] = i;
        }

        // descent sort by max frequency
        sort_cpuid_by_max_frequency(sorted_cpuids, &little_cluster_offset);
    }

    if (little_cluster_offset == 0)
    {
        fprintf(stderr, "SMP cpu powersave not supported\n");
        return -1;
    }

    // prepare affinity cpuid
    std::vector<int> cpuids;
    if (powersave == 0)
    {
        cpuids = sorted_cpuids;
    }
    else if (powersave == 1)
    {
        cpuids = std::vector<int>(sorted_cpuids.begin() + little_cluster_offset, sorted_cpuids.end());
    }
    else if (powersave == 2)
    {
        cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() +  + little_cluster_offset);
    }
    else
    {
        fprintf(stderr, "powersave %d not supported\n", powersave);
        return -1;
    }

#ifdef _OPENMP
    // set affinity for each thread
    int num_threads = cpuids.size();
    omp_set_num_threads(num_threads);
    std::vector<int> ssarets(num_threads, 0);
    #pragma omp parallel for
    for (int i=0; i<num_threads; i++)
    {
        ssarets[i] = set_sched_affinity(cpuids);
    }
    for (int i=0; i<num_threads; i++)
    {
        if (ssarets[i] != 0)
        {
            return -1;
        }
    }
#else
    int ssaret = set_sched_affinity(cpuids);
    if (ssaret != 0)
    {
        return -1;
    }
#endif

    g_powersave = powersave;

    return 0;
#elif __IOS__
    // thread affinity not supported on ios
    return -1;
#else
    // TODO
    (void) powersave;  // Avoid unused parameter warning.
    return -1;
#endif
}

int get_omp_num_threads()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

void set_omp_num_threads(int num_threads)
{
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
}

int get_omp_dynamic()
{
#ifdef _OPENMP
    return omp_get_dynamic();
#else
    return 0;
#endif
}

void set_omp_dynamic(int dynamic)
{
#ifdef _OPENMP
    omp_set_dynamic(dynamic);
#else
    (void)dynamic;
#endif
}

} // namespace ncnn
