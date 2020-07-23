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

#include "platform.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>    // __cpuid()
#include <immintrin.h> // _xgetbv()
#endif

#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#define __IOS__ 1
#endif
#endif

namespace ncnn {

#if defined __ANDROID__ || defined __linux__

// extract the ELF HW capabilities bitmap from /proc/self/auxv
static unsigned int get_elf_hwcap_from_proc_self_auxv()
{
    FILE* fp = fopen("/proc/self/auxv", "rb");
    if (!fp)
    {
        return 0;
    }

#define AT_HWCAP  16
#define AT_HWCAP2 26
#if __aarch64__

    struct
    {
        uint64_t tag;
        uint64_t value;
    } entry;
#else
    struct
    {
        unsigned int tag;
        unsigned int value;
    } entry;

#endif

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
#define HWCAP_ASIMD   (1 << 1)
#define HWCAP_ASIMDHP (1 << 10)
#else
// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_NEON  (1 << 12)
#define HWCAP_VFPv4 (1 << 16)
#endif

#endif // defined __ANDROID__ || defined __linux__

#if __IOS__
static unsigned int get_hw_cpufamily()
{
    unsigned int value = 0;
    size_t len = sizeof(value);
    sysctlbyname("hw.cpufamily", &value, &len, NULL, 0);
    return value;
}

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

static unsigned int g_hw_cpufamily = get_hw_cpufamily();
static cpu_type_t g_hw_cputype = get_hw_cputype();
static cpu_subtype_t g_hw_cpusubtype = get_hw_cpusubtype();
#endif // __IOS__

int cpu_support_arm_neon()
{
#if defined __ANDROID__ || defined __linux__
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
#if defined __ANDROID__ || defined __linux__
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
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMDHP;
#else
    return 0;
#endif
#elif __IOS__
#if __aarch64__
#ifndef CPUFAMILY_ARM_HURRICANE
#define CPUFAMILY_ARM_HURRICANE 0x67ceee93
#endif
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST 0x07d34b9f
#endif
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif
    return g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL || g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_x86_avx2()
{
#if defined(__x86_64__)
#ifdef _MSC_VER
    // TODO move to init function
    int cpu_info[4];
    __cpuid(cpu_info, 0);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    __cpuid(cpu_info, 1);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & 0x10000000) || !(cpu_info[2] & 0x04000000) || !(cpu_info[2] & 0x08000000))
        return 0;

    // check XSAVE enabled by kernel
    if ((_xgetbv(0) & 6) != 6)
        return 0;

    __cpuid(cpu_info, 7);
    return cpu_info[1] & 0x00000020;
#else
    // TODO gcc-specific
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#endif
#else
    return 0;
#endif
}

static int get_cpucount()
{
    int count = 0;
#if defined __ANDROID__ || defined __linux__
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp)
        return 1;

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
#elif __IOS__
    size_t len = sizeof(count);
    sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
#else
#ifdef _OPENMP
    count = omp_get_max_threads();
#else
    count = 1;
#endif // _OPENMP
#endif

    if (count < 1)
        count = 1;

    if (count > (int)sizeof(size_t) * 8)
    {
        NCNN_LOGE("more than %d cpu detected, thread affinity may not work properly :(", (int)sizeof(size_t) * 8);
    }

    return count;
}

static int g_cpucount = get_cpucount();

int get_cpu_count()
{
    return g_cpucount;
}

#if defined __ANDROID__ || defined __linux__
static int get_max_freq_khz(int cpuid)
{
    // first try, for all possible cpu
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
    {
        // second try, for online cpu
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
        fp = fopen(path, "rb");

        if (fp)
        {
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

            if (max_freq_khz != 0)
                return max_freq_khz;

            fp = NULL;
        }

        if (!fp)
        {
            // third try, for online cpu
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
            fp = fopen(path, "rb");

            if (!fp)
                return -1;

            int max_freq_khz = -1;
            fscanf(fp, "%d", &max_freq_khz);

            fclose(fp);

            return max_freq_khz;
        }
    }

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

static int set_sched_affinity(size_t thread_affinity_mask)
{
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define NCNN_CPU_SETSIZE 1024
#define __NCNN_NCPUBITS  (8 * sizeof(unsigned long))
    typedef struct
    {
        unsigned long __bits[NCNN_CPU_SETSIZE / __NCNN_NCPUBITS];
    } cpu_set_t;

#define NCNN_CPU_SET(cpu, cpusetp) \
    ((cpusetp)->__bits[(cpu) / __NCNN_NCPUBITS] |= (1UL << ((cpu) % __NCNN_NCPUBITS)))

#define NCNN_CPU_ZERO(cpusetp) \
    memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid = getpid();
#else
    pid_t pid = gettid();
#endif
#endif
    cpu_set_t mask;
    NCNN_CPU_ZERO(&mask);
    for (int i = 0; i < (int)sizeof(size_t) * 8; i++)
    {
        if (thread_affinity_mask & (1ul << i))
        {
            NCNN_CPU_SET(i, &mask);
        }
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret)
    {
        NCNN_LOGE("syscall error %d", syscallret);
        return -1;
    }

    return 0;
}
#endif // defined __ANDROID__ || defined __linux__

static int g_powersave = 0;

int get_cpu_powersave()
{
    return g_powersave;
}

int set_cpu_powersave(int powersave)
{
    if (powersave < 0 || powersave > 2)
    {
        NCNN_LOGE("powersave %d not supported", powersave);
        return -1;
    }

    size_t thread_affinity_mask = get_cpu_thread_affinity_mask(powersave);

    int ret = set_cpu_thread_affinity(thread_affinity_mask);
    if (ret != 0)
        return ret;

    g_powersave = powersave;

    return 0;
}

static size_t g_thread_affinity_mask_all = 0;
static size_t g_thread_affinity_mask_little = 0;
static size_t g_thread_affinity_mask_big = 0;

static int setup_thread_affinity_masks()
{
    g_thread_affinity_mask_all = (1ul << g_cpucount) - 1;

#if defined __ANDROID__ || defined __linux__
    int max_freq_khz_min = INT_MAX;
    int max_freq_khz_max = 0;
    std::vector<int> cpu_max_freq_khz(g_cpucount);
    for (int i = 0; i < g_cpucount; i++)
    {
        int max_freq_khz = get_max_freq_khz(i);

        //         NCNN_LOGE("%d max freq = %d khz", i, max_freq_khz);

        cpu_max_freq_khz[i] = max_freq_khz;

        if (max_freq_khz > max_freq_khz_max)
            max_freq_khz_max = max_freq_khz;
        if (max_freq_khz < max_freq_khz_min)
            max_freq_khz_min = max_freq_khz;
    }

    int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;
    if (max_freq_khz_medium == max_freq_khz_max)
    {
        g_thread_affinity_mask_little = 0;
        g_thread_affinity_mask_big = g_thread_affinity_mask_all;
        return 0;
    }

    for (int i = 0; i < g_cpucount; i++)
    {
        if (cpu_max_freq_khz[i] < max_freq_khz_medium)
            g_thread_affinity_mask_little |= (1ul << i);
        else
            g_thread_affinity_mask_big |= (1ul << i);
    }
#else
    // TODO implement me for other platforms
    g_thread_affinity_mask_little = 0;
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;
#endif

    return 0;
}

size_t get_cpu_thread_affinity_mask(int powersave)
{
    if (g_thread_affinity_mask_all == 0)
    {
        setup_thread_affinity_masks();
    }

    if (g_thread_affinity_mask_little == 0)
    {
        // SMP cpu powersave not supported
        // fallback to all cores anyway
        return g_thread_affinity_mask_all;
    }

    if (powersave == 0)
        return g_thread_affinity_mask_all;

    if (powersave == 1)
        return g_thread_affinity_mask_little;

    if (powersave == 2)
        return g_thread_affinity_mask_big;

    NCNN_LOGE("powersave %d not supported", powersave);

    // fallback to all cores anyway
    return g_thread_affinity_mask_all;
}

int set_cpu_thread_affinity(size_t thread_affinity_mask)
{
#if defined __ANDROID__ || defined __linux__
    int num_threads = 0;
    for (int i = 0; i < (int)sizeof(size_t) * 8; i++)
    {
        if (thread_affinity_mask & (1ul << i))
            num_threads++;
    }

#ifdef _OPENMP
    // set affinity for each thread
    set_omp_num_threads(num_threads);
    std::vector<int> ssarets(num_threads, 0);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; i++)
    {
        ssarets[i] = set_sched_affinity(thread_affinity_mask);
    }
    for (int i = 0; i < num_threads; i++)
    {
        if (ssarets[i] != 0)
            return -1;
    }
#else
    int ssaret = set_sched_affinity(thread_affinity_mask);
    if (ssaret != 0)
        return -1;
#endif

    return 0;
#elif __IOS__
    // thread affinity not supported on ios
    (void)thread_affinity_mask;
    return -1;
#else
    // TODO
    (void)thread_affinity_mask;
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

int get_omp_thread_num()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

int get_kmp_blocktime()
{
#if defined(_OPENMP) && __clang__
    return kmp_get_blocktime();
#else
    return 0;
#endif
}

void set_kmp_blocktime(int time_ms)
{
#if defined(_OPENMP) && __clang__
    kmp_set_blocktime(time_ms);
#else
    (void)time_ms;
#endif
}

} // namespace ncnn
