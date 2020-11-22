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
#if NCNN_SIMPLEOMP
#include "simpleomp.h"
#else
#include <omp.h>
#endif
#endif

#ifdef _MSC_VER
#include <intrin.h>    // __cpuid()
#include <immintrin.h> // _xgetbv()
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif

#if defined __ANDROID__ || defined __linux__
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#include <mach/mach.h>
#include <mach/machine.h>
#include <mach/thread_act.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define __IOS__ 1
#endif
// define missing cpu model for old sdk
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
#ifndef CPUFAMILY_ARM_FIRESTORM_ICESTORM
#define CPUFAMILY_ARM_FIRESTORM_ICESTORM 0x1b588bb3
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

#if __APPLE__
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
#endif // __APPLE__

#if defined __ANDROID__ || defined __linux__
CpuSet::CpuSet()
{
    disable_all();
}

void CpuSet::enable(int cpu)
{
    CPU_SET(cpu, &cpu_set);
}

void CpuSet::disable(int cpu)
{
    CPU_CLR(cpu, &cpu_set);
}

void CpuSet::disable_all()
{
    CPU_ZERO(&cpu_set);
}

bool CpuSet::is_enabled(int cpu) const
{
    return CPU_ISSET(cpu, &cpu_set);
}

int CpuSet::num_enabled() const
{
    int num_enabled = 0;
    for (int i = 0; i < (int)sizeof(cpu_set_t) * 8; i++)
    {
        if (is_enabled(i))
            num_enabled++;
    }

    return num_enabled;
}
#elif __APPLE__
CpuSet::CpuSet()
{
    disable_all();
}

void CpuSet::enable(int cpu)
{
    policy |= (1 << cpu);
}

void CpuSet::disable(int cpu)
{
    policy &= ~(1 << cpu);
}

void CpuSet::disable_all()
{
    policy = 0;
}

bool CpuSet::is_enabled(int cpu) const
{
    return policy & (1 << cpu);
}

int CpuSet::num_enabled() const
{
    int num_enabled = 0;
    for (int i = 0; i < (int)sizeof(policy) * 8; i++)
    {
        if (is_enabled(i))
            num_enabled++;
    }

    return num_enabled;
}
#else
CpuSet::CpuSet()
{
}

void CpuSet::enable(int /* cpu */)
{
}

void CpuSet::disable(int /* cpu */)
{
}

void CpuSet::disable_all()
{
}

bool CpuSet::is_enabled(int /* cpu */) const
{
    return true;
}

int CpuSet::num_enabled() const
{
    return get_cpu_count();
}
#endif

int cpu_support_arm_neon()
{
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMD;
#else
    return g_hwcaps & HWCAP_NEON;
#endif
#elif __APPLE__
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
#elif __APPLE__
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
#elif __APPLE__
#if __aarch64__
    return g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL || g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_x86_avx2()
{
#if (_M_AMD64 || __x86_64__) || (_M_IX86 || __i386__)
#if defined(_MSC_VER)
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
#elif defined(__clang__)
#if __clang_major__ >= 6
    __builtin_cpu_init();
#endif
    return __builtin_cpu_supports("avx2");
#elif defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    // TODO: other x86 compilers checking avx2 here
    NCNN_LOGE("AVX2 detection method is unknown for current compiler");
    return 0;
#endif
#else
    return 0;
#endif
}

static int get_cpucount()
{
    int count = 0;
#ifdef __EMSCRIPTEN__
    if (emscripten_has_threading_support())
        count = emscripten_num_logical_cores();
    else
        count = 1;
#elif defined __ANDROID__ || defined __linux__
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
#elif __APPLE__
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

    return count;
}

static int g_cpucount = get_cpucount();

int get_cpu_count()
{
    return g_cpucount;
}

int get_little_cpu_count()
{
    return get_cpu_thread_affinity_mask(1).num_enabled();
}

int get_big_cpu_count()
{
    return get_cpu_thread_affinity_mask(2).num_enabled();
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
            int nscan = fscanf(fp, "%d", &max_freq_khz);
            if (nscan != 1)
            {
                NCNN_LOGE("fscanf cpuinfo_max_freq error %d", nscan);
            }
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

static int set_sched_affinity(const CpuSet& thread_affinity_mask)
{
    // set affinity for thread
#if defined(__GLIBC__) || defined(__OHOS__)
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid = getpid();
#else
    pid_t pid = gettid();
#endif
#endif

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(cpu_set_t), &thread_affinity_mask.cpu_set);
    if (syscallret)
    {
        NCNN_LOGE("syscall error %d", syscallret);
        return -1;
    }

    return 0;
}
#endif // defined __ANDROID__ || defined __linux__

#if __APPLE__
static int set_sched_affinity(const CpuSet& thread_affinity_mask)
{
    // https://developer.apple.com/library/archive/releasenotes/Performance/RN-AffinityAPI/index.html
    // http://www.hybridkernel.com/2015/01/18/binding_threads_to_cores_osx.html
    // https://gist.github.com/Coneko/4234842

    // This is a quite outdated document. Apple will not allow developers to set CPU affinity.
    // In OS X 10.5 it worked, later it became a suggestion to OS X, then in 10.10 or so (as well in later ones), macOS will ignore any affinity settings.
    // see https://github.com/Tencent/ncnn/pull/2335#discussion_r528233919   --- AmeAkio

    int affinity_tag = THREAD_AFFINITY_TAG_NULL;
    for (int i = 0; i < (int)sizeof(thread_affinity_mask.policy) * 8; i++)
    {
        if (thread_affinity_mask.is_enabled(i))
        {
            affinity_tag = i + 1;
            break;
        }
    }

    mach_port_t tid = pthread_mach_thread_np(pthread_self());

    thread_affinity_policy_data_t policy_data;
    policy_data.affinity_tag = affinity_tag;
    int ret = thread_policy_set(tid, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy_data, THREAD_AFFINITY_POLICY_COUNT);
    if (ret && ret != KERN_NOT_SUPPORTED)
    {
        NCNN_LOGE("thread_policy_set error %d", ret);
        return -1;
    }

    return 0;
}
#endif // __APPLE__

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

    const CpuSet& thread_affinity_mask = get_cpu_thread_affinity_mask(powersave);

    int ret = set_cpu_thread_affinity(thread_affinity_mask);
    if (ret != 0)
        return ret;

    g_powersave = powersave;

    return 0;
}

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_big;

static int setup_thread_affinity_masks()
{
    g_thread_affinity_mask_all.disable_all();

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
        g_thread_affinity_mask_little.disable_all();
        g_thread_affinity_mask_big = g_thread_affinity_mask_all;
        return 0;
    }

    for (int i = 0; i < g_cpucount; i++)
    {
        if (cpu_max_freq_khz[i] < max_freq_khz_medium)
            g_thread_affinity_mask_little.enable(i);
        else
            g_thread_affinity_mask_big.enable(i);
    }
#elif __APPLE__
    // affinity info from cpu model
    if (g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL)
    {
        // 2 + 4
        g_thread_affinity_mask_big.enable(0);
        g_thread_affinity_mask_big.enable(1);
        g_thread_affinity_mask_little.enable(2);
        g_thread_affinity_mask_little.enable(3);
        g_thread_affinity_mask_little.enable(4);
        g_thread_affinity_mask_little.enable(5);
    }
    else if (g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM)
    {
        // 2 + 4 or 4 + 4
        if (get_cpu_count() == 6)
        {
            g_thread_affinity_mask_big.enable(0);
            g_thread_affinity_mask_big.enable(1);
            g_thread_affinity_mask_little.enable(2);
            g_thread_affinity_mask_little.enable(3);
            g_thread_affinity_mask_little.enable(4);
            g_thread_affinity_mask_little.enable(5);
        }
        else
        {
            g_thread_affinity_mask_big.enable(0);
            g_thread_affinity_mask_big.enable(1);
            g_thread_affinity_mask_big.enable(2);
            g_thread_affinity_mask_big.enable(3);
            g_thread_affinity_mask_little.enable(4);
            g_thread_affinity_mask_little.enable(5);
            g_thread_affinity_mask_little.enable(6);
            g_thread_affinity_mask_little.enable(7);
        }
    }
    else
    {
        // smp models
        g_thread_affinity_mask_little.disable_all();
        g_thread_affinity_mask_big = g_thread_affinity_mask_all;
    }
#else
    // TODO implement me for other platforms
    g_thread_affinity_mask_little.disable_all();
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;
#endif

    return 0;
}

const CpuSet& get_cpu_thread_affinity_mask(int powersave)
{
    setup_thread_affinity_masks();

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

int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask)
{
#if defined __ANDROID__ || defined __linux__
    int num_threads = thread_affinity_mask.num_enabled();

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
#elif __APPLE__
    int num_threads = thread_affinity_mask.num_enabled();

#ifdef _OPENMP
    // set affinity for each thread
    set_omp_num_threads(num_threads);
    std::vector<int> ssarets(num_threads, 0);
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; i++)
    {
        // assign one core for each thread
        int core = -1 - i;
        for (int j = 0; j < (int)sizeof(thread_affinity_mask.policy) * 8; j++)
        {
            if (thread_affinity_mask.is_enabled(j))
            {
                if (core == -1)
                {
                    core = j;
                    break;
                }
                else
                {
                    core++;
                }
            }
        }
        CpuSet this_thread_affinity_mask;
        if (core != -1 - i)
        {
            this_thread_affinity_mask.enable(core);
        }

        ssarets[i] = set_sched_affinity(this_thread_affinity_mask);
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
