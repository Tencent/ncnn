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

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#ifdef _MSC_VER
#include <intrin.h>    // __cpuid()
#include <immintrin.h> // _xgetbv()
#endif
#if defined(__clang__) || defined(__GNUC__)
#include <cpuid.h> // __get_cpuid() and __cpuid_count()
#endif
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif

#if defined __ANDROID__ || defined __linux__
#if defined __ANDROID__
#if __ANDROID_API__ >= 18
#include <sys/auxv.h> // getauxval()
#endif
#include <dlfcn.h>
#endif
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
// A11
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
// A12
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST 0x07d34b9f
#endif
// A13
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif
// A14
#ifndef CPUFAMILY_ARM_FIRESTORM_ICESTORM
#define CPUFAMILY_ARM_FIRESTORM_ICESTORM 0x1b588bb3
#endif
// A15
#ifndef CPUFAMILY_ARM_AVALANCHE_BLIZZARD
#define CPUFAMILY_ARM_AVALANCHE_BLIZZARD 0xda33d83d
#endif
// M1
#ifndef CPUFAMILY_AARCH64_FIRESTORM_ICESTORM
#define CPUFAMILY_AARCH64_FIRESTORM_ICESTORM 0x1b588bb3
#endif
#endif // __APPLE__

#if defined(__SSE3__)
#include <immintrin.h>
#endif

namespace ncnn {

#if defined __ANDROID__ || defined __linux__

#define AT_HWCAP  16
#define AT_HWCAP2 26

#if defined __ANDROID__
// Probe the system's C library for a 'getauxval' function and call it if
// it exits, or return 0 for failure. This function is available since API
// level 18.
//
// Note that getauxval() can't really be re-implemented here, because
// its implementation does not parse /proc/self/auxv. Instead it depends
// on values  that are passed by the kernel at process-init time to the
// C runtime initialization layer.
static unsigned int get_elf_hwcap_from_getauxval()
{
#if __ANDROID_API__ >= 18
    unsigned int hwcap = getauxval(AT_HWCAP);
    if (hwcap)
        return hwcap;
#endif

    typedef unsigned long getauxval_func_t(unsigned long);

    dlerror();
    void* libc_handle = dlopen("libc.so", RTLD_NOW);
    if (!libc_handle)
    {
        NCNN_LOGE("dlopen libc.so failed %s", dlerror());
        return 0;
    }

    unsigned int result = 0;
    getauxval_func_t* func = (getauxval_func_t*)dlsym(libc_handle, "getauxval");
    if (!func)
    {
        NCNN_LOGE("dlsym getauxval failed");
    }
    else
    {
        // Note: getauxval() returns 0 on failure. Doesn't touch errno.
        result = (unsigned int)(*func)(AT_HWCAP);
    }
    dlclose(libc_handle);

    return result;
}
#endif // defined __ANDROID__

// extract the ELF HW capabilities bitmap from /proc/self/auxv
static unsigned int get_elf_hwcap_from_proc_self_auxv()
{
    FILE* fp = fopen("/proc/self/auxv", "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen /proc/self/auxv failed");
        return 0;
    }

#if __aarch64__ || __mips64 || __riscv_xlen == 64
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

static unsigned int get_elf_hwcap()
{
#if defined __ANDROID__
    unsigned int hwcap = get_elf_hwcap_from_getauxval();
    if (hwcap)
        return hwcap;
#endif

    return get_elf_hwcap_from_proc_self_auxv();
}

static unsigned int g_hwcaps = get_elf_hwcap();

#if __aarch64__
// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_ASIMD   (1 << 1)
#define HWCAP_ASIMDHP (1 << 10)
#define HWCAP_ASIMDDP (1 << 20)
#else
// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_NEON  (1 << 12)
#define HWCAP_VFPv4 (1 << 16)
#endif

#if __mips__
// from arch/mips/include/uapi/asm/hwcap.h
#define HWCAP_MIPS_MSA     (1 << 1)
#define HWCAP_LOONGSON_MMI (1 << 11)
#endif

#if __riscv
// from arch/riscv/include/uapi/asm/hwcap.h
#define COMPAT_HWCAP_ISA_F (1 << ('F' - 'A'))
#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
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
    return g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL || g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_asimddp()
{
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    return g_hwcaps & HWCAP_ASIMDDP;
#else
    return 0;
#endif
#elif __APPLE__
#if __aarch64__
    return g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
static inline void x86_cpuid(int level, unsigned int out[4])
{
#if defined(_MSC_VER)
    __cpuid((int*)out, level);
#elif defined(__clang__) || defined(__GNUC__)
    __get_cpuid(level, out, out + 1, out + 2, out + 3);
#else
    NCNN_LOGE("x86_cpuid is unknown for current compiler");
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
#endif
}

static inline void x86_cpuid_sublevel(int level, int sublevel, unsigned int out[4])
{
#if defined(_MSC_VER)
    __cpuidex((int*)out, level, sublevel);
#elif defined(__clang__) || defined(__GNUC__)
    __cpuid_count(level, sublevel, out[0], out[1], out[2], out[3]);
#else
    NCNN_LOGE("x86_cpuid_sublevel is unknown for current compiler");
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
#endif
}

static inline int x86_get_xcr0()
{
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
    return _xgetbv(0);
#elif defined(__i386__) || defined(__x86_64__)
    int xcr0 = 0;
    asm(".byte 0x0f, 0x01, 0xd0"
        : "=a"(xcr0)
        : "c"(0)
        : "%edx");
    return xcr0;
#else
    NCNN_LOGE("x86_get_xcr0 is unknown for current compiler");
    return 0xffffffff; // assume it will work
#endif
}

static int get_cpu_support_x86_avx()
{
#if !NCNN_AVX
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 1)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    return 1;
}

static int get_cpu_support_x86_fma()
{
#if !NCNN_FMA
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    return cpu_info[2] & (1u << 12);
}

static int get_cpu_support_x86_xop()
{
#if !NCNN_XOP
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0x80000000, cpu_info);

    if (cpu_info[0] < 0x80000001)
        return 0;

    x86_cpuid(0x80000001, cpu_info);

    return cpu_info[2] & (1u << 11);
}

static int get_cpu_support_x86_f16c()
{
#if !NCNN_F16C
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 1)
        return 0;

    x86_cpuid(1, cpu_info);

    return cpu_info[2] & (1u << 29);
}

static int get_cpu_support_x86_avx2()
{
#if !NCNN_AVX2
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return cpu_info[1] & (1u << 5);
}

static int get_cpu_support_x86_avx_vnni()
{
#if !NCNN_AVXVNNI
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    x86_cpuid_sublevel(7, 1, cpu_info);
    return cpu_info[0] & (1u << 4);
}

static int get_cpu_support_x86_avx512()
{
#if !NCNN_AVX512
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    // check avx512 XSAVE enabled by kernel
    if ((x86_get_xcr0() & 0xe0) != 0xe0)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return (cpu_info[1] & (1u << 16)) && (cpu_info[1] & (1u << 17)) && (cpu_info[1] & (1u << 28)) && (cpu_info[1] & (1u << 30)) && (cpu_info[1] & (1u << 31));
}

static int get_cpu_support_x86_avx512_vnni()
{
#if !NCNN_AVX512VNNI
    return 0;
#endif
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0, cpu_info);

    int nIds = cpu_info[0];
    if (nIds < 7)
        return 0;

    x86_cpuid(1, cpu_info);
    // check AVX XSAVE OSXSAVE
    if (!(cpu_info[2] & (1u << 28)) || !(cpu_info[2] & (1u << 26)) || !(cpu_info[2] & (1u << 27)))
        return 0;

    // check XSAVE enabled by kernel
    if ((x86_get_xcr0() & 6) != 6)
        return 0;

    // check avx512 XSAVE enabled by kernel
    if ((x86_get_xcr0() & 0xe0) != 0xe0)
        return 0;

    x86_cpuid_sublevel(7, 0, cpu_info);
    return cpu_info[2] & (1u << 11);
}

static int g_cpu_support_x86_avx = get_cpu_support_x86_avx();
static int g_cpu_support_x86_fma = get_cpu_support_x86_fma();
static int g_cpu_support_x86_xop = get_cpu_support_x86_xop();
static int g_cpu_support_x86_f16c = get_cpu_support_x86_f16c();
static int g_cpu_support_x86_avx2 = get_cpu_support_x86_avx2();
static int g_cpu_support_x86_avx_vnni = get_cpu_support_x86_avx_vnni();
static int g_cpu_support_x86_avx512 = get_cpu_support_x86_avx512();
static int g_cpu_support_x86_avx512_vnni = get_cpu_support_x86_avx512_vnni();
#else  // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
static const int g_cpu_support_x86_avx = 0;
static const int g_cpu_support_x86_fma = 0;
static const int g_cpu_support_x86_xop = 0;
static const int g_cpu_support_x86_f16c = 0;
static const int g_cpu_support_x86_avx2 = 0;
static const int g_cpu_support_x86_avx_vnni = 0;
static const int g_cpu_support_x86_avx512 = 0;
static const int g_cpu_support_x86_avx512_vnni = 0;
#endif // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

int cpu_support_x86_avx()
{
    return g_cpu_support_x86_avx;
}

int cpu_support_x86_fma()
{
    return g_cpu_support_x86_fma;
}

int cpu_support_x86_xop()
{
    return g_cpu_support_x86_xop;
}

int cpu_support_x86_f16c()
{
    return g_cpu_support_x86_f16c;
}

int cpu_support_x86_avx2()
{
    return g_cpu_support_x86_avx2;
}

int cpu_support_x86_avx_vnni()
{
    return g_cpu_support_x86_avx_vnni;
}

int cpu_support_x86_avx512()
{
    return g_cpu_support_x86_avx512;
}

int cpu_support_x86_avx512_vnni()
{
    return g_cpu_support_x86_avx512_vnni;
}

int cpu_support_mips_msa()
{
#if defined __ANDROID__ || defined __linux__
#if __mips__
    return g_hwcaps & HWCAP_MIPS_MSA;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_loongson_mmi()
{
#if defined __ANDROID__ || defined __linux__
#if __mips__
    return g_hwcaps & HWCAP_LOONGSON_MMI;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_riscv_v()
{
#if defined __ANDROID__ || defined __linux__
#if __riscv
    return g_hwcaps & COMPAT_HWCAP_ISA_V;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_riscv_zfh()
{
#if defined __ANDROID__ || defined __linux__
#if __riscv
    // v + f does not imply zfh, but how to discover zfh properly ?
    // upstream issue https://github.com/riscv/riscv-isa-manual/issues/414
    return g_hwcaps & COMPAT_HWCAP_ISA_V && g_hwcaps & COMPAT_HWCAP_ISA_F;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_riscv_vlenb()
{
#if __riscv
    if (!cpu_support_riscv_v())
        return 0;

    int a = 0;
    asm volatile(
        ".word  0xc22026f3  \n" // csrr  a3, vlenb
        "mv     %0, a3      \n"
        : "=r"(a)
        :
        : "memory", "a3");
    return a;
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
    int big_cpu_count = get_cpu_thread_affinity_mask(2).num_enabled();
    return big_cpu_count ? big_cpu_count : g_cpucount;
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
#if defined(__BIONIC__)
    pid_t pid = gettid();
#else
    pid_t pid = syscall(SYS_gettid);
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
    else if (g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD)
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

#ifdef _OPENMP
    int num_threads = thread_affinity_mask.num_enabled();

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

static ncnn::ThreadLocalStorage tls_flush_denormals;

int get_flush_denormals()
{
#if defined(__SSE3__)
    return (int)reinterpret_cast<size_t>(tls_flush_denormals.get());
#else
    return 0;
#endif
}

int set_flush_denormals(int flush_denormals)
{
    if (flush_denormals < 0 || flush_denormals > 3)
    {
        NCNN_LOGE("denormals_zero %d not supported", flush_denormals);
        return -1;
    }
#if defined(__SSE3__)
    if (flush_denormals == 0)
    {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    }
    else if (flush_denormals == 1)
    {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    }
    else if (flush_denormals == 2)
    {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }
    else if (flush_denormals == 3)
    {
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    }

    tls_flush_denormals.set(reinterpret_cast<void*>((size_t)flush_denormals));
    return 0;
#else
    return 0;
#endif
}

} // namespace ncnn
