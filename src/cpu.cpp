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
#include <setjmp.h>
#include <signal.h>
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

#if defined _WIN32 && !(defined __MINGW32__)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <powerbase.h>
#endif

#if defined __ANDROID__ || defined __linux__
#if defined __ANDROID__
#if __ANDROID_API__ >= 18
#include <sys/auxv.h> // getauxval()
#endif
#include <sys/system_properties.h> // __system_property_get()
#include <dlfcn.h>
#endif
#include <ctype.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if __APPLE__
#include <mach/mach.h>
#include <mach/machine.h>
#include <mach/thread_act.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
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
// A16
#ifndef CPUFAMILY_ARM_EVEREST_SAWTOOTH
#define CPUFAMILY_ARM_EVEREST_SAWTOOTH 0x8765edea
#endif
// M1
#ifndef CPUFAMILY_AARCH64_FIRESTORM_ICESTORM
#define CPUFAMILY_AARCH64_FIRESTORM_ICESTORM 0x1b588bb3
#endif
// M2
#ifndef CPUFAMILY_AARCH64_AVALANCHE_BLIZZARD
#define CPUFAMILY_AARCH64_AVALANCHE_BLIZZARD 0xda33d83d
#endif
#endif // __APPLE__

#if defined(__SSE3__)
#include <immintrin.h>
#endif

#define RUAPU_IMPLEMENTATION
#include "ruapu.h"

// topology info
static int g_cpucount;
static int g_physical_cpucount;
static int g_powersave;
static ncnn::CpuSet g_cpu_affinity_mask_all;
static ncnn::CpuSet g_cpu_affinity_mask_little;
static ncnn::CpuSet g_cpu_affinity_mask_big;

// isa info
#if defined _WIN32
#if __aarch64__
static int g_cpu_support_arm_asimdhp;
static int g_cpu_support_arm_cpuid;
static int g_cpu_support_arm_asimddp;
static int g_cpu_support_arm_asimdfhm;
static int g_cpu_support_arm_bf16;
static int g_cpu_support_arm_i8mm;
static int g_cpu_support_arm_sve;
static int g_cpu_support_arm_sve2;
static int g_cpu_support_arm_svebf16;
static int g_cpu_support_arm_svei8mm;
static int g_cpu_support_arm_svef32mm;
#elif __arm__
static int g_cpu_support_arm_edsp;
static int g_cpu_support_arm_neon;
static int g_cpu_support_arm_vfpv4;
#endif // __aarch64__ || __arm__
#elif defined __ANDROID__ || defined __linux__
static unsigned int g_hwcaps;
static unsigned int g_hwcaps2;
#elif __APPLE__
static unsigned int g_hw_cpufamily;
static cpu_type_t g_hw_cputype;
static cpu_subtype_t g_hw_cpusubtype;
#if __aarch64__
static int g_hw_optional_arm_FEAT_FP16;
static int g_hw_optional_arm_FEAT_DotProd;
static int g_hw_optional_arm_FEAT_FHM;
static int g_hw_optional_arm_FEAT_BF16;
static int g_hw_optional_arm_FEAT_I8MM;
#endif // __aarch64__
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
static int g_cpu_support_x86_avx;
static int g_cpu_support_x86_fma;
static int g_cpu_support_x86_xop;
static int g_cpu_support_x86_f16c;
static int g_cpu_support_x86_avx2;
static int g_cpu_support_x86_avx_vnni;
static int g_cpu_support_x86_avx512;
static int g_cpu_support_x86_avx512_vnni;
static int g_cpu_support_x86_avx512_bf16;
static int g_cpu_support_x86_avx512_fp16;
#endif // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

static int g_cpu_level2_cachesize;
static int g_cpu_level3_cachesize;

// misc info
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
static int g_cpu_is_arm_a53_a55;
#endif // __aarch64__
#endif // defined __ANDROID__ || defined __linux__

static bool is_being_debugged()
{
#if defined _WIN32
    return IsDebuggerPresent();
#elif defined __ANDROID__ || defined __linux__
    // https://stackoverflow.com/questions/3596781/how-to-detect-if-the-current-process-is-being-run-by-gdb
    int status_fd = open("/proc/self/status", O_RDONLY);
    if (status_fd == -1)
        return false;

    char buf[4096];
    ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);
    close(status_fd);

    if (num_read <= 0)
        return false;

    buf[num_read] = '\0';
    const char tracerPidString[] = "TracerPid:";
    const char* tracer_pid_ptr = strstr(buf, tracerPidString);
    if (!tracer_pid_ptr)
        return false;

    for (const char* ch = tracer_pid_ptr + sizeof(tracerPidString) - 1; ch <= buf + num_read; ++ch)
    {
        if (isspace(*ch))
            continue;

        return isdigit(*ch) != 0 && *ch != '0';
    }

    return false;
#elif defined __APPLE__
    // https://stackoverflow.com/questions/2200277/detecting-debugger-on-mac-os-x
    struct kinfo_proc info;
    info.kp_proc.p_flag = 0;

    int mib[4];
    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();

    size_t size = sizeof(info);
    sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);

    return ((info.kp_proc.p_flag & P_TRACED) != 0);
#else
    // unknown platform :(
    fprintf(stderr, "unknown platform!\n");
    return false;
#endif
}

#if defined __ANDROID__ || defined __linux__

#define AT_HWCAP  16
#define AT_HWCAP2 26

#if __aarch64__
// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_ASIMD     (1 << 1)
#define HWCAP_ASIMDHP   (1 << 10)
#define HWCAP_CPUID     (1 << 11)
#define HWCAP_ASIMDDP   (1 << 20)
#define HWCAP_SVE       (1 << 22)
#define HWCAP_ASIMDFHM  (1 << 23)
#define HWCAP2_SVE2     (1 << 1)
#define HWCAP2_SVEI8MM  (1 << 9)
#define HWCAP2_SVEF32MM (1 << 10)
#define HWCAP2_SVEBF16  (1 << 12)
#define HWCAP2_I8MM     (1 << 13)
#define HWCAP2_BF16     (1 << 14)
#else
// from arch/arm/include/uapi/asm/hwcap.h
#define HWCAP_EDSP  (1 << 7)
#define HWCAP_NEON  (1 << 12)
#define HWCAP_VFPv4 (1 << 16)
#endif

#if __mips__
// from arch/mips/include/uapi/asm/hwcap.h
#define HWCAP_MIPS_MSA     (1 << 1)
#define HWCAP_LOONGSON_MMI (1 << 11)
#endif

#if __loongarch64
// from arch/loongarch/include/uapi/asm/hwcap.h
#define HWCAP_LOONGARCH_LSX  (1 << 4)
#define HWCAP_LOONGARCH_LASX (1 << 5)
#endif

#if __riscv
// from arch/riscv/include/uapi/asm/hwcap.h
#define COMPAT_HWCAP_ISA_F (1 << ('F' - 'A'))
#define COMPAT_HWCAP_ISA_V (1 << ('V' - 'A'))
#endif

#if defined __ANDROID__
// Probe the system's C library for a 'getauxval' function and call it if
// it exits, or return 0 for failure. This function is available since API
// level 18.
//
// Note that getauxval() can't really be re-implemented here, because
// its implementation does not parse /proc/self/auxv. Instead it depends
// on values  that are passed by the kernel at process-init time to the
// C runtime initialization layer.
static unsigned int get_elf_hwcap_from_getauxval(unsigned int type)
{
#if __ANDROID_API__ >= 18
    unsigned int hwcap = getauxval(type);
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
        result = (unsigned int)(*func)(type);
    }
    dlclose(libc_handle);

    return result;
}
#endif // defined __ANDROID__

// extract the ELF HW capabilities bitmap from /proc/self/auxv
static unsigned int get_elf_hwcap_from_proc_self_auxv(unsigned int type)
{
    FILE* fp = fopen("/proc/self/auxv", "rb");
    if (!fp)
    {
        NCNN_LOGE("fopen /proc/self/auxv failed");
        return 0;
    }

#if __aarch64__ || __mips64 || __riscv_xlen == 64 || __loongarch64
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

        if (entry.tag == type)
        {
            result = entry.value;
            break;
        }
    }

    fclose(fp);

    return result;
}

static unsigned int get_elf_hwcap(unsigned int type)
{
    unsigned int hwcap = 0;

#if defined __ANDROID__
    hwcap = get_elf_hwcap_from_getauxval(type);
#endif

    if (!hwcap)
        hwcap = get_elf_hwcap_from_proc_self_auxv(type);

#if defined __ANDROID__
#if __aarch64__
    if (type == AT_HWCAP)
    {
        // samsung exynos9810 on android pre-9 incorrectly reports armv8.2
        // for little cores, but big cores only support armv8.0
        // drop all armv8.2 features used by ncnn for preventing SIGILLs
        // ref https://reviews.llvm.org/D114523
        char arch[PROP_VALUE_MAX];
        int len = __system_property_get("ro.arch", arch);
        if (len > 0 && strncmp(arch, "exynos9810", 10) == 0)
        {
            hwcap &= ~HWCAP_ASIMDHP;
            hwcap &= ~HWCAP_ASIMDDP;
        }
    }
#endif // __aarch64__
#endif // defined __ANDROID__

    return hwcap;
}
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

static int get_hw_capability(const char* cap)
{
    int64_t value = 0;
    size_t len = sizeof(value);
    sysctlbyname(cap, &value, &len, NULL, 0);
    return value;
}
#endif // __APPLE__

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
static inline void x86_cpuid(int level, unsigned int out[4])
{
#if defined(_MSC_VER) && !defined(__clang__)
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
    unsigned int cpu_info[4] = {0};
    x86_cpuid(0x80000000, cpu_info);

    if (cpu_info[0] < 0x80000001)
        return 0;

    x86_cpuid(0x80000001, cpu_info);

    return cpu_info[2] & (1u << 11);
}

static int get_cpu_support_x86_f16c()
{
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
#if __APPLE__
    return ruapu_supports("avxvnni");
#else
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
#endif
}

static int get_cpu_support_x86_avx512()
{
#if __APPLE__
    return ruapu_supports("avx512f") && ruapu_supports("avx512bw") && ruapu_supports("avx512cd") && ruapu_supports("avx512dq") && ruapu_supports("avx512vl");
#else
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
#endif
}

static int get_cpu_support_x86_avx512_vnni()
{
#if __APPLE__
    return ruapu_supports("avx512vnni");
#else
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
#endif
}

static int get_cpu_support_x86_avx512_bf16()
{
#if __APPLE__
    return ruapu_supports("avx512bf16");
#else
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
    return cpu_info[0] & (1u << 5);
#endif
}

static int get_cpu_support_x86_avx512_fp16()
{
#if __APPLE__
    return ruapu_supports("avx512fp16");
#else
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
    return cpu_info[3] & (1u << 23);
#endif
}
#endif // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

static int get_cpucount()
{
    int count = 0;
#ifdef __EMSCRIPTEN__
    if (emscripten_has_threading_support())
        count = emscripten_num_logical_cores();
    else
        count = 1;
#elif (defined _WIN32 && !(defined __MINGW32__))
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    count = system_info.dwNumberOfProcessors;
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

#if defined __ANDROID__ || defined __linux__
static int get_thread_siblings(int cpuid)
{
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings", cpuid);

    FILE* fp = fopen(path, "rb");
    if (!fp)
        return -1;

    int thread_siblings = -1;
    int nscan = fscanf(fp, "%x", &thread_siblings);
    if (nscan != 1)
    {
        // ignore
    }

    fclose(fp);

    return thread_siblings;
}
#endif // defined __ANDROID__ || defined __linux__

static int get_physical_cpucount()
{
    int count = 0;
#if (defined _WIN32 && !(defined __MINGW32__))
    typedef BOOL(WINAPI * LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformation");
    if (glpi == NULL)
    {
        NCNN_LOGE("GetLogicalProcessorInformation is not supported");
        return g_cpucount;
    }

    DWORD return_length = 0;
    glpi(NULL, &return_length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(return_length);
    glpi(buffer, &return_length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    DWORD byte_offset = 0;
    while (byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= return_length)
    {
        if (ptr->Relationship == RelationProcessorCore)
        {
            count++;
        }

        byte_offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

    free(buffer);
#elif defined __ANDROID__ || defined __linux__
    std::vector<int> thread_set;
    for (int i = 0; i < g_cpucount; i++)
    {
        int thread_siblings = get_thread_siblings(i);
        if (thread_siblings == -1)
        {
            // ignore malformed one
            continue;
        }

        bool thread_siblings_exists = false;
        for (size_t j = 0; j < thread_set.size(); j++)
        {
            if (thread_set[j] == thread_siblings)
            {
                thread_siblings_exists = true;
                break;
            }
        }

        if (!thread_siblings_exists)
        {
            thread_set.push_back(thread_siblings);
            count++;
        }
    }
#elif __APPLE__
    size_t len = sizeof(count);
    sysctlbyname("hw.physicalcpu_max", &count, &len, NULL, 0);
#else
    count = g_cpucount;
#endif

    if (count > g_cpucount)
        count = g_cpucount;

    return count;
}

#if defined __ANDROID__ || defined __linux__
static int get_data_cache_size(int cpuid, int level)
{
    char path[256];

    // discover sysfs cache entry
    int indexid = -1;
    for (int i = 0;; i++)
    {
        // check level
        {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpuid, i);
            FILE* fp = fopen(path, "rb");
            if (!fp)
                break;

            int cache_level = -1;
            int nscan = fscanf(fp, "%d", &cache_level);
            fclose(fp);
            if (nscan != 1 || cache_level != level)
                continue;
        }

        // check type
        {
            sprintf(path, "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpuid, i);
            FILE* fp = fopen(path, "rb");
            if (!fp)
                break;

            char type[32];
            int nscan = fscanf(fp, "%31s", type);
            fclose(fp);
            if (nscan != 1 || (strcmp(type, "Data") != 0 && strcmp(type, "Unified") != 0))
                continue;
        }

        indexid = i;
        break;
    }

    if (indexid == -1)
    {
        // no sysfs entry
        return 0;
    }

    // get size
    int cache_size_K = 0;
    {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpuid, indexid);
        FILE* fp = fopen(path, "rb");
        if (!fp)
            return 0;

        int nscan = fscanf(fp, "%dK", &cache_size_K);
        fclose(fp);
        if (nscan != 1)
        {
            NCNN_LOGE("fscanf cache_size_K error %d", nscan);
            return 0;
        }
    }

    // parse shared_cpu_map mask
    ncnn::CpuSet shared_cpu_map;
    {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_map", cpuid, indexid);
        FILE* fp = fopen(path, "rb");
        if (!fp)
            return 0;

        char shared_cpu_map_str[256];
        int nscan = fscanf(fp, "%255s", shared_cpu_map_str);
        fclose(fp);
        if (nscan != 1)
        {
            NCNN_LOGE("fscanf shared_cpu_map error %d", nscan);
            return 0;
        }

        int len = strlen(shared_cpu_map_str);

        if (shared_cpu_map_str[0] == '0' && shared_cpu_map_str[1] == 'x')
        {
            // skip leading 0x
            len -= 2;
        }

        int ci = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            char x = shared_cpu_map_str[i];
            if (x & 1) shared_cpu_map.enable(ci + 0);
            if (x & 2) shared_cpu_map.enable(ci + 1);
            if (x & 4) shared_cpu_map.enable(ci + 2);
            if (x & 8) shared_cpu_map.enable(ci + 3);

            ci += 4;
        }
    }

    if (shared_cpu_map.num_enabled() == 1)
        return cache_size_K * 1024;

    // resolve physical cpu count in the shared_cpu_map
    int shared_physical_cpu_count = 0;
    {
        std::vector<int> thread_set;
        for (int i = 0; i < g_cpucount; i++)
        {
            if (!shared_cpu_map.is_enabled(i))
                continue;

            int thread_siblings = get_thread_siblings(i);
            if (thread_siblings == -1)
            {
                // ignore malformed one
                continue;
            }

            bool thread_siblings_exists = false;
            for (size_t j = 0; j < thread_set.size(); j++)
            {
                if (thread_set[j] == thread_siblings)
                {
                    thread_siblings_exists = true;
                    break;
                }
            }

            if (!thread_siblings_exists)
            {
                thread_set.push_back(thread_siblings);
                shared_physical_cpu_count++;
            }
        }
    }

    // return per-physical-core cache size with 4K aligned
    cache_size_K = (cache_size_K / shared_physical_cpu_count + 3) / 4 * 4;

    return cache_size_K * 1024;
}

static int get_big_cpu_data_cache_size(int level)
{
    if (g_cpu_affinity_mask_big.num_enabled() == 0)
    {
        // smp cpu
        return get_data_cache_size(0, level);
    }

    for (int i = 0; i < g_cpucount; i++)
    {
        if (g_cpu_affinity_mask_big.is_enabled(i))
        {
            return get_data_cache_size(i, level);
        }
    }

    // should never reach here, fallback to cpu0
    return get_data_cache_size(0, level);
}
#endif // defined __ANDROID__ || defined __linux__

static int get_cpu_level2_cachesize()
{
    int size = 0;
#if (defined _WIN32 && !(defined __MINGW32__))
    typedef BOOL(WINAPI * LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformation");
    if (glpi != NULL)
    {
        DWORD return_length = 0;
        glpi(NULL, &return_length);

        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(return_length);
        glpi(buffer, &return_length);

        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
        DWORD byte_offset = 0;
        while (byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= return_length)
        {
            if (ptr->Relationship == RelationCache)
            {
                PCACHE_DESCRIPTOR Cache = &ptr->Cache;
                if (Cache->Level == 2)
                {
                    size = std::max(size, (int)Cache->Size);
                }
            }

            byte_offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            ptr++;
        }

        free(buffer);
    }
#elif defined __ANDROID__ || defined __linux__
    size = get_big_cpu_data_cache_size(2);
#if defined(_SC_LEVEL2_CACHE_SIZE)
    if (size <= 0)
        size = sysconf(_SC_LEVEL2_CACHE_SIZE);
#endif
#elif __APPLE__
    // perflevel 0 is the higher performance cluster
    int cpusperl2 = get_hw_capability("hw.perflevel0.cpusperl2");
    int l2cachesize = get_hw_capability("hw.perflevel0.l2cachesize");
    size = cpusperl2 > 1 ? l2cachesize / cpusperl2 : l2cachesize;
#endif

    // fallback to a common value
    if (size <= 0)
    {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
        size = 64 * 1024;
        if (g_cpu_support_x86_avx)
            size = 128 * 1024;
        if (g_cpu_support_x86_avx2)
            size = 256 * 1024;
        if (g_cpu_support_x86_avx512)
            size = 1024 * 1024;
#elif __aarch64__
        size = 256 * 1024;
#elif __arm__
        size = 128 * 1024;
#else
        // is 64k still too large here ?
        size = 64 * 1024;
#endif
    }

    return size;
}

static int get_cpu_level3_cachesize()
{
    int size = 0;
#if (defined _WIN32 && !(defined __MINGW32__))
    typedef BOOL(WINAPI * LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformation");
    if (glpi != NULL)
    {
        DWORD return_length = 0;
        glpi(NULL, &return_length);

        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(return_length);
        glpi(buffer, &return_length);

        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
        DWORD byte_offset = 0;
        while (byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= return_length)
        {
            if (ptr->Relationship == RelationCache)
            {
                PCACHE_DESCRIPTOR Cache = &ptr->Cache;
                if (Cache->Level == 3)
                {
                    size = std::max(size, (int)Cache->Size);
                }
            }

            byte_offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            ptr++;
        }

        free(buffer);
    }
#elif defined __ANDROID__ || defined __linux__
    size = get_big_cpu_data_cache_size(3);
#if defined(_SC_LEVEL3_CACHE_SIZE)
    if (size <= 0)
        size = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
#elif __APPLE__
    // perflevel 0 is the higher performance cluster
    // get the size shared among all cpus
    size = get_hw_capability("hw.perflevel0.l3cachesize");
#endif

    // l3 cache size can be zero

    return size;
}

#if (defined _WIN32 && !(defined __MINGW32__))
static ncnn::CpuSet get_smt_cpu_mask()
{
    ncnn::CpuSet smt_cpu_mask;

    typedef BOOL(WINAPI * LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
    LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetLogicalProcessorInformation");
    if (glpi == NULL)
    {
        NCNN_LOGE("GetLogicalProcessorInformation is not supported");
        return smt_cpu_mask;
    }

    DWORD return_length = 0;
    glpi(NULL, &return_length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(return_length);
    glpi(buffer, &return_length);

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = buffer;
    DWORD byte_offset = 0;
    while (byte_offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= return_length)
    {
        if (ptr->Relationship == RelationProcessorCore)
        {
            ncnn::CpuSet smt_set;
            smt_set.mask = ptr->ProcessorMask;
            if (smt_set.num_enabled() > 1)
            {
                // this core is smt
                smt_cpu_mask.mask |= smt_set.mask;
            }
        }

        byte_offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

    free(buffer);

    return smt_cpu_mask;
}

static std::vector<int> get_max_freq_mhz()
{
    typedef struct _PROCESSOR_POWER_INFORMATION
    {
        ULONG Number;
        ULONG MaxMhz;
        ULONG CurrentMhz;
        ULONG MhzLimit;
        ULONG MaxIdleState;
        ULONG CurrentIdleState;
    } PROCESSOR_POWER_INFORMATION, *PPROCESSOR_POWER_INFORMATION;

    HMODULE powrprof = LoadLibrary(TEXT("powrprof.dll"));

    typedef LONG(WINAPI * LPFN_CNPI)(POWER_INFORMATION_LEVEL, PVOID, ULONG, PVOID, ULONG);
    LPFN_CNPI cnpi = (LPFN_CNPI)GetProcAddress(powrprof, "CallNtPowerInformation");
    if (cnpi == NULL)
    {
        NCNN_LOGE("CallNtPowerInformation is not supported");
        FreeLibrary(powrprof);
        return std::vector<int>(g_cpucount, 0);
    }

    DWORD return_length = sizeof(PROCESSOR_POWER_INFORMATION) * g_cpucount;
    PPROCESSOR_POWER_INFORMATION buffer = (PPROCESSOR_POWER_INFORMATION)malloc(return_length);

    cnpi(ProcessorInformation, NULL, 0, buffer, return_length);

    std::vector<int> ret;
    for (int i = 0; i < g_cpucount; i++)
    {
        ULONG max_mhz = buffer[i].MaxMhz;
        ret.push_back(max_mhz);
    }

    free(buffer);
    FreeLibrary(powrprof);
    return ret;
}

static int set_sched_affinity(const ncnn::CpuSet& thread_affinity_mask)
{
    DWORD_PTR prev_mask = SetThreadAffinityMask(GetCurrentThread(), thread_affinity_mask.mask);
    if (prev_mask == 0)
    {
        NCNN_LOGE("SetThreadAffinityMask failed %d", GetLastError());
        return -1;
    }

    return 0;
}
#endif // (defined _WIN32 && !(defined __MINGW32__))

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

static bool is_smt_cpu(int cpuid)
{
    // https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/stable/sysfs-devices-system-cpu#L68-72
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpu%d/topology/core_cpus_list", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp)
    {
        sprintf(path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpuid);
        fp = fopen(path, "rb");

        if (!fp)
            return false;
    }

    bool is_smt = false;
    while (!feof(fp))
    {
        char ch = fgetc(fp);
        if (ch == ',' || ch == '-')
        {
            is_smt = true;
            break;
        }
    }

    fclose(fp);

    return is_smt;
}

static int set_sched_affinity(const ncnn::CpuSet& thread_affinity_mask)
{
    // set affinity for thread
#if defined(__BIONIC__) && !defined(__OHOS__)
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
static int set_sched_affinity(const ncnn::CpuSet& thread_affinity_mask)
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

static void initialize_cpu_thread_affinity_mask(ncnn::CpuSet& mask_all, ncnn::CpuSet& mask_little, ncnn::CpuSet& mask_big)
{
    mask_all.disable_all();
    for (int i = 0; i < g_cpucount; i++)
    {
        mask_all.enable(i);
    }

#if (defined _WIN32 && !(defined __MINGW32__))
    // get max freq mhz for all cores
    int max_freq_mhz_min = INT_MAX;
    int max_freq_mhz_max = 0;
    std::vector<int> cpu_max_freq_mhz = get_max_freq_mhz();
    for (int i = 0; i < g_cpucount; i++)
    {
        int max_freq_mhz = cpu_max_freq_mhz[i];

        // NCNN_LOGE("%d max freq = %d khz", i, max_freq_mhz);

        if (max_freq_mhz > max_freq_mhz_max)
            max_freq_mhz_max = max_freq_mhz;
        if (max_freq_mhz < max_freq_mhz_min)
            max_freq_mhz_min = max_freq_mhz;
    }

    int max_freq_mhz_medium = (max_freq_mhz_min + max_freq_mhz_max) / 2;
    if (max_freq_mhz_medium == max_freq_mhz_max)
    {
        mask_little.disable_all();
        mask_big = mask_all;
        return;
    }

    ncnn::CpuSet smt_cpu_mask = get_smt_cpu_mask();

    for (int i = 0; i < g_cpucount; i++)
    {
        if (smt_cpu_mask.is_enabled(i))
        {
            // always treat smt core as big core
            mask_big.enable(i);
            continue;
        }

        if (cpu_max_freq_mhz[i] < max_freq_mhz_medium)
            mask_little.enable(i);
        else
            mask_big.enable(i);
    }
#elif defined __ANDROID__ || defined __linux__
    int max_freq_khz_min = INT_MAX;
    int max_freq_khz_max = 0;
    std::vector<int> cpu_max_freq_khz(g_cpucount);
    for (int i = 0; i < g_cpucount; i++)
    {
        int max_freq_khz = get_max_freq_khz(i);

        // NCNN_LOGE("%d max freq = %d khz", i, max_freq_khz);

        cpu_max_freq_khz[i] = max_freq_khz;

        if (max_freq_khz > max_freq_khz_max)
            max_freq_khz_max = max_freq_khz;
        if (max_freq_khz < max_freq_khz_min)
            max_freq_khz_min = max_freq_khz;
    }

    int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;
    if (max_freq_khz_medium == max_freq_khz_max)
    {
        mask_little.disable_all();
        mask_big = mask_all;
        return;
    }

    for (int i = 0; i < g_cpucount; i++)
    {
        if (is_smt_cpu(i))
        {
            // always treat smt core as big core
            mask_big.enable(i);
            continue;
        }

        if (cpu_max_freq_khz[i] < max_freq_khz_medium)
            mask_little.enable(i);
        else
            mask_big.enable(i);
    }
#elif __APPLE__
    int nperflevels = get_hw_capability("hw.nperflevels");
    if (nperflevels == 1)
    {
        // smp models
        mask_little.disable_all();
        mask_big = mask_all;
    }
    else
    {
        // two or more clusters, level0 is the high-performance cluster
        int perflevel0_logicalcpu = get_hw_capability("hw.perflevel0.logicalcpu_max");
        for (int i = 0; i < perflevel0_logicalcpu; i++)
        {
            mask_big.enable(i);
        }
        for (int i = perflevel0_logicalcpu; i < g_cpucount; i++)
        {
            mask_little.enable(i);
        }
    }
#else
    // TODO implement me for other platforms
    mask_little.disable_all();
    mask_big = mask_all;
#endif
}

#if defined __ANDROID__ || defined __linux__
#if __aarch64__
union midr_info_t
{
    struct __attribute__((packed))
    {
        unsigned int revision : 4;
        unsigned int part : 12;
        unsigned int architecture : 4;
        unsigned int variant : 4;
        unsigned int implementer : 8;
    };
    unsigned int midr;

    midr_info_t(unsigned int _midr)
        : midr(_midr)
    {
    }
};

static unsigned int get_midr_from_sysfs(int cpuid)
{
    char path[256];
    sprintf(path, "/sys/devices/system/cpu/cpu%d/regs/identification/midr_el1", cpuid);

    FILE* fp = fopen(path, "rb");
    if (!fp)
        return 0;

    unsigned int midr_el1 = 0;
    int nscan = fscanf(fp, "%x", &midr_el1);
    if (nscan != 1)
    {
        // ignore
    }

    fclose(fp);

    return midr_el1;
}

static int get_midr_from_proc_cpuinfo(std::vector<unsigned int>& midrs)
{
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp)
        return -1;

    midrs.resize(g_cpucount, 0);

    int cpuid = -1;
    midr_info_t midr_info(0);

    char line[1024];
    while (!feof(fp))
    {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (memcmp(line, "processor", 9) == 0)
        {
            // processor       : 4
            int id = -1;
            int nscan = sscanf(line, "%*[^:]: %d", &id);
            if (nscan != 1)
                continue;

            if (cpuid >= 0 && cpuid < g_cpucount)
            {
                if (midr_info.midr == 0)
                {
                    // shared midr
                    midrs[cpuid] = (unsigned int)-1;
                }
                else
                {
                    // save midr and reset
                    midrs[cpuid] = midr_info.midr;
                    for (int i = 0; i < g_cpucount; i++)
                    {
                        if (midrs[i] == (unsigned int)-1)
                            midrs[i] = midr_info.midr;
                    }
                }

                midr_info.midr = 0;
            }

            cpuid = id;
        }

        if (cpuid == -1)
            continue;

        if (memcmp(line, "CPU implementer", 15) == 0)
        {
            // CPU implementer : 0x51
            unsigned int id = 0;
            int nscan = sscanf(line, "%*[^:]: %x", &id);
            if (nscan != 1)
                continue;

            midr_info.implementer = id;
        }
        else if (memcmp(line, "CPU architecture", 16) == 0)
        {
            // CPU architecture: 8
            int id = 0;
            int nscan = sscanf(line, "%*[^:]: %d", &id);
            if (nscan != 1)
                continue;

            midr_info.architecture = id;
        }
        else if (memcmp(line, "CPU variant", 11) == 0)
        {
            // CPU variant     : 0xd
            int id = 0;
            int nscan = sscanf(line, "%*[^:]: %x", &id);
            if (nscan != 1)
                continue;

            midr_info.variant = id;
        }
        else if (memcmp(line, "CPU part", 8) == 0)
        {
            // CPU part        : 0x804
            int id = 0;
            int nscan = sscanf(line, "%*[^:]: %x", &id);
            if (nscan != 1)
                continue;

            midr_info.part = id;
        }
        else if (memcmp(line, "CPU revision", 12) == 0)
        {
            // CPU revision    : 14
            int id = 0;
            int nscan = sscanf(line, "%*[^:]: %d", &id);
            if (nscan != 1)
                continue;

            midr_info.revision = id;
        }
    }

    fclose(fp);

    if (cpuid >= 0 && cpuid < g_cpucount)
    {
        if (midr_info.midr == 0)
        {
            // shared midr
            midrs[cpuid] = (unsigned int)-1;
        }
        else
        {
            // save midr and reset
            midrs[cpuid] = midr_info.midr;
            for (int i = 0; i < g_cpucount; i++)
            {
                if (midrs[i] == (unsigned int)-1)
                    midrs[i] = midr_info.midr;
            }
        }

        midr_info.midr = 0;
    }

    // /proc/cpuinfo may only report little/online cores on old kernel
    if (g_cpu_affinity_mask_big.num_enabled() == g_cpucount)
    {
        // assign the remaining unknown midrs for smp cpu
        for (int i = 0; i < g_cpucount; i++)
        {
            if (midrs[i] == 0)
                midrs[i] = midr_info.midr;
        }
    }
    else
    {
        // clear the big core midrs for hmp cpu if they are the same as little cores
        unsigned int little_midr = 0;
        for (int i = 0; i < g_cpucount; i++)
        {
            if (g_cpu_affinity_mask_little.is_enabled(i))
            {
                little_midr = midrs[i];
                break;
            }
        }

        for (int i = 0; i < g_cpucount; i++)
        {
            if (g_cpu_affinity_mask_big.is_enabled(i))
            {
                if (midrs[i] == little_midr)
                {
                    midrs[i] = 0;
                }
            }
        }
    }

    return 0;
}

// return midr for the current running core
static unsigned int get_midr_from_register()
{
    uint64_t midr;
    asm volatile("mrs   %0, MIDR_EL1"
                 : "=r"(midr));

    return (unsigned int)midr;
}

static int get_sched_affinity(ncnn::CpuSet& thread_affinity_mask)
{
    // get affinity for thread
#if defined(__BIONIC__) && !defined(__OHOS__)
    pid_t pid = gettid();
#else
    pid_t pid = syscall(SYS_gettid);
#endif

    thread_affinity_mask.disable_all();

    int syscallret = syscall(__NR_sched_getaffinity, pid, sizeof(cpu_set_t), &thread_affinity_mask.cpu_set);
    if (syscallret)
    {
        // handle get error silently
        return -1;
    }

    return 0;
}

static int midr_is_a53_a55(unsigned int midr)
{
    // 0x 41 ? f d03 ? = arm cortex-a53
    // 0x 51 ? f 801 ? = qcom kryo200 a53
    // 0x 41 ? f d04 ? = arm cortex-a35
    // 0x 41 ? f d05 ? = arm cortex-a55
    // 0x 51 ? f 803 ? = qcom kryo300 a55
    // 0x 51 ? f 805 ? = qcom kryo400 a55

    midr_info_t midr_info(midr);

    return (midr_info.implementer == 0x41 && midr_info.part == 0xd03)
           || (midr_info.implementer == 0x51 && midr_info.part == 0x801)
           || (midr_info.implementer == 0x41 && midr_info.part == 0xd04)
           || (midr_info.implementer == 0x41 && midr_info.part == 0xd05)
           || (midr_info.implementer == 0x51 && midr_info.part == 0x803)
           || (midr_info.implementer == 0x51 && midr_info.part == 0x805);
}

static int detect_cpu_is_arm_a53_a55()
{
    int a53_a55_cpu_count = 0;

    // first try, iterate /sys/devices/system/cpu/cpuX/regs/identification/midr_el1
    bool sysfs_midr = true;
    for (int i = 0; i < g_cpucount; i++)
    {
        unsigned int midr = 0;

        // for kernel 4.7+
        midr = get_midr_from_sysfs(i);
        if (midr == 0)
        {
            sysfs_midr = false;
            break;
        }

        if (midr_is_a53_a55(midr))
        {
            a53_a55_cpu_count++;
        }
    }

    if (!sysfs_midr)
    {
        // second try, collect midr from /proc/cpuinfo
        std::vector<unsigned int> midrs;
        int ret = get_midr_from_proc_cpuinfo(midrs);
        if (ret == 0 && (int)midrs.size() == g_cpucount)
        {
            for (int i = 0; i < g_cpucount; i++)
            {
                if (midr_is_a53_a55(midrs[i]))
                {
                    a53_a55_cpu_count++;
                }
            }
        }
        else
        {
            // third try, assume all aarch64 little cores are a53/a55
            a53_a55_cpu_count = g_cpu_affinity_mask_little.num_enabled();
        }
    }

    if (a53_a55_cpu_count == 0)
        return 0; // all non a53/a55

    if (a53_a55_cpu_count == g_cpucount)
        return 1; // all a53/a55

    // little cores are a53/a55
    return 2;
}
#endif // __aarch64__
#endif // defined __ANDROID__ || defined __linux__

// the initialization
static void initialize_global_cpu_info()
{
    g_cpucount = get_cpucount();
    g_physical_cpucount = get_physical_cpucount();
    g_powersave = 0;
    initialize_cpu_thread_affinity_mask(g_cpu_affinity_mask_all, g_cpu_affinity_mask_little, g_cpu_affinity_mask_big);

#if (defined _WIN32 && (__aarch64__ || __arm__)) || __APPLE__
    if (!is_being_debugged())
    {
        ruapu_init();
    }
#endif

#if defined _WIN32
#if __aarch64__
    g_cpu_support_arm_cpuid = ruapu_supports("cpuid");
    g_cpu_support_arm_asimdhp = ruapu_supports("asimdhp") || IsProcessorFeaturePresent(43); // dp implies hp
    g_cpu_support_arm_asimddp = ruapu_supports("asimddp") || IsProcessorFeaturePresent(43); // 43 is PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
    g_cpu_support_arm_asimdfhm = ruapu_supports("asimdfhm");
    g_cpu_support_arm_bf16 = ruapu_supports("bf16");
    g_cpu_support_arm_i8mm = ruapu_supports("i8mm");
    g_cpu_support_arm_sve = ruapu_supports("sve");
    g_cpu_support_arm_sve2 = ruapu_supports("sve2");
    g_cpu_support_arm_svebf16 = ruapu_supports("svebf16");
    g_cpu_support_arm_svei8mm = ruapu_supports("svei8mm");
    g_cpu_support_arm_svef32mm = ruapu_supports("svef32mm");
#elif __arm__
    g_cpu_support_arm_edsp = ruapu_supports("edsp");
    g_cpu_support_arm_neon = 1; // all modern windows arm devices have neon
    g_cpu_support_arm_vfpv4 = ruapu_supports("vfpv4");
#endif // __aarch64__ || __arm__
#elif defined __ANDROID__ || defined __linux__
    g_hwcaps = get_elf_hwcap(AT_HWCAP);
    g_hwcaps2 = get_elf_hwcap(AT_HWCAP2);
#elif __APPLE__
    g_hw_cpufamily = get_hw_cpufamily();
    g_hw_cputype = get_hw_cputype();
    g_hw_cpusubtype = get_hw_cpusubtype();
#if __aarch64__
    g_hw_optional_arm_FEAT_FP16 = get_hw_capability("hw.optional.arm.FEAT_FP16");
    g_hw_optional_arm_FEAT_DotProd = get_hw_capability("hw.optional.arm.FEAT_DotProd");
    g_hw_optional_arm_FEAT_FHM = get_hw_capability("hw.optional.arm.FEAT_FHM");
    g_hw_optional_arm_FEAT_BF16 = get_hw_capability("hw.optional.arm.FEAT_BF16");
    g_hw_optional_arm_FEAT_I8MM = get_hw_capability("hw.optional.arm.FEAT_I8MM");
#endif // __aarch64__
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    g_cpu_support_x86_avx = get_cpu_support_x86_avx();
    g_cpu_support_x86_fma = get_cpu_support_x86_fma();
    g_cpu_support_x86_xop = get_cpu_support_x86_xop();
    g_cpu_support_x86_f16c = get_cpu_support_x86_f16c();
    g_cpu_support_x86_avx2 = get_cpu_support_x86_avx2();
    g_cpu_support_x86_avx_vnni = get_cpu_support_x86_avx_vnni();
    g_cpu_support_x86_avx512 = get_cpu_support_x86_avx512();
    g_cpu_support_x86_avx512_vnni = get_cpu_support_x86_avx512_vnni();
    g_cpu_support_x86_avx512_bf16 = get_cpu_support_x86_avx512_bf16();
    g_cpu_support_x86_avx512_fp16 = get_cpu_support_x86_avx512_fp16();
#endif // defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)

    g_cpu_level2_cachesize = get_cpu_level2_cachesize();
    g_cpu_level3_cachesize = get_cpu_level3_cachesize();

#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    g_cpu_is_arm_a53_a55 = detect_cpu_is_arm_a53_a55();
#endif // __aarch64__
#endif // defined __ANDROID__ || defined __linux__
}

static int g_cpu_info_initialized = 0;

static inline void try_initialize_global_cpu_info()
{
    if (!g_cpu_info_initialized)
    {
        initialize_global_cpu_info();
        g_cpu_info_initialized = 1;
    }
}

namespace ncnn {

#if (defined _WIN32 && !(defined __MINGW32__))
CpuSet::CpuSet()
{
    disable_all();
}

void CpuSet::enable(int cpu)
{
    mask |= ((ULONG_PTR)1 << cpu);
}

void CpuSet::disable(int cpu)
{
    mask &= ~((ULONG_PTR)1 << cpu);
}

void CpuSet::disable_all()
{
    mask = 0;
}

bool CpuSet::is_enabled(int cpu) const
{
    return mask & ((ULONG_PTR)1 << cpu);
}

int CpuSet::num_enabled() const
{
    int num_enabled = 0;
    for (int i = 0; i < (int)sizeof(mask) * 8; i++)
    {
        if (is_enabled(i))
            num_enabled++;
    }

    return num_enabled;
}
#elif defined __ANDROID__ || defined __linux__
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
    policy |= ((unsigned int)1 << cpu);
}

void CpuSet::disable(int cpu)
{
    policy &= ~((unsigned int)1 << cpu);
}

void CpuSet::disable_all()
{
    policy = 0;
}

bool CpuSet::is_enabled(int cpu) const
{
    return policy & ((unsigned int)1 << cpu);
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

int cpu_support_arm_edsp()
{
    try_initialize_global_cpu_info();
#if __arm__ && !__aarch64__
#if defined _WIN32
    return g_cpu_support_arm_edsp;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_EDSP;
#elif __APPLE__
    return g_hw_cputype == CPU_TYPE_ARM;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_neon()
{
    try_initialize_global_cpu_info();
#if __aarch64__
    return 1;
#elif __arm__
#if defined _WIN32
    return g_cpu_support_arm_neon;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_NEON;
#elif __APPLE__
    return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_vfpv4()
{
    try_initialize_global_cpu_info();
#if __aarch64__
    return 1;
#elif __arm__
#if defined _WIN32
    return g_cpu_support_arm_vfpv4;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_VFPv4;
#elif __APPLE__
    return g_hw_cputype == CPU_TYPE_ARM && g_hw_cpusubtype > CPU_SUBTYPE_ARM_V7S;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_asimdhp()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_asimdhp;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_ASIMDHP;
#elif __APPLE__
    return g_hw_optional_arm_FEAT_FP16
           || g_hw_cpufamily == CPUFAMILY_ARM_MONSOON_MISTRAL
           || g_hw_cpufamily == CPUFAMILY_ARM_VORTEX_TEMPEST
           || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER
           || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM
           || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD
           || g_hw_cpufamily == CPUFAMILY_ARM_EVEREST_SAWTOOTH;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_cpuid()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_cpuid;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_CPUID;
#elif __APPLE__
    return 0;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_asimddp()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_asimddp;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_ASIMDDP;
#elif __APPLE__
    return g_hw_optional_arm_FEAT_DotProd
           || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER
           || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM
           || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD
           || g_hw_cpufamily == CPUFAMILY_ARM_EVEREST_SAWTOOTH;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_asimdfhm()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_asimdfhm;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_ASIMDFHM;
#elif __APPLE__
    return g_hw_optional_arm_FEAT_FHM
           || g_hw_cpufamily == CPUFAMILY_ARM_LIGHTNING_THUNDER
           || g_hw_cpufamily == CPUFAMILY_ARM_FIRESTORM_ICESTORM
           || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD
           || g_hw_cpufamily == CPUFAMILY_ARM_EVEREST_SAWTOOTH;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_bf16()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_bf16;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_BF16;
#elif __APPLE__
    return g_hw_optional_arm_FEAT_BF16
           || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD
           || g_hw_cpufamily == CPUFAMILY_ARM_EVEREST_SAWTOOTH;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_i8mm()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_i8mm;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_I8MM;
#elif __APPLE__
    return g_hw_optional_arm_FEAT_I8MM
           || g_hw_cpufamily == CPUFAMILY_ARM_AVALANCHE_BLIZZARD
           || g_hw_cpufamily == CPUFAMILY_ARM_EVEREST_SAWTOOTH;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_sve()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_sve;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps & HWCAP_SVE;
#elif __APPLE__
    return 0; // no known apple cpu support armv8.6 sve
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_sve2()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_sve2;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_SVE2;
#elif __APPLE__
    return 0; // no known apple cpu support armv8.6 sve2
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_svebf16()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_svebf16;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_SVEBF16;
#elif __APPLE__
    return 0; // no known apple cpu support armv8.6 svebf16
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_svei8mm()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_svei8mm;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_SVEI8MM;
#elif __APPLE__
    return 0; // no known apple cpu support armv8.6 svei8mm
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_arm_svef32mm()
{
    try_initialize_global_cpu_info();
#if __aarch64__
#if defined _WIN32
    return g_cpu_support_arm_svef32mm;
#elif defined __ANDROID__ || defined __linux__
    return g_hwcaps2 & HWCAP2_SVEF32MM;
#elif __APPLE__
    return 0; // no known apple cpu support armv8.6 svef32mm
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_x86_avx()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx;
#else
    return 0;
#endif
}

int cpu_support_x86_fma()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_fma;
#else
    return 0;
#endif
}

int cpu_support_x86_xop()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_xop;
#else
    return 0;
#endif
}

int cpu_support_x86_f16c()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_f16c;
#else
    return 0;
#endif
}

int cpu_support_x86_avx2()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx2;
#else
    return 0;
#endif
}

int cpu_support_x86_avx_vnni()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx_vnni;
#else
    return 0;
#endif
}

int cpu_support_x86_avx512()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx512;
#else
    return 0;
#endif
}

int cpu_support_x86_avx512_vnni()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx512_vnni;
#else
    return 0;
#endif
}

int cpu_support_x86_avx512_bf16()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx512_bf16;
#else
    return 0;
#endif
}

int cpu_support_x86_avx512_fp16()
{
    try_initialize_global_cpu_info();
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return g_cpu_support_x86_avx512_fp16;
#else
    return 0;
#endif
}

int cpu_support_mips_msa()
{
    try_initialize_global_cpu_info();
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

int cpu_support_loongarch_lsx()
{
    try_initialize_global_cpu_info();
#if defined __ANDROID__ || defined __linux__
#if __loongarch64
    return g_hwcaps & HWCAP_LOONGARCH_LSX;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_loongarch_lasx()
{
    try_initialize_global_cpu_info();
#if defined __ANDROID__ || defined __linux__
#if __loongarch64
    return g_hwcaps & HWCAP_LOONGARCH_LASX;
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_support_loongson_mmi()
{
    try_initialize_global_cpu_info();
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
    try_initialize_global_cpu_info();
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
    try_initialize_global_cpu_info();
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
    try_initialize_global_cpu_info();
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

int get_cpu_count()
{
    try_initialize_global_cpu_info();
    return g_cpucount;
}

int get_little_cpu_count()
{
    try_initialize_global_cpu_info();
    return get_cpu_thread_affinity_mask(1).num_enabled();
}

int get_big_cpu_count()
{
    try_initialize_global_cpu_info();
    int big_cpu_count = get_cpu_thread_affinity_mask(2).num_enabled();
    return big_cpu_count ? big_cpu_count : g_cpucount;
}

int get_physical_cpu_count()
{
    try_initialize_global_cpu_info();
    return g_physical_cpucount;
}

int get_physical_little_cpu_count()
{
    try_initialize_global_cpu_info();
    if (g_physical_cpucount == g_cpucount)
        return get_little_cpu_count();

    return g_physical_cpucount * 2 - g_cpucount;
}

int get_physical_big_cpu_count()
{
    try_initialize_global_cpu_info();
    if (g_physical_cpucount == g_cpucount)
        return get_big_cpu_count();

    return g_cpucount - g_physical_cpucount;
}

int get_cpu_level2_cache_size()
{
    try_initialize_global_cpu_info();
    return g_cpu_level2_cachesize;
}

int get_cpu_level3_cache_size()
{
    try_initialize_global_cpu_info();
    return g_cpu_level3_cachesize;
}

int get_cpu_powersave()
{
    try_initialize_global_cpu_info();
    return g_powersave;
}

int set_cpu_powersave(int powersave)
{
    try_initialize_global_cpu_info();
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

const CpuSet& get_cpu_thread_affinity_mask(int powersave)
{
    try_initialize_global_cpu_info();
    if (powersave == 0)
        return g_cpu_affinity_mask_all;

    if (powersave == 1)
        return g_cpu_affinity_mask_little;

    if (powersave == 2)
        return g_cpu_affinity_mask_big;

    NCNN_LOGE("powersave %d not supported", powersave);

    // fallback to all cores anyway
    return g_cpu_affinity_mask_all;
}

int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask)
{
    try_initialize_global_cpu_info();
#if defined __ANDROID__ || defined __linux__ || (defined _WIN32 && !(defined __MINGW32__))
#ifdef _OPENMP
    int num_threads = thread_affinity_mask.num_enabled();

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

int is_current_thread_running_on_a53_a55()
{
    try_initialize_global_cpu_info();
#if defined __ANDROID__ || defined __linux__
#if __aarch64__
    if (g_cpu_is_arm_a53_a55 == 0)
        return 0; // all non a53/a55

    if (g_cpu_is_arm_a53_a55 == 1)
        return 1; // all a53/a55

    if (g_powersave == 2)
        return 0; // big clusters

    if (g_powersave == 1)
        return 1; // little clusters

    // little cores are a53/a55

    // use cpuid for retrieving midr since kernel 4.7+
    if (cpu_support_arm_cpuid())
    {
        unsigned int midr = get_midr_from_register();
        if (midr)
            return midr_is_a53_a55(midr);
    }

    // check if affinity cpuid is in the little ones
    CpuSet thread_cs;
    int ret = get_sched_affinity(thread_cs);
    if (ret != 0)
    {
        // no affinity capability
        return 0;
    }

    const CpuSet& little_cs = get_cpu_thread_affinity_mask(1);
    for (int i = 0; i < g_cpucount; i++)
    {
        if (!thread_cs.is_enabled(i))
            continue;

        if (!little_cs.is_enabled(i))
            return 0;
    }

    // all affinity cpuids are little core
    return 1;
#else
    return 0;
#endif // __aarch64__
#else
    return 0;
#endif // defined __ANDROID__ || defined __linux__
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

#if defined __ANDROID__ && defined(_OPENMP) && __clang__
#ifdef __cplusplus
extern "C" {
#endif
void __wrap___kmp_affinity_determine_capable(const char* /*env_var*/)
{
    // the internal affinity routines in llvm openmp call abort on __NR_sched_getaffinity / __NR_sched_setaffinity fails
    // ref KMPNativeAffinity::get_system_affinity/set_system_affinity in openmp/runtime/src/kmp_affinity.h
    // and cpu core goes offline in powersave mode on android, which triggers abort
    // ATM there is no known api for controlling the abort behavior
    // override __kmp_affinity_determine_capable with empty body to disable affinity regardless of KMP_AFFINITY env_var
    // ugly hack works >.<    --- nihui
}
#ifdef __cplusplus
} // extern "C"
#endif
#endif
