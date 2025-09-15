// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_CPU_H
#define NCNN_CPU_H

#include <stddef.h>

#if defined _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifndef NCNN_MAX_CPU_COUNT
#if defined(NCNN_WINDOWS_SERVER)
#define NCNN_MAX_CPU_COUNT 4096
#elif defined(_WIN32_WINNT) && (_WIN32_WINNT >= _WIN32_WINNT_WIN10)
#define NCNN_MAX_CPU_COUNT 512
#elif defined(_WIN32_WINNT) && (_WIN32_WINNT >= _WIN32_WINNT_WIN7)
#define NCNN_MAX_CPU_COUNT 256
#else
#define NCNN_MAX_CPU_COUNT 64
#endif
#endif

#define NCNN_CPU_MASK_GROUPS ((NCNN_MAX_CPU_COUNT + sizeof(ULONG_PTR) * 8 - 1) / (sizeof(ULONG_PTR) * 8))

#endif // _WIN32

#if defined __ANDROID__ || defined __linux__
#include <sched.h> // cpu_set_t
#endif //__ANDROID__ || __linux__

#include "platform.h"

namespace ncnn {

class NCNN_EXPORT CpuSet
{
public:
    CpuSet();
    void enable(int cpu);
    void disable(int cpu);
    void disable_all();
    bool is_enabled(int cpu) const;
    int num_enabled() const;

#if defined _WIN32
    void set_group_mask(int group, ULONG_PTR mask);
    ULONG_PTR get_group_mask(int group) const;
    int get_group_count() const;
#endif // defined _WIN32

public:
#if defined _WIN32
    ULONG_PTR mask_groups[NCNN_CPU_MASK_GROUPS];
#elif defined __ANDROID__ || defined __linux__
    cpu_set_t cpu_set;
#elif __APPLE__
    unsigned int policy;
#else
    int empty;
#endif
};

// test optional cpu features
// edsp = armv7 edsp
NCNN_EXPORT int cpu_support_arm_edsp();
// neon = armv7 neon or aarch64 asimd
NCNN_EXPORT int cpu_support_arm_neon();
// vfpv4 = armv7 fp16 + fma
NCNN_EXPORT int cpu_support_arm_vfpv4();
// asimdhp = aarch64 asimd half precision
NCNN_EXPORT int cpu_support_arm_asimdhp();
// cpuid = aarch64 cpuid info
NCNN_EXPORT int cpu_support_arm_cpuid();
// asimddp = aarch64 asimd dot product
NCNN_EXPORT int cpu_support_arm_asimddp();
// asimdfhm = aarch64 asimd fhm
NCNN_EXPORT int cpu_support_arm_asimdfhm();
// bf16 = aarch64 bf16
NCNN_EXPORT int cpu_support_arm_bf16();
// i8mm = aarch64 i8mm
NCNN_EXPORT int cpu_support_arm_i8mm();
// sve = aarch64 sve
NCNN_EXPORT int cpu_support_arm_sve();
// sve2 = aarch64 sve2
NCNN_EXPORT int cpu_support_arm_sve2();
// svebf16 = aarch64 svebf16
NCNN_EXPORT int cpu_support_arm_svebf16();
// svei8mm = aarch64 svei8mm
NCNN_EXPORT int cpu_support_arm_svei8mm();
// svef32mm = aarch64 svef32mm
NCNN_EXPORT int cpu_support_arm_svef32mm();

// avx = x86 avx
NCNN_EXPORT int cpu_support_x86_avx();
// fma = x86 fma
NCNN_EXPORT int cpu_support_x86_fma();
// xop = x86 xop
NCNN_EXPORT int cpu_support_x86_xop();
// f16c = x86 f16c
NCNN_EXPORT int cpu_support_x86_f16c();
// avx2 = x86 avx2 + fma + f16c
NCNN_EXPORT int cpu_support_x86_avx2();
// avx_vnni = x86 avx vnni
NCNN_EXPORT int cpu_support_x86_avx_vnni();
// avx_vnni_int8 = x86 avx vnni int8
NCNN_EXPORT int cpu_support_x86_avx_vnni_int8();
// avx_vnni_int16 = x86 avx vnni int16
NCNN_EXPORT int cpu_support_x86_avx_vnni_int16();
// avx_ne_convert = x86 avx ne convert
NCNN_EXPORT int cpu_support_x86_avx_ne_convert();
// avx512 = x86 avx512f + avx512cd + avx512bw + avx512dq + avx512vl
NCNN_EXPORT int cpu_support_x86_avx512();
// avx512_vnni = x86 avx512 vnni
NCNN_EXPORT int cpu_support_x86_avx512_vnni();
// avx512_bf16 = x86 avx512 bf16
NCNN_EXPORT int cpu_support_x86_avx512_bf16();
// avx512_fp16 = x86 avx512 fp16
NCNN_EXPORT int cpu_support_x86_avx512_fp16();

// lsx = loongarch lsx
NCNN_EXPORT int cpu_support_loongarch_lsx();
// lasx = loongarch lasx
NCNN_EXPORT int cpu_support_loongarch_lasx();

// msa = mips mas
NCNN_EXPORT int cpu_support_mips_msa();
// mmi = loongson mmi
NCNN_EXPORT int cpu_support_loongson_mmi();

// v = riscv vector
NCNN_EXPORT int cpu_support_riscv_v();
// zfh = riscv half-precision float
NCNN_EXPORT int cpu_support_riscv_zfh();
// zvfh = riscv vector half-precision float
NCNN_EXPORT int cpu_support_riscv_zvfh();
// xtheadvector = riscv xtheadvector
NCNN_EXPORT int cpu_support_riscv_xtheadvector();
// vlenb = riscv vector length in bytes
NCNN_EXPORT int cpu_riscv_vlenb();

// cpu info
NCNN_EXPORT int get_cpu_count();
NCNN_EXPORT int get_little_cpu_count();
NCNN_EXPORT int get_big_cpu_count();

NCNN_EXPORT int get_physical_cpu_count();
NCNN_EXPORT int get_physical_little_cpu_count();
NCNN_EXPORT int get_physical_big_cpu_count();

// cpu l2 varies from 64k to 1M, but l3 can be zero
NCNN_EXPORT int get_cpu_level2_cache_size();
NCNN_EXPORT int get_cpu_level3_cache_size();

// bind all threads on little clusters if powersave enabled
// affects HMP arch cpu like ARM big.LITTLE
// only implemented on android at the moment
// switching powersave is expensive and not thread-safe
// 0 = all cores enabled(default)
// 1 = only little clusters enabled
// 2 = only big clusters enabled
// return 0 if success for setter function
NCNN_EXPORT int get_cpu_powersave();
NCNN_EXPORT int set_cpu_powersave(int powersave);

// convenient wrapper
NCNN_EXPORT const CpuSet& get_cpu_thread_affinity_mask(int powersave);

// set explicit thread affinity
NCNN_EXPORT int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask);

// runtime thread affinity info
NCNN_EXPORT int is_current_thread_running_on_a53_a55();

// misc function wrapper for openmp routines
NCNN_EXPORT int get_omp_num_threads();
NCNN_EXPORT void set_omp_num_threads(int num_threads);

NCNN_EXPORT int get_omp_dynamic();
NCNN_EXPORT void set_omp_dynamic(int dynamic);

NCNN_EXPORT int get_omp_thread_num();

NCNN_EXPORT int get_kmp_blocktime();
NCNN_EXPORT void set_kmp_blocktime(int time_ms);

// need to flush denormals on Intel Chipset.
// Other architectures such as ARM can be added as needed.
// 0 = DAZ OFF, FTZ OFF
// 1 = DAZ ON , FTZ OFF
// 2 = DAZ OFF, FTZ ON
// 3 = DAZ ON,  FTZ ON
NCNN_EXPORT int get_flush_denormals();
NCNN_EXPORT int set_flush_denormals(int flush_denormals);

} // namespace ncnn

#endif // NCNN_CPU_H
