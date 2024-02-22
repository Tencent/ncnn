// SPDX-License-Identifier: MIT
// Copyright (c) 2024 nihui (https://github.com/nihui)
// Copyright (c) 2024 kernelbin (https://github.com/kernelbin)
//
// ruapu --- detect cpu isa features with single-file

#ifndef RUAPU_H
#define RUAPU_H

void ruapu_init();

int ruapu_supports(const char* isa);

#ifdef RUAPU_IMPLEMENTATION

#include <setjmp.h>
#include <string.h>

#if defined _WIN32

#include <windows.h>

#if WINAPI_FAMILY == WINAPI_FAMILY_APP
static int ruapu_detect_isa(const void* some_inst)
{
    // uwp does not support seh  :(
    (void)some_inst;
    return 0;
}
#else // WINAPI_FAMILY == WINAPI_FAMILY_APP
static int g_ruapu_sigill_caught = 0;
static jmp_buf g_ruapu_jmpbuf;

typedef const void* ruapu_some_inst;

static LONG CALLBACK ruapu_catch_sigill(struct _EXCEPTION_POINTERS* ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION)
    {
        g_ruapu_sigill_caught = 1;
        longjmp(g_ruapu_jmpbuf, -1);
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static int ruapu_detect_isa(const void* some_inst)
{
    g_ruapu_sigill_caught = 0;

    PVOID eh = AddVectoredExceptionHandler(1, ruapu_catch_sigill);

    if (setjmp(g_ruapu_jmpbuf) == 0)
    {
        ((void (*)())some_inst)();
    }

    RemoveVectoredExceptionHandler(eh);

    return g_ruapu_sigill_caught ? 0 : 1;
}
#endif // WINAPI_FAMILY == WINAPI_FAMILY_APP

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#ifdef _MSC_VER
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned char ruapu_some_##isa[] = { __VA_ARGS__, 0xc3 };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned char ruapu_some_##isa[] = { __VA_ARGS__, 0xc3 };
#endif

#elif __aarch64__ || defined(_M_ARM64)
#ifdef _MSC_VER
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xd65f03c0 };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xd65f03c0 };
#endif

#elif __arm__ || defined(_M_ARM)
#if __thumb__
#ifdef _MSC_VER
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0x4770 };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0x4770 };
#endif
#else
#ifdef _MSC_VER
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xe12fff1e };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xe12fff1e };
#endif
#endif

#endif

#elif defined __ANDROID__ || defined __linux__ || defined __APPLE__
#include <signal.h>

static int g_ruapu_sigill_caught = 0;
static sigjmp_buf g_ruapu_jmpbuf;

typedef void (*ruapu_some_inst)();

static void ruapu_catch_sigill(int signo, siginfo_t* si, void* data)
{
    (void)signo;
    (void)si;
    (void)data;

    g_ruapu_sigill_caught = 1;
    siglongjmp(g_ruapu_jmpbuf, -1);
}

static int ruapu_detect_isa(ruapu_some_inst some_inst)
{
    g_ruapu_sigill_caught = 0;

    struct sigaction sa = { 0 };
    struct sigaction old_sa;
    sa.sa_flags = SA_ONSTACK | SA_RESTART | SA_SIGINFO;
    sa.sa_sigaction = ruapu_catch_sigill;
    sigaction(SIGILL, &sa, &old_sa);

    if (sigsetjmp(g_ruapu_jmpbuf, 1) == 0)
    {
        some_inst();
    }

    sigaction(SIGILL, &old_sa, NULL);

    return g_ruapu_sigill_caught ? 0 : 1;
}

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".byte " #__VA_ARGS__ : : : ); }
#elif __aarch64__
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".word " #__VA_ARGS__ : : : ); }
#elif __arm__
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".word " #__VA_ARGS__ : : : ); }
#endif

#else // defined _WIN32 || defined __ANDROID__ || defined __linux__ || defined __APPLE__
typedef const void* ruapu_some_inst;
static int ruapu_detect_isa(const void* some_inst)
{
    // unknown platform, bare metal os ?
    (void)some_inst;
    return 0;
}

#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { }
#endif // defined _WIN32 || defined __ANDROID__ || defined __linux__ || defined __APPLE__

struct ruapu_isa_entry
{
    const char* isa;
    ruapu_some_inst inst;
    int capable;
};

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
RUAPU_INSTCODE(mmx, 0x0f, 0xdb, 0xc0) // pand mm0,mm0
RUAPU_INSTCODE(sse, 0x0f, 0x54, 0xc0) // andps xmm0,xmm0
RUAPU_INSTCODE(sse2, 0x66, 0x0f, 0xfe, 0xc0) // paddd xmm0,xmm0
RUAPU_INSTCODE(sse3, 0xf2, 0x0f, 0x7c, 0xc0) // haddps xmm0,xmm0
RUAPU_INSTCODE(ssse3, 0x66, 0x0f, 0x38, 0x06, 0xc0) // phsubd xmm0,xmm0
RUAPU_INSTCODE(sse41, 0x66, 0x0f, 0x38, 0x3d, 0xc0) // pmaxsd xmm0,xmm0
RUAPU_INSTCODE(sse42, 0x66, 0x0f, 0x38, 0x37, 0xc0) // pcmpgtq xmm0,xmm0
RUAPU_INSTCODE(sse4a, 0x66, 0x0f, 0x79, 0xc0) // extrq xmm0,xmm0
RUAPU_INSTCODE(xop, 0x8f, 0xe8, 0x78, 0xb6, 0xc0, 0x00)  // vpmadcswd xmm0,xmm0,xmm0,xmm0
RUAPU_INSTCODE(avx, 0xc5, 0xfc, 0x54, 0xc0) // vandps ymm0,ymm0,ymm0
RUAPU_INSTCODE(f16c, 0xc4, 0xe2, 0x7d, 0x13, 0xc0) // vcvtph2ps ymm0,xmm0
RUAPU_INSTCODE(fma, 0xc4, 0xe2, 0x7d, 0x98, 0xc0) // vfmadd132ps ymm0,ymm0,ymm0
RUAPU_INSTCODE(fma4, 0xc4, 0xe3, 0xfd, 0x68, 0xc0, 0x00) // vfmaddps ymm0,ymm0,ymm0,ymm0
RUAPU_INSTCODE(avx2, 0xc5, 0xfd, 0xfe, 0xc0) // vpaddd ymm0,ymm0,ymm0
RUAPU_INSTCODE(avx512f, 0x62, 0xf1, 0x7c, 0x48, 0x58, 0xc0) // vaddps zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512bw, 0x62, 0xf1, 0x7d, 0x48, 0xfd, 0xc0) // vpaddw zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512cd, 0x62, 0xf2, 0xfd, 0x48, 0x44, 0xc0) // vplzcntq zmm0,zmm0
RUAPU_INSTCODE(avx512dq, 0x62, 0xf1, 0x7c, 0x48, 0x54, 0xc0) // vandps zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512vl, 0x62, 0xf2, 0xfd, 0x28, 0x1f, 0xc0) // vpabsq ymm0,ymm0
RUAPU_INSTCODE(avx512vnni, 0x62, 0xf2, 0x7d, 0x48, 0x52, 0xc0) // vpdpwssd zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512bf16, 0x62, 0xf2, 0x7e, 0x48, 0x52, 0xc0) // vdpbf16ps zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512ifma, 0x62, 0xf2, 0xfd, 0x48, 0xb4, 0xc0) // vpmadd52luq zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512vbmi, 0x62, 0xf2, 0x7d, 0x48, 0x75, 0xc0) // vpermi2b zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512vbmi2, 0x62, 0xf2, 0x7d, 0x48, 0x71, 0xc0) // vpshldvd zmm0,zmm0,zmm0
RUAPU_INSTCODE(avx512fp16, 0x62, 0xf6, 0x7d, 0x48, 0x98, 0xc0) // vfmadd132ph zmm0,zmm0,zmm0
RUAPU_INSTCODE(avxvnni, 0xc4, 0xe2, 0x7d, 0x52, 0xc0) // vpdpwssd ymm0,ymm0,ymm0
RUAPU_INSTCODE(avxvnniint8, 0xc4, 0xe2, 0x7f, 0x50, 0xc0) // vpdpbssd ymm0,ymm0,ymm0
RUAPU_INSTCODE(avxifma, 0xc4, 0xe2, 0xfd, 0xb4, 0xc0) // vpmadd52luq ymm0,ymm0,ymm0

#elif __aarch64__ || defined(_M_ARM64)
RUAPU_INSTCODE(neon, 0x4e20d400) // fadd v0.4s,v0.4s,v0.4s
RUAPU_INSTCODE(vfpv4, 0x0e216800) // fcvtn v0.4h,v0.4s
RUAPU_INSTCODE(cpuid, 0xd5380000) // mrs x0,midr_el1
RUAPU_INSTCODE(asimdhp, 0x0e401400) // fadd v0.4h,v0.4h,v0.4h
RUAPU_INSTCODE(asimddp, 0x4e809400) // sdot v0.4h,v0.16b,v0.16b
RUAPU_INSTCODE(asimdfhm, 0x4e20ec00) // fmlal v0.4s,v0.4h,v0.4h
RUAPU_INSTCODE(bf16, 0x6e40ec00) // bfmmla v0.4h,v0.8h,v0.8h
RUAPU_INSTCODE(i8mm, 0x4e80a400) // smmla v0.4h,v0.16b,v0.16b
RUAPU_INSTCODE(sve, 0x65608000) // fmad z0.h,p0/m,z0.h,z0.h
RUAPU_INSTCODE(sve2, 0x44405000) // smlslb z0.h,z0.b,z0.b
RUAPU_INSTCODE(svebf16, 0x6460e400) // bfmmla z0.s,z0.h,z0.h
RUAPU_INSTCODE(svei8mm, 0x45009800) // smmla z0.s,z0.b,z0.b
RUAPU_INSTCODE(svef32mm, 0x64a0e400) // fmmla z0.s,z0.s,z0.s

#elif __arm__ || defined(_M_ARM)
#if __thumb__
RUAPU_INSTCODE(edsp, 0xfb20, 0x0000) // smlad r0,r0,r0,r0
RUAPU_INSTCODE(neon, 0xef00, 0x0d40) // vadd.f32 q0,q0,q0
RUAPU_INSTCODE(vfpv4, 0xffb6, 0x0600) // vcvt.f16.f32 d0,q0
#else
RUAPU_INSTCODE(edsp, 0xe7000010) // smlad r0,r0,r0,r0
RUAPU_INSTCODE(neon, 0xf2000d40) // vadd.f32 q0,q0,q0
RUAPU_INSTCODE(vfpv4, 0xf3b60600) // vcvt.f16.f32 d0,q0
#endif

#endif

#undef RUAPU_INSTCODE

#define RUAPU_ISAENTRY(isa) { #isa, (ruapu_some_inst)ruapu_some_##isa, 0 },

struct ruapu_isa_entry g_ruapu_isa_map[] = {

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
RUAPU_ISAENTRY(mmx)
RUAPU_ISAENTRY(sse)
RUAPU_ISAENTRY(sse2)
RUAPU_ISAENTRY(sse3)
RUAPU_ISAENTRY(ssse3)
RUAPU_ISAENTRY(sse41)
RUAPU_ISAENTRY(sse42)
RUAPU_ISAENTRY(sse4a)
RUAPU_ISAENTRY(xop)
RUAPU_ISAENTRY(avx)
RUAPU_ISAENTRY(f16c)
RUAPU_ISAENTRY(fma)
RUAPU_ISAENTRY(fma4)
RUAPU_ISAENTRY(avx2)
RUAPU_ISAENTRY(avx512f)
RUAPU_ISAENTRY(avx512bw)
RUAPU_ISAENTRY(avx512cd)
RUAPU_ISAENTRY(avx512dq)
RUAPU_ISAENTRY(avx512vl)
RUAPU_ISAENTRY(avx512vnni)
RUAPU_ISAENTRY(avx512bf16)
RUAPU_ISAENTRY(avx512ifma)
RUAPU_ISAENTRY(avx512vbmi)
RUAPU_ISAENTRY(avx512vbmi2)
RUAPU_ISAENTRY(avx512fp16)
RUAPU_ISAENTRY(avxvnni)
RUAPU_ISAENTRY(avxvnniint8)
RUAPU_ISAENTRY(avxifma)

#elif __aarch64__ || defined(_M_ARM64)
RUAPU_ISAENTRY(neon)
RUAPU_ISAENTRY(vfpv4)
RUAPU_ISAENTRY(cpuid)
RUAPU_ISAENTRY(asimdhp)
RUAPU_ISAENTRY(asimddp)
RUAPU_ISAENTRY(asimdfhm)
RUAPU_ISAENTRY(bf16)
RUAPU_ISAENTRY(i8mm)
RUAPU_ISAENTRY(sve)
RUAPU_ISAENTRY(sve2)
RUAPU_ISAENTRY(svebf16)
RUAPU_ISAENTRY(svei8mm)
RUAPU_ISAENTRY(svef32mm)

#elif __arm__ || defined(_M_ARM)
RUAPU_ISAENTRY(edsp)
RUAPU_ISAENTRY(neon)
RUAPU_ISAENTRY(vfpv4)

#endif
};

#undef RUAPU_ISAENTRY

void ruapu_init()
{
    for (size_t i = 0; i < sizeof(g_ruapu_isa_map) / sizeof(g_ruapu_isa_map[0]); i++)
    {
        g_ruapu_isa_map[i].capable = ruapu_detect_isa(g_ruapu_isa_map[i].inst);
    }
}

int ruapu_supports(const char* isa)
{
    for (size_t i = 0; i < sizeof(g_ruapu_isa_map) / sizeof(g_ruapu_isa_map[0]); i++)
    {
        if (strcmp(g_ruapu_isa_map[i].isa, isa) == 0)
        {
            return g_ruapu_isa_map[i].capable;
        }
    }

    return 0;
}

#endif // RUAPU_IMPLEMENTATION

#endif // RUAPU_H
