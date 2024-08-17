// SPDX-License-Identifier: MIT
// Copyright (c) 2024 nihui (https://github.com/nihui)
// Copyright (c) 2024 kernelbin (https://github.com/kernelbin)
//
// ruapu --- detect cpu isa features with single-file

#ifndef RUAPU_H
#define RUAPU_H

#ifdef __cplusplus
extern "C" {
#endif

void ruapu_init();

int ruapu_supports(const char* isa);

const char* const* ruapu_rua();

#ifdef RUAPU_IMPLEMENTATION

#include <stdint.h>
#include <string.h>

typedef void (*ruapu_some_inst)();

#if defined _WIN32

#include <windows.h>
#include <setjmp.h>

#if defined (_MSC_VER) // MSVC
static int ruapu_detect_isa(ruapu_some_inst some_inst)
{
    int g_ruapu_sigill_caught = 0;

    __try
    {
        some_inst();
    }
    __except (GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION ?
        EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
        g_ruapu_sigill_caught = 1;
    }

    return g_ruapu_sigill_caught ? 0 : 1;
}
#else
static int g_ruapu_sigill_caught = 0;
static jmp_buf g_ruapu_jmpbuf;

static LONG CALLBACK ruapu_catch_sigill(struct _EXCEPTION_POINTERS* ExceptionInfo)
{
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION)
    {
        g_ruapu_sigill_caught = 1;
        longjmp(g_ruapu_jmpbuf, -1);
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static int ruapu_detect_isa(ruapu_some_inst some_inst)
{
    g_ruapu_sigill_caught = 0;

    PVOID eh = AddVectoredExceptionHandler(1, ruapu_catch_sigill);

    if (setjmp(g_ruapu_jmpbuf) == 0)
    {
        some_inst();
    }

    RemoveVectoredExceptionHandler(eh);

    return g_ruapu_sigill_caught ? 0 : 1;
}
#endif // WINAPI_FAMILY == WINAPI_FAMILY_APP

#elif defined __ANDROID__ || defined __linux__ || defined __APPLE__ || defined __FreeBSD__ || defined __NetBSD__ || defined __OpenBSD__ || defined __DragonFly__ || defined __sun__
#include <signal.h>
#include <setjmp.h>

static int g_ruapu_sigill_caught = 0;
static sigjmp_buf g_ruapu_jmpbuf;

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

#elif defined __SYTERKIT__

#include <mmu.h>

static int g_ruapu_sigill_caught = 0;

void arm32_do_undefined_instruction(struct arm_regs_t *regs)
{
    g_ruapu_sigill_caught = 1;
    regs->pc += 4;
}

static int ruapu_detect_isa(ruapu_some_inst some_inst)
{
    g_ruapu_sigill_caught = 0;
    some_inst();
    return g_ruapu_sigill_caught ? 0 : 1;
}

#endif // defined _WIN32 || defined __ANDROID__ || defined __linux__ || defined __APPLE__ || defined __FreeBSD__ || defined __NetBSD__ || defined __OpenBSD__ || defined __DragonFly__ || defined __sun__ || defined __SYTERKIT__

#if defined _WIN32

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
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned short ruapu_some_##isa[] = { __VA_ARGS__, 0x4770 };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned short ruapu_some_##isa[] = { __VA_ARGS__, 0x4770 };
#endif
#else
#ifdef _MSC_VER
#define RUAPU_INSTCODE(isa, ...) __pragma(section(".text")) __declspec(allocate(".text")) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xe12fff1e };
#else
#define RUAPU_INSTCODE(isa, ...) __attribute__((section(".text"))) static unsigned int ruapu_some_##isa[] = { __VA_ARGS__, 0xe12fff1e };
#endif
#endif

#endif

#else // defined _WIN32

#if defined(__i386__) || defined(__x86_64__) || __s390x__
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".byte " #__VA_ARGS__ : : : ); }
#elif __aarch64__ || __arm__ || __mips__ || __riscv || __loongarch__
#if __thumb__
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".short " #__VA_ARGS__ : : : ); }
#else
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".word " #__VA_ARGS__ : : : ); }
#endif
#elif __powerpc__
#define RUAPU_INSTCODE(isa, ...) static void ruapu_some_##isa() { asm volatile(".long " #__VA_ARGS__ : : : ); }
#endif

#endif // defined _WIN32

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
RUAPU_INSTCODE(mmx, 0x0f, 0xdb, 0xc0, 0x0f, 0x77) // pand mm0,mm0 + emms
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
// TODO:avx512pf, vgatherpf1dps DWORD PTR [esp+zmm0*1]{k1}
RUAPU_INSTCODE(avx512er, 0x62, 0xf2, 0xfd, 0x48, 0xc8, 0xc0) //vexp2pd zmm0,zmm0
RUAPU_INSTCODE(avx5124fmaps, 0x67, 0x62, 0xf2, 0x7f, 0x48, 0x9a, 0x04, 0x24) //v4fmaddps zmm0,zmm0,XMMWORD PTR [esp]
RUAPU_INSTCODE(avx5124vnniw, 0x67, 0x62, 0xf2, 0x7f, 0x48, 0x52, 0x04, 0x24) //vp4dpwssd zmm0,zmm0,XMMWORD PTR [esp]
RUAPU_INSTCODE(avxvnni, 0xc4, 0xe2, 0x7d, 0x52, 0xc0) // vpdpwssd ymm0,ymm0,ymm0
RUAPU_INSTCODE(avxvnniint8, 0xc4, 0xe2, 0x7f, 0x50, 0xc0) // vpdpbssd ymm0,ymm0,ymm0
RUAPU_INSTCODE(avxvnniint16, 0xc4, 0xe2, 0x7e, 0xd2, 0xc0) // vpdpwsud ymm0,ymm0,ymm0
RUAPU_INSTCODE(avxifma, 0xc4, 0xe2, 0xfd, 0xb4, 0xc0) // vpmadd52luq ymm0,ymm0,ymm0
RUAPU_INSTCODE(amxfp16, 0xc4, 0xe2, 0x7b, 0x5c, 0xd1) // tdpfp16ps %tmm0, %tmm1, %tmm2
RUAPU_INSTCODE(amxbf16, 0xc4, 0xe2, 0x7a, 0x5c, 0xd1) // tdpbf16ps %tmm0, %tmm1, %tmm2
RUAPU_INSTCODE(amxint8, 0xc4, 0xe2, 0x7b, 0x5e, 0xd1) // tdpbssd %tmm0, %tmm1, %tmm2
RUAPU_INSTCODE(amxtile, 0xc4, 0xe2, 0x7a, 0x49, 0xc0) // tilezero %tmm0

#elif __aarch64__ || defined(_M_ARM64)
RUAPU_INSTCODE(neon, 0x4e20d400) // fadd v0.4s,v0.4s,v0.4s
RUAPU_INSTCODE(vfpv4, 0x1f000000) // fmadd s0,s0,s0,s0
RUAPU_INSTCODE(lse, 0xf82083e0, 0xf82083e0) // swp x0,x0,[sp] + swp x0,x0,[sp]
RUAPU_INSTCODE(cpuid, 0xd5380000) // mrs x0,midr_el1
RUAPU_INSTCODE(asimdrdm, 0x6e808400) // sqrdmlah v0.4s,v0.4s,v0.4s
RUAPU_INSTCODE(asimdhp, 0x0e401400) // fadd v0.4h,v0.4h,v0.4h
RUAPU_INSTCODE(asimddp, 0x4e809400) // sdot v0.4h,v0.16b,v0.16b
RUAPU_INSTCODE(asimdfhm, 0x4e20ec00) // fmlal v0.4s,v0.4h,v0.4h
RUAPU_INSTCODE(bf16, 0x6e40ec00) // bfmmla v0.4h,v0.8h,v0.8h
RUAPU_INSTCODE(i8mm, 0x4e80a400) // smmla v0.4h,v0.16b,v0.16b
RUAPU_INSTCODE(frint, 0x4e21e800) // frint32z v0.4s,v0.4s
RUAPU_INSTCODE(jscvt, 0x1e7e0000) // fjcvtzs w0,d0
RUAPU_INSTCODE(fcma, 0x6e80c400) // fcmla v0.4s,v0.4s,v0.4s,#0
RUAPU_INSTCODE(mte, 0xd96003e0) // ldg x0,[sp]
RUAPU_INSTCODE(mte2, 0xd9e003e0) // ldgm x0,[sp]
RUAPU_INSTCODE(sve, 0x65608000) // fmad z0.h,p0/m,z0.h,z0.h
RUAPU_INSTCODE(sve2, 0x44405000) // smlslb z0.h,z0.b,z0.b
RUAPU_INSTCODE(svebf16, 0x6460e400) // bfmmla z0.s,z0.h,z0.h
RUAPU_INSTCODE(svei8mm, 0x45009800) // smmla z0.s,z0.b,z0.b
RUAPU_INSTCODE(svef32mm, 0x64a0e400) // fmmla z0.s,z0.s,z0.s
RUAPU_INSTCODE(svef64mm, 0x64e0e400) // fmmla z0.d,z0.d,z0.d
RUAPU_INSTCODE(sme, 0x80800000) // fmopa za0.s,p0/m,p0/m,z0.s,z0.s
RUAPU_INSTCODE(smef16f16, 0x81800008) // fmopa za0.h,p0/m,p0/m,z0.h,z0.h
RUAPU_INSTCODE(smef64f64, 0x80c00000) // fmopa za0.d,p0/m,p0/m,z0.d,z0.d
RUAPU_INSTCODE(smei64i64, 0xa0c00000) // smopa za0.d,p0/m,p0/m,z0.h,z0.h
RUAPU_INSTCODE(pmull, 0x0e20e000) // pmull v0.8h,v0.8b,v0.8b
RUAPU_INSTCODE(crc32, 0x1ac04000) // crc32b w0,w0,w0
RUAPU_INSTCODE(aes, 0x4e285800) // aesd v0.16b,v0.16b
RUAPU_INSTCODE(sha1, 0x5e280800) // sha1h s0,s0
RUAPU_INSTCODE(sha2, 0x5e004000) // sha256h q0,q0,v0.4s
RUAPU_INSTCODE(sha3, 0xce000000) // eor3 v0.16b, v0.16b, v0.16b, v0.16b
RUAPU_INSTCODE(sha512, 0xce608000) // sha512h q0, q0, v0.2d
RUAPU_INSTCODE(sm3, 0xce60c000) // sm3partw1 v0.4s, v0.4s, v0.4s
RUAPU_INSTCODE(sm4, 0xcec08400) // sm4e v0.4s, v0.4s
RUAPU_INSTCODE(svepmull, 0x45006800) // pmullb z0.q,z0.d,z0.d
RUAPU_INSTCODE(svebitperm, 0x4500b000) // bext z0.b,z0.b,z0.b
RUAPU_INSTCODE(sveaes, 0x4522e400) // aesd z0.b,z0.b,z0.b
RUAPU_INSTCODE(svesha3, 0x4520f400) // rax1 z0.d,z0.d,z0.d
RUAPU_INSTCODE(svesm4, 0x4523e000) // sm4e z0.s,z0.s,z0.s
RUAPU_INSTCODE(amx, 0x00201220) // amx setup


#elif __arm__ || defined(_M_ARM)
#if __thumb__
RUAPU_INSTCODE(half, 0xf8bd, 0x0000) // ldrh r0,[sp]
RUAPU_INSTCODE(edsp, 0xfb20, 0x0000) // smlad r0,r0,r0,r0
RUAPU_INSTCODE(neon, 0xef00, 0x0d40) // vadd.f32 q0,q0,q0
RUAPU_INSTCODE(vfpv4, 0xeea0, 0x0a00) // vfma.f32 s0,s0,s0
RUAPU_INSTCODE(idiv, 0x2003, 0xfb90, 0xf0f0) // movs r0,#3 + sdiv r0,r0,r0
#else
RUAPU_INSTCODE(half, 0xe1dd00b0) // ldrh r0,[sp]
RUAPU_INSTCODE(edsp, 0xe7000010) // smlad r0,r0,r0,r0
RUAPU_INSTCODE(neon, 0xf2000d40) // vadd.f32 q0,q0,q0
RUAPU_INSTCODE(vfpv4, 0xeea00a00) // vfma.f32 s0,s0,s0
RUAPU_INSTCODE(idiv, 0xe3a00003, 0xe710f010) // movs r0,#3 + sdiv r0,r0,r0
#endif

#elif __mips__
RUAPU_INSTCODE(msa, 0x7900001b) // fmadd.w $w0,$w0,$w0
RUAPU_INSTCODE(mmi, 0x4b60000e) // pmaddhw $f0,$f0
RUAPU_INSTCODE(sx, 0xef48001e) // __lsx_vffloor_w
RUAPU_INSTCODE(asx, 0xec40001d) // __lasx_xfmadd_w
RUAPU_INSTCODE(msa2, 0x78000008) // __msa2_vperm_b
RUAPU_INSTCODE(crypto, 0x78010017) // __crypto_aes128_dec

#elif __powerpc__
RUAPU_INSTCODE(vsx, 0x104210c0) // vaddudm v2,v2,v2

#elif __s390x__
RUAPU_INSTCODE(zvector, 0xe7, 0x11, 0x12, 0x00, 0x10, 0x8f) // vfmasb v1,v1,v1,v1

#elif __loongarch__
RUAPU_INSTCODE(lsx, 0x700b0000) //vadd.w vr0, vr0, vr0
RUAPU_INSTCODE(lasx, 0x740b0000) //xvadd.w xr0, xr0, xr0

#elif __riscv
RUAPU_INSTCODE(i, 0x00a50533) // add a0,a0,a0
RUAPU_INSTCODE(m, 0x00200513, 0x02a50533, 0x02a54533) // addi a0,x0,2 mul a0,a0,a0 div a0,a0,a0
RUAPU_INSTCODE(a, 0x100122af, 0x185122af) // lr.w t0,(sp) + sc.w t0,t0,(sp)
RUAPU_INSTCODE(f, 0x10a57553) // fmul.s fa0,fa0,fa0
RUAPU_INSTCODE(d, 0x12a57553) // fmul.d fa0,fa0,fa0
RUAPU_INSTCODE(c, 0x0001952a) // add a0,a0,a0 + nop
RUAPU_INSTCODE(zba, 0x20a52533) // sh1add a0,a0,a0
RUAPU_INSTCODE(zbb, 0x60451513) // sext.b a0,a0,a0
RUAPU_INSTCODE(zbc, 0x0aa52533) // clmulr a0,a0,a0
RUAPU_INSTCODE(zbs, 0x48a51533) // bclr a0,a0,a0
RUAPU_INSTCODE(zbkb, 0x08a54533) // pack a0,a0,a0
RUAPU_INSTCODE(zbkc, 0x0aa53533) // clmulh a0,a0,a0
RUAPU_INSTCODE(zbkx, 0x28a52533) // xperm.n a0,a0,a0
RUAPU_INSTCODE(zfa, 0xf0108053) // fli.s ft0, min
RUAPU_INSTCODE(zfbfmin, 0x44807053) // fcvt.bf16.s ft0,ft0
RUAPU_INSTCODE(zfh, 0x04007053); // fadd.hs ft0, ft0, ft0
RUAPU_INSTCODE(zfhmin, 0xe4000553) // fmv.x.h a0, ft0
RUAPU_INSTCODE(zicond, 0x0ea55533) // czero.eqz a0,a0,a0
RUAPU_INSTCODE(zicsr, 0xc0102573); // csrr a0, time
RUAPU_INSTCODE(zifencei, 0x0000100f); // fence.i
RUAPU_INSTCODE(zmmul, 0x02a50533) // mul a0,a0,a0

RUAPU_INSTCODE(xtheadba, 0x00a5150b) // th.addsl a0,a0,a0,#0
RUAPU_INSTCODE(xtheadbb, 0x1005150b) // th.srri a0,a0,#0
RUAPU_INSTCODE(xtheadbs, 0x8805150b) // th.tst a0,a0,#0
RUAPU_INSTCODE(xtheadcondmov, 0x40a5150b) // th.mveqz a0,a0,a0
RUAPU_INSTCODE(xtheadfmemidx, 0x40a1650b) // th.flrw a0,sp,a0,#0
RUAPU_INSTCODE(xtheadfmv, 0xc005150b) // th.fmv.x.hw a0,fa0
RUAPU_INSTCODE(xtheadmac, 0x20a5150b) // th.mula a0,a0,a0
RUAPU_INSTCODE(xtheadmemidx, 0x1801450b) // th.lbia a0,(sp),#0,#0
RUAPU_INSTCODE(xtheadmempair, 0xe0a1450b) // th.lwd a0,a0,(sp),#0,3
RUAPU_INSTCODE(xtheadsync, 0x0180000b) // th.sync
RUAPU_INSTCODE(xtheadvdot, 0x8000600b) // th.vmaqa.vv v0,v0,v0

// RVV 1.0 support
// unimp (csrrw x0, cycle, x0)
#define RUAPU_RV_TRAP() asm volatile(".align 2\n.word 0xc0001073")
// vcsr is only defined in rvv 1.0, which doesn't exist in rvv 0.7.1 or xtheadvector.
// csrr x0, vcsr
#define RUAPU_RVV1P0_AVAIL() asm volatile(".align 2\n.word 0x00f02573")
// csrr res, vlenb
#define RUAPU_DETECT_ZVL(len) static void ruapu_some_zvl##len##b() { \
        RUAPU_RVV1P0_AVAIL(); \
        intptr_t res; \
        asm volatile(".align 2\n.insn i 0x73, 0x2, %0, x0, -990" : "=r"(res)); \
        if (res < len/8) RUAPU_RV_TRAP(); \
    }
RUAPU_DETECT_ZVL(32)
RUAPU_DETECT_ZVL(64)
RUAPU_DETECT_ZVL(128)
RUAPU_DETECT_ZVL(256)
RUAPU_DETECT_ZVL(512)
RUAPU_DETECT_ZVL(1024)
#undef RUAPU_DETECT_ZVL
// vsetvl res, zero, vtype
// check vill bits after vsetvl
#define RUAPU_RVV_INSTCODE(isa, vtype, ...) static void ruapu_some_##isa() { \
        RUAPU_RVV1P0_AVAIL(); \
        intptr_t res; \
        asm volatile(".align 2\n.insn r 0x57, 0x7, 0x40, %0, x0, %1" : "=r"(res) : "r"(vtype)); \
        if (res < 0) RUAPU_RV_TRAP(); \
        asm volatile(".align 2\n.word " #__VA_ARGS__ ); \
    }

RUAPU_RVV_INSTCODE(zvbb, 0, 0x4a862257) // vclz.v v4, v8 with SEW = 8
RUAPU_RVV_INSTCODE(zvbc, 0, 0x32842257) // vclmul.vv v4, v8, v8 with SEW = 8
RUAPU_RVV_INSTCODE(zvfh, 8, 0x02841257) // vfadd.vv v4, v8, v8 with SEW = 16
RUAPU_RVV_INSTCODE(zvfhmin, 8, 0x4a8a1257) // vfncvt.f.f.v v4, v8 with SEW = 16
RUAPU_RVV_INSTCODE(zvfbfmin, 8, 0x4a8e9257) // vfncvtbf16.f.f.w v4, v8 with SEW = 16
RUAPU_RVV_INSTCODE(zvfbfwma, 8, 0xee855257) // vfwmaccbf16.vf v4, fa0, v8 with SEW = 16
RUAPU_RVV_INSTCODE(zvkb, 0, 0x56860257) // vrol.vv v4, v8, v12 with SEW = 8
RUAPU_RVV_INSTCODE(v, 24, 0x22842257) // vaaddu.vv v4, v8, v8 with SEW = 64

#undef RUAPU_RVV_INSTCODE
#undef RUAPU_RV_TRAP
#undef RUAPU_RVV1P0_AVAIL
#endif

#undef RUAPU_INSTCODE

struct ruapu_isa_entry
{
    const char* isa;
    ruapu_some_inst inst;
};

#define RUAPU_ISAENTRY(isa) { #isa, (ruapu_some_inst)(void*)ruapu_some_##isa },

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
// TODO:avx512pf
RUAPU_ISAENTRY(avx512er)
RUAPU_ISAENTRY(avx5124fmaps)
RUAPU_ISAENTRY(avx5124vnniw)
RUAPU_ISAENTRY(avxvnni)
RUAPU_ISAENTRY(avxvnniint8)
RUAPU_ISAENTRY(avxvnniint16)
RUAPU_ISAENTRY(avxifma)
RUAPU_ISAENTRY(amxfp16)
RUAPU_ISAENTRY(amxbf16)
RUAPU_ISAENTRY(amxint8)
RUAPU_ISAENTRY(amxtile)

#elif __aarch64__ || defined(_M_ARM64)
RUAPU_ISAENTRY(neon)
RUAPU_ISAENTRY(vfpv4)
RUAPU_ISAENTRY(lse)
RUAPU_ISAENTRY(cpuid)
RUAPU_ISAENTRY(asimdrdm)
RUAPU_ISAENTRY(asimdhp)
RUAPU_ISAENTRY(asimddp)
RUAPU_ISAENTRY(asimdfhm)
RUAPU_ISAENTRY(bf16)
RUAPU_ISAENTRY(i8mm)
RUAPU_ISAENTRY(frint)
RUAPU_ISAENTRY(jscvt)
RUAPU_ISAENTRY(fcma)
RUAPU_ISAENTRY(mte)
RUAPU_ISAENTRY(mte2)
RUAPU_ISAENTRY(sve)
RUAPU_ISAENTRY(sve2)
RUAPU_ISAENTRY(svebf16)
RUAPU_ISAENTRY(svei8mm)
RUAPU_ISAENTRY(svef32mm)
RUAPU_ISAENTRY(svef64mm)
RUAPU_ISAENTRY(sme)
RUAPU_ISAENTRY(smef16f16)
RUAPU_ISAENTRY(smef64f64)
RUAPU_ISAENTRY(smei64i64)
RUAPU_ISAENTRY(pmull)
RUAPU_ISAENTRY(crc32)
RUAPU_ISAENTRY(aes)
RUAPU_ISAENTRY(sha1)
RUAPU_ISAENTRY(sha2)
RUAPU_ISAENTRY(sha3)
RUAPU_ISAENTRY(sha512)
RUAPU_ISAENTRY(sm3)
RUAPU_ISAENTRY(sm4)
RUAPU_ISAENTRY(svepmull)
RUAPU_ISAENTRY(svebitperm)
RUAPU_ISAENTRY(sveaes)
RUAPU_ISAENTRY(svesha3)
RUAPU_ISAENTRY(svesm4)
RUAPU_ISAENTRY(amx)

#elif __arm__ || defined(_M_ARM)
RUAPU_ISAENTRY(half)
RUAPU_ISAENTRY(edsp)
RUAPU_ISAENTRY(neon)
RUAPU_ISAENTRY(vfpv4)
RUAPU_ISAENTRY(idiv)

#elif __mips__
RUAPU_ISAENTRY(msa)
RUAPU_ISAENTRY(mmi)
RUAPU_ISAENTRY(sx)
RUAPU_ISAENTRY(asx)
RUAPU_ISAENTRY(msa2)
RUAPU_ISAENTRY(crypto)

#elif __powerpc__
RUAPU_ISAENTRY(vsx)

#elif __s390x__
RUAPU_ISAENTRY(zvector)

#elif __loongarch__
RUAPU_ISAENTRY(lsx)
RUAPU_ISAENTRY(lasx)

#elif __riscv
RUAPU_ISAENTRY(i)
RUAPU_ISAENTRY(m)
RUAPU_ISAENTRY(a)
RUAPU_ISAENTRY(f)
RUAPU_ISAENTRY(d)
RUAPU_ISAENTRY(c)
RUAPU_ISAENTRY(v)
RUAPU_ISAENTRY(zba)
RUAPU_ISAENTRY(zbb)
RUAPU_ISAENTRY(zbc)
RUAPU_ISAENTRY(zbs)
RUAPU_ISAENTRY(zbkb)
RUAPU_ISAENTRY(zbkc)
RUAPU_ISAENTRY(zbkx)
RUAPU_ISAENTRY(zfa)
RUAPU_ISAENTRY(zfbfmin)
RUAPU_ISAENTRY(zfh)
RUAPU_ISAENTRY(zfhmin)
RUAPU_ISAENTRY(zicond)
RUAPU_ISAENTRY(zicsr)
RUAPU_ISAENTRY(zifencei)
RUAPU_ISAENTRY(zmmul)
RUAPU_ISAENTRY(zvbb)
RUAPU_ISAENTRY(zvbc)
RUAPU_ISAENTRY(zvfh)
RUAPU_ISAENTRY(zvfhmin)
RUAPU_ISAENTRY(zvfbfmin)
RUAPU_ISAENTRY(zvfbfwma)
RUAPU_ISAENTRY(zvkb)
RUAPU_ISAENTRY(zvl32b)
RUAPU_ISAENTRY(zvl64b)
RUAPU_ISAENTRY(zvl128b)
RUAPU_ISAENTRY(zvl256b)
RUAPU_ISAENTRY(zvl512b)
RUAPU_ISAENTRY(zvl1024b)

RUAPU_ISAENTRY(xtheadba)
RUAPU_ISAENTRY(xtheadbb)
RUAPU_ISAENTRY(xtheadbs)
RUAPU_ISAENTRY(xtheadcondmov)
RUAPU_ISAENTRY(xtheadfmemidx)
RUAPU_ISAENTRY(xtheadfmv)
RUAPU_ISAENTRY(xtheadmac)
RUAPU_ISAENTRY(xtheadmemidx)
RUAPU_ISAENTRY(xtheadmempair)
RUAPU_ISAENTRY(xtheadsync)
RUAPU_ISAENTRY(xtheadvdot)

#elif __openrisc__
RUAPU_ISAENTRY(orbis32)
RUAPU_ISAENTRY(orbis64)
RUAPU_ISAENTRY(orfpx32)
RUAPU_ISAENTRY(orfpx64)
RUAPU_ISAENTRY(orvdx64)

#endif
};

#undef RUAPU_ISAENTRY

const char* g_ruapu_isa_supported[sizeof(g_ruapu_isa_map) / sizeof(g_ruapu_isa_map[0]) + 1] = { 0 };

#if defined __openrisc__
static void ruapu_detect_openrisc_isa()
{
    uint32_t value;
    uint16_t addr = U(0x0000);
    asm volatile ("l.mfspr %0, r0, %1" : "=r" (value) : "K" (addr));
    size_t j = 0;
    for (size_t i = 0; i < sizeof(g_ruapu_isa_map) / sizeof(g_ruapu_isa_map[0]); i++)
    {
        int capable = ((value) >> (5 + i)) & 0x1;
        if (capable)
        {
            g_ruapu_isa_supported[j] = g_ruapu_isa_map[i].isa;
            j++;
        }
    }
    g_ruapu_isa_supported[j] = 0;
}
#endif

void ruapu_init()
{
#if defined _WIN32 || defined __ANDROID__ || defined __linux__ || defined __APPLE__ || defined __FreeBSD__ || defined __NetBSD__ || defined __OpenBSD__ || defined __DragonFly__ || defined __sun__ || defined __SYTERKIT__
    size_t j = 0;
    for (size_t i = 0; i < sizeof(g_ruapu_isa_map) / sizeof(g_ruapu_isa_map[0]); i++)
    {
        int capable = ruapu_detect_isa(g_ruapu_isa_map[i].inst);
        if (capable)
        {
            g_ruapu_isa_supported[j] = g_ruapu_isa_map[i].isa;
            j++;
        }
    }
    g_ruapu_isa_supported[j] = 0;
#elif defined __openrisc__
    ruapu_detect_openrisc_isa();
#else
    // initialize g_ruapu_isa_map for baremetal here, default all zero
    // there is still ruapu_some_XYZ() functions available
    // but you have to work out your own signal handling
#warning ruapu does not support your baremetal os yet
#endif
}

int ruapu_supports(const char* isa)
{
    const char* const* isa_supported = g_ruapu_isa_supported;
    while (*isa_supported)
    {
        if (strcmp(*isa_supported, isa) == 0)
            return 1;

        isa_supported++;
    }

    return 0;
}

const char* const* ruapu_rua()
{
    return g_ruapu_isa_supported;
}

#endif // RUAPU_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // RUAPU_H
