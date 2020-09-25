// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cpu_flag_test.h"

namespace ncnn {
static void _ncnn_cpuid(int info_eax, int info_ecx, int* _cpu_info)
{
#if defined(_MSC_VER)
    // Visual C version uses intrinsic or inline x86 assembly.
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
    __cpuidex(_cpu_info, info_eax, info_ecx);
#elif defined(_M_IX86)
    __asm {
			mov eax, info_eax
			mov ecx, info_ecx
			mov edi, _cpu_info
			cpuid
			mov [edi], eax
			mov [edi + 4], ebx
			mov [edi + 8], ecx
			mov [edi + 12], edx
    }
#else // Visual C but not x86
    if (info_ecx == 0)
    {
        __cpuid(_cpu_info, info_eax);
    }
    else
    {
        _cpu_info[0] = _cpu_info[1] = _cpu_info[2] = _cpu_info[3] = 0;
    }
#endif
    // GCC version uses inline x86 assembly.
#else // defined(_MSC_VER)
    int info_ebx = 0, info_edx = 0;
    asm volatile(
#if defined(__i386__) && defined(__PIC__)
        // Preserve ebx for fpic 32 bit.
        "mov %%ebx, %%edi                          \n"
        "cpuid                                     \n"
        "xchg %%edi, %%ebx                         \n"
        : "=D"(info_ebx),
#else
        "cpuid                                     \n"
        : "=b"(info_ebx),
#endif //  defined( __i386__) && defined(__PIC__)
          "+a"(info_eax), "+c"(info_ecx), "=d"(info_edx));
    _cpu_info[0] = info_eax;
    _cpu_info[1] = info_ebx;
    _cpu_info[2] = info_ecx;
    _cpu_info[3] = info_edx;
#endif // defined(_MSC_VER)
}

static int _ncnn_get_xcr0()
{
    // For VS2010 and earlier emit can be used:
    //   _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0  // For VS2010 and earlier.
    //  __asm {
    //    xor        ecx, ecx    // xcr 0
    //    xgetbv
    //    mov        xcr0, eax
    //  }
    // For VS2013 and earlier 32 bit, the _xgetbv(0) optimizer produces bad code.
    // https://code.google.com/p/libyuv/issues/detail?id=529
#if defined(_M_IX86) && (_MSC_VER < 1900)
#pragma optimize("g", off)
#endif
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)) && !defined(__pnacl__) && !defined(__CLR_VER) && !defined(__native_client__)
    // X86 CPUs have xgetbv to detect OS saves high parts of ymm registers.
    int xcr0 = 0;
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
    xcr0 = (int)_xgetbv(0); // VS2010 SP1 required.  NOLINT
#elif defined(__i386__) || defined(__x86_64__)
    asm(".byte 0x0f, 0x01, 0xd0"
        : "=a"(xcr0)
        : "c"(0)
        : "%edx");
#endif // defined(__i386__) || defined(__x86_64__)
    return xcr0;
#else
    // xgetbv unavailable to query for OSSave support.  Return 0.
    return 0;
#endif // defined(_M_IX86) || defined(_M_X64) .. \
       // Return optimization to previous setting.
#if defined(_M_IX86) && (_MSC_VER < 1900)
#pragma optimize("g", on)
#endif
}

static CpuFlag _ncnn_x86_get_cpu_info()
{
    CpuFlag _cpu_info = CpuFlag_None;
    int _cpu_0[4] = {0};
    int _cpu_1[4] = {0};
    int _cpu_7[4] = {0};
    _ncnn_cpuid(0, 0, _cpu_0);
    _ncnn_cpuid(1, 0, _cpu_1);
    if (_cpu_0[0] >= 7)
    {
        _ncnn_cpuid(7, 0, _cpu_7);
    }
    _cpu_info = (CpuFlag)(CpuFlag_X86
                          | ((_cpu_1[3] & 0x04000000) ? CpuFlag_SSE2 : CpuFlag_None)
                          | ((_cpu_1[2] & 0x00000200) ? CpuFlag_SSSE3 : CpuFlag_None)
                          | ((_cpu_1[2] & 0x00080000) ? CpuFlag_SSE41 : CpuFlag_None)
                          | ((_cpu_1[2] & 0x00100000) ? CpuFlag_SSE42 : CpuFlag_None)
                          | ((_cpu_7[1] & 0x00000200) ? CpuFlag_ERMS : CpuFlag_None));

    // AVX requires OS saves YMM registers.
    if (((_cpu_1[2] & 0x1c000000) == 0x1c000000) /*AVX and OSXSave*/ && ((_ncnn_get_xcr0() & 6) == 6) /*Test OS saves YMM registers*/)
    {
        _cpu_info = (CpuFlag)(_cpu_info
                              | CpuFlag_AVX
                              | ((_cpu_7[1] & 0x00000020) ? CpuFlag_AVX2 : CpuFlag_None)
                              | ((_cpu_1[2] & 0x00001000) ? CpuFlag_FMA3 : CpuFlag_None)
                              | ((_cpu_1[2] & 0x20000000) ? CpuFlag_F16C : CpuFlag_None));

        // Detect AVX512bw
        if ((_ncnn_get_xcr0() & 0xe0) == 0xe0)
        {
            _cpu_info = (_cpu_info | (_cpu_7[1] & 0x40000000) ? CpuFlag_AVX512BW : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[1] & 0x80000000) ? CpuFlag_AVX512VL : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[2] & 0x00000002) ? CpuFlag_AVX512VBMI : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[2] & 0x00000040) ? CpuFlag_AVX512VBMI2 : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[2] & 0x00001000) ? CpuFlag_AVX512VBITALG : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[2] & 0x00004000) ? CpuFlag_AVX512VPOPCNTDQ : CpuFlag_None);
            _cpu_info = (_cpu_info | (_cpu_7[2] & 0x00000100) ? CpuFlag_GFNI : CpuFlag_None);
        }
    }
    return _cpu_info;
}

static CpuFlag _ncnn_mips__get_cpu_info(std::string cpuinfo_name, std::string ase)
{
    std::ifstream _ifs(cpuinfo_name, std::ios::binary);
    if (!_ifs.is_open())
    {
        if (ase == " msa")
        {
            return CpuFlag_MSA;
        }
        else if (ase == " mmi")
        {
            return CpuFlag_MMI;
        }
        else
        {
            return CpuFlag_None;
        }
    }
    std::string _s;
    while (getline(_ifs, _s))
    {
        if (_s.substr(0, 16) == "ASEs implemented")
        {
            return ase == " msa" ? CpuFlag_MSA : CpuFlag_None;
        }
        else if (_s.substr(0, 9) == "cpu model")
        {
            return ase == " mmi" ? CpuFlag_MMI : CpuFlag_None;
        }
    }
    return CpuFlag_None;
}

static CpuFlag _ncnn_arm__get_cpu_info(std::string cpuinfo_name)
{
    std::ifstream _ifs(cpuinfo_name, std::ios::binary);
    if (!_ifs.is_open())
    {
        // Assume Neon if /proc/cpuinfo is unavailable.
        // This will occur for Chrome sandbox for Pepper or Render process.
        return CpuFlag_NEON;
    }
    std::string _s;
    while (getline(_ifs, _s))
    {
        if (_s.substr(0, 8) == "Features")
        {
            size_t _p = _s.find(" neon");
            if (_p != std::string::npos)
            {
                if (_s.size() == _p + 5)
                {
                    return CpuFlag_NEON;
                }
                else
                {
                    char _ch = _s[_p + 5];
                    if (_ch == ' ' || _ch == '\r' || _ch == '\n')
                        return CpuFlag_NEON;
                }
            }
            // aarch64 uses asimd for Neon.
            if (_s.find(" asimd") != std::string::npos)
                return CpuFlag_NEON;
        }
    }
    return CpuFlag_None;
}

CpuFlagTest::CpuFlagTest()
{
#if !defined(__pnacl__) && !defined(__CLR_VER) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
    m_cpu_info = _ncnn_x86_get_cpu_info();
#endif
#if defined(__mips__) && defined(__linux__)
#if defined(__mips_msa)
    m_cpu_info = _ncnn_mips__get_cpu_info("/proc/cpuinfo", " msa");
#elif defined(_MIPS_ARCH_LOONGSON3A)
    m_cpu_info = _ncnn_mips__get_cpu_info("/proc/cpuinfo", " mmi");
#endif
    m_cpu_info = (CpuFlag)(m_cpu_info | CpuFlag_MIPS);
#endif
#if defined(__arm__) || defined(__aarch64__)
    // gcc -mfpu=neon defines __ARM_NEON__
    // __ARM_NEON__ generates code that requires Neon.  NaCL also requires Neon.
    // For Linux, /proc/cpuinfo can be tested but without that assume Neon.
#if defined(__ARM_NEON__) || defined(__native_client__) || !defined(__linux__)
    m_cpu_info = CpuFlag_NEON;
    // For aarch64(arm64), /proc/cpuinfo's feature is not complete, e.g. no neon
    // flag in it.
    // So for aarch64, neon enabling is hard coded here.
#endif
#if defined(__aarch64__)
    m_cpu_info = CpuFlag_NEON;
#else
    // Linux arm parse text file for neon detect.
    m_cpu_info = _ncnn_arm__get_cpu_info("/proc/cpuinfo");
#endif
    m_cpu_info = (CpuFlag)(m_cpu_info | CpuFlag_ARM);
#endif // __arm__
    m_cpu_info = (CpuFlag)(m_cpu_info | CpuFlag_Initialized);
}

bool CpuFlagTest::TestSupport(CpuFlag _flag)
{
    return !!(m_cpu_info & _flag);
}

std::string CpuFlagTest::GetSupportString()
{
    static std::map<CpuFlag, std::string> s_flags = {
        {CpuFlag_ARM, "ARM"},
        {CpuFlag_NEON, "NEON"},
        {CpuFlag_X86, "X86"},
        {CpuFlag_SSE2, "SSE2"},
        {CpuFlag_SSSE3, "SSSE3"},
        {CpuFlag_SSE41, "SSE4.1"},
        {CpuFlag_SSE42, "SSE4.2"},
        {CpuFlag_AVX, "AVX"},
        {CpuFlag_AVX2, "AVX2"},
        {CpuFlag_ERMS, "ERMS"},
        {CpuFlag_FMA3, "FMA3"},
        {CpuFlag_F16C, "F16C"},
        {CpuFlag_GFNI, "GFNI"},
        {CpuFlag_AVX512BW, "AVX512BW"},
        {CpuFlag_AVX512VL, "AVX512VL"},
        {CpuFlag_AVX512VBMI, "AVX512VBMI"},
        {CpuFlag_AVX512VBMI2, "AVX512VBMI2"},
        {CpuFlag_AVX512VBITALG, "AVX512VBITALG"},
        {CpuFlag_AVX512VPOPCNTDQ, "AVX512VPOPCNTDQ"},
        {CpuFlag_MIPS, "MIPS"},
        {CpuFlag_MSA, "MSA"},
        {CpuFlag_MMI, "MMI"},
    };
    static std::string s_ret = "";
    if (s_ret != "")
        return s_ret;
    for (auto [_flag, _name] : s_flags)
    {
        if (TestSupport(_flag))
        {
            if (s_ret != "")
                s_ret += " ";
            s_ret += _name;
        }
    }
    return s_ret;
}
} // namespace ncnn
