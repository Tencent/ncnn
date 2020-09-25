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

#ifndef NCNN_INTEL_INSTRUCTION_TEST_H
#define NCNN_INTEL_INSTRUCTION_TEST_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

namespace ncnn {
	enum CpuFlag {
		CpuFlag_None = 0x0,
		CpuFlag_Initialized = 0x1,
		// arm
		CpuFlag_ARM = 0x2,
		CpuFlag_NEON = 0x4,
		// x86
		CpuFlag_X86 = 0x10,
		CpuFlag_SSE2 = 0x20,
		CpuFlag_SSSE3 = 0x40,
		CpuFlag_SSE41 = 0x80,
		CpuFlag_SSE42 = 0x100,
		CpuFlag_AVX = 0x200,
		CpuFlag_AVX2 = 0x400,
		CpuFlag_ERMS = 0x800,
		CpuFlag_FMA3 = 0x1000,
		CpuFlag_F16C = 0x2000,
		CpuFlag_GFNI = 0x4000,
		CpuFlag_AVX512BW = 0x8000,
		CpuFlag_AVX512VL = 0x10000,
		CpuFlag_AVX512VBMI = 0x20000,
		CpuFlag_AVX512VBMI2 = 0x40000,
		CpuFlag_AVX512VBITALG = 0x80000,
		CpuFlag_AVX512VPOPCNTDQ = 0x100000,
		// mips
		CpuFlag_MIPS = 0x200000,
		CpuFlag_MSA = 0x400000,
		CpuFlag_MMI = 0x800000,
	};

	class CpuFlagTest {
	public:
		CpuFlagTest ();

		bool TestSupport (CpuFlag _flag);

		std::string GetSupportString ();

	private:
		CpuFlag m_cpu_info = CpuFlag_None;
	};
}

#endif //NCNN_INTEL_INSTRUCTION_TEST_H
