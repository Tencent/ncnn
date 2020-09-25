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
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>

#include <intrin.h>
#include <immintrin.h>

namespace ncnn {
enum class CpuFlag {
	None = 0x0,
	Initialized = 0x1,
	// arm
	ARM = 0x2,
	NEON = 0x4,
	// x86
	X86 = 0x10,
	SSE2 = 0x20,
	SSSE3 = 0x40,
	SSE41 = 0x80,
	SSE42 = 0x100,
	AVX = 0x200,
	AVX2 = 0x400,
	ERMS = 0x800,
	FMA3 = 0x1000,
	F16C = 0x2000,
	GFNI = 0x4000,
	AVX512BW = 0x8000,
	AVX512VL = 0x10000,
	AVX512VBMI = 0x20000,
	AVX512VBMI2 = 0x40000,
	AVX512VBITALG = 0x80000,
	AVX512VPOPCNTDQ = 0x100000,
	// mips
	MIPS = 0x200000,
	MSA = 0x400000,
	MMI = 0x800000,
};

class CpuFlagTest {
public:
	CpuFlagTest ();

	bool TestSupport (CpuFlag _flag);

	std::string GetSupportString ();

private:
	CpuFlag m_cpu_info = CpuFlag::None;
};
}

#endif //NCNN_INTEL_INSTRUCTION_TEST_H
