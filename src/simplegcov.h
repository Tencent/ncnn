// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_SIMPLEGCOV_H
#define NCNN_SIMPLEGCOV_H

#include "platform.h"

#if NCNN_SIMPLEGCOV

#include <stdint.h>

// This minimal gcov emitting implementation only supports the llvm abi

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*llvm_gcov_callback)();

NCNN_EXPORT void llvm_gcov_init(llvm_gcov_callback writeout, llvm_gcov_callback flush);

NCNN_EXPORT void llvm_gcda_start_file(const char* orig_filename, uint32_t version, uint32_t checksum);

NCNN_EXPORT void llvm_gcda_emit_function(uint32_t ident, uint32_t func_checksum, uint32_t cfg_checksum);

NCNN_EXPORT void llvm_gcda_emit_arcs(uint32_t num_counters, uint64_t* counters);

NCNN_EXPORT void llvm_gcda_summary_info();

NCNN_EXPORT void llvm_gcda_end_file();

#ifdef __cplusplus
}
#endif

#endif // NCNN_SIMPLEGCOV

#endif // NCNN_SIMPLEGCOV_H
