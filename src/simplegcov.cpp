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

#include "platform.h"

#if NCNN_SIMPLEGCOV

#include "simplegcov.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

struct gcda_function_arcs_info
{
    // function
    uint32_t ident;
    uint32_t func_checksum;
    uint32_t cfg_checksum;

    // arcs
    uint32_t num_counters;
    uint64_t* counters;
};

struct gcda_info
{
    const char* orig_filename;
    uint32_t version;
    uint32_t checksum;

    std::vector<gcda_function_arcs_info> fas;
};

static void write_gcda_files();

struct gdata
{
    std::vector<gcda_info> gcdas;

    gdata()
    {
        atexit(write_gcda_files);
    }
};

static struct gdata* g = 0;

static void write_gcda_files()
{
    fprintf(stderr, "write_gcda_files\n");

    // dump gcda
    std::vector<gcda_info>& gcdas = g->gcdas;

    for (size_t i = 0; i < gcdas.size(); i++)
    {
        const gcda_info& g = gcdas[i];

        fprintf(stderr, "write %s\n", g.orig_filename);

        FILE* fp = fopen(g.orig_filename, "wb");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed %d\n", g.orig_filename, errno);
            continue;
        }

        const uint32_t GCOV_DATA_MAGIC = 0x67636461;
        fwrite(&GCOV_DATA_MAGIC, 1, sizeof(uint32_t), fp);
        fwrite(&g.version, 1, sizeof(uint32_t), fp);
        fwrite(&g.checksum, 1, sizeof(uint32_t), fp);

        for (size_t j = 0; j < gcdas[i].fas.size(); j++)
        {
            const gcda_function_arcs_info& fa = gcdas[i].fas[j];

            const uint32_t GCOV_TAG_FUNCTION = 0x01000000;
            fwrite(&GCOV_TAG_FUNCTION, 1, sizeof(uint32_t), fp);
            const uint32_t function_size = 3;
            fwrite(&function_size, 1, sizeof(uint32_t), fp);
            fwrite(&fa.ident, 1, sizeof(uint32_t), fp);
            fwrite(&fa.func_checksum, 1, sizeof(uint32_t), fp);
            fwrite(&fa.cfg_checksum, 1, sizeof(uint32_t), fp);

            const uint32_t GCOV_TAG_COUNTER_BASE = 0x01a10000;
            fwrite(&GCOV_TAG_COUNTER_BASE, 1, sizeof(uint32_t), fp);
            const uint32_t arcs_size = fa.num_counters * 2;
            fwrite(&arcs_size, 1, sizeof(uint32_t), fp);
            for (uint32_t k = 0; k < fa.num_counters; k++)
            {
                fwrite(&fa.counters[k], 1, sizeof(uint64_t), fp);
            }
        }

        fclose(fp);
    }

    delete g;
    g = 0;
}

#ifdef __cplusplus
extern "C" {
#endif

void llvm_gcov_init(llvm_gcov_callback writeout, llvm_gcov_callback flush)
{
    // fprintf(stderr, "llvm_gcov_init %p %p\n", writeout, flush);

    if (g == 0)
    {
        g = new gdata;
    }

    writeout();
}

void llvm_gcda_start_file(const char* orig_filename, uint32_t version, uint32_t checksum)
{
    // fprintf(stderr, "llvm_gcda_start_file %s %u %u\n", orig_filename, version, checksum);

    std::vector<gcda_info>& gcdas = g->gcdas;

    gcda_info g;
    g.orig_filename = orig_filename;
    g.version = version;
    g.checksum = checksum;
    gcdas.push_back(g);
}

void llvm_gcda_emit_function(uint32_t ident, uint32_t func_checksum, uint32_t cfg_checksum)
{
    // fprintf(stderr, "llvm_gcda_emit_function %u %u %u\n", ident, func_checksum, cfg_checksum);

    std::vector<gcda_info>& gcdas = g->gcdas;

    gcda_function_arcs_info fa = {ident, func_checksum, cfg_checksum, 0, 0};
    gcdas.back().fas.push_back(fa);
}

void llvm_gcda_emit_arcs(uint32_t num_counters, uint64_t* counters)
{
    // fprintf(stderr, "llvm_gcda_emit_arcs %u %p\n", num_counters, counters);

    std::vector<gcda_info>& gcdas = g->gcdas;

    gcda_function_arcs_info& fa = gcdas.back().fas.back();
    fa.num_counters = num_counters;
    fa.counters = counters;
}

void llvm_gcda_summary_info()
{
    // fprintf(stderr, "llvm_gcda_summary_info\n");
}

void llvm_gcda_end_file()
{
    // fprintf(stderr, "llvm_gcda_end_file\n");
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEGCOV
