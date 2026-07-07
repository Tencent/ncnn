// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "../modelwriter.h"
#include "ncnnllm_quant.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* /*format*/, void* /*p*/) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class QuantNet : public ModelWriter
{
public:
    QuantNet();

    int quantize_num_threads;

    int save_table(const char* tablepath, int block_size, int weight_bits, int method) const;
};

QuantNet::QuantNet()
    : ModelWriter()
{
    quantize_num_threads = 1;
}

static void show_usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s [inparam] [inbin] [outtable] [(key=value)...]\n", argv0);
    fprintf(stderr, "  method=minmax/mseclip\n");
    fprintf(stderr, "  bits=4/6/8\n");
    fprintf(stderr, "  block=32/64/128\n");
    fprintf(stderr, "  thread=8\n");
    fprintf(stderr, "Sample usage:\n");
    fprintf(stderr, "  %s model.param model.bin model.llm.table method=mseclip bits=4 block=64\n", argv0);
}

int QuantNet::save_table(const char* tablepath, int block_size, int weight_bits, int method) const
{
    FILE* fp = fopen(tablepath, "wb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", tablepath);
        return -1;
    }

    int table_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type == "Gemm")
        {
            const ncnn::Gemm* gemm = (const ncnn::Gemm*)layers[i];

            const char* reason = 0;
            if (!is_supported_llm_gemm(gemm, &reason))
            {
                print_skip_gemm(gemm, reason);
                continue;
            }

            ncnn::Mat B_data_quantize_scales;
            const int ret = make_gemm_B_scales(gemm->B_data, block_size, weight_bits, method, B_data_quantize_scales, quantize_num_threads);
            if (ret != 0)
            {
                fclose(fp);
                return ret;
            }

            char key[256];
            snprintf(key, 256, "%s_param_1", gemm_name(gemm));

            if (write_llm_table_row(fp, key, weight_bits, block_size, method, B_data_quantize_scales) != 0)
            {
                fclose(fp);
                return -1;
            }

            fprintf(stderr, "write_llm_table %s dtype=%s block=%d method=%s\n", key, llm_quant_bits_to_dtype(weight_bits), block_size, llm_quant_method_to_string(method));
            table_count++;
        }
        else if (layers[i]->type == "MultiHeadAttention")
        {
            const ncnn::MultiHeadAttention* mha = (const ncnn::MultiHeadAttention*)layers[i];

            const char* reason = 0;
            if (!is_supported_llm_multiheadattention(mha, &reason))
            {
                print_skip_multiheadattention(mha, reason);
                continue;
            }

            const int qdim = mha->weight_data_size / mha->embed_dim;

            const ncnn::Mat q_weight_data = mha->q_weight_data.reshape(qdim, mha->embed_dim);
            const ncnn::Mat k_weight_data = mha->k_weight_data.reshape(mha->kdim, mha->embed_dim);
            const ncnn::Mat v_weight_data = mha->v_weight_data.reshape(mha->vdim, mha->embed_dim);
            const ncnn::Mat out_weight_data = mha->out_weight_data.reshape(mha->embed_dim, qdim);

            const ncnn::Mat weights[4] = {q_weight_data, k_weight_data, v_weight_data, out_weight_data};
            const int param_ids[4] = {0, 1, 2, 3};

            for (int j = 0; j < 4; j++)
            {
                ncnn::Mat weight_data_quantize_scales;
                const int ret = make_gemm_B_scales(weights[j], block_size, weight_bits, method, weight_data_quantize_scales, quantize_num_threads);
                if (ret != 0)
                {
                    fclose(fp);
                    return ret;
                }

                char key[256];
                snprintf(key, 256, "%s_param_%d", multiheadattention_name(mha), param_ids[j]);

                if (write_llm_table_row(fp, key, weight_bits, block_size, method, weight_data_quantize_scales) != 0)
                {
                    fclose(fp);
                    return -1;
                }

                fprintf(stderr, "write_llm_table %s dtype=%s block=%d method=%s\n", key, llm_quant_bits_to_dtype(weight_bits), block_size, llm_quant_method_to_string(method));
                table_count++;
            }
        }
    }

    fclose(fp);

    fprintf(stderr, "ncnn llm quant table created %d rows\n", table_count);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        show_usage(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            show_usage(argv[0]);
            return -1;
        }
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outtable = argv[3];

    int weight_bits = 6;
    int block_size = 64;
    const char* method = "minmax";
    int thread = 1;

    for (int i = 4; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];
        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            return -1;
        }

        eqs[0] = '\0';
        const char* key = kv;
        const char* value = eqs + 1;

        if (strcmp(key, "method") == 0)
            method = value;
        else if (strcmp(key, "bits") == 0)
            weight_bits = atoi(value);
        else if (strcmp(key, "block") == 0)
            block_size = atoi(value);
        else if (strcmp(key, "thread") == 0)
            thread = atoi(value);
        else
        {
            fprintf(stderr, "unrecognized arg %s\n", key);
            return -1;
        }
    }

    const int quantize_method = llm_quant_method_from_string(method);
    if (quantize_method < 0)
    {
        fprintf(stderr, "unsupported method=%s\n", method);
        return -1;
    }

    if (ncnn::gemm_weight_block_quantize_term(weight_bits, block_size) == 0)
    {
        show_usage(argv[0]);
        return -1;
    }

    if (thread < 1)
    {
        fprintf(stderr, "malformed thread %d\n", thread);
        return -1;
    }

    QuantNet table;
    table.storage_type = 1;
    table.quantize_num_threads = thread;
    table.opt.num_threads = thread;

    if (table.load_param(inparam) != 0)
        return -1;

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        if (table.load_model(dr) != 0)
            return -1;
    }
    else
    {
        if (table.load_model(inbin) != 0)
            return -1;
    }

    if (table.save_table(outtable, block_size, weight_bits, quantize_method) != 0)
        return -1;

    return 0;
}
