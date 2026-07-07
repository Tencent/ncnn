// Copyright 2019 BUG1989
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

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

class LLMWeightScale
{
public:
    LLMWeightScale()
    {
        used = false;
    }

public:
    std::string format;
    std::string dtype;
    std::string block;
    std::string scale_dtype;
    std::string scale_encoding;
    std::string method;
    ncnn::Mat scales;
    bool used;
};

static bool parse_int_string(const char* s, int& v)
{
    int nconsumed = 0;
    if (sscanf(s, "%d%n", &v, &nconsumed) != 1 || s[nconsumed] != '\0')
        return false;

    return true;
}

static bool parse_float_string(const char* s, float& v)
{
    int nconsumed = 0;
    if (sscanf(s, "%f%n", &v, &nconsumed) != 1 || s[nconsumed] != '\0')
        return false;

    return true;
}

static bool read_llm_scale_table(const char* filepath, std::map<std::string, LLMWeightScale>& llm_scale_table)
{
    llm_scale_table.clear();

    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "Open %s failed.\n", filepath);
        return false;
    }

    std::string key_str;
    std::vector<float> scales;

    std::vector<char> line(10240000);
    char* pch = NULL;
    int lineno = 0;

    while (!feof(fp))
    {
        char* s = fgets(line.data(), (int)line.size(), fp);
        if (!s)
            break;

        lineno++;
        line[strcspn(line.data(), "\r\n")] = 0;

        pch = strtok(line.data(), " \t");
        if (pch == NULL)
            continue;
        if (pch[0] == '#')
            continue;

        char key[256];
        sscanf(pch, "%255s", key);
        key_str = key;

        if (llm_scale_table.find(key_str) != llm_scale_table.end())
        {
            fprintf(stderr, "%s:%d duplicate key %s\n", filepath, lineno, key);
            fclose(fp);
            return false;
        }

        LLMWeightScale scale;
        scales.clear();

        bool coeff_started = false;
        bool has_format = false;
        bool has_dtype = false;
        bool has_block = false;
        bool has_scale_dtype = false;
        bool has_scale_encoding = false;

        while ((pch = strtok(NULL, " \t")) != NULL)
        {
            char* eqs = strchr(pch, '=');
            if (eqs && !coeff_started)
            {
                eqs[0] = '\0';
                const char* k = pch;
                const char* v = eqs + 1;

                if (v[0] == '\0')
                {
                    fprintf(stderr, "%s:%d malformed metadata %s=\n", filepath, lineno, k);
                    fclose(fp);
                    return false;
                }

                if (strcmp(k, "format") == 0)
                {
                    if (has_format)
                    {
                        fprintf(stderr, "%s:%d duplicate metadata %s\n", filepath, lineno, k);
                        fclose(fp);
                        return false;
                    }

                    has_format = true;
                    scale.format = v;
                }
                else if (strcmp(k, "dtype") == 0)
                {
                    if (has_dtype)
                    {
                        fprintf(stderr, "%s:%d duplicate metadata %s\n", filepath, lineno, k);
                        fclose(fp);
                        return false;
                    }

                    has_dtype = true;
                    scale.dtype = v;
                }
                else if (strcmp(k, "block") == 0)
                {
                    if (has_block)
                    {
                        fprintf(stderr, "%s:%d duplicate metadata %s\n", filepath, lineno, k);
                        fclose(fp);
                        return false;
                    }

                    has_block = true;
                    scale.block = v;
                }
                else if (strcmp(k, "scale_dtype") == 0)
                {
                    if (has_scale_dtype)
                    {
                        fprintf(stderr, "%s:%d duplicate metadata %s\n", filepath, lineno, k);
                        fclose(fp);
                        return false;
                    }

                    has_scale_dtype = true;
                    scale.scale_dtype = v;
                }
                else if (strcmp(k, "scale_encoding") == 0)
                {
                    if (has_scale_encoding)
                    {
                        fprintf(stderr, "%s:%d duplicate metadata %s\n", filepath, lineno, k);
                        fclose(fp);
                        return false;
                    }

                    has_scale_encoding = true;
                    scale.scale_encoding = v;
                }
                else if (strcmp(k, "method") == 0)
                {
                    scale.method = v;
                }
            }
            else
            {
                if (eqs && coeff_started)
                {
                    fprintf(stderr, "%s:%d key=value token after coefficients started\n", filepath, lineno);
                    fclose(fp);
                    return false;
                }

                coeff_started = true;

                float coeff = 0.f;
                if (!parse_float_string(pch, coeff))
                {
                    fprintf(stderr, "%s:%d malformed coefficient %s\n", filepath, lineno, pch);
                    fclose(fp);
                    return false;
                }

                if (!(coeff > 0.f) || !std::isfinite(coeff))
                {
                    fprintf(stderr, "%s invalid coefficient index=%d coeff=%f\n", key, (int)scales.size(), coeff);
                    fclose(fp);
                    return false;
                }

                scales.push_back(coeff);
            }
        }

        scale.scales = ncnn::Mat((int)scales.size(), (void*)scales.data()).clone();
        llm_scale_table[key_str] = scale;
    }

    fclose(fp);

    return true;
}

class NetQuantize : public ModelWriter
{
public:
    NetQuantize();

public:
    int quantize_gemm(int block_size, int weight_bits, int method);
    int quantize_gemm_from_table(std::map<std::string, LLMWeightScale>& llm_scale_table);
    int quantize_multiheadattention(int block_size, int weight_bits, int method);
    int quantize_multiheadattention_from_table(std::map<std::string, LLMWeightScale>& llm_scale_table);
};

NetQuantize::NetQuantize()
    : ModelWriter()
{
}

static void print_usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s [inparam] [inbin] [outparam] [outbin] [(key=value)...]\n", argv0);
    fprintf(stderr, "       %s [inparam] [inbin] [outparam] [outbin] [llm.table] [(key=value)...]\n", argv0);
    fprintf(stderr, "  method=minmax/mseclip\n");
    fprintf(stderr, "  bits=4/6/8\n");
    fprintf(stderr, "  block=32/64/128\n");
    fprintf(stderr, "Sample usage:\n");
    fprintf(stderr, "  %s model.param model.bin model-int4.param model-int4.bin method=mseclip bits=4 block=64\n", argv0);
    fprintf(stderr, "  %s model.param model.bin model-int4.param model-int4.bin model.llm.table\n", argv0);
}

static int llm_table_row_to_scales(const char* key, const LLMWeightScale& scale, int K, int N, int& weight_bits, int& block_size, ncnn::Mat& weight_data_quantize_scales)
{
    if (scale.format.empty() || scale.dtype.empty() || scale.block.empty() || scale.scale_dtype.empty() || scale.scale_encoding.empty())
    {
        fprintf(stderr, "%s missing mandatory metadata\n", key);
        return -1;
    }

    if (scale.format != "block_symmetric")
    {
        fprintf(stderr, "%s unsupported format=%s\n", key, scale.format.c_str());
        return -1;
    }

    weight_bits = llm_quant_dtype_to_bits(scale.dtype.c_str());
    if (weight_bits == 0)
    {
        fprintf(stderr, "%s unsupported dtype=%s\n", key, scale.dtype.c_str());
        return -1;
    }

    if (!parse_int_string(scale.block.c_str(), block_size))
    {
        fprintf(stderr, "%s malformed block=%s\n", key, scale.block.c_str());
        return -1;
    }

    if (llm_weight_block_quantize_term(weight_bits, block_size) == 0)
    {
        fprintf(stderr, "%s unsupported dtype=%s block=%d\n", key, scale.dtype.c_str(), block_size);
        return -1;
    }

    if (scale.scale_dtype != "fp32")
    {
        fprintf(stderr, "%s unsupported scale_dtype=%s\n", key, scale.scale_dtype.c_str());
        return -1;
    }

    if (scale.scale_encoding != "quant")
    {
        fprintf(stderr, "%s unsupported scale_encoding=%s\n", key, scale.scale_encoding.c_str());
        return -1;
    }

    const int block_count = (K + block_size - 1) / block_size;
    const int weight_scale_count = N * block_count;

    if (scale.scales.w != weight_scale_count)
    {
        fprintf(stderr, "%s coefficient count mismatch expected=%d got=%d\n", key, weight_scale_count, scale.scales.w);
        return -1;
    }

    weight_data_quantize_scales.create(block_count, N);
    if (weight_data_quantize_scales.empty())
        return -100;

    memcpy(weight_data_quantize_scales.data, scale.scales.data, weight_scale_count * sizeof(float));

    return 0;
}

static int llm_table_row_to_input_scales(const char* key, const LLMWeightScale& scale, int K, ncnn::Mat& weight_data_input_scales)
{
    if (scale.format.empty() || scale.scale_dtype.empty() || scale.scale_encoding.empty())
    {
        fprintf(stderr, "%s missing mandatory metadata\n", key);
        return -1;
    }

    if (scale.format != "input_scale")
    {
        fprintf(stderr, "%s unsupported format=%s\n", key, scale.format.c_str());
        return -1;
    }

    if (scale.scale_dtype != "fp32")
    {
        fprintf(stderr, "%s unsupported scale_dtype=%s\n", key, scale.scale_dtype.c_str());
        return -1;
    }

    if (scale.scale_encoding != "mul")
    {
        fprintf(stderr, "%s unsupported scale_encoding=%s\n", key, scale.scale_encoding.c_str());
        return -1;
    }

    if (scale.scales.w != K)
    {
        fprintf(stderr, "%s coefficient count mismatch expected=%d got=%d\n", key, K, scale.scales.w);
        return -1;
    }

    const float* ptr = scale.scales;
    for (int k = 0; k < K; k++)
    {
        if (!(ptr[k] > 0.f) || !std::isfinite(ptr[k]))
        {
            fprintf(stderr, "%s invalid input scale index=%d coeff=%f\n", key, k, ptr[k]);
            return -1;
        }
    }

    weight_data_input_scales = scale.scales.clone();
    if (weight_data_input_scales.empty())
        return -100;

    return 0;
}

static void print_unused_llm_table_rows(const std::map<std::string, LLMWeightScale>& llm_scale_table)
{
    for (std::map<std::string, LLMWeightScale>::const_iterator it = llm_scale_table.begin(); it != llm_scale_table.end(); ++it)
    {
        if (!it->second.used)
            fprintf(stderr, "warning: unused llm table row %s\n", it->first.c_str());
    }
}

int NetQuantize::quantize_gemm(int block_size, int weight_bits, int method)
{
    const int quantize_term = llm_weight_block_quantize_term(weight_bits, block_size);
    if (quantize_term == 0)
    {
        fprintf(stderr, "unsupported bits=%d or block=%d\n", weight_bits, block_size);
        return -1;
    }

    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "Gemm")
            continue;

        ncnn::Gemm* gemm = (ncnn::Gemm*)layers[i];

        const char* reason = 0;
        if (!is_supported_llm_gemm(gemm, &reason))
        {
            print_skip_gemm(gemm, reason);
            continue;
        }

        fprintf(stderr, "quantize_gemm bits=%d block_size=%d term=%d %s\n", weight_bits, block_size, quantize_term, gemm_name(gemm));

        ncnn::Mat B_data_quantized;
        ncnn::Mat B_data_quantize_scales;
        const int ret = make_and_pack_gemm_B(gemm->B_data, block_size, weight_bits, method, B_data_quantized, B_data_quantize_scales);
        if (ret != 0)
            return ret;

        gemm->B_data = B_data_quantized;
        gemm->B_data_quantize_scales = B_data_quantize_scales;
        gemm->quantize_term = quantize_term;

        quantized_count++;
    }

    fprintf(stderr, "quantized %d Gemm layers\n", quantized_count);

    return 0;
}

int NetQuantize::quantize_gemm_from_table(std::map<std::string, LLMWeightScale>& llm_scale_table)
{
    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "Gemm")
            continue;

        ncnn::Gemm* gemm = (ncnn::Gemm*)layers[i];
        char key[256];
        snprintf(key, 256, "%s_param_1", gemm_name(gemm));
        char input_scale_key[256];
        snprintf(input_scale_key, 256, "%s_param_1_input_scale", gemm_name(gemm));

        const char* reason = 0;
        if (!is_supported_llm_gemm(gemm, &reason))
        {
            std::map<std::string, LLMWeightScale>::iterator iter = llm_scale_table.find(key);
            if (iter != llm_scale_table.end())
            {
                iter->second.used = true;
                fprintf(stderr, "table row %s targets unsupported Gemm %s\n", key, reason);
                return -1;
            }

            print_skip_gemm(gemm, reason);
            continue;
        }

        std::map<std::string, LLMWeightScale>::iterator iter = llm_scale_table.find(key);
        if (iter == llm_scale_table.end())
        {
            fprintf(stderr, "skip_gemm %s missing table row %s\n", gemm_name(gemm), key);
            continue;
        }

        LLMWeightScale& scale = iter->second;
        scale.used = true;

        const int constantN = gemm->constantN;
        const int constantK = gemm->constantK;
        int weight_bits = 0;
        int block_size = 0;
        ncnn::Mat B_data_quantize_scales;
        int ret = llm_table_row_to_scales(key, scale, constantK, constantN, weight_bits, block_size, B_data_quantize_scales);
        if (ret != 0)
            return ret;

        std::map<std::string, LLMWeightScale>::iterator input_scale_iter = llm_scale_table.find(input_scale_key);
        const bool has_input_scale = input_scale_iter != llm_scale_table.end();

        ncnn::Mat B_data_input_scales;
        if (has_input_scale)
        {
            input_scale_iter->second.used = true;

            ret = llm_table_row_to_input_scales(input_scale_key, input_scale_iter->second, constantK, B_data_input_scales);
            if (ret != 0)
                return ret;
        }

        const int quantize_term = llm_weight_block_quantize_term(weight_bits, block_size, has_input_scale);
        fprintf(stderr, "quantize_gemm table dtype=%s block_size=%d term=%d method=%s %s\n", scale.dtype.c_str(), block_size, quantize_term, scale.method.c_str(), gemm_name(gemm));

        ncnn::Mat B_data_quantized;
        ret = pack_gemm_B_from_scales(gemm->B_data, B_data_quantize_scales, block_size, weight_bits, B_data_quantized);
        if (ret != 0)
            return ret;

        gemm->B_data = B_data_quantized;
        gemm->B_data_quantize_scales = B_data_quantize_scales;
        gemm->B_data_input_scales = B_data_input_scales;
        gemm->quantize_term = quantize_term;

        quantized_count++;
    }

    fprintf(stderr, "quantized %d Gemm layers\n", quantized_count);

    return 0;
}

int NetQuantize::quantize_multiheadattention(int block_size, int weight_bits, int method)
{
    const int quantize_term = llm_weight_block_quantize_term(weight_bits, block_size);
    if (quantize_term == 0)
    {
        fprintf(stderr, "unsupported bits=%d or block=%d\n", weight_bits, block_size);
        return -1;
    }

    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "MultiHeadAttention")
            continue;

        ncnn::MultiHeadAttention* mha = (ncnn::MultiHeadAttention*)layers[i];

        const char* reason = 0;
        if (!is_supported_llm_multiheadattention(mha, &reason))
        {
            print_skip_multiheadattention(mha, reason);
            continue;
        }

        fprintf(stderr, "quantize_multiheadattention bits=%d block_size=%d term=%d %s\n", weight_bits, block_size, quantize_term, multiheadattention_name(mha));

        const int qdim = mha->weight_data_size / mha->embed_dim;

        const ncnn::Mat q_weight_data = mha->q_weight_data.reshape(qdim, mha->embed_dim);
        const ncnn::Mat k_weight_data = mha->k_weight_data.reshape(mha->kdim, mha->embed_dim);
        const ncnn::Mat v_weight_data = mha->v_weight_data.reshape(mha->vdim, mha->embed_dim);
        const ncnn::Mat out_weight_data = mha->out_weight_data.reshape(mha->embed_dim, qdim);

        ncnn::Mat q_weight_data_quantized;
        ncnn::Mat k_weight_data_quantized;
        ncnn::Mat v_weight_data_quantized;
        ncnn::Mat out_weight_data_quantized;
        ncnn::Mat q_weight_data_quantize_scales;
        ncnn::Mat k_weight_data_quantize_scales;
        ncnn::Mat v_weight_data_quantize_scales;
        ncnn::Mat out_weight_data_quantize_scales;

        int ret = make_and_pack_gemm_B(q_weight_data, block_size, weight_bits, method, q_weight_data_quantized, q_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = make_and_pack_gemm_B(k_weight_data, block_size, weight_bits, method, k_weight_data_quantized, k_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = make_and_pack_gemm_B(v_weight_data, block_size, weight_bits, method, v_weight_data_quantized, v_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = make_and_pack_gemm_B(out_weight_data, block_size, weight_bits, method, out_weight_data_quantized, out_weight_data_quantize_scales);
        if (ret != 0)
            return ret;

        mha->q_weight_data = q_weight_data_quantized;
        mha->k_weight_data = k_weight_data_quantized;
        mha->v_weight_data = v_weight_data_quantized;
        mha->out_weight_data = out_weight_data_quantized;
        mha->q_weight_data_quantize_scales = q_weight_data_quantize_scales;
        mha->k_weight_data_quantize_scales = k_weight_data_quantize_scales;
        mha->v_weight_data_quantize_scales = v_weight_data_quantize_scales;
        mha->out_weight_data_quantize_scales = out_weight_data_quantize_scales;
        mha->quantize_term = quantize_term;

        quantized_count++;
    }

    fprintf(stderr, "quantized %d MultiHeadAttention layers\n", quantized_count);

    return 0;
}

int NetQuantize::quantize_multiheadattention_from_table(std::map<std::string, LLMWeightScale>& llm_scale_table)
{
    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "MultiHeadAttention")
            continue;

        ncnn::MultiHeadAttention* mha = (ncnn::MultiHeadAttention*)layers[i];

        char keys[4][256];
        char input_scale_keys[4][256];
        for (int j = 0; j < 4; j++)
        {
            snprintf(keys[j], 256, "%s_param_%d", multiheadattention_name(mha), j);
            snprintf(input_scale_keys[j], 256, "%s_param_%d_input_scale", multiheadattention_name(mha), j);
        }

        std::map<std::string, LLMWeightScale>::iterator iters[4];
        int present_count = 0;
        for (int j = 0; j < 4; j++)
        {
            iters[j] = llm_scale_table.find(keys[j]);
            if (iters[j] != llm_scale_table.end())
                present_count++;
        }

        if (present_count == 0)
            continue;

        for (int j = 0; j < 4; j++)
        {
            if (iters[j] != llm_scale_table.end())
                iters[j]->second.used = true;
        }

        if (present_count != 4)
        {
            fprintf(stderr, "MultiHeadAttention %s requires all table rows %s %s %s %s\n", multiheadattention_name(mha), keys[0], keys[1], keys[2], keys[3]);
            return -1;
        }

        const char* reason = 0;
        if (!is_supported_llm_multiheadattention(mha, &reason))
        {
            fprintf(stderr, "table rows target unsupported MultiHeadAttention %s %s\n", multiheadattention_name(mha), reason);
            return -1;
        }

        const int qdim = mha->weight_data_size / mha->embed_dim;

        const int Ks[4] = {qdim, mha->kdim, mha->vdim, mha->embed_dim};
        const int Ns[4] = {mha->embed_dim, mha->embed_dim, mha->embed_dim, qdim};

        std::map<std::string, LLMWeightScale>::iterator input_scale_iters[4];
        int input_scale_present_count = 0;
        for (int j = 0; j < 4; j++)
        {
            input_scale_iters[j] = llm_scale_table.find(input_scale_keys[j]);
            if (input_scale_iters[j] != llm_scale_table.end())
                input_scale_present_count++;
        }

        if (input_scale_present_count != 0)
        {
            for (int j = 0; j < 4; j++)
            {
                if (input_scale_iters[j] != llm_scale_table.end())
                    input_scale_iters[j]->second.used = true;
            }

            if (input_scale_present_count != 4)
            {
                fprintf(stderr, "MultiHeadAttention %s requires all input scale table rows %s %s %s %s\n", multiheadattention_name(mha), input_scale_keys[0], input_scale_keys[1], input_scale_keys[2], input_scale_keys[3]);
                return -1;
            }
        }

        int weight_bits[4];
        int block_size[4];
        ncnn::Mat weight_data_quantize_scales[4];
        for (int j = 0; j < 4; j++)
        {
            int ret = llm_table_row_to_scales(keys[j], iters[j]->second, Ks[j], Ns[j], weight_bits[j], block_size[j], weight_data_quantize_scales[j]);
            if (ret != 0)
                return ret;
        }

        for (int j = 1; j < 4; j++)
        {
            if (weight_bits[j] != weight_bits[0] || block_size[j] != block_size[0])
            {
                fprintf(stderr, "MultiHeadAttention %s table rows require same dtype/block for q/k/v/out\n", multiheadattention_name(mha));
                return -1;
            }
        }

        ncnn::Mat weight_data_input_scales[4];
        if (input_scale_present_count == 4)
        {
            for (int j = 0; j < 4; j++)
            {
                int ret = llm_table_row_to_input_scales(input_scale_keys[j], input_scale_iters[j]->second, Ks[j], weight_data_input_scales[j]);
                if (ret != 0)
                    return ret;
            }
        }

        const int quantize_term = llm_weight_block_quantize_term(weight_bits[0], block_size[0], input_scale_present_count == 4);
        fprintf(stderr, "quantize_multiheadattention table dtype=%s block_size=%d term=%d %s\n", iters[0]->second.dtype.c_str(), block_size[0], quantize_term, multiheadattention_name(mha));

        const ncnn::Mat q_weight_data = mha->q_weight_data.reshape(qdim, mha->embed_dim);
        const ncnn::Mat k_weight_data = mha->k_weight_data.reshape(mha->kdim, mha->embed_dim);
        const ncnn::Mat v_weight_data = mha->v_weight_data.reshape(mha->vdim, mha->embed_dim);
        const ncnn::Mat out_weight_data = mha->out_weight_data.reshape(mha->embed_dim, qdim);

        ncnn::Mat q_weight_data_quantized;
        ncnn::Mat k_weight_data_quantized;
        ncnn::Mat v_weight_data_quantized;
        ncnn::Mat out_weight_data_quantized;

        int ret = pack_gemm_B_from_scales(q_weight_data, weight_data_quantize_scales[0], block_size[0], weight_bits[0], q_weight_data_quantized);
        if (ret != 0)
            return ret;
        ret = pack_gemm_B_from_scales(k_weight_data, weight_data_quantize_scales[1], block_size[0], weight_bits[0], k_weight_data_quantized);
        if (ret != 0)
            return ret;
        ret = pack_gemm_B_from_scales(v_weight_data, weight_data_quantize_scales[2], block_size[0], weight_bits[0], v_weight_data_quantized);
        if (ret != 0)
            return ret;
        ret = pack_gemm_B_from_scales(out_weight_data, weight_data_quantize_scales[3], block_size[0], weight_bits[0], out_weight_data_quantized);
        if (ret != 0)
            return ret;

        mha->q_weight_data = q_weight_data_quantized;
        mha->k_weight_data = k_weight_data_quantized;
        mha->v_weight_data = v_weight_data_quantized;
        mha->out_weight_data = out_weight_data_quantized;
        mha->q_weight_data_quantize_scales = weight_data_quantize_scales[0];
        mha->k_weight_data_quantize_scales = weight_data_quantize_scales[1];
        mha->v_weight_data_quantize_scales = weight_data_quantize_scales[2];
        mha->out_weight_data_quantize_scales = weight_data_quantize_scales[3];
        mha->q_weight_data_input_scales = weight_data_input_scales[0];
        mha->k_weight_data_input_scales = weight_data_input_scales[1];
        mha->v_weight_data_input_scales = weight_data_input_scales[2];
        mha->out_weight_data_input_scales = weight_data_input_scales[3];
        mha->quantize_term = quantize_term;

        quantized_count++;
    }

    fprintf(stderr, "quantized %d MultiHeadAttention layers\n", quantized_count);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            print_usage(argv[0]);
            return -1;
        }
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* tablepath = 0;

    int kv_start = 5;
    if (argc >= 6 && strchr(argv[5], '=') == NULL)
    {
        tablepath = argv[5];
        kv_start = 6;
    }

    int weight_bits = 6;
    int block_size = 64;
    const char* method = "minmax";

    for (int i = kv_start; i < argc; i++)
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
        else
        {
            fprintf(stderr, "unrecognized arg %s\n", key);
            return -1;
        }
    }

    int quantize_method = LLM_QUANT_METHOD_MINMAX;
    if (!tablepath)
    {
        quantize_method = llm_quant_method_from_string(method);
        if (quantize_method < 0)
        {
            fprintf(stderr, "unsupported method=%s\n", method);
            return -1;
        }

        if (llm_weight_block_quantize_term(weight_bits, block_size) == 0)
        {
            print_usage(argv[0]);
            return -1;
        }
    }

    std::map<std::string, LLMWeightScale> llm_scale_table;
    if (tablepath)
    {
        if (!read_llm_scale_table(tablepath, llm_scale_table))
            return -1;
    }

    NetQuantize quantizer;
    quantizer.storage_type = 1; // keep existing prototype behavior for unrelated fp32 weights

    if (quantizer.load_param(inparam) != 0)
        return -1;

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        if (quantizer.load_model(dr) != 0)
            return -1;
    }
    else
    {
        if (quantizer.load_model(inbin) != 0)
            return -1;
    }

    if (tablepath)
    {
        if (quantizer.quantize_gemm_from_table(llm_scale_table) != 0)
            return -1;
        if (quantizer.quantize_multiheadattention_from_table(llm_scale_table) != 0)
            return -1;
        print_unused_llm_table_rows(llm_scale_table);
    }
    else if (quantizer.quantize_gemm(block_size, weight_bits, quantize_method) != 0)
    {
        return -1;
    }
    else if (quantizer.quantize_multiheadattention(block_size, weight_bits, quantize_method) != 0)
    {
        return -1;
    }

    quantizer.save(outparam, outbin);

    return 0;
}
