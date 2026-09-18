// Copyright 2019 BUG1989
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

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
        bits = 0;
        block_size = 0;
    }

public:
    int bits;
    int block_size;
    std::string qweight;
    ncnn::Mat scales;
};

static bool qweight_path_is_absolute(const char* path)
{
    if (path[0] == '/')
        return true;

#ifdef _WIN32
    if (strlen(path) >= 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/'))
        return true;
#endif

    return false;
}

static std::string resolve_qweight_path(const char* tablepath, const char* path)
{
    if (qweight_path_is_absolute(path))
        return std::string(path);

    const char* slash = strrchr(tablepath, '/');
#ifdef _WIN32
    const char* backslash = strrchr(tablepath, '\\');
    if (backslash && (!slash || backslash > slash))
        slash = backslash;
#endif

    if (!slash)
        return std::string(path);

    return std::string(tablepath, slash - tablepath + 1) + path;
}

static inline int sign_extend(int v, int bits)
{
    const int sign_bit = 1 << (bits - 1);
    return (v ^ sign_bit) - sign_bit;
}

static inline int unpack_signed_weight(const unsigned char* ptr, int k, int bits, int packed_k_bytes)
{
    const int bit_offset = k * bits;
    const int byte_offset = bit_offset / 8;
    const int bit_shift = bit_offset % 8;

    unsigned int v = ptr[byte_offset];
    if (byte_offset + 1 < packed_k_bytes)
        v |= (unsigned int)ptr[byte_offset + 1] << 8;

    const int mask = (1 << bits) - 1;
    return sign_extend((v >> bit_shift) & mask, bits);
}

static bool parse_int_string(const char* s, int& v)
{
    int nconsumed = 0;
    if (sscanf(s, "%d%n", &v, &nconsumed) != 1 || s[nconsumed] != '\0')
        return false;

    return true;
}

static bool read_line(FILE* fp, std::vector<char>& line)
{
    line.clear();

    char buf[4096];
    while (fgets(buf, sizeof(buf), fp))
    {
        const size_t len = strlen(buf);
        line.insert(line.end(), buf, buf + len);
        if (len > 0 && buf[len - 1] == '\n')
            break;
    }

    if (line.empty())
        return false;

    line.push_back('\0');
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

    std::vector<char> line;
    char* pch = NULL;

    while (read_line(fp, line))
    {
        line[strcspn(line.data(), "\r\n")] = 0;

        pch = strtok(line.data(), " \t");
        if (pch == NULL)
            continue;
        if (pch[0] == '#')
            continue;

        char key[256];
        sscanf(pch, "%255s", key);
        key_str = key;

        LLMWeightScale scale;
        scales.clear();

        bool coeff_started = false;
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
                    fprintf(stderr, "%s malformed metadata %s=\n", filepath, k);
                    fclose(fp);
                    return false;
                }

                if (strcmp(k, "bits") == 0)
                {
                    if (!parse_int_string(v, scale.bits))
                    {
                        fprintf(stderr, "%s malformed bits=%s\n", filepath, v);
                        fclose(fp);
                        return false;
                    }
                }
                else if (strcmp(k, "block") == 0)
                {
                    if (!parse_int_string(v, scale.block_size))
                    {
                        fprintf(stderr, "%s malformed block=%s\n", filepath, v);
                        fclose(fp);
                        return false;
                    }
                }
                else if (strcmp(k, "qweight") == 0)
                {
                    scale.qweight = resolve_qweight_path(filepath, v);
                }
                else if (strcmp(k, "method") != 0)
                {
                    fprintf(stderr, "%s unsupported metadata %s\n", filepath, k);
                    fclose(fp);
                    return false;
                }
            }
            else
            {
                if (eqs && coeff_started)
                {
                    fprintf(stderr, "%s key=value token after coefficients started\n", filepath);
                    fclose(fp);
                    return false;
                }

                coeff_started = true;

                float coeff = 0.f;
                if (sscanf(pch, "%f", &coeff) != 1)
                {
                    fprintf(stderr, "%s malformed coefficient %s\n", filepath, pch);
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

static int make_input_scaled_weight(const ncnn::Mat& weight_data, const ncnn::Mat& input_scales, ncnn::Mat& weight_data_scaled)
{
    const int K = weight_data.w;
    const int N = weight_data.h;

    if (input_scales.w != K)
    {
        fprintf(stderr, "input scale count mismatch expected=%d got=%d\n", K, input_scales.w);
        return -1;
    }

    weight_data_scaled.create(K, N);
    if (weight_data_scaled.empty())
        return -100;

    const float* input_scale_ptr = input_scales;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* outptr = weight_data_scaled.row(n);

        for (int k = 0; k < K; k++)
            outptr[k] = ptr[k] / input_scale_ptr[k];
    }

    return 0;
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
    if (scale.bits == 0 || scale.block_size == 0)
    {
        fprintf(stderr, "%s missing mandatory metadata\n", key);
        return -1;
    }

    weight_bits = scale.bits;
    block_size = scale.block_size;

    if (llm_weight_block_quantize_term(weight_bits, block_size) == 0)
    {
        fprintf(stderr, "%s unsupported bits=%d block=%d\n", key, weight_bits, block_size);
        return -1;
    }

    const int block_count = (K + block_size - 1) / block_size;
    const size_t weight_scale_count = (size_t)N * block_count;

    if ((size_t)scale.scales.w != weight_scale_count)
    {
        fprintf(stderr, "%s coefficient count mismatch expected=%zu got=%d\n", key, weight_scale_count, scale.scales.w);
        return -1;
    }

    weight_data_quantize_scales.create(block_count, N);
    if (weight_data_quantize_scales.empty())
        return -100;

    memcpy(weight_data_quantize_scales.data, scale.scales.data, weight_scale_count * sizeof(float));
    const float* scale_ptr = weight_data_quantize_scales;
    for (size_t i = 0; i < weight_scale_count; i++)
    {
        if (!(scale_ptr[i] > 0.f))
        {
            fprintf(stderr, "%s invalid weight scale index=%zu coeff=%f\n", key, i, scale_ptr[i]);
            return -1;
        }
    }

    return 0;
}

static int read_qweight_file(const char* key, const LLMWeightScale& scale, int K, int N, int weight_bits, ncnn::Mat& weight_data_quantized)
{
    const int packed_k_bytes = llm_weight_quantize_packed_k_bytes(K, weight_bits);
    const size_t qweight_bytes = (size_t)N * packed_k_bytes;

    FILE* fp = fopen(scale.qweight.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "%s fopen qweight %s failed\n", key, scale.qweight.c_str());
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    const long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size != (long)qweight_bytes)
    {
        fprintf(stderr, "%s qweight byte count mismatch expected=%zu got=%ld\n", key, qweight_bytes, file_size);
        fclose(fp);
        return -1;
    }

    weight_data_quantized.create(packed_k_bytes, N, (size_t)1u);
    if (weight_data_quantized.empty())
    {
        fclose(fp);
        return -100;
    }

    if (fread(weight_data_quantized.data, 1, qweight_bytes, fp) != qweight_bytes)
    {
        fprintf(stderr, "%s fread qweight %s failed\n", key, scale.qweight.c_str());
        fclose(fp);
        return -1;
    }

    fclose(fp);

    const int used_bits = (int)(((size_t)K * weight_bits) % 8);
    const unsigned char tail_mask = used_bits == 0 ? 0 : (unsigned char)(0xffu << used_bits);
    const int qmin = -(1 << (weight_bits - 1));

    for (int n = 0; n < N; n++)
    {
        const unsigned char* qptr = weight_data_quantized.row<const unsigned char>(n);

        if (tail_mask && (qptr[packed_k_bytes - 1] & tail_mask))
        {
            fprintf(stderr, "%s qweight tail padding bits are not zero row=%d\n", key, n);
            return -1;
        }

        for (int k = 0; k < K; k++)
        {
            const int q = unpack_signed_weight(qptr, k, weight_bits, packed_k_bytes);
            if (q == qmin)
            {
                fprintf(stderr, "%s qweight contains unsupported value row=%d k=%d q=%d\n", key, n, k, q);
                return -1;
            }
        }
    }

    return 0;
}

static int llm_table_row_to_qweight(const char* key, const LLMWeightScale& scale, int K, int N, int& weight_bits, int& block_size, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_quantized)
{
    if (scale.qweight.empty())
    {
        fprintf(stderr, "%s missing mandatory metadata\n", key);
        return -1;
    }

    int ret = llm_table_row_to_scales(key, scale, K, N, weight_bits, block_size, weight_data_quantize_scales);
    if (ret != 0)
        return ret;

    return read_qweight_file(key, scale, K, N, weight_bits, weight_data_quantized);
}

static int llm_table_row_to_input_scales(const char* key, const LLMWeightScale& scale, int K, ncnn::Mat& input_scales)
{
    if (scale.scales.w != K)
    {
        fprintf(stderr, "%s coefficient count mismatch expected=%d got=%d\n", key, K, scale.scales.w);
        return -1;
    }

    const float* ptr = scale.scales;
    for (int k = 0; k < K; k++)
    {
        if (!(ptr[k] > 0.f))
        {
            fprintf(stderr, "%s invalid input scale index=%d coeff=%f\n", key, k, ptr[k]);
            return -1;
        }
    }

    input_scales = scale.scales.clone();
    if (input_scales.empty())
        return -100;

    return 0;
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

        if (!is_supported_llm_gemm(gemm))
        {
            print_skip_gemm(gemm);
            continue;
        }

        fprintf(stderr, "quantize_gemm %s\n", gemm_name(gemm));

        ncnn::Mat B_data_quantized;
        ncnn::Mat B_data_quantize_scales;
        const int ret = quantize_weight_data(gemm->B_data, block_size, weight_bits, method, B_data_quantized, B_data_quantize_scales);
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

        if (!is_supported_llm_gemm(gemm))
        {
            std::map<std::string, LLMWeightScale>::iterator iter = llm_scale_table.find(key);
            if (iter != llm_scale_table.end())
            {
                fprintf(stderr, "table row %s targets unsupported Gemm\n", key);
                return -1;
            }

            print_skip_gemm(gemm);
            continue;
        }

        std::map<std::string, LLMWeightScale>::iterator iter = llm_scale_table.find(key);
        if (iter == llm_scale_table.end())
        {
            fprintf(stderr, "skip_gemm %s missing table row %s\n", gemm_name(gemm), key);
            continue;
        }

        LLMWeightScale& scale = iter->second;

        const int constantN = gemm->constantN;
        const int constantK = gemm->constantK;
        int weight_bits = 0;
        int block_size = 0;
        ncnn::Mat B_data_quantize_scales;
        ncnn::Mat B_data_quantized;
        const bool qweight_row = !scale.qweight.empty();
        int ret = 0;
        if (qweight_row)
            ret = llm_table_row_to_qweight(key, scale, constantK, constantN, weight_bits, block_size, B_data_quantize_scales, B_data_quantized);
        else
            ret = llm_table_row_to_scales(key, scale, constantK, constantN, weight_bits, block_size, B_data_quantize_scales);
        if (ret != 0)
            return ret;

        std::map<std::string, LLMWeightScale>::iterator input_scale_iter = llm_scale_table.find(input_scale_key);
        const bool has_input_scale = input_scale_iter != llm_scale_table.end();

        if (qweight_row && has_input_scale)
        {
            fprintf(stderr, "%s does not support input_scale with qweight yet\n", key);
            return -1;
        }

        ncnn::Mat B_data_input_scales;
        if (has_input_scale)
        {
            ret = llm_table_row_to_input_scales(input_scale_key, input_scale_iter->second, constantK, B_data_input_scales);
            if (ret != 0)
                return ret;
        }

        const int quantize_term = llm_weight_block_quantize_term(weight_bits, block_size, has_input_scale);
        fprintf(stderr, "quantize_gemm %s\n", gemm_name(gemm));

        if (!qweight_row)
        {
            if (has_input_scale)
            {
                ncnn::Mat B_data_scaled;
                ret = make_input_scaled_weight(gemm->B_data, B_data_input_scales, B_data_scaled);
                if (ret != 0)
                    return ret;

                ret = pack_weight_data(B_data_scaled, B_data_quantize_scales, block_size, weight_bits, B_data_quantized);
            }
            else
            {
                ret = pack_weight_data(gemm->B_data, B_data_quantize_scales, block_size, weight_bits, B_data_quantized);
            }
            if (ret != 0)
                return ret;
        }

        gemm->B_data = B_data_quantized;
        gemm->B_data_quantize_scales = B_data_quantize_scales;
        gemm->B_data_input_scales = B_data_input_scales;
        gemm->quantize_term = quantize_term;

        if (has_input_scale)
            llm_scale_table.erase(input_scale_iter);
        llm_scale_table.erase(iter);

        quantized_count++;
    }

    fprintf(stderr, "quantized %d Gemm layers\n", quantized_count);

    return quantized_count;
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

        if (!is_supported_llm_multiheadattention(mha))
        {
            print_skip_multiheadattention(mha);
            continue;
        }

        fprintf(stderr, "quantize_multiheadattention %s\n", multiheadattention_name(mha));

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

        int ret = quantize_weight_data(q_weight_data, block_size, weight_bits, method, q_weight_data_quantized, q_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = quantize_weight_data(k_weight_data, block_size, weight_bits, method, k_weight_data_quantized, k_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = quantize_weight_data(v_weight_data, block_size, weight_bits, method, v_weight_data_quantized, v_weight_data_quantize_scales);
        if (ret != 0)
            return ret;
        ret = quantize_weight_data(out_weight_data, block_size, weight_bits, method, out_weight_data_quantized, out_weight_data_quantize_scales);
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

        if (present_count != 4)
        {
            fprintf(stderr, "MultiHeadAttention %s requires all table rows %s %s %s %s\n", multiheadattention_name(mha), keys[0], keys[1], keys[2], keys[3]);
            return -1;
        }

        if (!is_supported_llm_multiheadattention(mha))
        {
            fprintf(stderr, "table rows target unsupported MultiHeadAttention %s\n", multiheadattention_name(mha));
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

        if (input_scale_present_count != 0 && input_scale_present_count != 4)
        {
            fprintf(stderr, "MultiHeadAttention %s requires all input scale table rows %s %s %s %s\n", multiheadattention_name(mha), input_scale_keys[0], input_scale_keys[1], input_scale_keys[2], input_scale_keys[3]);
            return -1;
        }

        int weight_bits[4];
        int block_size[4];
        bool qweight_rows[4];
        ncnn::Mat weight_data_quantize_scales[4];
        ncnn::Mat weight_data_quantized[4];
        for (int j = 0; j < 4; j++)
        {
            qweight_rows[j] = !iters[j]->second.qweight.empty();

            int ret = 0;
            if (qweight_rows[j])
                ret = llm_table_row_to_qweight(keys[j], iters[j]->second, Ks[j], Ns[j], weight_bits[j], block_size[j], weight_data_quantize_scales[j], weight_data_quantized[j]);
            else
                ret = llm_table_row_to_scales(keys[j], iters[j]->second, Ks[j], Ns[j], weight_bits[j], block_size[j], weight_data_quantize_scales[j]);
            if (ret != 0)
                return ret;
        }

        for (int j = 1; j < 4; j++)
        {
            if (weight_bits[j] != weight_bits[0] || block_size[j] != block_size[0])
            {
                fprintf(stderr, "MultiHeadAttention %s table rows require same bits/block for q/k/v/out\n", multiheadattention_name(mha));
                return -1;
            }

            if (qweight_rows[j] != qweight_rows[0])
            {
                fprintf(stderr, "MultiHeadAttention %s table rows require same format for q/k/v/out\n", multiheadattention_name(mha));
                return -1;
            }
        }

        ncnn::Mat input_scales[4];
        const bool has_input_scale = input_scale_present_count == 4;
        if (has_input_scale && qweight_rows[0])
        {
            fprintf(stderr, "MultiHeadAttention %s does not support input_scale with qweight yet\n", multiheadattention_name(mha));
            return -1;
        }

        if (has_input_scale)
        {
            for (int j = 0; j < 4; j++)
            {
                int ret = llm_table_row_to_input_scales(input_scale_keys[j], input_scale_iters[j]->second, Ks[j], input_scales[j]);
                if (ret != 0)
                    return ret;
            }
        }

        const int quantize_term = llm_weight_block_quantize_term(weight_bits[0], block_size[0], has_input_scale);
        fprintf(stderr, "quantize_multiheadattention %s\n", multiheadattention_name(mha));

        const ncnn::Mat q_weight_data = mha->q_weight_data.reshape(qdim, mha->embed_dim);
        const ncnn::Mat k_weight_data = mha->k_weight_data.reshape(mha->kdim, mha->embed_dim);
        const ncnn::Mat v_weight_data = mha->v_weight_data.reshape(mha->vdim, mha->embed_dim);
        const ncnn::Mat out_weight_data = mha->out_weight_data.reshape(mha->embed_dim, qdim);

        const ncnn::Mat weights[4] = {q_weight_data, k_weight_data, v_weight_data, out_weight_data};
        for (int j = 0; j < 4; j++)
        {
            if (qweight_rows[j])
                continue;

            int ret = 0;
            if (has_input_scale)
            {
                ncnn::Mat weight_data_scaled;
                ret = make_input_scaled_weight(weights[j], input_scales[j], weight_data_scaled);
                if (ret != 0)
                    return ret;

                ret = pack_weight_data(weight_data_scaled, weight_data_quantize_scales[j], block_size[0], weight_bits[0], weight_data_quantized[j]);
            }
            else
            {
                ret = pack_weight_data(weights[j], weight_data_quantize_scales[j], block_size[0], weight_bits[0], weight_data_quantized[j]);
            }
            if (ret != 0)
                return ret;
        }

        mha->q_weight_data = weight_data_quantized[0];
        mha->k_weight_data = weight_data_quantized[1];
        mha->v_weight_data = weight_data_quantized[2];
        mha->out_weight_data = weight_data_quantized[3];
        mha->q_weight_data_quantize_scales = weight_data_quantize_scales[0];
        mha->k_weight_data_quantize_scales = weight_data_quantize_scales[1];
        mha->v_weight_data_quantize_scales = weight_data_quantize_scales[2];
        mha->out_weight_data_quantize_scales = weight_data_quantize_scales[3];
        mha->q_weight_data_input_scales = input_scales[0];
        mha->k_weight_data_input_scales = input_scales[1];
        mha->v_weight_data_input_scales = input_scales[2];
        mha->out_weight_data_input_scales = input_scales[3];
        mha->quantize_term = quantize_term;

        for (int j = 0; j < 4; j++)
        {
            if (has_input_scale)
                llm_scale_table.erase(input_scale_iters[j]);
            llm_scale_table.erase(iters[j]);
        }

        quantized_count++;
    }

    fprintf(stderr, "quantized %d MultiHeadAttention layers\n", quantized_count);

    return quantized_count;
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
        {
            if (!parse_int_string(value, weight_bits))
            {
                fprintf(stderr, "malformed bits=%s\n", value);
                return -1;
            }
        }
        else if (strcmp(key, "block") == 0)
        {
            if (!parse_int_string(value, block_size))
            {
                fprintf(stderr, "malformed block=%s\n", value);
                return -1;
            }
        }
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

        if (quantize_method == LLM_QUANT_METHOD_AWQ || quantize_method == LLM_QUANT_METHOD_GPTQ)
        {
            fprintf(stderr, "method=%s requires llm.table\n", method);
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
    quantizer.storage_type = 1; // use fp16 where weight quant not applied

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
        int quantized_count = quantizer.quantize_gemm_from_table(llm_scale_table);
        if (quantized_count < 0)
            return -1;
        int quantized_count_mha = quantizer.quantize_multiheadattention_from_table(llm_scale_table);
        if (quantized_count_mha < 0)
            return -1;

        quantized_count += quantized_count_mha;

        if (!llm_scale_table.empty())
        {
            for (std::map<std::string, LLMWeightScale>::const_iterator it = llm_scale_table.begin(); it != llm_scale_table.end(); ++it)
                fprintf(stderr, "unused table row %s\n", it->first.c_str());
            return -1;
        }

        if (quantized_count == 0)
        {
            fprintf(stderr, "no layer quantized\n");
            return -1;
        }
    }
    else if (quantizer.quantize_gemm(block_size, weight_bits, quantize_method) != 0)
    {
        return -1;
    }
    else if (quantizer.quantize_multiheadattention(block_size, weight_bits, quantize_method) != 0)
    {
        return -1;
    }

    if (quantizer.save(outparam, outbin) != 0)
        return -1;

    return 0;
}
