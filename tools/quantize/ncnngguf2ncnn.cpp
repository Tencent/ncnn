// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "../modelwriter.h"

#include "layer/gemm.h"
#include "layer/rmsnorm.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

static const int gguf_tensor_f32 = 0;
static const int gguf_tensor_q8_0 = 8;

static const int qwen3_hidden_size = 1024;
static const int qwen3_vocab_size = 151936;
static const int qwen3_layer_count = 28;

static uint32_t load_u32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int64_t file_tell(FILE* fp)
{
#if defined(_WIN32)
    return _ftelli64(fp);
#else
    return ftello(fp);
#endif
}

static bool file_seek(FILE* fp, int64_t offset, int origin)
{
#if defined(_WIN32)
    return _fseeki64(fp, offset, origin) == 0;
#else
    if (offset > (int64_t)std::numeric_limits<off_t>::max())
        return false;
    return fseeko(fp, (off_t)offset, origin) == 0;
#endif
}

struct GGUFFile
{
    explicit GGUFFile(const char* path, const char* mode)
        : fp(fopen(path, mode))
    {
    }

    ~GGUFFile()
    {
        if (fp)
            fclose(fp);
    }

    bool close()
    {
        const int ret = fclose(fp);
        fp = 0;
        return ret == 0;
    }

    FILE* fp;

private:
    GGUFFile(const GGUFFile&);
    GGUFFile& operator=(const GGUFFile&);
};

struct GGUFTensor
{
    std::vector<uint64_t> dims;
    uint32_t type;
    uint64_t offset;
};

class GGUFReader
{
public:
    explicit GGUFReader(FILE* file)
        : fp(file), size(0), alignment(32), data_offset(0)
    {
    }

    bool parse();
    bool match_tensor(const std::string& name, uint32_t type, int k, int n) const;
    bool write_tensor(const std::string& name, FILE* out) const;

    std::map<std::string, uint32_t> metadata;
    std::string architecture;
    std::string rope_scaling;
    std::map<std::string, GGUFTensor> tensors;

private:
    bool skip(uint64_t bytes);
    bool read_u32(uint32_t& value);
    bool read_u64(uint64_t& value);
    bool read_string(std::string& value);
    bool skip_values(uint32_t type, uint64_t count);
    bool read_metadata();

    FILE* fp;
    uint64_t size;
    uint32_t alignment;
    uint64_t data_offset;
};

bool GGUFReader::skip(uint64_t bytes)
{
    const int64_t pos = file_tell(fp);
    return pos >= 0 && (uint64_t)pos <= size && bytes <= size - (uint64_t)pos && file_seek(fp, pos + (int64_t)bytes, SEEK_SET);
}

bool GGUFReader::read_u32(uint32_t& value)
{
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, fp) != 4)
        return false;
    value = load_u32(bytes);
    return true;
}

bool GGUFReader::read_u64(uint64_t& value)
{
    unsigned char bytes[8];
    if (fread(bytes, 1, 8, fp) != 8)
        return false;
    value = (uint64_t)load_u32(bytes) | ((uint64_t)load_u32(bytes + 4) << 32);
    return true;
}

bool GGUFReader::read_string(std::string& value)
{
    uint64_t length;
    if (!read_u64(length) || length == 0 || length > 65536)
        return false;
    value.resize((size_t)length);
    return fread(&value[0], 1, (size_t)length, fp) == length;
}

bool GGUFReader::skip_values(uint32_t type, uint64_t count)
{
    static const unsigned char sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type >= sizeof(sizes))
        return false;

    if (sizes[type])
        return count <= size / sizes[type] && skip(count * sizes[type]);

    if (type != 8 || count > size / 8)
        return false;

    for (uint64_t i = 0; i < count; i++)
    {
        uint64_t length;
        if (!read_u64(length) || !skip(length))
            return false;
    }

    return true;
}

bool GGUFReader::read_metadata()
{
    std::string name;
    uint32_t type;
    if (!read_string(name) || !read_u32(type))
        return false;

    if (name == "general.architecture")
        return type == 8 && architecture.empty() && read_string(architecture);

    if (name == "general.alignment")
        return type == 4 && read_u32(alignment) && alignment >= 8 && (alignment & (alignment - 1)) == 0;

    if (name == "qwen3.rope.scaling.type")
        return type == 8 && rope_scaling.empty() && read_string(rope_scaling);

    if (name.compare(0, 6, "qwen3.") == 0 && (type == 4 || type == 6))
    {
        uint32_t value;
        return read_u32(value) && metadata.insert(std::make_pair(name, value)).second;
    }

    if (type == 9)
    {
        uint32_t element_type;
        uint64_t count;
        return read_u32(element_type) && read_u64(count) && skip_values(element_type, count);
    }

    return skip_values(type, 1);
}

bool GGUFReader::parse()
{
    if (!file_seek(fp, 0, SEEK_END))
        return false;
    const int64_t end = file_tell(fp);
    if (end < 24 || !file_seek(fp, 0, SEEK_SET))
        return false;
    size = (uint64_t)end;

    unsigned char magic[4];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0 || !read_u32(version) || version != 3
            || !read_u64(tensor_count) || tensor_count > size / 24 || !read_u64(metadata_count) || metadata_count > size / 12)
        return false;

    for (uint64_t i = 0; i < metadata_count; i++)
    {
        if (!read_metadata())
            return false;
    }

    for (uint64_t i = 0; i < tensor_count; i++)
    {
        std::string name;
        uint32_t n_dims;
        GGUFTensor t;
        if (!read_string(name) || !read_u32(n_dims) || n_dims == 0 || n_dims > 4)
            return false;
        t.dims.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; j++)
        {
            if (!read_u64(t.dims[j]) || t.dims[j] == 0)
                return false;
        }
        if (!read_u32(t.type) || !read_u64(t.offset) || t.offset % alignment != 0 || !tensors.insert(std::make_pair(name, t)).second)
            return false;
    }

    const int64_t pos = file_tell(fp);
    if (pos < 0)
        return false;
    data_offset = ((uint64_t)pos + alignment - 1) & ~((uint64_t)alignment - 1);
    return data_offset < size;
}

bool GGUFReader::match_tensor(const std::string& name, uint32_t type, int k, int n) const
{
    std::map<std::string, GGUFTensor>::const_iterator it = tensors.find(name);
    if (it == tensors.end())
        return false;

    const GGUFTensor& t = it->second;
    const uint64_t count = (uint64_t)k * (n ? n : 1);
    if (t.type != type || t.dims.size() != (n ? 2u : 1u) || t.dims[0] != (uint64_t)k
            || (n && t.dims[1] != (uint64_t)n) || t.offset > size - data_offset)
        return false;

    if (type == gguf_tensor_q8_0 && k % 32 != 0)
        return false;

    const uint64_t bytes = type == gguf_tensor_f32 ? count * 4 : count / 32 * 34;
    return bytes <= size - data_offset - t.offset;
}

bool GGUFReader::write_tensor(const std::string& name, FILE* out) const
{
    const GGUFTensor& t = tensors.find(name)->second;
    if (!file_seek(fp, (int64_t)(data_offset + t.offset), SEEK_SET))
        return false;

    if (t.type == gguf_tensor_f32)
    {
        std::vector<unsigned char> data((size_t)t.dims[0] * 4);
        return fread(data.data(), 1, data.size(), fp) == data.size() && fwrite(data.data(), 1, data.size(), out) == data.size();
    }

    const size_t blocks = (size_t)(t.dims[0] / 32 * t.dims[1]);
    std::vector<unsigned char> scales(blocks * 2);
    unsigned char packed[1024 * 34];
    unsigned char codes[1024 * 32];
    for (size_t i = 0; i < blocks; i += 1024)
    {
        const size_t n = std::min(blocks - i, (size_t)1024);
        if (fread(packed, 34, n, fp) != n)
            return false;
        for (size_t j = 0; j < n; j++)
        {
            const unsigned char* record = packed + j * 34;
            if ((record[1] & 0x7c) == 0x7c)
            {
                fprintf(stderr, "%s has a non-finite Q8_0 weight scale\n", name.c_str());
                return false;
            }
            memcpy(scales.data() + (i + j) * 2, record, 2);
            memcpy(codes + j * 32, record + 2, 32);
        }
        if (fwrite(codes, 32, n, out) != n)
            return false;
    }

    const unsigned char padding[4] = {0};
    const size_t pad = alignSize(scales.size(), 4) - scales.size();
    return fwrite(scales.data(), 1, scales.size(), out) == scales.size() && fwrite(padding, 1, pad, out) == pad;
}

static bool match_qwen3_config(const GGUFReader& gguf)
{
    if (gguf.architecture != "qwen3" || (!gguf.rope_scaling.empty() && gguf.rope_scaling != "none"))
        return false;

    static const struct
    {
        const char* key;
        uint32_t value;
    } fields[] = {
        {"qwen3.block_count", qwen3_layer_count},
        {"qwen3.embedding_length", qwen3_hidden_size},
        {"qwen3.feed_forward_length", 3072},
        {"qwen3.attention.head_count", 16},
        {"qwen3.attention.head_count_kv", 8},
        {"qwen3.attention.key_length", 128},
        {"qwen3.attention.value_length", 128},
        {"qwen3.attention.layer_norm_rms_epsilon", 0x358637bd},
        {"qwen3.rope.freq_base", 0x49742400},
    };

    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
    {
        std::map<std::string, uint32_t>::const_iterator it = gguf.metadata.find(fields[i].key);
        if (it == gguf.metadata.end() || it->second != fields[i].value)
        {
            fprintf(stderr, "unsupported or missing GGUF metadata %s\n", fields[i].key);
            return false;
        }
    }

    return true;
}

static bool match_qwen3_decoder(const GGUFReader& gguf, const ncnn::Net& decoder, std::vector<std::string>& tensor_names)
{
    static const char* const norm_names[] = {"attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"};
    static const char* const gemm_names[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
    static const int out_sizes[] = {2048, 1024, 1024, 1024, 3072, 3072, 1024};
    static const int in_sizes[] = {1024, 1024, 1024, 2048, 1024, 1024, 3072};
    static const int op_order[] = {0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1};

    const std::vector<ncnn::Layer*>& layers = decoder.layers();
    int gemm_index = 0;
    int norm_index = 0;
    tensor_names.resize(layers.size());

    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];
        char tensor_name[64];
        char layer_name[64];
        int k;
        int n;
        uint32_t type;
        const int op_index = gemm_index + norm_index;
        if (layer->type == "Gemm")
        {
            if (gemm_index >= qwen3_layer_count * 7 || op_order[op_index % 11] != 1)
                return false;
            const int j = gemm_index % 7;
            snprintf(layer_name, sizeof(layer_name), "gemm_%d", gemm_index);
            snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s.weight", gemm_index / 7, gemm_names[j]);
            k = in_sizes[j];
            n = out_sizes[j];
            type = gguf_tensor_q8_0;
            const ncnn::Gemm* op = (const ncnn::Gemm*)layer;
            if (op->constantA || !op->constantB || op->transA || !op->transB || op->constant_broadcast_type_C != -1 || op->constantK != k || op->constantN != n || op->quantize_term)
                return false;
            gemm_index++;
        }
        else if (layer->type == "RMSNorm")
        {
            const int j = norm_index % 4;
            if (norm_index > qwen3_layer_count * 4 || op_order[op_index % 11] != 0)
                return false;
            snprintf(layer_name, sizeof(layer_name), "rmsn_%d", norm_index + qwen3_layer_count * 7);
            if (norm_index == qwen3_layer_count * 4)
                snprintf(tensor_name, sizeof(tensor_name), "output_norm.weight");
            else
                snprintf(tensor_name, sizeof(tensor_name), "blk.%d.%s.weight", norm_index / 4, norm_names[j]);
            k = j == 1 || j == 2 ? 128 : qwen3_hidden_size;
            n = 0;
            type = gguf_tensor_f32;
            const ncnn::RMSNorm* op = (const ncnn::RMSNorm*)layer;
            if (op->affine_size != k || !op->affine || op->eps != 1e-6f)
                return false;
            norm_index++;
        }
        else
        {
            const std::string& t = layer->type;
            if (t != "Input" && t != "Split" && t != "Reshape" && t != "Permute" && t != "RotaryEmbed"
                    && t != "SDPA" && t != "BinaryOp" && t != "Swish")
                return false;
            continue;
        }

        if (layer->name != layer_name || !gguf.match_tensor(tensor_name, type, k, n))
        {
            fprintf(stderr, "unexpected layer or missing/mismatched GGUF tensor %s for %s\n", tensor_name, layer->name.c_str());
            return false;
        }

        tensor_names[i] = tensor_name;
    }

    return gemm_index == qwen3_layer_count * 7 && norm_index == qwen3_layer_count * 4 + 1;
}

static bool write_decoder(const GGUFReader& gguf, ModelWriter& decoder, const std::vector<std::string>& tensor_names, const char* template_path, const std::string& prefix)
{
    const std::string param_path = prefix + ".decoder.ncnn.param";
    const std::string bin_path = prefix + ".decoder.ncnn.bin";
    if (param_path == template_path)
        return false;

    for (size_t i = 0; i < decoder.layers.size(); i++)
    {
        ncnn::Layer* layer = decoder.layers[i];
        if (layer->type == "Gemm")
        {
            ncnn::Gemm* op = (ncnn::Gemm*)layer;
            op->quantize_term = 900;
        }
    }

    GGUFFile bin(bin_path.c_str(), "wb");
    if (!bin.fp || decoder.save(param_path.c_str(), 0) != 0)
        return false;

    for (size_t i = 0; i < tensor_names.size(); i++)
    {
        if (!tensor_names[i].empty() && !gguf.write_tensor(tensor_names[i], bin.fp))
            return false;
    }

    return bin.close();
}

static bool write_embedding_and_output(const GGUFReader& gguf, const std::string& prefix, bool tied)
{
    GGUFFile embed_param((prefix + ".embed.ncnn.param").c_str(), "wb");
    GGUFFile embed_bin((prefix + ".embed.ncnn.bin").c_str(), "wb");
    GGUFFile proj_param((prefix + ".proj_out.ncnn.param").c_str(), "wb");
    if (!embed_param.fp || !embed_bin.fp || !proj_param.fp)
        return false;

    if (fprintf(embed_param.fp, "7767517\n2 2\nInput in0 0 1 in0\nEmbed embed 1 1 in0 out0 0=%d 1=%d 3=%d 18=900\n",
                qwen3_hidden_size, qwen3_vocab_size, qwen3_hidden_size * qwen3_vocab_size)
            < 0)
        return false;
    if (fprintf(proj_param.fp, "7767517\n2 2\nInput in0 0 1 in0\nGemm proj_out 1 1 in0 out0 3=1 5=1 8=%d 9=%d 18=900\n",
                qwen3_vocab_size, qwen3_hidden_size)
            < 0)
        return false;
    if (!gguf.write_tensor("token_embd.weight", embed_bin.fp))
        return false;

    if (!tied)
    {
        GGUFFile proj_bin((prefix + ".proj_out.ncnn.bin").c_str(), "wb");
        if (!proj_bin.fp || !gguf.write_tensor("output.weight", proj_bin.fp) || !proj_bin.close())
            return false;
    }

    return embed_param.close() && embed_bin.close() && proj_param.close();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s model.gguf decoder-template.ncnn.param output-prefix\n", argv[0]);
        fprintf(stderr, "supports Qwen3-0.6B Q8_0 with benchmark/models/llm/qwen3_0.6b_decoder.ncnn.param\n");
        return 1;
    }

    GGUFFile model(argv[1], "rb");
    if (!model.fp)
    {
        fprintf(stderr, "could not open GGUF %s\n", argv[1]);
        return 1;
    }

    GGUFReader gguf(model.fp);
    if (!gguf.parse() || !match_qwen3_config(gguf))
    {
        fprintf(stderr, "invalid GGUF v3 or unsupported Qwen3-0.6B configuration\n");
        return 1;
    }

    const bool tied = gguf.tensors.count("output.weight") == 0;
    if (!gguf.match_tensor("token_embd.weight", gguf_tensor_q8_0, qwen3_hidden_size, qwen3_vocab_size)
            || (!tied && !gguf.match_tensor("output.weight", gguf_tensor_q8_0, qwen3_hidden_size, qwen3_vocab_size)))
    {
        fprintf(stderr, "missing or invalid Q8_0 token embedding or output weight\n");
        return 1;
    }
    const size_t expected_tensors = qwen3_layer_count * (7 + 4) + (tied ? 2 : 3);
    if (gguf.tensors.size() != expected_tensors)
    {
        fprintf(stderr, "unexpected Qwen3-0.6B tensor count %zu (expected %zu)\n", gguf.tensors.size(), expected_tensors);
        return 1;
    }

    ModelWriter decoder;
    std::vector<std::string> tensor_names;
    if (decoder.load_param(argv[2]) != 0 || !match_qwen3_decoder(gguf, decoder, tensor_names))
    {
        fprintf(stderr, "decoder template or GGUF tensors do not match Qwen3-0.6B\n");
        return 1;
    }

    if (!write_decoder(gguf, decoder, tensor_names, argv[2], argv[3]) || !write_embedding_and_output(gguf, argv[3], tied))
    {
        fprintf(stderr, "error writing ncnn model for prefix %s\n", argv[3]);
        return 1;
    }

    fprintf(stderr, "converted Qwen3-0.6B with %s embedding and output weights\n", tied ? "tied" : "separate");
    fprintf(stderr, "use %s.%s.ncnn.bin with %s.proj_out.ncnn.param\n", argv[3], tied ? "embed" : "proj_out", argv[3]);
    return 0;
}
