// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "../modelwriter.h"
#include "../pnnx/src/json.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <fstream>
#include <sstream>

struct Safetensor
{
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset;
    uint64_t size;
};

class Safetensors
{
public:
    bool open(const char* path);
    bool find(const std::string& name, Safetensor& tensor) const;
    bool read(const Safetensor& tensor, uint64_t offset, void* ptr, size_t size);

private:
    std::ifstream file;
    pnnx::JsonValue metadata;
    uint64_t data_offset;
    uint64_t data_size;
};

bool Safetensors::open(const char* path)
{
    file.open(path, std::ios::binary);
    if (!file)
        return false;

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();
    if (file_size < 8)
        return false;

    file.seekg(0, std::ios::beg);
    unsigned char bytes[8];
    if (!file.read((char*)bytes, sizeof(bytes)))
        return false;

    uint64_t header_size = 0;
    for (int i = 0; i < 8; i++)
        header_size |= (uint64_t)bytes[i] << (i * 8);
    if (header_size == 0 || header_size > 100000000 || header_size > (uint64_t)(file_size - 8))
        return false;

    std::vector<char> header((size_t)header_size);
    if (!file.read(header.data(), (std::streamsize)header_size))
        return false;
    if (pnnx::parse_json(header.data(), header.size(), metadata) != 0 || metadata.type != pnnx::JsonValue::Object)
        return false;

    data_offset = 8 + header_size;
    data_size = (uint64_t)file_size - data_offset;
    return true;
}

bool Safetensors::find(const std::string& name, Safetensor& tensor) const
{
    const pnnx::JsonValue* item = metadata.get(name);
    if (!item)
        return false;

    const pnnx::JsonValue* dtype = item->get("dtype");
    const pnnx::JsonValue* shape = item->get("shape");
    const pnnx::JsonValue* offsets = item->get("data_offsets");
    if (!dtype || dtype->type != pnnx::JsonValue::String || !shape || shape->type != pnnx::JsonValue::Array || !offsets || offsets->type != pnnx::JsonValue::Array || offsets->a.size() != 2)
        return false;

    const pnnx::JsonValue& begin = offsets->a[0];
    const pnnx::JsonValue& end = offsets->a[1];
    if (begin.type != pnnx::JsonValue::Int || end.type != pnnx::JsonValue::Int || begin.i < 0 || end.i < begin.i || (uint64_t)end.i > data_size)
        return false;

    tensor.dtype = dtype->s;
    tensor.shape.resize(shape->a.size());
    tensor.offset = (uint64_t)begin.i;
    tensor.size = (uint64_t)(end.i - begin.i);
    for (size_t i = 0; i < shape->a.size(); i++)
    {
        if (shape->a[i].type != pnnx::JsonValue::Int || shape->a[i].i < 0)
            return false;
        tensor.shape[i] = shape->a[i].i;
    }

    return true;
}

bool Safetensors::read(const Safetensor& tensor, uint64_t offset, void* ptr, size_t size)
{
    if (offset > tensor.size || size > tensor.size - offset)
        return false;
    file.seekg((std::streamoff)(data_offset + tensor.offset + offset), std::ios::beg);
    return (bool)file.read((char*)ptr, (std::streamsize)size);
}

struct LinearMapping
{
    std::string module;
    int first_row;
    int parts;
};

typedef std::map<std::string, LinearMapping> LinearMappings;
typedef std::map<std::string, std::string> NormalizationMappings;

static bool load_mapping(const std::string& path, LinearMappings& mapping)
{
    std::ifstream file(path.c_str());
    if (!file)
        return false;

    std::string line;
    while (std::getline(file, line))
    {
        std::string::size_type comment = line.find('#');
        if (comment != std::string::npos)
            line.resize(comment);

        std::istringstream ss(line);
        std::string layer;
        if (!(ss >> layer))
            continue;

        LinearMapping m;
        m.parts = 0;
        std::string extra;
        if (!(ss >> m.module >> m.first_row) || m.first_row < 0 || (ss >> extra) || mapping.count(layer))
        {
            fprintf(stderr, "invalid mapping for layer %s\n", layer.c_str());
            return false;
        }
        mapping.insert(std::make_pair(layer, m));
    }

    return !file.bad() && !mapping.empty();
}

static void make_zimage_mapping(const std::string& group, LinearMappings& mapping, NormalizationMappings& norms)
{
    const bool modulation = group != "context_refiner";
    const int blocks = group == "layers" ? 30 : 2;
    const char* attention_ffn[] = {"attention.qkv", "attention.qkv", "attention.qkv", "attention.out", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"};
    const char* norm_types[] = {"attention_norm1", "attention.q_norm", "attention.k_norm", "attention_norm2", "ffn_norm1", "ffn_norm2"};
    int norm_index = group == "context_refiner" ? 16 : group == "noise_refiner" ? 22 : 330;

    int index = 0;
    for (int i = 0; i < blocks; i++)
    {
        const std::string prefix = group + "." + std::to_string(i) + ".";
        if (modulation)
        {
            const LinearMapping m = {prefix + "adaLN_modulation.0", 0, 1};
            mapping.insert(std::make_pair("gemm_" + std::to_string(index++), m));
        }
        for (int j = 0; j < 7; j++)
        {
            const LinearMapping m = {prefix + attention_ffn[j], j < 3 ? 3840 * j : 0, j < 3 ? 3 : 1};
            mapping.insert(std::make_pair("gemm_" + std::to_string(index++), m));
        }
        for (int j = 0; j < 6; j++)
            norms.insert(std::make_pair("rmsn_" + std::to_string(norm_index++), prefix + norm_types[j] + ".weight"));
    }
}

static const pnnx::JsonValue* find_quant_option(const pnnx::JsonValue& config, const char* name)
{
    const pnnx::JsonValue* value = config.get(name);
    if (value)
        return value;
    const pnnx::JsonValue* params = config.get("params");
    return params ? params->get(name) : 0;
}

static bool load_quant_config(Safetensors& checkpoint, const std::string& name, int K, int& rotation, int& quantize_term)
{
    Safetensor tensor;
    if (!checkpoint.find(name, tensor) || tensor.dtype != "U8" || tensor.shape.size() != 1 || (uint64_t)tensor.shape[0] != tensor.size || tensor.size == 0 || tensor.size > 65536)
        return false;

    std::vector<char> data((size_t)tensor.size);
    if (!checkpoint.read(tensor, 0, data.data(), data.size()))
        return false;

    pnnx::JsonValue config;
    if (pnnx::parse_json(data.data(), data.size(), config) != 0)
        return false;

    const pnnx::JsonValue* format = config.get("format");
    if (!format || format->type != pnnx::JsonValue::String || format->s != "int8_tensorwise")
        return false;

    const pnnx::JsonValue* convrot = find_quant_option(config, "convrot");
    const pnnx::JsonValue* groupsize = find_quant_option(config, "convrot_groupsize");
    const pnnx::JsonValue* full_precision = find_quant_option(config, "full_precision_matrix_mult");
    if ((convrot && convrot->type != pnnx::JsonValue::Bool)
            || (groupsize && (groupsize->type != pnnx::JsonValue::Int || groupsize->i <= 0 || groupsize->i > INT_MAX))
            || (full_precision && full_precision->type != pnnx::JsonValue::Bool))
        return false;

    rotation = 0;
    if (convrot && convrot->b)
    {
        rotation = groupsize ? (int)groupsize->i : 256;
        int group = rotation;
        while (group % 4 == 0)
            group /= 4;
        if (rotation < 4 || group != 1 || K % rotation != 0)
            return false;
    }
    quantize_term = full_precision && full_precision->b ? 901 : 902;

    return true;
}

static bool read_scales(Safetensors& checkpoint, const Safetensor& tensor, int64_t rows, int first_row, int N, int rotation, ncnn::Mat& scales)
{
    const bool is_scalar = tensor.shape.empty() || (tensor.shape.size() == 1 && tensor.shape[0] == 1);
    const bool is_rowwise = (tensor.shape.size() == 1 && tensor.shape[0] == rows) || (tensor.shape.size() == 2 && tensor.shape[0] == rows && tensor.shape[1] == 1);
    const uint64_t count = is_scalar ? 1 : (uint64_t)rows;
    if (tensor.dtype != "F32" || (!is_scalar && !is_rowwise) || (rotation && !is_rowwise) || count > UINT64_MAX / sizeof(float) || tensor.size != count * sizeof(float))
        return false;

    scales.create(N);
    if (scales.empty())
        return false;

    const int scale_count = is_scalar ? 1 : N;
    const uint64_t offset = is_scalar ? 0 : (uint64_t)first_row * sizeof(float);
    float* ptr = scales;
    if (!checkpoint.read(tensor, offset, ptr, (size_t)scale_count * sizeof(float)))
        return false;
    for (int i = 0; i < scale_count; i++)
    {
#if __BIG_ENDIAN__
        ncnn::swap_endianness_32(ptr + i);
#endif
        if (!isfinite(ptr[i]) || ptr[i] < 0.f)
            return false;
    }
    if (is_scalar)
        scales.fill(ptr[0]);

    return true;
}

static bool read_float_vector(Safetensors& checkpoint, const Safetensor& tensor, int64_t rows, int first_row, int N, ncnn::Mat& data)
{
    if (tensor.shape.size() != 1 || tensor.shape[0] != rows || (uint64_t)rows > UINT64_MAX / sizeof(float))
        return false;

    data.create(N);
    if (data.empty())
        return false;

    if (tensor.dtype == "F32" && tensor.size == (uint64_t)rows * sizeof(float))
    {
        float* ptr = data;
        if (!checkpoint.read(tensor, (uint64_t)first_row * sizeof(float), ptr, (size_t)N * sizeof(float)))
            return false;
#if __BIG_ENDIAN__
        for (int i = 0; i < N; i++)
            ncnn::swap_endianness_32(ptr + i);
#endif
    }
    else if ((tensor.dtype == "BF16" || tensor.dtype == "F16") && tensor.size == (uint64_t)rows * sizeof(unsigned short))
    {
        std::vector<unsigned short> ptr(N);
        if (!checkpoint.read(tensor, (uint64_t)first_row * sizeof(unsigned short), ptr.data(), (size_t)N * sizeof(unsigned short)))
            return false;
        for (int i = 0; i < N; i++)
        {
#if __BIG_ENDIAN__
            ncnn::swap_endianness_16(&ptr[i]);
#endif
            data[i] = tensor.dtype == "BF16" ? ncnn::bfloat16_to_float32(ptr[i]) : ncnn::float16_to_float32(ptr[i]);
        }
    }
    else
    {
        return false;
    }

    for (int i = 0; i < N; i++)
    {
        if (!isfinite(data[i]))
            return false;
    }
    return true;
}

class NetComfy : public ModelWriter
{
public:
    int import_int8(Safetensors& checkpoint, const LinearMappings& mapping);
    int import_zimage_norms(Safetensors& checkpoint, const NormalizationMappings& mapping);
};

int NetComfy::import_int8(Safetensors& checkpoint, const LinearMappings& mapping)
{
    std::set<std::string> imported;
    for (size_t i = 0; i < layers.size(); i++)
    {
        ncnn::Layer* layer = layers[i];
        LinearMappings::const_iterator it = mapping.find(layer->name);
        if (it == mapping.end())
            continue;

        if (!imported.insert(layer->name).second)
        {
            fprintf(stderr, "duplicate mapped ncnn layer %s\n", layer->name.c_str());
            return -1;
        }

        const LinearMapping& source = it->second;
        if (layer->type != "Gemm")
        {
            fprintf(stderr, "layer %s is %s, expected Gemm\n", layer->name.c_str(), layer->type.c_str());
            return -1;
        }

        ncnn::Gemm* gemm = (ncnn::Gemm*)layer;
        const int K = gemm->constantK;
        const int N = gemm->constantN;
        const bool constant_bias = gemm->constantC && gemm->constant_broadcast_type_C != -1;
        if (gemm->quantize_term != 0 || gemm->constantA || !gemm->constantB || !gemm->transB || K <= 0 || N <= 0 || (uint64_t)K * N > INT_MAX
                || (!gemm->B_data.empty() && gemm->B_data.elemsize != 4u) || (constant_bias && gemm->constant_broadcast_type_C != 4))
        {
            fprintf(stderr, "layer %s has unsupported Gemm weight layout\n", layer->name.c_str());
            return -1;
        }

        Safetensor weights;
        if (!checkpoint.find(source.module + ".weight", weights) || weights.dtype != "I8" || weights.shape.size() != 2 || weights.shape[1] != K || weights.size % K != 0
                || weights.size / K != (uint64_t)weights.shape[0] || (int64_t)source.first_row + N > weights.shape[0] || (source.parts && weights.shape[0] != (int64_t)source.parts * N))
        {
            fprintf(stderr, "missing or invalid weight %s for layer %s [%d,%d]\n", source.module.c_str(), layer->name.c_str(), N, K);
            return -1;
        }

        int rotation, mode;
        if (!load_quant_config(checkpoint, source.module + ".comfy_quant", K, rotation, mode))
        {
            fprintf(stderr, "missing or invalid int8_tensorwise metadata for %s\n", source.module.c_str());
            return -1;
        }

        Safetensor source_scales;
        ncnn::Mat scales;
        if (!checkpoint.find(source.module + ".weight_scale", source_scales) || !read_scales(checkpoint, source_scales, weights.shape[0], source.first_row, N, rotation, scales))
        {
            fprintf(stderr, "missing or invalid int8 scales for %s\n", source.module.c_str());
            return -1;
        }

        ncnn::Mat codes(K, N, (size_t)1u);
        if (codes.empty() || !checkpoint.read(weights, (uint64_t)source.first_row * K, codes.data, (size_t)K * N))
        {
            fprintf(stderr, "cannot read int8 weights for %s\n", source.module.c_str());
            return -1;
        }

        Safetensor source_bias;
        ncnn::Mat bias;
        const bool has_bias = checkpoint.find(source.module + ".bias", source_bias);
        if (constant_bias != has_bias || (has_bias && !read_float_vector(checkpoint, source_bias, weights.shape[0], source.first_row, N, bias)))
        {
            fprintf(stderr, "missing or invalid bias for %s\n", source.module.c_str());
            return -1;
        }

        gemm->B_data = codes;
        gemm->B_data_quantize_scales = scales;
        if (has_bias)
        {
            gemm->C_data = bias.reshape(N, 1);
            if (gemm->C_data.empty())
                return -100;
        }
        gemm->quantize_term = mode;
        gemm->weight_int8_rotation = rotation;
        fprintf(stderr, "%s <- %s rows %d..%d, rotation=%d, %s\n", layer->name.c_str(), source.module.c_str(), source.first_row, source.first_row + N - 1, rotation, mode == 901 ? "weight-only" : "W8A8");
    }

    if (imported.size() != mapping.size())
    {
        fprintf(stderr, "only %zu of %zu mapped layers found in ncnn graph\n", imported.size(), mapping.size());
        return -1;
    }

    fprintf(stderr, "imported %zu INT8 linear weights\n", imported.size());
    return 0;
}

int NetComfy::import_zimage_norms(Safetensors& checkpoint, const NormalizationMappings& mapping)
{
    std::set<std::string> imported;
    for (size_t i = 0; i < layers.size(); i++)
    {
        ncnn::Layer* layer = layers[i];
        if (layer->type == "RMSNorm")
        {
            ncnn::RMSNorm* norm = (ncnn::RMSNorm*)layer;
            NormalizationMappings::const_iterator it = mapping.find(layer->name);
            Safetensor tensor;
            const int count = norm->affine_size;
            if (it == mapping.end() || !imported.insert(layer->name).second || count <= 0 || !norm->affine
                    || !checkpoint.find(it->second, tensor) || !read_float_vector(checkpoint, tensor, count, 0, count, norm->gamma_data))
            {
                fprintf(stderr, "missing or invalid Z-Image RMSNorm weights for %s\n", layer->name.c_str());
                return -1;
            }
        }
        else if (layer->type == "Gemm")
        {
            const ncnn::Gemm* gemm = (const ncnn::Gemm*)layer;
            if (gemm->B_data.empty())
            {
                fprintf(stderr, "Z-Image Gemm %s has no imported weights\n", layer->name.c_str());
                return -1;
            }
        }
        else if (layer->type != "Input" && layer->type != "Split" && layer->type != "Slice" && layer->type != "Swish" && layer->type != "TanH"
                 && layer->type != "BinaryOp" && layer->type != "Permute" && layer->type != "Reshape" && layer->type != "RotaryEmbed" && layer->type != "SDPA")
        {
            fprintf(stderr, "unsupported layer %s (%s) in Z-Image refiner graph\n", layer->name.c_str(), layer->type.c_str());
            return -1;
        }
    }

    if (imported.size() != mapping.size())
    {
        fprintf(stderr, "only %zu of %zu Z-Image RMSNorm layers found\n", imported.size(), mapping.size());
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 7)
    {
        fprintf(stderr, "Usage: %s in.param in.bin|- checkpoint.safetensors out.param out.bin context_refiner|noise_refiner|layers|mapping.txt\n", argv[0]);
        fprintf(stderr, "Use - to load all weights from the checkpoint for a named Z-Image-Turbo refiner graph. "
                "A mapping file uses one 'gemm_layer comfy_module first_output_row' per line and requires in.bin.\n");
        return -1;
    }

    for (int i = 4; i <= 5; i++)
    {
        for (int j = 1; j < i; j++)
        {
            if (strcmp(argv[i], argv[j]) == 0)
            {
                fprintf(stderr, "output path %s is already in use by another argument\n", argv[i]);
                return -1;
            }
        }
    }

    const std::string group = argv[6];
    const bool use_base_weights = strcmp(argv[2], "-") != 0;
    LinearMappings mapping;
    NormalizationMappings norms;
    if (group == "context_refiner" || group == "noise_refiner" || group == "layers")
        make_zimage_mapping(group, mapping, norms);
    else if (!use_base_weights || !load_mapping(group, mapping))
    {
        fprintf(stderr, "cannot use layer mapping %s without a readable mapping file and an input .bin\n", group.c_str());
        return -1;
    }

    Safetensors checkpoint;
    if (!checkpoint.open(argv[3]))
    {
        fprintf(stderr, "cannot read safetensors checkpoint %s\n", argv[3]);
        return -1;
    }

    NetComfy net;
    net.storage_type = 2; // use bfloat16 for float weights carrying a storage tag
    if (net.load_param(argv[1]) != 0 || (use_base_weights && net.load_model(argv[2]) != 0) || net.import_int8(checkpoint, mapping) != 0)
        return -1;
    if (!use_base_weights && net.import_zimage_norms(checkpoint, norms) != 0)
        return -1;

    return net.save(argv[4], argv[5]);
}
