// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ncnndit_quant.h"

#include <sstream>

class NetQuantize : public ModelWriter
{
public:
    int quantize_gemm(const DiTQuantizeOptions& options);
    int quantize_gemm_from_table(std::istream& table);
};

static int quantize_dit_gemm(ncnn::Gemm* op, const DiTQuantizeParams& params, ncnn::Mat& scales, int num_threads)
{
    ncnn::Mat weight_data_quantized;
    const int ret = quantize_dit_weight(op, params, scales, &weight_data_quantized, num_threads);
    if (ret != 0)
        return ret;

    op->B_data = weight_data_quantized;
    op->B_data_quantize_scales = scales;
    op->transB = 1;
    op->weight_int8_rotation = params.rotation;
    op->quantize_term = params.bits == 8 ? (params.rotation ? 902 : 901) : llm_weight_block_quantize_term(params.bits, params.block_size);
    fprintf(stderr, "quantize %s %d x %d int%d block=%d convrot=%d\n", op->name.c_str(), op->constantN, op->constantK, params.bits, params.block_size, params.rotation);
    return 0;
}

int NetQuantize::quantize_gemm(const DiTQuantizeOptions& options)
{
    std::set<std::string> unused_include = options.include;
    std::set<std::string> unused_exclude = options.exclude;
    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        ncnn::Layer* layer = layers[i];
        if (layer->type != "Gemm")
            continue;

        ncnn::Gemm* op = (ncnn::Gemm*)layer;
        DiTQuantizeParams params = options.params;
        if (params.rotation == -1)
            params.rotation = dit_rotation_group(op->constantK);

        const bool explicitly_included = unused_include.erase(layer->name) != 0;
        const bool selected = unused_exclude.erase(layer->name) == 0 && (options.include.empty() || explicitly_included);
        const bool supported = is_supported_dit_gemm(op, params.rotation);
        if (!selected || !supported)
        {
            if (selected && (explicitly_included || (params.rotation > 0 && options.params.rotation > 0 && is_supported_dit_gemm(op, 0))))
            {
                fprintf(stderr, "unsupported weight shape or Gemm configuration for %s (convrot=%d, K=%d)\n",
                        layer->name.c_str(), params.rotation, op->constantK);
                return -1;
            }

            fprintf(stderr, "keep %s\n", layer->name.c_str());
            continue;
        }

        ncnn::Mat scales;
        const int ret = quantize_dit_gemm(op, params, scales, opt.num_threads);
        if (ret != 0)
            return ret;

        quantized_count++;
    }

    if (!unused_include.empty() || !unused_exclude.empty())
    {
        const std::string& name = !unused_include.empty() ? *unused_include.begin() : *unused_exclude.begin();
        fprintf(stderr, "Gemm layer %s was not found\n", name.c_str());
        return -1;
    }

    if (quantized_count == 0)
    {
        fprintf(stderr, "no Gemm layer selected for quantization\n");
        return -1;
    }

    fprintf(stderr, "%d Gemm layers quantized\n", quantized_count);
    return 0;
}

int NetQuantize::quantize_gemm_from_table(std::istream& table)
{
    std::set<std::string> seen;
    int quantized_count = 0;
    std::string line;

    while (std::getline(table, line))
    {
        const size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.resize(comment);
        std::istringstream row(line);
        std::string name;
        if (!(row >> name))
            continue;

        const int index = find_layer_index_by_name(name.c_str());
        if (index < 0 || layers[index]->type != "Gemm" || !seen.insert(name).second)
        {
            fprintf(stderr, "unknown or duplicate Gemm layer %s in DiT quantization table\n", name.c_str());
            return -1;
        }

        std::string bits;
        row >> bits;
        if (bits == "keep" && (row >> std::ws).eof())
            continue;

        std::string block, rotation, method, N, K;
        DiTQuantizeParams params;
        int n, k;
        if (!(row >> block >> rotation >> method >> N >> K)
            || !parse_dit_int(bits.c_str(), "bits=", params.bits)
            || !parse_dit_int(block.c_str(), "block=", params.block_size)
            || !parse_dit_int(rotation.c_str(), "convrot=", params.rotation)
            || method.compare(0, 7, "method=") != 0 || !parse_dit_int(N.c_str(), "N=", n) || !parse_dit_int(K.c_str(), "K=", k))
        {
            fprintf(stderr, "invalid DiT quantization table row for %s\n", name.c_str());
            return -1;
        }
        params.method = llm_quant_method_from_string(method.c_str() + 7);

        ncnn::Gemm* op = (ncnn::Gemm*)layers[index];
        if (!is_dit_quantize_params(params) || !is_supported_dit_gemm(op, params.rotation) || op->constantN != n || op->constantK != k)
        {
            fprintf(stderr, "invalid quantization settings or weight shape for %s\n", name.c_str());
            return -1;
        }

        ncnn::Mat scales;
        if (!(row >> std::ws).eof())
        {
            create_dit_weight_scales(scales, k, n, params);
            if (scales.empty())
                return -100;

            const int count = scales.w * scales.h;
            for (int j = 0; j < count; j++)
            {
                float s;
                if (!(row >> s) || !std::isfinite(s) || s < 0.f || (params.bits != 8 && s == 0.f))
                {
                    fprintf(stderr, "invalid scale %d for %s\n", j, name.c_str());
                    return -1;
                }
                scales[j] = s;
            }
            if (!(row >> std::ws).eof())
            {
                fprintf(stderr, "too many scales for %s\n", name.c_str());
                return -1;
            }
        }

        const int ret = quantize_dit_gemm(op, params, scales, opt.num_threads);
        if (ret != 0)
            return ret;

        quantized_count++;
    }

    if (table.bad() || quantized_count == 0)
    {
        fprintf(stderr, "no usable DiT quantization table\n");
        return -1;
    }

    fprintf(stderr, "%d Gemm layers quantized\n", quantized_count);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s in.param in.bin out.param out.bin [bits=4|6|8] [block=32|64|128] [method=minmax|mseclip] [convrot=auto|group] [include=file] [exclude=file] [storage=fp32|fp16|bf16]\n",
                argv[0]);
        fprintf(stderr, "       %s in.param in.bin out.param out.bin dit.table [storage=fp32|fp16|bf16]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* table_path = argc > 5 && strchr(argv[5], '=') == 0 ? argv[5] : 0;
    if (strcmp(inparam, outparam) == 0 || strcmp(inparam, outbin) == 0 || strcmp(inbin, outparam) == 0 || strcmp(inbin, outbin) == 0
        || strcmp(outparam, outbin) == 0 || (table_path && (strcmp(table_path, outparam) == 0 || strcmp(table_path, outbin) == 0)))
    {
        fprintf(stderr, "input and output paths must be distinct\n");
        return -1;
    }

    DiTQuantizeOptions options;
    int storage_type = 0;
    if (!parse_dit_quantize_options(argc, argv, table_path ? 6 : 5, table_path ? 0 : &options, &storage_type))
        return -1;

    std::ifstream table;
    if (table_path)
    {
        table.open(table_path);
        if (!table)
        {
            fprintf(stderr, "cannot open DiT quantization table %s\n", table_path);
            return -1;
        }
    }

    NetQuantize quantizer;
    quantizer.storage_type = storage_type;
    if (quantizer.load_param(inparam) != 0 || quantizer.load_model(inbin) != 0)
        return -1;

    const int ret = table_path ? quantizer.quantize_gemm_from_table(table) : quantizer.quantize_gemm(options);
    if (ret != 0)
        return ret;

    return quantizer.save(outparam, outbin);
}
