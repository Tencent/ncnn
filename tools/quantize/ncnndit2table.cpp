// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ncnndit_quant.h"

#include <iomanip>
#include <limits>

class QuantNet : public ModelWriter
{
public:
    int write_table(const DiTQuantizeOptions& options, std::ostream& table);
};

int QuantNet::write_table(const DiTQuantizeOptions& options, std::ostream& table)
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

            table << layer->name << " keep\n";
            continue;
        }

        ncnn::Mat scales;
        const int ret = quantize_dit_weight(op, params, scales, 0, opt.num_threads);
        if (ret != 0)
            return ret;

        table << op->name << " bits=" << params.bits << " block=" << params.block_size << " convrot=" << params.rotation
              << " method=" << llm_quant_method_to_string(params.method) << " N=" << op->constantN << " K=" << op->constantK;
        const float* ptr = scales;
        const int count = scales.w * scales.h;
        for (int j = 0; j < count; j++)
            table << ' ' << ptr[j];
        table << '\n';
        if (!table)
            return -1;

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

    fprintf(stderr, "%d Gemm layers in DiT quantization table\n", quantized_count);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s in.param in.bin out.dit.table [bits=4|6|8] [block=32|64|128] [method=minmax|mseclip] [convrot=auto|group] [include=file] [exclude=file]\n", argv[0]);
        return -1;
    }

    if (strcmp(argv[1], argv[3]) == 0 || strcmp(argv[2], argv[3]) == 0)
    {
        fprintf(stderr, "input and output paths must be distinct\n");
        return -1;
    }

    DiTQuantizeOptions options;
    if (!parse_dit_quantize_options(argc, argv, 4, &options, 0))
        return -1;

    QuantNet quantizer;
    if (quantizer.load_param(argv[1]) != 0 || quantizer.load_model(argv[2]) != 0)
        return -1;

    std::ofstream table(argv[3]);
    if (!table)
    {
        fprintf(stderr, "cannot write DiT quantization table %s\n", argv[3]);
        return -1;
    }

    table << std::setprecision(std::numeric_limits<float>::max_digits10);
    const int ret = quantizer.write_table(options, table);
    table.close();
    if (ret != 0 || !table)
    {
        fprintf(stderr, "failed to generate DiT quantization table %s\n", argv[3]);
        return -1;
    }

    return 0;
}
