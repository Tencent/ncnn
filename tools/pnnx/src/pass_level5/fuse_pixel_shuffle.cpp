// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_pixel_shuffle.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_pixel_shuffle_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.reshape          op_1        1 1 input 1 shape=%shape
Tensor.permute          op_2        1 1 1 2 dims=%dims
Tensor.reshape          op_3        1 1 2 out shape=%shape2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.PixelShuffle";
    }

    const char* name_str() const
    {
        return "pixelshuffle";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;
        const std::vector<int>& shape2 = captured_params.at("shape2").ai;
        const std::vector<int>& dims = captured_params.at("dims").ai;
        int size = shape.size();
        int size2 = shape2.size();

        if (size != 6 || size2 != 4) return false;

        int upscale_factor = shape[2];

        bool match_reshape = upscale_factor == shape[3] && shape[1] == shape2[1] && shape[4] * upscale_factor == shape2[2]
                             && shape[5] * upscale_factor == shape2[3];
        bool match_permute = dims == std::vector<int>{0, 1, 4, 2, 5, 3};

        return match_reshape & match_permute;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;

        int upscale_factor = shape[2];

        op->params["upscale_factor"] = upscale_factor;
    }
};

class fuse_pixel_shuffle_pass_1 : public fuse_pixel_shuffle_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.reshape          op_1        1 1 input 1 shape=%shape
Tensor.permute          op_2        1 1 1 2 dims=%dims
Tensor.reshape          op_3        1 1 2 out shape=%shape2
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_pixel_shuffle(Graph& graph)
{
    fuse_pixel_shuffle_pass a;
    fuse_pixel_shuffle_pass_1 a1;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
}

} // namespace pnnx
