// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_pixel_unshuffle.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_pixel_unshuffle_pass : public GraphRewriterPass
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
        return "nn.PixelUnshuffle";
    }

    const char* name_str() const
    {
        return "pixelunshuffle";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;
        const std::vector<int>& shape2 = captured_params.at("shape2").ai;
        const std::vector<int>& dims = captured_params.at("dims").ai;
        int size = shape.size();
        int size2 = shape2.size();

        if (shape.size() < 3 || shape2.size() < 3 || shape.size() != shape2.size() + 2) return false;

        int downscale_factor = shape[size - 1];

        bool match_reshape = downscale_factor == shape[size - 3] && shape[size - 2] == shape2[size2 - 1] && shape[size - 4] == shape2[size2 - 2]
                             && shape2[size2 - 3] == downscale_factor * downscale_factor * shape[size - 5];
        bool match_permute = dims == std::vector<int>{0, 1, 3, 5, 2, 4};

        return match_reshape & match_permute;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;

        int downscale_factor = shape.back();

        op->params["downscale_factor"] = downscale_factor;
    }
};

void fuse_pixel_unshuffle(Graph& graph)
{
    fuse_pixel_unshuffle_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
