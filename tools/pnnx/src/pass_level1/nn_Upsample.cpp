// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Upsample : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.upsampling.Upsample";
    }

    const char* type_str() const
    {
        return "nn.Upsample";
    }

    void write(Operator* op, const TorchGraphProxy& graph) const
    {
        const TorchNodeProxy* upsample_nearest1d = graph.find_node_by_kind("aten::upsample_nearest1d");
        const TorchNodeProxy* upsample_nearest_exact1d = graph.find_node_by_kind("aten::_upsample_nearest_exact1d");
        const TorchNodeProxy* upsample_linear1d = graph.find_node_by_kind("aten::upsample_linear1d");

        const TorchNodeProxy* upsample_nearest2d = graph.find_node_by_kind("aten::upsample_nearest2d");
        const TorchNodeProxy* upsample_nearest_exact2d = graph.find_node_by_kind("aten::_upsample_nearest_exact2d");
        const TorchNodeProxy* upsample_bilinear2d = graph.find_node_by_kind("aten::upsample_bilinear2d");
        const TorchNodeProxy* upsample_bicubic2d = graph.find_node_by_kind("aten::upsample_bicubic2d");

        const TorchNodeProxy* upsample_nearest3d = graph.find_node_by_kind("aten::upsample_nearest3d");
        const TorchNodeProxy* upsample_nearest_exact3d = graph.find_node_by_kind("aten::_upsample_nearest_exact3d");
        const TorchNodeProxy* upsample_trilinear3d = graph.find_node_by_kind("aten::upsample_trilinear3d");

        const TorchNodeProxy* upsample = 0;
        if (upsample_nearest1d)
        {
            upsample = upsample_nearest1d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_nearest_exact1d)
        {
            upsample = upsample_nearest_exact1d;
            op->params["mode"] = "nearest-exact";
        }
        else if (upsample_linear1d)
        {
            upsample = upsample_linear1d;
            op->params["mode"] = "linear";
        }
        else if (upsample_nearest2d)
        {
            upsample = upsample_nearest2d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_nearest_exact2d)
        {
            upsample = upsample_nearest_exact2d;
            op->params["mode"] = "nearest-exact";
        }
        else if (upsample_bilinear2d)
        {
            upsample = upsample_bilinear2d;
            op->params["mode"] = "bilinear";
        }
        else if (upsample_bicubic2d)
        {
            upsample = upsample_bicubic2d;
            op->params["mode"] = "bicubic";
        }
        else if (upsample_nearest3d)
        {
            upsample = upsample_nearest3d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_nearest_exact3d)
        {
            upsample = upsample_nearest_exact3d;
            op->params["mode"] = "nearest-exact";
        }
        else if (upsample_trilinear3d)
        {
            upsample = upsample_trilinear3d;
            op->params["mode"] = "trilinear";
        }

        if (upsample->hasNamedInput("output_size"))
        {
            op->params["size"] = upsample->namedInput("output_size");
        }

        if (upsample->hasNamedInput("scale_factors"))
        {
            op->params["scale_factor"] = upsample->namedInput("scale_factors");
        }

        if (upsample->hasNamedInput("align_corners"))
        {
            op->params["align_corners"] = upsample->namedInput("align_corners");
        }

        bool recompute_scale_factor = true;
        if (op->params.find("size") != op->params.end())
        {
            if (op->params.at("size").type == 2)
            {
                int s = op->params.at("size").i;
                if (s != 0)
                {
                    recompute_scale_factor = false;
                }
            }
            if (op->params.at("size").type == 5)
            {
                const std::vector<int>& size = op->params.at("size").ai;
                for (auto s : size)
                {
                    if (s != 0)
                    {
                        recompute_scale_factor = false;
                        break;
                    }
                }
            }
        }
        if (op->params.find("scale_factor") != op->params.end())
        {
            if (op->params.at("scale_factor").type != 0)
                recompute_scale_factor = false;
        }

        if (recompute_scale_factor)
        {
            op->params["size"] = Parameter();

            // FIXME does this param really counts ?
            // op->params["recompute_scale_factor"] = true;

            // resolve scale_factor in recompute scale graph
            std::vector<float> scale_factor;
            try
            {
                const TorchNodeProxy* size_list = graph.find_node_by_kind("prim::ListConstruct");
                const int size_list_input_count = size_list->input_count();
                for (int i = 0; i < size_list_input_count; i++)
                {
                    const TorchNodeProxy* scale_tensor = graph.find_producer_node_by_value(graph.find_producer_node_by_value(graph.find_producer_node_by_value(graph.find_producer_node_by_value(graph.find_producer_node_by_value(graph.find_producer_node_by_value(graph.find_producer_node_by_value(size_list->input(i))->input(0))->input(0))->input(0))->input(1))->input(0))->input(0));

                    // auto t = scale_tensor->t(torch::jit::attr::value);
                    // float s = (float)t.item<double>();
                    Parameter ps = scale_tensor->node;
                    float s = ps.f;

                    scale_factor.push_back(s);
                }

                op->params["scale_factor"] = scale_factor;
            }
            catch (...)
            {
                fprintf(stderr, "unhandled upsample recompute_scale_factor graph");
                graph.dump();
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Upsample)

} // namespace pnnx
