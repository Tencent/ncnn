// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "save_ncnn.h"

namespace pnnx {

static bool type_is_integer(int type)
{
    if (type == 1) return false;
    if (type == 2) return false;
    if (type == 3) return false;
    if (type == 4) return true;
    if (type == 5) return true;
    if (type == 6) return true;
    if (type == 7) return true;
    if (type == 8) return true;
    if (type == 9) return true;
    if (type == 10) return false;
    if (type == 11) return false;
    if (type == 12) return false;
    return false;
}

static const char* type_to_dtype_string(int type)
{
    if (type == 1) return "torch.float";
    if (type == 2) return "torch.double";
    if (type == 3) return "torch.half";
    if (type == 4) return "torch.int";
    if (type == 5) return "torch.long";
    if (type == 6) return "torch.short";
    if (type == 7) return "torch.int8";
    if (type == 8) return "torch.uint8";
    if (type == 9) return "torch.bool";
    if (type == 10) return "torch.complex64";
    if (type == 11) return "torch.complex128";
    if (type == 12) return "torch.complex32";
    return "null";
}

static bool string_is_positive_integer(const std::string& t)
{
    for (size_t i = 0; i < t.size(); i++)
    {
        if (t[i] < '0' || t[i] > '9')
            return false;
    }

    return true;
}

int save_ncnn(const Graph& g, const std::string& parampath, const std::string& binpath, const std::string& pypath)
{
    FILE* paramfp = fopen(parampath.c_str(), "wb");
    if (!paramfp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath.c_str());
        return -1;
    }

    FILE* binfp = fopen(binpath.c_str(), "wb");
    if (!binfp)
    {
        fprintf(stderr, "fopen %s failed\n", binpath.c_str());
        fclose(paramfp);
        return -1;
    }

    // magic
    fprintf(paramfp, "7767517\n");

    // op count and oprand count
    fprintf(paramfp, "%d %d\n", (int)g.ops.size(), (int)g.operands.size());

    for (const Operator* op : g.ops)
    {
        fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());

        for (const Operand* oprand : op->inputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const Operand* oprand : op->outputs)
        {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const auto& it : op->params)
        {
            const Parameter& param = it.second;

            if (!string_is_positive_integer(it.first))
            {
                fprintf(stderr, "ignore %s %s param %s=", op->type.c_str(), op->name.c_str(), it.first.c_str());

                if (param.type == 0)
                {
                    fprintf(stderr, "None");
                }
                if (param.type == 1)
                {
                    if (param.b)
                        fprintf(stderr, "True");
                    else
                        fprintf(stderr, "False");
                }
                if (param.type == 2)
                {
                    fprintf(stderr, "%d", param.i);
                }
                if (param.type == 3)
                {
                    fprintf(stderr, "%e", param.f);
                }
                if (param.type == 4)
                {
                    fprintf(stderr, "%s", param.s.c_str());
                }
                if (param.type == 5)
                {
                    fprintf(stderr, "(");
                    for (size_t i = 0; i < param.ai.size(); i++)
                    {
                        fprintf(stderr, "%d", param.ai[i]);
                        if (i + 1 != param.ai.size())
                            fprintf(stderr, ",");
                    }
                    fprintf(stderr, ")");
                }
                if (param.type == 6)
                {
                    fprintf(stderr, "(");
                    for (size_t i = 0; i < param.af.size(); i++)
                    {
                        fprintf(stderr, "%e", param.af[i]);
                        if (i + 1 != param.af.size())
                            fprintf(stderr, ",");
                    }
                    fprintf(stderr, ")");
                }
                if (param.type == 7)
                {
                    fprintf(stderr, "(");
                    for (size_t i = 0; i < param.as.size(); i++)
                    {
                        fprintf(stderr, "%s", param.as[i].c_str());
                        if (i + 1 != param.as.size())
                            fprintf(stderr, ",");
                    }
                    fprintf(stderr, ")");
                }
                fprintf(stderr, "\n");

                continue;
            }

            const int idkey = std::stoi(it.first);
            if (param.type == 2)
            {
                fprintf(paramfp, " %d=%d", idkey, param.i);
            }
            if (param.type == 3)
            {
                fprintf(paramfp, " %d=%e", idkey, param.f);
            }
            if (param.type == 5)
            {
                const int array_size = (int)param.ai.size();
                fprintf(paramfp, " %d=%d", -23300 - idkey, array_size);
                for (size_t i = 0; i < param.ai.size(); i++)
                {
                    fprintf(paramfp, ",%d", param.ai[i]);
                }
            }
            if (param.type == 6)
            {
                const int array_size = (int)param.af.size();
                fprintf(paramfp, " %d=%d", -23300 - idkey, array_size);
                for (size_t i = 0; i < param.af.size(); i++)
                {
                    fprintf(paramfp, ",%e", param.af[i]);
                }
            }
        }

        for (const auto& it : op->attrs)
        {
            //             fprintf(paramfp, " @%s=", it.first.c_str());

            const Attribute& attr = it.second;

            fwrite(attr.data.data(), attr.data.size(), 1, binfp);
        }

        //         if (op->inputnames.size() == op->inputs.size())
        //         {
        //             for (size_t i = 0; i < op->inputs.size(); i++)
        //             {
        //                 const Operand* oprand = op->inputs[i];
        //                 fprintf(paramfp, " $%s=%s", op->inputnames[i].c_str(), oprand->name.c_str());
        //             }
        //         }

        //         for (const Operand* oprand : op->outputs)
        //         {
        //             if (oprand->params.find("__batch_index") == oprand->params.end())
        //                 continue;
        //
        //             const int batch_index = oprand->params.at("__batch_index").i;
        //
        //             fprintf(paramfp, " #%s=%d", oprand->name.c_str(), batch_index);
        //         }

        //         for (const Operand* oprand : op->outputs)
        //         {
        //             if (oprand->shape.empty())
        //                 continue;
        //
        //             fprintf(paramfp, " #%s=", oprand->name.c_str());
        //
        //             fprintf(paramfp, "(");
        //             for (int64_t i = 0; i < oprand->shape.size() - 1; i++)
        //             {
        //                 fprintf(paramfp, "%d,", oprand->shape[i]);
        //             }
        //             if (oprand->shape.size() > 0)
        //                 fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
        //             fprintf(paramfp, ")");
        //
        //             fprintf(paramfp, type_to_string(oprand->type));
        //         }

        fprintf(paramfp, "\n");
    }

    fclose(paramfp);
    fclose(binfp);

    FILE* pyfp = fopen(pypath.c_str(), "wb");
    if (!pyfp)
    {
        fprintf(stderr, "fopen %s failed\n", pypath.c_str());
        return -1;
    }

    fprintf(pyfp, "import numpy as np\n");
    fprintf(pyfp, "import ncnn\n");
    fprintf(pyfp, "import torch\n");

    fprintf(pyfp, "\n");

    // test inference
    {
        fprintf(pyfp, "def test_inference():\n");
        fprintf(pyfp, "    torch.manual_seed(0)\n");

        for (int input_index = 0;; input_index++)
        {
            std::string input_name = std::string("in") + std::to_string(input_index);
            const Operand* r = g.get_operand(input_name);
            if (!r)
                break;

            if (type_is_integer(r->type))
            {
                fprintf(pyfp, "    %s = torch.randint(10, (", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d", r->shape[i]);
                    if (i + 1 != r->shape.size() || r->shape.size() == 1)
                        fprintf(pyfp, ", ");
                }
                fprintf(pyfp, "), dtype=%s)\n", type_to_dtype_string(r->type));
            }
            else
            {
                fprintf(pyfp, "    %s = torch.rand(", input_name.c_str());
                for (size_t i = 0; i < r->shape.size(); i++)
                {
                    fprintf(pyfp, "%d, ", r->shape[i]);
                }
                fprintf(pyfp, "dtype=%s)\n", type_to_dtype_string(r->type));
            }
        }

        fprintf(pyfp, "    out = []\n");
        fprintf(pyfp, "\n");

        fprintf(pyfp, "    with ncnn.Net() as net:\n");
        fprintf(pyfp, "         net.load_param(\"%s\")\n", parampath.c_str());
        fprintf(pyfp, "         net.load_model(\"%s\")\n", binpath.c_str());
        fprintf(pyfp, "\n");
        fprintf(pyfp, "         with net.create_extractor() as ex:\n");

        for (int input_index = 0;; input_index++)
        {
            std::string input_name = std::string("in") + std::to_string(input_index);
            const Operand* r = g.get_operand(input_name);
            if (!r)
                break;

            const int batch_index = r->params.at("__batch_index").i;
            if (batch_index != 233)
            {
                fprintf(pyfp, "            ex.input(\"%s\", ncnn.Mat(%s.squeeze(%d).numpy()).clone())\n", input_name.c_str(), input_name.c_str(), batch_index);
            }
            else
            {
                fprintf(pyfp, "            ex.input(\"%s\", ncnn.Mat(%s.numpy()).clone())\n", input_name.c_str(), input_name.c_str());
            }
        }

        fprintf(pyfp, "\n");

        for (int output_index = 0;; output_index++)
        {
            std::string output_name = std::string("out") + std::to_string(output_index);
            const Operand* r = g.get_operand(output_name);
            if (!r)
                break;

            fprintf(pyfp, "            _, %s = ex.extract(\"%s\")\n", output_name.c_str(), output_name.c_str());

            const int batch_index = r->params.at("__batch_index").i;
            if (batch_index != 233)
            {
                fprintf(pyfp, "            out.append(torch.from_numpy(np.array(%s)).unsqueeze(%d))\n", output_name.c_str(), batch_index);
            }
            else
            {
                fprintf(pyfp, "            out.append(torch.from_numpy(np.array(%s)))\n", output_name.c_str());
            }
        }

        fprintf(pyfp, "\n");

        fprintf(pyfp, "    if len(out) == 1:\n");
        fprintf(pyfp, "        return out[0]\n");
        fprintf(pyfp, "    else:\n");
        fprintf(pyfp, "        return tuple(out)\n");
    }

    fclose(pyfp);

    return 0;
}

} // namespace pnnx
