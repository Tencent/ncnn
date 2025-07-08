// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_einsum_equation.h"

#include <ctype.h>
#include <algorithm>
#include <map>
#include <vector>
#include "pass_level2.h"

namespace pnnx {

static void replaceAll(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

void normalize_einsum_equation(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
    {
        Operator* op = graph.ops[i];

        if (op->type != "torch.einsum")
            continue;

        std::string equation = op->params.at("equation").s;

        // remove all spaces
        equation.erase(std::remove_if(equation.begin(), equation.end(), isspace), equation.end());

        size_t equation_len = equation.size();

        std::map<char, char> xset;

        // find arrow ->
        size_t arrow = equation.find("->");
        if (arrow == std::string::npos)
        {
            // normalize to ijkl...
            for (size_t i = 0; i < equation_len; i++)
            {
                char x = equation[i];

                if ((x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                {
                    if (xset.find(x) == xset.end())
                        xset[x] = '\0';
                }
            }

            char x = 'i';
            for (auto& _xset : xset)
            {
                _xset.second = x++;
            }
        }
        else
        {
            // find dot pair ...
            size_t dot0 = equation.find("...");
            if (dot0 != std::string::npos)
            {
                size_t dot1 = equation.find("...", dot0);
                if (dot1 != std::string::npos)
                {
                    // replace dot pair with alphabet
                    const int input_rank = op->inputs[0]->shape.size();
                    if (input_rank > 0)
                    {
                        int missing_bits = input_rank - (arrow - 3);

                        std::string missing_bits_word;
                        for (int i = 0; i < missing_bits; i++)
                        {
                            missing_bits_word += '0' + i;
                        }

                        replaceAll(equation, "...", missing_bits_word);
                        equation_len = equation.size();
                        arrow = equation.find("->");
                    }
                }
            }

            std::string lhs = equation.substr(0, arrow);
            std::string rhs = equation.substr(arrow + 2);

            // normalize and sort to ijkl...
            std::vector<char> xs;
            for (size_t i = 0; i < rhs.size(); i++)
            {
                char x = rhs[i];

                if ((x >= '0' && x <= '9') || (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                {
                    if (xset.find(x) == xset.end())
                    {
                        xset[x] = '\0';
                        xs.push_back(x);
                    }
                }
            }

            for (size_t i = 0; i < lhs.size(); i++)
            {
                char x = lhs[i];

                if ((x >= '0' && x <= '9') || (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                {
                    if (xset.find(x) == xset.end())
                    {
                        xset[x] = '\0';
                        xs.push_back(x);
                    }
                }
            }

            for (auto& _xset : xset)
            {
                char x = _xset.first;
                size_t xsi = 0;
                for (size_t i = 0; i < xs.size(); i++)
                {
                    if (xs[i] == x)
                    {
                        xsi = i;
                        break;
                    }
                }

                _xset.second = 'i' + xsi;
            }
        }

        for (size_t i = 0; i < equation_len; i++)
        {
            char x = equation[i];

            if ((x >= '0' && x <= '9') || (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z'))
                equation[i] = xset[x];
        }

        // shorten jik->ijk to jik
        if (arrow != std::string::npos && op->inputs.size() == 1)
        {
            std::string lhs = equation.substr(0, arrow);
            std::string rhs = equation.substr(arrow + 2);

            if (lhs.size() == rhs.size())
            {
                equation = lhs;
            }
        }

        op->params["equation"] = equation;
    }
}

} // namespace pnnx
