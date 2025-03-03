// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "expression.h"

namespace ncnn {

int count_expression_blobs(const std::string& expr)
{
    int count = 0;

    std::string t;
    for (size_t i = 0; i < expr.size(); i++)
    {
        char ch = expr[i];

        if (ch == '(' || ch == ')' || ch == ',')
        {
            if (!t.empty())
            {
                if (t.size() == 2 && (t[0] >= '0' && t[0] <= '9') && (t[1] == 'w' || t[1] == 'h' || t[1] == 'd' || t[1] == 'c'))
                {
                    int blob_index = t[0] - '0';
                    count = std::max(count, blob_index + 1);
                }

                t.clear();
            }
        }
        else
        {
#if NCNN_SIMPLESTL
            t.resize(t.size() + 1);
            t[t.size() - 1] = ch;
#else
            t += ch;
#endif
        }
    }

    if (!t.empty())
    {
        if (t.size() == 2 && (t[0] >= '0' && t[0] <= '9') && (t[1] == 'w' || t[1] == 'h' || t[1] == 'd' || t[1] == 'c'))
        {
            int blob_index = t[0] - '0';
            count = std::max(count, blob_index + 1);
        }
    }

    return count;
}

std::vector<int> eval_list_expression(const std::string& expr, const std::vector<Mat>& blobs)
{
    // /(0w,2),*(0h,2),0c

    // split by , ( )
    //
    //     /
    //         0w
    //         2
    // -------------------
    //     *
    //         0h
    //         2
    // -------------------
    //     0c
    // -------------------

    // split by , ( )

    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (size_t i = 0; i < expr.size(); i++)
        {
            char ch = expr[i];

            if (ch == '(' || ch == ')' || ch == ',')
            {
                if (!t.empty())
                {
                    tokens.push_back(t);
                    t.clear();
                }
            }
            else
            {
#if NCNN_SIMPLESTL
                t.resize(t.size() + 1);
                t[t.size() - 1] = ch;
#else
                t += ch;
#endif
            }
        }

        if (!t.empty())
        {
            tokens.push_back(t);
        }
    }

    //      / 0w 2 * 0h 2 0c

    struct typed_value
    {
        int type; // 0=i 1=f
        union
        {
            int i;
            float f;
        };

        typed_value()
            : type(0), i(0)
        {
        }
        typed_value(int _i)
            : type(0), i(_i)
        {
        }
        typed_value(float _f)
            : type(1), f(_f)
        {
        }

        int to_int()
        {
            if (type == 0)
                return i;

            // trunc by default
            return (int)f;
        }
    };

    // scan and stack
    std::vector<typed_value> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        // + - * / 0w 0h 0d 0c 12345

        if (t.size() == 2 && (t[0] >= '0' && t[0] <= '9') && (t[1] == 'w' || t[1] == 'h' || t[1] == 'd' || t[1] == 'c'))
        {
            size_t blob_index = t[0] - '0';
            if (blob_index >= blobs.size())
            {
                NCNN_LOGE("shape expression blob index %d out of bound!", (int)blob_index);
                blob_index = 0;
            }

            const Mat& blob = blobs[blob_index].shape();
            int size;
            if (t[1] == 'w')
                size = blob.w;
            else if (t[1] == 'h')
                size = blob.h;
            else if (t[1] == 'd')
                size = blob.d;
            else // if (t[1] == 'c')
                size = blob.c;

            exprstack.push_back(size);
        }
        else if (t == "+" || t == "-" || t == "*" || t == "//" || t == "max" || t == "min")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
            typed_value tb = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
            typed_value tb = exprstack.back();
            exprstack.pop_back();
#endif

            if (ta.type == 0 && tb.type == 0)
            {
                int a = ta.i;
                int b = tb.i;

                if (t == "+")
                {
                    exprstack.push_back(a + b);
                }
                if (t == "-")
                {
                    exprstack.push_back(a - b);
                }
                if (t == "*")
                {
                    exprstack.push_back(a * b);
                }
                if (t == "//")
                {
                    if (b == 0)
                    {
                        NCNN_LOGE("expr divide by zero");
                        exprstack.push_back(a);
                    }
                    else
                    {
                        exprstack.push_back(a / b);
                    }
                }
                if (t == "max")
                {
                    exprstack.push_back(std::max(a, b));
                }
                if (t == "min")
                {
                    exprstack.push_back(std::min(a, b));
                }
            }
            else
            {
                float a = ta.type == 0 ? ta.i : ta.f;
                float b = tb.type == 0 ? tb.i : tb.f;

                if (t == "+")
                {
                    exprstack.push_back(a + b);
                }
                if (t == "-")
                {
                    exprstack.push_back(a - b);
                }
                if (t == "*")
                {
                    exprstack.push_back(a * b);
                }
                if (t == "//")
                {
                    exprstack.push_back(floorf(a / b));
                }
                if (t == "max")
                {
                    exprstack.push_back(std::max(a, b));
                }
                if (t == "min")
                {
                    exprstack.push_back(std::min(a, b));
                }
            }
        }
        else if (t == "abs" || t == "neg" || t == "sign" || t == "square")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
#endif

            if (ta.type == 0)
            {
                int a = ta.i;

                if (t == "abs")
                {
                    exprstack.push_back(a > 0 ? a : -a);
                }
                if (t == "neg")
                {
                    exprstack.push_back(-a);
                }
                if (t == "sign")
                {
                    exprstack.push_back(a > 0 ? 1 : (a == 0 ? 0 : -1));
                }
                if (t == "square")
                {
                    exprstack.push_back(a * a);
                }
            }
            else
            {
                float a = ta.f;

                if (t == "abs")
                {
                    exprstack.push_back(fabsf(a));
                }
                if (t == "neg")
                {
                    exprstack.push_back(-a);
                }
                if (t == "sign")
                {
                    exprstack.push_back(a > 0.f ? 1 : (a == 0.f ? 0 : -1));
                }
                if (t == "square")
                {
                    exprstack.push_back(a * a);
                }
            }
        }
        else if (t == "trunc" || t == "ceil" || t == "floor" || t == "round")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
#endif

            if (ta.type == 0)
            {
                int a = ta.i;
                exprstack.push_back(a);
            }
            else
            {
                float a = ta.f;

                if (t == "trunc")
                {
                    exprstack.push_back((int)a);
                }
                if (t == "ceil")
                {
                    exprstack.push_back((int)ceil(a));
                }
                if (t == "floor")
                {
                    exprstack.push_back((int)floor(a));
                }
                if (t == "round")
                {
                    exprstack.push_back((int)round(a));
                }
            }
        }
        else if (t == "acos"
                 || t == "acosh"
                 || t == "asin"
                 || t == "asinh"
                 || t == "atan"
                 || t == "atanh"
                 || t == "cos"
                 || t == "cosh"
                 || t == "erf"
                 || t == "exp"
                 || t == "log"
                 || t == "log10"
                 || t == "reciprocal"
                 || t == "rsqrt"
                 || t == "sin"
                 || t == "sinh"
                 || t == "sqrt"
                 || t == "tan"
                 || t == "tanh")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
#endif

            float a = ta.type == 0 ? ta.i : ta.f;

            if (t == "acos")
            {
                exprstack.push_back(acosf(a));
            }
            if (t == "acosh")
            {
                exprstack.push_back(acoshf(a));
            }
            if (t == "asin")
            {
                exprstack.push_back(asinf(a));
            }
            if (t == "asinh")
            {
                exprstack.push_back(asinhf(a));
            }
            if (t == "atan")
            {
                exprstack.push_back(atanf(a));
            }
            if (t == "atanh")
            {
                exprstack.push_back(atanhf(a));
            }
            if (t == "cos")
            {
                exprstack.push_back(cosf(a));
            }
            if (t == "cosh")
            {
                exprstack.push_back(coshf(a));
            }
            if (t == "erf")
            {
                exprstack.push_back(erff(a));
            }
            if (t == "exp")
            {
                exprstack.push_back(expf(a));
            }
            if (t == "log")
            {
                exprstack.push_back(logf(a));
            }
            if (t == "log10")
            {
                exprstack.push_back(log10f(a));
            }
            if (t == "reciprocal")
            {
                exprstack.push_back(1.f / a);
            }
            if (t == "rsqrt")
            {
                exprstack.push_back(1.f / sqrtf(a));
            }
            if (t == "sin")
            {
                exprstack.push_back(sinf(a));
            }
            if (t == "sinh")
            {
                exprstack.push_back(sinhf(a));
            }
            if (t == "sqrt")
            {
                exprstack.push_back(sqrtf(a));
            }
            if (t == "tan")
            {
                exprstack.push_back(tanf(a));
            }
            if (t == "tanh")
            {
                exprstack.push_back(tanhf(a));
            }
        }
        else if (t == "/"
                 || t == "atan2"
                 || t == "fmod"
                 || t == "pow"
                 || t == "remainder"
                 || t == "logaddexp")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
            typed_value tb = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
            typed_value tb = exprstack.back();
            exprstack.pop_back();
#endif

            float a = ta.type == 0 ? ta.i : ta.f;
            float b = tb.type == 0 ? tb.i : tb.f;

            if (t == "/")
            {
                exprstack.push_back(a / b);
            }
            if (t == "atan2")
            {
                exprstack.push_back(atan2f(a, b));
            }
            if (t == "fmod")
            {
                exprstack.push_back(fmodf(a, b));
            }
            if (t == "pow")
            {
                exprstack.push_back(powf(a, b));
            }
            if (t == "remainder")
            {
                float r = fmodf(a, b);
                if (a * b < 0)
                    r += b;
                exprstack.push_back(r);
            }
            if (t == "logaddexp")
            {
                exprstack.push_back(logf(expf(a) + expf(b)));
            }
        }
        else if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
        {
#if NCNN_SIMPLESTL
            typed_value ta = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
            typed_value tb = exprstack[exprstack.size() - 1];
            exprstack.resize(exprstack.size() - 1);
#else
            typed_value ta = exprstack.back();
            exprstack.pop_back();
            typed_value tb = exprstack.back();
            exprstack.pop_back();
#endif

            // assert ta.type == 0 && tb.type == 0

            int a = ta.i;
            int b = tb.i;

            if (t == "and")
            {
                exprstack.push_back(a & b);
            }
            if (t == "or")
            {
                exprstack.push_back(a | b);
            }
            if (t == "xor")
            {
                exprstack.push_back(a ^ b);
            }
            if (t == "lshift")
            {
                exprstack.push_back(a << b);
            }
            if (t == "rshift")
            {
                exprstack.push_back(a >> b);
            }
        }
        else
        {
            // literal
            int vi;
            float vf;
            int nscani = sscanf(t.c_str(), "%d", &vi);
            int nscanf = sscanf(t.c_str(), "%f", &vf);
            if (nscani == 1 && nscanf == 1 && vi == vf)
            {
                exprstack.push_back(vi);
            }
            else if (nscanf == 1)
            {
                exprstack.push_back(vf);
            }
            else
            {
                NCNN_LOGE("malformed literal token %s", t.c_str());
                exprstack.push_back(0);
            }
        }
    }

    std::vector<int> list;
#if NCNN_SIMPLESTL
    int size = exprstack[exprstack.size() - 1].to_int();
    exprstack.resize(exprstack.size() - 1);
#else
    int size = exprstack.back().to_int();
    exprstack.pop_back();
#endif
    list.push_back(size);
    while (!exprstack.empty())
    {
#if NCNN_SIMPLESTL
        size = exprstack[exprstack.size() - 1].to_int();
        exprstack.resize(exprstack.size() - 1);
#else
        size = exprstack.back().to_int();
        exprstack.pop_back();
#endif
        list.push_back(size);
    }

    // NCNN_LOGE("shape %s = %d %d", expr.c_str(), list[0], list[1]);

    return list;
}

} // namespace ncnn
