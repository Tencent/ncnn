// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expression.h"

#include <stdio.h> // sscanf

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

int eval_list_expression(const std::string& expr, const std::vector<Mat>& blobs, std::vector<int>& outlist)
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

    // scan and stack
    std::stack<typed_value> exprstack;
    for (int i = (int)tokens.size() - 1; i >= 0; i--)
    {
        const std::string& t = tokens[i];

        // NCNN_LOGE("t = %s", t.c_str());

        // + - * / 0w 0h 0d 0c 12345

        if (t.size() == 2 && (t[0] >= '0' && t[0] <= '9') && (t[1] == 'w' || t[1] == 'h' || t[1] == 'd' || t[1] == 'c'))
        {
            size_t blob_index = t[0] - '0';
            if (blob_index >= blobs.size())
            {
                NCNN_LOGE("shape expression blob index %d out of bound!", (int)blob_index);
                return -1;
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

            // NCNN_LOGE("t = %s  =>  %d", t.c_str(), size);

            exprstack.push(size);
        }
        else if (t == "+" || t == "-" || t == "*" || t == "//" || t == "max" || t == "min")
        {
            typed_value ta = exprstack.top();
            exprstack.pop();
            typed_value tb = exprstack.top();
            exprstack.pop();

            if (ta.type == 0 && tb.type == 0)
            {
                const int a = ta.i;
                const int b = tb.i;

                int r = 0;
                if (t == "+")
                {
                    r = a + b;
                }
                else if (t == "-")
                {
                    r = a - b;
                }
                else if (t == "*")
                {
                    r = a * b;
                }
                else if (t == "//")
                {
                    if (b == 0)
                    {
                        NCNN_LOGE("expr divide by zero");
                        return -1;
                    }
                    else
                    {
                        r = a / b;
                    }
                }
                else if (t == "max")
                {
                    r = std::max(a, b);
                }
                else // if (t == "min")
                {
                    r = std::min(a, b);
                }
                exprstack.push(r);
            }
            else
            {
                const float a = ta.type == 0 ? ta.i : ta.f;
                const float b = tb.type == 0 ? tb.i : tb.f;

                float r = 0.f;
                if (t == "+")
                {
                    r = a + b;
                }
                else if (t == "-")
                {
                    r = a - b;
                }
                else if (t == "*")
                {
                    r = a * b;
                }
                else if (t == "//")
                {
                    r = floorf(a / b);
                }
                else if (t == "max")
                {
                    r = std::max(a, b);
                }
                else // if (t == "min")
                {
                    r = std::min(a, b);
                }
                exprstack.push(r);
            }
        }
        else if (t == "abs" || t == "neg" || t == "sign" || t == "square")
        {
            typed_value ta = exprstack.top();
            exprstack.pop();

            if (ta.type == 0)
            {
                const int a = ta.i;

                int r = 0;
                if (t == "abs")
                {
                    r = a > 0 ? a : -a;
                }
                else if (t == "neg")
                {
                    r = -a;
                }
                else if (t == "sign")
                {
                    r = a > 0 ? 1 : (a == 0 ? 0 : -1);
                }
                else // if (t == "square")
                {
                    r = a * a;
                }
                exprstack.push(r);
            }
            else
            {
                const float a = ta.f;

                float r = 0;
                if (t == "abs")
                {
                    r = fabsf(a);
                }
                else if (t == "neg")
                {
                    r = -a;
                }
                else if (t == "sign")
                {
                    r = a > 0.f ? 1 : (a == 0.f ? 0 : -1);
                }
                else // if (t == "square")
                {
                    r = a * a;
                }
                exprstack.push(r);
            }
        }
        else if (t == "trunc" || t == "ceil" || t == "floor" || t == "round")
        {
            typed_value ta = exprstack.top();
            exprstack.pop();

            if (ta.type == 0)
            {
                const int a = ta.i;
                exprstack.push(a);
            }
            else
            {
                const float a = ta.f;

                int r = 0;
                if (t == "trunc")
                {
                    r = (int)a;
                }
                else if (t == "ceil")
                {
                    r = (int)ceil(a);
                }
                else if (t == "floor")
                {
                    r = (int)floor(a);
                }
                else // if (t == "round")
                {
                    r = (int)round(a);
                }
                exprstack.push(r);
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
            typed_value ta = exprstack.top();
            exprstack.pop();

            const float a = ta.type == 0 ? ta.i : ta.f;

            float r = 0;
            if (t == "acos")
            {
                r = acosf(a);
            }
            else if (t == "acosh")
            {
                r = acoshf(a);
            }
            else if (t == "asin")
            {
                r = asinf(a);
            }
            else if (t == "asinh")
            {
                r = asinhf(a);
            }
            else if (t == "atan")
            {
                r = atanf(a);
            }
            else if (t == "atanh")
            {
                r = atanhf(a);
            }
            else if (t == "cos")
            {
                r = cosf(a);
            }
            else if (t == "cosh")
            {
                r = coshf(a);
            }
            else if (t == "erf")
            {
                r = erff(a);
            }
            else if (t == "exp")
            {
                r = expf(a);
            }
            else if (t == "log")
            {
                r = logf(a);
            }
            else if (t == "log10")
            {
                r = log10f(a);
            }
            else if (t == "reciprocal")
            {
                r = 1.f / a;
            }
            else if (t == "rsqrt")
            {
                r = 1.f / sqrtf(a);
            }
            else if (t == "sin")
            {
                r = sinf(a);
            }
            else if (t == "sinh")
            {
                r = sinhf(a);
            }
            else if (t == "sqrt")
            {
                r = sqrtf(a);
            }
            else if (t == "tan")
            {
                r = tanf(a);
            }
            else // if (t == "tanh")
            {
                r = tanhf(a);
            }
            exprstack.push(r);
        }
        else if (t == "/"
                 || t == "atan2"
                 || t == "fmod"
                 || t == "pow"
                 || t == "remainder"
                 || t == "logaddexp")
        {
            typed_value ta = exprstack.top();
            exprstack.pop();
            typed_value tb = exprstack.top();
            exprstack.pop();

            const float a = ta.type == 0 ? ta.i : ta.f;
            const float b = tb.type == 0 ? tb.i : tb.f;

            float r = 0.f;
            if (t == "/")
            {
                r = a / b;
            }
            else if (t == "atan2")
            {
                r = atan2f(a, b);
            }
            else if (t == "fmod")
            {
                r = fmodf(a, b);
            }
            else if (t == "pow")
            {
                r = powf(a, b);
            }
            else if (t == "remainder")
            {
                r = fmodf(a, b);
                if (a * b < 0)
                    r += b;
            }
            else // if (t == "logaddexp")
            {
                r = logf(expf(a) + expf(b));
            }
            exprstack.push(r);
        }
        else if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
        {
            typed_value ta = exprstack.top();
            exprstack.pop();
            typed_value tb = exprstack.top();
            exprstack.pop();

            // assert ta.type == 0 && tb.type == 0

            const int a = ta.i;
            const int b = tb.i;

            int r = 0;
            if (t == "and")
            {
                r = a & b;
            }
            else if (t == "or")
            {
                r = a | b;
            }
            else if (t == "xor")
            {
                r = a ^ b;
            }
            else if (t == "lshift")
            {
                r = a << b;
            }
            else // if (t == "rshift")
            {
                r = a >> b;
            }
            exprstack.push(r);
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
                exprstack.push(vi);
            }
            else if (nscanf == 1)
            {
                exprstack.push(vf);
            }
            else
            {
                NCNN_LOGE("malformed literal token %s", t.c_str());
                return -1;
            }
        }
    }

    int size = exprstack.top().to_int();
    exprstack.pop();
    outlist.push_back(size);
    while (!exprstack.empty())
    {
        size = exprstack.top().to_int();
        exprstack.pop();
        outlist.push_back(size);
    }

    // NCNN_LOGE("shape %s = %d %d", expr.c_str(), list[0], list[1]);

    return 0;
}

} // namespace ncnn
