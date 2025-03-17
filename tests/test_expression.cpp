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

#include <stdio.h>
#include <stdarg.h>

#include "expression.h"

static int test_count_expression_blobs(const std::string& expr, int true_count)
{
    int count = ncnn::count_expression_blobs(expr);
    if (count != true_count)
    {
        fprintf(stderr, "test_count_expression_blobs failed expr=%s got %d\n", expr.c_str(), count);
        return -1;
    }

    return 0;
}

static int test_expression_0()
{
    return 0
           || test_count_expression_blobs("1,2,3,4,5,6", 0)
           || test_count_expression_blobs("-1,1h,2w", 3)
           || test_count_expression_blobs("2,9d,2c,-1", 10);
}

static int test_eval_list_expression(const std::string& expr, std::vector<ncnn::Mat>& blobs, int ndim, ...)
{
    // construct true list
    std::vector<int> true_list(ndim);
    va_list ap;
    va_start(ap, ndim);
    for (int i = 0; i < ndim; i++)
    {
        true_list[i] = va_arg(ap, int);
    }
    va_end(ap);

    std::vector<int> list;
    int er = ncnn::eval_list_expression(expr, blobs, list);
    if (er != 0)
        return -1;

    bool failed = false;
    if (list.size() != true_list.size())
    {
        failed = true;
    }
    else
    {
        for (size_t i = 0; i < list.size(); i++)
        {
            if (list[i] != true_list[i])
            {
                failed = true;
                break;
            }
        }
    }

    if (failed)
    {
        fprintf(stderr, "test_eval_list_expression failed expr=%s got [", expr.c_str());
        for (size_t i = 0; i < list.size(); i++)
        {
            fprintf(stderr, "%d", list[i]);
            if (i + 1 != list.size())
                fprintf(stderr, ",");
        }
        fprintf(stderr, "]\n");
        return -1;
    }

    return 0;
}

static int test_expression_1()
{
    std::vector<ncnn::Mat> blobs(2);
    blobs[0] = ncnn::Mat(100, 200, 44);
    blobs[1] = ncnn::Mat(10, 20, 2, 4);

    return 0
           || test_eval_list_expression("+(trunc(*(0w,0.5)),-(0c,10)),floor(/(1h,0.5)),+(0c,1c),round(2.0)", blobs, 4, 84, 40, 48, 2)
           || test_eval_list_expression("//(0w,3),+(0w,1w),-(0h,1h),*(0c,1c)", blobs, 4, 33, 110, 180, 176)
           || test_eval_list_expression("floor(//(0w,2.99)),round(+(0w,1.01)),trunc(-(0h,1.9)),ceil(*(1d,2.99))", blobs, 4, 33, 101, 198, 6)
           || test_eval_list_expression("round(*(abs(asin(sin(0w))),10.11)),ceil(*(abs(acos(cos(+(0w,3)))),10.11))", blobs, 2, 5, 25)
           || test_eval_list_expression("floor(*(abs(asinh(sinh(/(0w,100)))),10.11)),trunc(*(abs(acosh(cosh(*(0w,0.004)))),10.11))", blobs, 2, 10, 4)
           || test_eval_list_expression("round(*(abs(atan(tan(0w))),10.11)),ceil(*(abs(atanh(tanh(-(0w,99)))),10.11))", blobs, 2, 5, 11)
           || test_eval_list_expression("floor(min(max(*(square(sqrt(0w)),1.2121),100),120))", blobs, 1, 120)
           || test_eval_list_expression("min(max(trunc(*(log(exp(*(neg(0w),0.001))),-144)),15),20)", blobs, 1, 15)
           || test_eval_list_expression("round(*(erf(reciprocal(log10(1h))),999))", blobs, 1, 722)
           || test_eval_list_expression("ceil(pow(fmod(atan2(0w,1d),1c),14.14)),floor(logaddexp(remainder(0c,10),6))", blobs, 2, 495, 6)
           || test_eval_list_expression("floor(*(square(sqrt(0w)),1.2121))", blobs, 1, 121)
           || test_eval_list_expression("rshift(lshift(xor(or(and(1d,18),9),4),4),2)", blobs, 1, 60)
           || test_eval_list_expression("ceil(*(rsqrt(+(+(sign(1w),10),*(sign(-(neg(1d)),0.5),3))),100))", blobs, 1, 36);
}

static int test_expression_2()
{
    std::vector<ncnn::Mat> blobs(2);
    blobs[0] = ncnn::Mat(10, 20, 4);
    blobs[1] = ncnn::Mat(1, 2, 3, 4);

    // expect error blob index out of bound
    if (test_eval_list_expression("0w,1h,2c,1d", blobs, 0) != -1)
        return -1;

    // expect error divide by zero
    if (test_eval_list_expression("//(0w,-(0c,1c))", blobs, 0) != -1)
        return -1;

    // expect error malformed token
    if (test_eval_list_expression("1c,#(0w,1)", blobs, 0) != -1)
        return -1;
    if (test_eval_list_expression("1c,+(qwq,1w)", blobs, 0) != -1)
        return -1;

    return 0;
}

int main()
{
    return 0
           || test_expression_0()
           || test_expression_1()
           || test_expression_2();
}
