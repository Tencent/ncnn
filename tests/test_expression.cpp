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
           || test_count_expression_blobs("2,9d,2c,-1", 10)
           ;
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

    std::vector<int> list = ncnn::eval_list_expression(expr, blobs);
    if (list != true_list)
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
           ;
}

int main()
{
    return 0
           || test_expression_0()
           || test_expression_1();
}
