// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "testutil.h"

static ncnn::Mat IntArrayMat(int a0, int a1, int a2, int a3)
{
    ncnn::Mat m(4);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    p[3] = a3;
    return m;
}

static int test_gather(const ncnn::Mat& a, const ncnn::Mat& index, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(2);
    a0[0] = a;
    a0[1] = index;

    int ret = test_layer("Gather", pd, weights, a0);
    if (ret != 0)
    {
        fprintf(stderr, "test_gather failed a.dims=%d a=(%d %d %d %d) index.w=%d axis=%d\n", a.dims, a.w, a.h, a.d, a.c, index.w, axis);
    }

    return ret;
}

// 生成验证的index
static ncnn::Mat IntMat(int dims, int axis)
{
    if (dims == 1)
    {
        return IntArrayMat(1, 7, 9, 15);
    }
    else if (dims == 2)
    {
        if (axis == 0)
        {
            static const int int_data[] = {
                // 1, 3, 2,
                // 0, 3, 1,
                1,
                3,
                2,
                0,
                3,
                1,
            };
            ncnn::Mat in1(6);
            int* ptr1 = (int*)in1.data;
            memcpy(ptr1, int_data, sizeof(int_data));
            in1 = in1.reshape(3, 2); // 3D
            return in1;
        }
        else
        {
            static const int int_data[] = {
                // 1, 3, 2, 4, 6, 5,
                // 4, 3, 2, 1, 5, 6,
                1,
                3,
                2,
                4,
                6,
                5,
                4,
                3,
                2,
                1,
                5,
                6,
            };
            ncnn::Mat in1(12);
            int* ptr1 = (int*)in1.data;
            memcpy(ptr1, int_data, sizeof(int_data));
            in1 = in1.reshape(6, 2); // 3D
            return in1;
        }
    }
    else if (dims == 3)
    {
        if (axis == 0)
        {
            static const int int_data[] = {
                // 0, 1, 0,
                // 1, 0, 1,
                0,
                1,
                0,
                1,
                0,
                1,
            };
            ncnn::Mat in1(6);
            int* ptr1 = (int*)in1.data;
            memcpy(ptr1, int_data, sizeof(int_data));
            in1 = in1.reshape(1, 3, 2); // 3D
            return in1;
        }
        else if (axis == 1)
        {
            static const int int_data[] = {
                // 0, 1, 2,
                // 1, 2, 0,
                0,
                1,
                2,
                1,
                2,
                0,
            };
            ncnn::Mat in1(6);
            int* ptr1 = (int*)in1.data;
            memcpy(ptr1, int_data, sizeof(int_data));
            in1 = in1.reshape(1, 3, 2); // 3D
            return in1;
        }
        else if (axis == 2)
        {
            static const int int_data[] = {
                // 0, 1, 2, 3,
                // 2, 1, 0, 3,
                0,
                1,
                2,
                3,
                2,
                1,
                0,
                3,
            };
            ncnn::Mat in1(8);
            int* ptr1 = (int*)in1.data;
            memcpy(ptr1, int_data, sizeof(int_data));
            in1 = in1.reshape(4, 1, 2); // 3D
            return in1;
        }
    }
    else if (dims == 4)
    {
        static const int int_data[] = {
            // 0, 1, 0, 1, 0,
            // 1, 0, 1, 0, 1,
            // 0, 1, 0, 1, 0,
            // 1, 0, 1, 0, 1,

            // 1, 0, 1, 0, 1,
            // 0, 1, 0, 1, 0,
            // 1, 0, 1, 0, 1,
            // 0, 1, 0, 1, 0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,

            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
        };
        ncnn::Mat in1(40);
        int* ptr1 = (int*)in1.data;
        memcpy(ptr1, int_data, sizeof(int_data));
        in1 = in1.reshape(5, 4, 1, 2); // 3D
        return in1;
    }

    return ncnn::Mat();
}

static int test_gather_0()
{
    return 0
           || test_gather(RandomMat(3, 4, 5, 6), IntMat(4, 0), 0)
           || test_gather(RandomMat(4, 4, 5, 6), IntMat(4, 1), 1)
           || test_gather(RandomMat(3, 4, 7, 6), IntMat(4, 2), 2)
           || test_gather(RandomMat(7, 2, 5, 6), IntMat(4, 3), 3);
}

static int test_gather_1()
{
    return 0
           || test_gather(RandomMat(8, 4, 5), IntMat(3, 0), -3)
           || test_gather(RandomMat(4, 7, 5), IntMat(3, 1), -2)
           || test_gather(RandomMat(6, 4, 5), IntMat(3, 2), -1);
}

static int test_gather_2()
{
    return 0
           || test_gather(RandomMat(8, 6), IntMat(2, 0), 0)
           || test_gather(RandomMat(8, 7), IntMat(2, 1), 1);
}

static int test_gather_3()
{
    return 0
           || test_gather(RandomMat(18), IntArrayMat(1, 7, 9, 15), -1);
}

int main()
{
    SRAND(7767517);
    return 0
           || test_gather_0()
           || test_gather_1()
           || test_gather_2()
           || test_gather_3();
}