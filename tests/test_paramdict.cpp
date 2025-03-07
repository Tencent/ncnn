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

#include "datareader.h"
#include "paramdict.h"

class ParamDictTest : public ncnn::ParamDict
{
public:
    int load_param(const char* str);
    int load_param_bin(const unsigned char* mem);
};

int ParamDictTest::load_param(const char* str)
{
    const unsigned char* mem = (const unsigned char*)str;
    ncnn::DataReaderFromMemory dr(mem);
    return ncnn::ParamDict::load_param(dr);
}

int ParamDictTest::load_param_bin(const unsigned char* mem)
{
    ncnn::DataReaderFromMemory dr(mem);
    return ncnn::ParamDict::load_param_bin(dr);
}

static int test_paramdict_0()
{
    ParamDictTest pdt;
    pdt.load_param("0=100 1=1,-1,4,5,1,4 2=1.250000 -23303=5,0.1,0.2,-0.4,0.8,1.0 -23304=3,-1,10,-88");

    // int
    int typei = pdt.type(0);
    if (typei != 2)
    {
        fprintf(stderr, "test_paramdict int type failed %d != 2\n", typei);
        return -1;
    }
    int i = pdt.get(0, 0);
    if (i != 100)
    {
        fprintf(stderr, "test_paramdict int value failed %d != 100\n", i);
        return -1;
    }

    // int array
    int typeai = pdt.type(1);
    if (typeai != 5)
    {
        fprintf(stderr, "test_paramdict int array type failed %d != 5\n", typeai);
        return -1;
    }
    ncnn::Mat ai = pdt.get(1, ncnn::Mat());
    if (ai.w != 6)
    {
        fprintf(stderr, "test_paramdict int array size failed %d != 6\n", ai.w);
        return -1;
    }
    const int* p = ai;
    if (p[0] != 1 || p[1] != -1 || p[2] != 4 || p[3] != 5 || p[4] != 1 || p[5] != 4)
    {
        fprintf(stderr, "test_paramdict int array value failed %d %d %d %d %d %d\n", p[0], p[1], p[2], p[3], p[4], p[5]);
        return -1;
    }

    // float
    int typef = pdt.type(2);
    if (typef != 3)
    {
        fprintf(stderr, "test_paramdict float type failed %d != 3\n", typef);
        return -1;
    }
    float f = pdt.get(2, 0.f);
    if (f != 1.25f)
    {
        fprintf(stderr, "test_paramdict float value failed %f != 1.25f\n", f);
        return -1;
    }

    // float array
    int typeaf = pdt.type(3);
    if (typeaf != 6)
    {
        fprintf(stderr, "test_paramdict float array type failed %d != 6\n", typeaf);
        return -1;
    }
    ncnn::Mat af = pdt.get(3, ncnn::Mat());
    if (af.w != 5)
    {
        fprintf(stderr, "test_paramdict float array size failed %d != 5\n", af.w);
        return -1;
    }
    if (af[0] != 0.1f || af[1] != 0.2f || af[2] != -0.4f || af[3] != 0.8f || af[4] != 1.0f)
    {
        fprintf(stderr, "test_paramdict float array value failed %f %f %f %f %f\n", af[0], af[1], af[2], af[3], af[4]);
        return -1;
    }

    // int array
    typeai = pdt.type(4);
    if (typeai != 5)
    {
        fprintf(stderr, "test_paramdict int array type failed %d != 5\n", typeai);
        return -1;
    }
    ai = pdt.get(4, ncnn::Mat());
    if (ai.w != 3)
    {
        fprintf(stderr, "test_paramdict int array size failed %d != 3\n", ai.w);
        return -1;
    }
    p = ai;
    if (p[0] != -1 || p[1] != 10 || p[2] != -88)
    {
        fprintf(stderr, "test_paramdict int array value failed %d %d %d\n", p[0], p[1], p[2]);
        return -1;
    }

    return 0;
}

static int test_paramdict_1()
{
    ParamDictTest pdt;
    pdt.load_param("0=-1 1=4, 2=0.01 3=-1.45e-2,3.14");

    // int
    int typei = pdt.type(0);
    if (typei != 2)
    {
        fprintf(stderr, "test_paramdict int type failed %d != 2\n", typei);
        return -1;
    }
    int i = pdt.get(0, 0);
    if (i != -1)
    {
        fprintf(stderr, "test_paramdict int value failed %d != -1\n", i);
        return -1;
    }

    // int array
    int typeai = pdt.type(1);
    if (typeai != 5)
    {
        fprintf(stderr, "test_paramdict int array type failed %d != 5\n", typeai);
        return -1;
    }
    ncnn::Mat ai = pdt.get(1, ncnn::Mat());
    if (ai.w != 1)
    {
        fprintf(stderr, "test_paramdict int array size failed %d != 1\n", ai.w);
        return -1;
    }
    const int* p = ai;
    if (p[0] != 4)
    {
        fprintf(stderr, "test_paramdict int array value failed %d\n", p[0]);
        return -1;
    }

    // float
    int typef = pdt.type(2);
    if (typef != 3)
    {
        fprintf(stderr, "test_paramdict float type failed %d != 3\n", typef);
        return -1;
    }
    float f = pdt.get(2, 0.f);
    if (f != 0.01f)
    {
        fprintf(stderr, "test_paramdict float value failed %f != 0.01f\n", f);
        return -1;
    }

    // float array
    int typeaf = pdt.type(3);
    if (typeaf != 6)
    {
        fprintf(stderr, "test_paramdict float array type failed %d != 6\n", typeaf);
        return -1;
    }
    ncnn::Mat af = pdt.get(3, ncnn::Mat());
    if (af.w != 2)
    {
        fprintf(stderr, "test_paramdict float array size failed %d != 2\n", af.w);
        return -1;
    }
    if (af[0] != -0.0145f || af[1] != 3.14f)
    {
        fprintf(stderr, "test_paramdict float array value failed %f %f\n", af[0], af[1]);
        return -1;
    }

    return 0;
}

static int test_paramdict_2()
{
    ParamDictTest pdt;
    pdt.load_param("0=bij,bjk->bik 1=This_is_a_very_long_long_string 3=\"1,2,3 and 6.667          zzz\" 2=\"X\"");

    // string
    int types = pdt.type(0);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    std::string s = pdt.get(0, "");
    if (s != "bij,bjk->bik")
    {
        fprintf(stderr, "test_paramdict string text failed %s != bij,bjk->bik\n", s.c_str());
        return -1;
    }

    // string
    types = pdt.type(1);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(1, "");
    if (s != "This_is_a_very_long_long_string")
    {
        fprintf(stderr, "test_paramdict string text failed %s != This_is_a_very_long_long_string\n", s.c_str());
        return -1;
    }

    // string
    types = pdt.type(2);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(2, "");
    if (s != "X")
    {
        fprintf(stderr, "test_paramdict string text failed %s != X\n", s.c_str());
        return -1;
    }

    // string
    types = pdt.type(3);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(3, "");
    if (s != "1,2,3 and 6.667          zzz")
    {
        fprintf(stderr, "test_paramdict string text failed %s != \"1,2,3 and 6.667          zzz\"\n", s.c_str());
        return -1;
    }

    return 0;
}

static int test_paramdict_3()
{
    const unsigned char mem[] = {
        0x00, 0x00, 0x00, 0x00,
        0x64, 0x00, 0x00, 0x00,
        0xfb, 0xa4, 0xff, 0xff,
        0x06, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff,
        0x04, 0x00, 0x00, 0x00,
        0x05, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xa0, 0x3f,
        0xf9, 0xa4, 0xff, 0xff,
        0x05, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0xcc, 0x3d,
        0xcd, 0xcc, 0x4c, 0x3e,
        0xcd, 0xcc, 0xcc, 0xbe,
        0xcd, 0xcc, 0x4c, 0x3f,
        0x00, 0x00, 0x80, 0x3f,
        0x17, 0xff, 0xff, 0xff
    };

    ParamDictTest pdt;
    pdt.load_param_bin(mem);

    // int
    int typei = pdt.type(0);
    if (typei != 1)
    {
        fprintf(stderr, "test_paramdict int type failed %d != 1\n", typei);
        return -1;
    }
    int i = pdt.get(0, 0);
    if (i != 100)
    {
        fprintf(stderr, "test_paramdict int value failed %d != 100\n", i);
        return -1;
    }

    // int array
    int typeai = pdt.type(1);
    if (typeai != 4)
    {
        fprintf(stderr, "test_paramdict int array type failed %d != 4\n", typeai);
        return -1;
    }
    ncnn::Mat ai = pdt.get(1, ncnn::Mat());
    if (ai.w != 6)
    {
        fprintf(stderr, "test_paramdict int array size failed %d != 6\n", ai.w);
        return -1;
    }
    const int* p = ai;
    if (p[0] != 1 || p[1] != -1 || p[2] != 4 || p[3] != 5 || p[4] != 1 || p[5] != 4)
    {
        fprintf(stderr, "test_paramdict int array value failed %d %d %d %d %d %d\n", p[0], p[1], p[2], p[3], p[4], p[5]);
        return -1;
    }

    // float
    int typef = pdt.type(2);
    if (typef != 1)
    {
        fprintf(stderr, "test_paramdict float type failed %d != 1\n", typef);
        return -1;
    }
    float f = pdt.get(2, 0.f);
    if (f != 1.25f)
    {
        fprintf(stderr, "test_paramdict float value failed %f != 1.25f\n", f);
        return -1;
    }

    // float array
    int typeaf = pdt.type(3);
    if (typeaf != 4)
    {
        fprintf(stderr, "test_paramdict float array type failed %d != 4\n", typeaf);
        return -1;
    }
    ncnn::Mat af = pdt.get(3, ncnn::Mat());
    if (af.w != 5)
    {
        fprintf(stderr, "test_paramdict float array size failed %d != 5\n", af.w);
        return -1;
    }
    if (af[0] != 0.1f || af[1] != 0.2f || af[2] != -0.4f || af[3] != 0.8f || af[4] != 1.0f)
    {
        fprintf(stderr, "test_paramdict float array value failed %f %f %f %f %f\n", af[0], af[1], af[2], af[3], af[4]);
        return -1;
    }

    return 0;
}

static int test_paramdict_4()
{
    const unsigned char mem[] = {
        0x00, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff,
        0xfb, 0xa4, 0xff, 0xff,
        0x01, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00,
        0x0a, 0xd7, 0x23, 0x3c,
        0xf9, 0xa4, 0xff, 0xff,
        0x02, 0x00, 0x00, 0x00,
        0x68, 0x91, 0x6d, 0xbc,
        0xc3, 0xf5, 0x48, 0x40,
        0x17, 0xff, 0xff, 0xff
    };

    ParamDictTest pdt;
    pdt.load_param_bin(mem);

    // int
    int typei = pdt.type(0);
    if (typei != 1)
    {
        fprintf(stderr, "test_paramdict int type failed %d != 1\n", typei);
        return -1;
    }
    int i = pdt.get(0, 0);
    if (i != -1)
    {
        fprintf(stderr, "test_paramdict int value failed %d != -1\n", i);
        return -1;
    }

    // int array
    int typeai = pdt.type(1);
    if (typeai != 4)
    {
        fprintf(stderr, "test_paramdict int array type failed %d != 4\n", typeai);
        return -1;
    }
    ncnn::Mat ai = pdt.get(1, ncnn::Mat());
    if (ai.w != 1)
    {
        fprintf(stderr, "test_paramdict int array size failed %d != 1\n", ai.w);
        return -1;
    }
    const int* p = ai;
    if (p[0] != 4)
    {
        fprintf(stderr, "test_paramdict int array value failed %d\n", p[0]);
        return -1;
    }

    // float
    int typef = pdt.type(2);
    if (typef != 1)
    {
        fprintf(stderr, "test_paramdict float type failed %d != 1\n", typef);
        return -1;
    }
    float f = pdt.get(2, 0.f);
    if (f != 0.01f)
    {
        fprintf(stderr, "test_paramdict float value failed %f != 0.01f\n", f);
        return -1;
    }

    // float array
    int typeaf = pdt.type(3);
    if (typeaf != 4)
    {
        fprintf(stderr, "test_paramdict float array type failed %d != 4\n", typeaf);
        return -1;
    }
    ncnn::Mat af = pdt.get(3, ncnn::Mat());
    if (af.w != 2)
    {
        fprintf(stderr, "test_paramdict float array size failed %d != 2\n", af.w);
        return -1;
    }
    if (af[0] != -0.0145f || af[1] != 3.14f)
    {
        fprintf(stderr, "test_paramdict float array value failed %f %f\n", af[0], af[1]);
        return -1;
    }

    return 0;
}

static int test_paramdict_5()
{
    const unsigned char mem[] = {
        0x98, 0xa4, 0xff, 0xff,
        0x0c, 0x00, 0x00, 0x00,
        0x62, 0x69, 0x6a, 0x2c,
        0x62, 0x6a, 0x6b, 0x2d,
        0x3e, 0x62, 0x69, 0x6b,
        0x97, 0xa4, 0xff, 0xff,
        0x1f, 0x00, 0x00, 0x00,
        0x54, 0x68, 0x69, 0x73,
        0x5f, 0x69, 0x73, 0x5f,
        0x61, 0x5f, 0x76, 0x65,
        0x72, 0x79, 0x5f, 0x6c,
        0x6f, 0x6e, 0x67, 0x5f,
        0x6c, 0x6f, 0x6e, 0x67,
        0x5f, 0x73, 0x74, 0x72,
        0x69, 0x6e, 0x67, 0x00,
        0x96, 0xa4, 0xff, 0xff,
        0x01, 0x00, 0x00, 0x00,
        0x58, 0x00, 0x00, 0x00,
        0x17, 0xff, 0xff, 0xff
    };

    ParamDictTest pdt;
    pdt.load_param_bin(mem);

    // string
    int types = pdt.type(0);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    std::string s = pdt.get(0, "");
    if (s != "bij,bjk->bik")
    {
        fprintf(stderr, "test_paramdict string text failed %s != bij,bjk->bik\n", s.c_str());
        return -1;
    }

    // string
    types = pdt.type(1);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(1, "");
    if (s != "This_is_a_very_long_long_string")
    {
        fprintf(stderr, "test_paramdict string text failed %s != This_is_a_very_long_long_string\n", s.c_str());
        return -1;
    }

    // string
    types = pdt.type(2);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(2, "");
    if (s != "X")
    {
        fprintf(stderr, "test_paramdict string text failed %s != X\n", s.c_str());
        return -1;
    }

    return 0;
}

static int compare_paramdict(const ncnn::ParamDict& pd, const ncnn::ParamDict& pd0)
{
    for (int id = 0;; id++)
    {
        const int type0 = pd0.type(id);
        if (type0 == 0)
        {
            break;
        }
        else if (type0 == 2)
        {
            const int i0 = pd0.get(id, 0);
            int i = pd.get(id, 0);
            if (i != i0)
            {
                fprintf(stderr, "compare_paramdict int failed %d != %d\n", i, i0);
                return -1;
            }
        }
        else if (type0 == 3)
        {
            const float f0 = pd0.get(id, 0.f);
            int f = pd.get(id, 0.f);
            if (f != f0)
            {
                fprintf(stderr, "compare_paramdict float failed %f != %f\n", f, f0);
                return -1;
            }
        }
        else if (type0 == 5)
        {
            const ncnn::Mat ai0 = pd0.get(id, ncnn::Mat());
            ncnn::Mat ai = pd.get(id, ncnn::Mat());
            if (ai.w != ai0.w)
            {
                fprintf(stderr, "compare_paramdict int array size failed %d != %d\n", ai.w, ai0.w);
                return -1;
            }
            for (int q = 0; q < ai0.w; q++)
            {
                int i0 = ((const int*)ai0)[q];
                int i = ((const int*)ai)[q];
                if (i != i0)
                {
                    fprintf(stderr, "compare_paramdict int array element %d failed %d != %d\n", q, i, i0);
                    return -1;
                }
            }
        }
        else if (type0 == 6)
        {
            const ncnn::Mat af0 = pd0.get(id, ncnn::Mat());
            ncnn::Mat af = pd.get(id, ncnn::Mat());
            if (af.w != af0.w)
            {
                fprintf(stderr, "compare_paramdict float array size failed %d != %d\n", af.w, af0.w);
                return -1;
            }
            for (int q = 0; q < af0.w; q++)
            {
                float f0 = af0[q];
                float f = af[q];
                if (f != f0)
                {
                    fprintf(stderr, "compare_paramdict float array element %d failed %f != %f\n", q, f, f0);
                    return -1;
                }
            }
        }
        else if (type0 == 7)
        {
            const std::string s0 = pd0.get(id, "");
            std::string s = pd.get(id, "");
            if (s != s0)
            {
                fprintf(stderr, "compare_paramdict string failed %s != %s\n", s.c_str(), s0.c_str());
                return -1;
            }
        }
        else
        {
            fprintf(stderr, "unexpected paramdict type %d\n", type0);
            return -1;
        }
    }

    return 0;
}

static int test_paramdict_6()
{
    const int i0 = 11;
    const float f0 = -2.2f;
    const std::string s0 = "qwqwqwq";
    ncnn::Mat ai0(1);
    {
        int* p = ai0;
        p[0] = 233;
    }

    ncnn::Mat af0(4);
    {
        float* p = af0;
        p[0] = 2.33f;
        p[1] = -0.2f;
        p[2] = 0.f;
        p[3] = 9494.f;
    }

    ncnn::ParamDict pd0;
    pd0.set(1, i0);
    pd0.set(2, ai0);
    pd0.set(3, f0);
    pd0.set(4, af0);
    pd0.set(5, s0);

    // copy
    {
        ncnn::ParamDict pd(pd0);

        int ret = compare_paramdict(pd, pd0);
        if (ret != 0)
        {
            fprintf(stderr, "paramdict copy failed\n");
            return -1;
        }
    }

    // assign
    {
        ncnn::ParamDict pd;
        pd = pd0;

        int ret = compare_paramdict(pd, pd0);
        if (ret != 0)
        {
            fprintf(stderr, "paramdict assign failed\n");
            return -1;
        }
    }

    return 0;
}

int main()
{
    return 0
           || test_paramdict_0()
           || test_paramdict_1()
           || test_paramdict_2()
           || test_paramdict_3()
           || test_paramdict_4()
           || test_paramdict_5()
           || test_paramdict_6();
}
