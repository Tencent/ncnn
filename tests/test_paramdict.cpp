// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <stdio.h>
#include <limits.h>
#include <string.h>

#include "datareader.h"
#include "paramdict.h"

class ParamDictTest : public ncnn::ParamDict
{
public:
    using ncnn::ParamDict::load_param;
    using ncnn::ParamDict::load_param_bin;
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
    pdt.load_param("0=bij,bjk->bik 1=This_is_a_very_long_long_string 3=\"1,2,3 and 6.667          zzz\" 2=\"X\" 6=\"qwqwqwq\"");

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

    // string
    types = pdt.type(6);
    if (types != 7)
    {
        fprintf(stderr, "test_paramdict string type failed %d != 7\n", types);
        return -1;
    }
    s = pdt.get(6, "");
    if (s != "qwqwqwq")
    {
        fprintf(stderr, "test_paramdict string text failed %s != \"qwqwqwq\"\n", s.c_str());
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
    for (int id = 0; id < NCNN_MAX_PARAM_COUNT; id++)
    {
        const int type0 = pd0.type(id);
        if (type0 == 0)
        {
            if (pd.type(id) != 0)
                return -1;
            continue;
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
            float f = pd.get(id, 0.f);
            if (f != f0)
            {
                fprintf(stderr, "compare_paramdict float failed %f != %f\n", f, f0);
                return -1;
            }
        }
        else if (type0 == 4 || type0 == 5)
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

static int test_paramdict_access()
{
    ncnn::ParamDict pd;
    ncnn::Mat array(1);
    ((int*)array)[0] = 17;
    const std::string text = "text";
    pd.set(0, 10);
    pd.set(1, 2.5f);
    pd.set(2, array);
    pd.set(3, text);
    pd.set(31, 31);

    const int invalid_ids[] = {INT_MIN, -1, NCNN_MAX_PARAM_COUNT, INT_MAX};
    for (size_t j = 0; j < sizeof(invalid_ids) / sizeof(invalid_ids[0]); j++)
    {
        const int id = invalid_ids[j];
        pd.set(id, 1);
        pd.set(id, 1.f);
        pd.set(id, array);
        pd.set(id, text);
        if (pd.type(id) != 0 || pd.get(id, 7) != 7 || pd.get(id, 7.f) != 7.f
            || pd.get(id, array).data != array.data || pd.get(id, text) != text)
        {
            fprintf(stderr, "ParamDict invalid id access failed %d\n", id);
            return -1;
        }
    }
    if (pd.get(0, 0) != 10 || pd.get(31, 0) != 31)
        return -1;

    if (pd.get(0, 7.f) != 7.f || pd.get(1, 7) != 7
        || pd.get(2, 7) != 7 || pd.get(2, 7.f) != 7.f
        || pd.get(3, 7) != 7 || pd.get(3, 7.f) != 7.f
        || pd.get(0, array).data != array.data || pd.get(0, text) != text
        || pd.get(3, array).data != array.data || pd.get(2, text) != text)
    {
        fprintf(stderr, "ParamDict mismatched type access failed\n");
        return -1;
    }

    // retagging and copying must not expose an earlier array or string value
    pd.set(2, 22);
    pd.set(3, 33);
    ncnn::ParamDict copied(pd);
    ncnn::ParamDict assigned;
    assigned.set(2, array);
    assigned.set(3, text);
    assigned = pd;
    assigned = assigned;
    if (!copied.get(2, ncnn::Mat()).empty() || copied.get(3, std::string()) != ""
        || !assigned.get(2, ncnn::Mat()).empty() || assigned.get(3, std::string()) != ""
        || copied.get(2, 0) != 22 || assigned.get(3, 0) != 33)
    {
        fprintf(stderr, "ParamDict retagged copy access failed\n");
        return -1;
    }
    return 0;
}

static int check_text_result(const char* text, bool valid)
{
    ParamDictTest memory;
    if ((memory.load_param(text) == 0) != valid)
    {
        fprintf(stderr, "ParamDict memory parse result failed: %s\n", text);
        return -1;
    }
#if NCNN_STDIO
    // fscanf and sscanf have different consumption behavior on failed matches
    FILE* fp = tmpfile();
    if (!fp)
        return -1;
    fwrite(text, 1, strlen(text), fp);
    rewind(fp);
    ncnn::DataReaderFromStdio dr(fp);
    ParamDictTest stdio;
    const int ret = stdio.load_param(dr);
    fclose(fp);
    if ((ret == 0) != valid)
    {
        fprintf(stderr, "ParamDict stdio parse result failed: %s\n", text);
        return -1;
    }
#endif
    return 0;
}

static std::string make_param_string(size_t len, char c)
{
    std::string s;
    s.resize(len);
    for (size_t i = 0; i < len; i++)
        s[i] = c;
    return s;
}

static int test_paramdict_invalid_text()
{
    const char* malformed[] = {
        "-23304=-4", "-23304=-1", "-23304=-2147483648", "-23304=2147483648",
        "-2147483648=0", "2147483648=0", "999999999999999999999=0", "32=0", "-1=0", "-23332=0",
        "0", "0 1", "0=", "+=1", "0=2147483648", "0=-2147483649", "0=1junk", "0=1.0junk",
        "0=.", "0=-", "0=--1", "0=1e", "0=1e+", "0=1.2.3", "0=1e4294967295", "0=1e39",
        "0=\"", "0=\"unterminated", "0=\"abc\n", "0=\"abc\"suffix", "0=\"abc\"1=2",
        "-23300=1", "-23300=2,1", "-23300=1,1,2", "-23300=0,1", "-23300=1x,2",
        "0=1,,2", "0=1,2x", "0=1,2.5", "-23300=2,1,2.5", "0=1.0,2e", "-23300=1,1e+"
    };
    for (size_t i = 0; i < sizeof(malformed) / sizeof(malformed[0]); i++)
        if (check_text_result(malformed[i], false))
            return -1;

    const std::string long_number = "0=0." + make_param_string(128, '0') + "1";
    if (check_text_result(long_number.c_str(), false))
        return -1;
    const std::string long_string = "0=" + make_param_string(256, 'a');
    const std::string long_quoted = "0=\"" + make_param_string(256, 'a') + "\"";
    if (check_text_result(long_string.c_str(), false) || check_text_result(long_quoted.c_str(), false))
        return -1;
    return 0;
}

static int test_paramdict_text_boundaries()
{
    ParamDictTest pd;
    const char* text = "0=-2147483648\t1=2147483647\r\n2=\"\" 3=\" \" 4=1.0,2,-3 5=7, -23306=2,1.0,2 -23307=0 8=0e9999999999 9=1e-9999999999 10=.5 11=4294967296.0";
    if (check_text_result(text, true) || pd.load_param(text))
        return -1;
    if (pd.get(0, 0) != INT_MIN || pd.get(1, 0) != INT_MAX
        || pd.type(2) != 7 || pd.get(2, std::string("default")) != "" || pd.get(3, std::string()) != " "
        || pd.type(7) != 4 || !pd.get(7, ncnn::Mat()).empty()
        || pd.get(8, 1.f) != 0.f || pd.get(9, 1.f) != 0.f || pd.get(10, 0.f) != 0.5f
        || pd.get(11, 0.f) != 4294967296.f)
        return -1;
    ncnn::Mat a = pd.get(4, ncnn::Mat());
    ncnn::Mat b = pd.get(5, ncnn::Mat());
    ncnn::Mat c = pd.get(6, ncnn::Mat());
    if (a.w != 3 || a[0] != 1.f || a[1] != 2.f || a[2] != -3.f
        || b.w != 1 || ((const int*)b)[0] != 7 || c.w != 2 || c[0] != 1.f || c[1] != 2.f)
        return -1;

    const int lengths[] = {1, 14, 15, 16, 240, 241, 254, 255};
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++)
    {
        const std::string value = make_param_string(lengths[i], 'x');
        for (int quoted = 0; quoted < 2; quoted++)
        {
            const std::string input = quoted ? "0=\"" + value + "\" 1=19" : "0=" + value + " 1=19";
            if (check_text_result(input.c_str(), true) || pd.load_param(input.c_str())
                || pd.get(0, std::string()) != value || pd.get(1, 0) != 19)
            {
                fprintf(stderr, "ParamDict string boundary failed len=%d quoted=%d\n", lengths[i], quoted);
                return -1;
            }
        }
    }

    // leave the next layer header untouched for the caller
    const unsigned char* ptr = (const unsigned char*)"0=42\nReLU next 1 1 in out\n";
    ncnn::DataReaderFromMemory reader(ptr);
    char layer_type[5];
    if (pd.load_param(reader) || pd.get(0, 0) != 42 || reader.scan("%4s", layer_type) != 1 || strcmp(layer_type, "ReLU"))
        return -1;

    // exercise array growth and the final copy for both element types
    const int array_lengths[] = {16, 17, 32, 33, 257};
    for (size_t j = 0; j < sizeof(array_lengths) / sizeof(array_lengths[0]); j++)
        for (int floating = 0; floating < 2; floating++)
        {
            std::string input = "0=";
            for (int i = 0; i < array_lengths[j]; i++)
            {
                char value[32];
                snprintf(value, sizeof(value), "%s%d%s", i ? "," : "", i, floating ? ".0" : "");
                input += value;
            }
            input += " 1=42";
            if (check_text_result(input.c_str(), true) || pd.load_param(input.c_str()) || pd.get(1, 0) != 42)
                return -1;
            const ncnn::Mat values = pd.get(0, ncnn::Mat());
            if (values.w != array_lengths[j])
                return -1;
            for (int i = 0; i < values.w; i++)
                if (floating ? values[i] != (float)i : ((const int*)values)[i] != i)
                    return -1;
        }
    return 0;
}

class BoundedParamReader : public ncnn::DataReader
{
public:
    BoundedParamReader(const unsigned char* data, size_t size)
        : ptr(data), remaining(size)
    {
    }
    virtual size_t read(void* buf, size_t size) const
    {
        const size_t n = std::min(size, remaining);
        if (n != 0)
        {
            memcpy(buf, ptr, n);
            ptr += n;
            remaining -= n;
        }
        return n;
    }
private:
    mutable const unsigned char* ptr;
    mutable size_t remaining;
};

static void append_param_word(std::vector<unsigned char>& bytes, int value)
{
    const unsigned int v = (unsigned int)value;
    for (int i = 0; i < 4; i++)
        bytes.push_back((v >> (i * 8)) & 255);
}

static int test_paramdict_binary_bounds()
{
    const int ids[] = {-23304, -23404};
    const int lengths[] = {-1, -4, -8, INT_MIN};
    for (size_t i = 0; i < sizeof(ids) / sizeof(ids[0]); i++)
        for (size_t j = 0; j < sizeof(lengths) / sizeof(lengths[0]); j++)
        {
            std::vector<unsigned char> data;
            append_param_word(data, ids[i]);
            append_param_word(data, lengths[j]);
            append_param_word(data, -233);
            BoundedParamReader reader(data.data(), data.size());
            ParamDictTest pd;
            if (pd.load_param_bin(reader) == 0)
            {
                fprintf(stderr, "ParamDict negative binary length accepted id=%d len=%d\n", ids[i], lengths[j]);
                return -1;
            }
        }

    const int invalid_ids[] = {INT_MIN, -1, 32, INT_MAX, -23332, -23432};
    for (size_t i = 0; i < sizeof(invalid_ids) / sizeof(invalid_ids[0]); i++)
    {
        std::vector<unsigned char> data;
        append_param_word(data, invalid_ids[i]);
        append_param_word(data, -233);
        BoundedParamReader reader(data.data(), data.size());
        ParamDictTest pd;
        if (pd.load_param_bin(reader) == 0)
            return -1;
    }

    const int invalid_string_lengths[] = {256, INT_MAX};
    for (size_t i = 0; i < sizeof(invalid_string_lengths) / sizeof(invalid_string_lengths[0]); i++)
    {
        std::vector<unsigned char> data;
        append_param_word(data, -23400);
        append_param_word(data, invalid_string_lengths[i]);
        BoundedParamReader reader(data.data(), data.size());
        ParamDictTest pd;
        if (pd.load_param_bin(reader) == 0)
            return -1;
    }

    if (sizeof(size_t) == 4)
    {
        std::vector<unsigned char> data;
        append_param_word(data, -23300);
        append_param_word(data, 0x40000000); // byte count wraps on 32-bit targets
        BoundedParamReader reader(data.data(), data.size());
        ParamDictTest pd;
        if (pd.load_param_bin(reader) == 0 || check_text_result("-23300=1073741824", false))
            return -1;
    }

    // untyped binary scalars/arrays remain readable as both int and float
    std::vector<unsigned char> data;
    append_param_word(data, 0);
    append_param_word(data, 0x3f800000);
    append_param_word(data, -23301);
    append_param_word(data, 1);
    append_param_word(data, 0x3f800000);
    append_param_word(data, -23302);
    append_param_word(data, 0);
    append_param_word(data, -23403);
    append_param_word(data, 0);
    append_param_word(data, -23404);
    append_param_word(data, 1);
    append_param_word(data, 0x64636261); // only 'a' belongs to the string; padding is nonzero
    append_param_word(data, -23405);
    append_param_word(data, 3);
    append_param_word(data, 0x7f620061); // embedded NUL is part of the declared length
    append_param_word(data, -23406);
    append_param_word(data, 255);
    for (int i = 0; i < 256; i++)
        data.push_back('q');
    append_param_word(data, -233);
    ParamDictTest pd;
    BoundedParamReader reader(data.data(), data.size());
    if (pd.load_param_bin(reader) || pd.get(0, 0) != 0x3f800000 || pd.get(0, 0.f) != 1.f
        || pd.get(1, ncnn::Mat()).w != 1 || pd.get(1, ncnn::Mat())[0] != 1.f
        || pd.type(2) != 4 || !pd.get(2, ncnn::Mat()).empty()
        || pd.type(3) != 7 || pd.get(3, std::string("default")) != ""
        || pd.get(4, std::string()) != "a" || pd.get(5, std::string()).size() != 3
        || pd.get(5, std::string())[0] != 'a' || pd.get(5, std::string())[1] != '\0' || pd.get(5, std::string())[2] != 'b'
        || pd.get(6, std::string()) != make_param_string(255, 'q'))
        return -1;

    // every truncation, including missing EOP and short scalar/array/string data, fails
    for (size_t size = 0; size < data.size(); size++)
    {
        BoundedParamReader truncated(data.data(), size);
        ParamDictTest partial;
        if (partial.load_param_bin(truncated) == 0)
        {
            fprintf(stderr, "ParamDict truncated binary accepted at %zu\n", size);
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
           || test_paramdict_6()
           || test_paramdict_access()
           || test_paramdict_invalid_text()
           || test_paramdict_text_boundaries()
           || test_paramdict_binary_bounds();
}
