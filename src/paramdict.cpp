// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "paramdict.h"

#include "datareader.h"
#include "mat.h"
#include "platform.h"

#include <ctype.h>

#if NCNN_STDIO
#include <stdio.h>
#endif

namespace ncnn {

class ParamDictPrivate
{
public:
    struct
    {
        // 0 = null
        // 1 = int/float
        // 2 = int
        // 3 = float
        // 4 = array of int/float
        // 5 = array of int
        // 6 = array of float
        // 7 = string
        int type;
        union
        {
            int i;
            float f;
        };
        Mat v;
        std::string s;
    } params[NCNN_MAX_PARAM_COUNT];
};

ParamDict::ParamDict()
    : d(new ParamDictPrivate)
{
    clear();
}

ParamDict::~ParamDict()
{
    delete d;
}

ParamDict::ParamDict(const ParamDict& rhs)
    : d(new ParamDictPrivate)
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        int type = rhs.d->params[i].type;
        d->params[i].type = type;
        if (type == 1 || type == 2 || type == 3)
        {
            d->params[i].i = rhs.d->params[i].i;
        }
        else if (type == 7)
        {
            d->params[i].s = rhs.d->params[i].s;
        }
        else // if (type == 4 || type == 5 || type == 6)
        {
            d->params[i].v = rhs.d->params[i].v;
        }
    }
}

ParamDict& ParamDict::operator=(const ParamDict& rhs)
{
    if (this == &rhs)
        return *this;

    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        int type = rhs.d->params[i].type;
        d->params[i].type = type;
        if (type == 1 || type == 2 || type == 3)
        {
            d->params[i].i = rhs.d->params[i].i;
        }
        else if (type == 7)
        {
            d->params[i].s = rhs.d->params[i].s;
        }
        else // if (type == 4 || type == 5 || type == 6)
        {
            d->params[i].v = rhs.d->params[i].v;
        }
    }

    return *this;
}

int ParamDict::type(int id) const
{
    return d->params[id].type;
}

// TODO strict type check
int ParamDict::get(int id, int def) const
{
    return d->params[id].type ? d->params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return d->params[id].type ? d->params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return d->params[id].type ? d->params[id].v : def;
}

std::string ParamDict::get(int id, const std::string& def) const
{
    return d->params[id].type ? d->params[id].s : def;
}

void ParamDict::set(int id, int i)
{
    d->params[id].type = 2;
    d->params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    d->params[id].type = 3;
    d->params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    d->params[id].type = 4;
    d->params[id].v = v;
}

void ParamDict::set(int id, const std::string& s)
{
    d->params[id].type = 7;
    d->params[id].s = s;
}

void ParamDict::clear()
{
    for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
    {
        d->params[i].type = 0;
        d->params[i].i = 0;
        d->params[i].v = Mat();
        d->params[i].s.clear();
    }
}

#if NCNN_STRING
static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j = 0; j < 16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

static bool vstr_is_string(const char vstr[16])
{
    return isalpha(vstr[0]) || vstr[0] == '\"';
}

static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';
    if (*p == '+' || *p == '-')
    {
        p++;
    }

    // digits before decimal point or exponent
    unsigned int v1 = 0;
    while (isdigit(*p))
    {
        v1 = v1 * 10 + (*p - '0');
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if (*p == '.')
    {
        p++;

        unsigned int pow10 = 1;
        unsigned int v2 = 0;

        while (isdigit(*p))
        {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if (*p == 'e' || *p == 'E')
    {
        p++;

        // sign of exponent
        bool fact = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits of exponent
        unsigned int expon = 0;
        while (isdigit(*p))
        {
            expon = expon * 10 + (*p - '0');
            p++;
        }

        double scale = 1.0;
        while (expon >= 8)
        {
            scale *= 1e8;
            expon -= 8;
        }
        while (expon > 0)
        {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    //     fprintf(stderr, "v = %f\n", v);
    return sign ? (float)v : (float)-v;
}

int ParamDict::load_param(const DataReader& dr)
{
    clear();

    // 0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0
    // 3=0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (dr.scan("%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (id >= NCNN_MAX_PARAM_COUNT)
        {
            NCNN_LOGE("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)", id, NCNN_MAX_PARAM_COUNT);
            return -1;
        }

        if (is_array)
        {
            // old style array
            int len = 0;
            int nscan = dr.scan("%d", &len);
            if (nscan != 1)
            {
                NCNN_LOGE("ParamDict read array length failed");
                return -1;
            }

            d->params[id].v.create(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = dr.scan(",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict read array element failed");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float* ptr = d->params[id].v;
                    ptr[j] = vstr_to_float(vstr);
                }
                else
                {
                    int* ptr = d->params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                    if (nscan != 1)
                    {
                        NCNN_LOGE("ParamDict parse array element failed");
                        return -1;
                    }
                }

                d->params[id].type = is_float ? 6 : 5;
            }

            continue;
        }

        char vstr[16];
        char comma[4];
        int nscan = dr.scan("%15[^,\n ]", vstr);
        if (nscan != 1)
        {
            NCNN_LOGE("ParamDict read value failed");
            return -1;
        }

        bool is_string = vstr_is_string(vstr);
        if (is_string)
        {
            // scan the remaining string
            char vstr2[256];
            vstr2[241] = '\0'; // max 255 = 15 + 240

            if (vstr[0] == '\"')
            {
                int len = 0;
                while (vstr[len] != '\0')
                    len++;
                char end = vstr[len - 1];
                if (end != '\"')
                {
                    nscan = dr.scan("%255[^\"\n]\"", vstr2);
                }
                else
                    nscan = 0; // already ended with a quote, no need to scan more
            }
            else
            {
                nscan = dr.scan("%255[^\n ]", vstr2);
            }
            if (nscan == 1)
            {
                if (vstr2[241] != '\0')
                {
                    NCNN_LOGE("string too long (id=%d)", id);
                    return -1;
                }

                if (vstr[0] == '\"')
                    d->params[id].s = std::string(&vstr[1]) + vstr2;
                else
                    d->params[id].s = std::string(vstr) + vstr2;
            }
            else
            {
                if (vstr[0] == '\"')
                    d->params[id].s = std::string(&vstr[1]);
                else
                    d->params[id].s = std::string(vstr);
            }

            if (d->params[id].s[d->params[id].s.size() - 1] == '\"')
                d->params[id].s.resize(d->params[id].s.size() - 1);

            d->params[id].type = 7;

            continue;
        }

        bool is_float = vstr_is_float(vstr);

        nscan = dr.scan("%1[,]", comma);
        is_array = nscan == 1;

        if (is_array)
        {
            std::vector<float> af;
            std::vector<int> ai;

            if (is_float)
            {
                af.push_back(vstr_to_float(vstr));
            }
            else
            {
                int v = 0;
                nscan = sscanf(vstr, "%d", &v);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict parse value failed");
                    return -1;
                }

                ai.push_back(v);
            }

            while (1)
            {
                nscan = dr.scan("%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    break;
                }

                if (is_float)
                {
                    af.push_back(vstr_to_float(vstr));
                }
                else
                {
                    int v = 0;
                    nscan = sscanf(vstr, "%d", &v);
                    if (nscan != 1)
                    {
                        NCNN_LOGE("ParamDict parse value failed");
                        return -1;
                    }

                    ai.push_back(v);
                }

                nscan = dr.scan("%1[,]", comma);
                if (nscan != 1)
                {
                    break;
                }
            }

            if (is_float)
            {
                d->params[id].v.create((int)af.size());
                memcpy(d->params[id].v.data, af.data(), af.size() * 4);
            }
            else
            {
                d->params[id].v.create((int)ai.size());
                memcpy(d->params[id].v.data, ai.data(), ai.size() * 4);
            }

            d->params[id].type = is_float ? 6 : 5;
        }
        else
        {
            if (is_float)
            {
                d->params[id].f = vstr_to_float(vstr);
            }
            else
            {
                nscan = sscanf(vstr, "%d", &d->params[id].i);
                if (nscan != 1)
                {
                    NCNN_LOGE("ParamDict parse value failed");
                    return -1;
                }
            }

            d->params[id].type = is_float ? 3 : 2;
        }
    }

    return 0;
}
#endif // NCNN_STRING

int ParamDict::load_param_bin(const DataReader& dr)
{
    clear();

    //     binary 0
    //     binary 100
    //     binary 1
    //     binary 1.250000
    //     binary 3 | array_bit
    //     binary 5
    //     binary 0.1
    //     binary 0.2
    //     binary 0.4
    //     binary 0.8
    //     binary 1.0
    //     binary -233(EOP)

    int id = 0;
    size_t nread;
    nread = dr.read(&id, sizeof(int));
    if (nread != sizeof(int))
    {
        NCNN_LOGE("ParamDict read id failed %zd", nread);
        return -1;
    }

#if __BIG_ENDIAN__
    swap_endianness_32(&id);
#endif

    while (id != -233)
    {
        bool is_array = id <= -23300;
        bool is_string = id <= -23400;
        if (is_string)
        {
            id = -id - 23400;
        }
        else if (is_array)
        {
            id = -id - 23300;
        }

        if (id >= NCNN_MAX_PARAM_COUNT)
        {
            NCNN_LOGE("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)", id, NCNN_MAX_PARAM_COUNT);
            return -1;
        }

        if (is_string)
        {
            int len = 0;
            nread = dr.read(&len, sizeof(int));
            if (nread != sizeof(int))
            {
                NCNN_LOGE("ParamDict read array length failed %zd", nread);
                return -1;
            }

#if __BIG_ENDIAN__
            swap_endianness_32(&len);
#endif

            if (len > 255)
            {
                NCNN_LOGE("string too long (id=%d)", id);
                return -1;
            }

            size_t len_padded = (len + 3) / 4 * 4;
            std::vector<char> tmpstr(len_padded + 1);

            char* ptr = (char*)tmpstr.data();
            nread = dr.read(ptr, len_padded);
            if (nread != len_padded)
            {
                NCNN_LOGE("ParamDict read string failed %zd", nread);
                return -1;
            }

            tmpstr[len_padded] = '\0';

            d->params[id].s = tmpstr.data();

            d->params[id].type = 7;
        }
        else if (is_array)
        {
            int len = 0;
            nread = dr.read(&len, sizeof(int));
            if (nread != sizeof(int))
            {
                NCNN_LOGE("ParamDict read array length failed %zd", nread);
                return -1;
            }

#if __BIG_ENDIAN__
            swap_endianness_32(&len);
#endif

            d->params[id].v.create(len);

            float* ptr = d->params[id].v;
            nread = dr.read(ptr, sizeof(float) * len);
            if (nread != sizeof(float) * len)
            {
                NCNN_LOGE("ParamDict read array element failed %zd", nread);
                return -1;
            }

#if __BIG_ENDIAN__
            for (int i = 0; i < len; i++)
            {
                swap_endianness_32(ptr + i);
            }
#endif

            d->params[id].type = 4;
        }
        else
        {
            nread = dr.read(&d->params[id].f, sizeof(float));
            if (nread != sizeof(float))
            {
                NCNN_LOGE("ParamDict read value failed %zd", nread);
                return -1;
            }

#if __BIG_ENDIAN__
            swap_endianness_32(&d->params[id].f);
#endif

            d->params[id].type = 1;
        }

        nread = dr.read(&id, sizeof(int));
        if (nread != sizeof(int))
        {
            NCNN_LOGE("ParamDict read EOP failed %zd", nread);
            return -1;
        }

#if __BIG_ENDIAN__
        swap_endianness_32(&id);
#endif
    }

    return 0;
}

} // namespace ncnn
