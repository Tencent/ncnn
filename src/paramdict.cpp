// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "paramdict.h"

#include "allocator.h"
#include "datareader.h"
#include "mat.h"
#include "platform.h"

#include <float.h>
#include <limits.h>

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
    if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        return 0;

    return d->params[id].type;
}

int ParamDict::get(int id, int def) const
{
    const int t = type(id);
    return t == 1 || t == 2 ? d->params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    const int t = type(id);
    return t == 1 || t == 3 ? d->params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    const int t = type(id);
    return t == 4 || t == 5 || t == 6 ? d->params[id].v : def;
}

std::string ParamDict::get(int id, const std::string& def) const
{
    return type(id) == 7 ? d->params[id].s : def;
}

void ParamDict::set(int id, int i)
{
    if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        return;

    d->params[id].type = 2;
    d->params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        return;

    d->params[id].type = 3;
    d->params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        return;

    d->params[id].type = 4;
    d->params[id].v = v;
}

void ParamDict::set(int id, const std::string& s)
{
    if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        return;

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

static bool valid_array_length(size_t len)
{
    // leave room for Mat alignment, the reference count and fastMalloc overhead
    const size_t max_len = ((size_t)-1 - 15 - sizeof(int) - sizeof(void*) - NCNN_MALLOC_ALIGN - NCNN_MALLOC_OVERREAD) / sizeof(float);
    return len <= INT_MAX && len <= max_len;
}

#if NCNN_STRING
static bool vstr_is_float(const char* vstr)
{
    return strchr(vstr, '.') || strchr(vstr, 'e') || strchr(vstr, 'E');
}

static bool vstr_to_int(const char* p, int& v)
{
    const bool negative = *p == '-';
    if (*p == '+' || *p == '-')
        p++;

    if (*p < '0' || *p > '9')
        return false;

    const unsigned int limit = negative ? (unsigned int)INT_MAX + 1u : (unsigned int)INT_MAX;
    unsigned int magnitude = 0;
    while (*p >= '0' && *p <= '9')
    {
        const unsigned int digit = *p++ - '0';
        if (magnitude > (limit - digit) / 10)
            return false;
        magnitude = magnitude * 10 + digit;
    }
    if (*p != '\0')
        return false;

    v = negative ? (magnitude == (unsigned int)INT_MAX + 1u ? INT_MIN : -(int)magnitude) : (int)magnitude;
    return true;
}

static bool vstr_to_float(const char* p, float& value)
{
    const bool negative = *p == '-';
    if (*p == '+' || *p == '-')
        p++;

    double v = 0.0;
    bool has_digit = false;
    while (*p >= '0' && *p <= '9')
    {
        has_digit = true;
        v = v * 10.0 + (*p++ - '0');
    }
    if (*p == '.')
    {
        p++;
        double scale = 0.1;
        while (*p >= '0' && *p <= '9')
        {
            has_digit = true;
            v += (*p++ - '0') * scale;
            scale *= 0.1;
        }
    }
    if (!has_digit)
        return false;

    if (*p == 'e' || *p == 'E')
    {
        p++;
        const bool negative_exponent = *p == '-';
        if (*p == '+' || *p == '-')
            p++;
        if (*p < '0' || *p > '9')
            return false;

        // saturate the exponent instead of overflowing or looping over its value
        unsigned int exponent = 0;
        while (*p >= '0' && *p <= '9')
        {
            if (exponent < 1024)
                exponent = std::min(1024u, exponent * 10 + (*p - '0'));
            p++;
        }
        if (v != 0.0)
        {
            double scale = 1.0;
            double base = 10.0;
            while (exponent)
            {
                if (exponent & 1)
                    scale *= base;
                exponent >>= 1;
                if (exponent)
                    base *= base;
            }
            v = negative_exponent ? v / scale : v * scale;
        }
    }
    // allow rounding to FLT_MAX, but reject the midpoint that rounds to infinity
    const double half_ulp = (double)FLT_MAX / ((1u << FLT_MANT_DIG) - 1) * 0.5;
    if (*p != '\0' || v >= (double)FLT_MAX + half_ulp)
        return false;

    // keep the conversion within the finite float range
    v = std::min(v, (double)FLT_MAX);
    value = negative ? (float)-v : (float)v;
    return true;
}

static int scan_numeric_value(const DataReader& dr, char vstr[128])
{
    if (dr.scan("%127[^, \t\r\n\v\f]", vstr) != 1)
        return 0;

    // a field-width limit must not silently split a numeric token
    char extra[2];
    if (strlen(vstr) == 127 && dr.scan("%1[^, \t\r\n\v\f]", extra) == 1)
        return -1;

    return 1;
}

int ParamDict::load_param(const DataReader& dr)
{
    clear();

    // stop before the next layer name, leaving it for Net to parse
    char idstr[16];
    while (dr.scan(" %15[+-0123456789]", idstr) == 1)
    {
        int id;
        char delimiter[2];
        if (!vstr_to_int(idstr, id) || dr.scan("%1[=]", delimiter) != 1)
        {
            NCNN_LOGE("ParamDict invalid parameter id or missing equals sign");
            return -1;
        }

        const bool old_array = id <= -23300;
        if (old_array)
            id = -(id + 23300);

        if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
        {
            NCNN_LOGE("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)", id, NCNN_MAX_PARAM_COUNT);
            return -1;
        }

        if (old_array)
        {
            char vstr[128];
            int len;
            if (scan_numeric_value(dr, vstr) != 1 || !vstr_to_int(vstr, len) || !valid_array_length((size_t)len))
            {
                NCNN_LOGE("ParamDict invalid array length (id=%d)", id);
                return -1;
            }

            Mat v(len);
            if (len > 0 && v.empty())
            {
                NCNN_LOGE("ParamDict array allocation failed (id=%d, len=%d)", id, len);
                return -1;
            }

            bool is_float = false;
            for (int j = 0; j < len; j++)
            {
                if (dr.scan("%1[,]", delimiter) != 1 || scan_numeric_value(dr, vstr) != 1)
                {
                    NCNN_LOGE("ParamDict read array element failed");
                    return -1;
                }
                if (j == 0)
                    is_float = vstr_is_float(vstr);

                const bool ok = is_float ? vstr_to_float(vstr, ((float*)v)[j]) : vstr_to_int(vstr, ((int*)v)[j]);
                if (!ok)
                {
                    NCNN_LOGE("ParamDict invalid array element (id=%d, index=%d)", id, j);
                    return -1;
                }
            }
            // extra elements are not a new parameter or the next layer
            if (dr.scan("%1[,]", delimiter) == 1)
            {
                NCNN_LOGE("ParamDict array length mismatch (id=%d)", id);
                return -1;
            }

            d->params[id].v = v;
            d->params[id].type = len == 0 ? 4 : is_float ? 6 : 5;
            continue;
        }

        char first[2];
        if (dr.scan("%1[\"a-zA-Z]", first) == 1)
        {
            char text[256] = {0};
            const bool quoted = first[0] == '\"';
            if (quoted)
            {
                dr.scan("%255[^\"\r\n]", text);
                if (dr.scan("%1[\"]", delimiter) != 1)
                {
                    NCNN_LOGE("ParamDict unterminated or too long string (id=%d)", id);
                    return -1;
                }
            }
            else
            {
                text[0] = first[0];
                dr.scan("%254[^ \t\r\n\v\f]", text + 1);
            }
            if (dr.scan("%1[^ \t\r\n\v\f]", delimiter) == 1)
            {
                NCNN_LOGE("ParamDict invalid string suffix or string too long (id=%d)", id);
                return -1;
            }

            d->params[id].s = text;
            d->params[id].type = 7;
            continue;
        }

        char vstr[128];
        if (scan_numeric_value(dr, vstr) != 1)
        {
            NCNN_LOGE("ParamDict read value failed");
            return -1;
        }

        const bool is_float = vstr_is_float(vstr);
        float f = 0.f;
        int i = 0;
        if (!(is_float ? vstr_to_float(vstr, f) : vstr_to_int(vstr, i)))
        {
            NCNN_LOGE("ParamDict invalid numeric value (id=%d)", id);
            return -1;
        }

        if (dr.scan("%1[,]", delimiter) == 1)
        {
            Mat values(16);
            if (values.empty())
            {
                NCNN_LOGE("ParamDict array allocation failed (id=%d)", id);
                return -1;
            }
            int len = 1;
            if (is_float)
                ((float*)values)[0] = f;
            else
                ((int*)values)[0] = i;

            while (1)
            {
                const int nscan = scan_numeric_value(dr, vstr);
                // a trailing comma is the established syntax for a one-element array
                if (nscan == 0)
                {
                    if (dr.scan("%1[,]", delimiter) == 1)
                        return -1;
                    break;
                }
                if (nscan < 0 || !valid_array_length((size_t)len + 1) || !(is_float ? vstr_to_float(vstr, f) : vstr_to_int(vstr, i)))
                {
                    NCNN_LOGE("ParamDict invalid array element (id=%d)", id);
                    return -1;
                }
                if (len == values.w)
                {
                    size_t capacity = (size_t)values.w * 2;
                    if (!valid_array_length(capacity))
                        capacity = (size_t)len + 1;
                    Mat grown((int)capacity);
                    if (grown.empty())
                    {
                        NCNN_LOGE("ParamDict array allocation failed (id=%d)", id);
                        return -1;
                    }
                    memcpy(grown.data, values.data, (size_t)len * sizeof(float));
                    values = grown;
                }
                if (is_float)
                    ((float*)values)[len] = f;
                else
                    ((int*)values)[len] = i;
                len++;

                if (dr.scan("%1[,]", delimiter) != 1)
                    break;
            }

            Mat v(len);
            if (v.empty())
            {
                NCNN_LOGE("ParamDict array allocation failed (id=%d)", id);
                return -1;
            }
            memcpy(v.data, values.data, (size_t)len * sizeof(float));
            d->params[id].v = v;
            d->params[id].type = is_float ? 6 : 5;
        }
        else
        {
            if (is_float)
                d->params[id].f = f;
            else
                d->params[id].i = i;
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
            id = -(id + 23400);
        }
        else if (is_array)
        {
            id = -(id + 23300);
        }

        if (id < 0 || id >= NCNN_MAX_PARAM_COUNT)
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

            if (len < 0 || len > 255)
            {
                NCNN_LOGE("invalid string length %d (id=%d)", len, id);
                return -1;
            }

            size_t len_padded = (len + 3) / 4 * 4;
            char tmpstr[256];
            char* ptr = tmpstr;
            nread = dr.read(ptr, len_padded);
            if (nread != len_padded)
            {
                NCNN_LOGE("ParamDict read string failed %zd", nread);
                return -1;
            }

            d->params[id].s.resize(len);
            if (len > 0)
                memcpy(&d->params[id].s[0], tmpstr, len);

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

            if (!valid_array_length((size_t)len))
            {
                NCNN_LOGE("ParamDict invalid array length %d (id=%d)", len, id);
                return -1;
            }

            Mat v(len);
            if (len > 0 && v.empty())
            {
                NCNN_LOGE("ParamDict array allocation failed (id=%d, len=%d)", id, len);
                return -1;
            }

            float* ptr = v;
            nread = len == 0 ? 0 : dr.read(ptr, sizeof(float) * len);
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

            d->params[id].v = v;
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
