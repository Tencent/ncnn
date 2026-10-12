// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"

#include <stdio.h>
#include <string.h>

static int test_datareader_memory()
{
    const unsigned char data[] = {1, 2, 3, 4};
    const unsigned char* mem = data;
    ncnn::DataReaderFromMemory dr(mem, sizeof(data));

    const void* ref = 0;
    if (dr.reference((size_t)-1, &ref) != 0 || ref != 0 || mem != data)
    {
        fprintf(stderr, "test_datareader_memory reference bounds failed\n");
        return -1;
    }

    if (dr.reference(2, &ref) != 2 || ref != data || mem != data + 2)
    {
        fprintf(stderr, "test_datareader_memory reference failed\n");
        return -1;
    }

    unsigned char buf[4] = {0};
    if (dr.read(buf, (size_t)-1) != 2 || mem != data + sizeof(data) || memcmp(buf, data + 2, 2) || buf[2] != 0)
    {
        fprintf(stderr, "test_datareader_memory short read failed\n");
        return -1;
    }

    for (int i = 0; i < 2; i++)
    {
        if (dr.read(buf, 1) != 0 || dr.reference(1, &ref) != 0 || mem != data + sizeof(data) || ref != data)
        {
            fprintf(stderr, "test_datareader_memory eof failed\n");
            return -1;
        }
    }

    // the original constructor retains unbounded read and reference behavior
    mem = data;
    ncnn::DataReaderFromMemory dr2(mem);
    if (dr2.read(buf, 2) != 2 || memcmp(buf, data, 2) || dr2.reference(2, &ref) != 2 || ref != data + 2 || mem != data + sizeof(data))
    {
        fprintf(stderr, "test_datareader_memory unbounded reader failed\n");
        return -1;
    }

    return 0;
}

static int test_datareader_empty()
{
    const unsigned char data[] = {1};
    for (int i = 0; i < 2; i++)
    {
        const unsigned char* start = i == 0 ? 0 : data;
        const unsigned char* mem = start;
        ncnn::DataReaderFromMemory dr(mem, 0);
        const void* ref = data;
        unsigned char buf = 0;
        if (dr.read(0, 0) != 0 || dr.reference(0, &ref) != 0 || ref != start)
        {
            fprintf(stderr, "test_datareader_empty zero request failed\n");
            return -1;
        }

        for (int j = 0; j < 2; j++)
        {
            if (dr.read(&buf, 1) != 0 || dr.reference(1, &ref) != 0 || mem != start || ref != start || buf != 0)
            {
                fprintf(stderr, "test_datareader_empty bounds failed\n");
                return -1;
            }
#if NCNN_STRING
            int value = 0;
            if (dr.scan("%d", &value) != 0 || mem != start || value != 0)
            {
                fprintf(stderr, "test_datareader_empty scan failed\n");
                return -1;
            }
#endif
        }
    }

    return 0;
}

#if NCNN_STRING
static int test_datareader_scan_bounds()
{
    const unsigned char data[] = "1234 X";
    const unsigned char* mem = data;
    ncnn::DataReaderFromMemory dr(mem, 2);
    int value = 0;
    if (dr.scan("%d", &value) != 1 || value != 12 || mem != data + 2)
    {
        fprintf(stderr, "test_datareader_scan_bounds scan failed\n");
        return -1;
    }

    unsigned char buf = 0;
    const void* ref = 0;
    if (dr.read(&buf, 1) != 0 || dr.reference(1, &ref) != 0 || dr.scan("%d", &value) != 0 || mem != data + 2 || buf != 0 || ref != 0)
    {
        fprintf(stderr, "test_datareader_scan_bounds eof failed\n");
        return -1;
    }

    // scanning and binary reads share the same cursor and remaining size
    const unsigned char mixed[] = {'7', ' ', 'X', 'Y'};
    mem = mixed;
    ncnn::DataReaderFromMemory dr2(mem, sizeof(mixed));
    if (dr2.scan("%d", &value) != 1 || value != 7 || mem != mixed + 1 || dr2.read(&buf, 1) != 1 || buf != ' '
            || dr2.reference(2, &ref) != 2 || ref != mixed + 2 || mem != mixed + sizeof(mixed) || dr2.scan("%d", &value) != 0)
    {
        fprintf(stderr, "test_datareader_scan_bounds mixed read failed\n");
        return -1;
    }

    return 0;
}

static int test_datareader_scan_mixed()
{
    const unsigned char data[] = {'A', 'B', 'C', '1', '2', '\n', '3', '4', '\n', '5', '6', '!', '\n', 'X', 'Y', 'Z', '\n', '7', ' ', '8'};
    const unsigned char* mem = data;
    ncnn::DataReaderFromMemory dr(mem, sizeof(data));
    unsigned char buf[5] = {0};
    const void* ref = 0;
    if (dr.read(buf, 1) != 1 || buf[0] != 'A' || dr.reference(2, &ref) != 2 || ref != data + 1 || mem != data + 3)
    {
        fprintf(stderr, "test_datareader_scan_mixed initial binary read failed\n");
        return -1;
    }

    int value = 0;
    if (dr.scan("%d", &value) != 1 || value != 12 || mem != data + 5
            || dr.read(buf, 1) != 1 || buf[0] != '\n' || dr.scan("%d", &value) != 1 || value != 34 || mem != data + 8
            || dr.reference(1, &ref) != 1 || ref != data + 8 || dr.scan("%d", &value) != 1 || value != 56 || mem != data + 11)
    {
        fprintf(stderr, "test_datareader_scan_mixed mixed read failed\n");
        return -1;
    }

    if (dr.reference((size_t)-1, &ref) != 0 || ref != data + 8 || dr.scan("%d", &value) != 0 || value != 56 || mem != data + 11)
    {
        fprintf(stderr, "test_datareader_scan_mixed failed scan or reference consumed input\n");
        return -1;
    }

    if (dr.scan("!%5c", buf) != 1 || memcmp(buf, "\nXYZ\n", sizeof(buf)) || mem != data + 17)
    {
        fprintf(stderr, "test_datareader_scan_mixed cross-line scan failed\n");
        return -1;
    }

    if (dr.scan("%d", &value) != 1 || value != 7 || mem != data + 18 || dr.scan("%d", &value) != 1 || value != 8 || mem != data + sizeof(data))
    {
        fprintf(stderr, "test_datareader_scan_mixed consecutive scans failed\n");
        return -1;
    }

    if (dr.scan("%d", &value) != 0 || dr.read(buf, 1) != 0 || dr.reference(1, &ref) != 0 || mem != data + sizeof(data))
    {
        fprintf(stderr, "test_datareader_scan_mixed eof failed\n");
        return -1;
    }

    return 0;
}

static int test_datareader_scan()
{
    const char* formats[] = {"%d", "%i", "%x", "x=%d!", "%d \n", "%d!", "%d"};
    const char* inputs[] = {" \n1234 X", "010 X", "ff X", "x=7!tail", "7 \r\n \tX", "7X", "X"};
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
    {
        const unsigned char* start = (const unsigned char*)inputs[i];
        const unsigned char* mem = start;
        const unsigned char* mem2 = start;
        ncnn::DataReaderFromMemory dr(mem, strlen(inputs[i]));
        ncnn::DataReaderFromMemory dr2(mem2);
        int value = 0;
        int value2 = 0;
        const int ret = dr.scan(formats[i], &value);
        const int ret2 = dr2.scan(formats[i], &value2);
        if (ret != ret2 || value != value2 || mem != mem2)
        {
            fprintf(stderr, "test_datareader_scan integer failed format=%s\n", formats[i]);
            return -1;
        }
    }

    const char* string_formats[] = {"%2s", " %5c", "%[^!]", "%3[^\r\n]", "%[^\r\n]"};
    const char* string_inputs[] = {" \nhello X", " \tab\ncdX", "ab\ncd!tail", "abcde\r\n", "\nnext"};
    for (size_t i = 0; i < sizeof(string_formats) / sizeof(string_formats[0]); i++)
    {
        const size_t size = strlen(string_inputs[i]);
        // omit the null terminator from the actual allocation
        unsigned char* data = new unsigned char[size];
        memcpy(data, string_inputs[i], size);
        const unsigned char* mem = data;
        const unsigned char* mem2 = (const unsigned char*)string_inputs[i];
        ncnn::DataReaderFromMemory dr(mem, size);
        ncnn::DataReaderFromMemory dr2(mem2);
        char value[32] = {0};
        char value2[32] = {0};
        const int ret = dr.scan(string_formats[i], value);
        const int ret2 = dr2.scan(string_formats[i], value2);
        const bool failed = ret != ret2 || memcmp(value, value2, sizeof(value)) || mem - data != mem2 - (const unsigned char*)string_inputs[i];
        delete[] data;
        if (failed)
        {
            fprintf(stderr, "test_datareader_scan string failed format=%s\n", string_formats[i]);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_STRING

int main()
{
    return 0
           || test_datareader_memory()
           || test_datareader_empty()
#if NCNN_STRING
           || test_datareader_scan_bounds()
           || test_datareader_scan_mixed()
           || test_datareader_scan()
#endif
           ;
}
