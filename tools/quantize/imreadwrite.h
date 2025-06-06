// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef IMREADWRITE_H
#define IMREADWRITE_H

#include <limits.h>
#include <string.h>
#include "allocator.h"
#include "mat.h"

#ifndef NCNN_XADD
using ncnn::NCNN_XADD;
#endif

typedef unsigned char uchar;

enum
{
    CV_LOAD_IMAGE_UNCHANGED = -1,
    CV_LOAD_IMAGE_GRAYSCALE = 0,
    CV_LOAD_IMAGE_COLOR = 1,
};

enum
{
    CV_IMWRITE_JPEG_QUALITY = 1
};

// minimal opencv style data structure implementation
namespace cv {

#define CV_8UC1  1
#define CV_8UC3  3
#define CV_8UC4  4
#define CV_32FC1 4

struct Mat
{
    Mat()
        : data(0), refcount(0), rows(0), cols(0), c(0)
    {
    }

    Mat(int _rows, int _cols, int flags)
        : data(0), refcount(0)
    {
        create(_rows, _cols, flags);
    }

    // copy
    Mat(const Mat& m)
        : data(m.data), refcount(m.refcount)
    {
        if (refcount)
            NCNN_XADD(refcount, 1);

        rows = m.rows;
        cols = m.cols;
        c = m.c;
    }

    Mat(int _rows, int _cols, int flags, void* _data)
        : data((unsigned char*)_data), refcount(0)
    {
        rows = _rows;
        cols = _cols;
        c = flags;
    }

    ~Mat()
    {
        release();
    }

    // assign
    Mat& operator=(const Mat& m)
    {
        if (this == &m)
            return *this;

        if (m.refcount)
            NCNN_XADD(m.refcount, 1);

        release();

        data = m.data;
        refcount = m.refcount;

        rows = m.rows;
        cols = m.cols;
        c = m.c;

        return *this;
    }

    void create(int _rows, int _cols, int flags)
    {
        release();

        rows = _rows;
        cols = _cols;
        c = flags;

        if (total() > 0)
        {
            // refcount address must be aligned, so we expand totalsize here
            size_t totalsize = (total() + 3) >> 2 << 2;
            data = (uchar*)ncnn::fastMalloc(totalsize + (int)sizeof(*refcount));
            refcount = (int*)(((uchar*)data) + totalsize);
            *refcount = 1;
        }
    }

    void release()
    {
        if (refcount && NCNN_XADD(refcount, -1) == 1)
            ncnn::fastFree(data);

        data = 0;

        rows = 0;
        cols = 0;
        c = 0;

        refcount = 0;
    }

    Mat clone() const
    {
        if (empty())
            return Mat();

        Mat m(rows, cols, c);

        if (total() > 0)
        {
            memcpy(m.data, data, total());
        }

        return m;
    }

    bool empty() const
    {
        return data == 0 || total() == 0;
    }

    int type() const
    {
        return c;
    }

    size_t total() const
    {
        return (size_t)cols * rows * c;
    }

    uchar* data;

    // pointer to the reference counter;
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    int rows;
    int cols;

    int c;
};

enum ImreadModes
{
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1
};

Mat imread(const std::string& path, int flags = IMREAD_COLOR);

enum ImwriteFlags
{
    IMWRITE_JPEG_QUALITY = 1
};

bool imwrite(const std::string& path, const Mat& m, const std::vector<int>& params = std::vector<int>());

} // namespace cv

#endif // IMREADWRITE_H
