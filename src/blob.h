// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_BLOB_H
#define NCNN_BLOB_H

#include "mat.h"
#include "platform.h"

namespace ncnn {

class NCNN_EXPORT Blob
{
public:
    // empty
    Blob();

public:
#if NCNN_STRING
    // blob name
    std::string name;
#endif // NCNN_STRING
    // layer index which produce this blob as output
    int producer;
    // layer index which need this blob as input
    int consumer;
    // shape hint
    Mat shape;
};

} // namespace ncnn

#endif // NCNN_BLOB_H
