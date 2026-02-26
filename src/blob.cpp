// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "blob.h"

namespace ncnn {

Blob::Blob()
{
    producer = -1;
    consumer = -1;
}

} // namespace ncnn
