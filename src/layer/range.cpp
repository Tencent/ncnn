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

#include "range.h"

#include <math.h>

namespace ncnn {

Range::Range()
{
    one_blob_only = false;
    support_inplace = false;
}

int Range::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2 || bottom_blobs.size() > 3)
        return -100;

    const Mat& start = bottom_blobs[0];
    const Mat& limit = bottom_blobs[1];
    const Mat& delta = bottom_blobs.size() == 3 ? bottom_blobs[2] : Mat();
    Mat& output = top_blobs[0];

    if (start.w * start.h * start.d * start.c != 1 || limit.w * limit.h * limit.d * limit.c != 1 || (!delta.empty() && delta.w * delta.h * delta.d * delta.c != 1))
        return -100;

    const int* start_ptr = start;
    const int* limit_ptr = limit;
    const int* delta_ptr = delta;

    int start_val = start_ptr[0];
    int limit_val = limit_ptr[0];
    int delta_val = delta.empty() ? 1 : delta_ptr[0];

    if (delta_val == 0)
        return -100;

    int number_of_elements = static_cast<int>(ceil(static_cast<float>(limit_val - start_val) / delta_val));
    if (number_of_elements < 0)
        number_of_elements = 0;

    output.create(number_of_elements, start.elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    int* outptr = output;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < number_of_elements; i++)
    {
        outptr[i] = start_val + (i * delta_val);
    }

    return 0;
}

} // namespace ncnn
