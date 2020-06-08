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

#include "argmax.h"
#include <algorithm>
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(ArgMax)

ArgMax::ArgMax()
{
    one_blob_only = true;
}

int ArgMax::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);  // id 0, default 0;
    return 0;
}

int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<float> vec;
    int space, max_length;

    if (axis == 0) {
        top_blob.create(bottom_blob.w, bottom_blob.h, 1, 4u, opt.blob_allocator);
        space = bottom_blob.w * bottom_blob.h;
        max_length = bottom_blob.c;
    } else if (axis == 1) {
        top_blob.create(bottom_blob.w, 1, bottom_blob.c, 4u, opt.blob_allocator);
        space = bottom_blob.w;
        max_length = bottom_blob.h;
    } else if (axis == 2) {
        top_blob.create(1, bottom_blob.h, bottom_blob.c, 4u, opt.blob_allocator);
        space = 1;
        max_length = bottom_blob.w;
    } else
        return -1001;
        
    if (top_blob.empty())
        return -1001;

    const float* ptr = bottom_blob;
    float* outptr = top_blob;
    int size = top_blob.w * top_blob.h * top_blob.c;
    //printf("size: %d, space: %d\n", size, space);
    
    //#pragma omp parallel for num_threads(opt.num_threads)
    for (int i=0; i<size; i++) {
        for (int j=0; j<max_length; j++) {
            vec.push_back(ptr[(i / space * max_length + j) * space + i % space]);
            //printf("%f ", ptr[(i / space * max_length + j) * space + i % space]);
        }
        outptr[i] = argmax(vec.begin(), vec.end());
        //printf(" res: %f\n", outptr[i]);
        vec.clear();
    }
    
    return 0;
}

} // namespace ncnn
