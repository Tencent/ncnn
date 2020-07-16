// Copyright 2016 SoundAI Technology Co., Ltd. (author: Charles Wang)
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

#include "statisticspooling.h"

#include <float.h>
#include <limits.h>
#include <math.h>

namespace ncnn {

StatisticsPooling::StatisticsPooling()
{
    one_blob_only = true;
    support_inplace = false;
}

int StatisticsPooling::load_param(const ParamDict& pd)
{
    include_stddev = pd.get(0, 0);

    return 0;
}

int StatisticsPooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    size_t elemsize = bottom_blob.elemsize;

    int out_channels = channels;
    if (include_stddev)
    {
        out_channels *= 2;
    }

    top_blob.create(out_channels, elemsize, opt.blob_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

        float mean = 0.f;
        for (int i = 0; i < size; i++)
        {
            mean += ptr[i];
        }
        top_blob[q] = mean / w / h;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = channels; q < out_channels; q++)
    {
        const float* ptr = bottom_blob.channel(q - channels);

        float std = 0.f;
        for (int i = 0; i < size; i++)
        {
            std += pow((ptr[i] - top_blob[q - channels]), 2);
        }
        top_blob[q] = sqrt(std / w / h);
    }

    return 0;
}

} // namespace ncnn
