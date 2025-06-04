// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "testutil.h"

static int test_inversespectrogram(int frames, int freqs, int n_fft, int returns, int hoplen, int winlen, int window_type, int center, int normalized)
{
    ncnn::Mat a = RandomMat(2, frames, freqs);

    ncnn::ParamDict pd;
    pd.set(0, n_fft);
    pd.set(1, returns);
    pd.set(2, hoplen);
    pd.set(3, winlen);
    pd.set(4, window_type);
    pd.set(5, center);
    pd.set(7, normalized);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("InverseSpectrogram", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_inversespectrogram failed frames=%d freqs=%d n_fft=%d returns=%d hoplen=%d winlen=%d window_type=%d center=%d normalized=%d\n", frames, freqs, n_fft, returns, hoplen, winlen, window_type, center, normalized);
    }

    return ret;
}

static int test_inversespectrogram_0()
{
    return 0
           || test_inversespectrogram(17, 1, 1, 0, 1, 1, 0, 1, 0)
           || test_inversespectrogram(39, 9, 17, 0, 7, 15, 0, 0, 1)
           || test_inversespectrogram(128, 6, 10, 0, 2, 7, 1, 1, 1)
           || test_inversespectrogram(255, 17, 17, 1, 14, 17, 2, 0, 0)
           || test_inversespectrogram(124, 28, 55, 2, 12, 55, 1, 1, 2);
}

int main()
{
    SRAND(7767517);

    return test_inversespectrogram_0();
}
