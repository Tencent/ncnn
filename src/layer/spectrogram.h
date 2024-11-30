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

#ifndef LAYER_SPECTROGRAM_H
#define LAYER_SPECTROGRAM_H

#include "layer.h"

namespace ncnn {

class Spectrogram : public Layer
{
public:
    Spectrogram();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    int n_fft;
    int power;
    int hoplen;
    int winlen;
    int window_type; // 0=ones 1=hann 2=hamming
    int center;
    int pad_type;   // 0=CONSTANT 1=REPLICATE 2=REFLECT
    int normalized; // 0=disabled 1=sqrt(n_fft) 2=window-l2-energy
    int onesided;

    Mat window_data;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_H
