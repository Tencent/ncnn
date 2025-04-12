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

static int test_spectrogram(int size, int n_fft, int power, int hoplen, int winlen, int window_type, int center, int pad_type, int normalized, int onesided)
{
    ncnn::Mat a = RandomMat(size);

    ncnn::ParamDict pd;
    pd.set(0, n_fft);
    pd.set(1, power);
    pd.set(2, hoplen);
    pd.set(3, winlen);
    pd.set(4, window_type);
    pd.set(5, center);
    pd.set(6, pad_type);
    pd.set(7, normalized);
    pd.set(8, onesided);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Spectrogram", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_spectrogram failed size=%d n_fft=%d power=%d hoplen=%d winlen=%d window_type=%d center=%d pad_type=%d normalized=%d onesided=%d\n", size, n_fft, power, hoplen, winlen, window_type, center, pad_type, normalized, onesided);
    }

    return ret;
}

static int test_spectrogram_0()
{
    return 0
           || test_spectrogram(17, 1, 0, 1, 1, 0, 1, 0, 0, 0)
           || test_spectrogram(39, 17, 0, 7, 15, 0, 0, 0, 1, 0)
           || test_spectrogram(128, 10, 0, 2, 7, 1, 1, 1, 1, 1)
           || test_spectrogram(255, 17, 1, 14, 17, 2, 0, 0, 0, 1)
           || test_spectrogram(124, 55, 2, 12, 55, 1, 1, 2, 2, 0);
}

static int test_spectrogram_eval(int size, int n_fft, int power, int hoplen, int winlen, int window_type, int center, int pad_type, int normalized, int onesided,float * in,float * std)
{
    ncnn::Layer * layer = ncnn::create_layer("Spectrogram");

    ncnn::ParamDict pd;
    pd.set(0, n_fft);
    pd.set(1, power);
    pd.set(2, hoplen);
    pd.set(3, winlen);
    pd.set(4, window_type);
    pd.set(5, center);
    pd.set(6, pad_type);
    pd.set(7, normalized);
    pd.set(8, onesided);

    ncnn::Mat input = ncnn::Mat(size);
    memcpy(input, in, size * sizeof(float));

    ncnn::Mat output;

    ncnn::Option opt;
    opt.num_threads = 2;

    layer->load_param(pd);
    layer->create_pipeline(opt);
    layer->forward(input, output, opt);
    layer->destroy_pipeline(opt);

    const float epsilon = 1e-6;

    for (int i = 0; i < output.c; i++)
    {
        float * output_data = output.channel(i);
        for (int j = 0; j < output.h; j++)
        {
            for (int k = 0; k < output.w; k++)
            {
                if (fabs(output_data[j * output.w + k] - std[i * output.h * output.w + j * output.w + k]) > epsilon)
                {
                    fprintf(stderr, "test_spectrogram failed size=%d n_fft=%d power=%d hoplen=%d winlen=%d window_type=%d center=%d pad_type=%d normalized=%d onesided=%d\n", size, n_fft, power, hoplen, winlen, window_type, center, pad_type, normalized, onesided);
                    return 1;
                }
            }
        }
    }

    delete layer;
    return 0;
}

static int test_spectrogram_1()
{
    float input_0[16] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    float std_0[] = {
        0.05000000f, 0.40000001f, 0.80000001f, 1.20000005f, 1.59999990f, 2.00000000f, 2.40000010f, 2.79999995f, 0.75000000f, 0.05000000f, 0.22360681f, 0.41231057f, 0.60827625f, 0.80622578f, 1.00498760f, 1.20415950f, 1.40356684f, 0.75000000f, 0.05000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000006f, 0.00000000f, 0.00000000f, 0.00000000f, 0.75000000f
    };
    float std_1[] = {
        0.80000001f, 1.20000005f, 1.59999990f, 2.00000000f, 2.40000010f, 0.68649411f, 1.02670193f, 1.36751485f, 1.70857072f, 2.04974818f, 0.41231057f, 0.60827625f, 0.80622578f, 1.00498760f, 1.20415950f, 0.13684234f, 0.18942842f, 0.24475159f, 0.30130789f, 0.35851428f, 0.00000000f, 0.00000000f, 0.00000006f, 0.00000000f, 0.00000000f
    };
    float std_2[] = {
        0.28284273f, 0.49497476f, 0.70710677f, 0.24271232f, 0.42322639f, 0.60407096f, 0.14577380f, 0.25000000f, 0.35531676f, 0.04838108f, 0.07667736f, 0.10652842f, 0.00000000f, 0.00000002f, 0.00000000f, 0.04838108f, 0.07667736f, 0.10652842f, 0.14577380f, 0.25000000f, 0.35531676f, 0.24271232f, 0.42322639f, 0.60407096f
    };

    return
    test_spectrogram_eval(16, 4, 1, 2, 4, 1, 1, 0, 0, 1, input_0, std_0)
    || test_spectrogram_eval(16, 8, 1, 2, 4, 1, 0, 0, 0, 1, input_0, std_1)
    || test_spectrogram_eval(16, 8, 1, 3, 4, 1, 0, 0, 1, 0, input_0, std_2);

}

int main()
{
    SRAND(7767517);

    return test_spectrogram_0() || test_spectrogram_1();
}
