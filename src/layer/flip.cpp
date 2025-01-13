// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "flip.h"

namespace ncnn {

Flip::Flip()
{
    one_blob_only = true;
}

int Flip::load_param(const ParamDict& pd)
{
    axis = pd.get(0, Mat());
    // 调试
    // const int *axis_ptr = axis;
    // printf("axis_len = %d\n", axis.w);
    // printf("axis[0] = %d\n", axis_ptr[0]);
    return 0;
}

int Flip::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // 已知参数
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    // 校准输入参数
    if (axis.w > 4)
    {
        return -1;
    }
    const int* axis_ptr = axis;

    if (dims == 1)
    {
        // 1D 只有一种情况
        top_blob.create(w, elemsize, opt.blob_allocator);
        const float* ptr = bottom_blob;
        float* outptr = top_blob;
        for (int i = 0; i < w; i++)
        {
            outptr[i] = ptr[w - 1 - i];
        }
    }
    else if (dims == 2)
    {
        // 2D 有三种，安装上下、左右和上下左右同时翻转;[-2/0上下翻转, -1/1左右翻转,交叉为上下左右翻转]
        top_blob.create(w, h, elemsize, opt.blob_allocator);
        if (axis.w == 1)
        {
            if (axis_ptr[0] == -2 || axis_ptr[0] == 0)
            {
                // 按照行翻转
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.row(h - 1 - i); // 从最后一行开始
                    float* outptr = top_blob.row(i);               // 输出到当前行

                    // 直接复制整行数据
                    memcpy(outptr, ptr, w * sizeof(float));
                }
            }
            else
            {
                // 按照列翻转
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.row(i);
                    float* outptr = top_blob.row(i);

                    // 使用临时buffer存储反转的行数据
                    std::vector<float> line_buffer(w);
                    for (int j = 0; j < w; j++)
                    {
                        line_buffer[j] = ptr[w - 1 - j];
                    }

                    // 一次性复制整行
                    memcpy(outptr, line_buffer.data(), w * sizeof(float));
                }
            }
        }
        else
        {
            // 当axis.w=2时，上下左右都翻转
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(h - 1 - i); // 从最后一行开始读取
                float* outptr = top_blob.row(i);               // 输出到当前行

                // 每行内左右翻转
                for (int j = 0; j < w; j++)
                {
                    outptr[j] = ptr[w - 1 - j]; // 反向读取每行像素
                }
            }
        }
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        if (axis.w == 1)
        {
            // w、h、c
            // 约定到正数，简化后续判断
            int axis0 = axis_ptr[0] < 0 ? 3 + axis_ptr[0] : axis_ptr[0];
            if (axis0 == 0)
            {
                // -3/0 整体上下翻转
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const float* ptr = bottom_blob.channel(channels - 1 - i).row(j); // 从最后一个channel开始
                        float* outptr = top_blob.channel(i).row(j);
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
            else if (axis0 == 1)
            {
                // -2/1 整体内部上下翻转
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const float* ptr = bottom_blob.channel(i).row(h - 1 - j);
                        float* outptr = top_blob.channel(i).row(j);
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
            else
            {
                // -1/2 整体左右翻转
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const float* ptr = bottom_blob.channel(i).row(j);
                        float* outptr = top_blob.channel(i).row(j);
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = ptr[w - 1 - k];
                        }
                    }
                }
            }
        }
        else if (axis.w == 2)
        {
            // ch、cw、hw
            int axis0 = axis_ptr[0] < 0 ? 3 + axis_ptr[0] : axis_ptr[0];
            int axis1 = axis_ptr[1] < 0 ? 3 + axis_ptr[1] : axis_ptr[1];
            int axis_sum = axis0 + axis1;
            if (axis_sum == 1)
            {
                // 对应ch
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        // 组合两种翻转：channel维度和行维度同时翻转
                        const float* ptr = bottom_blob.channel(channels - 1 - i).row(h - 1 - j);
                        float* outptr = const_cast<float*>(top_blob.channel(i).row(j));
                        memcpy(outptr, ptr, w * sizeof(float));
                    }
                }
            }
            else if (axis_sum == 2)
            {
                // 对应cw
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const float* ptr = bottom_blob.channel(channels - 1 - i).row(j);
                        float* outptr = top_blob.channel(i).row(j);
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = ptr[w - 1 - k];
                        }
                    }
                }
            }
            else if (axis_sum == 3)
            {
                // 对应hw
                for (int i = 0; i < channels; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        const float* ptr = bottom_blob.channel(i).row(h - 1 - j);
                        float* outptr = top_blob.channel(i).row(j);

                        // 增加左右翻转
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = ptr[w - 1 - k];
                        }
                    }
                }
            }
        }
        else
        {
            // whc
            for (int i = 0; i < channels; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    const float* ptr = bottom_blob.channel(channels - 1 - i).row(h - 1 - j);
                    float* outptr = top_blob.channel(i).row(j);

                    // 左右翻转实现完全倒序
                    for (int k = 0; k < w; k++)
                    {
                        outptr[k] = ptr[w - 1 - k];
                    }
                }
            }
        }
    }
    else if (dims == 4)
    {
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (axis.w == 1)
        {
            // w、h、d、c
            int axis0 = axis_ptr[0] < 0 ? 4 + axis_ptr[0] : axis_ptr[0];
            if (axis0 == 0)
            {
                // -4/0 整体上下翻转 torch中按c维度翻转
                for (int c = 0; c < channels; c++) // 遍历channels=3
                {
                    int flipped_c = channels - 1 - c; // 计算channels翻转位置

                    for (int z = 0; z < d; z++) // 遍历d=2维度
                    {
                        for (int j = 0; j < h; j++) // 遍历行
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + j);
                            float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(z * h + j));
                            memcpy(outptr, ptr, w * sizeof(float));
                        }
                    }
                }
            }
            else if (axis0 == 1)
            {
                // -3/1 torh中按d维度内部上下翻转
                for (int i = 0; i < channels; i++) // 遍历channels
                {
                    for (int z = 0; z < d; z++) // 遍历d维度
                    {
                        for (int j = 0; j < h; j++) // 遍历h维度
                        {
                            // 翻转d维度的数据读取位置
                            const float* ptr = bottom_blob.channel(i).row((d - 1 - z) * h + j);
                            float* outptr = const_cast<float*>(top_blob.channel(i).row(z * h + j));
                            // 逐行复制w元素
                            memcpy(outptr, ptr, w * sizeof(float));
                        }
                    }
                }
            }
            else if (axis0 == 2)
            {
                // -2/2 按torch中H维度翻转 上下
                for (int i = 0; i < channels; i++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            const float* ptr = bottom_blob.channel(i).row(z * h + (h - 1 - j));
                            float* outptr = top_blob.channel(i).row(z * h + j);
                            memcpy(outptr, ptr, w * sizeof(float));
                        }
                    }
                }
            }
            else
            {
                // -1/3 按torch中W维度翻转 左右
                for (int i = 0; i < channels; i++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            const float* ptr = bottom_blob.channel(i).row(z * h + j);
                            float* outptr = top_blob.channel(i).row(z * h + j);
                            for (int k = 0; k < w; k++)
                            {
                                outptr[k] = ptr[w - 1 - k];
                            }
                        }
                    }
                }
            }
        }
        else if (axis.w == 2)
        {
            // dc1、dh2、dw3、ch3、cw4、hw5
            int axis0 = axis_ptr[0] < 0 ? 4 + axis_ptr[0] : axis_ptr[0];
            int axis1 = axis_ptr[1] < 0 ? 4 + axis_ptr[1] : axis_ptr[1];
            int axis_sum = axis0 + axis1;
            if (axis_sum == 1)
            {
                // 对应dc
                for (int c = 0; c < channels; c++) // 遍历channels
                {
                    int flipped_c = channels - 1 - c; // 翻转后的channel位置

                    for (int z = 0; z < d; z++) // 遍历d维度
                    {
                        int flipped_d = d - 1 - z; // 翻转后的d位置

                        for (int j = 0; j < h; j++) // 遍历行
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + j);
                            float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(flipped_d * h + j));
                            memcpy(outptr, ptr, w * sizeof(float));
                        }
                    }
                }
            }
            else if (axis_sum == 2)
            {
                // 对应dh
                for (int c = 0; c < channels; c++) // 遍历 channels=2 维度
                {
                    int flipped_c = channels - 1 - c; // 计算 c 维度翻转位置 (0→1, 1→0)

                    for (int z = 0; z < d; z++) // 遍历 d=3 维度
                    {
                        // 按翻转顺序逐行复制 h 维度数据
                        for (int i = 0; i < h; i++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + i);
                            float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(z * h + (h - 1 - i))); // 保持z维度顺序,翻转h维度
                            memcpy(outptr, ptr, w * sizeof(float));                                                   // 按行复制，保持 w 维度顺序
                        }
                    }
                }
            }
            else if (axis_sum == 3)
            {
                // 对应dw；有一个为0或3
                if (axis0 == 0 || axis0 == 3)
                {
                    // 对应dw
                    for (int c = 0; c < channels; c++)
                    {
                        int flipped_c = channels - 1 - c; // 翻转c维度

                        for (int z = 0; z < d; z++) // d维度保持不变
                        {
                            for (int j = 0; j < h; j++) // h维度保持不变
                            {
                                const float* ptr = bottom_blob.channel(c).row(z * h + j);
                                float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(z * h + j));

                                // 翻转w维度
                                for (int k = 0; k < w; k++)
                                {
                                    outptr[k] = ptr[w - 1 - k];
                                }
                            }
                        }
                    }
                }
                else
                {
                    // 对应ch
                    for (int c = 0; c < channels; c++)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            int flipped_d = d - 1 - z;

                            for (int j = 0; j < h; j++)
                            {
                                int flipped_h = h - 1 - j;
                                // 读取源数据
                                const float* ptr = bottom_blob.channel(c).row(z * h + j);
                                float* outptr = const_cast<float*>(top_blob.channel(c).row(flipped_d * h + flipped_h));
                                memcpy(outptr, ptr, w * sizeof(float));
                            }
                        }
                    }
                }
            }
            else if (axis_sum == 4)
            {
                // 对应cw
                for (int c = 0; c < channels; c++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        int flipped_d = d - 1 - z; // 翻转 d 维度

                        for (int j = 0; j < h; j++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + j);
                            float* outptr = const_cast<float*>(top_blob.channel(c).row(flipped_d * h + j)); // c维度保持不变

                            // 翻转 w 维度
                            for (int k = 0; k < w; k++)
                            {
                                outptr[k] = ptr[w - 1 - k];
                            }
                        }
                    }
                }
            }
            else
            {
                // 对应hw
                for (int c = 0; c < channels; c++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        for (int j = 0; j < h; j++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + j);
                            float* outptr = const_cast<float*>(top_blob.channel(c).row(z * h + (h - 1 - j))); // 翻转 h 维度

                            // 翻转 w 维度
                            for (int k = 0; k < w; k++)
                            {
                                outptr[k] = ptr[w - 1 - k];
                            }
                        }
                    }
                }
            }
        }
        else if (axis.w == 3)
        {
            return 0; // 在线debug
            // dch3、dcw4、chw6
            int axis0 = axis_ptr[0] < 0 ? 4 + axis_ptr[0] : axis_ptr[0];
            int axis1 = axis_ptr[1] < 0 ? 4 + axis_ptr[1] : axis_ptr[1];
            int axis2 = axis_ptr[2] < 0 ? 4 + axis_ptr[2] : axis_ptr[2];
            int axis_sum = axis0 + axis1 + axis2;
            if (axis_sum == 3)
            {
                // 对应dch，除w外，其余全翻转
                for (int c = 0; c < channels; c++)
                {
                    int flipped_c = channels - 1 - c; // 翻转c维度

                    for (int z = 0; z < d; z++)
                    {
                        int flipped_d = d - 1 - z; // 翻转d维度

                        for (int i = 0; i < h; i++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + i);
                            float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(flipped_d * h + (h - 1 - i))); // 翻转h维度
                            memcpy(outptr, ptr, w * sizeof(float));                                                           // w维度保持不变
                        }
                    }
                }
            }
            else if (axis_sum == 4)
            {
                // 对应dcw，除h外，其余全翻转
                for (int c = 0; c < channels; c++)
                {
                    int flipped_c = channels - 1 - c; // 翻转c维度

                    for (int z = 0; z < d; z++)
                    {
                        int flipped_d = d - 1 - z; // 翻转d维度

                        for (int i = 0; i < h; i++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + i);
                            float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(flipped_d * h + i)); // h维度保持不变

                            // 翻转w维度
                            for (int k = 0; k < w; k++)
                            {
                                outptr[k] = ptr[w - 1 - k];
                            }
                        }
                    }
                }
            }
            else if (axis_sum == 6)
            {
                // 对应chw,除了c外全翻转
                for (int c = 0; c < channels; c++) // c维度保持不变
                {
                    for (int z = 0; z < d; z++)
                    {
                        int flipped_d = d - 1 - z; // 翻转d维度

                        for (int i = 0; i < h; i++)
                        {
                            const float* ptr = bottom_blob.channel(c).row(z * h + i);
                            float* outptr = const_cast<float*>(top_blob.channel(c).row(flipped_d * h + (h - 1 - i))); // 翻转h维度

                            // 翻转w维度
                            for (int k = 0; k < w; k++)
                            {
                                outptr[k] = ptr[w - 1 - k];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // dchw全部翻转
            for (int c = 0; c < channels; c++)
            {
                int flipped_c = channels - 1 - c; // 翻转c维度

                for (int z = 0; z < d; z++)
                {
                    int flipped_d = d - 1 - z; // 翻转d维度

                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr = bottom_blob.channel(c).row(z * h + i);
                        float* outptr = const_cast<float*>(top_blob.channel(flipped_c).row(flipped_d * h + (h - 1 - i))); // 翻转h维度

                        // 翻转w维度
                        for (int k = 0; k < w; k++)
                        {
                            outptr[k] = ptr[w - 1 - k];
                        }
                    }
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

} // namespace ncnn
