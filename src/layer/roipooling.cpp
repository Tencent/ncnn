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

#include "roipooling.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(ROIPooling)

ROIPooling::ROIPooling()
{
}

#if NCNN_STDIO
#if NCNN_STRING
int ROIPooling::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %f",
                       &pooled_width, &pooled_height, &spatial_scale);
    if (nscan != 3)
    {
        fprintf(stderr, "ROIPooling load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
#endif // NCNN_STRING
int ROIPooling::load_param_bin(FILE* paramfp)
{
    fread(&pooled_width, sizeof(int), 1, paramfp);

    fread(&pooled_height, sizeof(int), 1, paramfp);

    fread(&spatial_scale, sizeof(float), 1, paramfp);

    return 0;
}
#endif // NCNN_STDIO

int ROIPooling::load_param(const unsigned char*& mem)
{
    pooled_width = *(int*)(mem);
    mem += 4;

    pooled_height = *(int*)(mem);
    mem += 4;

    spatial_scale = *(float*)(mem);
    mem += 4;

    return 0;
}

int ROIPooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    const Mat& roi_blob = bottom_blobs[1];
    int num_roi = roi_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(pooled_width, pooled_height, channels);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: max pool over R
    #pragma omp parallel for
    for (int n = 0; n < num_roi; n++)
    {
        const float* roi_ptr = roi_blob.data + 4 * n;

        int roi_x = round(roi_ptr[0] * spatial_scale);
        int roi_y = round(roi_ptr[1] * spatial_scale);
        int roi_w = round(roi_ptr[2] * spatial_scale);
        int roi_h = round(roi_ptr[3] * spatial_scale);

        float bin_size_w = (float)roi_w / (float)pooled_width;
        float bin_size_h = (float)roi_h / (float)pooled_height;

        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    // Compute pooling region for this output unit:
                    //  start (included) = floor(ph * roi_height / pooled_height)
                    //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height)
                    int hstart = roi_y + floor((float)(ph) * bin_size_h);
                    int wstart = roi_x + floor((float)(pw) * bin_size_w);
                    int hend = roi_y + ceil((float)(ph + 1) * bin_size_h);
                    int wend = roi_x + ceil((float)(pw + 1) * bin_size_w);

                    hstart = std::min(std::max(hstart, 0), h);
                    wstart = std::min(std::max(wstart, 0), w);
                    hend = std::min(std::max(hend, 0), h);
                    wend = std::min(std::max(wend, 0), w);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    float max = is_empty ? 0.f : ptr[hstart * w + wstart];

                    for (int y = hstart; y < hend; y++)
                    {
                        for (int x = wstart; x < wend; x++)
                        {
                            int index = y * w + x;
                            max = std::max(max, ptr[index]);
                        }
                    }

                    outptr[pw] = max;
                }

                outptr += pooled_width;
            }
        }

    }

    return 0;
}

} // namespace ncnn
