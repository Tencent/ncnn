// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "absval.h"
#include "thread.h"

namespace ncnn {

AbsVal::AbsVal()
{
    one_blob_only = true;
    support_inplace = true;
}

int AbsVal::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    if (opt.num_threads > 64)
    {
        ThreadWorkspace workspace;
        workspace.layer = (Layer*)this;
        MutilThread thread(workspace, opt);
        std::vector<Mat> workspace_blobs;
        workspace_blobs.push_back(bottom_top_blob);
        thread.join(workspace_blobs);
        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0)
                ptr[i] = -ptr[i];
        }
    }

    return 0;
}

int AbsVal::forward_thread(void* workspace)
{
    ThreadInfoExc* info = (ThreadInfoExc*)workspace;
    Mat& bottom_top_blob = info->mats->at(0);
    if (bottom_top_blob.elemsize == 1)
    {
        int8_t* ptr = (int8_t*)bottom_top_blob.data;
        const int8_t flag = 1 << 7;
        for (size_t i = info->start_index; i < info->end_index; i++)
        {
            if (ptr[i] & flag)
            {
                ptr[i] = -ptr[i];
            }
        }
    }
    else if (bottom_top_blob.elemsize == 2)
    {
        int16_t* ptr = (int16_t*)bottom_top_blob.data;
        const int16_t flag = 1 << 15;
        for (size_t i = info->start_index; i < info->end_index; i++)
        {
            if (ptr[i] & flag)
            {
                ptr[i] = -ptr[i];
            }
        }
    }
    else
    {
        float* ptr = (float*)bottom_top_blob.data;
        for (size_t i = info->start_index; i < info->end_index; i++)
        {
            if (ptr[i] < 0)
            {
                ptr[i] = -ptr[i];
            }
        }
    }

    return 0;
}

} // namespace ncnn
