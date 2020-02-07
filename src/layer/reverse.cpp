#include "reverse.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Reverse)

Reverse::Reverse()
{
    one_blob_only = true;
    support_inplace = false;
}


int Reverse::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels= bottom_blob.c;


    top_blob.create(w,h,channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int q=0; q<channels; q++)//8
    {
        const float* ptr = bottom_blob.channel(q);
        float* target = top_blob.channel(channels-q-1);
        for (int i = 0; i < h; i++)//1
        {
            for (int j = 0; i < w; i++)//256
            {
               target[j+h*i] = ptr[j+h*i];
            }
        }
    }

    
    return 0;
}

} // namespace ncnn