// License

#include "celu.h"

#include <math.h>

namespace ncnn {

CELU::CELU()
{
    one_blob_only = true;
    support_inplace = true;
}

int CELU::load_param(const ParamDict& pd)
{
    alpha = pd.get(0, 1.f);

    return 0;
}

int CELU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i] / alpha) - 1.f) * alpha;
        }
    }

    return 0;
}

} // namespace ncnn
