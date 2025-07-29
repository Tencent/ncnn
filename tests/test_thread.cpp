#include "testutil.h"
#include "thread.h"

class TestLayer : public ncnn::Layer
{
public:
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt)
    {
        ThreadWorkspace workspace;
        workspace.layer = (Layer*)this;
        MutilThread thread(workspace, opt);
        std::vector<Mat> workspace_blobs;
        workspace_blobs.push_back(bottom_top_blob);
        thread.join(workspace_blobs);
        return 0;
    }
    virtual int forward_thread(void* workspace)
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
};

static int test_thread(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("TestLayer", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_thread failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_thread_0(){
    return 0
          || test_thread(RandomMat(5,6,7,24))
          || test_thread(RandomMat(5,6,7,12))
          || test_thread(RandomMat(5,6,7,13));

}

static int test_thread_1(){
    return 0
          || test_thread(RandomMat(5,7,24))
          || test_thread(RandomMat(5,6,24))
          || test_thread(RandomMat(7,9,24));
}

static int test_thread_2(){
    return 0
          || test_thread(RandomMat(7,12))
          || test_thread(RandomMat(5,12))
          || test_thread(RandomMat(9,12));
}

static int test_thread_3(){
    return 0
          || test_thread(RandomMat(7))
          || test_thread(RandomMat(128))
          || test_thread(RandomMat(256));
}

int main()
{
    return 0 
           || test_thread_0()
           || test_thread_1()
           || test_thread_2()
           || test_thread_3();
}