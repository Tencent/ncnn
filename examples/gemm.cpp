#include "net.h"
#include <stdio.h>
#include <sys/time.h>


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s m, n, k\n", argv[0]);
        return -1;
    }
    int m, n, k;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    printf("m = %d, n = %d, k = %d\n", m, n, k);
    ncnn::Mat a = ncnn::Mat(m, k);
    a.fill(1.f);
    ncnn::Mat b = ncnn::Mat(k, n);
    b.fill(2.f);
    // ncnn::Mat c = ncnn::Mat(4096, 4096);
    // 构建NCNN的net，并加载转换好的模型
    ncnn::Net net;
    net.load_param("test_gemm.ncnn.param");
    net.load_model("test_gemm.ncnn.model.bin");

 // 创建网络提取器，设置网络输入，线程数，light模式等等
    ncnn::Extractor ex = net.create_extractor();
    // ex.set_light_mode(false);
    // ex.set_num_threads(1);
    ex.input("in0", a);
    ex.input("in1", b);
 // 调用extract接口，完成网络推理，获得输出结果
    ncnn::Mat c;
    // add timeit code
    struct timeval start_time, end_time;
    long elapsed_time;

    // 获取起始时间
    gettimeofday(&start_time, NULL);

    // 计时

    ex.extract("out0", c);
    gettimeofday(&end_time, NULL);
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000L + (end_time.tv_usec - start_time.tv_usec);
    printf("Time elapsed: %ld microseconds\n", elapsed_time);
    // printf("%f\n", c[0]);
    return 0;
}