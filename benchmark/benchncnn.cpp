#include <float.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include <iostream>
#include <numeric>
#include <algorithm>
#include "benchmark.h"
#include "cpu.h"
#include "net.h"

namespace ncnn {

// always return empty weights
class ModelBinFromEmpty : public ModelBin
{
public:
    virtual Mat load(int w, int /*type*/) const { return Mat(w); }
};

class BenchNet : public Net
{
public:
    int load_model()
    {
        // load file
        int ret = 0;

        ModelBinFromEmpty mb;
        for (size_t i=0; i<layers.size(); i++)
        {
            Layer* layer = layers[i];

            int lret = layer->load_model(mb);
            if (lret != 0)
            {
                fprintf(stderr, "layer load_model %d failed\n", (int)i);
                ret = -1;
                break;
            }
        }

        return ret;
    }
};

} // namespace ncnn

static int g_loop_count = 80;

void benchmark(const char* comment, void (*init)(ncnn::Net&), void (*run)(const ncnn::Net&))
{
    ncnn::BenchNet net;

    init(net);

    net.load_model();

    // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
    Sleep(10 * 1000);
#else
    sleep(10);
#endif

    // warm up
    for (int i=0; i<30; i++)  run(net);
    std::vector<double> times(g_loop_count);
    for (int i=0; i<g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        run(net);

        double end = ncnn::get_current_time();

        double time = end - start;
        times[i] = time;
        usleep(20000);
    }

    std::cout << "Test for " << g_loop_count << " times, cost time list:" << std::endl;
    for(int i=0; i<g_loop_count; i++)
        std::cout << times[i] << " ";
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin()+5, times.end()-5, 0);  // 掐头去尾
    std::cout << std::endl;
    fprintf(stderr, "%16s  min = %7.2f  max = %7.2f  avg = %7.2f\n", comment, times[0], times[g_loop_count-1], sum / (g_loop_count-10));
}

void squeezenet_init(ncnn::Net& net)
{
    net.load_param("squeezenet.param");
}

void squeezenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_init(ncnn::Net& net)
{
    net.load_param("mobilenet.param");
}

void mobilenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void mobilenet_v2_init(ncnn::Net& net)
{
    net.load_param("mobilenet_v2.param");
}

void mobilenet_v2_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void shufflenet_init(ncnn::Net& net)
{
    net.load_param("shufflenet.param");
}

void shufflenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1000", out);
}

void googlenet_init(ncnn::Net& net)
{
    net.load_param("googlenet.param");
}

void googlenet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void resnet18_init(ncnn::Net& net)
{
    net.load_param("resnet18.param");
}

void resnet18_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void alexnet_init(ncnn::Net& net)
{
    net.load_param("alexnet.param");
}

void alexnet_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void vgg16_init(ncnn::Net& net)
{
    net.load_param("vgg16.param");
}

void vgg16_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(224, 224, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
}

void squeezenet_ssd_init(ncnn::Net& net)
{
    net.load_param("squeezenet_ssd.param");
}

void squeezenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

void mobilenet_ssd_init(ncnn::Net& net)
{
    net.load_param("mobilenet_ssd.param");
}

void mobilenet_ssd_run(const ncnn::Net& net)
{
    ncnn::Extractor ex = net.create_extractor();

    ncnn::Mat in(227, 227, 3);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
}

int main(int argc, char** argv)
{
    int loop_count = 80;
    int num_threads = ncnn::get_cpu_count();
    int powersave = 0;

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }

    g_loop_count = loop_count;

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());

    // run
    // benchmark("squeezenet", squeezenet_init, squeezenet_run);
    benchmark("mobilenet", mobilenet_init, mobilenet_run);

    // benchmark("mobilenet_v2", mobilenet_v2_init, mobilenet_v2_run);

    // benchmark("shufflenet", shufflenet_init, shufflenet_run);

    // benchmark("googlenet", googlenet_init, googlenet_run);

    // benchmark("resnet18", resnet18_init, resnet18_run);

    // benchmark("alexnet", alexnet_init, alexnet_run);

    // benchmark("vgg16", vgg16_init, vgg16_run);

    // benchmark("squeezenet-ssd", squeezenet_ssd_init, squeezenet_ssd_run);

    // benchmark("mobilenet-ssd", mobilenet_ssd_init, mobilenet_ssd_run);

    return 0;
}
