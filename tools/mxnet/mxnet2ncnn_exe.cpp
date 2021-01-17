#include "mxnet2ncnn.h"

int main(int argc, char** argv)
{
    const char* jsonpath = argv[1];
    const char* parampath = argv[2];
    const char* ncnn_prototxt = argc >= 5 ? argv[3] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 5 ? argv[4] : "ncnn.bin";

    FILE* mxnet_json = fopen(jsonpath, "rb");
    if (!mxnet_json)
    {
        fprintf(stderr, "fopen %s failed\n", jsonpath);
    }

    FILE* mxnet_param = fopen(parampath, "rb");
    if (!mxnet_param)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    convert(mxnet_json, mxnet_param, pp, bp);

    fclose(pp);
    fclose(bp);

    return 0;
}
