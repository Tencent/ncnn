#include <fstream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "onnx2ncnn.h"

static bool read_proto_from_binary(const char* filepath, onnx::ModelProto* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int main(int argc, char** argv)
{
    const char* onnxpb = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    onnx::ModelProto model;

    // load
    bool s1 = read_proto_from_binary(onnxpb, &model);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    convert(model, pp, bp);

    fclose(pp);
    fclose(bp);

    return 0;
}
