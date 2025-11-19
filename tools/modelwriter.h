// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdint.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "layer/batchnorm.h"
#include "layer/bias.h"
#include "layer/binaryop.h"
#include "layer/celu.h"
#include "layer/clip.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/convolution1d.h"
#include "layer/convolution3d.h"
#include "layer/convolutiondepthwise.h"
#include "layer/convolutiondepthwise1d.h"
#include "layer/convolutiondepthwise3d.h"
#include "layer/copyto.h"
#include "layer/crop.h"
#include "layer/cumulativesum.h"
#include "layer/deconvolution.h"
#include "layer/deconvolution1d.h"
#include "layer/deconvolution3d.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/deconvolutiondepthwise1d.h"
#include "layer/deconvolutiondepthwise3d.h"
#include "layer/deformableconv2d.h"
#include "layer/detectionoutput.h"
#include "layer/diag.h"
#include "layer/dropout.h"
#include "layer/eltwise.h"
#include "layer/elu.h"
#include "layer/embed.h"
#include "layer/exp.h"
#include "layer/expanddims.h"
#include "layer/flatten.h"
#include "layer/fold.h"
#include "layer/gelu.h"
#include "layer/gemm.h"
#include "layer/glu.h"
#include "layer/gridsample.h"
#include "layer/groupnorm.h"
#include "layer/gru.h"
#include "layer/hardsigmoid.h"
#include "layer/hardswish.h"
#include "layer/innerproduct.h"
#include "layer/input.h"
#include "layer/instancenorm.h"
#include "layer/interp.h"
#include "layer/layernorm.h"
#include "layer/log.h"
#include "layer/lrn.h"
#include "layer/lstm.h"
#include "layer/matmul.h"
#include "layer/memorydata.h"
#include "layer/mvn.h"
#include "layer/multiheadattention.h"
#include "layer/normalize.h"
#include "layer/padding.h"
#include "layer/permute.h"
#include "layer/pixelshuffle.h"
#include "layer/pooling.h"
#include "layer/pooling1d.h"
#include "layer/pooling3d.h"
#include "layer/power.h"
#include "layer/prelu.h"
#include "layer/priorbox.h"
#include "layer/proposal.h"
#include "layer/psroipooling.h"
#include "layer/quantize.h"
#include "layer/reduction.h"
#include "layer/relu.h"
#include "layer/reorg.h"
#include "layer/requantize.h"
#include "layer/reshape.h"
#include "layer/rmsnorm.h"
#include "layer/rnn.h"
#include "layer/roialign.h"
#include "layer/roipooling.h"
#include "layer/rotaryembed.h"
#include "layer/scale.h"
#include "layer/sdpa.h"
#include "layer/shufflechannel.h"
#include "layer/slice.h"
#include "layer/softmax.h"
#include "layer/split.h"
#include "layer/squeeze.h"
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/unfold.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"

// for gen_random_weight
#include "../tests/prng.h"

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

class MemoryFootprintAllocator : public ncnn::Allocator
{
public:
    MemoryFootprintAllocator()
    {
        current_memory_usage = 0;
        memory_footprint = 0;
    }

    virtual void* fastMalloc(size_t size)
    {
        ncnn::MutexLockGuard g(lock);
        void* ptr = ncnn::fastMalloc(size);
        bookkeeper[ptr] = size;
        current_memory_usage += size;
        memory_footprint = std::max(memory_footprint, current_memory_usage);
        return ptr;
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::MutexLockGuard g(lock);
        size_t size = bookkeeper[ptr];
        current_memory_usage -= size;
        bookkeeper.erase(bookkeeper.find(ptr));
        ncnn::fastFree(ptr);
    }

public:
    int current_memory_usage;
    int memory_footprint;
    ncnn::Mutex lock;
    std::map<void*, size_t> bookkeeper;
};

class CustomLayer : public ncnn::Layer
{
public:
    virtual int load_param(const ncnn::ParamDict& pd)
    {
        mpd = pd;
        return 0;
    }

    void write_param(FILE* pp)
    {
        for (int i = 0; i < NCNN_MAX_PARAM_COUNT; i++)
        {
            int type = mpd.type(i);
            if (type == 0)
                continue;

            if (type == 2)
            {
                fprintf(pp, " %d=%d", i, mpd.get(i, 0));
            }
            if (type == 3)
            {
                fprintf(pp, " %d=%e", i, mpd.get(i, 0.f));
            }
            if (type == 5)
            {
                ncnn::Mat v = mpd.get(i, ncnn::Mat());
                int len = v.w;
                fprintf(pp, " %d=%d", -i - 23300, len);
                const int* p = v;
                for (int j = 0; j < len; j++)
                {
                    fprintf(pp, ",%d", p[j]);
                }
            }
            if (type == 6)
            {
                ncnn::Mat v = mpd.get(i, ncnn::Mat());
                int len = v.w;
                fprintf(pp, " %d=%d", -i - 23300, len);
                const float* p = v;
                for (int j = 0; j < len; j++)
                {
                    fprintf(pp, ",%e", p[j]);
                }
            }
        }
    }

public:
    ncnn::ParamDict mpd;
};

DEFINE_LAYER_CREATOR(CustomLayer)

class ModelWriter : public ncnn::Net
{
public:
    ModelWriter();

    virtual ncnn::Layer* create_custom_layer(const char* type);

    std::vector<ncnn::Blob>& blobs;
    std::vector<ncnn::Layer*>& layers;

    bool has_custom_layer;

public:
    // 0=fp32 1=fp16
    int storage_type;

    int gen_random_weight;

    // Cut param and bin -1=no cut
    int cutstart;
    int cutend;

public:
    int set_cutparam(const char* cutstartname, const char* cutendname);

    int shape_inference();
    int estimate_memory_footprint();

public:
    int fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp);
    int fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp);

    int fwrite_weight_tag_data(const ncnn::Mat& data, FILE* bp, float a = -1.2f, float b = 1.2f);
    int fwrite_weight_data(const ncnn::Mat& data, FILE* bp, float a = -1.2f, float b = 1.2f);

    int save(const char* parampath, const char* binpath);
};

ModelWriter::ModelWriter()
    : blobs(mutable_blobs()), layers(mutable_layers())
{
    opt.lightmode = false;
    has_custom_layer = false;
    gen_random_weight = false;
    cutstart = -1;
    cutend = -1;

    SRAND(7767517);
}

ncnn::Layer* ModelWriter::create_custom_layer(const char* type)
{
    ncnn::Layer* layer = Net::create_custom_layer(type);
    if (layer)
        return layer;

    fprintf(stderr, "create_custom_layer %s\n", type);

    register_custom_layer(type, CustomLayer_layer_creator);

    has_custom_layer = true;

    return Net::create_custom_layer(type);
}

int ModelWriter::set_cutparam(const char* cutstartname, const char* cutendname)
{
    if (cutstartname != nullptr)
    {
        int layindex = find_layer_index_by_name(cutstartname);
        if (layindex >= 0)
        {
            cutstart = layindex;
            fprintf(stderr, "cutstart layer %d:%s\n", layindex, cutstartname);
        }
        else
        {
            fprintf(stderr, "not find target cutstart layer %s\n", cutstartname);
            return -1;
        }
    }

    if (cutendname != nullptr)
    {
        int layindex = find_layer_index_by_name(cutendname);
        if (layindex >= 0)
        {
            cutend = layindex;
            fprintf(stderr, "cutend layer %d:%s\n", layindex, cutendname);
        }
        else
        {
            fprintf(stderr, "not find target cutend layer %s\n", cutendname);
            return -1;
        }
    }

    return 0;
}

int ModelWriter::shape_inference()
{
    if (has_custom_layer)
    {
        fprintf(stderr, "model has custom layer, shape_inference skipped\n");
        return -1;
    }

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    // recreate layer pipeline for param and weight changes
    for (size_t i = 0; i < layer_count; i++)
    {
        ncnn::Layer* layer = layers[i];

        layer->destroy_pipeline(opt);

        int cret = layer->create_pipeline(opt);
        if (cret != 0)
        {
            NCNN_LOGE("layer create_pipeline %d %s failed", (int)i, layer->name.c_str());
            return -1;
        }
    }

    ncnn::Extractor ex = create_extractor();
    ex.set_light_mode(true);

    // prepare Input blobs
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        if (layer->type != "Input")
            continue;

        ncnn::Input* input = (ncnn::Input*)layer;

        int w = input->w;
        int h = input->h;
        int c = input->c;

        int dims = 0;
        if (w == 0 && h == 0 && c == 0) dims = 0;
        if (w != 0 && h == 0 && c == 0) dims = 1;
        if (w != 0 && h != 0 && c == 0) dims = 2;
        if (w != 0 && h != 0 && c != 0) dims = 3;

        if (dims == 0)
        {
            fprintf(stderr, "Input layer %s without shape info, shape_inference skipped\n", layer->name.c_str());
            return -1;
        }

        ncnn::Mat m;
        if (dims == 1) m.create(w);
        if (dims == 2) m.create(w, h);
        if (dims == 3) m.create(w, h, c);

        ex.input(layer->tops[0], m);
    }

    // prepare blobs with predefined shape
    for (size_t i = 0; i < blob_count; i++)
    {
        const ncnn::Blob& blob = blobs[i];

        int dims = blob.shape.dims;
        int w = blob.shape.w;
        int h = blob.shape.h;
        int c = blob.shape.c;

        if (dims == 0)
            continue;

        ncnn::Mat m;
        if (dims == 1) m.create(w);
        if (dims == 2) m.create(w, h);
        if (dims == 3) m.create(w, h, c);

        m.fill(0.f);

        ex.input(int(i), m);
    }

    fprintf(stderr, "shape_inference\n");

    // resolve all layer output blob shape
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            ncnn::Mat m;
            ex.extract(top_blob_index, m);

            blobs[top_blob_index].shape = m;
        }
    }

    // assign all layer blob shape
    for (size_t i = 0; i < layer_count; i++)
    {
        ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer->bottom_shapes.resize(layer->bottoms.size());
        for (size_t j = 0; j < layer->bottoms.size(); j++)
        {
            int bottom_blob_index = layer->bottoms[j];

            layer->bottom_shapes[j] = blobs[bottom_blob_index].shape;
        }

        layer->top_shapes.resize(layer->tops.size());
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            layer->top_shapes[j] = blobs[top_blob_index].shape;

            //             fprintf(stderr, "%d %4d %4d %4d | %2d %s\n", blobs[top_blob_index].shape.dims, blobs[top_blob_index].shape.w, blobs[top_blob_index].shape.h, blobs[top_blob_index].shape.c, top_blob_index, blobs[top_blob_index].name.c_str());
        }
    }

    return 0;
}

int ModelWriter::estimate_memory_footprint()
{
    if (has_custom_layer)
    {
        fprintf(stderr, "model has custom layer, estimate_memory_footprint skipped\n");
        return -1;
    }

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    MemoryFootprintAllocator allocator;

    ncnn::Extractor ex = create_extractor();
    ex.set_light_mode(true);

    ex.set_blob_allocator(&allocator);
    ex.set_workspace_allocator(&allocator);

    // prepare Input blobs
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        if (layer->type != "Input")
            continue;

        ncnn::Input* input = (ncnn::Input*)layer;

        int w = input->w;
        int h = input->h;
        int c = input->c;

        int dims = 0;
        if (w == 0 && h == 0 && c == 0) dims = 0;
        if (w != 0 && h == 0 && c == 0) dims = 1;
        if (w != 0 && h != 0 && c == 0) dims = 2;
        if (w != 0 && h != 0 && c != 0) dims = 3;

        if (dims == 0)
        {
            fprintf(stderr, "Input layer %s without shape info, estimate_memory_footprint skipped\n", layer->name.c_str());
            return -1;
        }

        ncnn::Mat m;
        if (dims == 1) m.create(w, 4u, &allocator);
        if (dims == 2) m.create(w, h, 4u, &allocator);
        if (dims == 3) m.create(w, h, c, 4u, &allocator);

        ex.input(layer->tops[0], m);

        fprintf(stderr, "input = %s\n", blobs[layer->tops[0]].name.c_str());
    }

    // find output blobs and do inference
    std::vector<ncnn::Mat> outputs;
    for (size_t i = 0; i < blob_count; i++)
    {
        const ncnn::Blob& blob = blobs[i];

        if (blob.producer == -1 || blob.consumer != -1)
            continue;

        if (layers[blob.producer]->type == "ncnnfused")
            continue;

        // treat blob without any consumers as output
        ncnn::Mat m;
        ex.extract(int(i), m);
        outputs.push_back(m);

        fprintf(stderr, "extract = %s\n", blob.name.c_str());
    }

    fprintf(stderr, "estimated memory footprint = %.2f KB = %.2f MB\n", allocator.memory_footprint / 1024.f, allocator.memory_footprint / 1024.f / 1024.f);

    return 0;
}

int ModelWriter::fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp)
{
    const int count = m.w;
    const int* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%d", ptr[i]);
    }

    return 0;
}

int ModelWriter::fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp)
{
    const int count = m.w;
    const float* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%e", ptr[i]);
    }

    return 0;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static void replace_denormals_with_zero(float* data, size_t data_length)
{
    const int total = static_cast<int>(data_length);
    for (size_t i = 0; i < data_length; ++i)
    {
        float value = data[i];

        if (fabsf(value) < 1e-30 && fabsf(value) != 0.f)
        {
            data[i] = 0.f;
        }
    }
}

static float RandomFloat(float a = -1.2f, float b = 1.2f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

static void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f)
{
    if (m.elemsize == 4)
    {
        for (size_t i = 0; i < m.total(); i++)
        {
            m[i] = RandomFloat(a, b);
        }
    }
    else if (m.elemsize == 2)
    {
        unsigned short* p = m;
        for (size_t i = 0; i < m.total(); i++)
        {
            p[i] = ncnn::float32_to_float16(RandomFloat(a, b));
        }
    }
    else if (m.elemsize == 1)
    {
        signed char* p = m;
        for (size_t i = 0; i < m.total(); i++)
        {
            p[i] = (signed char)RandomFloat(-127, 127);
        }
    }
}

int ModelWriter::fwrite_weight_tag_data(const ncnn::Mat& data, FILE* bp, float a, float b)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.d * data.c);
    if (gen_random_weight)
        Randomize(data_flattened, a, b);

    if (data_flattened.elemsize == 4)
    {
        if (storage_type == 1)
        {
            const int tag = 0x01306B47; // fp16 magic
            fwrite(&tag, sizeof(int), 1, bp);
            ncnn::Mat data_flattened_fp16;
            ncnn::cast_float32_to_float16(data_flattened, data_flattened_fp16);
            fwrite(data_flattened_fp16.data, data_flattened_fp16.elemsize, data_flattened_fp16.w, bp);
        }
        else
        {
            const int tag = 0; // fp32 magic
            fwrite(&tag, sizeof(int), 1, bp);
            replace_denormals_with_zero(data_flattened, data_flattened.w);
            fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);
        }
    }
    else if (data_flattened.elemsize == 2)
    {
        const int tag = 0x01306B47; // fp16 magic
        fwrite(&tag, sizeof(int), 1, bp);
        fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);
    }
    else if (data_flattened.elemsize == 1)
    {
        const int tag = 0x000D4B38; // int8 magic
        fwrite(&tag, sizeof(int), 1, bp);
        fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);
    }
    else
    {
        fprintf(stderr, "unknown weight data type %d\n", (int)data_flattened.elemsize);
    }

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    size_t nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int ModelWriter::fwrite_weight_data(const ncnn::Mat& data, FILE* bp, float a, float b)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.d * data.c);
    if (gen_random_weight)
        Randomize(data_flattened, a, b);

    if (data_flattened.elemsize == 4) // fp32
    {
        replace_denormals_with_zero(data_flattened, data_flattened.w);
    }

    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    size_t nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int ModelWriter::save(const char* parampath, const char* binpath)
{
    uint64_t mac = 0;

    FILE* pp = fopen(parampath, "wb");
    FILE* bp = fopen(binpath, "wb");

    fprintf(pp, "7767517\n");

    const size_t layer_count = layers.size();

    int layer_count_fused = 0;
    std::set<std::string> blob_names;
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer_count_fused++;

        size_t bottom_count = layer->bottoms.size();
        for (size_t j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            blob_names.insert(blobs[bottom_blob_index].name);
        }

        size_t top_count = layer->tops.size();
        for (size_t j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            blob_names.insert(blobs[top_blob_index].name);
        }
    }

    size_t blob_count_fused = blob_names.size();

    fprintf(pp, "%d %zd\n", layer_count_fused, blob_count_fused);

    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        if (cutstart > 0 && i < cutstart)
            continue;

        if (cutend > 0 && i > cutend)
            continue;

        size_t bottom_count = layer->bottoms.size();
        size_t top_count = layer->tops.size();

        fprintf(pp, "%-24s %-24s %zd %zd", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

        for (size_t j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            fprintf(pp, " %s", blobs[bottom_blob_index].name.c_str());
        }
        for (size_t j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            fprintf(pp, " %s", blobs[top_blob_index].name.c_str());
        }

        // write shape hints
        bool shape_ready = true;
        for (size_t j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];

            int dims = blobs[top_blob_index].shape.dims;
            if (dims == 0)
            {
                shape_ready = false;
                break;
            }
        }
        if (shape_ready)
        {
            fprintf(pp, " -23330=%zd", top_count * 4);
            for (size_t j = 0; j < top_count; j++)
            {
                int top_blob_index = layer->tops[j];

                int dims = blobs[top_blob_index].shape.dims;
                int w = blobs[top_blob_index].shape.w;
                int h = blobs[top_blob_index].shape.h;
                int c = blobs[top_blob_index].shape.c;

                fprintf(pp, ",%d,%d,%d,%d", dims, w, h, c);
            }
        }

        // custom op
        if (layer->typeindex & ncnn::LayerType::CustomBit)
        {
            ((CustomLayer*)layer)->write_param(pp);

            fprintf(pp, "\n");

            continue;
        }

        ncnn::Layer* layer_default = ncnn::create_layer_cpu(layer->typeindex);

        ncnn::ParamDict pd;
        layer_default->load_param(pd);

#define fprintf_param_value(format, phase)                                  \
    {                                                                       \
        if (op->phase != op_default->phase) fprintf(pp, format, op->phase); \
    }

        if (layer->type == "BatchNorm")
        {
            ncnn::BatchNorm* op = (ncnn::BatchNorm*)layer;
            ncnn::BatchNorm* op_default = (ncnn::BatchNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%e", eps)

            fwrite_weight_data(op->slope_data, bp);
            fwrite_weight_data(op->mean_data, bp);
            fwrite_weight_data(op->var_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "Bias")
        {
            ncnn::Bias* op = (ncnn::Bias*)layer;
            ncnn::Bias* op_default = (ncnn::Bias*)layer_default;

            fprintf_param_value(" 0=%d", bias_data_size)

            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "BinaryOp")
        {
            ncnn::BinaryOp* op = (ncnn::BinaryOp*)layer;
            ncnn::BinaryOp* op_default = (ncnn::BinaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            fprintf_param_value(" 1=%d", with_scalar)
            fprintf_param_value(" 2=%e", b)
        }
        else if (layer->type == "CELU")
        {
            ncnn::CELU* op = (ncnn::CELU*)layer;
            ncnn::CELU* op_default = (ncnn::CELU*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
        }
        else if (layer->type == "Clip")
        {
            ncnn::Clip* op = (ncnn::Clip*)layer;
            ncnn::Clip* op_default = (ncnn::Clip*)layer_default;

            fprintf_param_value(" 0=%e", min)
            fprintf_param_value(" 1=%e", max)
        }
        else if (layer->type == "Concat")
        {
            ncnn::Concat* op = (ncnn::Concat*)layer;
            ncnn::Concat* op_default = (ncnn::Concat*)layer_default;

            fprintf_param_value(" 0=%d", axis)
        }
        else if (layer->type == "Convolution")
        {
            ncnn::Convolution* op = (ncnn::Convolution*)layer;
            ncnn::Convolution* op_default = (ncnn::Convolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 19=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);

#if NCNN_INT8
                // write int8_scale data
                if (op->int8_scale_term)
                {
                    fwrite_weight_data(op->weight_data_int8_scales, bp, 90, 100);
                    fwrite_weight_data(op->bottom_blob_int8_scales, bp, 0.001, 1);
                    fwrite_weight_data(op->top_blob_int8_scales, bp, 0.001, 1);
                }
#endif // NCNN_INT8
            }

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_h * op->kernel_w * outw * outh * outc * inc;
            }
        }
        else if (layer->type == "Convolution1D")
        {
            ncnn::Convolution1D* op = (ncnn::Convolution1D*)layer;
            ncnn::Convolution1D* op_default = (ncnn::Convolution1D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            fprintf_param_value(" 2=%d", dilation_w)
            fprintf_param_value(" 3=%d", stride_w)
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 19=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);
            }

            if (shape_ready)
            {
                int inh = blobs[layer->bottoms[0]].shape.h;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;

                mac += (uint64_t)op->kernel_w * outw * outh * inh;
            }
        }
        else if (layer->type == "Convolution3D")
        {
            ncnn::Convolution3D* op = (ncnn::Convolution3D*)layer;
            ncnn::Convolution3D* op_default = (ncnn::Convolution3D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
                if (op->kernel_d != op->kernel_w) fprintf(pp, " 21=%d", op->kernel_d);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
                if (op->dilation_d != op->dilation_w) fprintf(pp, " 22=%d", op->dilation_d);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
                if (op->stride_d != op->stride_w) fprintf(pp, " 23=%d", op->stride_d);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
                if (op->pad_front != op->pad_left) fprintf(pp, " 24=%d", op->pad_front);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            {
                if (op->pad_behind != op->pad_front) fprintf(pp, " 17=%d", op->pad_behind);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outd = blobs[layer->tops[0]].shape.d;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_d * op->kernel_h * op->kernel_w * outw * outh * outd * outc * inc;
            }
        }
        else if (layer->type == "ConvolutionDepthWise")
        {
            ncnn::ConvolutionDepthWise* op = (ncnn::ConvolutionDepthWise*)layer;
            ncnn::ConvolutionDepthWise* op_default = (ncnn::ConvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 19=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);

#if NCNN_INT8
                // write int8_scale data
                if (op->int8_scale_term == 1 || op->int8_scale_term == 101)
                {
                    op->bottom_blob_int8_scales.w = 1;
                }
                if (op->int8_scale_term == 2 || op->int8_scale_term == 102)
                {
                    op->weight_data_int8_scales.w = 1;
                    op->bottom_blob_int8_scales.w = 1;
                }
                if (op->int8_scale_term > 100)
                {
                    op->top_blob_int8_scales.w = 1;
                }

                if (op->int8_scale_term)
                {
                    fwrite_weight_data(op->weight_data_int8_scales, bp, 90, 100);
                    fwrite_weight_data(op->bottom_blob_int8_scales, bp, 0.001, 1);
                    fwrite_weight_data(op->top_blob_int8_scales, bp, 0.001, 1);
                }
#endif // NCNN_INT8
            }

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_h * op->kernel_w * outw * outh * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "ConvolutionDepthWise1D")
        {
            ncnn::ConvolutionDepthWise1D* op = (ncnn::ConvolutionDepthWise1D*)layer;
            ncnn::ConvolutionDepthWise1D* op_default = (ncnn::ConvolutionDepthWise1D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            fprintf_param_value(" 2=%d", dilation_w)
            fprintf_param_value(" 3=%d", stride_w)
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 19=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                if (op->bias_term)
                {
                    fwrite_weight_data(op->bias_data, bp);
                }
            }

            if (shape_ready)
            {
                int inh = blobs[layer->bottoms[0]].shape.h;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;

                mac += (uint64_t)op->kernel_w * outw * (outh / op->group) * (inh / op->group) * op->group;
            }
        }
        else if (layer->type == "ConvolutionDepthWise3D")
        {
            ncnn::ConvolutionDepthWise3D* op = (ncnn::ConvolutionDepthWise3D*)layer;
            ncnn::ConvolutionDepthWise3D* op_default = (ncnn::ConvolutionDepthWise3D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
                if (op->kernel_d != op->kernel_w) fprintf(pp, " 21=%d", op->kernel_d);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
                if (op->dilation_d != op->dilation_w) fprintf(pp, " 22=%d", op->dilation_d);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
                if (op->stride_d != op->stride_w) fprintf(pp, " 23=%d", op->stride_d);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
                if (op->pad_front != op->pad_left) fprintf(pp, " 24=%d", op->pad_front);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            {
                if (op->pad_behind != op->pad_front) fprintf(pp, " 17=%d", op->pad_behind);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outd = blobs[layer->tops[0]].shape.d;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_d * op->kernel_h * op->kernel_w * outw * outh * outd * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "CopyTo")
        {
            ncnn::CopyTo* op = (ncnn::CopyTo*)layer;
            ncnn::CopyTo* op_default = (ncnn::CopyTo*)layer_default;

            fprintf_param_value(" 0=%d", woffset)
            fprintf_param_value(" 1=%d", hoffset)
            fprintf_param_value(" 13=%d", doffset)
            fprintf_param_value(" 2=%d", coffset)
            {
                if (!op->starts.empty()) fprintf_param_int_array(9, op->starts, pp);
            }
            {
                if (!op->axes.empty()) fprintf_param_int_array(11, op->axes, pp);
            }
        }
        else if (layer->type == "Crop")
        {
            ncnn::Crop* op = (ncnn::Crop*)layer;
            ncnn::Crop* op_default = (ncnn::Crop*)layer_default;

            fprintf_param_value(" 0=%d", woffset)
            fprintf_param_value(" 1=%d", hoffset)
            fprintf_param_value(" 13=%d", doffset)
            fprintf_param_value(" 2=%d", coffset)
            fprintf_param_value(" 3=%d", outw)
            fprintf_param_value(" 4=%d", outh)
            fprintf_param_value(" 14=%d", outd)
            fprintf_param_value(" 5=%d", outc)
            fprintf_param_value(" 6=%d", woffset2)
            fprintf_param_value(" 7=%d", hoffset2)
            fprintf_param_value(" 15=%d", doffset2)
            fprintf_param_value(" 8=%d", coffset2)
            {
                if (!op->starts.empty()) fprintf_param_int_array(9, op->starts, pp);
            }
            {
                if (!op->ends.empty()) fprintf_param_int_array(10, op->ends, pp);
            }
            {
                if (!op->axes.empty()) fprintf_param_int_array(11, op->axes, pp);
            }
            {
                if (op->starts_expr != op_default->starts_expr) fprintf(pp, " 19=\"%s\"", op->starts_expr.c_str());
            }
            {
                if (op->ends_expr != op_default->ends_expr) fprintf(pp, " 20=\"%s\"", op->ends_expr.c_str());
            }
            {
                if (op->axes_expr != op_default->axes_expr) fprintf(pp, " 21=\"%s\"", op->axes_expr.c_str());
            }
        }
        else if (layer->type == "CumulativeSum")
        {
            ncnn::CumulativeSum* op = (ncnn::CumulativeSum*)layer;
            ncnn::CumulativeSum* op_default = (ncnn::CumulativeSum*)layer_default;

            fprintf_param_value(" 0=%d", axis)
        }
        else if (layer->type == "Deconvolution")
        {
            ncnn::Deconvolution* op = (ncnn::Deconvolution*)layer;
            ncnn::Deconvolution* op_default = (ncnn::Deconvolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
            }
            fprintf_param_value(" 20=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 21=%d", op->output_h);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 28=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);
            }

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_h * op->kernel_w * inw * inh * outc * inc;
            }
        }
        else if (layer->type == "Deconvolution1D")
        {
            ncnn::Deconvolution1D* op = (ncnn::Deconvolution1D*)layer;
            ncnn::Deconvolution1D* op_default = (ncnn::Deconvolution1D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            fprintf_param_value(" 2=%d", dilation_w)
            fprintf_param_value(" 3=%d", stride_w)
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            fprintf_param_value(" 20=%d", output_w)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 28=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);
            }

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int outh = blobs[layer->tops[0]].shape.h;

                mac += (uint64_t)op->kernel_w * inw * outh * inh;
            }
        }
        else if (layer->type == "Deconvolution3D")
        {
            ncnn::Deconvolution3D* op = (ncnn::Deconvolution3D*)layer;
            ncnn::Deconvolution3D* op_default = (ncnn::Deconvolution3D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
                if (op->kernel_d != op->kernel_w) fprintf(pp, " 21=%d", op->kernel_d);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
                if (op->dilation_d != op->dilation_w) fprintf(pp, " 22=%d", op->dilation_d);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
                if (op->stride_d != op->stride_w) fprintf(pp, " 23=%d", op->stride_d);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
                if (op->pad_front != op->pad_left) fprintf(pp, " 24=%d", op->pad_front);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            {
                if (op->pad_behind != op->pad_front) fprintf(pp, " 17=%d", op->pad_behind);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
                if (op->output_pad_behind != op->output_pad_right) fprintf(pp, " 20=%d", op->output_pad_behind);
            }
            fprintf_param_value(" 25=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 26=%d", op->output_h);
                if (op->output_d != op->output_w) fprintf(pp, " 27=%d", op->output_d);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int ind = blobs[layer->bottoms[0]].shape.d;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_d * op->kernel_h * op->kernel_w * inw * inh * ind * outc * inc;
            }
        }
        else if (layer->type == "DeconvolutionDepthWise")
        {
            ncnn::DeconvolutionDepthWise* op = (ncnn::DeconvolutionDepthWise*)layer;
            ncnn::DeconvolutionDepthWise* op_default = (ncnn::DeconvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
            }
            fprintf_param_value(" 20=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 21=%d", op->output_h);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 28=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);
            }

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_h * op->kernel_w * inw * inh * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "DeconvolutionDepthWise1D")
        {
            ncnn::DeconvolutionDepthWise1D* op = (ncnn::DeconvolutionDepthWise1D*)layer;
            ncnn::DeconvolutionDepthWise1D* op_default = (ncnn::DeconvolutionDepthWise1D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            fprintf_param_value(" 2=%d", dilation_w)
            fprintf_param_value(" 3=%d", stride_w)
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            fprintf_param_value(" 20=%d", output_w)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 28=%d", dynamic_weight)

            if (op->dynamic_weight == 0)
            {
                fwrite_weight_tag_data(op->weight_data, bp);
                fwrite_weight_data(op->bias_data, bp);
            }

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int outh = blobs[layer->tops[0]].shape.h;

                mac += (uint64_t)op->kernel_w * inw * (outh / op->group) * (inh / op->group) * op->group;
            }
        }
        else if (layer->type == "DeconvolutionDepthWise3D")
        {
            ncnn::DeconvolutionDepthWise3D* op = (ncnn::DeconvolutionDepthWise3D*)layer;
            ncnn::DeconvolutionDepthWise3D* op_default = (ncnn::DeconvolutionDepthWise3D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
                if (op->kernel_d != op->kernel_w) fprintf(pp, " 21=%d", op->kernel_d);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
                if (op->dilation_d != op->dilation_w) fprintf(pp, " 22=%d", op->dilation_d);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
                if (op->stride_d != op->stride_w) fprintf(pp, " 23=%d", op->stride_d);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
                if (op->pad_front != op->pad_left) fprintf(pp, " 24=%d", op->pad_front);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            {
                if (op->pad_behind != op->pad_front) fprintf(pp, " 17=%d", op->pad_behind);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
                if (op->output_pad_behind != op->output_pad_right) fprintf(pp, " 20=%d", op->output_pad_behind);
            }
            fprintf_param_value(" 25=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 26=%d", op->output_h);
                if (op->output_d != op->output_w) fprintf(pp, " 27=%d", op->output_d);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int ind = blobs[layer->bottoms[0]].shape.d;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_d * op->kernel_h * op->kernel_w * inw * inh * ind * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "DeformableConv2D")
        {
            ncnn::DeformableConv2D* op = (ncnn::DeformableConv2D*)layer;
            ncnn::DeformableConv2D* op_default = (ncnn::DeformableConv2D*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += (uint64_t)op->kernel_h * op->kernel_w * inw * inh * outc * inc;
            }
        }
        else if (layer->type == "DetectionOutput")
        {
            ncnn::DetectionOutput* op = (ncnn::DetectionOutput*)layer;
            ncnn::DetectionOutput* op_default = (ncnn::DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%e", nms_threshold)
            fprintf_param_value(" 2=%d", nms_top_k)
            fprintf_param_value(" 3=%d", keep_top_k)
            fprintf_param_value(" 4=%e", confidence_threshold)
            fprintf_param_value(" 5=%e", variances[0])
            fprintf_param_value(" 6=%e", variances[1])
            fprintf_param_value(" 7=%e", variances[2])
            fprintf_param_value(" 8=%e", variances[3])
        }
        else if (layer->type == "Diag")
        {
            ncnn::Diag* op = (ncnn::Diag*)layer;
            ncnn::Diag* op_default = (ncnn::Diag*)layer_default;

            fprintf_param_value(" 0=%d", diagonal)
        }
        else if (layer->type == "Dropout")
        {
            ncnn::Dropout* op = (ncnn::Dropout*)layer;
            ncnn::Dropout* op_default = (ncnn::Dropout*)layer_default;

            fprintf_param_value(" 0=%e", scale)
        }
        else if (layer->type == "Eltwise")
        {
            ncnn::Eltwise* op = (ncnn::Eltwise*)layer;
            ncnn::Eltwise* op_default = (ncnn::Eltwise*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            {
                if (!op->coeffs.empty()) fprintf_param_float_array(1, op->coeffs, pp);
            }
        }
        else if (layer->type == "ELU")
        {
            ncnn::ELU* op = (ncnn::ELU*)layer;
            ncnn::ELU* op_default = (ncnn::ELU*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
        }
        else if (layer->type == "Embed")
        {
            ncnn::Embed* op = (ncnn::Embed*)layer;
            ncnn::Embed* op_default = (ncnn::Embed*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", input_dim)
            fprintf_param_value(" 2=%d", bias_term)
            fprintf_param_value(" 3=%d", weight_data_size)
            fprintf_param_value(" 18=%d", int8_scale_term)

            fwrite_weight_tag_data(op->weight_data, bp);
            if (op->bias_term)
            {
                fwrite_weight_data(op->bias_data, bp);
            }

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                ncnn::Mat weight_data_int8_scales(1);
                weight_data_int8_scales[0] = op->weight_data_int8_scale;
                fwrite_weight_data(weight_data_int8_scales, bp, 90, 100);
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "Exp")
        {
            ncnn::Exp* op = (ncnn::Exp*)layer;
            ncnn::Exp* op_default = (ncnn::Exp*)layer_default;

            fprintf_param_value(" 0=%e", base)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "ExpandDims")
        {
            ncnn::ExpandDims* op = (ncnn::ExpandDims*)layer;
            ncnn::ExpandDims* op_default = (ncnn::ExpandDims*)layer_default;

            {
                if (!op->axes.empty()) fprintf_param_int_array(3, op->axes, pp);
            }
        }
        else if (layer->type == "Fold")
        {
            ncnn::Fold* op = (ncnn::Fold*)layer;
            ncnn::Fold* op_default = (ncnn::Fold*)layer_default;

            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 20=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 21=%d", op->output_h);
            }
        }
        else if (layer->type == "GELU")
        {
            ncnn::GELU* op = (ncnn::GELU*)layer;
            ncnn::GELU* op_default = (ncnn::GELU*)layer_default;

            fprintf_param_value(" 0=%d", fast_gelu)
        }
        else if (layer->type == "Gemm")
        {
            ncnn::Gemm* op = (ncnn::Gemm*)layer;
            ncnn::Gemm* op_default = (ncnn::Gemm*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
            fprintf_param_value(" 2=%d", transA)
            fprintf_param_value(" 3=%d", transB)
            fprintf_param_value(" 4=%d", constantA)
            fprintf_param_value(" 5=%d", constantB)
            fprintf_param_value(" 6=%d", constantC)
            fprintf_param_value(" 7=%d", constantM)
            fprintf_param_value(" 8=%d", constantN)
            fprintf_param_value(" 9=%d", constantK)
            fprintf_param_value(" 10=%d", constant_broadcast_type_C)
            fprintf_param_value(" 11=%d", output_N1M)
            fprintf_param_value(" 12=%d", output_elempack)
            fprintf_param_value(" 13=%d", output_elemtype)
            fprintf_param_value(" 14=%d", output_transpose)
            fprintf_param_value(" 18=%d", int8_scale_term)
            fprintf_param_value(" 20=%d", constant_TILE_M)
            fprintf_param_value(" 21=%d", constant_TILE_N)
            fprintf_param_value(" 22=%d", constant_TILE_K)

            if (op->constantA == 1)
            {
                fwrite_weight_tag_data(op->A_data, bp);
            }
            if (op->constantB == 1)
            {
                fwrite_weight_tag_data(op->B_data, bp);
            }
            if (op->constantC == 1 && op->constant_broadcast_type_C != -1)
            {
                fwrite_weight_tag_data(op->C_data, bp);
            }

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                if (op->constantA == 1)
                {
                    fwrite_weight_data(op->A_data_int8_scales, bp, 90, 100);
                }
                if (op->constantB == 1)
                {
                    ncnn::Mat B_data_int8_scales(1);
                    B_data_int8_scales[0] = op->B_data_int8_scale;
                    fwrite_weight_data(B_data_int8_scales, bp, 90, 100);
                }
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "GLU")
        {
            ncnn::GLU* op = (ncnn::GLU*)layer;
            ncnn::GLU* op_default = (ncnn::GLU*)layer_default;

            fprintf_param_value(" 0=%d", axis)
        }
        else if (layer->type == "GridSample")
        {
            ncnn::GridSample* op = (ncnn::GridSample*)layer;
            ncnn::GridSample* op_default = (ncnn::GridSample*)layer_default;

            fprintf_param_value(" 0=%d", sample_type)
            fprintf_param_value(" 1=%d", padding_mode)
            fprintf_param_value(" 2=%d", align_corner)
            fprintf_param_value(" 3=%d", permute_fusion)
        }
        else if (layer->type == "GroupNorm")
        {
            ncnn::GroupNorm* op = (ncnn::GroupNorm*)layer;
            ncnn::GroupNorm* op_default = (ncnn::GroupNorm*)layer_default;

            fprintf_param_value(" 0=%d", group)
            fprintf_param_value(" 1=%d", channels)
            fprintf_param_value(" 2=%e", eps)
            fprintf_param_value(" 3=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
            fwrite_weight_data(op->beta_data, bp);
        }
        else if (layer->type == "GRU")
        {
            ncnn::GRU* op = (ncnn::GRU*)layer;
            ncnn::GRU* op_default = (ncnn::GRU*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", weight_data_size)
            fprintf_param_value(" 2=%d", direction)
            fprintf_param_value(" 8=%d", int8_scale_term)

            fwrite_weight_tag_data(op->weight_xc_data, bp);
            fwrite_weight_tag_data(op->bias_c_data, bp);
            fwrite_weight_tag_data(op->weight_hc_data, bp);

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                fwrite_weight_data(op->weight_xc_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->weight_hc_data_int8_scales, bp, 90, 100);
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "HardSigmoid")
        {
            ncnn::HardSigmoid* op = (ncnn::HardSigmoid*)layer;
            ncnn::HardSigmoid* op_default = (ncnn::HardSigmoid*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
        }
        else if (layer->type == "HardSwish")
        {
            ncnn::HardSwish* op = (ncnn::HardSwish*)layer;
            ncnn::HardSwish* op_default = (ncnn::HardSwish*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
        }
        else if (layer->type == "InnerProduct")
        {
            ncnn::InnerProduct* op = (ncnn::InnerProduct*)layer;
            ncnn::InnerProduct* op_default = (ncnn::InnerProduct*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", bias_term)
            fprintf_param_value(" 2=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                fwrite_weight_data(op->weight_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->bottom_blob_int8_scales, bp, 0.001, 1);
            }
#endif // NCNN_INT8

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;

                mac += (uint64_t)inw * inh * inc * outw;
            }
        }
        else if (layer->type == "Input")
        {
            ncnn::Input* op = (ncnn::Input*)layer;
            ncnn::Input* op_default = (ncnn::Input*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
        }
        else if (layer->type == "InstanceNorm")
        {
            ncnn::InstanceNorm* op = (ncnn::InstanceNorm*)layer;
            ncnn::InstanceNorm* op_default = (ncnn::InstanceNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%e", eps)
            fprintf_param_value(" 2=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
            fwrite_weight_data(op->beta_data, bp);
        }
        else if (layer->type == "Interp")
        {
            ncnn::Interp* op = (ncnn::Interp*)layer;
            ncnn::Interp* op_default = (ncnn::Interp*)layer_default;

            fprintf_param_value(" 0=%d", resize_type)
            fprintf_param_value(" 1=%e", height_scale)
            fprintf_param_value(" 2=%e", width_scale)
            fprintf_param_value(" 3=%d", output_height)
            fprintf_param_value(" 4=%d", output_width)
            fprintf_param_value(" 5=%d", dynamic_target_size)
            fprintf_param_value(" 6=%d", align_corner)
            {
                if (op->size_expr != op_default->size_expr) fprintf(pp, " 9=\"%s\"", op->size_expr.c_str());
            }
        }
        else if (layer->type == "LayerNorm")
        {
            ncnn::LayerNorm* op = (ncnn::LayerNorm*)layer;
            ncnn::LayerNorm* op_default = (ncnn::LayerNorm*)layer_default;

            fprintf_param_value(" 0=%d", affine_size)
            fprintf_param_value(" 1=%e", eps)
            fprintf_param_value(" 2=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
            fwrite_weight_data(op->beta_data, bp);
        }
        else if (layer->type == "Log")
        {
            ncnn::Log* op = (ncnn::Log*)layer;
            ncnn::Log* op_default = (ncnn::Log*)layer_default;

            fprintf_param_value(" 0=%e", base)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "LRN")
        {
            ncnn::LRN* op = (ncnn::LRN*)layer;
            ncnn::LRN* op_default = (ncnn::LRN*)layer_default;

            fprintf_param_value(" 0=%d", region_type)
            fprintf_param_value(" 1=%d", local_size)
            fprintf_param_value(" 2=%e", alpha)
            fprintf_param_value(" 3=%e", beta)
            fprintf_param_value(" 4=%e", bias)
        }
        else if (layer->type == "LSTM")
        {
            ncnn::LSTM* op = (ncnn::LSTM*)layer;
            ncnn::LSTM* op_default = (ncnn::LSTM*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", weight_data_size)
            fprintf_param_value(" 2=%d", direction)
            fprintf_param_value(" 3=%d", hidden_size)
            fprintf_param_value(" 8=%d", int8_scale_term)

            fwrite_weight_tag_data(op->weight_xc_data, bp);
            fwrite_weight_tag_data(op->bias_c_data, bp);
            fwrite_weight_tag_data(op->weight_hc_data, bp);

            if (op->num_output != op->hidden_size)
            {
                fwrite_weight_tag_data(op->weight_hr_data, bp);
            }

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                fwrite_weight_data(op->weight_xc_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->weight_hc_data_int8_scales, bp, 90, 100);
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "MatMul")
        {
            ncnn::MatMul* op = (ncnn::MatMul*)layer;
            ncnn::MatMul* op_default = (ncnn::MatMul*)layer_default;

            fprintf_param_value(" 0=%d", transB)
        }
        else if (layer->type == "MemoryData")
        {
            ncnn::MemoryData* op = (ncnn::MemoryData*)layer;
            ncnn::MemoryData* op_default = (ncnn::MemoryData*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
            fprintf_param_value(" 11=%d", d)
            fwrite_weight_data(op->data, bp);
        }
        else if (layer->type == "MultiHeadAttention")
        {
            ncnn::MultiHeadAttention* op = (ncnn::MultiHeadAttention*)layer;
            ncnn::MultiHeadAttention* op_default = (ncnn::MultiHeadAttention*)layer_default;

            fprintf_param_value(" 0=%d", embed_dim)
            fprintf_param_value(" 1=%d", num_heads)
            fprintf_param_value(" 2=%d", weight_data_size)
            fprintf_param_value(" 3=%d", kdim)
            fprintf_param_value(" 4=%d", vdim)
            fprintf_param_value(" 5=%d", attn_mask)
            fprintf_param_value(" 6=%e", scale)
            fprintf_param_value(" 18=%d", int8_scale_term)

            fwrite_weight_tag_data(op->q_weight_data, bp);
            fwrite_weight_data(op->q_bias_data, bp);
            fwrite_weight_tag_data(op->k_weight_data, bp);
            fwrite_weight_data(op->k_bias_data, bp);
            fwrite_weight_tag_data(op->v_weight_data, bp);
            fwrite_weight_data(op->v_bias_data, bp);
            fwrite_weight_tag_data(op->out_weight_data, bp);
            fwrite_weight_data(op->out_bias_data, bp);

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                fwrite_weight_data(op->q_weight_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->k_weight_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->v_weight_data_int8_scales, bp, 90, 100);
                ncnn::Mat out_weight_data_int8_scales(1);
                out_weight_data_int8_scales[0] = op->out_weight_data_int8_scale;
                fwrite_weight_data(out_weight_data_int8_scales, bp, 90, 100);
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "MVN")
        {
            ncnn::MVN* op = (ncnn::MVN*)layer;
            ncnn::MVN* op_default = (ncnn::MVN*)layer_default;

            fprintf_param_value(" 0=%d", normalize_variance)
            fprintf_param_value(" 1=%d", across_channels)
            fprintf_param_value(" 2=%e", eps)
        }
        else if (layer->type == "Normalize")
        {
            ncnn::Normalize* op = (ncnn::Normalize*)layer;
            ncnn::Normalize* op_default = (ncnn::Normalize*)layer_default;

            fprintf_param_value(" 0=%d", across_spatial)
            fprintf_param_value(" 1=%d", channel_shared)
            fprintf_param_value(" 2=%e", eps)
            fprintf_param_value(" 3=%d", scale_data_size)
            fprintf_param_value(" 4=%d", across_channel)
            fprintf_param_value(" 9=%d", eps_mode)

            fwrite_weight_data(op->scale_data, bp);
        }
        else if (layer->type == "Padding")
        {
            ncnn::Padding* op = (ncnn::Padding*)layer;
            ncnn::Padding* op_default = (ncnn::Padding*)layer_default;

            fprintf_param_value(" 0=%d", top)
            fprintf_param_value(" 1=%d", bottom)
            fprintf_param_value(" 2=%d", left)
            fprintf_param_value(" 3=%d", right)
            fprintf_param_value(" 4=%d", type)
            fprintf_param_value(" 5=%e", value)
            fprintf_param_value(" 6=%d", per_channel_pad_data_size)
            fprintf_param_value(" 7=%d", front)
            fprintf_param_value(" 8=%d", behind)

            fwrite_weight_data(op->per_channel_pad_data, bp);
        }
        else if (layer->type == "Permute")
        {
            ncnn::Permute* op = (ncnn::Permute*)layer;
            ncnn::Permute* op_default = (ncnn::Permute*)layer_default;

            fprintf_param_value(" 0=%d", order_type)
        }
        else if (layer->type == "PixelShuffle")
        {
            ncnn::PixelShuffle* op = (ncnn::PixelShuffle*)layer;
            ncnn::PixelShuffle* op_default = (ncnn::PixelShuffle*)layer_default;

            fprintf_param_value(" 0=%d", upscale_factor)
            fprintf_param_value(" 1=%d", mode)
        }
        else if (layer->type == "Pooling")
        {
            ncnn::Pooling* op = (ncnn::Pooling*)layer;
            ncnn::Pooling* op_default = (ncnn::Pooling*)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 12=%d", op->stride_h);
            }
            fprintf_param_value(" 3=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 13=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 14=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 15=%d", op->pad_bottom);
            }
            fprintf_param_value(" 4=%d", global_pooling)
            fprintf_param_value(" 5=%d", pad_mode)
            fprintf_param_value(" 6=%d", avgpool_count_include_pad)
            fprintf_param_value(" 7=%d", adaptive_pooling)
            fprintf_param_value(" 8=%d", out_w)
            {
                if (op->out_h != op->out_w) fprintf(pp, " 18=%d", op->out_h);
            }
        }
        else if (layer->type == "Pooling1D")
        {
            ncnn::Pooling1D* op = (ncnn::Pooling1D*)layer;
            ncnn::Pooling1D* op_default = (ncnn::Pooling1D*)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
            fprintf_param_value(" 1=%d", kernel_w)
            fprintf_param_value(" 2=%d", stride_w)
            fprintf_param_value(" 3=%d", pad_left)
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 14=%d", op->pad_right);
            }
            fprintf_param_value(" 4=%d", global_pooling)
            fprintf_param_value(" 5=%d", pad_mode)
            fprintf_param_value(" 6=%d", avgpool_count_include_pad)
            fprintf_param_value(" 7=%d", adaptive_pooling)
            fprintf_param_value(" 8=%d", out_w)
        }
        else if (layer->type == "Pooling3D")
        {
            ncnn::Pooling3D* op = (ncnn::Pooling3D*)layer;
            ncnn::Pooling3D* op_default = (ncnn::Pooling3D*)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
                if (op->kernel_d != op->kernel_w) fprintf(pp, " 21=%d", op->kernel_d);
            }
            fprintf_param_value(" 2=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 12=%d", op->stride_h);
                if (op->stride_d != op->stride_w) fprintf(pp, " 22=%d", op->stride_d);
            }
            fprintf_param_value(" 3=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 13=%d", op->pad_top);
                if (op->pad_front != op->pad_left) fprintf(pp, " 23=%d", op->pad_front);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 14=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 15=%d", op->pad_bottom);
            }
            {
                if (op->pad_behind != op->pad_front) fprintf(pp, " 16=%d", op->pad_behind);
            }
            fprintf_param_value(" 4=%d", global_pooling)
            fprintf_param_value(" 5=%d", pad_mode)
            fprintf_param_value(" 6=%d", avgpool_count_include_pad)
            fprintf_param_value(" 7=%d", adaptive_pooling)
            fprintf_param_value(" 8=%d", out_w)
            {
                if (op->out_h != op->out_w) fprintf(pp, " 18=%d", op->out_h);
                if (op->out_d != op->out_w) fprintf(pp, " 28=%d", op->out_d);
            }
        }
        else if (layer->type == "Power")
        {
            ncnn::Power* op = (ncnn::Power*)layer;
            ncnn::Power* op_default = (ncnn::Power*)layer_default;

            fprintf_param_value(" 0=%e", power)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "PReLU")
        {
            ncnn::PReLU* op = (ncnn::PReLU*)layer;
            ncnn::PReLU* op_default = (ncnn::PReLU*)layer_default;

            fprintf_param_value(" 0=%d", num_slope)

            fwrite_weight_data(op->slope_data, bp);
        }
        else if (layer->type == "PriorBox")
        {
            ncnn::PriorBox* op = (ncnn::PriorBox*)layer;
            ncnn::PriorBox* op_default = (ncnn::PriorBox*)layer_default;

            {
                if (!op->min_sizes.empty()) fprintf_param_float_array(0, op->min_sizes, pp);
            }
            {
                if (!op->max_sizes.empty()) fprintf_param_float_array(1, op->max_sizes, pp);
            }
            {
                if (!op->aspect_ratios.empty()) fprintf_param_float_array(2, op->aspect_ratios, pp);
            }
            fprintf_param_value(" 3=%e", variances[0])
            fprintf_param_value(" 4=%e", variances[1])
            fprintf_param_value(" 5=%e", variances[2])
            fprintf_param_value(" 6=%e", variances[3])
            fprintf_param_value(" 7=%d", flip)
            fprintf_param_value(" 8=%d", clip)
            fprintf_param_value(" 9=%d", image_width)
            fprintf_param_value(" 10=%d", image_height)
            fprintf_param_value(" 11=%e", step_width)
            fprintf_param_value(" 12=%e", step_height)
            fprintf_param_value(" 13=%e", offset)
        }
        else if (layer->type == "Proposal")
        {
            ncnn::Proposal* op = (ncnn::Proposal*)layer;
            ncnn::Proposal* op_default = (ncnn::Proposal*)layer_default;

            fprintf_param_value(" 0=%d", feat_stride)
            fprintf_param_value(" 1=%d", base_size)
            fprintf_param_value(" 2=%d", pre_nms_topN)
            fprintf_param_value(" 3=%d", after_nms_topN)
            fprintf_param_value(" 4=%e", nms_thresh)
            fprintf_param_value(" 5=%d", min_size)
        }
        else if (layer->type == "PSROIPooling")
        {
            ncnn::PSROIPooling* op = (ncnn::PSROIPooling*)layer;
            ncnn::PSROIPooling* op_default = (ncnn::PSROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
            fprintf_param_value(" 3=%d", output_dim)
        }
        else if (layer->type == "Quantize")
        {
            ncnn::Quantize* op = (ncnn::Quantize*)layer;
            ncnn::Quantize* op_default = (ncnn::Quantize*)layer_default;

            fprintf_param_value(" 0=%d", scale_data_size)

            fwrite_weight_data(op->scale_data, bp);
        }
        else if (layer->type == "Reduction")
        {
            ncnn::Reduction* op = (ncnn::Reduction*)layer;
            ncnn::Reduction* op_default = (ncnn::Reduction*)layer_default;

            fprintf_param_value(" 0=%d", operation)
            fprintf_param_value(" 1=%d", reduce_all)
            fprintf_param_value(" 2=%e", coeff)
            {
                if (!op->axes.empty()) fprintf_param_int_array(3, op->axes, pp);
            }
            fprintf_param_value(" 4=%d", keepdims)

            // HACK
            if (!op->axes.empty())
            {
                int fixbug0 = 1;
                fprintf(pp, " 5=%d", fixbug0);
            }
        }
        else if (layer->type == "ReLU")
        {
            ncnn::ReLU* op = (ncnn::ReLU*)layer;
            ncnn::ReLU* op_default = (ncnn::ReLU*)layer_default;

            fprintf_param_value(" 0=%e", slope)
        }
        else if (layer->type == "Reorg")
        {
            ncnn::Reorg* op = (ncnn::Reorg*)layer;
            ncnn::Reorg* op_default = (ncnn::Reorg*)layer_default;

            fprintf_param_value(" 0=%d", stride)
            fprintf_param_value(" 1=%d", mode)
        }
        else if (layer->type == "Requantize")
        {
            ncnn::Requantize* op = (ncnn::Requantize*)layer;
            ncnn::Requantize* op_default = (ncnn::Requantize*)layer_default;

            fprintf_param_value(" 0=%d", scale_in_data_size)
            fprintf_param_value(" 1=%d", scale_out_data_size)
            fprintf_param_value(" 2=%d", bias_data_size)
            fprintf_param_value(" 3=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(4, op->activation_params, pp);
            }

            fwrite_weight_data(op->scale_in_data, bp);
            fwrite_weight_data(op->scale_out_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "Reshape")
        {
            ncnn::Reshape* op = (ncnn::Reshape*)layer;
            ncnn::Reshape* op_default = (ncnn::Reshape*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 11=%d", d)
            fprintf_param_value(" 2=%d", c)
            {
                if (op->shape_expr != op_default->shape_expr) fprintf(pp, " 6=\"%s\"", op->shape_expr.c_str());
            }
        }
        else if (layer->type == "RMSNorm")
        {
            ncnn::RMSNorm* op = (ncnn::RMSNorm*)layer;
            ncnn::RMSNorm* op_default = (ncnn::RMSNorm*)layer_default;

            fprintf_param_value(" 0=%d", affine_size)
            fprintf_param_value(" 1=%e", eps)
            fprintf_param_value(" 2=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
        }
        else if (layer->type == "RNN")
        {
            ncnn::RNN* op = (ncnn::RNN*)layer;
            ncnn::RNN* op_default = (ncnn::RNN*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", weight_data_size)
            fprintf_param_value(" 2=%d", direction)
            fprintf_param_value(" 8=%d", int8_scale_term)

            fwrite_weight_tag_data(op->weight_xc_data, bp);
            fwrite_weight_tag_data(op->bias_c_data, bp);
            fwrite_weight_tag_data(op->weight_hc_data, bp);

#if NCNN_INT8
            // write int8_scale data
            if (op->int8_scale_term)
            {
                fwrite_weight_data(op->weight_xc_data_int8_scales, bp, 90, 100);
                fwrite_weight_data(op->weight_hc_data_int8_scales, bp, 90, 100);
            }
#endif // NCNN_INT8
        }
        else if (layer->type == "ROIAlign")
        {
            ncnn::ROIAlign* op = (ncnn::ROIAlign*)layer;
            ncnn::ROIAlign* op_default = (ncnn::ROIAlign*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
            fprintf_param_value(" 3=%d", sampling_ratio)
            fprintf_param_value(" 4=%d", aligned)
            fprintf_param_value(" 5=%d", version)
        }
        else if (layer->type == "ROIPooling")
        {
            ncnn::ROIPooling* op = (ncnn::ROIPooling*)layer;
            ncnn::ROIPooling* op_default = (ncnn::ROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
        }
        else if (layer->type == "RotaryEmbed")
        {
            ncnn::RotaryEmbed* op = (ncnn::RotaryEmbed*)layer;
            ncnn::RotaryEmbed* op_default = (ncnn::RotaryEmbed*)layer_default;

            fprintf_param_value(" 0=%d", interleaved)
        }
        else if (layer->type == "Scale")
        {
            ncnn::Scale* op = (ncnn::Scale*)layer;
            ncnn::Scale* op_default = (ncnn::Scale*)layer_default;

            fprintf_param_value(" 0=%d", scale_data_size)
            fprintf_param_value(" 1=%d", bias_term)

            fwrite_weight_data(op->scale_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "SDPA")
        {
            ncnn::SDPA* op = (ncnn::SDPA*)layer;
            ncnn::SDPA* op_default = (ncnn::SDPA*)layer_default;

            fprintf_param_value(" 5=%d", attn_mask)
            fprintf_param_value(" 6=%e", scale)
            fprintf_param_value(" 7=%d", kv_cache)
            fprintf_param_value(" 18=%d", int8_scale_term)
        }
        else if (layer->type == "ShuffleChannel")
        {
            ncnn::ShuffleChannel* op = (ncnn::ShuffleChannel*)layer;
            ncnn::ShuffleChannel* op_default = (ncnn::ShuffleChannel*)layer_default;

            fprintf_param_value(" 0=%d", group)
            fprintf_param_value(" 1=%d", reverse)
        }
        else if (layer->type == "Slice")
        {
            ncnn::Slice* op = (ncnn::Slice*)layer;
            ncnn::Slice* op_default = (ncnn::Slice*)layer_default;

            {
                if (!op->slices.empty()) fprintf_param_int_array(0, op->slices, pp);
            }
            fprintf_param_value(" 1=%d", axis)
        }
        else if (layer->type == "Softmax")
        {
            ncnn::Softmax* op = (ncnn::Softmax*)layer;
            ncnn::Softmax* op_default = (ncnn::Softmax*)layer_default;

            fprintf_param_value(" 0=%d", axis)

            // HACK
            if (op->axis != 0)
            {
                int fixbug0 = 1;
                fprintf(pp, " 1=%d", fixbug0);
            }
        }
        else if (layer->type == "Squeeze")
        {
            ncnn::Squeeze* op = (ncnn::Squeeze*)layer;
            ncnn::Squeeze* op_default = (ncnn::Squeeze*)layer_default;

            fprintf_param_value(" 0=%d", squeeze_w)
            fprintf_param_value(" 1=%d", squeeze_h)
            fprintf_param_value(" 11=%d", squeeze_d)
            fprintf_param_value(" 2=%d", squeeze_c)
            {
                if (!op->axes.empty()) fprintf_param_int_array(3, op->axes, pp);
            }
        }
        else if (layer->type == "Threshold")
        {
            ncnn::Threshold* op = (ncnn::Threshold*)layer;
            ncnn::Threshold* op_default = (ncnn::Threshold*)layer_default;

            fprintf_param_value(" 0=%e", threshold)
        }
        else if (layer->type == "UnaryOp")
        {
            ncnn::UnaryOp* op = (ncnn::UnaryOp*)layer;
            ncnn::UnaryOp* op_default = (ncnn::UnaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
        }
        else if (layer->type == "Unfold")
        {
            ncnn::Unfold* op = (ncnn::Unfold*)layer;
            ncnn::Unfold* op_default = (ncnn::Unfold*)layer_default;

            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%e", pad_value)
        }
        else if (layer->type == "YoloDetectionOutput")
        {
            ncnn::YoloDetectionOutput* op = (ncnn::YoloDetectionOutput*)layer;
            ncnn::YoloDetectionOutput* op_default = (ncnn::YoloDetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%e", confidence_threshold)
            fprintf_param_value(" 3=%e", nms_threshold)
            {
                if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp);
            }
        }
        else if (layer->type == "Yolov3DetectionOutput")
        {
            ncnn::Yolov3DetectionOutput* op = (ncnn::Yolov3DetectionOutput*)layer;
            ncnn::Yolov3DetectionOutput* op_default = (ncnn::Yolov3DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%e", confidence_threshold)
            fprintf_param_value(" 3=%e", nms_threshold)
            {
                if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp);
            }
            {
                if (!op->mask.empty()) fprintf_param_int_array(5, op->mask, pp);
            }
            {
                if (!op->anchors_scale.empty()) fprintf_param_float_array(6, op->anchors_scale, pp);
            }
        }

#undef fprintf_param_value

        fprintf(pp, "\n");

        delete layer_default;
    }

    fclose(pp);
    fclose(bp);

    if (mac)
    {
        fprintf(stderr, "mac = %llu = %.2f M\n", static_cast<long long unsigned>(mac), mac / 1000000.0);
    }

    return 0;
}
