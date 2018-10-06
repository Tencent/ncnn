#include <cstdio>
#include <vector>

#include <layer.h>
#include <layer/convolution.h>
#include <layer/convolutiondepthwise.h>
#include <layer/batchnorm.h>
#include <layer/scale.h>
#include <net.h>

#define xstr(a) str(a)
#define str(a) #a

#define FuseTwoLayers(a_t, b_t, handler)  \
        for (size_t i = 0; i < net.layers.size() - 1; i++) {    \
            auto *layer = net.layers[i];    \
            auto *next_layer = net.layers[i + 1];   \
            auto type = layer->type;    \
            if (type == xstr(a_t)) {    \
                if (next_layer->type == xstr(b_t)) {  \
                    a_t *layer_a = static_cast<a_t *>(layer);    \
                    b_t *layer_b = static_cast<b_t *>(next_layer); \
                    handler(layer_a, layer_b);  \
                    net.layers.erase(net.layers.begin() + (i + 1)); \
                    for (size_t j = i + 1; j < net.layers.size(); j++) {    \
                        auto *layer2 = net.layers[j];   \
                        for (size_t k = 0; k < layer2->bottoms.size(); k++) {    \
                            if (layer2->bottoms[k] == layer_b->tops[0]) {    \
                                layer2->bottoms[k] = layer_a->tops[0];   \
                            }   \
                        }   \
                    }   \
                }   \
            }   \
        }   \
        \

namespace ncnn {
class OpFuser {
private:
    void FuseDWConvBN(ConvolutionDepthWise *conv, BatchNorm *bn) {
        auto &conv_weight = conv->weight_data;        // for group==input_channel: [channel, h, w]       
        auto &bn_a = bn->a_data, bn_b = bn->b_data;     // [channel]
        size_t step = conv_weight.total() / conv->num_output;   // TODO: support group > 1 && != input_channel
        for (int n = 0; n < conv->num_output; n++) {
            for (size_t s = 0; s < step; s++) {
                conv_weight[n * step + s] *= bn_b[n];
            }
        }
        if (conv->bias_term) {
            for (int n = 0; n < conv->num_output; n++) {
                auto &conv_bias = conv->bias_data;
                conv_bias[n] = conv_bias[n] * bn_b[n] + bn_a[n];
            }
        } else {
            conv->bias_term = 1;
            conv->bias_data = bn_a;
        }
    }
    void FuseDWConvScale(ConvolutionDepthWise *conv, Scale *scale) {
        auto &conv_weight = conv->weight_data;       // for group==input_channel: [channel, h, w]
        auto &scale_blob = scale->scale_data;     // [channel]
        size_t step = conv_weight.total() / conv->num_output;
        for (int n = 0; n < conv->num_output; n++) {
            for (size_t s = 0; s < step; s++) {
                conv_weight[n * step + s] *= scale_blob[n];
            }
        }
        if (scale->bias_term) {
            auto &scale_bias = scale->bias_data;
            for (int n = 0; n < conv->num_output; n++) {
                if (conv->bias_term) {
                    auto &conv_bias = conv->bias_data;
                    conv_bias[n] = conv_bias[n] * scale_blob[n] + scale_bias[n];
                } else {
                    conv->bias_term = 1;
                    conv->bias_data = scale_bias;
                }
            }
        }
        if (scale->bias_term) {
            auto &scale_bias = scale->bias_data;
            if (conv->bias_term) {
                auto &conv_bias = conv->bias_data;
                for (int n = 0; n < conv->num_output; n++) {
                    conv_bias[n] = conv_bias[n] * scale_blob[n] + scale_bias[n];
                }
            } else {
                conv->bias_term = 1;
                conv->bias_data = scale_bias;
            }
        }
    }
    void FuseConvBN(Convolution *conv, BatchNorm *bn) {
        auto &conv_weight = conv->weight_data;        // [output_channel, input_channel, h, w]
        auto &bn_a = bn->a_data, bn_b = bn->b_data;     // [channel]
        size_t step = conv_weight.total() / conv->num_output;
        for (int n = 0; n < conv->num_output; n++) {
            for (size_t s = 0; s < step; s++) {
                conv_weight[n * step + s] *= bn_b[n];
            }
        }
        if (conv->bias_term) {
            for (int n = 0; n < conv->num_output; n++) {
                auto &conv_bias = conv->bias_data;
                conv_bias[n] = conv_bias[n] * bn_b[n] + bn_a[n];
            }
        } else {
            conv->bias_term = 1;
            conv->bias_data = bn_a;
        }
    }
    void FuseConvScale(Convolution *conv, Scale *scale) {
        auto &conv_weight = conv->weight_data;        // [output_channel, input_channel, h, w]
        auto &scale_blob = scale->scale_data;     // [channel]
        size_t step = conv_weight.total() / conv->num_output;
        for (int n = 0; n < conv->num_output; n++) {
            for (size_t s = 0; s < step; s++) {
                conv_weight[n * step + s] *= scale_blob[n];
            }
        }
        if (scale->bias_term) {
            auto &scale_bias = scale->bias_data;
            if (conv->bias_term) {
                auto &conv_bias = conv->bias_data;
                for (int n = 0; n < conv->num_output; n++) {
                    conv_bias[n] = conv_bias[n] * scale_blob[n] + scale_bias[n];
                }
            } else {
                conv->bias_term = 1;
                conv->bias_data = scale_bias;
            }
        }
    }
public:
    void fuse(const char* parampath, const char* modelpath, const char* fusedparampath, const char* fusedmodelpath) {
        Net net;
        auto ret = net.load_param(parampath);
        if (ret != 0) {
            fprintf(stderr, "load_param failed");
            return;
        }
        ret = net.load_model(modelpath);
        if (ret != 0) {
            fprintf(stderr, "load_model failed");
            return;
        }
        
        FuseTwoLayers(Convolution, BatchNorm, FuseConvBN);
        FuseTwoLayers(Convolution, Scale, FuseConvScale);
        FuseTwoLayers(ConvolutionDepthWise, BatchNorm, FuseDWConvBN);
        FuseTwoLayers(ConvolutionDepthWise, Scale, FuseDWConvScale);

        FILE* pp;
        pp = fopen(fusedparampath, "w");
        net.save_param(pp);
        fclose(pp);
        FILE* mp;
        mp = fopen(fusedmodelpath, "w");
        net.save_model(mp);
        fclose(mp);
    }
};
}

int main(int argc, char** argv) {
    ncnn::OpFuser fuser;
    fuser.fuse(argv[1], argv[2], argv[3], argv[4]);
}
