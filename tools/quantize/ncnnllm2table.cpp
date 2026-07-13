// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <float.h>
#include <string>
#include <vector>

// npy format header
#include "npy.hpp"

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "../modelwriter.h"
#include "ncnnllm_quant.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* /*format*/, void* /*p*/) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class QuantActStat
{
public:
    QuantActStat()
    {
        width = 0;
        count = 0;
        max_sample_count = 0;
    }

public:
    int width;
    int count;
    int max_sample_count;
    ncnn::Mat sum_abs;
    std::vector<float> samples;
};

class QuantNet : public ModelWriter
{
public:
    QuantNet();

public:
    std::vector<std::vector<std::string> > listspaths;
    std::vector<std::vector<int> > shapes;
    std::vector<int> input_blobs;
    std::vector<QuantActStat> gemm_act_stats;
    std::vector<QuantActStat> mha_act_stats;
    int quantize_num_threads;
    int file_type;
    int awq_steps;
    int awq_samples;
    int awq_inner_method;
    float awq_max_scale;
    int gptq_samples;
    float gptq_damp;
    bool use_calibration_dataset;

public:
    int init(int method);
    int collect_activation_stats();
    int save_table(const char* tablepath, int block_size, int weight_bits, int method) const;
};

QuantNet::QuantNet()
    : ModelWriter()
{
    quantize_num_threads = 1;
    file_type = 1;
    awq_steps = 20;
    awq_samples = 128;
    awq_inner_method = LLM_QUANT_METHOD_MINMAX;
    awq_max_scale = 16.f;
    gptq_samples = 128;
    gptq_damp = 0.01f;
    use_calibration_dataset = false;
}

static void show_usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s [inparam] [inbin] [outtable] [(key=value)...]\n", argv0);
    fprintf(stderr, "       %s [inparam] [inbin] [caliblist] [outtable] [(key=value)...]\n", argv0);
    fprintf(stderr, "  method=minmax/mseclip/awq/gptq\n");
    fprintf(stderr, "  bits=4/6/8\n");
    fprintf(stderr, "  block=32/64/128\n");
    fprintf(stderr, "  thread=8\n");
    fprintf(stderr, "  type=1\n");
    fprintf(stderr, "  shape=[w,h,...]\n");
    fprintf(stderr, "Sample usage:\n");
    fprintf(stderr, "  %s model.param model.bin model.llm.table method=mseclip bits=4 block=64\n", argv0);
    fprintf(stderr, "  %s model.param model.bin calib.list model.llm.table method=awq bits=4 block=64 shape=[4096]\n", argv0);
    fprintf(stderr, "  %s model.param model.bin calib.list model.llm.table method=gptq bits=4 block=64 shape=[4096]\n", argv0);
}

static ncnn::Mat read_npy(const std::vector<int>& shape, const std::string& npypath)
{
    npy::npy_data<float> d;
    try
    {
        d = npy::read_npy<float>(npypath);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "npy::read_npy %s exception: %s\n", npypath.c_str(), e.what());
        return ncnn::Mat();
    }

    const std::vector<unsigned long>& npy_shape = d.shape;
    const size_t dims = shape.size();

    if (dims != npy_shape.size())
    {
        fprintf(stderr, "%s expect %d dims, but got %d\n", npypath.c_str(), (int)dims, (int)npy_shape.size());
        return ncnn::Mat();
    }

    for (size_t i = 0; i < dims; i++)
    {
        if ((unsigned long)shape[i] != npy_shape[dims - 1 - i])
        {
            fprintf(stderr, "%s shape mismatch\n", npypath.c_str());
            return ncnn::Mat();
        }
    }

    if (dims == 1)
        return ncnn::Mat(shape[0], (void*)d.data.data()).reshape(shape[0]).clone();
    if (dims == 2)
        return ncnn::Mat(shape[0] * shape[1], (void*)d.data.data()).reshape(shape[0], shape[1]).clone();
    if (dims == 3)
        return ncnn::Mat(shape[0] * shape[1] * shape[2], (void*)d.data.data()).reshape(shape[0], shape[1], shape[2]).clone();
    if (dims == 4)
        return ncnn::Mat(shape[0] * shape[1] * shape[2] * shape[3], (void*)d.data.data()).reshape(shape[0], shape[1], shape[2], shape[3]).clone();

    fprintf(stderr, "%s dims %d is unsupported\n", npypath.c_str(), (int)dims);
    return ncnn::Mat();
}

static std::vector<std::vector<std::string> > parse_comma_path_list(char* s)
{
    std::vector<std::vector<std::string> > aps;

    char* pch = strtok(s, ",");
    while (pch != NULL)
    {
        FILE* fp = fopen(pch, "rb");
        if (!fp)
        {
            fprintf(stderr, "fopen %s failed\n", pch);
            break;
        }

        std::vector<std::string> paths;

        char line[1024];
        while (!feof(fp))
        {
            char* ss = fgets(line, 1024, fp);
            if (!ss)
                break;

            char filepath[256];
            int nscan = sscanf(line, "%255s", filepath);
            if (nscan != 1)
                continue;

            paths.push_back(std::string(filepath));
        }

        fclose(fp);

        aps.push_back(paths);

        pch = strtok(NULL, ",");
    }

    return aps;
}

static std::vector<std::vector<int> > parse_comma_int_array_list(char* s)
{
    std::vector<std::vector<int> > aai;

    char* pch = strtok(s, "[]");
    while (pch != NULL)
    {
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1)
        {
            pch += nconsumed;

            std::vector<int> ai;
            ai.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1)
            {
                pch += nconsumed;

                ai.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            aai.push_back(ai);
        }

        pch = strtok(NULL, "[]");
    }

    return aai;
}

static std::string make_qweight_filename(const char* tablepath, const char* key, std::string& qweight_path)
{
    const char* slash = strrchr(tablepath, '/');
#ifdef _WIN32
    const char* backslash = strrchr(tablepath, '\\');
    if (backslash && (!slash || backslash > slash))
        slash = backslash;
#endif

    const char* table_base = slash ? slash + 1 : tablepath;
    std::string qweight_name = std::string(table_base) + "." + key + ".qweight";
    for (size_t i = 0; i < qweight_name.size(); i++)
    {
        if (qweight_name[i] == '/' || qweight_name[i] == '\\' || qweight_name[i] == ':' || qweight_name[i] == ' ')
            qweight_name[i] = '_';
    }

    qweight_path = qweight_name;
    if (slash)
        qweight_path = std::string(tablepath, slash - tablepath + 1) + qweight_name;

    return qweight_name;
}

static int write_raw_mat_file(const char* path, const ncnn::Mat& m)
{
    FILE* fp = fopen(path, "wb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    const size_t logical_count = (size_t)m.w * m.h * m.d * m.c;
    if (fwrite(m.data, m.elemsize, logical_count, fp) != logical_count)
    {
        fprintf(stderr, "fwrite %s failed\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    return 0;
}

static int init_act_stat(QuantActStat& stat, int width, int max_sample_count)
{
    stat.width = width;
    stat.count = 0;
    stat.max_sample_count = max_sample_count;

    stat.sum_abs.create(width);
    if (stat.sum_abs.empty())
        return -100;

    stat.sum_abs.fill(0.f);
    stat.samples.clear();
    stat.samples.reserve((size_t)width * max_sample_count);

    return 0;
}

static void collect_act_row(QuantActStat& stat, const float* ptr)
{
    const int width = stat.width;
    float* sum_abs_ptr = stat.sum_abs;

    for (int k = 0; k < width; k++)
    {
        const float v = (float)fabs(ptr[k]);
        sum_abs_ptr[k] += v;
    }

    if ((int)(stat.samples.size() / width) < stat.max_sample_count)
    {
        for (int k = 0; k < width; k++)
            stat.samples.push_back(ptr[k]);
    }

    stat.count++;
}

static void collect_act_rows(QuantActStat& stat, const ncnn::Mat& m)
{
    if (m.dims == 1)
    {
        collect_act_row(stat, (const float*)m);
        return;
    }

    if (m.dims == 2)
    {
        for (int y = 0; y < m.h; y++)
            collect_act_row(stat, m.row(y));
        return;
    }

    if (m.dims == 3)
    {
        for (int q = 0; q < m.c; q++)
        {
            const ncnn::Mat m0 = m.channel(q);
            for (int y = 0; y < m.h; y++)
            {
                collect_act_row(stat, m0.row(y));
            }
        }
        return;
    }

    if (m.dims == 4)
    {
        for (int q = 0; q < m.c; q++)
        {
            const ncnn::Mat m0 = m.channel(q);
            for (int z = 0; z < m.d; z++)
            {
                const ncnn::Mat m1 = m0.depth(z);
                for (int y = 0; y < m.h; y++)
                {
                    collect_act_row(stat, m1.row(y));
                }
            }
        }
        return;
    }
}

static void resolve_mha_bottom_blob_index(const ncnn::MultiHeadAttention* mha, int bottom_blob_count, int& q_blob_i, int& k_blob_i, int& v_blob_i, int& attn_mask_i, int& cached_xk_i, int& cached_xv_i)
{
    if (mha->kv_cache)
    {
        if (mha->attn_mask)
        {
            // assert bottom_blob_count == 4/5/6
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                attn_mask_i = 1;
                cached_xk_i = 2;
                cached_xv_i = 3;
            }
            if (bottom_blob_count == 5)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                attn_mask_i = 2;
                cached_xk_i = 3;
                cached_xv_i = 4;
            }
            if (bottom_blob_count == 6)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                attn_mask_i = 3;
                cached_xk_i = 4;
                cached_xv_i = 5;
            }
        }
        else
        {
            // assert bottom_blob_count == 3/4/5
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                cached_xk_i = 1;
                cached_xv_i = 2;
            }
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                cached_xk_i = 2;
                cached_xv_i = 3;
            }
            if (bottom_blob_count == 5)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                cached_xk_i = 3;
                cached_xv_i = 4;
            }
        }
    }
    else
    {
        if (mha->attn_mask)
        {
            // assert bottom_blob_count == 2/3/4
            if (bottom_blob_count == 2)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
                attn_mask_i = 1;
            }
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
                attn_mask_i = 2;
            }
            if (bottom_blob_count == 4)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
                attn_mask_i = 3;
            }
        }
        else
        {
            // assert bottom_blob_count == 1/2/3
            if (bottom_blob_count == 1)
            {
                q_blob_i = 0;
                k_blob_i = 0;
                v_blob_i = 0;
            }
            if (bottom_blob_count == 2)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 1;
            }
            if (bottom_blob_count == 3)
            {
                q_blob_i = 0;
                k_blob_i = 1;
                v_blob_i = 2;
            }
        }
    }
}

static int collect_mha_out_act_rows(const ncnn::MultiHeadAttention* mha, const ncnn::Mat& q_blob, const ncnn::Mat& k_blob, const ncnn::Mat& v_blob, const ncnn::Mat& attn_mask_blob, const ncnn::Mat& cached_xk_blob, const ncnn::Mat& cached_xv_blob, int q_blob_i, int k_blob_i, int v_blob_i, QuantActStat& stat, int num_threads)
{
    const int embed_dim = mha->embed_dim;
    const int num_heads = mha->num_heads;
    const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = mha->weight_data_size / embed_dim;

    const int src_seqlen = q_blob.h;
    const int cur_seqlen = k_blob.h;
    const int past_seqlen = mha->kv_cache && !cached_xk_blob.empty() ? cached_xk_blob.w : 0;
    const int dst_seqlen = past_seqlen > 0 ? (q_blob_i == k_blob_i ? (past_seqlen + cur_seqlen) : past_seqlen) : cur_seqlen;

    ncnn::Mat q_affine;
    q_affine.create(src_seqlen, embed_dim);
    if (q_affine.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < src_seqlen; i++)
    {
        const float* kptr = (const float*)mha->q_weight_data;

        for (int j = 0; j < embed_dim; j++)
        {
            const float* ptr = q_blob.row(i);

            float sum = mha->q_bias_data[j];
            for (int k = 0; k < qdim; k++)
                sum += *ptr++ * *kptr++;

            q_affine.row(j)[i] = sum * mha->scale;
        }
    }

    ncnn::Mat k_affine;
    if (past_seqlen > 0 && q_blob_i != k_blob_i)
    {
        k_affine = cached_xk_blob;
    }
    else
    {
        k_affine.create(dst_seqlen, embed_dim);
        if (k_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < embed_dim; i++)
                memcpy(k_affine.row(i), cached_xk_blob.row(i), past_seqlen * sizeof(float));
        }

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const float* kptr = (const float*)mha->k_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = k_blob.row(i);

                float sum = mha->k_bias_data[j];
                for (int k = 0; k < mha->kdim; k++)
                    sum += *ptr++ * *kptr++;

                k_affine.row(j)[past_seqlen + i] = sum;
            }
        }
    }

    ncnn::Mat v_affine;
    if (past_seqlen > 0 && q_blob_i != v_blob_i)
    {
        v_affine = cached_xv_blob;
    }
    else
    {
        v_affine.create(dst_seqlen, embed_dim);
        if (v_affine.empty())
            return -100;

        if (past_seqlen > 0)
        {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < embed_dim; i++)
                memcpy(v_affine.row(i), cached_xv_blob.row(i), past_seqlen * sizeof(float));
        }

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < cur_seqlen; i++)
        {
            const float* kptr = (const float*)mha->v_weight_data;

            for (int j = 0; j < embed_dim; j++)
            {
                const float* ptr = v_blob.row(i);

                float sum = mha->v_bias_data[j];
                for (int k = 0; k < mha->vdim; k++)
                    sum += *ptr++ * *kptr++;

                v_affine.row(j)[past_seqlen + i] = sum;
            }
        }
    }

    ncnn::Mat qk_cross;
    qk_cross.create(dst_seqlen, src_seqlen, num_heads);
    if (qk_cross.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const ncnn::Mat q_affine_head = q_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
        const ncnn::Mat k_affine_head = k_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
        ncnn::Mat qk_cross_head = qk_cross.channel(q);

        for (int i = 0; i < src_seqlen; i++)
        {
            float* outptr = qk_cross_head.row(i);

            for (int j = 0; j < dst_seqlen; j++)
            {
                float sum = 0.f;
                for (int l = 0; l < embed_dim_per_head; l++)
                    sum += q_affine_head.row(l)[i] * k_affine_head.row(l)[j];

                outptr[j] = sum;
            }
        }
    }

    if (mha->attn_mask)
    {
        #pragma omp parallel for num_threads(num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const ncnn::Mat& maskm = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(q) : attn_mask_blob;
            ncnn::Mat qk_cross_head = qk_cross.channel(q);

            for (int i = 0; i < src_seqlen; i++)
            {
                const float* mptr = maskm.row(i);
                float* outptr = qk_cross_head.row(i);

                for (int j = 0; j < dst_seqlen; j++)
                    outptr[j] += mptr[j];
            }
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        ncnn::Mat qk_cross_head = qk_cross.channel(q);

        for (int i = 0; i < src_seqlen; i++)
        {
            float* ptr = qk_cross_head.row(i);

            float max = -FLT_MAX;
            for (int j = 0; j < dst_seqlen; j++)
                max = std::max(max, ptr[j]);

            float sum = 0.f;
            for (int j = 0; j < dst_seqlen; j++)
            {
                ptr[j] = (float)expf(ptr[j] - max);
                sum += ptr[j];
            }

            for (int j = 0; j < dst_seqlen; j++)
                ptr[j] /= sum;
        }
    }

    ncnn::Mat qkv_cross;
    qkv_cross.create(src_seqlen, embed_dim);
    if (qkv_cross.empty())
        return -100;

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const ncnn::Mat qk_cross_head = qk_cross.channel(q);
        const ncnn::Mat v_affine_head = v_affine.row_range(q * embed_dim_per_head, embed_dim_per_head);
        ncnn::Mat qkv_cross_head = qkv_cross.row_range(q * embed_dim_per_head, embed_dim_per_head);

        for (int i = 0; i < src_seqlen; i++)
        {
            for (int j = 0; j < embed_dim_per_head; j++)
            {
                const float* qkptr = qk_cross_head.row(i);
                const float* vptr = v_affine_head.row(j);

                float sum = 0.f;
                for (int k = 0; k < dst_seqlen; k++)
                    sum += *qkptr++ * *vptr++;

                qkv_cross_head.row(j)[i] = sum;
            }
        }
    }

    std::vector<float> row(embed_dim);
    for (int i = 0; i < src_seqlen; i++)
    {
        for (int k = 0; k < embed_dim; k++)
            row[k] = qkv_cross.row(k)[i];

        collect_act_row(stat, row.data());
    }

    return 0;
}

static int make_input_scaled_weight(const ncnn::Mat& weight_data, const ncnn::Mat& input_scales, ncnn::Mat& weight_data_scaled)
{
    const int K = weight_data.w;
    const int N = weight_data.h;

    weight_data_scaled.create(K, N);
    if (weight_data_scaled.empty())
        return -100;

    const float* input_scale_ptr = input_scales;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* outptr = weight_data_scaled.row(n);

        for (int k = 0; k < K; k++)
            outptr[k] = ptr[k] / input_scale_ptr[k];
    }

    return 0;
}

static int make_dequant_weight(const ncnn::Mat& weight_data, const ncnn::Mat& weight_data_quantize_scales, int block_size, int weight_bits, ncnn::Mat& weight_data_dequantized)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int block_count = (K + block_size - 1) / block_size;

    weight_data_dequantized.create(K, N);
    if (weight_data_dequantized.empty())
        return -100;

    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        const float* scale_ptr = weight_data_quantize_scales.row(n);
        float* outptr = weight_data_dequantized.row(n);

        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = std::min(block_size, K - k0);
            const float scale = scale_ptr[b];

            for (int k = 0; k < max_kk; k++)
            {
                const int q = float2int_weight(ptr[k0 + k] * scale, weight_bits);
                outptr[k0 + k] = q / scale;
            }
        }
    }

    return 0;
}

static double calc_awq_reconstruction_error(const ncnn::Mat& weight_data, const ncnn::Mat& input_scales, const ncnn::Mat& weight_data_dequantized, const QuantActStat& stat)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int sample_count = (int)(stat.samples.size() / K);

    const float* input_scale_ptr = input_scales;
    double error = 0.0;

    for (int i = 0; i < sample_count; i++)
    {
        const float* sample_ptr = &stat.samples[(size_t)i * K];

        for (int n = 0; n < N; n++)
        {
            const float* wptr = weight_data.row(n);
            const float* deqptr = weight_data_dequantized.row(n);

            double ref = 0.0;
            double out = 0.0;
            for (int k = 0; k < K; k++)
            {
                const float x = sample_ptr[k];
                ref += (double)x * wptr[k];
                out += (double)(x * input_scale_ptr[k]) * deqptr[k];
            }

            const double diff = ref - out;
            error += diff * diff;
        }
    }

    return error;
}

static int make_awq_input_scales(const ncnn::Mat& weight_data, const QuantActStat& stat, int block_size, int weight_bits, int inner_method, int awq_steps, float awq_max_scale, ncnn::Mat& input_scales, ncnn::Mat& weight_data_quantize_scales, int num_threads)
{
    const int K = weight_data.w;
    const int N = weight_data.h;

    ncnn::Mat act_mean(K);
    ncnn::Mat weight_mean(K);
    if (act_mean.empty() || weight_mean.empty())
        return -100;

    const float* sum_abs_ptr = stat.sum_abs;
    float* act_mean_ptr = act_mean;
    for (int k = 0; k < K; k++)
        act_mean_ptr[k] = sum_abs_ptr[k] / stat.count;

    weight_mean.fill(0.f);
    float* weight_mean_ptr = weight_mean;
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        for (int k = 0; k < K; k++)
            weight_mean_ptr[k] += (float)fabs(ptr[k]);
    }
    for (int k = 0; k < K; k++)
        weight_mean_ptr[k] /= N;

    ncnn::Mat best_input_scales;
    ncnn::Mat best_quantize_scales;
    double best_error = DBL_MAX;

    const int search_steps = awq_steps;
    for (int s = 0; s <= search_steps; s++)
    {
        ncnn::Mat input_scales1(K);
        if (input_scales1.empty())
            return -100;

        float* input_scale_ptr = input_scales1;
        if (s == 0)
        {
            for (int k = 0; k < K; k++)
                input_scale_ptr[k] = 1.f;
        }
        else
        {
            const float alpha = (float)s / search_steps;
            double logsum = 0.0;
            for (int k = 0; k < K; k++)
            {
                float raw = powf((weight_mean_ptr[k] + 1e-6f) / (act_mean_ptr[k] + 1e-6f), alpha);
                if (raw < 1e-12f)
                    raw = 1e-12f;
                input_scale_ptr[k] = raw;
                logsum += ::log((double)raw);
            }

            const float geomean = (float)::exp(logsum / K);
            for (int k = 0; k < K; k++)
            {
                float v = input_scale_ptr[k] / geomean;
                if (v < 1.f / awq_max_scale)
                    v = 1.f / awq_max_scale;
                if (v > awq_max_scale)
                    v = awq_max_scale;

                input_scale_ptr[k] = v;
            }
        }

        ncnn::Mat weight_data_scaled;
        int ret = make_input_scaled_weight(weight_data, input_scales1, weight_data_scaled);
        if (ret != 0)
            return ret;

        ncnn::Mat quantize_scales1;
        ret = make_weight_scales(weight_data_scaled, block_size, weight_bits, inner_method, quantize_scales1, num_threads);
        if (ret != 0)
            return ret;

        ncnn::Mat weight_data_dequantized;
        ret = make_dequant_weight(weight_data_scaled, quantize_scales1, block_size, weight_bits, weight_data_dequantized);
        if (ret != 0)
            return ret;

        const double error = calc_awq_reconstruction_error(weight_data, input_scales1, weight_data_dequantized, stat);
        if (error < best_error)
        {
            best_error = error;
            best_input_scales = input_scales1;
            best_quantize_scales = quantize_scales1;
        }
    }

    input_scales = best_input_scales.clone();
    weight_data_quantize_scales = best_quantize_scales.clone();
    if (input_scales.empty() || weight_data_quantize_scales.empty())
        return -100;

    return 0;
}

static int cholesky_inverse_gptq_hessian(std::vector<double>& h, int n)
{
    for (int i = n - 1; i >= 0; i--)
    {
        double diag = h[(size_t)i * n + i];
        for (int k = i + 1; k < n; k++)
            diag -= h[(size_t)i * n + k] * h[(size_t)i * n + k];

        if (!(diag > 0.0))
            return -1;

        diag = sqrt(diag);
        h[(size_t)i * n + i] = diag;

        for (int j = 0; j < i; j++)
        {
            double v = h[(size_t)j * n + i];
            for (int k = i + 1; k < n; k++)
                v -= h[(size_t)j * n + k] * h[(size_t)i * n + k];

            h[(size_t)j * n + i] = v / diag;
        }
    }

    for (int i = n - 1; i >= 0; i--)
    {
        const double diag = h[(size_t)i * n + i];

        for (int j = n - 1; j > i; j--)
        {
            double v = 0.0;
            for (int k = i + 1; k <= j; k++)
                v += h[(size_t)i * n + k] * h[(size_t)k * n + j];

            h[(size_t)i * n + j] = -v / diag;
        }

        h[(size_t)i * n + i] = 1.0 / diag;
    }

    return 0;
}

static int make_gptq_qweight(const ncnn::Mat& weight_data, const QuantActStat& stat, int block_size, int weight_bits, float damp, ncnn::Mat& weight_data_quantize_scales, ncnn::Mat& weight_data_quantized, int num_threads)
{
    const int K = weight_data.w;
    const int N = weight_data.h;
    const int block_count = (K + block_size - 1) / block_size;
    const int packed_k_bytes = llm_weight_quantize_packed_k_bytes(K, weight_bits);
    const int sample_count = (int)(stat.samples.size() / K);

    weight_data_quantize_scales.create(block_count, N);
    weight_data_quantized.create(packed_k_bytes, N, (size_t)1u);
    if (weight_data_quantize_scales.empty() || weight_data_quantized.empty())
        return -100;

    memset(weight_data_quantized.data, 0, weight_data_quantized.total() * weight_data_quantized.elemsize);

    #pragma omp parallel for num_threads(num_threads)
    for (int n = 0; n < N; n++)
    {
        const float* ptr = weight_data.row(n);
        float* scale_ptr = weight_data_quantize_scales.row(n);
        for (int b = 0; b < block_count; b++)
        {
            const int k0 = b * block_size;
            const int max_kk = std::min(block_size, K - k0);
            scale_ptr[b] = choose_weight_scale(ptr + k0, max_kk, weight_bits, LLM_QUANT_METHOD_MINMAX);
        }
    }

    std::vector<double> H((size_t)K * K, 0.0);

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < K; i++)
    {
        for (int j = i; j < K; j++)
        {
            double v = 0.0;
            for (int s = 0; s < sample_count; s++)
            {
                const float* x = &stat.samples[(size_t)s * K];
                v += (double)x[i] * x[j];
            }

            H[(size_t)i * K + j] = v;
        }
    }

    double diag_mean = 0.0;
    for (int i = 0; i < K; i++)
        diag_mean += H[(size_t)i * K + i];
    diag_mean /= K;

    for (int i = 0; i < K; i++)
        H[(size_t)i * K + i] += damp * diag_mean;

    if (cholesky_inverse_gptq_hessian(H, K) != 0)
    {
        fprintf(stderr, "gptq hessian decomposition failed\n");
        return -1;
    }

    // H is the upper cholesky factor of the inverse hessian
    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<double> work(K);
        std::vector<double> error(block_size);

        #pragma omp for
        for (int n = 0; n < N; n++)
        {
            const float* ptr = weight_data.row(n);
            for (int k = 0; k < K; k++)
                work[k] = ptr[k];

            const float* scale_ptr = weight_data_quantize_scales.row(n);
            unsigned char* qptr = weight_data_quantized.row<unsigned char>(n);

            for (int k0 = 0; k0 < K; k0 += block_size)
            {
                const int max_kk = std::min(block_size, K - k0);
                const int k1 = k0 + max_kk;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    const int k = k0 + kk;
                    const float scale = scale_ptr[k / block_size];
                    const double w = work[k];
                    const int q = float2int_weight((float)(w * scale), weight_bits);
                    const double deq = (double)q / scale;
                    const double err = (w - deq) / H[(size_t)k * K + k];

                    pack_signed_weight(qptr, k, weight_bits, q);
                    error[kk] = err;

                    for (int j = k; j < k1; j++)
                        work[j] -= err * H[(size_t)k * K + j];
                }

                for (int j = k1; j < K; j++)
                {
                    double err = 0.0;
                    for (int kk = 0; kk < max_kk; kk++)
                        err += error[kk] * H[(size_t)(k0 + kk) * K + j];

                    work[j] -= err;
                }
            }
        }
    }

    return 0;
}

int QuantNet::init(int method)
{
    if (method != LLM_QUANT_METHOD_AWQ && method != LLM_QUANT_METHOD_GPTQ)
        return 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "Input")
            input_blobs.push_back(layer->tops[0]);
    }

    const int max_sample_count = method == LLM_QUANT_METHOD_AWQ ? awq_samples : gptq_samples;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type == "Gemm")
        {
            const ncnn::Gemm* gemm = (const ncnn::Gemm*)layers[i];

            if (!is_supported_llm_gemm(gemm))
                continue;

            QuantActStat stat;
            int ret = init_act_stat(stat, gemm->constantK, max_sample_count);
            if (ret != 0)
                return ret;

            gemm_act_stats.push_back(stat);
        }
        else if (layers[i]->type == "MultiHeadAttention")
        {
            const ncnn::MultiHeadAttention* mha = (const ncnn::MultiHeadAttention*)layers[i];

            if (!is_supported_llm_multiheadattention(mha))
                continue;

            const size_t mha_act_offset = mha_act_stats.size();
            mha_act_stats.resize(mha_act_offset + 4);

            const int qdim = mha->weight_data_size / mha->embed_dim;
            const int Ks[4] = {qdim, mha->kdim, mha->vdim, mha->embed_dim};

            for (int j = 0; j < 4; j++)
            {
                int ret = init_act_stat(mha_act_stats[mha_act_offset + j], Ks[j], max_sample_count);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int check_calibration_input(const QuantNet& table)
{
    const int input_blob_count = (int)table.input_blobs.size();
    if (input_blob_count == 0)
    {
        fprintf(stderr, "no input blob found\n");
        return -1;
    }

    if ((int)table.listspaths.size() != input_blob_count)
    {
        fprintf(stderr, "calibration list count mismatch expected=%d got=%d\n", input_blob_count, (int)table.listspaths.size());
        return -1;
    }

    if ((int)table.shapes.size() != input_blob_count)
    {
        fprintf(stderr, "shape count mismatch expected=%d got=%d\n", input_blob_count, (int)table.shapes.size());
        return -1;
    }

    if (table.file_type != 1)
    {
        fprintf(stderr, "ncnnllm2table calibration supports type=1 only\n");
        return -1;
    }

    if (table.listspaths[0].empty())
    {
        fprintf(stderr, "calibration list is empty\n");
        return -1;
    }

    const int file_count = (int)table.listspaths[0].size();
    for (int i = 1; i < input_blob_count; i++)
    {
        if ((int)table.listspaths[i].size() != file_count)
        {
            fprintf(stderr, "calibration list size mismatch input=%d expected=%d got=%d\n", i, file_count, (int)table.listspaths[i].size());
            return -1;
        }
    }

    return 0;
}

int QuantNet::collect_activation_stats()
{
    if (check_calibration_input(*this) != 0)
        return -1;

    const int input_blob_count = (int)input_blobs.size();
    const int file_count = (int)listspaths[0].size();

    for (int i = 0; i < file_count; i++)
    {
        if (i % 100 == 0)
            fprintf(stderr, "collect_activation_stats %.2f%% [ %d / %d ]\n", i * 100.f / file_count, i, file_count);

        ncnn::Extractor ex = create_extractor();
        ex.set_light_mode(false);

        for (int j = 0; j < input_blob_count; j++)
        {
            ncnn::Mat in = read_npy(shapes[j], listspaths[j][i]);
            if (in.empty())
                return -1;

            ex.input(input_blobs[j], in);
        }

        size_t gemm_act_index = 0;
        size_t mha_act_index = 0;

        for (size_t j = 0; j < layers.size(); j++)
        {
            if (layers[j]->type == "Gemm")
            {
                const ncnn::Gemm* gemm = (const ncnn::Gemm*)layers[j];

                if (!is_supported_llm_gemm(gemm))
                    continue;

                ncnn::Mat bottom_blob;
                ex.extract(gemm->bottoms[0], bottom_blob);

                collect_act_rows(gemm_act_stats[gemm_act_index], bottom_blob);

                gemm_act_index++;
            }
            else if (layers[j]->type == "MultiHeadAttention")
            {
                const ncnn::MultiHeadAttention* mha = (const ncnn::MultiHeadAttention*)layers[j];

                if (!is_supported_llm_multiheadattention(mha))
                    continue;

                int q_blob_i = 0;
                int k_blob_i = 0;
                int v_blob_i = 0;
                int attn_mask_i = 0;
                int cached_xk_i = 0;
                int cached_xv_i = 0;
                resolve_mha_bottom_blob_index(mha, (int)mha->bottoms.size(), q_blob_i, k_blob_i, v_blob_i, attn_mask_i, cached_xk_i, cached_xv_i);

                ncnn::Mat q_blob;
                ncnn::Mat k_blob;
                ncnn::Mat v_blob;
                ncnn::Mat attn_mask_blob;
                ncnn::Mat cached_xk_blob;
                ncnn::Mat cached_xv_blob;

                ex.extract(mha->bottoms[q_blob_i], q_blob);
                ex.extract(mha->bottoms[k_blob_i], k_blob);
                ex.extract(mha->bottoms[v_blob_i], v_blob);
                if (mha->attn_mask)
                    ex.extract(mha->bottoms[attn_mask_i], attn_mask_blob);
                if (mha->kv_cache)
                {
                    ex.extract(mha->bottoms[cached_xk_i], cached_xk_blob);
                    ex.extract(mha->bottoms[cached_xv_i], cached_xv_blob);
                }

                const size_t mha_act_offset = mha_act_index * 4;
                collect_act_rows(mha_act_stats[mha_act_offset], q_blob);
                collect_act_rows(mha_act_stats[mha_act_offset + 1], k_blob);
                collect_act_rows(mha_act_stats[mha_act_offset + 2], v_blob);

                int ret = collect_mha_out_act_rows(mha, q_blob, k_blob, v_blob, attn_mask_blob, cached_xk_blob, cached_xv_blob, q_blob_i, k_blob_i, v_blob_i, mha_act_stats[mha_act_offset + 3], quantize_num_threads);
                if (ret != 0)
                    return ret;

                mha_act_index++;
            }
        }
    }

    fprintf(stderr, "collect_activation_stats 100.00%% [ %d / %d ]\n", file_count, file_count);

    return 0;
}

int QuantNet::save_table(const char* tablepath, int block_size, int weight_bits, int method) const
{
    FILE* fp = fopen(tablepath, "wb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", tablepath);
        return -1;
    }

    int table_count = 0;
    size_t gemm_act_index = 0;
    size_t mha_act_index = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type == "Gemm")
        {
            const ncnn::Gemm* gemm = (const ncnn::Gemm*)layers[i];

            if (!is_supported_llm_gemm(gemm))
            {
                print_skip_gemm(gemm);
                continue;
            }

            char key[256];
            snprintf(key, 256, "%s_param_1", gemm_name(gemm));

            ncnn::Mat B_data_quantize_scales;
            ncnn::Mat B_data_input_scales;
            ncnn::Mat B_data_quantized;
            int ret = 0;
            if (method == LLM_QUANT_METHOD_AWQ)
            {
                ret = make_awq_input_scales(gemm->B_data, gemm_act_stats[gemm_act_index], block_size, weight_bits, awq_inner_method, awq_steps, awq_max_scale, B_data_input_scales, B_data_quantize_scales, quantize_num_threads);
            }
            else if (method == LLM_QUANT_METHOD_GPTQ)
            {
                ret = make_gptq_qweight(gemm->B_data, gemm_act_stats[gemm_act_index], block_size, weight_bits, gptq_damp, B_data_quantize_scales, B_data_quantized, quantize_num_threads);
            }
            else
            {
                ret = make_weight_scales(gemm->B_data, block_size, weight_bits, method, B_data_quantize_scales, quantize_num_threads);
            }
            if (ret != 0)
            {
                fclose(fp);
                return ret;
            }

            if (method == LLM_QUANT_METHOD_GPTQ)
            {
                std::string qweight_path;
                const std::string qweight_name = make_qweight_filename(tablepath, key, qweight_path);
                ret = write_raw_mat_file(qweight_path.c_str(), B_data_quantized);
                if (ret != 0)
                {
                    fclose(fp);
                    return ret;
                }

                if (write_llm_qweight_table_row(fp, key, weight_bits, block_size, method, qweight_name.c_str(), B_data_quantize_scales) != 0)
                {
                    fclose(fp);
                    return -1;
                }
            }
            else if (write_llm_table_row(fp, key, weight_bits, block_size, method, B_data_quantize_scales) != 0)
            {
                fclose(fp);
                return -1;
            }

            fprintf(stderr, "write_llm_table %s\n", key);
            table_count++;

            if (method == LLM_QUANT_METHOD_AWQ)
            {
                char input_scale_key[512];
                snprintf(input_scale_key, 512, "%s_input_scale", key);

                if (write_llm_input_scale_row(fp, input_scale_key, method, B_data_input_scales) != 0)
                {
                    fclose(fp);
                    return -1;
                }

                fprintf(stderr, "write_llm_table %s\n", input_scale_key);
                table_count++;
            }

            gemm_act_index++;
        }
        else if (layers[i]->type == "MultiHeadAttention")
        {
            const ncnn::MultiHeadAttention* mha = (const ncnn::MultiHeadAttention*)layers[i];

            if (!is_supported_llm_multiheadattention(mha))
            {
                print_skip_multiheadattention(mha);
                continue;
            }

            const int qdim = mha->weight_data_size / mha->embed_dim;

            const ncnn::Mat q_weight_data = mha->q_weight_data.reshape(qdim, mha->embed_dim);
            const ncnn::Mat k_weight_data = mha->k_weight_data.reshape(mha->kdim, mha->embed_dim);
            const ncnn::Mat v_weight_data = mha->v_weight_data.reshape(mha->vdim, mha->embed_dim);
            const ncnn::Mat out_weight_data = mha->out_weight_data.reshape(mha->embed_dim, qdim);

            const ncnn::Mat weights[4] = {q_weight_data, k_weight_data, v_weight_data, out_weight_data};

            const size_t mha_act_offset = mha_act_index * 4;
            for (int j = 0; j < 4; j++)
            {
                char key[256];
                snprintf(key, 256, "%s_param_%d", multiheadattention_name(mha), j);

                ncnn::Mat weight_data_quantize_scales;
                ncnn::Mat weight_data_input_scales;
                ncnn::Mat weight_data_quantized;
                int ret = 0;
                if (method == LLM_QUANT_METHOD_AWQ)
                {
                    ret = make_awq_input_scales(weights[j], mha_act_stats[mha_act_offset + j], block_size, weight_bits, awq_inner_method, awq_steps, awq_max_scale, weight_data_input_scales, weight_data_quantize_scales, quantize_num_threads);
                }
                else if (method == LLM_QUANT_METHOD_GPTQ)
                {
                    ret = make_gptq_qweight(weights[j], mha_act_stats[mha_act_offset + j], block_size, weight_bits, gptq_damp, weight_data_quantize_scales, weight_data_quantized, quantize_num_threads);
                }
                else
                {
                    ret = make_weight_scales(weights[j], block_size, weight_bits, method, weight_data_quantize_scales, quantize_num_threads);
                }
                if (ret != 0)
                {
                    fclose(fp);
                    return ret;
                }

                if (method == LLM_QUANT_METHOD_GPTQ)
                {
                    std::string qweight_path;
                    const std::string qweight_name = make_qweight_filename(tablepath, key, qweight_path);
                    ret = write_raw_mat_file(qweight_path.c_str(), weight_data_quantized);
                    if (ret != 0)
                    {
                        fclose(fp);
                        return ret;
                    }

                    if (write_llm_qweight_table_row(fp, key, weight_bits, block_size, method, qweight_name.c_str(), weight_data_quantize_scales) != 0)
                    {
                        fclose(fp);
                        return -1;
                    }
                }
                else if (write_llm_table_row(fp, key, weight_bits, block_size, method, weight_data_quantize_scales) != 0)
                {
                    fclose(fp);
                    return -1;
                }

                fprintf(stderr, "write_llm_table %s\n", key);
                table_count++;

                if (method == LLM_QUANT_METHOD_AWQ)
                {
                    char input_scale_key[512];
                    snprintf(input_scale_key, 512, "%s_input_scale", key);

                    if (write_llm_input_scale_row(fp, input_scale_key, method, weight_data_input_scales) != 0)
                    {
                        fclose(fp);
                        return -1;
                    }

                    fprintf(stderr, "write_llm_table %s\n", input_scale_key);
                    table_count++;
                }
            }

            mha_act_index++;
        }
    }

    fclose(fp);

    fprintf(stderr, "ncnn llm quant table created %d rows\n", table_count);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        show_usage(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            show_usage(argv[0]);
            return -1;
        }
    }

    char* inparam = argv[1];
    char* inbin = argv[2];
    char* caliblist = 0;
    char* outtable = 0;
    int kv_start = 4;

    if (argc >= 5 && strchr(argv[3], '=') == NULL && strchr(argv[4], '=') == NULL)
    {
        caliblist = argv[3];
        outtable = argv[4];
        kv_start = 5;
    }
    else
    {
        outtable = argv[3];
        kv_start = 4;
    }

    int weight_bits = 6;
    int block_size = 64;
    const char* method = "minmax";
    const char* awq_inner = "minmax";
    int thread = 1;

    QuantNet table;
    table.use_calibration_dataset = caliblist != 0;
    if (caliblist)
        table.listspaths = parse_comma_path_list(caliblist);

    for (int i = kv_start; i < argc; i++)
    {
        char* kv = argv[i];
        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            return -1;
        }

        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "method") == 0)
            method = value;
        else if (strcmp(key, "bits") == 0)
            weight_bits = atoi(value);
        else if (strcmp(key, "block") == 0)
            block_size = atoi(value);
        else if (strcmp(key, "thread") == 0)
            thread = atoi(value);
        else if (strcmp(key, "type") == 0)
            table.file_type = atoi(value);
        else if (strcmp(key, "shape") == 0)
            table.shapes = parse_comma_int_array_list(value);
        else if (strcmp(key, "awq_steps") == 0)
            table.awq_steps = atoi(value);
        else if (strcmp(key, "awq_samples") == 0)
            table.awq_samples = atoi(value);
        else if (strcmp(key, "awq_max_scale") == 0)
            table.awq_max_scale = (float)atof(value);
        else if (strcmp(key, "awq_inner") == 0)
            awq_inner = value;
        else if (strcmp(key, "gptq_samples") == 0)
            table.gptq_samples = atoi(value);
        else if (strcmp(key, "gptq_damp") == 0)
            table.gptq_damp = (float)atof(value);
        else
        {
            fprintf(stderr, "unrecognized arg %s\n", key);
            return -1;
        }
    }

    const int quantize_method = llm_quant_method_from_string(method);
    if (quantize_method < 0)
    {
        fprintf(stderr, "unsupported method=%s\n", method);
        return -1;
    }

    if ((quantize_method == LLM_QUANT_METHOD_AWQ || quantize_method == LLM_QUANT_METHOD_GPTQ) && !table.use_calibration_dataset)
    {
        fprintf(stderr, "method=%s requires calibration list\n", method);
        return -1;
    }

    if (quantize_method != LLM_QUANT_METHOD_AWQ && quantize_method != LLM_QUANT_METHOD_GPTQ && table.use_calibration_dataset)
    {
        fprintf(stderr, "calibration list is only used by method=awq/gptq\n");
        return -1;
    }

    if (llm_weight_block_quantize_term(weight_bits, block_size) == 0)
    {
        show_usage(argv[0]);
        return -1;
    }

    if (thread < 1)
    {
        fprintf(stderr, "malformed thread %d\n", thread);
        return -1;
    }

    if (quantize_method == LLM_QUANT_METHOD_AWQ)
    {
        table.awq_inner_method = llm_quant_method_from_string(awq_inner);
        if (table.awq_inner_method != LLM_QUANT_METHOD_MINMAX && table.awq_inner_method != LLM_QUANT_METHOD_MSECLIP)
        {
            fprintf(stderr, "unsupported awq_inner=%s\n", awq_inner);
            return -1;
        }

        if (table.awq_steps < 0)
        {
            fprintf(stderr, "malformed awq_steps %d\n", table.awq_steps);
            return -1;
        }

        if (table.awq_samples < 1)
        {
            fprintf(stderr, "malformed awq_samples %d\n", table.awq_samples);
            return -1;
        }

        if (!(table.awq_max_scale > 1.f))
        {
            fprintf(stderr, "malformed awq_max_scale %f\n", table.awq_max_scale);
            return -1;
        }
    }

    if (quantize_method == LLM_QUANT_METHOD_GPTQ)
    {
        if (table.gptq_samples < 1)
        {
            fprintf(stderr, "malformed gptq_samples %d\n", table.gptq_samples);
            return -1;
        }

        if (!(table.gptq_damp >= 0.f))
        {
            fprintf(stderr, "malformed gptq_damp %f\n", table.gptq_damp);
            return -1;
        }
    }

    table.quantize_num_threads = thread;
    table.opt.num_threads = thread;
    table.opt.use_packing_layout = false;
    table.opt.use_fp16_packed = false;
    table.opt.use_fp16_storage = false;
    table.opt.use_fp16_arithmetic = false;
    table.opt.use_bf16_storage = false;

    if (table.load_param(inparam) != 0)
        return -1;

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        if (table.load_model(dr) != 0)
            return -1;
    }
    else
    {
        if (table.load_model(inbin) != 0)
            return -1;
    }

    if (table.init(quantize_method) != 0)
        return -1;

    if (quantize_method == LLM_QUANT_METHOD_AWQ || quantize_method == LLM_QUANT_METHOD_GPTQ)
    {
        if (table.collect_activation_stats() != 0)
            return -1;
    }

    if (table.save_table(outtable, block_size, weight_bits, quantize_method) != 0)
        return -1;

    return 0;
}
