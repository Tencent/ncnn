// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// yolo11 vulkan preprocess + postprocess example
// demonstrate:
// 1. custom vulkan operator with inline GLSL compute shader
// 2. net vulkan zero-copy via VkMat chaining
// 3. GLSL-based image preprocessing (BGR->RGB + bilinear resize + normalize + letterbox)
// 4. GLSL-based postprocessing (DFL decode + bbox decode + parallel NMS)

#include "layer.h"
#include "net.h"

#include "gpu.h"
#include "pipeline.h"
#include "command.h"
#include "allocator.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include <string.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// custom vulkan layer: YoloPreprocess
// performs BGR->RGB + bilinear resize + normalize(1/255) + letterbox pad(114/255) on GPU
class YoloPreprocess : public ncnn::Layer
{
public:
    YoloPreprocess();

    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);

    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

public:
    // runtime letterbox geometry
    int target_size;
    float scale;
    int pad_left;
    int pad_top;
    int dst_w;
    int dst_h;
    int src_w;
    int src_h;
    int stride;

private:
    ncnn::Pipeline* pipeline_preprocess;
};

DEFINE_LAYER_CREATOR(YoloPreprocess)

YoloPreprocess::YoloPreprocess()
{
    support_vulkan = true;
    one_blob_only = true;

    pipeline_preprocess = 0;

    target_size = 640;
    scale = 1.f;
    pad_left = 0;
    pad_top = 0;
    dst_w = 0;
    dst_h = 0;
    src_w = 0;
    src_h = 0;
    stride = 0;
}

// GLSL compute shader for yolo preprocess
// input:  interleaved BGR uint8 raw bytes (binding 0)  -- read via uint[] + bit-shift
// output: planar RGB sfp (binding 1)  -- sfp auto-adapts to fp32/fp16/bf16 storage
// uses afp for arithmetic precision, buffer_st1 for storage write
static const char yolo_preprocess_comp[] = R"(
#version 450

layout (push_constant) uniform parameter
{
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
    int pad_left;
    int pad_top;
    float scale;
    int stride;     // bytes per row of source image
    int dst_cstep;
} p;

layout (binding = 0) readonly buffer src_blob { uint src_data[]; };
layout (binding = 1) writeonly buffer dst_blob { sfp dst_data[]; };

// read one byte from the uint-packed buffer (little-endian host layout)
uint read_u8(int byte_idx)
{
    int word_idx = byte_idx / 4;
    int byte_offset = byte_idx % 4;
    return (src_data[word_idx] >> (byte_offset * 8)) & 0xFFu;
}

void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    if (gx >= p.dst_w || gy >= p.dst_h)
        return;

    int dst_idx = gy * p.dst_w + gx;

    afp r, g, b;

    int resize_w = int(float(p.src_w) * p.scale);
    int resize_h = int(float(p.src_h) * p.scale);

    if (gx < p.pad_left || gx >= p.pad_left + resize_w ||
        gy < p.pad_top  || gy >= p.pad_top  + resize_h)
    {
        r = afp(114.0 / 255.0);
        g = afp(114.0 / 255.0);
        b = afp(114.0 / 255.0);
    }
    else
    {
        float src_x = (float(gx - p.pad_left) + 0.5f) / p.scale - 0.5f;
        float src_y = (float(gy - p.pad_top)  + 0.5f) / p.scale - 0.5f;

        int x0 = int(floor(src_x));
        int y0 = int(floor(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = src_x - float(x0);
        float fy = src_y - float(y0);

        x0 = clamp(x0, 0, p.src_w - 1);
        y0 = clamp(y0, 0, p.src_h - 1);
        x1 = clamp(x1, 0, p.src_w - 1);
        y1 = clamp(y1, 0, p.src_h - 1);

        int y0_offset = y0 * p.stride;
        int y1_offset = y1 * p.stride;

        // B channel
        float b00 = float(read_u8(y0_offset + x0 * 3 + 0));
        float b01 = float(read_u8(y0_offset + x1 * 3 + 0));
        float b10 = float(read_u8(y1_offset + x0 * 3 + 0));
        float b11 = float(read_u8(y1_offset + x1 * 3 + 0));
        float bf = mix(mix(b00, b01, fx), mix(b10, b11, fx), fy) / 255.0;

        // G channel
        float g00 = float(read_u8(y0_offset + x0 * 3 + 1));
        float g01 = float(read_u8(y0_offset + x1 * 3 + 1));
        float g10 = float(read_u8(y1_offset + x0 * 3 + 1));
        float g11 = float(read_u8(y1_offset + x1 * 3 + 1));
        float gf = mix(mix(g00, g01, fx), mix(g10, g11, fx), fy) / 255.0;

        // R channel
        float r00 = float(read_u8(y0_offset + x0 * 3 + 2));
        float r01 = float(read_u8(y0_offset + x1 * 3 + 2));
        float r10 = float(read_u8(y1_offset + x0 * 3 + 2));
        float r11 = float(read_u8(y1_offset + x1 * 3 + 2));
        float rf = mix(mix(r00, r01, fx), mix(r10, r11, fx), fy) / 255.0;

        // BGR -> RGB
        r = afp(rf);
        g = afp(gf);
        b = afp(bf);
    }

    buffer_st1(dst_data, dst_idx + 0 * p.dst_cstep, r);
    buffer_st1(dst_data, dst_idx + 1 * p.dst_cstep, g);
    buffer_st1(dst_data, dst_idx + 2 * p.dst_cstep, b);
}
)";

int YoloPreprocess::create_pipeline(const ncnn::Option& opt)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(yolo_preprocess_comp, (int)strlen(yolo_preprocess_comp), opt, spirv);
    if (ret != 0)
    {
        NCNN_LOGE("compile_spirv_module failed %d", ret);
        return -1;
    }

    pipeline_preprocess = new ncnn::Pipeline(vkdev);
    pipeline_preprocess->set_optimal_local_size_xyz(8, 8, 1);
    pipeline_preprocess->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());

    return 0;
}

int YoloPreprocess::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_preprocess;
    pipeline_preprocess = 0;

    return 0;
}

int YoloPreprocess::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    int elempack = 1;
    size_t elemsize = (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
                       ? elempack * 2u : elempack * 4u;

    top_blob.create(dst_w, dst_h, 3, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    int dst_cstep = (int)ncnn::alignSize(dst_w * dst_h * (int)elemsize, 16) / (int)elemsize;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<ncnn::vk_constant_type> constants(9);
    constants[0].i = src_w;
    constants[1].i = src_h;
    constants[2].i = dst_w;
    constants[3].i = dst_h;
    constants[4].i = pad_left;
    constants[5].i = pad_top;
    constants[6].f = scale;
    constants[7].i = stride;
    constants[8].i = dst_cstep;

    ncnn::VkMat dispatcher;
    dispatcher.w = dst_w;
    dispatcher.h = dst_h;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_preprocess, bindings, constants, dispatcher);

    return 0;
}

// YoloPostprocess: GPU-based post-processing for YOLO11
// 1. generate proposals from model output (softmax DFL + bbox decode + coordinate transform)
// 2. parallel NMS (each thread checks all higher-score boxes)
class YoloPostprocess
{
public:
    YoloPostprocess();

    int create_pipeline(ncnn::VulkanDevice* _vkdev, const ncnn::Option& opt);
    int destroy_pipeline(const ncnn::Option& opt);

    int generate(const ncnn::VkMat& pred, ncnn::VkMat& proposals, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
    int nms(const ncnn::VkMat& proposals, ncnn::VkMat& picked, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

    float prob_threshold;
    float nms_threshold;
    int num_class;
    int num_anchor;
    int img_w, img_h;
    int pad_left, pad_top;
    float scale;

private:
    ncnn::VulkanDevice* vkdev;
public:
    ncnn::Pipeline* pipeline_generate_pack1;
    ncnn::Pipeline* pipeline_generate_pack4;
    ncnn::Pipeline* pipeline_nms;
};

// GLSL compute shader for generating YOLO proposals (pack1)
// input:  model pred output 144-dim per anchor (binding 0)  -- sfp for ncnn storage compat
// output: proposals float6 per anchor (x0,y0,x1,y1,score,label) (binding 1)
static const char yolo_generate_comp_pack1[] = R"(
#version 450

layout (push_constant) uniform parameter
{
    int num_anchor;
    int num_class;
    float prob_threshold;
    int img_w;
    int img_h;
    int pad_left;
    int pad_top;
    float scale;
    int grid0;
    int grid1;
    int grid2;
    int stride0;
    int stride1;
    int stride2;
} p;

layout (binding = 0) readonly buffer pred_blob { sfp pred_data[]; };
layout (binding = 1) writeonly buffer proposals_blob { float proposals_data[]; };

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= p.num_anchor)
        return;

    int grid0_sq = p.grid0 * p.grid0;
    int grid01_sq = grid0_sq + p.grid1 * p.grid1;

    int stride;
    int grid_x, grid_y;
    if (idx < grid0_sq)
    {
        stride = p.stride0;
        grid_x = idx % p.grid0;
        grid_y = idx / p.grid0;
    }
    else if (idx < grid01_sq)
    {
        int idx2 = idx - grid0_sq;
        stride = p.stride1;
        grid_x = idx2 % p.grid1;
        grid_y = idx2 / p.grid1;
    }
    else
    {
        int idx2 = idx - grid01_sq;
        stride = p.stride2;
        grid_x = idx2 % p.grid2;
        grid_y = idx2 / p.grid2;
    }

    int base = idx * 144;

    // find max class score
    afp max_score = afp(-1e9);
    int label = -1;
    for (int c = 0; c < p.num_class; c++)
    {
        afp s = buffer_ld1(pred_data, base + 64 + c);
        if (s > max_score)
        {
            max_score = s;
            label = c;
        }
    }
    afp score = afp(1.0) / (afp(1.0) + exp(-max_score));

    float x0, y0, x1, y1;
    if (float(score) < p.prob_threshold)
    {
        x0 = y0 = x1 = y1 = 0.0;
        score = afp(0.0);
        label = -1;
    }
    else
    {
        // DFL softmax for l, t, r, b
        afp l = afp(0.0), t = afp(0.0), r = afp(0.0), b = afp(0.0);
        for (int k = 0; k < 4; k++)
        {
            afp vals[16];
            afp maxv = afp(-1e9);
            int offset = base + k * 16;
            for (int i = 0; i < 16; i++)
            {
                vals[i] = buffer_ld1(pred_data, offset + i);
                maxv = max(maxv, vals[i]);
            }
            afp sum = afp(0.0);
            for (int i = 0; i < 16; i++)
            {
                vals[i] = exp(vals[i] - maxv);
                sum += vals[i];
            }
            afp expect = afp(0.0);
            for (int i = 0; i < 16; i++)
            {
                expect += afp(float(i)) * vals[i] / sum;
            }

            if (k == 0) l = expect * afp(float(stride));
            else if (k == 1) t = expect * afp(float(stride));
            else if (k == 2) r = expect * afp(float(stride));
            else b = expect * afp(float(stride));
        }

        float pb_cx = (float(grid_x) + 0.5) * float(stride);
        float pb_cy = (float(grid_y) + 0.5) * float(stride);

        x0 = (pb_cx - float(l) - float(p.pad_left)) / p.scale;
        y0 = (pb_cy - float(t) - float(p.pad_top)) / p.scale;
        x1 = (pb_cx + float(r) - float(p.pad_left)) / p.scale;
        y1 = (pb_cy + float(b) - float(p.pad_top)) / p.scale;

        // clip to original image
        x0 = clamp(x0, 0.0, float(p.img_w - 1));
        y0 = clamp(y0, 0.0, float(p.img_h - 1));
        x1 = clamp(x1, 0.0, float(p.img_w - 1));
        y1 = clamp(y1, 0.0, float(p.img_h - 1));
    }

    proposals_data[idx * 6 + 0] = x0;
    proposals_data[idx * 6 + 1] = y0;
    proposals_data[idx * 6 + 2] = x1;
    proposals_data[idx * 6 + 3] = y1;
    proposals_data[idx * 6 + 4] = float(score);
    proposals_data[idx * 6 + 5] = float(label);
}
)";

// GLSL compute shader for generating YOLO proposals (pack4)
// input:  model pred output 144-dim per anchor (binding 0)  -- sfp for ncnn storage compat
// output: proposals float6 per anchor (x0,y0,x1,y1,score,label) (binding 1)
static const char yolo_generate_comp_pack4[] = R"(
#version 450

layout (push_constant) uniform parameter
{
    int num_anchor;
    int num_class;
    float prob_threshold;
    int img_w;
    int img_h;
    int pad_left;
    int pad_top;
    float scale;
    int grid0;
    int grid1;
    int grid2;
    int stride0;
    int stride1;
    int stride2;
} p;

layout (binding = 0) readonly buffer pred_blob { sfp pred_data[]; };
layout (binding = 1) writeonly buffer proposals_blob { float proposals_data[]; };

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= p.num_anchor)
        return;

    int grid0_sq = p.grid0 * p.grid0;
    int grid01_sq = grid0_sq + p.grid1 * p.grid1;

    int stride;
    int grid_x, grid_y;
    if (idx < grid0_sq)
    {
        stride = p.stride0;
        grid_x = idx % p.grid0;
        grid_y = idx / p.grid0;
    }
    else if (idx < grid01_sq)
    {
        int idx2 = idx - grid0_sq;
        stride = p.stride1;
        grid_x = idx2 % p.grid1;
        grid_y = idx2 / p.grid1;
    }
    else
    {
        int idx2 = idx - grid01_sq;
        stride = p.stride2;
        grid_x = idx2 % p.grid2;
        grid_y = idx2 / p.grid2;
    }

    int comp = idx % 4;
    int packed_idx = idx / 4;
    int base = packed_idx * 576;

    // find max class score
    afp max_score = afp(-1e9);
    int label = -1;
    for (int c = 0; c < p.num_class; c++)
    {
        afp s = buffer_ld1(pred_data, base + (64 + c) * 4 + comp);
        if (s > max_score)
        {
            max_score = s;
            label = c;
        }
    }
    afp score = afp(1.0) / (afp(1.0) + exp(-max_score));

    float x0, y0, x1, y1;
    if (float(score) < p.prob_threshold)
    {
        x0 = y0 = x1 = y1 = 0.0;
        score = afp(0.0);
        label = -1;
    }
    else
    {
        // DFL softmax for l, t, r, b
        afp l = afp(0.0), t = afp(0.0), r = afp(0.0), b = afp(0.0);
        for (int k = 0; k < 4; k++)
        {
            afp vals[16];
            afp maxv = afp(-1e9);
            int offset = base + k * 64;
            for (int i = 0; i < 16; i++)
            {
                vals[i] = buffer_ld1(pred_data, offset + i * 4 + comp);
                maxv = max(maxv, vals[i]);
            }
            afp sum = afp(0.0);
            for (int i = 0; i < 16; i++)
            {
                vals[i] = exp(vals[i] - maxv);
                sum += vals[i];
            }
            afp expect = afp(0.0);
            for (int i = 0; i < 16; i++)
            {
                expect += afp(float(i)) * vals[i] / sum;
            }

            if (k == 0) l = expect * afp(float(stride));
            else if (k == 1) t = expect * afp(float(stride));
            else if (k == 2) r = expect * afp(float(stride));
            else b = expect * afp(float(stride));
        }

        float pb_cx = (float(grid_x) + 0.5) * float(stride);
        float pb_cy = (float(grid_y) + 0.5) * float(stride);

        x0 = (pb_cx - float(l) - float(p.pad_left)) / p.scale;
        y0 = (pb_cy - float(t) - float(p.pad_top)) / p.scale;
        x1 = (pb_cx + float(r) - float(p.pad_left)) / p.scale;
        y1 = (pb_cy + float(b) - float(p.pad_top)) / p.scale;

        // clip to original image
        x0 = clamp(x0, 0.0, float(p.img_w - 1));
        y0 = clamp(y0, 0.0, float(p.img_h - 1));
        x1 = clamp(x1, 0.0, float(p.img_w - 1));
        y1 = clamp(y1, 0.0, float(p.img_h - 1));
    }

    proposals_data[idx * 6 + 0] = x0;
    proposals_data[idx * 6 + 1] = y0;
    proposals_data[idx * 6 + 2] = x1;
    proposals_data[idx * 6 + 3] = y1;
    proposals_data[idx * 6 + 4] = float(score);
    proposals_data[idx * 6 + 5] = float(label);
}
)";

// GLSL compute shader for parallel NMS
// input:  proposals float6 per anchor (binding 0)
// output: picked int per anchor (1=keep, 0=suppress) (binding 1)
static const char yolo_nms_comp[] = R"(
#version 450

layout (push_constant) uniform parameter
{
    int num_anchor;
    float nms_threshold;
} p;

layout (binding = 0) readonly buffer proposals_blob { float proposals_data[]; };
layout (binding = 1) writeonly buffer picked_blob { int picked_data[]; };

float intersection_area(int i, int j)
{
    float x0 = max(proposals_data[i * 6 + 0], proposals_data[j * 6 + 0]);
    float y0 = max(proposals_data[i * 6 + 1], proposals_data[j * 6 + 1]);
    float x1 = min(proposals_data[i * 6 + 2], proposals_data[j * 6 + 2]);
    float y1 = min(proposals_data[i * 6 + 3], proposals_data[j * 6 + 3]);
    float w = max(x1 - x0, 0.0);
    float h = max(y1 - y0, 0.0);
    return w * h;
}

void main()
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= p.num_anchor)
        return;

    float score_i = proposals_data[i * 6 + 4];
    if (score_i <= 0.0)
    {
        picked_data[i] = 0;
        return;
    }

    float area_i = (proposals_data[i * 6 + 2] - proposals_data[i * 6 + 0])
                 * (proposals_data[i * 6 + 3] - proposals_data[i * 6 + 1]);

    int keep = 1;
    for (int j = 0; j < p.num_anchor; j++)
    {
        if (i == j)
            continue;

        float score_j = proposals_data[j * 6 + 4];
        if (score_j < score_i)
            continue;
        if (score_j == score_i && j >= i)
            continue;

        int label_i = int(proposals_data[i * 6 + 5]);
        int label_j = int(proposals_data[j * 6 + 5]);
        if (label_i != label_j)
            continue;

        float inter = intersection_area(i, j);
        float area_j = (proposals_data[j * 6 + 2] - proposals_data[j * 6 + 0])
                     * (proposals_data[j * 6 + 3] - proposals_data[j * 6 + 1]);
        float union_area = area_i + area_j - inter;

        if (union_area > 0.0 && inter / union_area > p.nms_threshold)
        {
            keep = 0;
            break;
        }
    }

    picked_data[i] = keep;
}
)";

YoloPostprocess::YoloPostprocess()
{
    vkdev = 0;
    pipeline_generate_pack1 = 0;
    pipeline_generate_pack4 = 0;
    pipeline_nms = 0;

    prob_threshold = 0.25f;
    nms_threshold = 0.45f;
    num_class = 80;
    num_anchor = 8400;
    img_w = 0;
    img_h = 0;
    pad_left = 0;
    pad_top = 0;
    scale = 1.f;
}

int YoloPostprocess::create_pipeline(ncnn::VulkanDevice* _vkdev, const ncnn::Option& opt)
{
    vkdev = _vkdev;

    // compile generate shader for pack1
    {
        std::vector<uint32_t> spirv;
        int ret = ncnn::compile_spirv_module(yolo_generate_comp_pack1, (int)strlen(yolo_generate_comp_pack1), opt, spirv);
        if (ret != 0)
        {
            NCNN_LOGE("compile generate pack1 spirv failed %d", ret);
            return -1;
        }
        pipeline_generate_pack1 = new ncnn::Pipeline(vkdev);
        pipeline_generate_pack1->set_optimal_local_size_xyz(256, 1, 1);
        ret = pipeline_generate_pack1->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
        if (ret != 0)
        {
            NCNN_LOGE("pipeline_generate_pack1 create failed %d", ret);
            return ret;
        }
        NCNN_LOGE("pipeline_generate_pack1 created pipeline=%lu", (unsigned long)pipeline_generate_pack1->pipeline());
    }

    // compile generate shader for pack4
    {
        std::vector<uint32_t> spirv;
        int ret = ncnn::compile_spirv_module(yolo_generate_comp_pack4, (int)strlen(yolo_generate_comp_pack4), opt, spirv);
        if (ret != 0)
        {
            NCNN_LOGE("compile generate pack4 spirv failed %d", ret);
            return -1;
        }
        pipeline_generate_pack4 = new ncnn::Pipeline(vkdev);
        pipeline_generate_pack4->set_optimal_local_size_xyz(256, 1, 1);
        ret = pipeline_generate_pack4->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
        if (ret != 0)
        {
            NCNN_LOGE("pipeline_generate_pack4 create failed %d", ret);
            return ret;
        }
        NCNN_LOGE("pipeline_generate_pack4 created pipeline=%lu", (unsigned long)pipeline_generate_pack4->pipeline());
    }

    // compile nms shader
    {
        std::vector<uint32_t> spirv;
        int ret = ncnn::compile_spirv_module(yolo_nms_comp, (int)strlen(yolo_nms_comp), opt, spirv);
        if (ret != 0)
        {
            NCNN_LOGE("compile nms spirv failed %d", ret);
            return -1;
        }
        pipeline_nms = new ncnn::Pipeline(vkdev);
        pipeline_nms->set_optimal_local_size_xyz(256, 1, 1);
        ret = pipeline_nms->create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>());
        if (ret != 0)
        {
            NCNN_LOGE("pipeline_nms create failed %d", ret);
            return ret;
        }
        NCNN_LOGE("pipeline_nms created pipeline=%lu", (unsigned long)pipeline_nms->pipeline());
    }

    return 0;
}

int YoloPostprocess::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_generate_pack1;
    pipeline_generate_pack1 = 0;
    delete pipeline_generate_pack4;
    pipeline_generate_pack4 = 0;
    delete pipeline_nms;
    pipeline_nms = 0;
    return 0;
}

int YoloPostprocess::generate(const ncnn::VkMat& pred, ncnn::VkMat& proposals, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    proposals.create(6, num_anchor, 1, 4u, 1, opt.blob_vkallocator);
    if (proposals.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = pred;
    bindings[1] = proposals;

    std::vector<ncnn::vk_constant_type> constants(14);
    constants[0].i = num_anchor;
    constants[1].i = num_class;
    constants[2].f = prob_threshold;
    constants[3].i = img_w;
    constants[4].i = img_h;
    constants[5].i = pad_left;
    constants[6].i = pad_top;
    constants[7].f = scale;
    constants[8].i = 80;   // grid0
    constants[9].i = 40;   // grid1
    constants[10].i = 20;  // grid2
    constants[11].i = 8;   // stride0
    constants[12].i = 16;  // stride1
    constants[13].i = 32;  // stride2

    ncnn::VkMat dispatcher;
    dispatcher.w = num_anchor;
    dispatcher.h = 1;
    dispatcher.c = 1;

    ncnn::Pipeline* pipeline_generate;
    if (pred.elempack == 1)
        pipeline_generate = pipeline_generate_pack1;
    else if (pred.elempack == 4)
        pipeline_generate = pipeline_generate_pack4;
    else
    {
        NCNN_LOGE("unsupported pred elempack %d", pred.elempack);
        return -1;
    }
    cmd.record_pipeline(pipeline_generate, bindings, constants, dispatcher);
    return 0;
}

int YoloPostprocess::nms(const ncnn::VkMat& proposals, ncnn::VkMat& picked, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    picked.create(1, num_anchor, 1, 4u, 1, opt.blob_vkallocator);
    if (picked.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = proposals;
    bindings[1] = picked;

    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].i = num_anchor;
    constants[1].f = nms_threshold;

    ncnn::VkMat dispatcher;
    dispatcher.w = num_anchor;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_nms, bindings, constants, dispatcher);
    return 0;
}

static int detect_yolo11_vk(const cv::Mat& bgr, std::vector<Object>& objects)
{
    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // ultralytics/cfg/models/v8/yolo11.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;

    int dst_w = w + wpad;
    int dst_h = h + hpad;

    // ===== Vulkan zero-copy preprocess + inference + postprocess =====
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    {
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;

        // step 1~2: upload + preprocess in cmd1
        ncnn::VkCompute cmd1(vkdev);

#if defined(USE_NCNN_SIMPLEOCV)
        int stride = img_w * 3;
        const uchar* bgr_data = bgr.data;
#else
        cv::Mat bgr_cont = bgr.isContinuous() ? bgr : bgr.clone();
        int stride = (int)bgr_cont.step[0];  // bytes per row
        const uchar* bgr_data = bgr_cont.data;
#endif

        // manually upload uint8 data via staging buffer to avoid ncnn's convert_packing
        // (convert_packing does not support int8 -> fp16/fp32)
        ncnn::VkMat staging_vkmat;
        staging_vkmat.create(stride, img_h, 1, 1u, 1, opt.staging_vkallocator);
        memcpy(staging_vkmat.mapped_ptr(), bgr_data, stride * img_h);
        staging_vkmat.allocator->flush(staging_vkmat.data);
        staging_vkmat.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        staging_vkmat.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        ncnn::VkMat in_vkmat;
        cmd1.record_clone(staging_vkmat, in_vkmat, opt);

        YoloPreprocess preprocess_layer;
        preprocess_layer.vkdev = vkdev;
        preprocess_layer.target_size = target_size;
        preprocess_layer.scale = scale;
        preprocess_layer.pad_left = wpad / 2;
        preprocess_layer.pad_top = hpad / 2;
        preprocess_layer.dst_w = dst_w;
        preprocess_layer.dst_h = dst_h;
        preprocess_layer.src_w = img_w;
        preprocess_layer.src_h = img_h;
        preprocess_layer.stride = stride;
        int ret = preprocess_layer.create_pipeline(opt);
        if (ret != 0)
        {
            NCNN_LOGE("preprocess create_pipeline failed %d", ret);
            return -1;
        }

        ncnn::VkMat pre_vkmat;
        ret = preprocess_layer.forward(in_vkmat, pre_vkmat, cmd1, opt);
        if (ret != 0)
        {
            NCNN_LOGE("preprocess forward failed %d", ret);
            return -1;
        }
        cmd1.submit_and_wait();
        fprintf(stderr, "pre_vkmat: w=%d h=%d c=%d elemsize=%zu elempack=%d\n",
                pre_vkmat.w, pre_vkmat.h, pre_vkmat.c, pre_vkmat.elemsize, pre_vkmat.elempack);

        // step 3~5: inference + postprocess + download in cmd2
        ncnn::Net yolo11;
        yolo11.opt.use_vulkan_compute = true;
        yolo11.opt.blob_vkallocator = blob_vkallocator;
        yolo11.opt.workspace_vkallocator = blob_vkallocator;
        yolo11.opt.staging_vkallocator = staging_vkallocator;

        yolo11.load_param("yolo11n.ncnn.param");
        yolo11.load_model("yolo11n.ncnn.bin");

        ncnn::Extractor ex = yolo11.create_extractor();
        ex.input("in0", pre_vkmat);

        ncnn::VkCompute cmd2(vkdev);

        // model inference
        ncnn::VkMat out_vkmat;
        ex.extract("out0", out_vkmat, cmd2);
        fprintf(stderr, "out_vkmat from extract: w=%d h=%d c=%d elemsize=%zu elempack=%d\n",
                out_vkmat.w, out_vkmat.h, out_vkmat.c, out_vkmat.elemsize, out_vkmat.elempack);

        // GPU postprocess (generate proposals + NMS)
        // shader uses sfp/buffer_ld1 for pred_data, supports both pack1 and pack4
        YoloPostprocess postprocess;
        ret = postprocess.create_pipeline(vkdev, opt);
        if (ret != 0)
        {
            NCNN_LOGE("postprocess create_pipeline failed %d", ret);
            return -1;
        }
        postprocess.prob_threshold = prob_threshold;
        postprocess.nms_threshold = nms_threshold;
        postprocess.num_class = 80;
        postprocess.num_anchor = 8400;
        postprocess.img_w = img_w;
        postprocess.img_h = img_h;
        postprocess.pad_left = wpad / 2;
        postprocess.pad_top = hpad / 2;
        postprocess.scale = scale;

        ncnn::VkMat proposals_vkmat;
        ncnn::VkMat picked_vkmat;
        ret = postprocess.generate(out_vkmat, proposals_vkmat, cmd2, opt);
        if (ret != 0)
        {
            NCNN_LOGE("postprocess generate failed %d", ret);
            return -1;
        }
        ret = postprocess.nms(proposals_vkmat, picked_vkmat, cmd2, opt);
        if (ret != 0)
        {
            NCNN_LOGE("postprocess nms failed %d", ret);
            return -1;
        }

        // download results
        ncnn::Mat proposals_mat;
        ncnn::Mat picked_mat;
        cmd2.record_download(proposals_vkmat, proposals_mat, opt);
        cmd2.record_download(picked_vkmat, picked_mat, opt);
        cmd2.submit_and_wait();

        float* proposals_data = (float*)proposals_mat.data;
        int* picked_data = (int*)picked_mat.data;
        for (int i = 0; i < postprocess.num_anchor; i++)
        {
            if (picked_data[i] == 0)
                continue;
            float x0 = proposals_data[i * 6 + 0];
            float y0 = proposals_data[i * 6 + 1];
            float x1 = proposals_data[i * 6 + 2];
            float y1 = proposals_data[i * 6 + 3];
            float score = proposals_data[i * 6 + 4];
            int label = (int)proposals_data[i * 6 + 5];

            if (label < 0 || score < prob_threshold)
                continue;

            Object obj;
            obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            obj.label = label;
            obj.prob = score;
            objects.push_back(obj);
        }

        postprocess.destroy_pipeline(opt);
        preprocess_layer.destroy_pipeline(opt);

        // blob_vkallocator->mappable indicates unified memory (iGPU), allowing zero-copy read via mapped_ptr()
    } // all ncnn objects destroyed here before reclaiming allocators

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54),
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolo11_vk(m, objects);

    draw_objects(m, objects);

    return 0;
}
